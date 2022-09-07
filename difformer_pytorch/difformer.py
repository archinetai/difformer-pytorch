from inspect import isfunction
from math import pi, sqrt
from typing import Any, Callable, Optional, Tuple, TypeVar, Union

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops_exts import rearrange_many
from torch import Tensor, einsum, nn
from typing_extensions import TypeGuard

T = TypeVar("T")

"""
Utils
"""


def exists(val: Optional[T]) -> TypeGuard[T]:
    return val is not None


def default(val: Optional[T], d: Union[Callable[..., T], T]) -> T:
    if exists(val):
        return val
    return d() if isfunction(d) else d


"""
Diffusion
"""


class Distribution:
    def __call__(self, num_samples: int, device: torch.device):
        raise NotImplementedError()


class LogNormalDistribution(Distribution):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(
        self, num_samples, device: torch.device = torch.device("cpu")
    ) -> Tensor:
        normal = self.mean + self.std * torch.randn((num_samples,), device=device)
        return normal.exp()


class Schedule(nn.Module):
    """Interface used by different schedules"""

    def forward(self, num_steps: int, device: torch.device) -> Tensor:
        raise NotImplementedError()


class RhoSchedule(Schedule):
    """https://arxiv.org/abs/2206.00364 equation 5"""

    def __init__(self, sigma_min: float, sigma_max: float, rho: float = 7.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def forward(self, num_steps: int, device: Any) -> Tensor:
        rho_inv = 1.0 / self.rho
        steps = torch.arange(num_steps, device=device, dtype=torch.float32)
        sigmas = (
            self.sigma_max**rho_inv
            + (steps / (num_steps - 1))
            * (self.sigma_min**rho_inv - self.sigma_max**rho_inv)
        ) ** self.rho
        sigmas = F.pad(sigmas, pad=(0, 1), value=0.0)
        return sigmas


class Sampler(nn.Module):
    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        raise NotImplementedError()


class AEulerSampler(Sampler):
    def get_sigmas(self, sigma: float, sigma_next: float) -> Tuple[float, float]:
        sigma_up = sqrt(sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2)
        sigma_down = sqrt(sigma_next**2 - sigma_up**2)
        return sigma_up, sigma_down

    def step(self, x: Tensor, fn: Callable, sigma: float, sigma_next: float) -> Tensor:
        # Sigma steps
        sigma_up, sigma_down = self.get_sigmas(sigma, sigma_next)
        # Derivative at sigma (∂x/∂sigma)
        d = (x - fn(x, sigma=sigma)) / sigma
        # Euler method
        x_next = x + d * (sigma_down - sigma)
        # Add randomness
        x_next = x_next + torch.randn_like(x) * sigma_up
        return x_next

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        x = sigmas[0] * noise
        # Denoise to sample
        for i in range(num_steps - 1):
            x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore # noqa
        return x


class Diffusion(nn.Module):
    """Elucidated Diffusion: https://arxiv.org/abs/2206.00364"""

    def __init__(
        self,
        net: nn.Module,
        *,
        sigma_distribution: Distribution,
        sigma_data: float,
    ):
        super().__init__()

        self.net = net
        self.sigma_data = sigma_data
        self.sigma_distribution = sigma_distribution

    def get_scale_weights(self, sigmas: Tensor) -> Tuple[Tensor, ...]:
        sigma_data = self.sigma_data
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")
        c_skip = (sigma_data**2) / (sigmas_padded**2 + sigma_data**2)
        c_out = (
            sigmas_padded * sigma_data * (sigma_data**2 + sigmas_padded**2) ** -0.5
        )
        c_in = (sigmas_padded**2 + sigma_data**2) ** -0.5
        c_noise = torch.log(sigmas) * 0.25
        return c_skip, c_out, c_in, c_noise

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        batch, device = x_noisy.shape[0], x_noisy.device

        assert exists(sigmas) ^ exists(sigma), "Either sigmas or sigma must be provided"

        # If sigma provided use the same for all batch items (used for sampling)
        if exists(sigma):
            sigmas = torch.full(size=(batch,), fill_value=sigma).to(device)

        assert exists(sigmas)

        # Predict network output and add skip connection
        c_skip, c_out, c_in, c_noise = self.get_scale_weights(sigmas)
        x_pred = self.net(c_in * x_noisy, c_noise, **kwargs)
        x_denoised = c_skip * x_noisy + c_out * x_pred

        # Dynamic thresholding
        return x_denoised.clamp(-1.0, 1.0)

    def loss_weight(self, sigmas: Tensor) -> Tensor:
        # Computes weight depending on data distribution
        return (sigmas**2 + self.sigma_data**2) * (sigmas * self.sigma_data) ** -2

    def forward(self, x: Tensor, noise: Tensor = None, **kwargs) -> Tensor:
        batch, device = x.shape[0], x.device

        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch, device=device)
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")

        # Add noise to input
        noise = default(noise, lambda: torch.randn_like(x))
        x_noisy = x + sigmas_padded * noise

        # Compute denoised values
        x_denoised = self.denoise_fn(x_noisy, sigmas=sigmas, **kwargs)

        # Compute weighted loss
        losses = F.mse_loss(x_denoised, x, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")
        losses = losses * self.loss_weight(sigmas)
        loss = losses.mean()

        return loss


class DiffusionSampler(nn.Module):
    def __init__(
        self,
        diffusion: Diffusion,
        *,
        sampler: Sampler,
        sigma_schedule: Schedule,
        num_steps: Optional[int] = None,
    ):
        super().__init__()
        self.denoise_fn = diffusion.denoise_fn
        self.sampler = sampler
        self.sigma_schedule = sigma_schedule
        self.num_steps = num_steps

    @torch.no_grad()
    def forward(
        self, noise: Tensor, num_steps: Optional[int] = None, **kwargs
    ) -> Tensor:
        device = noise.device
        num_steps = default(num_steps, self.num_steps)  # type: ignore
        assert exists(num_steps), "Parameter `num_steps` must be provided"
        # Compute sigmas using schedule
        sigmas = self.sigma_schedule(num_steps, device)
        # Append additional kwargs to denoise_fn (used e.g. for conditional model)
        fn = lambda *a, **ka: self.denoise_fn(*a, **{**ka, **kwargs})  # noqa
        # Sample using sampler
        x = self.sampler(noise, fn=fn, sigmas=sigmas, num_steps=num_steps)
        x = x.clamp(-1.0, 1.0)
        return x


"""
Transformer
"""


def attention_mask(
    sim: Tensor,
    mask: Tensor,
) -> Tensor:
    mask = rearrange(mask, "b j -> b 1 1 j")
    max_neg_value = -torch.finfo(sim.dtype).max
    sim = sim.masked_fill(~mask, max_neg_value)
    return sim


class LayerNorm(nn.Module):
    def __init__(self, features: int, *, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.bias = bias
        self.eps = eps
        self.g = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        norm = (x - mean) * (var + self.eps).rsqrt() * self.g
        return norm + self.b if self.bias else norm


class AttentionBase(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int = 64,
        num_heads: int = 8,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        self.scale = head_features**-0.5
        self.num_heads = num_heads
        mid_features = head_features * num_heads
        out_features = out_features if exists(out_features) else features

        self.to_out = nn.Sequential(
            nn.Linear(in_features=mid_features, out_features=out_features, bias=False),
            LayerNorm(features=out_features, bias=False),
        )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        mask: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
    ) -> Tensor:

        # Split heads, scale queries
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.num_heads)
        q = q * self.scale

        # Compute similarity matrix with bias and mask
        sim = einsum("... n d, ... m d -> ... n m", q, k)
        sim = sim + attention_bias if exists(attention_bias) else sim
        sim = attention_mask(sim, mask) if exists(mask) else sim

        # Get attention matrix with softmax
        attn = sim.softmax(dim=-1, dtype=torch.float32)

        # Compute values
        out = einsum("... n j, ... j d -> ... n d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int = 64,
        num_heads: int = 8,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        mid_features = head_features * num_heads

        self.norm = LayerNorm(features, bias=False)
        self.to_qkv = nn.Linear(
            in_features=features, out_features=mid_features * 3, bias=False
        )
        self.attention = AttentionBase(
            features,
            num_heads=num_heads,
            head_features=head_features,
            out_features=out_features,
        )

    def forward(self, x: Tensor, *, mask: Optional[Tensor] = None) -> Tensor:
        x = self.norm(x)
        q, k, v = torch.chunk(self.to_qkv(x), chunks=3, dim=-1)
        x = self.attention(q, k, v, mask=mask)
        return x


def FeedForward(features: int, multiplier: int = 2) -> nn.Module:
    mid_features = int(features * multiplier)
    return nn.Sequential(
        LayerNorm(features, bias=False),
        nn.Linear(in_features=features, out_features=mid_features, bias=False),
        nn.GELU(),
        LayerNorm(mid_features, bias=False),
        nn.Linear(in_features=mid_features, out_features=features, bias=False),
    )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int = 64,
        num_heads: int = 8,
        multiplier: int = 2,
    ):
        super().__init__()

        self.attention = Attention(
            features=features, head_features=head_features, num_heads=num_heads
        )

        self.feed_forward = FeedForward(features=features, multiplier=multiplier)

    def forward(self, x: Tensor, *, mask: Tensor = None) -> Tensor:
        x = self.attention(x, mask=mask) + x
        x = self.feed_forward(x) + x
        return x


class LearnedPositionalEmbedding(nn.Module):
    """Used for continuous time"""

    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def TimePositionalEmbedding(dim: int, out_features: int) -> nn.Module:
    return nn.Sequential(
        LearnedPositionalEmbedding(dim),
        nn.Linear(in_features=dim + 1, out_features=out_features),
    )


class ContinuousTransformer(nn.Module):
    def __init__(
        self,
        *,
        features: int,
        context_features: int,
        num_blocks: int,
        head_features: int = 64,
        num_heads: int = 8,
        multiplier: int = 4,
    ):
        super().__init__()
        self.features = features
        time_features = features * 2

        self.to_time = nn.Sequential(
            TimePositionalEmbedding(dim=features, out_features=time_features),
            nn.SiLU(),
            nn.Linear(in_features=time_features, out_features=time_features),
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    features=features + context_features,
                    head_features=head_features,
                    num_heads=num_heads,
                    multiplier=multiplier,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, x: Tensor, t: Tensor, context: Tensor) -> Tensor:
        n = x.shape[1]
        # Concat context
        x = torch.cat([x, context], dim=2)
        # Concat time token
        t = rearrange(self.to_time(t), "b d -> b 1 d")
        x = torch.cat([x, t], dim=1)
        # Feed into transformer
        for block in self.blocks:
            x = block(x)
        # Remove extra token and context features
        x = x[:, 0:n, 0 : self.features]
        return x


class TokenEmbedding(nn.Module):
    def __init__(self, num_tokens: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_tokens, embedding_dim=embedding_dim
        )

    def get_ids(self, x: Tensor) -> Tensor:
        b = x.shape[0]
        e = repeat(self.embedding.weight, "n d -> b n d", b=b)
        sim = torch.cdist(x, e, p=2)
        indices = sim.argmax(dim=-1)
        return indices

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(self.embedding(x))


"""
Difformer
"""


class DifformerBase(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        diffusion_sigma_distribution: Distribution,
        diffusion_sigma_data: float,
    ):
        super().__init__()
        assert (
            embedding_dim % num_heads == 0
        ), "embedding_dim must be divisible by num_heads"

        self.token_embedding = TokenEmbedding(
            num_tokens=num_tokens, embedding_dim=embedding_dim
        )

        self.transformer = ContinuousTransformer(
            features=embedding_dim,
            context_features=embedding_dim,
            num_blocks=num_layers,
            head_features=embedding_dim // num_heads,
            num_heads=num_heads,
            multiplier=4,
        )

        self.diffusion = Diffusion(
            net=self.transformer,
            sigma_distribution=diffusion_sigma_distribution,
            sigma_data=diffusion_sigma_data,
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        tokens = self.token_embedding(x)
        tokens_masked = self.token_embedding(x.masked_fill(~mask, 0))
        return self.diffusion(tokens, context=tokens_masked)

    def sample(
        self,
        x: Tensor,
        num_steps: int,
        sigma_schedule: Schedule,
        sampler: Sampler,
        mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        # Compute masked tokens embedding and start noise
        x_masked = x.masked_fill(~mask, 0) if exists(mask) else x
        tokens_masked = self.token_embedding(x_masked)
        noise = torch.randn_like(tokens_masked)
        # Sample unmasked embedding
        diffusion_sampler = DiffusionSampler(
            diffusion=self.diffusion,
            num_steps=num_steps,
            sampler=sampler,
            sigma_schedule=sigma_schedule,
        )
        embedding = diffusion_sampler(noise, context=tokens_masked, **kwargs)
        # Convert back into tokens
        indices = self.token_embedding.get_ids(embedding)
        return indices


class Difformer(DifformerBase):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            num_heads=8,
            diffusion_sigma_distribution=LogNormalDistribution(-3.0, 1.0),
            diffusion_sigma_data=0.1,
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})

    def sample(self, *args, **kwargs):
        default_kwargs = dict(
            sigma_schedule=RhoSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            sampler=AEulerSampler(),
        )
        return super().sample(*args, **{**default_kwargs, **kwargs})
