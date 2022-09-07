
# Difformer - PyTorch (Experimental)

Diffusion based transformer, in PyTorch.

```bash
pip install difformer-pytorch
```
[![PyPI - Python Version](https://img.shields.io/pypi/v/difformer-pytorch?style=flat&colorA=black&colorB=black)](https://pypi.org/project/difformer-pytorch/)


## Usage
```python
from difformer_pytorch import Difformer

num_tokens = 1000

difformer = Difformer(
    num_tokens=num_tokens,
    embedding_dim=512,
    num_layers=6
)

# Input tokens and mask
x = torch.randint(0, num_tokens, (1, 1024))
mask = torch.ones_like(x).bool()

# Train difformer to demask
loss = difformer(x, mask=mask)
loss.backward()

# Sample unmasked prediction given masked start sequence
sampled = difformer.sample(
    x,
    mask=mask,
    num_steps=5
) # [1, 1024]

```
