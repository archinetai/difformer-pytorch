
# Difformer - PyTorch (Experimental)

Diffusion based transformer, in PyTorch.

```bash
pip install difformer-pytorch
```
[![PyPI - Python Version](https://img.shields.io/pypi/v/difformer-pytorch?style=flat&colorA=black&colorB=black)](https://pypi.org/project/difformer-pytorch/)


## Usage

### Token based
```python
from difformer_pytorch import Difformer

num_tokens = 1000

difformer = Difformer(
    num_tokens=num_tokens,
    embedding_dim=512,
    num_layers=6
)

# Input tokens and mask
tokens = torch.randint(0, num_tokens, (1, 1024))
mask = torch.ones_like(x).bool()

# Train difformer to demask
loss = difformer(tokens=tokens, mask=mask)
loss.backward()

# Sample unmasked prediction given masked start sequence
sampled = difformer.sample(
    tokens=tokens,
    mask=mask,
    num_steps=5
) # [1, 1024]

```

### Embedding based
```py
from difformer_pytorch import Difformer

difformer = Difformer(
    embedding_dim=512,
    num_layers=6
)

# Input embedding and mask
embedding = torch.randn(1, 1024, 512)
mask = torch.ones(1, 1024).bool()

# Train difformer
loss = difformer(embedding=embedding, mask=mask)
loss.backward()

# Sample prediction given masked start embedding
sampled = difformer.sample(
    embedding=embedding,
    mask=mask, # Optional mask to apply on embeddings
    num_steps=5
) # [1, 1024, 512]
```
