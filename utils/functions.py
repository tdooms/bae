from itertools import product
from torch import Tensor
from jaxtyping import Float

def hoyer_density(x: Float[Tensor, "... x"], dim: int = -1):
    """Computes Hoyer's density measure (1=dense, 0=sparse) along a specified dim."""
    size = x.size(dim) if isinstance(dim, int) else product(x.size(d) for d in dim)
    return (x.norm(p=1, dim=dim) / x.norm(p=2, dim=dim) - 1.0) / (size**0.5 - 1.0)

def hoyer_sparsity(x: Float[Tensor, "... x"], dim: int = -1):
    """Computes Hoyer's sparsity measure (0=dense, 1=sparse) along a specified dim."""
    size = x.size(dim) if isinstance(dim, int) else product(x.size(d) for d in dim)
    return ((size**0.5 - 1.0) - (x.norm(p=1, dim=dim) / x.norm(p=2, dim=dim))) / (size**0.5 - 1.0)

def generalized_effective_dimension(v: Float[Tensor, "... x"], p: float = 4.0) ->  Float[Tensor, "..."]:
    """Computes the rough amount of 'big' elements in a vector."""
    vp = v.abs().pow(p)
    return vp.sum(dim=-1).pow(2) / vp.pow(2).sum(dim=-1)

def effective_dimension(tensor, dim=-1):
    return tensor.abs().sum(dim=dim).pow(2) / tensor.pow(2).sum(dim=dim)

def inverted_participation_ratio(tensor, dim: int = -1):
    tensor = tensor / tensor.norm(p=2, dim=dim, keepdim=True)
    return 1.0 / (tensor.pow(4).sum(dim=dim))