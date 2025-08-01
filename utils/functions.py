from itertools import product
from torch import Tensor
from jaxtyping import Float

def inv_hoyer(x: Float[Tensor, "... x"], dim: int = -1):
    """Computes 1 - Hoyer's sparsity measure along a specified dim."""
    size = x.size(dim) if isinstance(dim, int) else product(x.size(d) for d in dim)
    return (x.norm(p=1, dim=dim) / x.norm(p=2, dim=dim) - 1.0) / (size**0.5 - 1.0)

def hoyer(x: Float[Tensor, "... x"], dim: int = -1):
    """Computes Hoyer's sparsity measure along a specified dim."""
    size = x.size(dim) if isinstance(dim, int) else product(x.size(d) for d in dim)
    return ((size**0.5 - 1.0) - (x.norm(p=1, dim=dim) / x.norm(p=2, dim=dim))) / (size**0.5 - 1.0)

def generalized_participation_ratio(v:  Float[Tensor, "... x"], p: float = 4.0) ->  Float[Tensor, "..."]:
    """Computes the rough amount of 'big' elements in a vector."""
    vp = v.abs().pow(p)
    return vp.sum(dim=-1).pow(2) / vp.pow(2).sum(dim=-1)