import torch

from jaxtyping import Float
from torch import Tensor

def tiled_product(
    x: Float[Tensor, "... hid"],
    l: Float[Tensor, "hid left"],
    r: Float[Tensor, "hid right"],
    tiles: int,
    inds: list
):
    """Efficient tiled evaluation of a feature-kernel product."""
    
    # View matrices as tiles to be computed iteratively
    x = x.view(*x.shape[:-1], tiles, -1)
    r = r.view(tiles, -1, r.shape[1])
    l = l.view(tiles, -1, l.shape[1])
    
    # Define the einsum pattern and its constituent matrices
    sub = lambda i: (x[..., i, :], l[i], r[i])
    pattern = "...h, hl, hr, ...k, kl, kr -> ..."
    return sum(((i == j) + 1) * torch.einsum(pattern, *sub(i), *sub(j)) for i, j in inds)

def tiled_masked_product(
    x: Float[Tensor, "... hid"],
    l: Float[Tensor, "hid left"],
    r: Float[Tensor, "hid right"],
    tiles: int,
    inds: list
):
    """Efficient tiled evaluation of a masked feature-kernel product."""
    
    # View matrices as tiles to be computed iteratively
    x = x.view(*x.shape[:-1], tiles, -1)
    r = r.view(tiles, -1, r.shape[1])
    l = l.view(tiles, -1, l.shape[1])
    
    # Define the einsum pattern and its constituent matrices
    like = dict(dtype=x.dtype, device=x.device)
    sub = lambda i: (x[..., i, :], l[i], r[i])
    pattern = "...h, hl, hr, ...k, kl, kr, hk -> ..."
    return sum(((i == j) + 1) * torch.einsum(pattern, *sub(i), *sub(j), gram_block(r.shape[0], tiles, i, j, **like)) for i, j in inds)

def gram_block(total: int, tiles: int, i: int, j: int, dtype=torch.float16, device=None):
    """Compute a part of the lower-triangular gram matrix (L^T L)."""
    assert total % tiles == 0, "Total must be divisible by tiles"
    size = total // tiles
    
    i = torch.arange(size * i, size * (i+1), device=device, dtype=torch.long)
    j = torch.arange(size * j, size * (j+1), device=device, dtype=torch.long)
    return (total - torch.maximum(i[:, None], j[None, :])).to(dtype) / total