import torch

def tiled_product(x, l, r, tiles: int, inds: list):
    """Efficient evaluation of the feature-kernel product."""
    
    # View matrices as tiles to be computed iteratively
    x = x.view(*x.shape[:-1], tiles, -1)
    r = r.view(tiles, -1, r.shape[1])
    l = l.view(tiles, -1, l.shape[1])
    
    # Define the einsum pattern and its constituent matrices
    sub = lambda i: (x[..., i, :], l[i], r[i])
    pattern = "...h, hl, hr, ...k, kl, kr -> ..."
    return sum(((i == j) + 1) * torch.einsum(pattern, *sub(i), *sub(j)) for i, j in inds)

def blocked_masked_inner(f, left, right, inds):
    """Efficient evaluation of the kernelised inner product."""
    pattern = "...h, hl, hr, ...k, kl, kr, hk -> ..."
    like = dict(device=f.device, dtype=f.dtype)
    sub = lambda s, e: (f[..., s:e], left[s:e], right[s:e])
    return sum((2 if m1 != m2 else 1) * torch.einsum(pattern, *sub(*m1), *sub(*m2), tril_gram_block(f.size(-1), m1, m2, **like)) for m1, m2 in inds)

def tril_gram_block(n, m1, m2, dtype=torch.float16, device=None):
    i = torch.arange(*m1, device=device, dtype=torch.long)
    j = torch.arange(*m2, device=device, dtype=torch.long)
    return (n - torch.maximum(i[:, None], j[None, :])).to(dtype) / n