
# %%
import torch

def tiled_contraction(f, L, R, tile_size=1024):
    """
    Efficiently compute (f @ f.T) * (L @ L.T) * (R @ R.T) using simple tiling.
    
    Tiles the computation as: f A f = f_0 A_00 f_0 + f_0 A_01 f_1 + f_1 A_10 f_0 + f_1 A_11 f_1 + ...
    
    Args:
        f: tensor of shape (..., h, k) 
        L: tensor of shape (h, l)
        R: tensor of shape (k, r)  
        tile_size: size of tiles to process at once
        
    Returns:
        scalar result per batch dimension
    """
    h, k = f.shape[-2:]
    result = torch.zeros(f.shape[:-2], device=f.device, dtype=f.dtype)
    
    # Precompute R @ R.T once (k x k)
    RR = torch.einsum('kr,jr->kj', R, R)
    
    # Tile over the h dimension
    for i in range(0, h, tile_size):
        i_end = min(i + tile_size, h)
        for j in range(0, h, tile_size):
            j_end = min(j + tile_size, h)
            
            # Get f tiles
            f_i = f[..., i:i_end, :]  # (..., tile_h, k)
            f_j = f[..., j:j_end, :]  # (..., tile_h, k)
            
            # Get L @ L.T tile
            LL_tile = torch.einsum('il,jl->ij', L[i:i_end], L[j:j_end])  # (tile_h, tile_h)
            
            # Compute contribution: f_i @ RR @ f_j.T * LL_tile
            result += torch.einsum('...ik,kj,...jl,il->...', f_i, RR, f_j, LL_tile)
    
    return result



batch_size = 2
h, k, l, r = 100, 80, 50, 60

f = torch.randn(batch_size, h, k)
L = torch.randn(h, l) 
R = torch.randn(k, r)

# Reference computation
def reference_einsum(f, L, R):
    return torch.einsum("...h,...k,hl,kl,hr,kr->...", f, f, L, L, R, R)

# Test
result_ref = reference_einsum(f, L, R)
result_tiled = tiled_contraction(f, L, R, tile_size=32)

print(f"Reference result: {result_ref}")
print(f"Tiled result: {result_tiled}")
print(f"Max difference: {torch.abs(result_ref - result_tiled).max()}")

# %%