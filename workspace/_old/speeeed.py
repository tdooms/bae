# %%

import torch
import time
import torch.backends.opt_einsum as oe

from itertools import combinations_with_replacement
from einops import einsum

oe.strategy = 'auto-hq'

torch.set_grad_enabled(False)
# %%
def reference(f, L, R):
    return einsum(f, f, L, L, R, R, "... h1, ... h2, h1 l, h2 l, h1 r, h2 r ->...")

def tiled(f, L, R, block=4096):
    size = L.size(0)
    result = torch.zeros_like(f[..., 0])
    
    for s1, s2 in combinations_with_replacement(range(0, size, block), 2):
        scale = 2 if s1 != s2 else 1
        e1, e2 = min(s1 + block, size), min(s2 + block, size)
        result += scale * torch.einsum("...h, ...k, hl, kl, hr, kr -> ...", f[..., s1:e1], f[..., s2:e2], L[s1:e1], L[s2:e2], R[s1:e1], R[s2:e2])

    return result

# %%
import torch
import time
import gc

@torch.no_grad()
def _time(fn, *args, iters=5, **kw):
    fn(*args, **kw)
    dev = args[0].device
    if dev.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): _ = fn(*args, **kw)
    if dev.type == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

def clean_up():
    gc.collect()
    torch.cuda.empty_cache()

def safe_time(fn, *args, **kwargs):
    """Run timing with OOM protection"""
    try:
        return _time(fn, *args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        clean_up()
        return None

def benchmark_sizes(device="cuda"):
    B, D = 1024*4, 1024
    H_sizes =[1024 * 2**i for i in range(8)]
    
    print(f"{'H':>6} {'Reference':>12} {'Quick':>12} {'Speedup':>8}")
    print("-" * 45)
    
    for H in H_sizes:
        f = torch.randn(B, H, device=device, dtype=torch.float16)
        L = torch.randn(H, D, device=device, dtype=torch.float16)
        R = torch.randn(H, D, device=device, dtype=torch.float16)
        
        clean_up()
        t_quick = safe_time(lambda *args: tiled(*args, block=2048), f, L, R, iters=3)
        clean_up()
        t_ref = safe_time(lambda *args: tiled(*args, block=4096), f, L, R, iters=3)
        clean_up()
        
        
        ref_str = "OOM" if t_ref is None else f"{t_ref*1e3:.1f}ms"
        tiled_str = "OOM" if t_quick is None else f"{t_quick*1e3:.1f}ms"
        
        if t_ref and t_quick:
            speedup = f"{t_ref/t_quick:.2f}x"
        else:
            speedup = "N/A"
            
        print(f"{H:>6} {ref_str:>12} {tiled_str:>12} {speedup:>8}")
        
        # Skip remaining sizes if tiled also OOMs
        if t_quick is None:
            break

benchmark_sizes()
# %%
torch.set_grad_enabled(True)
# --- large (your sizes) ---
B, H, D = 256*16, 2*1024, 1024
f = torch.randn(B, H, device="cuda")
L = torch.randn(H, D, device="cuda")
R = torch.randn(H, D, device="cuda")

# pick tiles automatically from budget
t_naive = reference(f, L, R)
t_dense = tiled(f, L, R)

torch.testing.assert_close(t_naive, t_dense, rtol=1e-3, atol=1e-3)

# Test gradients
f.requires_grad_(True)
L.requires_grad_(True)
R.requires_grad_(True)

# Compute reference gradients
loss_ref = reference(f, L, R).sum()
loss_ref.backward()
f_grad_ref = f.grad.clone()
L_grad_ref = L.grad.clone()
R_grad_ref = R.grad.clone()

# Clear gradients
f.grad = None
L.grad = None
R.grad = None

# Compute tiled gradients
loss_tiled = tiled(f, L, R).sum()
loss_tiled.backward()

# Check gradient equality
torch.testing.assert_close(f.grad, f_grad_ref, rtol=1e-3, atol=1e-3)
torch.testing.assert_close(L.grad, L_grad_ref, rtol=1e-3, atol=1e-3)
torch.testing.assert_close(R.grad, R_grad_ref, rtol=1e-3, atol=1e-3)
print("Gradients match!")
