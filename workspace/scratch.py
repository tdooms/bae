# %%

import torch
import plotly.express as px
from einops import rearrange, einsum
# %%
a = torch.tril(torch.ones(5, 5))
v = torch.randn(5).abs()
# %%
q = einsum(a, a, v.cumsum(0), "mid in1, mid in2, mid -> in1 in2")
px.imshow(q, color_continuous_scale='RdBu', color_continuous_midpoint=0)
# %%
q = einsum(a, a, v, "mid in1, mid in2, in1 -> in1 in2")
px.imshow(q, color_continuous_scale='RdBu', color_continuous_midpoint=0)
# %%

a = torch.ones(5, 5)
v = torch.randn(5).abs()

q = einsum(a, a, v.cumsum(0), "mid in1, mid in2, mid -> in1 in2")
px.imshow(q, color_continuous_scale='RdBu', color_continuous_midpoint=0)
# %%

f = torch.ones(5, 5)
b = einsum(f, f, f, "out mid, mid in1, mid in2 -> out in1 in2")
px.imshow(b, color_continuous_scale='RdBu', color_continuous_midpoint=0, facet_col=0)
# %%
m = torch.tril(torch.ones(5, 5))
d = torch.randn(5, 5).abs()

y0 = m @ d
y1 = d.cumsum(0)

px.imshow(torch.stack([y0, y1], dim=0), color_continuous_scale='RdBu', color_continuous_midpoint=0, facet_col=0)
# %%

m = torch.tril(torch.ones(5, 5))
g = (m @ m.T) / 5
x = torch.randn(5)

print(einsum(x, x, g, "i, j, i j -> "))
print(x.flip(0).cumsum(0).pow(2).sum() / 5)

px.imshow(g)
# %%
def bilinear_with_H(x, A):
    """
    Compute x^T (A ⊙ H) x  where H_{ij} = min(i,j)+1,
    in O(n^2) time, O(n) extra memory.
    """
    n = x.size(0)
    idx = torch.arange(n, device=x.device, dtype=x.dtype)   # [0,1,2,…,n-1]
    c   = idx + 1                                           # [1,2,3,…,n]

    # 1) full mat–vec:       u_i = sum_j A_{ij} x_j
    u = A.matmul(x)                # (n,)

    # 2) lower‐triangular mv: v_i = sum_{j ≤ i} A_{ij} x_j
    L = torch.tril(A)
    v = L.matmul(x)                # (n,)

    # 3) weighted lower mv:   w_i = sum_{j ≤ i} A_{ij} x_j * (j+1)
    w = L.matmul(x * c)            # (n,)

    # combine:
    #   α_i = w_i
    #   β_i = u_i – v_i
    #   S = sum_i x_i*(α_i + (i+1)*β_i)
    return ( x * w + (c * x) * (u - v) ).sum()


n = 6
q = torch.randn(n, n).abs()
m = torch.tril(torch.ones(n, n))
g = (m @ m.T) * q
x = torch.ones(n)

print(bilinear_with_H(torch.ones(n), q))
print(einsum(x, x, g, "i, j, i j -> "))

# %%
def min_kernel(x, A, weight) -> torch.Tensor:
    L = torch.tril(A)
    
    u = einsum(A, x, "hid inp, ... inp -> ... hid")
    v = einsum(L, x, "out inp, inp -> out")
    w = einsum(L, x * weight, "out inp, inp -> out")

    return (x * w + (weight * x) * (u - v)).sum()

n = 6
a = torch.randn(n, n).abs()
m = torch.tril(torch.ones(n, n))
w = torch.randn(n)

g = (m @ torch.diag(w) @ m.T) * a
x = torch.ones(n)

print(min_kernel(x, a, w.cumsum(0)))
print(einsum(x, x, g, "i, j, i j -> "))
# %%