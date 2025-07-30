# %%

import torch
import plotly.express as px
from einops import rearrange, einsum

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