# %%
%load_ext autoreload
%autoreload 2

import torch
import plotly.express as px

# %%
torch.set_grad_enabled(False)
m = Monarch(4)
px.imshow(m.matrix(), color_continuous_scale='RdBu', color_continuous_midpoint=0)
# %%

x = torch.randn(16)
y0 = m(x)
y1 = m.matrix() @ x
y0, y1
# %%
import torch
from einops import rearrange, einsum

A = torch.randn(5, 5)
B = torch.randn(5, 5)
D = torch.diag(torch.randn(5))
x = torch.randn(5)

print(einsum(x, A, D, x, "i, i j, i j, j ->"))
einsum(x, A, x, "i, i j, j ->") * einsum(x, D, x, "i, i j, j ->")
