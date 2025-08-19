# %%
%load_ext autoreload
%autoreload 2

import torch
import plotly.express as px

# %%
torch.set_grad_enabled(False)

m = torch.tril(torch.ones(20, 20))

px.imshow(m[:, 5:10].T @ m[:, 5:10])
# %%

px.imshow(torch.arange(1, 6).flip(0).expand(5, -1))
# %%
(torch.tril(torch.ones(5, 5)) @ torch.tril(torch.ones(5, 5))) + k
# %%
from einops import einsum
triu = torch.triu(torch.ones(5, 5))
order = einsum(triu, triu, "f1 reg, f2 reg -> f1 f2")
px.imshow(order)
# %%
