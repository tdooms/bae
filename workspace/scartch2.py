# %%
%load_ext autoreload
%autoreload 2

from oldcoders.monarch import Monarch
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