# %%

import torch
import plotly.express as px

a = torch.triu(torch.ones(5, 5))
px.imshow((a @ a.T).cpu(), color_continuous_scale='RdBu', color_continuous_midpoint=0)
# %%

# %%
