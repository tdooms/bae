# %%
%load_ext autoreload
%autoreload 2

import torch
import plotly.express as px

# %%
torch.set_grad_enabled(False)

m = torch.tril(torch.ones(20, 20))