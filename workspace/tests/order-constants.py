# %%
%load_ext autoreload
%autoreload 2

from autoencoders import Autoencoder
import torch
import plotly.express as px

torch.set_grad_enabled(False)
# %%

params = dict(d_model=512, layer=18, expansion=16, alpha=0.1)
autoencoder = Autoencoder.from_config("ordered", **params)

# px.bar(autoencoder.counts)
px.bar(autoencoder.htail)
# %%