# %%
%load_ext autoreload
%autoreload 2

from tqdm import tqdm
from autoencoder import Autoencoder, Placeholder
import plotly.express as px
import torch
# %%
torch.set_grad_enabled(False)
model = Placeholder("Qwen/Qwen3-0.6B-Base", d_model=1024, dtype=torch.float32)
coder = Autoencoder.from_config(model, "vanilla", layer=18, expansion=4, alpha=0.2).eval().cuda()
# coder = Autoencoder.load(model, "vanilla", layer=18, expansion=16, alpha=0.2, tags=[]).eval()
# %%
tn = coder.network('a') | coder.network('b')
tn.contract(all, output_inds=[]).data