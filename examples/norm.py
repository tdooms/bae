# %%
%load_ext autoreload
%autoreload 2

from transformers import AutoModelForCausalLM, AutoTokenizer
from coders import Autoencoder, Placeholder

from tqdm import tqdm
from scipy import stats
from quimb.tensor import Tensor
from itertools import product
from tqdm import tqdm

import plotly.express as px
import torch
import gc

import coders

# %%
torch.set_grad_enabled(False)
model = Placeholder(d_model=1024, name="Qwen3-0.6B-Base")
# %%
coder = Autoencoder.load(model, layer=18, expansion=16, root='weights/incremental').half()
# %%
norm = coder.left.norm(dim=-1) * coder.right.norm(dim=-1) * coder.down.norm(dim=0)
px.scatter(y=norm.cpu(), x=list(range(len(norm))), template='plotly_white', labels=dict(x="Index", y="Norm"))
# %%
