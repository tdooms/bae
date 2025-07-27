# %%
%load_ext autoreload
%autoreload 2

from transformers import AutoModelForCausalLM, AutoTokenizer
from coders.sparse import Autoencoder, Placeholder

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
model = Placeholder(d_model=1024, name="Qwen/Qwen3-0.6B-Base")
# %%
# Construct a similarity matrix for all autoencoders
tree = None
k = 24

sims = torch.zeros((k, k), dtype=torch.float32)

for i, j in tqdm(product(range(k), repeat=2)):
    if i == j:
        sims[i, j] = 1.0
        continue
    
    coder0 = Autoencoder.load(model, layer=i, expansion=16, root='weights/incremental').half()
    coder1 = Autoencoder.load(model, layer=j, expansion=16, root='weights/incremental').half()

    tn0 = coder0.network() / 20
    tn1 = coder1.network() / 20

    tree = (tn0 & tn0).contraction_tree(optimize='optimal') if tree is None else tree
    
    cross = (tn0 & tn1).contract(all, [], optimize=tree)
    self1 = (tn0 & tn0).contract(all, [], optimize=tree)
    self2 = (tn1 & tn1).contract(all, [], optimize=tree)
    
    sim = (2 * cross.sqrt() / (self1.sqrt() + self2.sqrt())).item()
    sims[i, j] = sim

    gc.collect()
    torch.cuda.empty_cache()

torch.save(sims, "sims.pt")
px.imshow(sims.cpu(), color_continuous_midpoint=0.5, color_continuous_scale='RdBu')
# %%
px.imshow(sims.cpu(), color_continuous_scale="Viridis", color_continuous_midpoint=0.5,)
# %%
# Run a single sample to check similarity
# coder0 = Autoencoder.load(model, layer=17, expansion=16, root='weights').half()
coder0 = Autoencoder.load(model, layer=18, expansion=16, root='weights/incremental').half()
coder1 = Autoencoder.load(model, layer=18, expansion=16, root='weights').half()

tn0 = coder0.network() / 20
tn1 = coder1.network() / 20

tree = (tn0 & tn0).contraction_tree(optimize='optimal')
cross = (tn0 & tn1).contract(all, [], optimize=tree)
self1 = (tn0 & tn0).contract(all, [], optimize=tree)
self2 = (tn1 & tn1).contract(all, [], optimize=tree)

print("Similarity:", (2 * cross.sqrt() / (self1.sqrt() + self2.sqrt())).item())
# %%