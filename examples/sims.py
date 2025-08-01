# %%
%load_ext autoreload
%autoreload 2

from coders.asparse import Autoencoder, Placeholder
from tqdm import tqdm
from itertools import product
from tqdm import tqdm

import plotly.express as px
import torch
# %%
torch.set_grad_enabled(False)
model = Placeholder(d_model=1024, name="Qwen/Qwen3-0.6B-Base")

# coders = [Autoencoder.load(model, layer=i, expansion=16, root='weights/vanilla').half() for i in tqdm(range(24))]
coders = [Autoencoder.load(model, layer=18, expansion=16, root='weights/asparse_sweep', tags=[str(i)]).half() for i in tqdm(range(1, 11))]
# %%
# Compute the similarity between all pairs of coders, this can take a few minutes.
# Similarity is computed the normalised MSE between the tensors representing the autocoders.

def similarity(a, b, reg=20.0):
    matrix = (a.sym('a') | b.sym('b')).contract(all, output_inds=['h:a', 'h:b']).data / reg
    return matrix.pow(2).sum()

results = [similarity(a, b) for a, b in tqdm(list(product(coders, coders)))]
results = torch.stack(results).reshape(len(coders), len(coders))

sims = (2 * results.sqrt()) / (results.diag().sqrt()[:, None] + results.diag().sqrt()[None, :])
px.imshow(sims.cpu().numpy(), color_continuous_scale='RdBu', color_continuous_midpoint=0.5).show()
# %%
