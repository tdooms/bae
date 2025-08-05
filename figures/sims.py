# %%
%load_ext autoreload
%autoreload 2

import plotly.express as px
import torch

from tqdm import tqdm
from itertools import product
from tqdm import tqdm
from autoencoder import Autoencoder, Placeholder

# %%
torch.set_grad_enabled(False)
model = Placeholder("Qwen/Qwen3-0.6B-Base", d_model=1024)

# def polar(m):   # express polar decomposition in terms of singular-value decomposition
#     U, S, Vh = torch.linalg.svd(m.float())
#     u = U @ Vh
#     p = Vh.T.conj() @ S.diag().to(dtype=m.dtype) @ Vh
#     return  u, p

def norm(a, b):
    """Compute the norm of the kernel matrix"""
    matrix = (a.network('a') | b.network('b')).contract(all, output_inds=['h:a', 'h:b']).data
    return matrix.pow(2).mean().item()
# %%
# Create figure x, takes about ~15 minutes on my machine
coders = [Autoencoder.load(model, "rainbow", layer=i, expansion=16, alpha=0.1).eval().half() for i in tqdm(range(10))]

norms = [norm(a, b) for a, b in tqdm(list(product(coders, coders)))]
norms = torch.tensor(norms).reshape(len(coders), len(coders))

sims = (2 * norms.sqrt()) / (norms.diag().sqrt()[:, None] + norms.diag().sqrt()[None, :])
# %%
names = ["<b>Rainbow</b>"]
stacked = torch.stack([sims], dim=0)
fig = px.imshow(stacked.cpu(), color_continuous_scale='RdBu', color_continuous_midpoint=0.5, facet_col=0, width=800, height=400)
fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
fig.for_each_annotation(lambda a: a.update(text=names[int(a.text.split("=")[-1])], font_size=16))
fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
fig.show()
# %%

# results = [similarities(a, b) for a, b in tqdm(list(product(coders, coders)))]
# norms = torch.tensor([r['norm'] for r in results]).reshape(len(coders), len(coders))
# partials = torch.tensor([r['partial'] for r in results]).reshape(len(coders), len(coders))
# sparsities = torch.tensor([r['sparsity'] for r in results]).reshape(len(coders), len(coders))
# sims = (2 * norms.sqrt()) / (norms.diag().sqrt()[:, None] + norms.diag().sqrt()[None, :])

# names = ["Frobenius", "Permutation", "Sparsity"]
# stacked = torch.stack([sims, partials, sparsities], dim=0)
# fig = px.imshow(stacked.cpu(), color_continuous_scale='RdBu', color_continuous_midpoint=0.5, facet_col=0)
# fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
# fig.for_each_annotation(lambda a: a.update(text=names[int(a.text.split("=")[-1])], font_size=16))
# fig.show()
# %%

# coders = [Autoencoder.load(model, "rainbow", layer=18, expansion=16, alpha=i/10).half() for i in tqdm(range(10))]