# %%
%load_ext autoreload
%autoreload 2

import plotly.express as px
import torch

from tqdm import tqdm
from itertools import product
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from figures.constants import FONT

from autoencoder import Autoencoder, Placeholder
from utils.functions import hoyer

# %%
torch.set_grad_enabled(False)
model = Placeholder(d_model=1024, name="Qwen/Qwen3-0.6B-Base")

coders = [Autoencoder.load(model, "rainbow", layer=18, expansion=16, alpha=i/10).half() for i in tqdm(range(11))]
# %%
# Compute the similarity between all pairs of coders, this can take a few minutes.
# Similarity is computed the normalised error between the tensors representing the autocoders.

def similarities(a, b, reg=20.0):
    # Compute the norm of the kernel matrix
    tn = (a.network('a') | b.network('b'))
    matrix = tn.contract(all, output_inds=['h:a', 'h:b']).data / reg
    norm = matrix.pow(2).sum()
    
    # # Compute the best permutation of the kernel matrix using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(matrix.abs().cpu().numpy(), maximize=True)
    perm = matrix[row_ind, col_ind].pow(2).sum() / norm
    
    return perm

perms = [similarities(a, b) for a, b in tqdm(list(product(coders, coders)))]
perms = torch.tensor(perms).reshape(len(coders), len(coders))
# %%
fig = px.imshow(perms.cpu(), color_continuous_scale='RdBu', color_continuous_midpoint=0.5, height=400, width=600)
fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=22, b=10), font=FONT)
fig.show()
# %%
a, b = coders[0], coders[0]
tn = (a.network('a') | b.network('b'))
matrix = tn.contract(all, output_inds=['f:a', 'f:b']).data / 20.0
# vals, vecs = torch.linalg.eigh(matrix)

torch.linalg.svdvals((coders[0].left + coders[0].right)[torch.tensor([1273, 10943, 10091,  1822])].float())

# px.histogram(matrix.flatten().cpu())

# row_ind, col_ind = linear_sum_assignment(matrix.abs().cpu().numpy(), maximize=True)
# partial = matrix[row_ind, col_ind].pow(2).sum() / norm

# print(f"Norm: {norm:.4f}, Partial: {partial:.4f}")