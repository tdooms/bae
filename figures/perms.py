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

from autoencoders import Autoencoder, Placeholder
from utils.functions import hoyer
from itertools import combinations_with_replacement

# %%
torch.set_grad_enabled(False)
model = Placeholder(d_model=1024, name="Qwen/Qwen3-0.6B-Base")

coders = [Autoencoder.load(model, "vanilla", layer=18, expansion=16, alpha=i/10, hf=True).half() for i in tqdm(range(11))]
# %%
# Compute the best permutation between pairs of coders, this takes like half an hour.
# Similarity is computed as the fraction of the permutation matrix norm and the full frobenius norm.

def metrics(a, b, reg=20.0):
    # Compute the norm of the kernel matrix
    tn = (a.network('a') | b.network('b'))
    matrix = tn.contract(all, output_inds=['f:a', 'f:b']).data / reg

    # This is just way too slow man
    # # Compute the best permutation of the kernel matrix using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(matrix.abs().cpu().numpy(), maximize=True)
    perm = matrix[row_ind, col_ind].pow(2).sum().sqrt() / matrix.pow(2).sum().sqrt()
    return perm

# Only sample half the combinations since this should be symmetric
perms = torch.zeros(len(coders), len(coders))

for i, j in tqdm(list(combinations_with_replacement(range(len(coders)), 2))):
    perm = metrics(coders[i], coders[j])
    perms[i, j] = perm
    perms[j, i] = perm
# %%
fig = px.imshow(perms, color_continuous_scale='RdBu', color_continuous_midpoint=0.5, height=400, width=443, zmin=0, zmax=1)
fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=22, b=10), font=FONT)
fig.for_each_annotation(lambda a: a.update(text=['Absolute', 'Relative'][int(a.text.split("=")[-1])], font_size=16))
fig.show()
# %%
fig.write_image("C:/Users/thoma/Downloads/permutation.svg")
# %%