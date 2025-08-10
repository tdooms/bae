# %%
%load_ext autoreload
%autoreload 2

import plotly.express as px
import torch

from tqdm import tqdm
from itertools import product
from tqdm import tqdm
from autoencoder import Autoencoder, Placeholder
from figures.constants import FONT

# %%
torch.set_grad_enabled(False)
model = Placeholder("Qwen/Qwen3-0.6B-Base", d_model=1024)

def norm(a, b, kind):
    """Compute the norm of the kernel matrix"""
    tn = a.network('a') | b.network('b')
    matrix = tn.contract(all, output_inds=['h:a', 'h:b'], optimize="auto-hq").data if kind in ["mixed", "rainbow"] else tn.contract(all, output_inds=['f:a', 'f:b']).data
    return matrix.pow(2).mean().item()
# %%
sims = []
# Create figure x, takes about ~4 minutes on my machine
for kind in ["vanilla", "mixed", "ordered", "rainbow"]:
    coders = [Autoencoder.load(model, kind, layer=18, expansion=16, alpha=i/10, root="//wsl.localhost/Ubuntu/home/thomas/bae/weights").eval().half() for i in tqdm(range(11))]

    norms = [norm(a, b, kind) for a, b in tqdm(list(product(coders, coders)))]
    norms = torch.tensor(norms).reshape(len(coders), len(coders))

    sims += [(2 * norms.sqrt()) / (norms.diag().sqrt()[:, None] + norms.diag().sqrt()[None, :])]
sims = torch.stack(sims, dim=0)
# %%
names = ["<b>Vanilla</b>", "<b>Mixed</b>", "<b>Ordered</b>", "<b>Combined</b>"]
fig = px.imshow(sims.cpu(), color_continuous_scale='RdBu', color_continuous_midpoint=0.5, facet_col=0, width=1170, height=300)
fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
fig.for_each_annotation(lambda a: a.update(text=names[int(a.text.split("=")[-1])], font_size=16))
fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=22, b=10), font=FONT)
fig.show()
# %%
fig.write_image("C:/Users/thoma/Downloads/sparsities.svg")
# %%
from scipy.optimize import linear_sum_assignment
d = coders[0].config.d_features

a = Autoencoder.load(model, "rainbow", layer=18, expansion=16, alpha=0.1).eval().half()
b = Autoencoder.load(model, "rainbow", layer=18, expansion=16, alpha=0.2).eval().half()

tn = (a.network('a') | b.network('b'))
matrix = tn.contract(all, output_inds=['h:a', 'h:b']).data / 20.0
norm = matrix.pow(2).sum()

row_ind, col_ind = linear_sum_assignment(matrix.abs().cpu().numpy(), maximize=True)
partial = matrix[row_ind, col_ind].pow(2).sum() / norm

print(f"Norm: {norm:.4f}, Partial: {partial:.4f}")
# %%
