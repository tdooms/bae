# %%
%load_ext autoreload
%autoreload 2

import plotly.express as px
import torch

from tqdm import tqdm
from itertools import product
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from autoencoder import Autoencoder, Placeholder
from utils.functions import hoyer

# %%
torch.set_grad_enabled(False)
model = Placeholder(d_model=1024, name="Qwen/Qwen3-0.6B-Base")

# coders = [Autoencoder.load(model, "mixed", layer=18, expansion=16, alpha=i/10).half() for i in tqdm(range(5))]

coders = [Autoencoder.load(model, "mixed", layer=18, expansion=16, alpha=0.2, tags=[f"checkpoint-{i*1024}"]).half() for i in tqdm(range(1, 15))]

# coders = [Autoencoder.load(model, "mixed", layer=18, expansion=16, alpha=i/10).half() for i in tqdm(range(5))]
# coders = [Autoencoder.load(model, "mixed", layer=18, expansion=16, alpha=0.2, tags=[f'repeat{i}']).half() for i in tqdm(range(10))]
# %%
# Compute the similarity between all pairs of coders, this can take a few minutes.
# Similarity is computed the normalised MSE between the tensors representing the autocoders.

def similarities(a, b, reg=20.0, lite=False):
    # Compute the norm of the kernel matrix
    matrix = (a.network('a') | b.network('b')).contract(all, output_inds=['h:a', 'h:b']).data / reg
    norm = matrix.pow(2).sum()
    
    # Quick return if we're only interested in the norm
    if lite: return dict(norm=norm.item(), partial=1.0, sparsity=1.0)
    
    # Compute the best permutation of the kernel matrix using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(matrix.abs().cpu().numpy(), maximize=True)
    partial = matrix[row_ind, col_ind].pow(2).sum() / norm
    
    # Compute the sparsity, energy ratio, and procrustes error
    sparsity = hoyer(matrix).mean()

    return dict(norm=norm, partial=partial, sparsity=sparsity)

results = [similarities(a, b) for a, b in tqdm(list(product(coders, coders)))]
norms = torch.tensor([r['norm'] for r in results]).reshape(len(coders), len(coders))
partials = torch.tensor([r['partial'] for r in results]).reshape(len(coders), len(coders))
sparsities = torch.tensor([r['sparsity'] for r in results]).reshape(len(coders), len(coders))
sims = (2 * norms.sqrt()) / (norms.diag().sqrt()[:, None] + norms.diag().sqrt()[None, :])

names = ["Frobenius", "Permutation", "Sparsity"]
stacked = torch.stack([sims, partials, sparsities], dim=0)
fig = px.imshow(stacked.cpu(), color_continuous_scale='RdBu', color_continuous_midpoint=0.5, facet_col=0)
fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
fig.for_each_annotation(lambda a: a.update(text=names[int(a.text.split("=")[-1])], font_size=16))
fig.show()
# %%

kernel = coders[0].down @ ((coders[0].left @ coders[1].left.T) * (coders[0].right @ coders[1].right.T)) @ coders[1].down.T
px.histogram(kernel[2].cpu(), log_y=True)