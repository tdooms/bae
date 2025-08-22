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
    matrix = tn.contract(all, output_inds=['h:a', 'h:b'], optimize="auto-hq").data if kind in ["mixed", "combined"] else tn.contract(all, output_inds=['f:a', 'f:b']).data
    return matrix.pow(2).mean().item()
# %%
sims = []
# Create figure x, takes about ~4 minutes on my machine
for kind in ["vanilla", "mixed", "ordered", "combined"]:
    coders = [Autoencoder.load(model, kind, layer=18, expansion=16, alpha=i/10, hf=True).eval().half() for i in tqdm(range(11))]

    norms = [norm(a, b, kind) for a, b in tqdm(list(product(coders, coders)))]
    norms = torch.tensor(norms).reshape(len(coders), len(coders))

    sims += [(2 * norms.sqrt()) / (norms.diag().sqrt()[:, None] + norms.diag().sqrt()[None, :])]
sims = torch.stack(sims, dim=0)
# %%
names = ["<b>Vanilla</b>", "<b>Mixed</b>", "<b>Ordered</b>", "<b>Combined</b>"]
fig = px.imshow(sims.cpu(), color_continuous_scale='RdBu', color_continuous_midpoint=0.5, facet_col=0, width=1170, height=300)
fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
fig.for_each_annotation(lambda a: a.update(text=names[int(a.text.split("=")[-1])], font_size=16))
fig.update_layout(showlegend=False, margin=dict(l=30, r=10, t=22, b=15))

# Add arrow pointing from sparse to dense
fig.add_annotation(
    x=-0.01, y=0.04,
    xref="paper", yref="paper",
    showarrow=True,
    ax=0, ay=-150,
    arrowcolor="#282828",
    axref="pixel", ayref="pixel",
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2
)

fig.add_annotation(
    x=-0.01, y=0.96,
    xref="paper", yref="paper",
    showarrow=True,
    ax=0, ay=150,
    arrowcolor="#282828",
    axref="pixel", ayref="pixel",
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2
)

fig.add_annotation(
    text="<b>dense</b>",
    x=-0.032, y=0.93,
    font=dict(size=14),
    xref="paper", yref="paper",
    showarrow=False,
    textangle=-90
)
fig.add_annotation(
    text="<b>sparse</b>",
    x=-0.032, y=0.07,
    font=dict(size=14),
    xref="paper", yref="paper",
    showarrow=False,
    textangle=-90
)

for start, end in [(0, 0.235), (0.255, 0.49), (0.51, 0.745), (0.765, 1.0)]:
    fig.add_annotation(
        x=start, y=0.015,
        xref="paper", yref="paper",
        showarrow=True,
        ax=150, ay=0,
        arrowcolor="#282828",
        axref="pixel", ayref="pixel",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2
    )

    fig.add_annotation(
        x=end, y=0.015,
        xref="paper", yref="paper",
        showarrow=True,
        ax=-150, ay=0,
        arrowcolor="#282828",
        axref="pixel", ayref="pixel",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2
    )

    fig.add_annotation(
        text="<b>dense</b>",
        x=start + 0.03, y=-0.06,
        font=dict(size=14),
        xref="paper", yref="paper",
        xanchor="center",
        showarrow=False,
    )
    fig.add_annotation(
        text="<b>sparse</b>",
        x=end - 0.03, y=-0.06,
        font=dict(size=14),
        xref="paper", yref="paper",
        xanchor="center",
        showarrow=False,
    )

fig.update_layout(font=FONT)
fig.show()
# %%
fig.write_image("C:/Users/thoma/Downloads/frobenius.pdf")
# %%