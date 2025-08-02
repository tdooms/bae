# %%
%load_ext autoreload
%autoreload 2

from transformers import AutoModelForCausalLM, AutoTokenizer
from oldcoders.asparse import Autoencoder, Placeholder
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

from tqdm import tqdm
from scipy import stats
from quimb.tensor import Tensor
from itertools import product
from tqdm import tqdm

import plotly.express as px
import torch
import gc

# %%
torch.set_grad_enabled(False)
name = "Qwen/Qwen3-0.6B-Base"

tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cuda")
tokenize = lambda dataset: tokenizer(dataset["text"], truncation=True, padding=True, max_length=256)

dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True).with_format("torch")
dataset = Dataset.from_list(list(dataset.take(2**10))).with_format("torch")
dataset = dataset.map(tokenize, batched=True)
# %%
coder = Autoencoder.load(model, layer=18, expansion=16, root='weights/asparse').half()
# coder = Autoencoder.load(model, layer=18, expansion=16, root='weights/incremental').half()
# %%
norm = coder.left.norm(dim=-1) * coder.right.norm(dim=-1) * coder.down.norm(dim=0)
px.scatter(y=norm.cpu(), x=list(range(len(norm))), template='plotly_white', labels=dict(x="Index", y="Norm")).show()
px.histogram(norm.cpu(), nbins=100, template='plotly_white', labels=dict(x="Norm", y="Count")).show()
# %%
max_steps = 2**2
loader = DataLoader(dataset, batch_size=32, shuffle=False)
acts = []

for batch, _ in tqdm(zip(loader, range(max_steps)), total=max_steps):
    batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    output = coder(**batch)
    acts += [output['features'].half()]
    del output

acts = torch.cat(acts, dim=0)

import gc
torch.cuda.empty_cache()
gc.collect()
# %%
# Look at the correlation between activation norm and weight norm
anorm = acts.flatten(0, -2).norm(dim=0)
px.histogram(anorm.cpu(), nbins=100, template='plotly_white', labels=dict(x="Norm", y="Count")).show()
px.scatter(y=anorm.cpu(), x=norm.cpu(), template='plotly_white', labels=dict(x="Norm", y="Norm")).show()
# %%
d = coder.down / coder.down.norm(dim=0, keepdim=True)
q = d.T @ d - torch.eye(d.shape[0], device=d.device, dtype=d.dtype)
px.imshow(q[256:512, 256:512].cpu(), color_continuous_scale='RdBu', template='plotly_white', color_continuous_midpoint=0)
# %%
inds = q.flatten().topk(k=100, largest=False).indices
inds = torch.unravel_index(inds, q.shape)[0]
px.imshow(q[inds][:, inds].cpu(), color_continuous_scale='RdBu', template='plotly_white', color_continuous_midpoint=0)
# %%
def feature_dimensionality(W: torch.Tensor) -> torch.Tensor:
    W_hat = W / (W.norm(dim=1, keepdim=True) + 1e-8)
    return W.norm(dim=1).pow(2) / ((W_hat @ W.T).pow(2).sum(dim=1) + 1e-8)

d = coder.down / coder.down.norm(dim=0, keepdim=True)
dims = feature_dimensionality(d.T)
# px.histogram(dims.cpu(), nbins=100, template='plotly_white', labels=dict(x="Dimensionality", y="Count")).show()
px.scatter(y=dims.cpu(), x=list(range(len(dims))), template='plotly_white', labels=dict(x="Index", y="Dimensionality")).show()
# %%
# 484
def participation_ratio(v: torch.Tensor) -> torch.Tensor:
    v2 = v.pow(2)
    return v2.sum(dim=-1).pow(2) / (v2.pow(2).sum(dim=-1) + 1e-8)

def generalized_participation_ratio(v: torch.Tensor, p: float = 4.0) -> torch.Tensor:
    vp = v.abs().pow(p)
    numerator = vp.sum(dim=-1).pow(2)
    denominator = (vp.pow(2)).sum(dim=-1) + 1e-8
    return numerator / denominator

px.histogram(generalized_participation_ratio(q).cpu(), template='plotly_white', labels=dict(x="Index", y="Participation Ratio"), log_y=True).show()
px.scatter(y=generalized_participation_ratio(q).cpu(), x=list(range(len(q))), template='plotly_white', labels=dict(x="Index", y="Participation Ratio")).show()
# %%
q[4462].abs().topk(k=5)
# %%
