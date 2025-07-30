# %%
%load_ext autoreload
%autoreload 2

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from datasets import load_dataset
from utils.vis import Vis
from coders.vsparse import Autoencoder

from einops import rearrange, einsum
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import stats

import plotly.express as px
import torch
import pandas as pd
# %%
torch.set_grad_enabled(False)
name = "Qwen/Qwen3-0.6B-Base"

tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cuda")
tokenize = lambda dataset: tokenizer(dataset["text"], truncation=True, padding=True, max_length=256)

dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True).with_format("torch")
dataset = Dataset.from_list(list(dataset.take(2**12))).with_format("torch")
dataset = dataset.map(tokenize, batched=True)
# %%
coder = Autoencoder.load(model, layer=18, expansion=16, root='weights').half()
# %%
vis = Vis(coder, tokenizer, dataset, max_steps=2**6, batch_size=2**5)
# %%
# This generates a set of candidate manifolds
# It's a bit inconsistent currently, many dense features are in this list, this probably has a reason but it's as of yet unclear.
# It still requires some manual filtering
q = einsum(coder.down, coder.down, "mid out, mid inp -> out inp")
q = q - torch.diag_embed(torch.diagonal(q, dim1=-2, dim2=-1))
q.topk(dim=1, k=2).values[:, 0].topk(k=50).indices
# %%
# idx = 812  # capital how/How
# idx = 7973   # according to?
idx = 2697
q = einsum(coder.down, coder.down, "mid out, mid inp -> out inp")
px.histogram(q[idx].cpu()).show()
vis(q[idx].topk(k=3).indices.tolist(), k=5)

inds = q[idx].topk(k=3).indices
scales = q[idx].topk(k=3).values
# %%
d = einsum(coder.down[:, inds], coder.down, "mid out, mid inp -> out inp")
l = coder.left
r = coder.right

loader = DataLoader(dataset, batch_size=32, shuffle=False)
max_steps = 2**6
vals, acts, inputs = [], [], []

for batch, _ in tqdm(zip(loader, range(max_steps)), total=max_steps):
    batch = {k: v.to(coder.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}    
    cache = coder(**batch)
    
    inputs += [batch['input_ids']]
    acts += [cache['x']]
    vals += [einsum(d, r, l, cache['x'], cache['x'], "out mid, mid in1, mid in2, ... in1, ... in2 -> ... out")]
    # vals += [einsum(r[inds], l[inds], cache['x'], cache['x'], "out in1, out in2, ... in1, ... in2 -> ... out")]

vals, acts, inputs = map(torch.cat, [vals, acts, inputs])
vals, acts, inputs = vals.flatten(0, -2), acts.flatten(0, -2), inputs.flatten()
# %%
k = 3000
max_vals, max_inds = einsum(vals, scales, "... o, o -> ...").topk(k=k, dim=0, largest=True)
min_vals, min_inds = einsum(vals, scales, "... o, o -> ...").topk(k=k, dim=0, largest=False)
top_vals, top_inds = torch.cat([max_vals, min_vals], dim=0), torch.cat([max_inds, min_inds], dim=0)

top_acts = acts[top_inds]
tokens = [s.replace('Ġ', ' ') for s in tokenizer.convert_ids_to_tokens(inputs[top_inds].cpu())]

b = einsum(d, l, r, "out mid, mid in1, mid in2 -> out in1 in2")
u, s, v = torch.svd(b.flatten(1).float())
v = 0.5 * (v.view(1024, 1024, 3).permute(2, 0, 1) + v.view(1024, 1024, 3).permute(2, 1, 0))

_, vecs = torch.linalg.eigh(v)
proj = torch.stack([vecs[0, :, 0]*s[0], vecs[0, :, -1]*s[0], vecs[1, :, 0]*s[1], vecs[1, :, -1]*s[1], vecs[2, :, 0]*s[2], vecs[2, :, -1]*s[2]], dim=0)
u, s, v = torch.svd(proj)

p = einsum(top_acts, v[:, :3].half(), "batch d, d out -> batch out").cpu().numpy()
fig = px.scatter_3d(
    x=p[:, 0],
    y=p[:, 1],
    z=p[:, 2],
    color=top_vals.cpu(),
    # color=vals[..., 0][top_inds].cpu(),
    color_continuous_midpoint=0.0,
    color_continuous_scale="RdBu",
    hover_name=tokens,
    # hover_name=top_inds.cpu(),
    height=600, 
    width=800
)

fig.update_layout(
    scene_camera=dict(
        eye=dict(x=0.7, y=-0.7, z=1.2),  # Camera position
        center=dict(x=0, y=0, z=0),     # Point camera looks at
        up=dict(x=0, y=0, z=1)          # Up direction
    )
)

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False), coloraxis_showscale=False)
# %%
