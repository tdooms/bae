# %%
%load_ext autoreload
%autoreload 2

from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import Dataset, load_dataset
from einops import einsum

from utils.vis import Vis
from utils.manifold import Manifold
from utils.functions import *
from oldcoders.asparse import Autoencoder

import plotly.express as px
import torch
# %%
torch.set_grad_enabled(False)
name = "Qwen/Qwen3-0.6B-Base"

tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cuda")
tokenize = lambda dataset: tokenizer(dataset["text"], truncation=True, padding=True, max_length=256)

dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True).with_format("torch")
dataset = Dataset.from_list(list(dataset.take(2**11))).with_format("torch")
dataset = dataset.map(tokenize, batched=True)
# %%
# coder = Autoencoder.load(model, layer=18, expansion=16, root='weights/sweep', tags=['10']).half()
coder = Autoencoder.load(model, layer=18, expansion=16, root='weights/sweep', tags=['2']).half()
vis = Vis(coder, tokenizer, dataset, max_steps=2**4, batch_size=2**5)
# %%
d = coder.down / coder.down.norm(dim=0, keepdim=True)
g = d.T @ d

# I recommend not looking at the say top 5/10-ish. 
# There's some dense features which I don't quite understand yet.
# Their manifolds are interesting though.

gpr = generalized_effective_dimension(g)
px.scatter(y=gpr.cpu(), x=list(range(gpr.size(-1))), template='plotly_white', title="Number of active elements in the overlap matrix").show()
print(gpr.topk(50).indices.tolist())
# %%
# These are for the coder with '10' tag
# idx = 602
# idx = 11023
# idx = 7695

# These are for the coder with '2' tag
# idx = 10313  # abbreviation manifold
# idx = 3338 # ( manifold
# idx = 7695 # triangle for paper
# idx = 2062 # colon prediction for paper
# idx = 15620 # numbers!
idx = 11746

fig = px.histogram(g[idx].cpu(), template='plotly_white', log_y=True, width=500, height=300, range_x=[-1.1, 1.1])
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False).show()

vals, inds = g[idx].abs().topk(k=5)
vis(inds[:5].tolist(), k=3)
# %%
# density = einsum(coder.down[:, 11746], coder.down, coder.left, coder.right, "out, out mid, mid in1, mid in2 -> in1 in2")
# density = einsum(coder.down[2], coder.left, coder.right, "mid, mid in1, mid in2 -> in1 in2")
density = einsum(vals, coder.left[inds], coder.right[inds], "out, out in1, out in2 -> in1 in2")
density = 0.5 * (density + density.T)

manifold = Manifold(dataset, coder.hooked, tokenizer, density, max_steps=2**5)
manifold.spectrum().show()

manifold(k=30_000)
# %%