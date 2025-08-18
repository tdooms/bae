# %%
%load_ext autoreload
%autoreload 2

from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import Dataset, load_dataset
from einops import einsum

from utils.feature import Feature
from utils.manifold import Manifold
from utils.functions import *
from autoencoder import Autoencoder

import plotly.express as px
import torch
# %%
torch.set_grad_enabled(False)
# name = "Qwen/Qwen3-0.6B-Base"
name = "google/gemma-3-270m"

tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto", device_map="cuda")
tokenize = lambda dataset: tokenizer(dataset["text"], truncation=True, padding=True, max_length=256)

dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True).with_format("torch")
dataset = Dataset.from_list(list(dataset.take(2**11))).with_format("torch")
dataset = dataset.map(tokenize, batched=True)
# %%
# coder = Autoencoder.load(model, "mani", layer=18, expansion=16, alpha=1.0).half()
coder = Autoencoder.load(model, "mixed", layer=12, expansion=16, alpha=1.0)
vis = Feature(coder, tokenizer, dataset, max_steps=2**4, batch_size=2**5)
# %%
idx = 5

# l = torch.cat([coder.left.bias[:, None], coder.left.weight], dim=1)
# r = torch.cat([coder.right.bias[:, None], coder.right.weight], dim=1)
# d = coder.down.weight[idx]

l = coder.left
r = coder.right
d = coder.down

form = einsum(d[idx], l, r, "out, out in1, out in2 -> in1 in2")
form = 0.5 * (form + form.T)

vals, vecs = torch.linalg.eigh(form.float())

fig = px.line(torch.cat([vals[:20], vals[-20:]]).cpu(), template='plotly_white', width=500, height=300)
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False).show()

vis(idx, k=5)
# %%
manifold = Manifold(dataset, coder.hooked, tokenizer, form, max_steps=2**4)
manifold(k=2**15)
# %%