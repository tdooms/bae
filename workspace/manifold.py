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
name = "Qwen/Qwen3-0.6B-Base"

tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cuda")
tokenize = lambda dataset: tokenizer(dataset["text"], truncation=True, padding=True, max_length=256)

dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True).with_format("torch")
dataset = Dataset.from_list(list(dataset.take(2**11))).with_format("torch")
dataset = dataset.map(tokenize, batched=True)
# %%
# coder = Autoencoder.load(model, layer=18, expansion=16, root='weights/sweep', tags=['10']).half()
coder = Autoencoder.load(model, "biased", layer=18, expansion=16, alpha=1.0).half()
vis = Feature(coder, tokenizer, dataset, max_steps=2**4, batch_size=2**5)
# %%
d = coder.down / coder.down.norm(dim=0, keepdim=True)
g = d.T @ d

gpr = generalized_effective_dimension(g)
px.scatter(y=gpr.cpu(), x=list(range(gpr.size(-1))), template='plotly_white', title="Number of active elements in the overlap matrix").show()
print(gpr.topk(200).indices.tolist()[150:])
# %%
# ---- Manifolds in the paper ----
# idx = 9788 # this/these/those
idx = 5881 # although/while
# idx = 15612 # none/not/any/little
# idx = 3569 # first number in year (circle)

# ---- Other cool manifolds ----
# idx = 13346 # make/made
# idx = 3732 # predict numerals, numeric literals
# idx = 3294 # numbers in years

fig = px.histogram(g[idx].cpu(), template='plotly_white', log_y=True, width=500, height=300, range_x=[-1.1, 1.1])
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False).show()

vals, inds = g[idx].abs().topk(k=10)
vis(inds[:5].tolist(), k=3)
# %%
form = einsum(vals, coder.left[inds], coder.right[inds], "out, out in1, out in2 -> in1 in2")
# density = einsum(torch.randn(5), torch.randn(5, 1024), torch.randn(5, 1024), "out, out in1, out in2 -> in1 in2").to("cuda")
form = 0.5 * (form + form.T)

manifold = Manifold(dataset, coder.hooked, tokenizer, form, max_steps=2**6)
manifold.spectrum().show()

manifold(k=75_000)
# %%