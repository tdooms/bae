# %%
%load_ext autoreload
%autoreload 2

import torch
import plotly.express as px



# %%
torch.set_grad_enabled(False)

m = torch.tril(torch.ones(20, 20))

px.imshow(m[:, 5:10].T @ m[:, 5:10])
# %%

px.imshow(torch.arange(1, 6).flip(0).expand(5, -1))
# %%
(torch.tril(torch.ones(5, 5)) @ torch.tril(torch.ones(5, 5))) + k
# %%
from einops import einsum
triu = torch.triu(torch.ones(5, 5))
order = einsum(triu, triu, "f1 reg, f2 reg -> f1 f2")
px.imshow(order)
# %%
%load_ext autoreload
%autoreload 2

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset

import torch
from utils.vis import Vis
from autoencoder import Autoencoder
# %%
torch.set_grad_enabled(False)
name = "Qwen/Qwen3-0.6B-Base"

autoencoder = Autoencoder.from_pretrained(name, kind="vanilla", layer=18, expansion=16, alpha=0.3, tags=['test']).cuda().type(torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto", device_map="auto")

tokenize = lambda dataset: tokenizer(dataset["text"], truncation=True, padding=True, max_length=256)

dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True).with_format("torch")
dataset = Dataset.from_list(list(dataset.take(2**11))).with_format("torch")
dataset = dataset.map(tokenize, batched=True)
# %%
vis = Vis(model, autoencoder, dataset, tokenizer, batches=32)
# %%
vis(list(range(10)))