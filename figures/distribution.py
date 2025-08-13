# %%
%load_ext autoreload
%autoreload 2

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from autoencoder import Autoencoder
from utils.functions import inv_hoyer
from figures.constants import FONT

import pandas as pd
import plotly.express as px
import torch
import numpy as np
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
max_steps = 2**4
coder = Autoencoder.load(model, "vanilla", layer=18, expansion=16, alpha=1.0, tags=[], hf=True).eval().half()
loader = DataLoader(dataset, batch_size=32, shuffle=False)
acts = []

for batch, _ in zip(loader, range(max_steps)):
    batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    acts += [coder(**batch)['features'].half()]
acts = torch.cat(acts, dim=0)
# %%
fig = px.histogram(acts[..., 0].flatten().cpu().neg(), template="plotly_white", log_y=True, width=600, height=300)
fig.update_layout(showlegend=False, font=FONT, margin=dict(l=10, r=10, t=10, b=10))
fig.update_xaxes(title_text="Activation Value").update_yaxes(title_text="Count")
# %%