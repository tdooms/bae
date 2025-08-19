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
coder = Autoencoder.load(model, "ordered", layer=18, expansion=16, alpha=0.05, tags=['v2']).eval().half()
loader = DataLoader(dataset, batch_size=32, shuffle=False)
acts = []

for batch, _ in zip(loader, range(2)):
    batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    acts += [coder(**batch)['features'].half()]

acts = torch.cat(acts, dim=0)
hoyer = inv_hoyer(acts.flatten(0, -2), dim=0)
l0 = (acts.flatten(0, -2).abs() > 0.01).float().mean(dim=0)
df = pd.DataFrame.from_dict(dict(hoyer=hoyer.cpu(), l0=l0.cpu()))

fig = px.scatter(df, x=df.index, y="hoyer", opacity=0.3, marginal_y="histogram", template="plotly_white")
fig.update_yaxes(title="<b>Density</b>", range=(0, 1.005)).update_xaxes(title="<b>Latent index</b>", range=(0, 1024*16), row=1, col=1, showgrid=False)
fig.update_layout(yaxis2=dict(title=""), xaxis2=dict(title=""))
fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), width=500, height=300)
fig.update_layout(font=FONT)
fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2).update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)
fig.show()
# %%