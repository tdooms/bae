# %%
%load_ext autoreload
%autoreload 2

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from autoencoder import Autoencoder
from utils.functions import inv_hoyer

import plotly.express as px
import torch
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
max_steps = 2**2
hoyer, l0 = [], []
    
for i in tqdm(range(11)):
    coder = Autoencoder.load(model, "rainbow", layer=18, expansion=16, alpha=i/10).eval().half()
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    acts = []

    for batch, _ in zip(loader, range(max_steps)):
        batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
        acts += [coder(**batch)['features'].half()]

    acts = torch.cat(acts, dim=0)
    hoyer += [inv_hoyer(acts.flatten(0, -2), dim=0)]
    l0 += [(acts.flatten(0, -2).abs() > 0.05).sum(dim=-1)]

hoyer = torch.stack(hoyer, dim=0)
l0 = torch.stack(l0, dim=0)
# %%
inds = torch.tensor([0, 1, -1])
colors = [px.colors.sequential.Viridis[i] for i in [0, 3, 8]]

fig = px.histogram(hoyer[inds].cpu().T, log_y=False, barmode='overlay', template='plotly_white', color_discrete_sequence=colors) 
fig.update_layout(yaxis_title="", showlegend=False)
fig.update_yaxes(showticklabels=False).update_xaxes(title="<b>Hoyer sparsity</b>")
fig.show()
# %%
px.histogram(l0.cpu().T, barmode='overlay', log_y=False)
# %%