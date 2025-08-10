# %%
%load_ext autoreload
%autoreload 2

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import einsum

from autoencoder import Autoencoder
from utils.functions import inv_hoyer
from figures.constants import FONT, COLORS

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

loader = DataLoader(dataset, batch_size=64, shuffle=False)
batch = next(iter(loader))
batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}

def incremental(_coder):
    features = _coder(**batch)['features'].half()
    
    kernel = _coder.kernel()
    mask = torch.zeros_like(kernel[0])

    recons = []
    x = list(range(0, _coder.config.d_features + 1, 64))
    for k in tqdm(x):
        mask[:k] = 1
        recons += [einsum(features * mask, features * mask, kernel, "... d1, ... d2, d1 d2 -> ...").mean().item()]

    return pd.DataFrame({'x': x, 'recons': recons, 'kind': [coder.config.kind] * len(recons)})

# Get reconstructions for ordered coder
coder = Autoencoder.load(model, "ordered", layer=18, expansion=16, alpha=0.1, tags=[], root="//wsl.localhost/Ubuntu/home/thomas/bae/weights").eval().half()
ordered = incremental(coder)

# Get reconstructions for vanilla coder
coder = Autoencoder.load(model, "vanilla", layer=18, expansion=16, alpha=0.1, tags=[], root="//wsl.localhost/Ubuntu/home/thomas/bae/weights").eval().half()
vanilla = incremental(coder)

# Reorder vanilla reconstructions by largest increases
vanilla_data = vanilla.copy()
vanilla_recons_values = np.array(vanilla_data['recons'])
vanilla_diffs = np.diff(vanilla_recons_values)

# Get indices sorted by largest differences (descending)
sorted_indices = np.argsort(-vanilla_diffs)

# Create new ordering that puts largest increases first
reordered_recons = [vanilla_recons_values[0]]  # Start with first value
for idx in sorted_indices:
    reordered_recons.append(reordered_recons[-1] + vanilla_diffs[idx])

# Create new dataframe with reordered vanilla data
reordered = pd.DataFrame({
    'x': vanilla_data['x'], 
    'recons': reordered_recons, 
    'kind': ['reordered'] * len(reordered_recons)
})

df = pd.concat([ordered, vanilla, reordered], ignore_index=True)
# %%
fig = px.line(df, x='x', y='recons', color='kind', template="plotly_white", width=600, height=400, color_discrete_map=COLORS)
fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), font=FONT)
fig.update_xaxes(title_text="<b>Latent index</b>")
fig.update_yaxes(title_text="<b>Reconstruction error</b>", range=(0, 0.805))
fig.update_layout(showlegend=True, legend=dict(title="", orientation="h", x=0.5, xanchor="center", y=1.02, yanchor="bottom"))
fig.update_traces(line=dict(width=3))
# %%
fig.write_image("C:/Users/thoma/Downloads/ordered-recons.svg")
# %%