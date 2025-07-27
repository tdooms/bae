# %%
%load_ext autoreload
%autoreload 2

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset

from utils.vis import Vis
from coders.sparse import Autoencoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange

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
coder = Autoencoder.load(model, layer=18, expansion=16, root='weights').half()
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

def hoyer(x, dim=-1):
    return (x.norm(p=1, dim=dim) / x.norm(p=2, dim=dim) - 1.0) / (x.shape[-1]**0.5 - 1.0)

w = hoyer(rearrange(acts, "... d -> (...) d").T.cpu())
# w = w[acts.norm(dim=(0, 1)).cpu() > 0.1]
px.histogram(w).show()
px.scatter(x=list(range(len(w))), y=w.cpu(), template='plotly_white', labels=dict(x="Index", y="Hoyer Norm")).show()
print(w.topk(k=10, largest=False))
# %%
px.histogram(acts[20, 20].cpu(), log_y=True)
# %%
