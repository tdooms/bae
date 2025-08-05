# %%
%load_ext autoreload
%autoreload 2

from transformers import AutoModelForCausalLM, AutoTokenizer
from autoencoder import Autoencoder
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

from tqdm import tqdm
from scipy import stats
from quimb.tensor import Tensor, TensorNetwork
from itertools import product
from tqdm import tqdm

import plotly.express as px
import torch
import gc

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
coder = Autoencoder.load(model, "mixed", layer=18, expansion=16, alpha=0.2, tags=[]).eval().half()
# %%
loader = DataLoader(dataset, batch_size=2, shuffle=False)

batch = next(iter(loader))
batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}

_, cache = coder.hooked(**batch)
x = (cache['acts'] * cache['acts'].square().sum(-1, keepdim=True).rsqrt())

inputs = [Tensor(x, inds=["b", "s", f"in:{i}"], tags=['X']) for i in range(3)]

tn = coder.network('a').reindex({"in:0": "in:2", "in:1": "in:3"}) | coder.network('a')
ev = TensorNetwork([tn, *inputs])
# ev.draw()
# %%
x_hat = ev.contract(all, output_inds=["b", "s", "in:3"]).data

print((x_hat - x).norm() / x.norm())
# px.bar(torch.stack([x, x_hat])[:, 0, 31, :256].T.cpu(), barmode='group')
# px.bar((x_hat / x)[0, 32, :256].cpu(), log_y=True)
# %%
loader = DataLoader(dataset, batch_size=2, shuffle=False)

batch = next(iter(loader))
batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}

_, cache = coder.hooked(**batch)
x = (cache['acts'] * cache['acts'].square().sum(-1, keepdim=True).rsqrt())
x = x[:, 1:3]

inputs = [Tensor(x, inds=["b", "s", f"in:{i}"], tags=['X']) for i in range(2)]

tn = coder.network('a').reindex({"in:0": "in:2", "in:1": "in:3"}) | coder.network('a')
ev = TensorNetwork([tn, *inputs])

x_hat = ev.contract(all, output_inds=["b", "s", "in:2", "in:3"]).data
# %%
sym = 0.5 * (x_hat[0, 0] + x_hat[0, 0].T).float()
vals, vecs = torch.linalg.eigh(sym)
px.line(vals.cpu())
# %%
px.bar(torch.stack([x[0, 0],vecs[:, -1]])[:, :256].T.cpu(), barmode='group').show()
print((x[0, 0] - vecs[:, -1]).norm())
