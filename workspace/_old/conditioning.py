# %%
%load_ext autoreload
%autoreload 2

from transformers import AutoModelForCausalLM, AutoTokenizer
from autoencoders import Autoencoder
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
from torch.nn import CrossEntropyLoss

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
coder = Autoencoder.load(model, "mixed", layer=18, expansion=24, alpha=0.1, tags=[]).eval().half()
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
# px.bar((x_hat / x)[0, 32, :256].cpu())

px.scatter(x=x[0, 32].cpu(), y=(x_hat / x)[0, 32].abs().cpu(), log_y=True)
# %%
loader = DataLoader(dataset, batch_size=2, shuffle=False)

batch = next(iter(loader))
batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}

_, cache = coder.hooked(**batch)
x = (cache['acts'] * cache['acts'].square().sum(-1, keepdim=True).rsqrt())
x = x[:, :16]

inputs = [Tensor(x, inds=["b", "s", f"in:{i}"], tags=['X']) for i in range(2)]

tn = coder.network('a').reindex({"in:0": "in:2", "in:1": "in:3"}) | coder.network('a')
ev = TensorNetwork([tn, *inputs])

prod = ev.contract(all, output_inds=["b", "s", "in:2", "in:3"]).data

idx = 11
sym = 0.5 * (prod[0, idx] + prod[0, idx].T).float()
vals, vecs = torch.linalg.eigh(sym)
px.line(vals.cpu()).show()

x_hat = vecs[:, -1]
px.bar(torch.stack([x[0, idx], x_hat])[:, :256].T.cpu(), barmode='group').show()
print((x[0, idx] - x_hat).norm())
# %%
# Replace the representations at layer 18 with the reconstructed ones and compare the loss

# Prepare a new batch for evaluation
loader = DataLoader(dataset, batch_size=2, shuffle=False)
batch = next(iter(loader))
batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}

# Get original activations
_, cache = coder.hooked(**batch)
original_acts = cache['acts']

# Function to patch activations at layer 18
def patch_activations(module, input, output):
    ret = input[0].clone()
    # ret[0, idx, :] = x_hat
    ret[0, idx, :] = x_hat[0, idx, :]
    return ret,

# Register hook to replace activations at layer 18
handle = model.model.layers[18].register_forward_hook(patch_activations)
outputs = model(**batch)
handle.remove()

loss_fn = CrossEntropyLoss()
print(loss_fn(outputs.logits[0, idx], batch['input_ids'][0, idx+1]))

outputs = model(**batch)
print(loss_fn(outputs.logits[0, idx], batch['input_ids'][0, idx+1]))


# Shift logits and labels for next-token prediction
# shift_logits = outputs.logits[0, :, :].contiguous()
# shift_labels = batch['input_ids'][0, 1:5].contiguous()

# loss_reconstructed = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

# # Remove hook and compute loss with original activations
# handle.remove()
# outputs_orig = model(**batch)
# shift_logits_orig = outputs_orig.logits[:, :-1, :].contiguous()
# shift_labels_orig = batch['input_ids'][:, 1:].contiguous()
# loss_original = loss_fn(shift_logits_orig.view(-1, shift_logits_orig.size(-1)), shift_labels_orig.view(-1))

# print(f"Loss with reconstructed activations: {loss_reconstructed.item()}")
# print(f"Loss with original activations: {loss_original.item()}")

# %%
