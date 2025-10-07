# %%
%load_ext autoreload
%autoreload 2

from itertools import product
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from autoencoders import Autoencoder
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm

from autoencoder.utils import Hooked, Input, Muon, BatchSampler

import torch
import wandb

from torch.backends import opt_einsum
opt_einsum.strategy = "auto-hq"
# %%
model_name = "Qwen/Qwen3-0.6B-Base"
# name = "Qwen/Qwen3-1.7B-Base"
# name = "google/gemma-3-270m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenize = lambda dataset: tokenizer(dataset["text"], truncation=True, padding=True, max_length=256)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
model = torch.compile(model)

dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True).with_format("torch")
dataset = dataset.map(tokenize, batched=True)
# %%
max_steps = 2**10

params = dict(d_model=model.config.hidden_size, layer=18, expansion=16, alpha=0.3, tags=['test'])
autoencoder = Autoencoder.from_config("vanilla", **params).cuda().type(torch.bfloat16)

optimizer = Muon(list(autoencoder.parameters()), lr=0.03, weight_decay=0, nesterov=True, momentum=0.95)
scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=max_steps)

hooked = Hooked(model, Input(model.model.layers[autoencoder.config.layer]))
progress = tqdm(zip(range(max_steps), BatchSampler(hooked, dataset)), total=max_steps)

name = model_name.lower().split('/')[-1] + '-' + autoencoder.config.name
# run = wandb.init(project="coder", name=name, config=autoencoder.config)

for step, (acts, batch) in progress:
    scale = torch.tensor(min(1.0, step / 256))
    loss, metrics = autoencoder.loss_fn(acts, batch['attention_mask'], scale)

    metrics['train/scale'] = scale.item()
    metrics['train/loss'] = loss.item()
    metrics['train/learning_rate'] = scheduler.get_last_lr()[0]
    metrics['train/grad_norm'] = torch.nn.utils.get_total_norm(autoencoder.parameters())
    metrics['train/weight_norm'] = torch.nn.utils.get_total_norm([p.data for p in autoencoder.parameters()])

    progress.set_description(f"loss = {float(loss):.3f}")
    # run.log(metrics, step=step, commit=True)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
# run.finish()
# %%
autoencoder.save(model_name)
# %%