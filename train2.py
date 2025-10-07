# %%
%load_ext autoreload
%autoreload 2

from itertools import product
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from autoencoders import Autoencoder
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm

from autoencoders.utils import Hooked, Input, Muon, BatchSampler

import torch
import wandb

from torch.backends import opt_einsum
opt_einsum.strategy = "auto-hq"
# %%
path = f"SimpleStories/SimpleStories-35M"
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype="auto")

tokenizer = AutoTokenizer.from_pretrained(path)
tokenizer.pad_token = tokenizer.eos_token
tokenize = lambda batch: tokenizer(batch["story"], truncation=True, padding=True, max_length=512)

dataset = load_dataset("SimpleStories/SimpleStories", split="train[:5%]")
dataset = dataset.map(tokenize, batched=True)
# %%
max_steps = 2**10

params = dict(d_model=model.config.hidden_size, layer=18, expansion=16, alpha=0.3, tags=['test'])
autoencoder = Autoencoder.from_config("vanilla", **params).cuda().type(torch.bfloat16)

optimizer = Muon(list(autoencoder.parameters()), lr=1.0, weight_decay=0, nesterov=True, momentum=0.95)
scheduler = LinearLR(optimizer, start_factor=0.03, end_factor=0.0, total_iters=max_steps)

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