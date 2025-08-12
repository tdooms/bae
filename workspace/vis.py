# %%
%load_ext autoreload
%autoreload 2

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset

from utils import Feature
from autoencoder import Autoencoder

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
# coder = Autoencoder.load(model, "mixed", layer=18, expansion=16, alpha=1.0, tags=[]).eval().half()
# coder = Autoencoder.load(model, "biased", layer=18, expansion=16, alpha=1.0, tags=[]).eval().half()
coder = Autoencoder.load(model, "vanilla", layer=18, expansion=16, alpha=0.1, tags=[]).eval().half()
# %%
vis = Feature(coder, tokenizer, dataset, max_steps=2**5, batch_size=2**5)
# %%
vis(list(range(20)), dark=True, k=5)
# %%
vis(13, k=20)