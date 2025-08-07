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
name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenize = lambda dataset: tokenizer(dataset["text"], truncation=True, padding=True, max_length=256)

model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto", device_map="auto")
model = torch.compile(model)

dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True).with_format("torch")
dataset = Dataset.from_list(list(dataset.take(2**12))).with_format("torch")
dataset = dataset.map(tokenize, batched=True)
# %%
coder = Autoencoder.load(model, "mixed", layer=8, expansion=16, alpha=0.1, tags=[]).eval().half()
# %%
vis = Feature(coder, tokenizer, dataset, max_steps=2**5, batch_size=2**5)
# %%
vis(list(range(20, 40)), dark=True, k=3)
# %%
vis(13, k=20)