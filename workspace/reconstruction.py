# %%
%load_ext autoreload
%autoreload 2

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset

from utils import Feature
from autoencoder import Autoencoder

import torch
import plotly.express as px
# %%
torch.set_grad_enabled(False)
name = "Qwen/Qwen3-0.6B-Base"

tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cuda")
tokenize = lambda dataset: tokenizer(dataset["text"], truncation=True, padding=True, max_length=256)

dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True).with_format("torch")
dataset = Dataset.from_list(list(dataset.take(2**2))).with_format("torch")
dataset = dataset.map(tokenize, batched=True)
# %%
coder = Autoencoder.load(model, "ordinary", layer=18, expansion=16, alpha=0.1, tags=[]).half()
# %%
input_ids, attention_mask = dataset[0]["input_ids"].unsqueeze(0).cuda(), dataset[0]["attention_mask"].unsqueeze(0).cuda()
output = coder(input_ids, attention_mask)
loss, features, acts = output['loss'], output['features'], output['acts']
recons = torch.nn.functional.linear(features, coder.decoder)

px.bar(torch.stack([recons, acts]).cpu()[:, 0, 50, 256:512].T, barmode='group').show()
print(loss)
# %%
features.flatten(0, 1).sum(0).topk(10)
# %%
px.histogram(features[..., 6575].flatten().cpu())