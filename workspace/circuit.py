# %%

%load_ext autoreload
%autoreload 2

import plotly.express as px
import torch

from tqdm import tqdm
from autoencoder import Autoencoder

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from utils import Feature, visualise_circuit
# %%
torch.set_grad_enabled(False)
name = "Qwen/Qwen3-0.6B-Base"

tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cuda")
tokenize = lambda dataset: tokenizer(dataset["text"], truncation=True, padding=True, max_length=256)

dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True).with_format("torch")
dataset = Dataset.from_list(list(dataset.take(2**10))).with_format("torch")
dataset = dataset.map(tokenize, batched=True)

coders = [Autoencoder.load(model, "ordered", layer=i, expansion=16, alpha=0.1).eval().half() for i in tqdm(range(18, 22))]
# %%
networks = [out.network('out') | inp.network('inp') for out, inp in zip(coders[1:], coders[:-1])]
matrices = [net.contract(all, output_inds=['f:out', 'f:inp']).data for net in networks]

import gc
torch.cuda.empty_cache()
gc.collect()
# %%
vis = Feature(coders[1], tokenizer, dataset, max_steps=2**4, batch_size=2**5)
# %%
vis(19, k=5)
# %%
visualise_circuit(matrices, current=torch.arange(5), k=10)
# %%