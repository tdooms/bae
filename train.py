# %%
%load_ext autoreload
%autoreload 2

from itertools import product
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from autoencoders import Autoencoder

import torch
import wandb
import os

from torch.backends import opt_einsum
opt_einsum.strategy = "auto-hq"
# %%
name = "Qwen/Qwen3-0.6B-Base"
# name = "Qwen/Qwen3-1.7B-Base"
# name = "google/gemma-3-270m"

tokenizer = AutoTokenizer.from_pretrained(name)
tokenize = lambda dataset: tokenizer(dataset["text"], truncation=True, padding=True, max_length=256)

model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto", device_map="auto")
model = torch.compile(model)

train = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True).with_format("torch")
train = train.map(tokenize, batched=True)
# %%
# for i, k in product([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["vanilla", "ordered", "mixed", "combined"]):
for i, k in product([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["combined"]):
    coder = Autoencoder.from_config(model, k, layer=18, expansion=16, alpha=i, bottleneck=2, tags=['v2'])
    project = "coder"
    # project = None

    args = TrainingArguments(
        seed=0,
        output_dir="_checkpoints",  
        logging_steps=10,
        save_total_limit=5,
        save_steps=512,
        per_device_train_batch_size=32,
        do_eval=False,
        report_to="wandb" if project else "none",
        remove_unused_columns=True,
        bf16=True,
        gradient_accumulation_steps=1,
        max_steps=2**10,
        max_grad_norm=1000,
        run_name=f"{model.name_or_path.split('/')[-1]}-{coder.config.name}",
    )

    trainer = Trainer(
        model=coder,
        optimizers=coder.optimizers(args.max_steps),
        args=args,
        train_dataset=train,
        processing_class=tokenizer,
    )

    os.environ["WANDB_PROJECT"] = project or ""
    os.environ["WANDB_DIR"] = "_logs"
    os.environ["WANDB_CONSOLE"] = "wrap"

    trainer.train()
    coder.save()
    
    wandb.run.tags = ['sparsity-sweep']
    wandb.finish()
# %%
import gc
gc.collect()
torch.cuda.empty_cache()
# %%