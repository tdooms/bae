# %%
%load_ext autoreload
%autoreload 2

from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
# from coders.incremental import Autoencoder
# from coders.vsparse import Autoencoder
from coders.sparse import Autoencoder

import torch
import wandb
import os
# %%
name = "Qwen/Qwen3-0.6B-Base"

tokenizer = AutoTokenizer.from_pretrained(name)
tokenize = lambda dataset: tokenizer(dataset["text"], truncation=True, padding=True, max_length=256)

model = AutoModelForCausalLM.from_pretrained(name,torch_dtype="auto", device_map="auto")
model = torch.compile(model)

train = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True).with_format("torch")
train = train.map(tokenize, batched=True)
# %%
for i in [18]:
    coder = Autoencoder.from_config(model, layer=i, expansion=16, alpha=0.2, tags=[])
    project = "coder"
    # project = None

    args = TrainingArguments(
        output_dir="_checkpoints",
        logging_steps=10,
        save_total_limit=5,
        per_device_train_batch_size=32,
        do_eval=False,
        report_to="wandb" if project else "none",
        remove_unused_columns=True,
        bf16=True,
        gradient_accumulation_steps=2,
        max_steps=2**10,
        max_grad_norm=1000,
        run_name=f"{model.name_or_path}-{coder.config.name}",
    )

    trainer = Trainer(
        model=coder,
        optimizers=coder.optimizers(args.max_steps, cooldown=0.5),
        args=args,
        train_dataset=train,
        processing_class=tokenizer,
    )

    os.environ["WANDB_PROJECT"] = project or ""
    os.environ["WANDB_DIR"] = "_logs"
    os.environ["WANDB_CONSOLE"] = "wrap"

    trainer.train()
    coder.save()

    wandb.finish()
# %%
import plotly.express as px
norm = (coder.left.norm(dim=-1) * coder.right.norm(dim=-1) * coder.down.norm(dim=0)).detach()
px.scatter(y=norm.cpu(), x=list(range(len(norm))), template='plotly_white', labels=dict(x="Index", y="Norm"))
# %%