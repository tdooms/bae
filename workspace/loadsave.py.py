# %%
%load_ext autoreload
%autoreload 2

from tqdm import tqdm
from autoencoder import Autoencoder, Placeholder, Config
import plotly.express as px
import torch
from safetensors.torch import save_file, load_file, load_model, save_model
from huggingface_hub import hf_hub_download, HfApi
import os
import json

# %%
torch.set_grad_enabled(False)
model = Placeholder("Qwen/Qwen3-0.6B-Base", d_model=1024, dtype=torch.float32)
repo = f"tdooms/qwen3-0.6b-base-scope"

for i in range(24):
    base = model.name_or_path.split('/')[-1].lower()
    name = Config(kind="vanilla", layer=i, expansion=16, alpha=0.1, beta=0.0, tags=[], d_model=0).name
    folder = f"weights/{base}/{name}"

    HfApi().upload_folder(folder_path=folder, path_in_repo=name, repo_id=repo)

# %%





# %%

# base = "qwen3-0.6b-base"

# for folder_name in os.listdir(f"weights/{base}"):
#     folder_path = os.path.join(f"weights/{base}", folder_name)
#     if os.path.isdir(folder_path):
#         config_path = os.path.join(folder_path, "config.json")
#         if os.path.exists(config_path):
#             with open(config_path, 'r') as f:
#                 config = json.load(f)
#                 config["base"] = base
#             with open(config_path, 'w') as f:
#                 json.dump(config, f, indent=2)
            