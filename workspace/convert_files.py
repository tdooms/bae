# %%
%load_ext autoreload
%autoreload 2

import os
import torch
from tqdm import tqdm

from autoencoder import Autoencoder, Config, Placeholder
from safetensors.torch import save_file, load_file
import json

folder = "_checkpoints222"
model = Placeholder(d_model=1024, name="Qwen/Qwen3-0.6B-Base")

for filename in tqdm(os.listdir(folder)):
    path = os.path.join(folder, filename)
    
    if int(path.split('-')[-1]) % 1024 != 0:
        continue
    
    with open(f"{path}/config.json", "r") as file:
        config = json.load(file)
        del config['kind']
        del config['architectures']
        del config['torch_dtype']
        del config['d_model']
        del config['transformers_version']
        del config['kwargs']
        config['tags'] = ["checkpoint-" + path.split('-')[-1]]
    
    coder = Autoencoder.from_config(model, "mixed", **config)
    state = load_file(f"{path}/model.safetensors", device='cpu')
    coder.load_state_dict(state)
    coder.save()
    
    # _, _, _, layer, expansion, a = path[:-3].split('/')[-1].split('-')
    # coder = Autoencoder.from_config(model, layer=int(layer[1:]), expansion=int(expansion[1:]))
    # coder.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))

    # root = "weights"
    # out = model.name_or_path.split('/')[-1]
    # f = f"{root}/{out}/mixed-l{coder.config.layer}-x{coder.config.expansion}-a{int(a)*10}-b0"
    # print(f)
    # os.makedirs(f, exist_ok=True)
    # conf = Config(kind="mixed", layer=coder.config.layer, expansion=coder.config.expansion, d_model=1024, alpha=int(a)/10, beta=0.0)
    # save_file(coder.state_dict(), f"{f}/model.safetensors")
    # conf.save(f)
# %%