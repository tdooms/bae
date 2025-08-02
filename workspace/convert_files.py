# %%
%load_ext autoreload
%autoreload 2

import os
import torch
from tqdm import tqdm

from oldcoders.asparse import Autoencoder
from autoencoder.base import Config, Placeholder
from safetensors.torch import save_file

folder = "weights/vanilla-sweep"
model = Placeholder(d_model=1024, name="Qwen/Qwen3-0.6B-Base")

for filename in tqdm(os.listdir(folder)):
    path = os.path.join(folder, filename)
    _, _, _, layer, expansion, a = path[:-3].split('/')[-1].split('-')
    coder = Autoencoder.from_config(model, layer=int(layer[1:]), expansion=int(expansion[1:]))
    coder.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))

    root = "weights"
    out = model.name_or_path.split('/')[-1]
    f = f"{root}/{out}/mixed-l{coder.config.layer}-x{coder.config.expansion}-a{int(a)*10}-b0"
    print(f)
    os.makedirs(f, exist_ok=True)
    conf = Config(kind="mixed", layer=coder.config.layer, expansion=coder.config.expansion, d_model=1024, alpha=int(a)/10, beta=0.0)
    save_file(coder.state_dict(), f"{f}/model.safetensors")
    conf.save(f)
# %%