# %%
%load_ext autoreload
%autoreload 2

import os
import torch
from tqdm import tqdm

from autoencoder import Autoencoder, Mixed, Placeholder

model = Placeholder("Qwen/Qwen3-0.6B-Base", d_model=1024)

coder = Autoencoder.load(model, "mixed", layer=18, expansion=16)
coder
# %%