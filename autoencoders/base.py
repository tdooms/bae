import torch
import os
import json
import shutil

from torch import nn
from abc import abstractmethod

from safetensors.torch import load_model, save_model
from huggingface_hub import hf_hub_download, HfApi
from jaxtyping import Float
from torch import Tensor
from typing import List

class Config:
    def __init__(
        self,
        layer: int,                     # Layer to hook the model at
        d_model: int,                   # Model dimension at the hook point
        expansion: int = 16,            # Expansion factor for the autoencoder
        alpha: float = 0.0,             # Regularisation for activation-based/batch-wise Hoyer sparsity
        beta: float = 0.0,              # Regularisation for weight-based/feature-wise Hoyer sparsity
        kind: str = "undefined",        # The autoencoder kind (e.g., "vanilla", "ordered", etc.)
        bottleneck: int | None = None,  # Bottleneck expansion factor for the Mixed autoencoders
        tags: List[str] | None = None,  # Tags to identify the model
        **kwargs                        # Absorbs any extra arguments
    ):
        self.layer = layer
        self.d_model = d_model
        self.expansion = expansion
        self.bottleneck = bottleneck
        self.alpha = alpha
        self.beta = beta
        self.kind = kind
        self.tags = tags or []
        self.kwargs = kwargs
   
    @property
    def d_features(self) -> int:
        return int(self.d_model * self.expansion)
   
    @property
    def d_bottleneck(self) -> int:
        return int(self.d_model * (self.bottleneck or 2))
    
    @property
    def name(self):
        return '-'.join([self.kind, f"l{self.layer:02d}", f"x{self.expansion}", f"a{int(self.alpha*100)}", f"b{int(self.beta*100)}", *self.tags])
        

class Autoencoder(nn.Module):
    """A sparse tensor network autoencoder class."""
    _subclasses = {}
    
    def __init_subclass__(cls, kind=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if kind is None: raise ValueError("Subclasses must declare a model 'kind' attribute.")
        Autoencoder._subclasses[kind] = cls
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        
    def _like(self):
        return dict(device=self.device, dtype=self.dtype)
    
    @staticmethod
    def from_config(kind, **kwargs):
        return Autoencoder._subclasses[kind].from_config(**kwargs)
    
    def save(self, base: str, root: str = "weights", push_to_hub: bool = False, delete_local: bool = False):
        base = base.lower().split('/')[-1]
        folder = f"{root}/{base}/{self.config.name}"
        os.makedirs(folder, exist_ok=True)
        
        save_model(self, f"{folder}/model.safetensors")
        json.dump(vars(self.config), open(f'{folder}/config.json', 'w'), indent=2)
        
        if push_to_hub:
            repo_id = f"tdooms/{base}-scope"
            HfApi().upload_folder(folder_path=folder, path_in_repo=self.config.name, repo_id=repo_id)
        
        if delete_local:
            shutil.rmtree(folder)

    @staticmethod
    def load(base: str, kind: str, layer: int, expansion: int, alpha: float = 0.0, beta: float = 0.0, tags=[], hf=False):
        """Load an autoencoder from HuggingFace ``hf = True`` or local storage ``hf = False``."""
        base = base.lower().split('/')[-1]
        name = Config(kind=kind, layer=layer, expansion=expansion, alpha=alpha, beta=beta, tags=tags, d_model=0).name
        
        if hf:
            config_path = hf_hub_download(repo_id=f"tdooms/{base}-scope", filename=f"{name}/config.json")
            model_path = hf_hub_download(repo_id=f"tdooms/{base}-scope", filename=f"{name}/model.safetensors")
        else:
            config_path = f"weights/{base.lower()}/{name}/config.json"
            model_path = f"weights/{base.lower()}/{name}/model.safetensors"

        config = Config(**json.load(open(config_path)))
        coder = Autoencoder._subclasses[config.kind](config)

        load_model(coder, model_path)
        return coder

    @abstractmethod
    def loss_fn(self, x: Float[Tensor, "... acts"], mask: Float[Tensor, "..."], step: Tensor): pass
    
    @abstractmethod
    def forward(self, x: Float[Tensor, "... acts"]): pass
    
    @abstractmethod
    def network(self, mod='inp'): pass