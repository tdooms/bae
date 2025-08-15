import torch
import wandb
import os
import json
import shutil

from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from utils import Hooked, Input
from abc import abstractmethod
from types import SimpleNamespace

from safetensors.torch import load_model, save_model
from huggingface_hub import hf_hub_download, HfApi

from itertools import combinations_with_replacement

def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum()

def hoyer(x):
    # TODO: This should (maybe) be computed over the unmasked elements only.
    size = x.size(0) * x.size(1)
    return (x.norm(p=1, dim=(0, 1)) / x.norm(p=2, dim=(0, 1)) - 1.0) / (size**0.5 - 1.0)

def precompute_indices(total, block=4096):
    """Compute the indices for the tiles in the inner product"""
    lst = combinations_with_replacement(range(0, total, block), 2)
    return [(s1, min(s1 + block, total), s2, min(s2 + block, total)) for s1, s2 in lst]

def tiled_inner_product(f, L, R, indices):
    """Block-wise evalation of the kernelised inner product, leveraging the symmetry."""
    result = torch.zeros_like(f[..., 0])
    
    for s1, e1, s2, e2 in indices:
        scale = 2 if s1 != s2 else 1
        result += scale * torch.einsum("...h, ...k, hl, kl, hr, kr -> ...", f[..., s1:e1], f[..., s2:e2], L[s1:e1], L[s2:e2], R[s1:e1], R[s2:e2])

    return result

class Placeholder:
    """Use as a placeholder for a model when constrained for memory (there's probably a better way to do this)."""
    def __init__(self, name, d_model, device="cuda", dtype=torch.float16):
        self.config = SimpleNamespace(hidden_size=d_model)
        self.model = SimpleNamespace(layers=[nn.Identity()] * 100)
        self.name_or_path = name
        
        self.device = device
        self.dtype = dtype

class Config(PretrainedConfig):
    """Simple configuration class for the model below."""
    def __init__(
        self,
        layer: int | None = None,       # Layer to hook the model at
        d_model: int | None = None,     # Model dimension at the hook point
        n_ctx: int = 256,               # Max sampled context length
        expansion: int = 16,            # Expansion factor for the autoencoder
        bottleneck: int | None = None,  # Bottleneck expansion factor for the Mixed autoencoders
        alpha: float = 0.0,             # Regularisation for activation-based/batch-wise Hoyer sparsity
        beta: float = 0.0,              # Regularisation for weight-based/feature-wise Hoyer sparsity
        warmup: int = 256,              # Warmup steps for the regularisation
        kind: str = "undefined",        # The autoencoder kind (e.g., "vanilla", "ordered", etc.)
        tags: list = [],                # Tags to identify the model
        base: str | None = None,        # Base model name (e.g., "Qwen/Qwen3-0.6B-Base")
        **kwargs
    ):
        super().__init__(**kwargs)
        # assert layer is not None, "Layer must be specified for the autoencoder."
        # assert d_model is not None, "Model dimension must be specified for the autoencoder."
        
        self.layer = layer
        self.d_model = d_model
        self.n_ctx = n_ctx
        
        self.expansion = expansion
        self.bottleneck = bottleneck
        self.alpha = alpha
        self.beta = beta
        self.warmup = warmup
        
        self.kind = kind
        self.tags = tags
        self.base = base
        self.kwargs = kwargs
    
    @property
    def d_features(self):
        return int(self.d_model * self.expansion)
    
    @property
    def d_bottleneck(self):
        return int(self.d_model * (self.bottleneck or 2))
    
    @property
    def name(self):
        return '-'.join([self.kind, f"l{self.layer:02d}", f"x{self.expansion}", f"a{int(self.alpha*100)}", f"b{int(self.beta*100)}", *self.tags])
        

class Autoencoder(PreTrainedModel):
    """A sparse tensor network autoencoder class."""
    _subclasses = {}
    
    def __init_subclass__(cls, kind=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if kind is None: raise ValueError("Subclasses must declare a model 'kind' attribute.")
        Autoencoder._subclasses[kind] = cls
        
    def __init__(self, model, config) -> None:
        super().__init__(config)
        self.steps = 0
        
        # TODO: Make this more flexible
        layer = model.model.layers[config.layer]
        self.hooked = Hooked(model, acts=Input(layer))

    def _like(self):
        return dict(device=self.device, dtype=self.dtype)
    
    def save(self, root="weights", push_to_hub=False, delete_local=False):
        folder = f"{root}/{self.config.base}/{self.config.name}"
        os.makedirs(folder, exist_ok=True)
        
        save_model(self, f"{folder}/model.safetensors")
        json.dump(vars(self.config), open(f'{folder}/config.json', 'w'), indent=2)
        
        if push_to_hub:
            repo_id = f"tdooms/{self.config.model}-scope"
            HfApi().upload_folder(folder_path=folder, path_in_repo=self.config.name, repo_id=repo_id)
        
        if delete_local:
            shutil.rmtree(folder)
            
    @staticmethod
    def from_config(model, kind, **kwargs):
        # TODO: Make this more flexible
        d_model = model.config.hidden_size
        base = model.name_or_path.split('/')[-1].lower()
        return Autoencoder._subclasses[kind].from_config(model, d_model=d_model, base=base, **kwargs)

    @staticmethod
    def load(model, kind, layer, expansion, alpha=0.0, beta=0.0, tags=[], hf=False, device="cuda"):
        """Load an autoencoder from HuggingFace ``hf = True`` or local storage ``hf = False``."""
        base = model.name_or_path.split('/')[-1].lower()
        name = Config(kind=kind, layer=layer, expansion=expansion, alpha=alpha, beta=beta, tags=tags, d_model=0).name
        
        if hf:
            repo = f"tdooms/{base}-scope"
            config_path = hf_hub_download(repo_id=repo, filename=f"{name}/config.json")
            model_path = hf_hub_download(repo_id=repo, filename=f"{name}/model.safetensors")
        else:
            config_path = f"weights/{base}/{name}/config.json"
            model_path = f"weights/{base}/{name}/model.safetensors"

        config = Config(**json.load(open(config_path)))
        coder = Autoencoder._subclasses[config.kind](model, config)

        load_model(coder, model_path, device=device)
        return coder.to(device) # I'm a bit confused why this is necessary

    @abstractmethod
    def loss(self, acts, mask): pass
    
    @abstractmethod
    def features(self, acts): pass
    
    @abstractmethod
    def network(self, mod='inp'): pass
    
    @abstractmethod
    def optimizers(self, max_steps, lr=0.03): pass
    
    def forward(self, input_ids, attention_mask, **kwargs):
        # Sample and normalise the inputs using the L2 norm
        with torch.no_grad():
            _, cache = self.hooked(input_ids[..., :256])
            acts = cache['acts'].type(self.dtype)
            acts = acts * acts.square().sum(-1, keepdim=True).rsqrt()

        if self.training:
            self.steps += 0.5 # Gradient accumulation steps are 2, so we increment by 0.5, which is ugly
            alpha = self.config.alpha * min(1.0, self.steps / self.config.warmup)
            
            loss, features, metrics = self.loss(acts, attention_mask, alpha)
            if wandb.run is not None: wandb.log(metrics | dict(alpha=alpha), commit=False)
            return dict(loss=loss, features=features, acts=acts)
        else:
            features = self.features(acts)
            return dict(features=features, acts=acts)