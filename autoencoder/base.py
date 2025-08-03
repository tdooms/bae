import torch
import wandb
import os
import json

from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from utils import Hooked, Input
from abc import abstractmethod
from safetensors.torch import save_file, load_file
from types import SimpleNamespace

class Placeholder:
    """Use as a placeholder for a model when constrained for memory (there's probably a better way to do this)."""
    def __init__(self, d_model, name, device="cuda", dtype=torch.float16):
        self.config = SimpleNamespace(hidden_size=d_model)
        self.body = [nn.Identity()] * 100
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
        expansion: int = 8,             # Expansion factor for the autoencoder
        alpha: float = 0.0,             # Regularisation for activation-based/batch-wise Hoyer sparsity
        beta: float = 0.0,              # Regularisation for weight-based/feature-wise Hoyer sparsity
        kind: str = "undefined",        # The autoencoder kind (e.g., "vanilla", "ordered", etc.)
        tags: list = [],                # Tags to identify the model
        **kwargs
    ):
        super().__init__(**kwargs)
        # assert layer is not None, "Layer must be specified for the autoencoder."
        # assert d_model is not None, "Model dimension must be specified for the autoencoder."
        
        self.layer = layer
        self.d_model = d_model
        self.n_ctx = n_ctx
        
        self.expansion = expansion
        self.alpha = alpha
        self.beta = beta
        
        self.kind = kind
        self.tags = tags
        self.kwargs = kwargs
    
    @property
    def d_features(self):
        return int(self.d_model * self.expansion)
    
    @property
    def name(self):
        return '-'.join([self.kind, f"l{self.layer}", f"x{self.expansion}", f"a{int(self.alpha*100)}", f"b{int(self.beta*100)}", *self.tags])
        

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
        layer = model.body[config.layer] if hasattr(model, 'body') else model.model.layers[config.layer]
        self.hooked = Hooked(model, acts=Input(layer))

    def _like(self):
        return dict(device=self.hooked.model.device, dtype=self.hooked.model.dtype)
    
    def alpha(self):
        return self.config.alpha * max(min(1.0, (self.steps - 100) / 512.0), 0.0)
    
    @staticmethod
    def from_config(model, kind, **kwargs):
        # TODO: Make this more flexible
        d_model = model.config.hidden_size
        return Autoencoder._subclasses[kind].from_config(model, d_model=d_model, **kwargs)
    
    def save(self, root="weights"):
        folder = f"{root}/{self.hooked.model.name_or_path.split('/')[-1]}/{self.config.name}"
        os.makedirs(folder, exist_ok=True)
        
        save_file(self.state_dict(), f"{folder}/model.safetensors")
        open(f"{folder}/config.json", 'w').write(self.config.to_json_string())
    
    @staticmethod
    def load(model, kind, layer, expansion, alpha=0.0, beta=0.0, tags=[], root="weights", device='cuda'):
        name = Config(kind=kind, layer=layer, expansion=expansion, alpha=alpha, beta=beta, tags=tags, d_model=0).name
        folder = f"{root}/{model.name_or_path.split('/')[-1]}/{name}"

        config = Config(**json.load(open(f"{folder}/config.json")))
        coder = Autoencoder._subclasses[config.kind](model, config)
        
        state = load_file(f"{folder}/model.safetensors", device=device)
        coder.load_state_dict(state)
        return coder.to(device)

    @abstractmethod
    def loss(self, acts, features): pass
    
    @abstractmethod
    def features(self, acts): pass
    
    @abstractmethod
    def network(self, mod='inp'): pass
    
    @abstractmethod
    def optimizers(self, max_steps, lr=0.01, cooldown=0.5): pass
    
    def forward(self, input_ids, attention_mask, **kwargs):
        self.steps += 1
        
        # Normalise the inputs using the L2 norm (not RMS norm)
        with torch.no_grad():
            _, cache = self.hooked(input_ids[..., :256])
            acts = cache['acts'].type(self.dtype)
            acts = acts * acts.pow(2).sum(-1, keepdim=True).rsqrt()
        
        if self.training:
            loss, features, metrics = self.loss(acts)
            if wandb.run is not None: wandb.log(metrics, commit=False)
            return dict(loss=loss, features=features, acts=acts)
        else:
            features = self.features(acts)
            return dict(features=features, acts=acts)