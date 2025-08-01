from sympy import re
import torch
import wandb
import os

from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from torch.optim.lr_scheduler import LambdaLR
from utils import Muon, Hooked, Input
from einops import einsum

from quimb.tensor import Tensor, TensorNetwork



class Placeholder:
    """Use as a placeholder for a model when constrained for memory (there's probably a better way to do this)."""
    def __init__(self, d_model, name):
        self.config = Config(d_model=d_model)
        self.body = [nn.Identity()] * 24
        self.name_or_path = name
        
        
# def min_kernel(x, A, weight) -> torch.Tensor:
#     L = torch.tril(A)
    
#     u = einsum(A, x, "hid inp, ... inp -> ... hid")
#     v = einsum(L, x, "out inp, inp -> out")
#     w = einsum(L, x * weight, "out inp, inp -> out")

#     return (x * w + (weight * x) * (u - v)).sum()

class Config(PretrainedConfig):
    """Simple configuration class for the model below."""
    def __init__(
        self,
        layer: int | None = None,       # Layer to hook the model at
        d_model: int | None = None,     # Model dimension at the hook point
        expansion: int = 8,             # Expansion factor for the autoencoder
        tags: list = [],                # Tags for the model
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.layer = layer
        self.expansion = expansion
        self.d_model = d_model
        self.tags = tags
        self.kwargs = kwargs
    
    @property
    def d_features(self):
        return int(self.d_model * self.expansion)
    
    @property
    def d_hidden(self):
        return int(self.d_model * 2)
    
    @property
    def name(self):
        return '-'.join([f"l{self.layer}", f"x{self.expansion}", *self.tags])

class Autoencoder(PreTrainedModel):
    """A sparse tensor network autoencoder class."""
    def __init__(self, model, config) -> None:
        super().__init__(config)
        
        layer = model.body[config.layer] if hasattr(model, 'body') else model.model.layers[config.layer]
        self.hooked = Hooked(model, x=Input(layer))
        
        self.left = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.right = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.down = nn.Parameter(torch.empty(config.d_hidden, config.d_features))
        
        # This currently can't be changed since it's hardcoded in the constraint function.
        self.weights = nn.Buffer((torch.arange(1, config.d_features + 1).flip(0) / self.config.d_features).sqrt(), persistent=False)
        
        mask = torch.triu(torch.ones(config.d_features, config.d_features))
        self.mask = nn.Buffer((mask @ mask.T) / config.d_features, persistent=False)
    
        torch.nn.init.xavier_uniform_(self.left.data)
        torch.nn.init.xavier_uniform_(self.right.data)
        torch.nn.init.xavier_uniform_(self.down.data)
        
    @classmethod
    def from_pretrained(cls, repo, model, device='cuda', **kwargs):
        config = Config.from_pretrained(repo, repo=repo)
        return super(Autoencoder, Autoencoder).from_pretrained(repo, model=model, config=config, device_map=device, **kwargs)
    
    @staticmethod
    def from_config(model, **kwargs):
        d_model = getattr(model.config, 'hidden_size', None) or getattr(model.config, 'd_model', None)
        return Autoencoder(model, Config(d_model=d_model, **kwargs))
    
    def save(self, root="weights"):
        os.makedirs(root, exist_ok=True)
        prefix = self.hooked.model.name_or_path.split('/')[1]
        torch.save(self.state_dict(), f"{root}/{prefix}-{self.config.name}.pt")
        
    @staticmethod
    def load(model, layer, expansion, root="weights", device='cuda', **kwargs):
        coder = Autoencoder.from_config(model, layer=layer, expansion=expansion, **kwargs)
        prefix = model.name_or_path.split('/')[1]
        coder.load_state_dict(torch.load(f"{root}/{prefix}-{coder.config.name}.pt", weights_only=True, map_location=device))
        return coder.to(device)
    
    def _like(self):
        return dict(device=self.left.device, dtype=self.left.dtype)
    
    def kernel(self):
        return self.down @ ((self.left @ self.left.T) * (self.right @ self.right.T)) @ self.down.T

    def forward(self, input_ids, **kwargs):
        # Normalise the inputs using the L2 norm instead of RMS norm
        with torch.no_grad():
            _, acts = self.hooked(input_ids[..., :256])
            x = acts['x'].type(self.dtype)
            x = x * x.pow(2).sum(-1, keepdim=True).rsqrt()
        
        # Compute the features as well as hidden/clustered activations
        f = einsum(self.left, self.right, x, x, "feat in1, feat in2, ... in1, ... in2 -> ... feat")
        o = einsum(self.down, self.weights, f, "hid feat, feat, ... feat -> ... hid")
        
        kernel = self.kernel()
        reg = einsum(f, f, self.down, self.down, kernel, self.mask, "... f1, ... f2, h1 f1, h2 f2, h1 h2, f1 f2 -> ...")
        loss = (reg - 2 * o.pow(2).sum(-1) + 1.0).mean()
        
        with torch.no_grad():
            h = einsum(self.down, f, "hid feat, ... feat -> ... hid")
            selv = einsum(h, h, kernel, "... h1, ... h2, h1 h2 -> ...")
            mse = (selv - 2 * h.pow(2).sum(-1) + 1.0).mean()
        
        if wandb.run is not None: wandb.log(dict(mse=mse), commit=False)
        return dict(loss=loss, features=f)
    
    def optimizers(self, max_steps, lr=0.01, cooldown=0.5):
        params = [self.left, self.right, self.down]
        
        optimizer = Muon(params, lr=lr, weight_decay=0, momentum=0.95, nesterov=False)
        scheduler = LambdaLR(optimizer, lambda step: min(1.0, (1.0 / (cooldown - 1.0)) * ((step / max_steps) - 1.0)))
        
        return optimizer, scheduler