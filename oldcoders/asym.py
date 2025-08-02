from sympy import re
import torch
import wandb
import os

from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from torch.optim.lr_scheduler import LambdaLR
from utils import Muon, Hooked, Input
from einops import einsum

class Placeholder:
    """Use as a placeholder for a model when constrained for memory (there's probably a better way to do this)."""
    def __init__(self, d_model, name):
        self.config = Config(d_model=d_model)
        self.body = [nn.Identity()] * 24
        self.name_or_path = name

class Config(PretrainedConfig):
    """Simple configuration class for the model below."""
    def __init__(
        self,
        layer: int | None = None,       # Layer to hook the model at
        d_model: int | None = None,     # Model dimension at the hook point
        expansion: int = 8,             # Expansion factor for the autoencoder
        tags: list = [],                # Tags for the model
        alpha: float = 0.0,             # Batch sparsity
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.layer = layer
        self.expansion = expansion
        self.alpha = alpha
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
        self.steps = 0
        
        layer = model.body[config.layer] if hasattr(model, 'body') else model.model.layers[config.layer]
        self.hooked = Hooked(model, x=Input(layer))
        
        self.left1 = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.right1 = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.left2 = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.right2 = nn.Parameter(torch.empty(config.d_features, config.d_model))
        
        self.down = nn.Parameter(torch.empty(config.d_hidden, config.d_features))

        torch.nn.init.xavier_uniform_(self.left1.data)
        torch.nn.init.xavier_uniform_(self.right1.data)
        torch.nn.init.xavier_uniform_(self.left2.data)
        torch.nn.init.xavier_uniform_(self.right2.data)
        
        torch.nn.init.xavier_uniform_(self.down.data)

        # x = torch.empty((32, 256, 1024*16), device="meta")
        # self.contract = "bsx,bsy,ul,dl,ur,dr,cu,cy,cd,cx->bs"
        # self.path, _ = oe.contract_path(self.contract, x, x, *self.tensors(), optimize='auto-hq')

    def alpha(self):
        self.steps += 1
        return self.config.alpha * max(min(1.0, (self.steps - 100) / 512.0), 0.0)
    
    # def tensors(self):
    #     return [self.left.data, self.left.data, self.right.data, self.right.data, self.down.data, self.down.data, self.down.data, self.down.data]
    # # selv = oe.contract(self.contract, f, f, *self.tensors(), optimize=self.path)
    
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

    def forward(self, input_ids, **kwargs):
        # Normalise the inputs using the L2 norm instead of RMS norm
        with torch.no_grad():
            _, acts = self.hooked(input_ids[..., :256])
            x = acts['x'].type(self.dtype)
            x = x * x.pow(2).sum(-1, keepdim=True).rsqrt()
        
        # Compute the features
        f1 = einsum(self.left1, self.right1, x, x, "feat in1, feat in2, ... in1, ... in2 -> ... feat")
        f2 = einsum(self.left2, self.right2, x, x, "feat in1, feat in2, ... in1, ... in2 -> ... feat")

        # Compute the regularisation term
        hoyer1 = (f1.norm(p=1, dim=(0, 1)) / f1.norm(p=2, dim=(0, 1)) - 1.0).mean() / ((f1.size(0) * f1.size(1))**0.5 - 1.0)
        hoyer2 = (f2.norm(p=1, dim=(0, 1)) / f2.norm(p=2, dim=(0, 1)) - 1.0).mean() / ((f2.size(0) * f2.size(1))**0.5 - 1.0)
        reg = 1.0 - self.alpha() * (hoyer1 + hoyer2) / 2.0

        # Compute the reconstruction kernel and the hidden activations
        kernel = (self.left2 @ self.left2.T) * (self.right2 @ self.right2.T)
        h = einsum(self.down, self.down, f1, "cluster hid, cluster feat, ... feat -> ... hid")

        # Compute the self and cross terms of the loss
        selv = einsum(h, h, kernel, "... in1, ... in2, in1 in2 -> ...")
        cross = einsum(f1, self.down, self.down, f2, "... f1, q f1, q f2, ... f2 -> ...")

        # Compute the reconstruction and the loss
        mse = (selv - 2*cross + 1.0).mean()
        loss = (selv*reg - 2*cross*reg + 1.0).mean()

        if wandb.run is not None: wandb.log(dict(mse=mse, reg=(hoyer1 + hoyer2)/2.0), commit=False)
        return dict(loss=loss)

    def optimizers(self, max_steps, lr=0.01, cooldown=0.5):
        params = [self.left1, self.right1, self.left2, self.right2, self.down]

        optimizer = Muon(params, lr=lr, weight_decay=0, momentum=0.95, nesterov=False)
        scheduler = LambdaLR(optimizer, lambda step: min(1.0, (1.0 / (cooldown - 1.0)) * ((step / max_steps) - 1.0)))
        
        return optimizer, scheduler