from sympy import re
import torch
import wandb
import os

from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from torch.optim.lr_scheduler import LambdaLR
from utils import Muon, Hooked, Input
from einops import einsum, rearrange
from quimb.tensor import Tensor, TensorNetwork

# class Monarch(nn.Module):
#     def __init__(self, dim, rank=1):
#         super().__init__()
#         self.dim = dim
        
#         self.a = nn.Parameter(torch.empty(rank, dim, dim, dim))
#         self.b = nn.Parameter(torch.empty(rank, dim, dim, dim))
        
#         nn.init.xavier_uniform_(self.a.data)
#         nn.init.xavier_uniform_(self.b.data)
    
#     def forward(self, x):
#         x = rearrange(x, '... (i1 inp) -> ... i1 inp', inp=self.dim)
#         x = einsum(x, self.a, self.b, '... i1 inp, rank i1 o1 inp, rank i1 o1 out -> ... o1 out')
#         return rearrange(x, '... o1 out -> ... (o1 out)')
    
#     def matrix(self):
#         t = einsum(self.a, self.b, 'rank i1 o1 inp, rank i1 o1 out -> o1 out i1 inp')
#         return rearrange(t, 'o1 out i1 inp -> (o1 out) (i1 inp)')

class Monarch(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        # Now interpret R[i] and L[i] as “out×in” blocks
        self.R = nn.Parameter(torch.empty(m, m, m))
        self.L = nn.Parameter(torch.empty(m, m, m))
        nn.init.xavier_uniform_(self.R)
        nn.init.xavier_uniform_(self.L)

    def forward(self, x):
        # x: [..., m*m] → [..., m, m]
        B = x.shape[:-1]
        x = x.view(*B, self.m, self.m)  

        # 1) Block-diag R: y[i,j] = sum_k  R[i,j,k] * x[i,k]
        x = torch.einsum('... i k, i j k -> ... i j', x, self.R)

        # 2) Transpose blocks (the “perfect shuffle” P)
        x = x.transpose(-2, -1)         

        # 3) Block-diag L: y[i,j] = sum_k  L[i,j,k] * x[i,k]
        x = torch.einsum('... i k, i j k -> ... i j', x, self.L)

        # 4) Transpose back and flatten
        x = x.transpose(-2, -1)         
        return x.reshape(*B, self.m * self.m)

    def matrix(self):
        """
        Build the full (m^2×m^2) matrix by pushing the identity
        through `forward`.  Now m(x) and m.matrix() @ x will agree exactly.
        """
        I = torch.eye(self.m * self.m,
                      device=self.R.device,
                      dtype=self.R.dtype)
        return self.forward(I).T
    
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
        
        self.left = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.right = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.down = Monarch(2**7)
        
        torch.nn.init.xavier_uniform_(self.left.data)
        torch.nn.init.xavier_uniform_(self.right.data)

    def _like(self):
        return dict(device=self.left.device, dtype=self.left.dtype)
    
    def alpha(self):
        self.steps += 1
        return self.config.alpha * max(min(1.0, (self.steps - 100) / 512.0), 0.0)
    
    def kernel(self):
        d = self.down.matrix()
        return d @ ((self.left @ self.left.T) * (self.right @ self.right.T)) @ d.T
    
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
        
        # Compute the features as well as hidden/clustered activations
        f = einsum(self.left, self.right, x, x, "feat in1, feat in2, ... in1, ... in2 -> ... feat")
        h = self.down(f)
        
        # Compute the regularisation term
        hoyer = (f.norm(p=1, dim=(0, 1)) / f.norm(p=2, dim=(0, 1)) - 1.0).mean() / ((f.size(0) * f.size(1))**0.5 - 1.0)
        reg = 1.0 - self.alpha() * hoyer
        
        # Compute the self and cross terms of the loss
        selv = einsum(h, h, self.kernel(), "... h1, ... h2, h1 h2 -> ...")
        cross = h.pow(2).sum(-1)
        
        # Compute the reconstruction and the loss; the last term is always one since we normalised the inputs
        mse = (selv - 2*cross + 1.0).mean()
        loss = (selv*reg - 2*cross*reg + 1.0).mean()
        
        # Log and return the loss and statistics
        if wandb.run is not None: wandb.log(dict(mse=mse, reg=hoyer), commit=False)
        return dict(loss=loss, features=f, x=x)

    def optimizers(self, max_steps, lr=0.01, cooldown=0.5):
        params = [self.left, self.right, self.down.L, self.down.R]
        
        optimizer = Muon(params, lr=lr, weight_decay=0, momentum=0.95, nesterov=False)
        scheduler = LambdaLR(optimizer, lambda step: min(1.0, (1.0 / (cooldown - 1.0)) * ((step / max_steps) - 1.0)))
        
        return optimizer, scheduler