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
        
        layer = model.body[config.layer] if hasattr(model, 'body') else model.model.layers[config.layer]
        self.hooked = Hooked(model, x=Input(layer))
        
        self.left = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.right = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.down = nn.Parameter(torch.empty(config.d_hidden, config.d_features))
        
        torch.nn.init.xavier_uniform_(self.left.data)
        torch.nn.init.xavier_uniform_(self.right.data)
        torch.nn.init.xavier_uniform_(self.down.data)
        
        self.steps = -100
        
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
    
    def encoder(self, side='in', dtype=torch.float16):
        return Tensor(self.right.type(dtype), inds=[f'f:{side}', f'{side}:1'], tags=['R']) \
             & Tensor(self.left.type(dtype), inds=[f'f:{side}', f'{side}:0'], tags=['L'])

    def sym(self, side='in', dtype=torch.float16):
        u = torch.stack([self.left + self.right, self.left - self.right], dim=0).type(dtype)
        s = torch.tensor([1, -1], device=u.device, dtype=dtype)
        
        return Tensor(u, inds=[f"s:{side}", f'f:{side}', f'{side}:0'], tags=['U']) \
             & Tensor(u, inds=[f"s:{side}", f'f:{side}', f'{side}:1'], tags=['U']) \
             & Tensor(s, inds=[f's:{side}'], tags=['S'])

    def mixer(self, dtype=torch.float16):
        return Tensor(self.down.type(dtype), inds=['f:hid', 'f:in'], tags=['D']) \
             & Tensor(self.down.type(dtype), inds=['f:hid', 'f:out'], tags=['D'])

    def network(self, dtype=torch.float16):
        return self.sym('in', dtype) & self.sym('out', dtype) & self.mixer(dtype)
        
    # def loss(self, x):
    #     [Tensor(x, inds=['batch', 'seq', f'in:{i}']) for i in range(2)]

    def forward(self, input_ids, **kwargs):
        # Normalise the inputs using the L2 norm instead of RMS norm
        with torch.no_grad():
            _, acts = self.hooked(input_ids[..., :256])
            x = acts['x'].type(self.dtype)
            x = x * x.pow(2).sum(-1, keepdim=True).rsqrt()
        
        # Compute various hidden activations (f = features, h = hidden, m = middle)
        f = einsum(self.left, self.right, x, x, "feat in1, feat in2, ... in1, ... in2 -> ... feat")
        
        self.steps += 1
        alpha = self.config.alpha * max(min(1.0, self.steps / 400.0), 0.0)
        hoyer = (f.norm(p=1, dim=(0, 1)) / f.norm(p=2, dim=(0, 1)) - 1.0).mean() / ((f.size(0) * f.size(1))**0.5 - 1.0)
        reg = 1.0 - alpha * hoyer

        # Compute the linearised interaction matrix
        inner = (self.left @ self.left.T) * (self.right @ self.right.T)
        inner = einsum(self.down, self.down, self.down, self.down, inner, "md id, md od, mu iu, mu ou, od ou -> id iu")
        
        # Compute the constituent parts of the loss
        selv = einsum(f, f, inner, "... in1, ... in2, in1 in2 -> ...") * reg
        cross = einsum(f, f, self.down, self.down, "... f1, ... f2, h f1, h f2 -> ...") * reg
        const = einsum(x, x, "... d, ... d -> ...").pow(2)
        
        mse = (selv - 2*cross + const).mean()
        if wandb.run is not None: wandb.log(dict(mse=mse, reg=hoyer), commit=False)

        return dict(loss=mse, features=f, x=x)

    def optimizers(self, max_steps, lr=0.01, cooldown=0.5):
        params = [self.left, self.right, self.down]
        
        optimizer = Muon(params, lr=lr, weight_decay=0, momentum=0.95, nesterov=False)
        scheduler = LambdaLR(optimizer, lambda step: min(1.0, (1.0 / (cooldown - 1.0)) * ((step / max_steps) - 1.0)))
        
        return optimizer, scheduler