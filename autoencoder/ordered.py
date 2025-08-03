import torch

from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from einops import einsum
from quimb.tensor import Tensor

from utils import Muon
from autoencoder.base import Autoencoder, Config

class Ordered(Autoencoder, kind="ordered"):
    """A tensor-based autoencoder class which mixes its features."""
    def __init__(self, model, config) -> None:
        super().__init__(model, config)
        
        self.weights = nn.Buffer((torch.arange(1, config.d_features + 1).flip(0) / config.d_features).sqrt(), persistent=False)
        self.mask = nn.Buffer(torch.triu(torch.ones(config.d_features, config.d_features)), persistent=False)

        self.left = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.right = nn.Parameter(torch.empty(config.d_features, config.d_model))
        
        torch.nn.init.xavier_uniform_(self.left.data)
        torch.nn.init.xavier_uniform_(self.right.data)
    
    @staticmethod
    def from_config(model, **kwargs):
        return Ordered(model, Config(kind="ordered", **kwargs))

    def kernel(self):
        return (self.left @ self.left.T) * (self.right @ self.right.T)
    
    def features(self, acts):
        return einsum(self.left, acts, "feat inp, ... inp -> ... feat") * einsum(self.right, acts, "feat inp, ... inp -> ... feat")
    
    def network(self, mod='inp'):
        u = torch.stack([self.left + self.right, self.left - self.right], dim=0)
        
        return Tensor(u, inds=[f"s:{mod}", f'f:{mod}', f'i:0'], tags=['U']) \
             & Tensor(u, inds=[f"s:{mod}", f'f:{mod}', f'i:1'], tags=['U']) \
             & Tensor(torch.tensor([1, -1], **self._like()) / 4.0, inds=[f's:{mod}'], tags=['S'])
             
    def loss(self, acts):
        f = self.features(acts)
        o = einsum(self.weights, f, "feat, ... feat -> ... feat")
        
        # Compute the regularisation term
        counts = torch.arange(1, self.config.d_features + 1, **self._like())
        hoyer = (f.norm(p=1, dim=(0, 1)) / f.norm(p=2, dim=(0, 1)) - 1.0) / ((f.size(0)*f.size(1))**0.5 - 1.0)
        reg = (1 - self.alpha() * (hoyer.cumsum(dim=0) / counts)) / self.config.d_features
        inner = einsum(self.mask, reg, self.mask, "out hid, hid, inp hid -> out inp")
        
        kernel = self.kernel()
        orecons = einsum(f, f, kernel * inner, "... f1, ... f2, f1 f2 -> ...")
        loss = (orecons - 2 * o.pow(2).sum(-1) + 1.0).mean()
        
        with torch.no_grad():
            recons = einsum(f, f, kernel, "... h1, ... h2, h1 h2 -> ...")
            mse = (recons - 2 * f.pow(2).sum(-1) + 1.0).mean()
        
        return loss, f, dict(mse=mse, reg=reg)
    
    def optimizers(self, max_steps, lr=0.01, cooldown=0.5):
        optimizer = Muon(list(self.parameters()), lr=lr, weight_decay=0, momentum=0.95, nesterov=False)
        scheduler = LambdaLR(optimizer, lambda step: min(1.0, (1.0 / (cooldown - 1.0)) * ((step / max_steps) - 1.0)))
        return optimizer, scheduler
