import torch

from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from einops import einsum
from quimb.tensor import Tensor

from utils import Muon
from autoencoder.base import Autoencoder, Config

class Vanilla(Autoencoder, kind="vanilla"):
    """A tensor-based autoencoder class which mixes its features."""
    def __init__(self, model, config) -> None:
        super().__init__(model, config)

        self.left = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.right = nn.Parameter(torch.empty(config.d_features, config.d_model))
        
        torch.nn.init.xavier_uniform_(self.left.data)
        torch.nn.init.xavier_uniform_(self.right.data)
    
    @staticmethod
    def from_config(model, **kwargs):
        return Autoencoder(model, Config(kind="vanilla", **kwargs))
    
    def kernel(self):
        return (self.left @ self.left.T) * (self.right @ self.right.T)

    def network(self, mod='inp'):
        u = torch.stack([self.left + self.right, self.left - self.right], dim=0)

        return Tensor(u, inds=[f"s:{mod}", f'f:{mod}', f'i:0'], tags=['U']) \
             & Tensor(u, inds=[f"s:{mod}", f'f:{mod}', f'i:1'], tags=['U']) \
             & Tensor(self.down, inds=[f'h:{mod}', f'f:{mod}'], tags=['D']) \
             & Tensor(torch.tensor([1, -1], **self._like()) / 4.0, inds=[f's:{mod}'], tags=['S'])
    
    def features(self, acts):
        return einsum(self.left, acts, "feat inp, ... inp -> ... feat") * einsum(self.right, acts, "feat inp, ... inp -> ... feat")

    def loss(self, acts):
        f = self.features(acts)
        
        # Compute the regularisation term
        hoyer = (f.norm(p=1, dim=(0, 1)) / f.norm(p=2, dim=(0, 1)) - 1.0).mean() / ((f.size(0) * f.size(1))**0.5 - 1.0)
        reg = 1.0 - self.alpha() * hoyer
        
        # Compute the self and cross terms of the loss
        recons = einsum(f, f, self.kernel(), "... h1, ... h2, h1 h2 -> ...")
        cross = f.pow(2).sum(-1)

        # Compute the reconstruction and the loss
        mse = (recons - 2*cross + 1.0).mean()
        loss = (recons * reg - 2*cross*reg + 1.0).mean()
        
        return loss, f, dict(mse=mse, reg=hoyer)

    def optimizers(self, max_steps, lr=0.01, cooldown=0.5):
        optimizer = Muon(list(self.parameters()), lr=lr, weight_decay=0, momentum=0.95, nesterov=False)
        scheduler = LambdaLR(optimizer, lambda step: min(1.0, (1.0 / (cooldown - 1.0)) * ((step / max_steps) - 1.0)))
        return optimizer, scheduler