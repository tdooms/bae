import torch

from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from einops import einsum
from quimb.tensor import Tensor

from utils import Muon
from autoencoder.base import Autoencoder, Config, hoyer, masked_mean

class Vanilla(Autoencoder, kind="vanilla"):
    """A tensor-based autoencoder class which mixes its features."""
    def __init__(self, model, config) -> None:
        super().__init__(model, config)

        self.left = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.right = nn.Parameter(torch.empty(config.d_features, config.d_model))
        
        torch.nn.init.orthogonal_(self.left.data)
        torch.nn.init.orthogonal_(self.right.data)

    @staticmethod
    def from_config(model, **kwargs):
        return Vanilla(model, Config(kind="vanilla", **kwargs))
    
    def kernel(self):
        return (self.left @ self.left.T) * (self.right @ self.right.T)

    def network(self, mod='inp'):
        u = torch.stack([self.left + self.right, self.left - self.right], dim=0)

        return Tensor(u, inds=[f"s:{mod}", f'f:{mod}', f'i:0'], tags=['U']) \
             & Tensor(u, inds=[f"s:{mod}", f'f:{mod}', f'i:1'], tags=['U']) \
             & Tensor(torch.tensor([1, -1], **self._like()) / 4.0, inds=[f's:{mod}'], tags=['S'])
    
    def features(self, acts):
        return nn.functional.linear(acts, self.left) * nn.functional.linear(acts, self.right)

    @torch.compile(fullgraph=True)
    def loss(self, acts, mask, alpha):
        f = self.features(acts)
        
        # Compute the regularisation term
        sparsity = hoyer(f).mean()
        reg = 1.0 - alpha * sparsity
        
        # Compute the self and cross terms of the loss
        recons = einsum(f, f, self.kernel(), "... h1, ... h2, h1 h2 -> ...")
        cross = f.square().sum(-1)

        # Compute the reconstruction and the loss
        mse = masked_mean(recons - 2 * cross + 1.0, mask)
        loss = masked_mean(recons * reg - 2 * cross * reg + 1.0, mask)

        return loss, f, dict(mse=mse, reg=sparsity)

    def optimizers(self, max_steps, lr=0.01, cooldown=0.5):
        optimizer = Muon(list(self.parameters()), lr=lr, weight_decay=0, momentum=0.95, nesterov=False)
        scheduler = LambdaLR(optimizer, lambda step: min(1.0, (1.0 / (cooldown - 1.0)) * ((step / max_steps) - 1.0)))
        return optimizer, scheduler