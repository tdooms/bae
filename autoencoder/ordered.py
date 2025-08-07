import torch

from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from einops import einsum
from quimb.tensor import Tensor

from utils import Muon
from autoencoder.base import Autoencoder, Config, hoyer, masked_mean

class Ordered(Autoencoder, kind="ordered"):
    """A tensor-based autoencoder class which mixes its features."""
    def __init__(self, model, config) -> None:
        super().__init__(model, config)
        
        # self.triu = nn.Buffer(torch.triu(torch.ones(config.d_features, config.d_features)), persistent=False)
        self.counts = nn.Buffer(torch.arange(1, self.config.d_features + 1), persistent=False)

        self.left = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.right = nn.Parameter(torch.empty(config.d_features, config.d_model))
        
        torch.nn.init.orthogonal_(self.left.data)
        torch.nn.init.orthogonal_(self.right.data)
    
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
    
    def loss(self, acts, mask, alpha):
        f = self.features(acts)
        
        # Compute the regularisation term
        sparsity = hoyer(f)
        reg = (1 - alpha * (sparsity.cumsum(dim=0) / self.counts)) / self.config.d_features
        
        # Compute the reconstruction and the ordering kernels
        kernel = self.kernel()
        order = einsum(self.triu, reg, self.triu, "f1 reg, reg, f2 reg -> f1 f2")
        
        # Compute the self and cross terms of the loss and combine them
        # NOTE: one could cumsum reg, but it's slower than the matmul for some reason
        recons = einsum(f, f, kernel * order, "... f1, ... f2, f1 f2 -> ...")
        cross = einsum(f, f, self.triu, reg, "... f, ... f, f reg, reg -> ...")
        loss = masked_mean(recons - 2 * cross + 1.0, mask)
        
        # Compute the reconstruction error without the regularisation
        with torch.no_grad():
            recons = einsum(f, f, kernel, "... f1, ... f2, f1 f2 -> ...")
            mse = masked_mean(recons - 2 * f.pow(2).sum(-1) + 1.0, mask)
        
        return loss, f, dict(mse=mse, reg=sparsity.mean())
    
    def optimizers(self, max_steps, lr=0.01, cooldown=0.5):
        optimizer = Muon(list(self.parameters()), lr=lr, weight_decay=0, momentum=0.95, nesterov=False)
        scheduler = LambdaLR(optimizer, lambda step: min(1.0, (1.0 / (cooldown - 1.0)) * ((step / max_steps) - 1.0)))
        return optimizer, scheduler
