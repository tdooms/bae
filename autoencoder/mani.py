import torch

from torch import nn
from torch.optim.lr_scheduler import LinearLR
from einops import einsum
from quimb.tensor import Tensor

from utils import Muon
from autoencoder.base import Autoencoder, Config, hoyer, masked_mean


class Mani(Autoencoder, kind="mani"):
    """A tensor-based autoencoder class which mixes its features."""
    def __init__(self, model, config) -> None:
        super().__init__(model, config)
        
        self.left = nn.Linear(config.d_model, config.d_features, bias=True)
        self.right = nn.Linear(config.d_model, config.d_features, bias=True)
        self.down = nn.Linear(config.d_features, config.d_bottleneck, bias=True)
        
        torch.nn.init.orthogonal_(self.left.weight)
        torch.nn.init.orthogonal_(self.right.weight)
        torch.nn.init.orthogonal_(self.down.data)

    @staticmethod
    def from_config(model, **kwargs):
        return Mani(model, Config(kind="mani", **kwargs))
    
    def network(self, mod='inp'):
        u = torch.stack([self.left + self.right, self.left - self.right], dim=0)
        
        return Tensor(u, inds=[f"s:{mod}", f'f:{mod}', f'in:0'], tags=['U']) \
             & Tensor(u, inds=[f"s:{mod}", f'f:{mod}', f'in:1'], tags=['U']) \
             & Tensor(self.down, inds=[f'h:{mod}', f'f:{mod}'], tags=['D']) \
             & Tensor(torch.tensor([1, -1], **self._like()) / 4.0, inds=[f's:{mod}'], tags=['S'])
    
    def kernel(self):
        return self.down @ ((self.left @ self.left.T) * (self.right @ self.right.T)) @ self.down.T
    
    def features(self, acts):
        return nn.functional.linear(acts, self.left) * nn.functional.linear(acts, self.right)
    
    @torch.compile(fullgraph=True)
    def loss(self, acts, mask, alpha):
        # Compute the features and hidden representation
        f = self.features(acts)
        h = nn.functional.linear(f, self.down)
        
        # Compute the regularisation term
        sparsity = hoyer(h).mean()
        reg = 1.0 - alpha * sparsity 
        
        # Compute the self and cross terms of the loss
        recons = einsum(h, h, self.kernel(), "... h1, ... h2, h1 h2 -> ...")
        cross = h.square().sum(-1)
        
        # Compute the reconstruction and the loss
        error = masked_mean(recons - 2 * cross + 1.0, mask)
        loss = masked_mean(recons * reg - 2 * cross * reg + 1.0, mask)

        return loss, f, dict(mse=error, reg=sparsity)
    
    def optimizers(self, max_steps, lr=0.03):
        optimizer = Muon(list(self.parameters()), lr=lr, weight_decay=0, nesterov=False)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=max_steps)
        return optimizer, scheduler