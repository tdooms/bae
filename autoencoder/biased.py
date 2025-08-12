import torch

from torch import nn
from torch.optim.lr_scheduler import LambdaLR, LinearLR
from einops import einsum
from quimb.tensor import Tensor

from utils import Muon
from autoencoder.base import Autoencoder, Config, hoyer, masked_mean

class Biased(Autoencoder, kind="biased"):
    """A tensor-based autoencoder class which mixes its features."""
    def __init__(self, model, config) -> None:
        super().__init__(model, config)

        self.left = nn.Linear(config.d_model, config.d_features, bias=True)
        self.right = nn.Linear(config.d_model, config.d_features, bias=True)

        torch.nn.init.orthogonal_(self.left.weight)
        torch.nn.init.orthogonal_(self.right.weight)
        
        torch.nn.init.zeros_(self.left.bias)
        torch.nn.init.zeros_(self.right.bias)

    @staticmethod
    def from_config(model, **kwargs):
        return Biased(model, Config(kind="biased", **kwargs))
    
    def kernel(self):
        l = torch.cat([self.left.bias[:, None], self.left.weight], dim=1)
        r = torch.cat([self.right.bias[:, None], self.right.weight], dim=1)
        return (l @ l.T) * (r @ r.T)

    def network(self, mod='inp'):
        u = torch.stack([self.left + self.right, self.left - self.right], dim=0)

        return Tensor(u, inds=[f"s:{mod}", f'f:{mod}', f'i:0'], tags=['U']) \
             & Tensor(u, inds=[f"s:{mod}", f'f:{mod}', f'i:1'], tags=['U']) \
             & Tensor(torch.tensor([1, -1], **self._like()) / 4.0, inds=[f's:{mod}'], tags=['S'])
    
    def features(self, acts):
        return self.left(acts) * self.right(acts)

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
        error = masked_mean(recons - 2 * cross + 4.0, mask) / 4.0
        loss = masked_mean(recons * reg - 2 * cross * reg + 4.0, mask) / 4.0

        return loss, f, dict(mse=error, reg=sparsity)

    def optimizers(self, max_steps, lr=0.03, cooldown=0.5):
        optimizer = Muon(list(self.parameters()), lr=lr, weight_decay=0, momentum=0.95, nesterov=False)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=max_steps)
        return optimizer, scheduler