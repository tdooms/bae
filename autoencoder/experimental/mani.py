import torch

from torch import nn
from torch.optim.lr_scheduler import LinearLR
from einops import einsum
from quimb.tensor import Tensor

from utils import Muon
from autoencoders.base import Autoencoder, Config, hoyer_density, masked_mean


class Mani(Autoencoder, kind="mani"):
    """A tensor-based autoencoder class which mixes its features."""
    def __init__(self, model, config) -> None:
        super().__init__(model, config)
        
        self.left = nn.Linear(config.d_model, config.d_features, bias=True)
        self.right = nn.Linear(config.d_model, config.d_features, bias=True)
        self.down = nn.Linear(config.d_features, config.d_bottleneck, bias=False)
        
        self.offset = nn.Parameter(torch.zeros(config.d_bottleneck))
        
        torch.nn.init.orthogonal_(self.left.weight)
        torch.nn.init.orthogonal_(self.right.weight)
        torch.nn.init.orthogonal_(self.down.weight)
        
        torch.nn.init.zeros_(self.left.bias)
        torch.nn.init.zeros_(self.right.bias)

    @staticmethod
    def from_config(model, **kwargs):
        return Mani(model, Config(kind="mani", **kwargs))
    
    def network(self, mod='inp'):
        pass
    
    def kernel(self):
        l = torch.cat([self.left.bias[:, None], self.left.weight], dim=1)
        r = torch.cat([self.right.bias[:, None], self.right.weight], dim=1)
        return self.down.weight @ ((l @ l.T) * (r @ r.T)) @ self.down.weight.T

    def features(self, acts):
        return nn.functional.linear(self.left(acts) * self.right(acts), self.down.weight) + self.offset
    
    @torch.compile(fullgraph=True)
    def loss(self, acts, mask, alpha):
        # Compute the features and hidden representation
        # f = self.features(acts)
        
        # # Compute the regularisation term
        # sparsity = hoyer(f + self.offset).mean()
        # reg = 1.0 - alpha * sparsity 
        
        # # Compute the self and cross terms of the loss
        # recons = einsum(f, f, self.kernel(), "... f1, ... f2, f1 f2 -> ...")
        # cross = f.square().sum(-1)
        
        # # Compute the reconstruction and the loss
        # error = masked_mean(recons - 2 * cross + 4.0, mask) / 4.0
        # loss = masked_mean(recons * reg - 2 * cross * reg + 4.0, mask) / 4.0

        # return loss, f, dict(mse=error, reg=sparsity)
        pass
    
    def optimizers(self, max_steps, lr=0.03):
        optimizer = Muon(list(self.parameters()), lr=lr, weight_decay=0, nesterov=False)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=max_steps)
        return optimizer, scheduler