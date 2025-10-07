import torch

from torch import nn
from torch.optim.lr_scheduler import LinearLR
from einops import einsum
from quimb.tensor import Tensor

from utils import Muon

from autoencoders.base import Autoencoder, Config, masked_mean, block_indices

class TopK(nn.Module):
    """A module that selects the top-k features from the input."""
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        indices = x.topk(k=self.k, dim=-1).indices

        mask = torch.zeros_like(x)
        mask.scatter_(-1, indices, 1)
        
        return x * mask
    
class Lifted(Autoencoder, kind="lifted"):
    """A tensor-based autoencoder class which mixes its features."""
    def __init__(self, model, config) -> None:
        super().__init__(model, config)

        self.ileft = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.iright = nn.Parameter(torch.empty(config.d_features, config.d_model))
        
        self.oleft = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.oright = nn.Parameter(torch.empty(config.d_features, config.d_model))

        self.act = TopK(500)

        torch.nn.init.orthogonal_(self.ileft.data)
        torch.nn.init.orthogonal_(self.iright.data)

        self.oleft.data = self.ileft.data.contiguous().clone()
        self.oright.data = self.iright.data.contiguous().clone()

        self.inds = block_indices(config.d_features)

    @staticmethod
    def from_config(model, **kwargs):
        return Lifted(model, Config(kind="lifted", **kwargs))
    
    def kernel(self):
        return (self.oleft @ self.oleft.T) * (self.oright @ self.oright.T)

    def network(self, mod='inp'):
        pass
    
    def features(self, acts):
        return self.act(nn.functional.linear(acts, self.ileft) * nn.functional.linear(acts, self.iright))

    @torch.compile(fullgraph=True)
    def loss(self, acts, mask, alpha):
        f1 = self.features(acts)
        f2 = nn.functional.linear(acts, self.oleft) * nn.functional.linear(acts, self.oright)
        
        # Compute the self and cross terms of the loss
        recons = einsum(f1, f1, self.kernel(), "... h1, ... h2, h1 h2 -> ...")
        # recons = tiled_inner_product(f1, self.oleft, self.oright, self.inds)
        cross = (f1 * f2).sum(-1)

        # Compute the reconstruction and the loss
        error = masked_mean(recons - 2 * cross + 1.0, mask)
        return error, f1, dict(mse=error, reg=0)

    def optimizers(self, max_steps, lr=0.03):
        optimizer = Muon(list(self.parameters()), lr=lr, weight_decay=0, nesterov=False)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=max_steps)
        return optimizer, scheduler