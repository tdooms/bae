import torch

from torch import nn
from quimb.tensor import Tensor
from itertools import combinations_with_replacement

from autoencoders.base import Autoencoder, Config
from autoencoders.utils import tiled_product

class Mixed(Autoencoder, kind="mixed"):
    """A tensor-based autoencoder class which mixes its features."""
    def __init__(self, config) -> None:
        super().__init__(config)
        self.tiles = 4
        self.inds = list(combinations_with_replacement(range(0, self.tiles), 2))
        
        self.left = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.right = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.down = nn.Parameter(torch.empty(config.d_bottleneck, config.d_features))
        
        torch.nn.init.orthogonal_(self.left.data)
        torch.nn.init.orthogonal_(self.right.data)
        torch.nn.init.orthogonal_(self.down.data)

    @staticmethod
    def from_config(**kwargs):
        return Mixed(Config(kind="mixed", **kwargs))

    def network(self, mod='inp'):
        u = torch.stack([self.left + self.right, self.left - self.right], dim=0)
        
        return Tensor(u, inds=[f"s:{mod}", f'f:{mod}', f'in:0'], tags=['U']) \
             & Tensor(u, inds=[f"s:{mod}", f'f:{mod}', f'in:1'], tags=['U']) \
             & Tensor(self.down, inds=[f'h:{mod}', f'f:{mod}'], tags=['D']) \
             & Tensor(torch.tensor([1, -1], **self._like()) / 4.0, inds=[f's:{mod}'], tags=['S'])
    
    def forward(self, x):
        x = x * x.square().sum(dim=-1, keepdim=True).rsqrt()
        return nn.functional.linear(x, self.left) * nn.functional.linear(x, self.right)
    
    # @torch.compile(fullgraph=True)
    def loss_fn(self, x, mask, scale):
        # Compute the features and hidden representation
        f = self.features(x)
        h = nn.functional.linear(f, self.down)
        g = nn.functional.linear(h, self.down.T)
        
        # Compute the regularisation term
        density = (f.norm(p=1, dim=0) / f.norm(p=2, dim=0) - 1.0).mean() / (f.size(0)**0.5 - 1.0)
        
        # Compute the self and cross terms of the loss
        recons = tiled_product(g, self.left, self.right, self.tiles, self.inds)
        cross = h.square().sum(-1)
        
        # Compute the reconstruction and the loss
        error = ((recons - 2 * cross).sum() / mask.sum()) + 1.0
        return error + scale * self.config.alpha * density, dict(mse=error, reg=density)