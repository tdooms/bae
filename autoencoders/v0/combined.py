import torch

from torch import nn
from quimb.tensor import Tensor
from itertools import combinations_with_replacement

from autoencoders.base import Autoencoder, Config
from autoencoders.utils import tiled_masked_product, tiled_product

class Combined(Autoencoder, kind="combined"):
    """A tensor-based autoencoder class which mixes its features."""
    def __init__(self, config) -> None:
        super().__init__(config)
        self.tiles = 4
        self.inds = list(combinations_with_replacement(range(0, self.tiles), 2))
        
        ints = torch.arange(1, self.config.d_features + 1, dtype=torch.float).flip(0)
        self.counts = nn.Buffer(ints / self.config.d_features, persistent=False)
        self.htail = nn.Buffer(ints.reciprocal().cumsum(0).flip(0), persistent=False)

        self.left = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.right = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.down = nn.Parameter(torch.empty(config.d_bottleneck, config.d_features))
        
        torch.nn.init.orthogonal_(self.left.data)
        torch.nn.init.orthogonal_(self.right.data)
        torch.nn.init.orthogonal_(self.down.data)

    @staticmethod
    def from_config(**kwargs):
        return Combined(Config(kind="combined", **kwargs))
    
    def network(self, mod='inp'):
        u = torch.stack([self.left + self.right, self.left - self.right], dim=0)
        
        return Tensor(u, inds=[f"s:{mod}", f'f:{mod}', 'i:0'], tags=['U']) \
             & Tensor(u, inds=[f"s:{mod}", f'f:{mod}', 'i:1'], tags=['U']) \
             & Tensor(self.down, inds=[f'h:{mod}', f'f:{mod}'], tags=['D']) \
             & Tensor(torch.tensor([1, -1], **self._like()) / 4.0, inds=[f's:{mod}'], tags=['S'])
    
    def forward(self, x):
        x = x * x.square().sum(dim=-1, keepdim=True).rsqrt()
        return nn.functional.linear(x, self.left) * nn.functional.linear(x, self.right)
    
    @torch.compile(fullgraph=True)
    def loss_fn(self, x, mask, scale):
        # Compute the features
        f = (self(x) * mask[..., None]).flatten(0, -2)
        h = nn.functional.linear(f, self.down)
        g = nn.functional.linear(h, self.down.T)
        
        # Compute the regularisation term (mean of average prefix sum, phew)
        density = (f.norm(p=1, dim=0) / f.norm(p=2, dim=0) - 1.0).mean() / (f.size(0)**0.5 - 1.0)
        reg = self.config.alpha * scale * (density * self.htail).mean()

        # Compute the self and cross terms of the loss and combine them
        recons = tiled_masked_product(g, self.left, self.right, self.tiles, self.inds)
        cross = (f * g * self.counts).sum(-1)
        loss = ((recons - 2 * cross).sum() / mask.sum()) + 1.0
        
        # Compute the reconstruction error without the regularisation
        # This is solely for logging purposes and can be removed if needed
        with torch.no_grad():
            recons = tiled_product(g, self.left, self.right, self.tiles, self.inds)
            cross = h.square().sum(-1)
            error = ((recons - 2 * cross).sum() / mask.sum()) + 1.0

        return loss + reg, dict(mse=error, reg=density.mean())
