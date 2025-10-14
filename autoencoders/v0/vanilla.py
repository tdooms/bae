import torch

from torch import nn
from quimb.tensor import Tensor
from itertools import combinations_with_replacement

from autoencoders.base import Autoencoder, Config
from autoencoders.utils import tiled_product

class Vanilla(Autoencoder, kind="vanilla"):
    """A tensor-based autoencoder class which mixes its features."""
    def __init__(self, config) -> None:
        super().__init__(config)
        self.tiles = 4
        self.inds = list(combinations_with_replacement(range(0, self.tiles), 2))

        self.left = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.right = nn.Parameter(torch.empty(config.d_features, config.d_model))
        
        torch.nn.init.orthogonal_(self.left.data)
        torch.nn.init.orthogonal_(self.right.data)

    @staticmethod
    def from_config(**kwargs):
        return Vanilla(Config(kind="vanilla", **kwargs))

    def network(self, mod='inp'):
        u = torch.stack([self.left + self.right, self.left - self.right], dim=0)

        return Tensor(u, inds=[f"s:{mod}", f'f:{mod}', 'i:0'], tags=['U']) \
             & Tensor(u, inds=[f"s:{mod}", f'f:{mod}', 'i:1'], tags=['U']) \
             & Tensor(torch.tensor([1, -1], **self._like()) / 4.0, inds=[f's:{mod}'], tags=['S'])

    def forward(self, x):
        x = x * x.square().sum(dim=-1, keepdim=True).rsqrt()
        return nn.functional.linear(x, self.left) * nn.functional.linear(x, self.right)

    @torch.compile(fullgraph=True)
    def loss_fn(self, x, mask, scale):
        # Mask input and compute features and their hoyer density
        features = (self(x) * mask[..., None]).flatten(0, -2)
        density = (features.norm(p=1, dim=0) / features.norm(p=2, dim=0) - 1.0).mean() / (features.size(0)**0.5 - 1.0)

        # Compute the reconstruction error terms
        recons = tiled_product(features, self.left, self.right, self.tiles, self.inds)
        cross = features.square().sum(-1)

        # Compute the (masked) mean of the reconstruction errors (x^2 - 2xy + 1) * mask / mask.sum()
        error = ((recons - 2 * cross).sum() / mask.sum()) + 1.0
        return error + scale * self.config.alpha * density, dict(mse=error, reg=density)