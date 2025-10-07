import torch

from torch import nn
from torch.optim.lr_scheduler import LinearLR
from quimb.tensor import Tensor

from utils import Muon
from autoencoders.base import Autoencoder, Config, hoyer_density, masked_mean, blocked_inner

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

    def network(self, mod='inp'):
        u = torch.stack([self.left + self.right, self.left - self.right], dim=0)

        return Tensor(u, inds=[f"s:{mod}", f'f:{mod}', 'i:0'], tags=['U']) \
             & Tensor(u, inds=[f"s:{mod}", f'f:{mod}', 'i:1'], tags=['U']) \
             & Tensor(torch.tensor([1, -1], **self._like()) / 4.0, inds=[f's:{mod}'], tags=['S'])

    def features(self, acts):
        return nn.functional.linear(acts, self.left) * nn.functional.linear(acts, self.right)

    @torch.compile(fullgraph=True)
    def loss(self, acts, mask, alpha):
        f = self.features(acts)
        density = hoyer_density(f).mean()

        recons = blocked_inner(f, self.left, self.right, self.inds)
        cross = f.square().sum(-1)

        error = masked_mean(recons - 2 * cross + 1.0, mask)
        return error + alpha * density, f, dict(mse=error, reg=density)

    def optimizers(self, max_steps, lr=0.03):
        optimizer = Muon(list(self.parameters()), lr=lr, weight_decay=0, nesterov=False)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=max_steps)
        return optimizer, scheduler