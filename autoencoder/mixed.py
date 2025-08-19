import torch

from torch import nn
from torch.optim.lr_scheduler import LinearLR
from quimb.tensor import Tensor

from utils import Muon
from autoencoder.base import Autoencoder, Config, hoyer_density, masked_mean, blocked_inner

class Mixed(Autoencoder, kind="mixed"):
    """A tensor-based autoencoder class which mixes its features."""
    def __init__(self, model, config) -> None:
        super().__init__(model, config)
        
        self.left = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.right = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.down = nn.Parameter(torch.empty(config.d_bottleneck, config.d_features))
        
        torch.nn.init.orthogonal_(self.left.data)
        torch.nn.init.orthogonal_(self.right.data)
        torch.nn.init.orthogonal_(self.down.data)

    @staticmethod
    def from_config(model, **kwargs):
        return Mixed(model, Config(kind="mixed", **kwargs))
    
    def network(self, mod='inp'):
        u = torch.stack([self.left + self.right, self.left - self.right], dim=0)
        
        return Tensor(u, inds=[f"s:{mod}", f'f:{mod}', f'in:0'], tags=['U']) \
             & Tensor(u, inds=[f"s:{mod}", f'f:{mod}', f'in:1'], tags=['U']) \
             & Tensor(self.down, inds=[f'h:{mod}', f'f:{mod}'], tags=['D']) \
             & Tensor(torch.tensor([1, -1], **self._like()) / 4.0, inds=[f's:{mod}'], tags=['S'])
    
    def features(self, acts):
        return nn.functional.linear(acts, self.left) * nn.functional.linear(acts, self.right)
    
    @torch.compile(fullgraph=True)
    def loss(self, acts, mask, alpha):
        # Compute the features and hidden representation
        f = self.features(acts)
        h = nn.functional.linear(f, self.down)
        g = nn.functional.linear(h, self.down.T)
        
        # Compute the regularisation term
        density = hoyer_density(f).mean()
        
        # Compute the self and cross terms of the loss
        recons = blocked_inner(g, self.left, self.right, self.inds)
        cross = h.square().sum(-1)
        
        # Compute the reconstruction and the loss
        error = masked_mean(recons - 2 * cross + 1.0, mask)
        return error + alpha * density, f, dict(mse=error, reg=density)
    
    def optimizers(self, max_steps, lr=0.03):
        optimizer = Muon(list(self.parameters()), lr=lr, weight_decay=0, nesterov=False)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=max_steps)
        return optimizer, scheduler