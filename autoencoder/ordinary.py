import torch

from torch import nn
from torch.optim.lr_scheduler import LinearLR
from utils import Muon

from autoencoder.base import Autoencoder, Config, hoyer, masked_mean

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

class Ordinary(Autoencoder, kind="ordinary"):
    """A tensor-based autoencoder class which mixes its features."""
    def __init__(self, model, config) -> None:
        super().__init__(model, config)
        self.activation = TopK(50)
        
        self.encoder = nn.Parameter(torch.empty(config.d_features, config.d_model))
        self.decoder = nn.Parameter(torch.empty(config.d_model, config.d_features))

        torch.nn.init.xavier_uniform_(self.encoder.data)
        self.decoder.data = self.encoder.data.T.detach().contiguous().clone()

    @staticmethod
    def from_config(model, **kwargs):
        return Ordinary(model, Config(kind="ordinary", **kwargs))
    
    def kernel(self):
        raise NotImplementedError("Ordinary autoencoders do not use a kernel.")

    def network(self, mod='inp'):
        raise NotImplementedError("Ordinary autoencoders do not use a network representation.")
    
    def features(self, acts):
        return self.activation(nn.functional.linear(acts, self.encoder))

    # @torch.compile(fullgraph=True)
    def loss(self, acts, mask, _):
        f = self.features(acts)
        recons = nn.functional.linear(f, self.decoder)

        sparsity = hoyer(f).mean()
        
        error = masked_mean((recons - acts).pow(2).sum(dim=-1), mask)
        return error, f, dict(mse=error, reg=sparsity)

    def optimizers(self, max_steps, lr=0.03):
        optimizer = Muon(list(self.parameters()), lr=lr, weight_decay=0, nesterov=False)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=max_steps)
        return optimizer, scheduler