import torch
from dataclasses import dataclass

@dataclass
class BatchSampler:
    model: any
    dataset: any
    batch_size: int = 32
    n_ctx: int = 256
    device: str = 'cuda'
    keys: tuple = ('input_ids', 'attention_mask')
    
    @torch.no_grad()
    def __iter__(self):
        for batch in self.dataset.iter(batch_size=self.batch_size):
            batch = {k: v.to(self.device)[..., :self.n_ctx] for k, v in batch.items() if k in self.keys}
            yield self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']), batch

@dataclass
class IndexSampler:
    model: any
    dataset: any
    device: str = 'cuda'
    keys: tuple = ('input_ids', 'attention_mask')
    
    @torch.no_grad()
    def __getitem__(self, idx):
        batch = {key: self.dataset[key][idx].to(self.device) for key in self.keys}
        return self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']), batch
