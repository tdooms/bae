import torch
from dataclasses import dataclass

@dataclass
class BatchSampler:
    model: any
    dataset: any
    batch_size: int = 32
    device: str = 'cuda'
    
    @torch.no_grad()
    def __iter__(self):
        for batch in self.dataset.iter(batch_size=self.batch_size):
            batch = {k: v.to(self.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            yield self.model(**batch), batch

@dataclass
class IndexSampler:
    model: any
    dataset: any
    device: str = 'cuda'
    
    @torch.no_grad()
    def __getitem__(self, idx):
        batch = {key: self.dataset[key][idx].to(self.device) for key in ['input_ids', 'attention_mask']}
        return self.model(**batch), batch
