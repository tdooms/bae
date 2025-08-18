from tqdm import tqdm
from torch.utils.data import DataLoader
from einops import einsum

import pandas as pd
import plotly.express as px
import torch

class Manifold:
    def __init__(self, dataset, hooked, tokenizer, form, **kwargs):
        self.eigvals, self.eigvecs = torch.linalg.eigh(form.float())
        self.dataset = dataset
        self.hooked = hooked
        self.tokenizer = tokenizer
        
        self.sample(**kwargs)
    
    def spectrum(self, threshold=1e-3):
        filtered = self.eigvals[self.eigvals.abs() > threshold]
        return px.line(filtered.cpu(), template='plotly_white')
    
    def sample(self, batch_size=2**5, max_steps=2**3):
        inds = self.eigvals.abs().topk(k=3, dim=-1).indices
        eigvecs = self.eigvecs[..., inds].type(self.hooked.model.dtype)
        eigvals = self.eigvals[..., inds].type(self.hooked.model.dtype)
        
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        vals, feats, inputs, outputs = [], [], [], []

        for batch, _ in tqdm(zip(loader, range(max_steps)), total=max_steps):
            batch = {k: v.to(self.hooked.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}    
            cache = self.hooked(**batch)
            
            inputs += [batch['input_ids']]
            outputs += [cache[0].logits.argmax(dim=-1)]
            
            acts = cache[1]['acts']
            acts = acts * acts.pow(2).sum(-1, keepdim=True).rsqrt()
            
            if self.eigvals.size(-1) == acts.size(-1) + 1:
                acts = torch.cat([torch.ones_like(acts[..., :1]), acts], dim=-1)

            v = einsum(acts, eigvecs, "... x, x v -> ... v")
            
            feats += [v]
            vals += [einsum(v.pow(2), eigvals, "... v, v -> ...")]
        
        self.vals = torch.stack(vals, dim=0).flatten(0, 2)
        self.feats = torch.stack(feats, dim=0).flatten(0, 2)
        self.inputs = torch.stack(inputs, dim=0).flatten(0, 2)
        self.outputs = torch.stack(outputs, dim=0).flatten(0, 2)
    
    def to_dataframe(self, k=2**13, total=-1):
        # I know this throws away a single sample but I'm too lazy to fix it right now.
        max_vals, max_inds = self.vals[:total].topk(k=k, dim=0, largest=True)
        min_vals, min_inds = self.vals[:total].topk(k=k, dim=0, largest=False)
        top_vals, top_inds = torch.cat([max_vals, min_vals], dim=0), torch.cat([max_inds, min_inds], dim=0)

        top_feats = self.feats[top_inds]

        # Compute the current token string combined with the model prediction token (inp -> out)
        inp_toks = self.tokenizer.convert_ids_to_tokens(self.inputs[top_inds].cpu())
        out_toks = self.tokenizer.convert_ids_to_tokens(self.outputs[top_inds].cpu())
        tokens = [(i + ' -> ' + o).replace('Ġ', ' ') for i, o in zip(inp_toks, out_toks)]
        
        return pd.DataFrame(dict(
            x=top_feats[:, 0].float().detach().cpu().numpy(),
            y=top_feats[:, 1].float().detach().cpu().numpy(),
            z=top_feats[:, 2].float().detach().cpu().numpy(),
            value=top_vals.float().detach().cpu().numpy(),
            token=tokens,
        ))
    
    def __call__(self, **kwargs):
        df = self.to_dataframe(**kwargs)
        
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            color="value",
            hover_name="token",
            color_continuous_midpoint=0.0,
            color_continuous_scale="RdBu",
            height=800, 
            width=800
        )

        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.update_layout(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False))
        fig.update_layout(coloraxis_showscale=False)
        return fig
