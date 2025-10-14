from tqdm import tqdm
from torch.utils.data import DataLoader
from einops import einsum

import pandas as pd
import plotly.express as px
import torch

from autoencoders.utils import Input, Hooked

class Manifold:
    def __init__(self, model, autoencoder, dataset, tokenizer, form, **kwargs):
        self.eigvals, self.eigvecs = torch.linalg.eigh(form.float())
        self.dataset = dataset
        self.hooked = Hooked(model, Input(model.model.layers[autoencoder.config.layer]))
        self.tokenizer = tokenizer
        
        self.sample(**kwargs)
    
    def spectrum(self, threshold=1e-3):
        filtered = self.eigvals[self.eigvals.abs() > threshold]
        
        fig = px.line(filtered.cpu(), template='plotly_white', width=500, height=200)
        return fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False)

    def sample(self, batch_size=2**5, batches=2**3):
        inds = self.eigvals.abs().topk(k=3, dim=-1).indices
        
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        vals, feats, inputs, outputs = [], [], [], []

        for batch, _ in tqdm(zip(loader, range(batches)), total=batches):
            batch = {k: v.to(self.hooked.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}    
            output, acts = self.hooked(**batch, output=True)
            
            inputs += [batch['input_ids']]
            outputs += [output.logits.argmax(dim=-1)]
            
            acts = acts * acts.pow(2).sum(-1, keepdim=True).rsqrt()
            
            if self.eigvals.size(-1) == acts.size(-1) + 1:
                acts = torch.cat([torch.ones_like(acts[..., :1]), acts], dim=-1)

            v = einsum(acts, self.eigvecs.to(self.hooked.dtype), "... x, x v -> ... v")
            
            feats += [v[..., inds]]
            vals += [einsum(v.pow(2), self.eigvals.to(self.hooked.dtype), "... v, v -> ... v").pow(2).sum(-1)]
        
        self.vals = torch.stack(vals, dim=0).flatten(0, 2)
        self.feats = torch.stack(feats, dim=0).flatten(0, 2)
        self.inputs = torch.stack(inputs, dim=0).flatten(0, 2)
        self.outputs = torch.stack(outputs, dim=0).flatten(0, 2)
    
    def to_dataframe(self, k=2**13, total=-1):
        k = min(k, self.vals.size(0)-1)
        # I know this throws away a single sample but I'm too lazy to fix it right now.
        max_vals, max_inds = self.vals[:total].topk(k=k, dim=0, largest=True)
        max_feats = self.feats[max_inds]

        # Compute the current token string combined with the model prediction token (inp -> out)
        inp_toks = self.tokenizer.convert_ids_to_tokens(self.inputs[max_inds].cpu())
        out_toks = self.tokenizer.convert_ids_to_tokens(self.outputs[max_inds].cpu())
        tokens = [(i + ' -> ' + o).replace('Ġ', ' ') for i, o in zip(inp_toks, out_toks)]
        
        return pd.DataFrame(dict(
            x=max_feats[:, 0].float().detach().cpu().numpy(),
            y=max_feats[:, 1].float().detach().cpu().numpy(),
            z=max_feats[:, 2].float().detach().cpu().numpy(),
            value=max_vals.float().detach().cpu().numpy(),
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
            hover_data={"x": False, "y": False, "z": False, "value": False, "token": False},
            color_continuous_midpoint=0.0,
            color_continuous_scale="RdBu",
            height=800, 
            width=800
        )

        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.update_layout(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False))
        fig.update_layout(coloraxis_showscale=False)
        return fig
