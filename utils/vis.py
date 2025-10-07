from utils.hook import Input, Lazy, Hooked
from tqdm import tqdm
import torch

def to_black_diverging_color(values):
    """Simple red-black-blue color map."""
    r = torch.where(values < 0.5, 0.0, (values - 0.5) * 2.0)
    g = torch.zeros_like(values)
    b = torch.where(values > 0.5, 0.0, (0.5 - values) * 2.0)
    
    return (torch.stack([r, g, b], dim=-1) * 255).int()

class Vis:
    def __init__(self, model, autoencoder, dataset, tokenizer, batches=16, batch_size=32):
        hooked = Hooked(model, Input(model.model.layers[autoencoder.config.layer]))
        self.lazy = Lazy(lambda **batch: autoencoder(hooked(**batch)), dataset, device=model.device)
        self.tokenizer = tokenizer
        
        iterator = map(lambda i: self.lazy[i*batch_size:(i+1)*batch_size][0], range(batches))
        extrema = torch.cat([torch.stack([f.max(-2).values, f.min(-2).values], dim=-1) for f in tqdm(iterator, total=batches)], dim=0)
        maxima, minima = extrema.unbind(-1)

        _, max_inds = maxima.topk(dim=0, k=100)
        _, min_inds = minima.topk(dim=0, k=100, largest=False)
        
        self.acts = dict(max=max_inds.cpu(), min=min_inds.cpu())
        
    @staticmethod
    def color_str(str, color):
        r, g, b = color
        str = (str[2:] if str.startswith("##") else " " + str).replace('\n', ' ')
        return f"\033[48;2;{int(r)};{int(g)};{int(b)}m{str}\033[0m"
    
    @staticmethod
    def color_line(line, colors, start, end):
        return "".join([Vis.color_str(line[i], colors[i]) for i in range(start, end)]).replace('Ġ', '')

    def color_inputs(self, inds, feature, view=range(-10, 10), dark=True, largest=True):
        features, batch = self.lazy[inds]
        values = features[..., feature]
        tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in batch['input_ids']]
        
        normalized = -(values / values.abs().max()) / 2.0 + 0.5
        colors = to_black_diverging_color(normalized.cpu())
        
        for line, color, value in zip(tokens, colors, values):
            vals, inds = value.topk(k=1, dim=-1, largest=largest)
            start, end = max(0, inds.item() + view.start), min(len(color), len(line), inds.item() + view.stop)

            print(f"{vals.item():<4.2f}:  {Vis.color_line(line, color, start, end)}")
        print()
    
    def __call__(self, *args, k=3, **kwargs):
        assert k <= 100, "Amount must be less than or equal to 100"
        
        # TODO: be somewhat smarter about this
        args = [x for arg in args for x in ([arg] if not isinstance(arg, (list, tuple)) else arg)]
        
        for feature in args:
            print(f"Feature {feature}:")
            indices = self.acts["max"][:, feature][:k].cpu()
            self.color_inputs(indices, feature, largest=True, **kwargs)
            
            indices = self.acts["min"][:, feature][:k].cpu()
            self.color_inputs(indices, feature, largest=False, **kwargs)
            print()