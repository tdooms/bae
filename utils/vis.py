from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot
from matplotlib.colors import LinearSegmentedColormap
import torch

def create_black_diverging_cmap():
    colors = [
        (0.0,  (0.0, 0.0, 0.7)),    # Dark blue
        (0.25, (0.0, 0.4, 1.0)),    # Medium blue
        (0.45, (0.2, 0.2, 0.2)),    # Near-black
        (0.5,  (0.0, 0.0, 0.0)),    # Pure black
        (0.55, (0.2, 0.2, 0.2)),    # Near-black
        (0.75, (1.0, 0.4, 0.0)),    # Medium red
        (1.0,  (0.7, 0.0, 0.0))     # Dark red
    ]
    
    positions = [x[0] for x in colors]
    rgb_colors = [x[1] for x in colors]
    return LinearSegmentedColormap.from_list('black_diverging', list(zip(positions, rgb_colors)))

class Vis:
    def __init__(self, model, tokenizer, dataset, acts=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.acts = acts if acts is not None else self.max_activations(**kwargs)
    
    def max_activations(self, batch_size=32, max_steps=2**6, k=500):
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        maxima, minima = [], []
        
        for batch, _ in tqdm(zip(loader, range(max_steps)), total=max_steps):
            batch = {k: v.to(self.model.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            acts = self.model(**batch)['features']
            maxima.append(acts.max(1).values)
            minima.append(acts.min(1).values)
        
        maxima = torch.cat(maxima, dim=0).topk(dim=0, k=k).indices
        minima = torch.cat(minima, dim=0).topk(dim=0, k=k, largest=False).indices
        return dict(max=maxima, min=minima)
    
    @staticmethod
    def color_str(str, color):
        r, g, b = color
        str = str[2:] if str.startswith("##") else " " + str
        return f"\033[48;2;{int(r)};{int(g)};{int(b)}m{str}\033[0m"
    
    @staticmethod
    def color_line(line, colors, start, end):
        return "".join([Vis.color_str(line[i], colors[i]) for i in range(start, end)]).replace('Ġ', '')

    def color_inputs(self, batch, feature, view=range(-30, 20), dark=True, largest=True):
        features = self.model(**batch)['features']
        values = features[..., feature]
        tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in batch['input_ids']]
        normalized = -(values / values.abs().max()) / 2.0 + 0.5
        
        colors = create_black_diverging_cmap()(normalized.cpu())[..., :3] if dark else pyplot.cm.bwr(normalized.cpu())[..., :3]
        colors = (colors * 255).astype(int)
        
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
            batch = {key: self.dataset[key][indices].cuda() for key in ['input_ids', 'attention_mask']}
            self.color_inputs(batch, feature, largest=True, **kwargs)
            
            indices = self.acts["min"][:, feature][:k].cpu()
            batch = {key: self.dataset[key][indices].cuda() for key in ['input_ids', 'attention_mask']}
            self.color_inputs(batch, feature, largest=False, **kwargs)
            print()