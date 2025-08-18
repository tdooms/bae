from einops import *
from collections import defaultdict

class Input:
    def __init__(self, inner):
        self.inner = inner

class Output:
    def __init__(self, inner):
        self.inner = inner

class Hooked:
    """A simple combination of NNSight and TransformerLens since I didn't want either as dependency."""
    def __init__(self, model, *args, **kwargs):
        self.device = model.device
        self.dtype = model.dtype
        self.model = model
        self.modules = kwargs | {i: arg for i, arg in enumerate(args)}
        self.activations = defaultdict()
    
    def hook(self, name, module):
        kind = 'input' if isinstance(module, Input) else 'output'
        def cache(_module, input, output):
            self.activations[name] = dict(input=input[0], output=output)[kind]
            
        return module.inner.register_forward_hook(cache)
        
    def __call__(self, *args, **kwargs):
        handles = [self.hook(name, module) for name, module in self.modules.items()]
        logits = self.model(*args, **kwargs)
        _ = [handle.remove() for handle in handles]
        
        cache = self.activations
        self.activations = defaultdict()
        return logits, cache