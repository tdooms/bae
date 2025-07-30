from itertools import product

def inv_hoyer(x, dim=0):
    size = x.size(dim) if isinstance(dim, int) else product(x.size(d) for d in dim)
    return (x.norm(p=1, dim=dim) / x.norm(p=2, dim=dim) - 1.0) / (size**0.5 - 1.0)

def hoyer(x, dim=0):
    size = x.size(dim) if isinstance(dim, int) else product(x.size(d) for d in dim)
    return ((size**0.5 - 1.0) - (x.norm(p=1, dim=dim) / x.norm(p=2, dim=dim))) / (size**0.5 - 1.0)