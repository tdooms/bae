# Bilinear Autoencoders

This is the official repository for the paper [finding manifolds with bilinear autoencoders](https://openreview.net/pdf?id=ybJXIh4vcF).

## Quickstart

This repo uses [uv](https://docs.astral.sh/uv/). The setup should be as easy as installing it and running:

```bash
uv sync
```

Once setup, an autoencoder can be loaded as follows:

```python
# Load the autoencoder used to generate the manifold from the paper
autoencoder = Autoencoder.load("Qwen/Qwen3-0.6B-Base", "mixed", layer=18, expansion=16, alpha=1.0, hf=True)
```

## Documentation

Work in progress, the idea is to have a precise walkthrough of some more critical implementation details.

## Organisation

### Autoencoders

The meat of the repo, pending some rewrites and documentation.

``v0`` contains the autoencoders from the original paper.
``v1`` contains improved autoencoders that use biases to describe more general manifold.

There is also a subfolder ``experimental`` which contains some semi-working versions for quick iterations.

### Workspace

I usually use two repos: one scientific and one cleaned up version.
I'm experimenting with just having the one, where I also openly keep my working files, which is this folder.
These are extremely messy and probably won't contain any valuable insights.

### Tutorials

Upon request (or whim), I'll add tutorials here. Please [reach out](https://tdooms.github.io/) if you have any ideas or suggestions (have a low threshold to do this).

### Figures

Reproduce any figure from the paper quickly.

### Utils

Additional functions and classes that I often needed.

### Public

Contains the frontend code for the manifold visualisation.
