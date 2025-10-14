# %% Test wether the ``gram_block`` function produces the expected matrix.
%load_ext autoreload
%autoreload 2

from autoencoders.utils import gram_block
import torch
import plotly.express as px

torch.set_grad_enabled(False)
# %%
px.imshow(gram_block(1024, 4, 0, 0).cpu(), color_continuous_scale='Blues').show()
px.imshow(gram_block(1024, 4, 1, 0).cpu(), color_continuous_scale='Blues').show()
px.imshow(gram_block(1024, 4, 0, 2).cpu(), color_continuous_scale='Blues').show()
px.imshow(gram_block(1024, 4, 3, 3).cpu(), color_continuous_scale='Blues').show()
# %%