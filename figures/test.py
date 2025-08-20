# %%
%load_ext autoreload
%autoreload 2

import wandb
api = wandb.Api()

filters = {"tags": "sparsity-sweep", "config.tags": ["v2"]}
runs = api.runs(path="tdooms/coder", filters=filters)

len(runs)
# %%