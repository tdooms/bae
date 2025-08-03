# %%
import wandb
import pandas as pd
import plotly.express as px

from itertools import islice
from tqdm import tqdm

api = wandb.Api()
runs = api.runs("tdooms/coder")
# %%
results = []
for run in tqdm(islice(runs, 103-10, 103)):
    history = run.history()
    _, _, _, _, layer, expansion, alpha, _ = run.name.rsplit('-')
    
    mse = history["mse"].tail(20).mean()
    reg = history["reg"].tail(20).mean()
    
    results.append(dict(expansion=int(expansion[1:]), alpha=int(alpha[1:])/100, mse=mse, reg=reg))


df = pd.DataFrame(results)
df['expansion'] = df['expansion'].astype(int)
df['alpha'] = df['alpha'].astype(float)
df = df.sort_values(['expansion', 'alpha'], ascending=[False, False])
print(df)
# %%
fig = px.line(df, x="mse", y="reg", color="expansion", template="plotly_white")
fig
# %%