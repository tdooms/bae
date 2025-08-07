# %%
%load_ext autoreload
%autoreload 2

import wandb
import pandas as pd
import plotly.express as px

from itertools import islice, chain
from tqdm import tqdm

from figures.constants import COLORS, FONT

api = wandb.Api()
runs = api.runs("tdooms/coder")
# %%

combined_sweep = list(islice(runs, 130-11, 130))
mixed_sweep = list(islice(runs, 165-11, 165))
vanilla_sweep = list(islice(runs, 176-11, 176))
ordered_sweep = list(islice(runs, 182-6, 182)) + list(islice(runs, 248-5, 248))

results = []
for run in tqdm(chain(ordered_sweep, mixed_sweep, vanilla_sweep, combined_sweep)):
    # print(run.name)
    history = run.history()
    _, _, _, kind, layer, expansion, alpha, _ = run.name.rsplit('-')
    
    mse = history["mse"].min()
    reg = history["reg"].min()
    
    if kind == "rainbow":
        kind = "combined"
    
    results.append(dict(kind=kind, alpha=int(alpha[1:])/100, mse=mse, reg=reg))

df = pd.DataFrame(results)
df['alpha'] = df['alpha'].astype(float)
df = df.sort_values(['kind', 'alpha'], ascending=[False, False])
print(df)
# %%
fig = px.line(df, x="mse", y="reg", color="kind", template="plotly_white", markers=True, width=600, height=400, log_y=True, color_discrete_map=COLORS)
fig.update_xaxes(title="<b>Mean Squared Error</b>")
fig.update_traces(marker=dict(size=8))
fig.update_yaxes(title="<b>Hoyer</b>")
fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))

fig.add_annotation(
    text="Vanilla",
    xref="x", 
    yref="y",
    x=0.281, 
    y=-0.23,
    showarrow=False,
    font=dict(size=16),
)

fig.add_annotation(
    text="Mixed",
    xref="x", 
    yref="y",
    x=0.236, 
    y=-0.22,
    showarrow=False,
    font=dict(size=16),
)

fig.add_annotation(
    text="Combined",
    xref="x", 
    yref="y",
    x=0.2997, 
    y=-0.19,
    showarrow=False,
    font=dict(size=16),
)

fig.add_annotation(
    text="Ordered",
    xref="x", 
    yref="y",
    x=0.325, 
    y=-0.20,
    showarrow=False,
    font=dict(size=16),
)

fig.update_layout(font=FONT)
fig
# %%
