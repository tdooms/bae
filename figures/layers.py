# %%
import wandb
import pandas as pd
import plotly.express as px

from itertools import islice, chain
from tqdm import tqdm

from figures.constants import COLORS, FONT

api = wandb.Api()
runs = api.runs("tdooms/coder")
# %%

combined_sweep = list(islice(runs, 154-24, 154))
ordered_sweep = list(islice(runs, 243-24, 243))
mixed_sweep = list(islice(runs, 219-24, 219))
vanilla_sweep = list(islice(runs, 272-24, 272))
ordinary_sweep = list(islice(runs, 297-24, 297))

results = []
for run in tqdm(chain(combined_sweep, ordered_sweep, mixed_sweep, vanilla_sweep, ordinary_sweep)):
    history = run.history()
    _, _, _, kind, layer, expansion, alpha, _ = run.name.rsplit('-')
    
    mse = history["mse"].min()
    reg = history["reg"].min()
    
    if kind == "rainbow":
        kind = "combined"
    if kind == "ordinary":
        kind = "baseline"

    results.append(dict(kind=kind, alpha=int(alpha[1:])/100, mse=mse, reg=reg, layer=int(layer[1:])))

df = pd.DataFrame(results)
df['alpha'] = df['alpha'].astype(float)
df = df.sort_values(['kind', 'alpha'], ascending=[False, False])
print(df)
# %%
order = {"kind": ["vanilla", "mixed", "ordered", "combined", "baseline"]}
fig = px.bar(df, y='mse', x="layer", color="kind", template="plotly_white", barmode='group', color_discrete_map=COLORS, category_orders=order, width=1000, height=400)
fig.update_xaxes(title="<b>Layer</b>", tickvals=df['layer'].unique())
fig.update_yaxes(title="<b>Reconstruction error</b>")
fig.update_layout(font=FONT, margin=dict(l=10, r=10, t=10, b=10))
fig.update_layout(showlegend=True, legend=dict(title="", orientation="h", x=0.5, xanchor="center", y=1.02, yanchor="bottom"))
# %%
fig.write_image("C:/Users/thoma/Downloads/layer-recons.svg")