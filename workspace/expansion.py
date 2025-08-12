# %%
import wandb
import pandas as pd
import plotly.express as px

from itertools import islice, chain
from tqdm import tqdm
from scipy.optimize import curve_fit
import numpy as np

api = wandb.Api()
runs = api.runs("tdooms/coder")
# %%

sweep = list(islice(runs, 194-6, 194))

results = []
for run in tqdm(chain(sweep)):
    # print(run.name)
    history = run.history()
    _, _, _, kind, layer, expansion, alpha, _ = run.name.rsplit('-')
    
    mse = history["mse"].min()
    reg = history["reg"].min()
    
    results.append(dict(kind=kind, expansion=int(expansion[1:]), alpha=int(alpha[1:])/100, mse=mse, reg=reg))

df = pd.DataFrame(results)
df['alpha'] = df['alpha'].astype(float)
df = df.sort_values(['kind', 'alpha'], ascending=[False, False])
print(df)
# %%
px.scatter(df, x='mse', y='reg', color='expansion', template="plotly_white")
# px.scatter(df, x='expansion', y='mse', template="plotly_white")
# Add the implicit datapoints

# def power_law(x, a, b, alpha):
#     return a / (x + b)**alpha

# expanded = df.copy()
# # Add datapoint for expansion=0 with mse=1
# zero_expansion = pd.DataFrame({'kind': [''], 'expansion': [0], 'alpha': [0.], 'mse': [1.0], 'reg': [0.0]})
# expanded = pd.concat([df, zero_expansion], ignore_index=True)

# params, _ = curve_fit(power_law, expanded['expansion'], expanded['mse'], p0=[1.0, 1.0, 1.0])
# # Generate points for the fitted curve
# x_fit = np.linspace(0, 24, 1000)
# y_fit = power_law(x_fit, *params)

# # Create the plot with both scatter and fitted line
# fig = px.scatter(expanded, x='expansion', y='mse', template="plotly_white")
# fig.add_scatter(x=x_fit, y=y_fit, mode='lines', name='Power law fit', line=dict(color='red'))
# fig.show()