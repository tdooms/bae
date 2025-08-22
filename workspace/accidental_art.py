# %%
import torch
import plotly.graph_objects as go

# --- params ---
k, n, lims = 0.6, 64, 1.5

# --- grid + field F = (x^2+z^2)/(x^2+y^2+z^2+0.01) - k ---
xs = torch.linspace(-lims, lims, n)
ys = torch.linspace(-lims, lims, n)
zs = torch.linspace(-lims, lims, n)
X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
F = (X**2 + Z**2) / (X**2 + Y**2 + Z**2 + 0.01) - k

x = X.flatten().numpy(); y = Y.flatten().numpy(); z = Z.flatten().numpy(); v = F.flatten().numpy()

fig = go.Figure()

# (A) Filled region: all voxels where F >= 0 (semi-transparent)
fig.add_trace(go.Volume(
    x=x, y=y, z=z, value=v,
    isomin=0.0, isomax=float(F.max()),   # show the whole "inside" set
    opacity=0.1, surface_count=18, showscale=False
))

# (B) Sharp boundary: the level set F=0, with caps to "close" at the box
fig.add_trace(go.Isosurface(
    x=x, y=y, z=z, value=v,
    isomin=0.0, isomax=0.0, surface_count=1,
    caps=dict(x_show=True, y_show=True, z_show=True),
    showscale=False, opacity=0.5
))

fig.update_layout(scene=dict(aspectmode="data"), width=800, height=800, margin=dict(l=0, r=0, b=0, t=0))
fig.update_layout(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False))
fig.show()
# %%