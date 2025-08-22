# %%
import torch
import plotly.graph_objects as go

# --- parameters ---
k = 0.95          # choose any 0 < k < 1
n = 64           # grid resolution (increase for smoother surface)
lims = 1.5        # plot range in each axis

# --- grid + field ---
xs = torch.linspace(-lims, lims, n)
ys = torch.linspace(-lims, lims, n)
zs = torch.linspace(-lims, lims, n)
X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")

F = (X**2 + Z**2) / (X**2 + Y**2 + Z**2 + 0.01)  # implicit field

# --- plot isosurface F=0 ---
fig = go.Figure(go.Isosurface(
    x=X.flatten().numpy(),
    y=Y.flatten().numpy(),
    z=Z.flatten().numpy(),
    value=F.flatten().numpy(),
    colorscale='Blues',
    isomin=0.95, isomax=1.0,
    cmin=0.92, cmax=1.02,
    caps=dict(x_show=True, y_show=True, z_show=True)
))

# Second isosurface
G = -(X**2 + Y**2 + Z**2 + X*Y - X*Z) / (X**2 + Y**2 + Z**2 + 0.01)

fig.add_trace(go.Isosurface(
    x=X.flatten().numpy(),
    y=Y.flatten().numpy(),
    z=Z.flatten().numpy(),
    value=G.flatten().numpy(),
    colorscale='Reds',
    isomin=-0.3, isomax=0.0,
    cmin=-0.31, cmax=-0.28,
    caps=dict(x_show=True, y_show=True, z_show=True)
))

fig.update_layout(scene=dict(aspectmode="data"), width=600, height=600, margin=dict(l=0, r=0, b=0, t=0))
fig.update_layout(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False))
fig.update_traces(showscale=False)
fig.update_layout(scene_camera=dict(eye=dict(x=1.2, y=1.2, z=0.8)))
fig.show()
# %%

import torch
import plotly.graph_objects as go

# --- grid ---
n, lims = 400, 2.0
xs = torch.linspace(-lims, lims, n)
ys = torch.linspace(-lims, lims, n)
X, Y = torch.meshgrid(xs, ys, indexing="ij")
den = X**2 + Y**2 + 0.001

# --- fields ---
F1 = (0.01*X**2 + 0.4*X*Y + 0.98*Y**2) / den
F2 = (0.69*X**2 - 0.69*X*Y + 0.69*Y**2) / den
F3 = (0.95*X**2 + 0.5*X*Y) / den

# colors (explicit hex)
C1, C2, C3 = "#1f77b4", "#d62728", "#2ca02c"

fig = go.Figure()

def add_region(F, color):
    # Filled set {F > 1}
    fig.add_trace(go.Heatmap(
        x=xs.numpy(), y=ys.numpy(),
        z=(F > 1).to(dtype=torch.float32).numpy().T,
        showscale=False, opacity=0.28, hoverinfo="skip",
        colorscale=[[0, "rgba(0,0,0,0)"], [1, color]],
        zmin=0, zmax=1
    ))
    # Boundary F = 1 with constant color
    fig.add_trace(go.Contour(
        x=xs.numpy(), y=ys.numpy(), z=F.numpy().T,
        contours=dict(start=1, end=1, size=1, coloring="lines"),
        colorscale=[[0, color], [1, color]],  # constant color
        autocolorscale=False, showscale=False,
        line_width=2, hoverinfo="skip"
    ))

add_region(F1, C1)
add_region(F2, C2)
add_region(F3, C3)

# clean white layout
fig.update_layout(
    width=700, height=700,
    margin=dict(l=0, r=0, t=0, b=0),
    plot_bgcolor="white", paper_bgcolor="white",
    template=None  # avoid template color overrides
)
fig.update_xaxes(visible=False, constrain="domain", scaleanchor="y", scaleratio=1)
fig.update_yaxes(visible=False)

fig.show()
# %%