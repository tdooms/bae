# %%
import numpy as np
import plotly.express as px

# 1) Reproducible random parameters for R, G, B channels
np.random.seed(51)
params = [np.random.uniform(-5, 5, size=3) for _ in range(3)]
print("Parameters (a, b, c) for R, G, B channels:", params)

# 2) Build a 2D grid
n = 500
x = np.linspace(-2, 2, n)
y = np.linspace(-2, 2, n)
X, Y = np.meshgrid(x, y)

# 3) Compute |f| for each (a,b,c) and normalize
channels = []
for a, b, c in params:
    Z = (a * X**2 + b * X * Y + c * Y**2) / (X**2 + Y**2)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)   # mask origin
    Z_pos = np.clip(Z**2, 0, None)               # zero out negatives
    Z_norm = (Z_pos - Z_pos.min()) / (Z_pos.max() - Z_pos.min())
    channels.append(Z_norm)

# 4) Stack into an RGB image array
RGB = np.stack(channels, axis=-1)

# 5) Plot with Plotly
fig = px.imshow(
    RGB,
    origin='lower',
    aspect='equal',
    title='RGB Heatmap of |(a·x² + b·x·y + c·y²)/(x² + y²)|',
    labels={'x': 'x', 'y': 'y'}
)
fig.show()
# %%

import numpy as np
import plotly.graph_objects as go

# Suppress numpy warnings
np.seterr(divide='ignore', invalid='ignore')

# 1) Random parameters for R, G, B channels: 6 per quadratic form
np.random.seed(42)
params = [np.random.uniform(-5, 5, size=6) for _ in range(3)]
print("Parameters (a, b, c, d, e, f) for R, G, B:", params)

# 2) Build spherical grid
phi_res, theta_res = 50, 100
phi = np.linspace(0, np.pi, phi_res)
theta = np.linspace(0, 2*np.pi, theta_res)
phi_grid, theta_grid = np.meshgrid(phi, theta, indexing='ij')
X = np.sin(phi_grid) * np.cos(theta_grid)
Y = np.sin(phi_grid) * np.sin(theta_grid)
Z = np.cos(phi_grid)

# 3) Compute, clamp negatives, normalize
channels = []
for (a, b, c, d, e, f) in params:
    num = a*X**2 + b*Y**2 + c*Z**2 + d*X*Y + e*X*Z + f*Y*Z
    denom = X**2 + Y**2 + Z**2
    denom_safe = np.where(denom == 0, 1.0, denom)
    F = num / denom_safe
    F_pos = np.clip(F, 0, None)           # clamp negatives
    F_norm = (F_pos - F_pos.min()) / (F_pos.max() - F_pos.min())
    channels.append(F_norm)

# 4) Stack channels into RGB and flatten
RGB = np.stack(channels, axis=-1)
RGB_flat = RGB.reshape(-1, 3)

# 5) Create mesh indices
i_faces, j_faces, k_faces = [], [], []
for i in range(phi_res-1):
    for j in range(theta_res-1):
        v0 = i*theta_res + j
        v1 = (i+1)*theta_res + j
        v2 = (i+1)*theta_res + (j+1)
        v3 = i*theta_res + (j+1)
        i_faces += [v0, v0]
        j_faces += [v1, v2]
        k_faces += [v2, v3]

# Convert to CSS colors
vertex_colors = [
    f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
    for r, g, b in RGB_flat
]

# 6) Plot with Plotly
fig = go.Figure(data=[go.Mesh3d(
    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
    i=i_faces, j=j_faces, k=k_faces,
    vertexcolor=vertex_colors,
    flatshading=True, showscale=False
)])
fig.update_layout(
    title="Colored Sphere of 3D Quadratic Forms → RGB",
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='data'
    )
)
fig.show()
# %%