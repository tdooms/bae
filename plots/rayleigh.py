# %%
import numpy as np
import plotly.graph_objects as go

# 1) Define a symmetric matrix A
A = np.array([[2, 1],
              [1, 3]])

# 2) Build a grid of points on [-1,1]×[-1,1]
n_points = 200
x = np.linspace(-1.5, 1.5, n_points)
y = np.linspace(-1.5, 1.5, n_points)
X, Y = np.meshgrid(x, y)

# 3) Compute the Rayleigh quotient R(x,y) = (x^T A x) / (x^T x)
#    Here x = [X, Y]^T, so:
R = (A[0,0]*X**2 + 2*A[0,1]*X*Y + A[1,1]*Y**2) / (X**2 + Y**2 + 1e-5)
# Mask the singularity at (0,0)
# R[(X == 0) & (Y == 0)] = np.nan

# 4) Create the 3D surface plot
fig = go.Figure(
    data=go.Surface(
        x=X,
        y=Y,
        z=R,
        colorscale='RdBu',        # choose any plotly colorscale
        cmin=np.nanmin(R),
        cmax=np.nanmax(R),
        showscale=True
    )
)

fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    width=600, height=600
)
fig.update_traces(showscale=False)

# 6) Render
fig.show()
# %%