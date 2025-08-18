import torch
import plotly.graph_objects as go
from collections import defaultdict

def visualise_circuit(matrices, current, k=10):
    """This code is written by GPT-5 and extremely bugged, must rewrite."""
    node_idx, labels, layers = {}, [], []
    
    def nid(L, i):
        key = f"L{L}_{int(i)}"
        if key not in node_idx:
            node_idx[key] = len(labels); labels.append(key); layers.append(L)
        return node_idx[key]

    for c in current.tolist(): nid(0, c)

    edges = []  # (u,v,flow,signed)
    for L, W in enumerate(matrices):                         # W: [H_{L+1}, H_L]
        W = W.detach().cpu()
        sub = W[:, current]                                   # restrict to frontier
        t = min(k, sub.numel())
        vals, flat = sub.abs().flatten().topk(t)             # top-k pairs
        rows = flat // sub.size(1)                           # target indices in sub
        cols_in_frontier = flat % sub.size(1)                # source indices in frontier
        cols = current[cols_in_frontier]                     # original source ids

        # --- capacity-1 normalization (incomplete flows) ---
        # Row-normalize (each source i has capacity 1)
        row_sums = torch.zeros(current.numel(), dtype=torch.float16)
        row_sums.scatter_add_(0, cols_in_frontier, vals)
        row_scale = torch.zeros_like(row_sums)
        nz = row_sums > 0
        row_scale[nz] = 1.0 / row_sums[nz]
        flow = vals * row_scale[cols_in_frontier]            # now per-source sum ≤ 1 (==1 for nz rows)

        # Cap columns (each target j has capacity 1)
        # Map each target row to its position in the next frontier
        nxt, inv = torch.unique(rows, sorted=True, return_inverse=True)
        col_sums = torch.zeros(nxt.numel(), dtype=torch.float16)
        col_sums.scatter_add_(0, inv, flow)
        col_scale = torch.ones_like(col_sums)
        big = col_sums > 1
        col_scale[big] = 1.0 / col_sums[big]
        flow = flow * col_scale[inv]                         # now column sums ≤ 1; rows stay ≤ 1
        # ---------------------------------------------------

        # register next-layer nodes
        for r in nxt.tolist(): nid(L+1, int(r))

        # add edges with normalized flow as value; keep signed for hover
        for r, c, f in zip(rows.tolist(), cols.tolist(), flow.tolist()):
            if f == 0: continue
            u, v = nid(L, c), nid(L+1, r)
            edges.append((u, v, float(f), float(W[r, c])))

        current = nxt  # breadth ≤ k

    # simple per-layer coordinates
    byL = defaultdict(list)
    for i, L in enumerate(layers): byL[L].append(i)
    x = [0.0]*len(labels); y = [0.0]*len(labels)
    Lmax = max(byL) if byL else 1
    for L, ids in byL.items():
        ids.sort(key=lambda i: int(labels[i].split('_')[1]))
        n = len(ids)
        for j, i in enumerate(ids):
            x[i] = L / max(1, Lmax)
            y[i] = 0.5 if n == 1 else j / (n-1)

    src = [u for u, v, _, _ in edges]
    tgt = [v for u, v, _, _ in edges]
    val = [f for _, _, f, _ in edges]          # normalized “partial transition”
    sgn = [w for _, _, _, w in edges]          # signed weight for hover

    # (optional) node usage hovers: inflow/outflow fractions
    out_use = [0.0]*len(labels); in_use = [0.0]*len(labels)
    for u,v,f,_ in edges:
        out_use[u] += f
        in_use[v]  += f
    node_custom = [(in_use[i], out_use[i]) for i in range(len(labels))]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=labels, x=x, y=y, pad=8, thickness=12, line=dict(width=0.5),
            customdata=node_custom,
            hovertemplate="in: %{customdata[0]:.3f}<br>out: %{customdata[1]:.3f}<extra></extra>"
        ),
        link=dict(
            source=src, target=tgt, value=val, customdata=sgn,
            hovertemplate="signed weight: %{customdata:.4f}<extra></extra>"
        )
    ))
    return fig.update_layout(height=700)
    