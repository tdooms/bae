# %% 
%load_ext autoreload
%autoreload 2

import torch

def _row_norms(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row ℓ2 norms with floor."""
    return torch.linalg.norm(X, dim=1, keepdim=True).clamp_min(eps)

def normalize_rows(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise ℓ2 normalization."""
    return X / _row_norms(X, eps)

def cross_kernel(B: torch.Tensor, Bp: torch.Tensor) -> torch.Tensor:
    """
    C = B @ Bp.T  (k x k')
    Pairwise inner products ⟨b_i, b'_j⟩ in the lifted space.
    """
    return B @ Bp.T

def cross_cosine(B: torch.Tensor, Bp: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Cosine similarity matrix between features (k x k'):
    cos_ij = ⟨b_i, b'_j⟩ / (||b_i|| ||b'_j||).
    Interprets 'how similar are the two features as functions'.
    """
    NB  = normalize_rows(B, eps)
    NBp = normalize_rows(Bp, eps)
    return NB @ NBp.T  # (k x k')

def coefficients_in_other(
    B: torch.Tensor, Bp: torch.Tensor, damping: float = 1e-6, use_pinv: bool = False
) -> torch.Tensor:
    """
    For each b_i (row of B), compute α_i ∈ R^{k'} such that:
        α_i = argmin_α ||Bp^T α - b_i||_2
    Closed-form (all at once):
        Let Gp = Bp @ Bp.T  (k' x k') and Y = Bp @ B.T (k' x k).
        Solve Gp X = Y  for X (k' x k). Then α = X.T (k x k').
    This equals α_i = (Bp Bp^T)^+ Bp b_i^T (pseudoinverse).
    'damping' adds λI to stabilize near-rank-deficient Gp.
    """
    kprime = Bp.size(0)
    Gp = Bp @ Bp.T
    if damping > 0:
        Gp = Gp + damping * torch.eye(kprime, device=Gp.device, dtype=Gp.dtype)
    Y = Bp @ B.T
    if use_pinv:
        X = torch.linalg.pinv(Gp) @ Y
    else:
        # Prefer solve; Gp is PSD + damping ⇒ well-conditioned SPD
        X = torch.linalg.solve(Gp, Y)
    alpha = X.T  # (k x k')
    return alpha

def reconstruction_fidelity(
    B: torch.Tensor, Bp: torch.Tensor, damping: float = 1e-6, use_pinv: bool = False, eps: float = 1e-12
) -> torch.Tensor:
    """
    ρ_i = ||P' b_i|| / ||b_i|| ∈ [0,1], where P' is the projector onto span(rows(Bp)).
    Efficient Gram-space form (no D x D objects):
        Let Gp = Bp @ Bp.T (k' x k'), Y = Bp @ B.T (k' x k).
        Then (P' b_i) norm^2 = y_i^T Gp^+ y_i  where y_i is column i of Y.
        Vectorized: solve Z = Gp^+ Y, then s = colwise ⟨Y, Z⟩; ρ = sqrt(s)/||b_i||.
    Interprets 'how completely B' represents each feature of B'.
    """
    kprime = Bp.size(0)
    Gp = Bp @ Bp.T
    if damping > 0:
        Gp = Gp + damping * torch.eye(kprime, device=Gp.device, dtype=Gp.dtype)
    Y = Bp @ B.T                         # (k' x k)
    if use_pinv:
        Z = torch.linalg.pinv(Gp) @ Y    # (k' x k)
    else:
        Z = torch.linalg.solve(Gp, Y)
    s = (Y * Z).sum(dim=0)               # length-k, each is y_i^T Gp^+ y_i
    rho = torch.sqrt(s.clamp_min(0.0)) / _row_norms(B, eps).squeeze(1)
    return rho  # (k,)

def top1_greedy_match(cos: torch.Tensor):
    """
    Greedy one-to-one assignment from rows to cols maximizing cosine.
    Returns index tensor 'match_j' of length k and the matched cosines.
    If k != k', unassigned rows map to -1 with cosine 0.
    """
    k, kp = cos.shape
    match_j = torch.full((k,), -1, dtype=torch.long, device=cos.device)
    taken = torch.zeros(kp, dtype=torch.bool, device=cos.device)
    # Flatten and sort pairs by score desc
    vals, flat_idx = torch.sort(cos.reshape(-1), descending=True)
    rows = flat_idx // kp
    cols = flat_idx % kp
    assigned = 0
    for r, c, v in zip(rows.tolist(), cols.tolist(), vals.tolist()):
        if match_j[r] == -1 and not taken[c]:
            match_j[r] = c
            taken[c] = True
            assigned += 1
            if assigned == min(k, kp):
                break
    matched_vals = torch.where(match_j >= 0, cos[torch.arange(k, device=cos.device), match_j], torch.zeros(k, device=cos.device))
    return match_j, matched_vals

def mapping_entropy(alpha: torch.Tensor, base: float = torch.e, eps: float = 1e-12) -> torch.Tensor:
    """
    Entropy of |α_i| per row (as a distribution over B' features).
    Lower ⇒ sparser, nearer one-to-one; higher ⇒ diffuse mixture.
    """
    p = alpha.abs()
    p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)
    ent = -(p * (p.clamp_min(eps)).log()).sum(dim=1)
    if base == 2:
        ent = ent / torch.log(torch.tensor(2.0, device=alpha.device, dtype=alpha.dtype))
    return ent  # (k,)

def factorwise_cosines(
    L: torch.Tensor, R: torch.Tensor, Lp: torch.Tensor, Rp: torch.Tensor, eps: float = 1e-12
):
    """
    Left/right factor cosines and their product.
    Assumes rows are features: L,R ∈ R^{k x d}, Lp,Rp ∈ R^{k' x d}.
    With row-normalized factors, product equals cos(b_i, b'_j) for rank-1 matrices b_i=vec(l_i r_i^T).
    Returns (cos_L, cos_R, cos_B) each (k x k').
    """
    Ln  = normalize_rows(L, eps)
    Rn  = normalize_rows(R, eps)
    Lpn = normalize_rows(Lp, eps)
    Rpn = normalize_rows(Rp, eps)
    cos_L = Ln @ Lpn.T   # (k x k')
    cos_R = Rn @ Rpn.T   # (k x k')
    cos_B = cos_L * cos_R
    return cos_L, cos_R, cos_B
# %%