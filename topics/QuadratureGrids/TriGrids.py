"""
TriGrids.py — Triangular and wedge grid generation.

Core module: barycentric grids, generalized triangular grids, clipped
triangle (wedge) grids, and Lagrange basis evaluation.
No I/O, no plotting.
"""
import numpy as np


REF_VERTS = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
])


# ── Barycentric grid ──────────────────────────────────────────────────────────

def barycentric_grid(n, verts=REF_VERTS):
    """
    Generate the barycentric lattice of order n on a triangle.

    Returns
    -------
    bary : (N, 3) array — barycentric coordinates (lambda0, lambda1, lambda2)
    xy   : (N, 2) array — physical (x, y) coordinates
    idx  : dict mapping (i,j,k) -> row index
    """
    bary = []
    idx = {}
    r = 0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            bary.append([i / n, j / n, k / n])
            idx[(i, j, k)] = r
            r += 1
    bary = np.array(bary)
    xy = bary @ verts
    return bary, xy, idx


def subdivide_triangles(n, idx=None):
    """
    Build the triangulation of the barycentric grid of order n.
    Returns list of (i0, i1, i2) index triples.
    """
    if idx is None:
        _, _, idx = barycentric_grid(n)
    tris = []
    for i in range(n):
        for j in range(n - i):
            k = n - i - j
            a = idx[(i, j, k)]
            b = idx[(i + 1, j, k - 1)]
            c = idx[(i, j + 1, k - 1)]
            tris.append((a, b, c))
            if k > 1:
                d = idx[(i + 1, j, k - 1)]
                e = idx[(i + 1, j + 1, k - 2)]
                f = idx[(i, j + 1, k - 1)]
                tris.append((d, e, f))
    return tris


# ── Generalized triangular grid (nu ≠ nv) ─────────────────────────────────────

def general_tri_grid(nu, nv, verts=REF_VERTS):
    """
    Generalized triangular grid with nu subdivisions along edge v0-v1
    and nv subdivisions along edge v0-v2.

    Returns
    -------
    nodes : (N, 2) array       — physical (x, y) coordinates
    tris  : (T, 3) int array   — triangle vertex index triples
    idx   : dict (i,j) -> row  — index of regular grid nodes
    """
    eps = 1e-10
    nodes = []
    idx = {}

    def _to_xy(u, v):
        return verts[0] + u * (verts[1] - verts[0]) + v * (verts[2] - verts[0])

    for j in range(nv + 1):
        for i in range(nu + 1):
            if i / nu + j / nv <= 1 + eps:
                idx[(i, j)] = len(nodes)
                nodes.append(_to_xy(i / nu, j / nv))

    diag_idx = {}

    def _diag_node(u):
        v = 1.0 - u
        i_m = round(u * nu)
        j_m = round(v * nv)
        if 0 <= i_m <= nu and 0 <= j_m <= nv:
            if abs(i_m / nu - u) < eps and abs(j_m / nv - v) < eps:
                return idx[(i_m, j_m)]
        key = round(u, 12)
        if key not in diag_idx:
            diag_idx[key] = len(nodes)
            nodes.append(_to_xy(u, v))
        return diag_idx[key]

    tris = []
    for j in range(nv):
        for i in range(nu):
            a = idx.get((i, j))
            if a is None:
                continue
            if abs(i / nu + j / nv - 1.0) < eps:
                continue
            b = idx.get((i + 1, j))
            c = idx.get((i, j + 1))
            d = idx.get((i + 1, j + 1))

            if b is not None and c is not None:
                tris.append((a, b, c))
                if d is not None:
                    tris.append((b, d, c))
            elif b is not None and c is None:
                p = _diag_node(i / nu)
                q = _diag_node((i + 1) / nu)
                tris.append((a, b, q))
                tris.append((a, q, p))
            elif b is None and c is not None:
                p = _diag_node(1 - j / nv)
                q = _diag_node(1 - (j + 1) / nv)
                tris.append((a, p, q))
                tris.append((a, q, c))
            else:
                p = _diag_node(1 - j / nv)
                q = _diag_node(i / nu)
                tris.append((a, p, q))

    nodes = np.array(nodes)
    return nodes, np.array(tris), idx


# ── Wedge grid (clipped triangle) ──────────────────────────────────────────────

def wedge_grid(n, m, v0=0.0, verts=REF_VERTS):
    """
    Build a clipped-triangle (wedge) grid.

    Parameters
    ----------
    n : int   — subdivisions along the diagonal (outer perimeter) edge
    m : int   — subdivisions along the two radial edges (m <= n)
    v0 : float — radial position of inner edge (0 <= v0 < 1)

    Returns
    -------
    nodes : (N, 2) array       — physical (x, y) coordinates
    tris  : (T, 3) int array   — triangle vertex indices
    bary  : (N, 3) array       — barycentric coords of kept nodes
    info  : dict               — metadata: n, m, v0, clip, layers, counts
    """
    assert 0 < m <= n, f"need 0 < m <= n, got m={m} n={n}"
    assert 0.0 <= v0 < 1.0, f"need 0 <= v0 < 1, got v0={v0}"

    clip = n - m
    bary_full, xy_full, idx_full = barycentric_grid(n, verts)
    tris_full = subdivide_triangles(n, idx_full)

    keep = set()
    for (i, j, k), r in idx_full.items():
        if i + j >= clip:
            keep.add(r)

    old_to_new = {}
    new_nodes = []
    new_bary = []
    for r in sorted(keep):
        old_to_new[r] = len(new_nodes)
        new_nodes.append(xy_full[r])
        new_bary.append(bary_full[r])

    new_tris = []
    for (a, b, c) in tris_full:
        if a in keep and b in keep and c in keep:
            new_tris.append((old_to_new[a], old_to_new[b], old_to_new[c]))

    new_nodes = np.array(new_nodes)
    new_bary = np.array(new_bary)

    s_inner_orig = clip / n
    if s_inner_orig > 0:
        s = new_bary[:, 0] + new_bary[:, 1]
        mask = s > 1e-15
        s_new = np.empty_like(s)
        s_new[~mask] = v0
        s_new[mask] = v0 + (s[mask] - s_inner_orig) / (1.0 - s_inner_orig) * (1.0 - v0)
        scale = np.ones_like(s)
        scale[mask] = s_new[mask] / s[mask]
        u = new_bary[:, 0] * scale
        v = new_bary[:, 1] * scale
        new_nodes = np.column_stack([u, v])
        new_bary = np.column_stack([u, v, 1.0 - u - v])

    layers = {}
    for r in range(len(new_bary)):
        s_val = new_bary[r, 0] + new_bary[r, 1]
        if abs(s_val - v0) < 1e-12:
            layer = 0
        elif abs(s_val - 1.0) < 1e-12:
            layer = m
        else:
            layer = round((s_val - v0) / (1.0 - v0) * m)
        layers.setdefault(layer, []).append(r)

    info = {
        'n': n, 'm': m, 'v0': v0, 'clip': clip,
        'n_layers': m + 1,
        'counts': {k: len(v) for k, v in sorted(layers.items())},
        'layers': layers,
    }

    return new_nodes, np.array(new_tris), new_bary, info


# ── Barycentric Lagrange basis ────────────────────────────────────────────────

def _bary_from_xy(points, verts=REF_VERTS):
    """Convert (M,2) physical points to (M,3) barycentric coords."""
    M = np.vstack([verts.T, np.ones(3)]).T
    rhs = np.hstack([points, np.ones((points.shape[0], 1))])
    bary = np.linalg.solve(M.T, rhs.T).T
    return bary


def barycentric_lagrange_basis(eval_pts, n, verts=REF_VERTS):
    """
    Evaluate all N=(n+1)(n+2)/2 Lagrange basis functions on the
    barycentric grid of order n at the given physical points.

    Returns
    -------
    vals : (M, N) array — basis function values at each eval point
    bary_nodes : (N, 3) — barycentric coords of the nodes
    """
    bary_nodes, _, idx = barycentric_grid(n, verts)
    bary_eval = _bary_from_xy(eval_pts, verts)

    M = eval_pts.shape[0]
    N = bary_nodes.shape[0]
    vals = np.ones((M, N))

    for (i, j, k), node_idx in idx.items():
        for p in range(i):
            vals[:, node_idx] *= (n * bary_eval[:, 0] - p) / (i - p)
        for q in range(j):
            vals[:, node_idx] *= (n * bary_eval[:, 1] - q) / (j - q)
        for r in range(k):
            vals[:, node_idx] *= (n * bary_eval[:, 2] - r) / (k - r)

    return vals, bary_nodes
