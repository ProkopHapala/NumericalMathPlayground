"""
TrussSolver.py — Linear algebra solvers for truss systems.

This is the **solver layer** of the LinearElasticity module system.
It provides matrix-free operators, patch construction, iterative solvers,
and direct solvers — but knows nothing about visualization.

Responsibilities
----------------
1. **Graph utilities** — `build_adjacency`, `classify_edges`, `bfs_distances`.
2. **Matrix-free operators** — `compute_edge_data`, `matvec_A`,
   `compute_diagonal_3x3`, `invert_3x3_blocks`, `assemble_dense_A`.
3. **Patch construction** — `make_grid_patch`, `build_patches_grid`,
   `build_patches_grid_nonoverlap`, `build_patches_1d`,
   `build_patches_1d_nonoverlap`, `build_patches` (graph-based).
4. **Patch data setup** — `setup_patch_data` (precompute local matrices,
   LU factorizations, weights, RHS).
5. **Patch relaxation** — `relax_patch_from_global` (inner Jacobi sweeps),
   `relax_patch_direct` (single direct local solve via precomputed LU).
6. **Iterative solvers** — `solve_global_jacobi`, `solve_block_jacobi`,
   `solve_alternating_patches`, `solve_block_jacobi_direct`,
   `solve_alternating_patches_direct`. All support heavy-ball momentum
   (beta can be scalar or per-iteration array; beta[0] is forced to 0
   when scalar, since no previous velocity exists).
7. **Dynamic / frequency-domain solvers** (from VibrationProbing) —
   `dynamic_stiffness`, `cholesky_factor`, `cholesky_solve`,
   `solve_response`, `mechanical_greens_probing`, `expand_displacement`.
8. **Miscellaneous assembly** — `assemble_weighted_stiffness`,
   `classify_edge`, `build_stiffness_hex`.

Role in the system
------------------
- **Truss.py**: geometry, mesh, assembly, bookkeeping.
- **TrussSolver.py** (this file): all linear algebra and iterative solvers.
- **TrussPlotting.py**: reusable plotting functions.
- Scripts: thin wrappers that combine the three modules.

Notes
-----
- All solver functions return `(x, residuals)` where `residuals` has
  length `n_iter + 1` (index 0 = initial residual before any update).
- No matplotlib dependency.
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def build_adjacency(edges, n_nodes):
    adj = [[] for _ in range(n_nodes)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    return adj


def classify_edges(edges, nx, ny):
    """Split edges into straight (horiz/vert) and diagonal on a grid.
    Returns (straight_edges, diag_edges)."""
    straight = []
    diag = []
    for i, j in edges:
        ix_i, iy_i = i % nx, i // nx
        ix_j, iy_j = j % nx, j // nx
        if ix_i == ix_j or iy_i == iy_j:
            straight.append((i, j))
        else:
            diag.append((i, j))
    return straight, diag


def bfs_distances(adj, source, n_nodes):
    dist = np.full(n_nodes, -1, dtype=np.int32)
    dist[source] = 0
    queue = [source]
    head = 0
    while head < len(queue):
        u = queue[head]; head += 1
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


# ---------------------------------------------------------------------------
# Matrix-free operators (axial springs with rest length)
# Each edge (i,j) has rest direction n = (pos_j - pos_i)/L and stiffness
# k_eff = k_spring / L^2.  The stiffness contribution is k_eff * n⊗n (3x3),
# NOT k*I.  This resists only axial deformation and preserves rest length.
# ---------------------------------------------------------------------------

def compute_edge_data(pos, ei, ej, k_arr, dim=3):
    """Precompute per-edge: unit direction n, effective stiffness k_eff = k/L^2."""
    n_edges = len(ei)
    n_dirs = np.zeros((n_edges, dim))
    k_eff = np.zeros(n_edges)
    for e in range(n_edges):
        d = pos[int(ej[e])] - pos[int(ei[e])]
        L = np.linalg.norm(d)
        if L > 1e-12:
            n_dirs[e] = d / L
            k_eff[e] = k_arr[e] / (L * L)
        else:
            n_dirs[e] = np.array([1.0, 0.0, 0.0])[:dim]
            k_eff[e] = k_arr[e]
    return n_dirs, k_eff


def matvec_A(x, ei, ej, k_eff, n_dirs, mass_dt2):
    """A*x, A = M/dt^2 + K (axial springs). x: (N,dim) -> (N,dim)."""
    Ax = mass_dt2[:, None] * x
    if len(ei) > 0:
        diff = x[ei] - x[ej]  # (n_edges, dim)
        dot = np.sum(diff * n_dirs, axis=1)  # (n_edges,) projection along n
        contrib = (k_eff * dot)[:, None] * n_dirs  # (n_edges, dim)
        np.add.at(Ax, ei, contrib)
        np.add.at(Ax, ej, -contrib)
    return Ax


def matvec_A_flat(x_flat, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=3):
    """Flat-DOF wrapper for matvec_A: x_flat (N*dim,) -> (N*dim,)."""
    x = x_flat.reshape(n_nodes, dim)
    Ax = matvec_A(x, ei, ej, k_eff, n_dirs, mass_dt2)
    return Ax.reshape(-1)


def compute_diagonal_3x3(ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=3):
    """D_i = (m_i/dt^2)*I_3 + sum_e k_eff_e * n_e⊗n_e  (3x3 per node)."""
    D = np.zeros((n_nodes, dim, dim))
    for n in range(n_nodes):
        D[n] = mass_dt2[n] * np.eye(dim)
    if len(ei) > 0:
        for e in range(len(ei)):
            nn = np.outer(n_dirs[e], n_dirs[e])  # (dim, dim)
            D[int(ei[e])] += k_eff[e] * nn
            D[int(ej[e])] += k_eff[e] * nn
    return D


def invert_3x3_blocks(D):
    """Invert (N, dim, dim) batch of small SPD matrices. Singular -> zero."""
    N, dim, _ = D.shape
    Dinv = np.empty_like(D)
    for n in range(N):
        try:
            Dinv[n] = np.linalg.inv(D[n])
        except np.linalg.LinAlgError:
            Dinv[n] = np.zeros((dim, dim))
    return Dinv


def assemble_dense_A(ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=3, fixed_nodes=None):
    """Dense A = M/dt^2 + K (axial springs) with Dirichlet rows set to identity."""
    ndof = dim * n_nodes
    A = np.zeros((ndof, ndof))
    for n in range(n_nodes):
        for d in range(dim):
            A[n*dim+d, n*dim+d] = mass_dt2[n]
    for e in range(len(ei)):
        i, j = int(ei[e]), int(ej[e])
        nn = np.outer(n_dirs[e], n_dirs[e]) * k_eff[e]  # (dim, dim)
        for di in range(dim):
            for dj in range(dim):
                A[i*dim+di, i*dim+dj] += nn[di, dj]
                A[j*dim+di, j*dim+dj] += nn[di, dj]
                A[i*dim+di, j*dim+dj] -= nn[di, dj]
                A[j*dim+di, i*dim+dj] -= nn[di, dj]
    if fixed_nodes:
        for f in fixed_nodes:
            for d in range(dim):
                idx = f * dim + d
                A[idx, :] = 0
                A[idx, idx] = 1.0
    return A


# ---------------------------------------------------------------------------
# Patch construction
# ---------------------------------------------------------------------------

def make_grid_patch(nx, ny, ix0, iy0, ix1, iy1, free_mask):
    verts = []
    for iy in range(iy0, iy1):
        for ix in range(ix0, ix1):
            n = iy * nx + ix
            if free_mask[n]:
                verts.append(n)
    if len(verts) == 0:
        return None
    verts = sorted(verts)
    v2l = {v: a for a, v in enumerate(verts)}
    return {
        'vertices': np.array(verts),
        'v2l': v2l,
        'n_vertices': len(verts),
        'n_core': len(verts),
        'n_halo': 0,
        'core_mask_loc': np.ones(len(verts), dtype=bool),
        'grid_ix0': ix0, 'grid_iy0': iy0,
        'grid_ix1': ix1, 'grid_iy1': iy1,
    }


def build_patches_grid(nx, ny, block_size=4, overlap=1, free_mask=None):
    """Structured overlapping rectangular patches on a regular grid."""
    if free_mask is None:
        free_mask = np.ones(nx * ny, dtype=bool)
    step = max(1, block_size - overlap)
    nx_blocks = max(1, (nx - overlap + step - 1) // step)
    ny_blocks = max(1, (ny - overlap + step - 1) // step)
    patches = []
    for by in range(ny_blocks):
        for bx in range(nx_blocks):
            ix0 = min(bx * step, nx - block_size)
            iy0 = min(by * step, ny - block_size)
            pat = make_grid_patch(nx, ny, ix0, iy0, ix0 + block_size, iy0 + block_size, free_mask)
            if pat is not None:
                patches.append(pat)
    return patches


def build_patches_grid_nonoverlap(nx, ny, block_size=4, shift_x=0, shift_y=0, free_mask=None):
    """Non-overlapping clipped rectangular patches. A shifted set covers previous seams."""
    if free_mask is None:
        free_mask = np.ones(nx * ny, dtype=bool)
    x_breaks = sorted(set([0, nx] + [x for x in range(shift_x, nx, block_size) if 0 < x < nx]))
    y_breaks = sorted(set([0, ny] + [y for y in range(shift_y, ny, block_size) if 0 < y < ny]))
    patches = []
    for iy0, iy1 in zip(y_breaks[:-1], y_breaks[1:]):
        for ix0, ix1 in zip(x_breaks[:-1], x_breaks[1:]):
            pat = make_grid_patch(nx, ny, ix0, iy0, ix1, iy1, free_mask)
            if pat is not None:
                patches.append(pat)
    return patches


def build_patches_1d(nx, ny, block_size, overlap, free_mask):
    """1D overlapping patches spanning full height (ny), segmented along x."""
    if free_mask is None:
        free_mask = np.ones(nx * ny, dtype=bool)
    step = max(1, block_size - overlap)
    nx_blocks = max(1, (nx - overlap + step - 1) // step)
    patches = []
    for bx in range(nx_blocks):
        ix0 = min(bx * step, max(0, nx - block_size))
        ix1 = min(ix0 + block_size, nx)
        pat = make_grid_patch(nx, ny, ix0, 0, ix1, ny, free_mask)
        if pat is not None:
            patches.append(pat)
    return patches


def build_patches_1d_nonoverlap(nx, ny, block_size, shift_x, free_mask):
    """1D non-overlapping patches spanning full height, shifted along x."""
    if free_mask is None:
        free_mask = np.ones(nx * ny, dtype=bool)
    x_breaks = sorted(set([0, nx] + [x for x in range(shift_x, nx, block_size) if 0 < x < nx]))
    patches = []
    for ix0, ix1 in zip(x_breaks[:-1], x_breaks[1:]):
        pat = make_grid_patch(nx, ny, ix0, 0, ix1, ny, free_mask)
        if pat is not None:
            patches.append(pat)
    return patches


def build_patches(adj, n_nodes, free_mask, target_core=16, halo_radius=1):
    """Overlapping patches via greedy farthest-point seeds + BFS core + halo."""
    free_nodes = np.where(free_mask)[0]
    if len(free_nodes) == 0:
        return []
    seeds = [int(free_nodes[0])]
    min_dist = bfs_distances(adj, seeds[0], n_nodes).astype(float)
    min_dist[~free_mask] = 0
    while len(seeds) < max(1, len(free_nodes) // target_core):
        nxt = int(np.argmax(min_dist))
        if not free_mask[nxt] or min_dist[nxt] <= 0:
            break
        seeds.append(nxt)
        d = bfs_distances(adj, nxt, n_nodes).astype(float)
        min_dist = np.minimum(min_dist, d)
        min_dist[~free_mask] = 0
    n_patches = len(seeds)
    all_dist = np.zeros((n_nodes, n_patches), dtype=np.int32)
    for p, s in enumerate(seeds):
        all_dist[:, p] = bfs_distances(adj, s, n_nodes)
    core_assign = np.argmin(all_dist, axis=1)
    patches = []
    for p in range(n_patches):
        core = np.where((core_assign == p) & free_mask)[0]
        if len(core) == 0:
            continue
        core_set = set(core.tolist())
        halo = set()
        frontier = set(core.tolist())
        for _ in range(halo_radius):
            new_front = set()
            for vv in frontier:
                for uu in adj[vv]:
                    if uu not in core_set and uu not in halo:
                        halo.add(uu)
                        new_front.add(uu)
            frontier = new_front
        verts = sorted(core_set | halo)
        v2l = {vv: a for a, vv in enumerate(verts)}
        core_mask_loc = np.array([vv in core_set for vv in verts], dtype=bool)
        patches.append({
            'vertices': np.array(verts),
            'v2l': v2l,
            'n_vertices': len(verts),
            'n_core': len(core_set),
            'n_halo': len(halo),
            'core_mask_loc': core_mask_loc,
        })
    return patches


# ---------------------------------------------------------------------------
# Patch data setup (precompute local matrices, LU, weights, RHS)
# ---------------------------------------------------------------------------

def setup_patch_data(patches, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, free_mask, dim=3,
                     pos=None, gravity=None, local_dirichlet=False):
    """Precompute per-patch data. local_dirichlet=True uses full boundary ghost edges."""
    n_edges = len(k_eff)
    # Global 3x3 diagonal
    D_global_3x3 = compute_diagonal_3x3(ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim)
    # Global "stiffness norm" per node (trace of K_ii block) for weight derivation
    D_spr_trace = np.zeros(n_nodes)
    if n_edges > 0:
        for e in range(n_edges):
            nn = np.outer(n_dirs[e], n_dirs[e])
            tr = k_eff[e] * np.trace(nn)  # = k_eff[e]
            D_spr_trace[int(ei[e])] += tr
            D_spr_trace[int(ej[e])] += tr
    D_global_trace = mass_dt2 + D_spr_trace  # trace of D_i

    edge_patches = [[] for _ in range(n_edges)]
    for e in range(n_edges):
        i, j = int(ei[e]), int(ej[e])
        i_free = free_mask[i]
        j_free = free_mask[j]
        if not i_free and not j_free:
            continue
        for pi, pat in enumerate(patches):
            if local_dirichlet:
                if (i_free and i in pat['v2l']) or (j_free and j in pat['v2l']):
                    edge_patches[e].append(pi)
            else:
                if i_free and i in pat['v2l'] and (not j_free or j in pat['v2l']):
                    edge_patches[e].append(pi)
                elif j_free and j in pat['v2l'] and (not i_free or i in pat['v2l']):
                    edge_patches[e].append(pi)

    for pi, pat in enumerate(patches):
        n_loc = pat['n_vertices']
        verts = pat['vertices']
        # Collect ghost nodes connected by local Dirichlet/boundary edges
        ghost_nodes = set()
        for e in range(n_edges):
            if pi in edge_patches[e]:
                i, j = int(ei[e]), int(ej[e])
                if i not in pat['v2l']:
                    ghost_nodes.add(i)
                if j not in pat['v2l']:
                    ghost_nodes.add(j)
        ghost_nodes = sorted(ghost_nodes)
        # Extended vertex list: patch vertices + ghost nodes
        ext_verts = list(verts) + ghost_nodes
        ext_v2l = {v: a for a, v in enumerate(ext_verts)}
        n_ext = len(ext_verts)
        # Local stiffness trace (only for actual patch vertices, not ghosts)
        D_sp_loc_trace = np.zeros(n_ext)
        lei, lej, lk_eff, ln_dirs = [], [], [], []
        for e in range(n_edges):
            if pi in edge_patches[e]:
                alpha = 1.0 if local_dirichlet else 1.0 / len(edge_patches[e])
                li = ext_v2l[int(ei[e])]
                lj = ext_v2l[int(ej[e])]
                lei.append(li); lej.append(lj)
                lk_eff.append(alpha * k_eff[e])
                ln_dirs.append(n_dirs[e])
                D_sp_loc_trace[li] += alpha * k_eff[e]
                D_sp_loc_trace[lj] += alpha * k_eff[e]
        # mass split: alpha_m = D_spr_loc / D_spr_global (only for patch vertices)
        alpha_m = np.ones(n_ext)
        for a in range(n_ext):
            vv = ext_verts[a]
            if a >= n_loc:
                alpha_m[a] = 0.0
            elif local_dirichlet:
                alpha_m[a] = 1.0
            elif not free_mask[vv]:
                alpha_m[a] = 0.0
            elif D_spr_trace[vv] > 1e-30:
                alpha_m[a] = D_sp_loc_trace[a] / D_spr_trace[vv]
            else:
                cnt = sum(1 for p in patches if vv in p['v2l'])
                alpha_m[a] = 1.0 / max(1, cnt)
        # Local 3x3 diagonal (for all ext vertices including ghosts)
        D_loc_3x3 = np.zeros((n_ext, dim, dim))
        for a in range(n_ext):
            vv = ext_verts[a]
            if free_mask[vv]:
                D_loc_3x3[a] = alpha_m[a] * mass_dt2[vv] * np.eye(dim)
        for idx in range(len(lei)):
            nn = np.outer(ln_dirs[idx], ln_dirs[idx])
            D_loc_3x3[lei[idx]] += lk_eff[idx] * nn
            D_loc_3x3[lej[idx]] += lk_eff[idx] * nn
        # Only patch vertices (not ghosts) get W and are written back
        D_loc_trace_arr = np.array([np.trace(D_loc_3x3[a]) for a in range(n_ext)])
        W = np.zeros(n_ext)
        for a in range(n_loc):  # only patch vertices
            vv = verts[a]
            if D_global_trace[vv] > 1e-30:
                W[a] = D_loc_trace_arr[a] / D_global_trace[vv]
        # Precompute local Dinv (3x3) — only for free vertices
        D_loc_inv = invert_3x3_blocks(D_loc_3x3)
        pat['ext_verts'] = np.array(ext_verts)
        pat['n_ext'] = n_ext
        pat['lei'] = np.array(lei, dtype=np.int32)
        pat['lej'] = np.array(lej, dtype=np.int32)
        pat['lk_eff'] = np.array(lk_eff)
        pat['ln_dirs'] = np.array(ln_dirs)
        pat['D_loc_3x3'] = D_loc_3x3
        pat['D_loc_inv'] = D_loc_inv
        pat['W'] = W
        pat['alpha_m'] = alpha_m
        pat['local_mass_dt2'] = alpha_m * mass_dt2[np.array(ext_verts)]
        pat['local_free'] = np.array([(a < n_loc) and free_mask[v] for a, v in enumerate(ext_verts)], dtype=bool)
        # Precompute local RHS: b_loc = A_loc * y_rest + alpha_m * f_gravity
        if pos is not None:
            y_loc = pos[np.array(ext_verts)]
            A_loc_y = pat['local_mass_dt2'][:, None] * y_loc
            if len(pat['lei']) > 0:
                diff = y_loc[pat['lei']] - y_loc[pat['lej']]
                dot = np.sum(diff * pat['ln_dirs'], axis=1)
                contrib = (pat['lk_eff'] * dot)[:, None] * pat['ln_dirs']
                np.add.at(A_loc_y, pat['lei'], contrib)
                np.add.at(A_loc_y, pat['lej'], -contrib)
            f_loc = np.zeros((n_ext, dim))
            if gravity is not None:
                f_loc = alpha_m[:, None] * gravity[np.array(ext_verts)]
            pat['b_loc'] = A_loc_y + f_loc
            # Assemble dense local matrix A_loc and precompute LU factorization
            A_loc_dense = np.zeros((n_ext * dim, n_ext * dim))
            for a in range(n_ext):
                for d in range(dim):
                    A_loc_dense[a*dim+d, a*dim+d] = pat['local_mass_dt2'][a]
            for idx in range(len(lei)):
                nn = np.outer(ln_dirs[idx], ln_dirs[idx]) * lk_eff[idx]
                li_idx = lei[idx] * dim
                lj_idx = lej[idx] * dim
                for di in range(dim):
                    for dj in range(dim):
                        A_loc_dense[li_idx+di, li_idx+dj] += nn[di, dj]
                        A_loc_dense[lj_idx+di, lj_idx+dj] += nn[di, dj]
                        A_loc_dense[li_idx+di, lj_idx+dj] -= nn[di, dj]
                        A_loc_dense[lj_idx+di, li_idx+dj] -= nn[di, dj]
            # Dirichlet: fix ghost nodes (non-free) to identity rows
            for a in range(n_ext):
                if not pat['local_free'][a]:
                    for d in range(dim):
                        idx_dof = a * dim + d
                        A_loc_dense[idx_dof, :] = 0
                        A_loc_dense[idx_dof, idx_dof] = 1.0
            pat['A_loc_lu'] = lu_factor(A_loc_dense)
    # vertex -> incident patches
    vp = [[] for _ in range(n_nodes)]
    for pi, pat in enumerate(patches):
        for a in range(pat['n_vertices']):
            vp[pat['vertices'][a]].append((pi, a))
    D_global_inv = invert_3x3_blocks(D_global_3x3)
    return vp, D_global_3x3, D_global_inv


# ---------------------------------------------------------------------------
# Patch relaxation
# ---------------------------------------------------------------------------

def relax_patch_from_global(pat, x, omega_inner, n_inner):
    ext_verts = pat['ext_verts']
    x_loc = x[ext_verts].copy()
    b_loc = pat['b_loc']
    lei, lej = pat['lei'], pat['lej']
    lk_eff, ln_dirs = pat['lk_eff'], pat['ln_dirs']
    lmd = pat['local_mass_dt2']
    D_loc_inv = pat['D_loc_inv']
    lf = pat['local_free']
    for inner in range(n_inner):
        Ax_loc = lmd[:, None] * x_loc
        if len(lei) > 0:
            diff = x_loc[lei] - x_loc[lej]
            dot = np.sum(diff * ln_dirs, axis=1)
            contrib = (lk_eff * dot)[:, None] * ln_dirs
            np.add.at(Ax_loc, lei, contrib)
            np.add.at(Ax_loc, lej, -contrib)
        r_loc = b_loc - Ax_loc
        dx_loc = np.zeros_like(x_loc)
        for a in np.where(lf)[0]:
            dx_loc[a] = D_loc_inv[a] @ r_loc[a]
        x_loc = x_loc + omega_inner * dx_loc
        x_loc[~lf] = x[ext_verts][~lf]
    return x_loc


def relax_patch_direct(pat, x):
    """Direct local solve: x_loc = A_loc^{-1} * b_loc using precomputed LU."""
    ext_verts = pat['ext_verts']
    b_loc = pat['b_loc']
    n_ext = pat['n_ext']
    dim = 3
    # Build RHS: b_loc for free DOFs, current x for ghost/fixed DOFs
    rhs = b_loc.copy()
    x_current = x[ext_verts]
    for a in range(n_ext):
        if not pat['local_free'][a]:
            for d in range(dim):
                rhs[a, d] = x_current[a, d]
    rhs_flat = rhs.reshape(-1)
    x_loc_flat = lu_solve(pat['A_loc_lu'], rhs_flat)
    return x_loc_flat.reshape(n_ext, dim)


# ---------------------------------------------------------------------------
# Iterative solvers
# ---------------------------------------------------------------------------

def solve_global_jacobi(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=3,
                        free_mask=None, omega=0.8, beta=0.5, n_iter=100):
    """Global Jacobi with heavy-ball momentum. beta can be scalar or array of length n_iter."""
    D = compute_diagonal_3x3(ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim)
    Dinv = invert_3x3_blocks(D)
    if free_mask is None:
        free_mask = np.ones(n_nodes, dtype=bool)
    if np.isscalar(beta):
        betas = np.full(n_iter, beta)
        betas[0] = 0.0
    else:
        betas = np.asarray(beta)
    x = x0.copy()
    v = np.zeros_like(x)
    residuals = []
    b_norm = np.linalg.norm(b[free_mask]) + 1e-30
    r0 = b - matvec_A(x, ei, ej, k_eff, n_dirs, mass_dt2)
    residuals.append(np.linalg.norm(r0[free_mask]) / b_norm)
    for it in range(n_iter):
        r = b - matvec_A(x, ei, ej, k_eff, n_dirs, mass_dt2)
        dx = np.zeros_like(x)
        for n in np.where(free_mask)[0]:
            dx[n] = Dinv[n] @ r[n]
        v = betas[it] * v + omega * dx
        x = x + v
        x[~free_mask] = x0[~free_mask]
        r = b - matvec_A(x, ei, ej, k_eff, n_dirs, mass_dt2)
        residuals.append(np.linalg.norm(r[free_mask]) / b_norm)
    return x, residuals


def solve_block_jacobi(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=3,
                       patches=None, vp=None, free_mask=None, D_global_inv=None,
                       omega=0.8, beta=0.5, omega_inner=1.0,
                       n_outer=20, n_inner=5, use_scalar_weight=False):
    """Block Jacobi with overlapping patches and heavy-ball momentum.
    beta can be scalar or array of length n_outer."""
    if free_mask is None:
        free_mask = np.ones(n_nodes, dtype=bool)
    if D_global_inv is None:
        D_global = compute_diagonal_3x3(ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim)
        D_global_inv = invert_3x3_blocks(D_global)
    if np.isscalar(beta):
        betas = np.full(n_outer, beta)
        betas[0] = 0.0
    else:
        betas = np.asarray(beta)
    x = x0.copy()
    v = np.zeros_like(x)
    residuals = []
    b_norm = np.linalg.norm(b[free_mask]) + 1e-30
    r0 = b - matvec_A(x, ei, ej, k_eff, n_dirs, mass_dt2)
    residuals.append(np.linalg.norm(r0[free_mask]) / b_norm)
    for outer in range(n_outer):
        # Phase 1: patch relaxation (local memory simulation)
        preds = []
        for pat in patches:
            preds.append(relax_patch_from_global(pat, x, omega_inner, n_inner))
        # Phase 2: assembly (only write to patch vertices, not ghosts)
        x_pre = x.copy()
        dx_all = np.zeros_like(x)
        if use_scalar_weight:
            for pi, pat in enumerate(patches):
                verts = pat['vertices']
                n_loc = pat['n_vertices']
                corr = pat['W'][:n_loc, None] * (preds[pi][:n_loc] - x_pre[verts])
                np.add.at(dx_all, verts, corr)
        else:
            for pi, pat in enumerate(patches):
                verts = pat['vertices']
                n_loc = pat['n_vertices']
                Dp = pat['D_loc_3x3']
                dxi = preds[pi][:n_loc] - x_pre[verts]
                for a, vv in enumerate(verts):
                    dx_all[vv] += D_global_inv[vv] @ (Dp[a] @ dxi[a])
        v[free_mask] = betas[outer] * v[free_mask] + omega * dx_all[free_mask]
        x = x + v
        x[~free_mask] = x0[~free_mask]
        # residual
        r = b - matvec_A(x, ei, ej, k_eff, n_dirs, mass_dt2)
        residuals.append(np.linalg.norm(r[free_mask]) / b_norm)
    return x, residuals


def solve_alternating_patches(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=3,
                              patch_sets=None, free_mask=None,
                              omega=0.8, beta=0.0, omega_inner=1.0,
                              n_outer=20, n_inner=5):
    """Alternating shifted non-overlapping patches. beta can be scalar or array."""
    if free_mask is None:
        free_mask = np.ones(n_nodes, dtype=bool)
    if np.isscalar(beta):
        betas = np.full(n_outer, beta)
        betas[0] = 0.0
    else:
        betas = np.asarray(beta)
    x = x0.copy()
    v = np.zeros_like(x)
    residuals = []
    b_norm = np.linalg.norm(b[free_mask]) + 1e-30
    r0 = b - matvec_A(x, ei, ej, k_eff, n_dirs, mass_dt2)
    residuals.append(np.linalg.norm(r0[free_mask]) / b_norm)
    for outer in range(n_outer):
        patches = patch_sets[outer % len(patch_sets)]
        x_candidate = x.copy()
        for pat in patches:
            x_loc = relax_patch_from_global(pat, x, omega_inner, n_inner)
            n_loc = pat['n_vertices']
            verts = pat['vertices']
            x_candidate[verts] = x_loc[:n_loc]
        dx = x_candidate - x
        v[free_mask] = betas[outer] * v[free_mask] + omega * dx[free_mask]
        x = x + v
        x[~free_mask] = x0[~free_mask]
        r = b - matvec_A(x, ei, ej, k_eff, n_dirs, mass_dt2)
        residuals.append(np.linalg.norm(r[free_mask]) / b_norm)
    return x, residuals


def solve_block_jacobi_direct(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=3,
                              patches=None, vp=None, free_mask=None, D_global_inv=None,
                              omega=0.8, beta=0.0, use_scalar_weight=False,
                              n_outer=20):
    """Block Jacobi with direct local solves (precomputed LU) and heavy-ball.
    Each outer iteration does 1 direct solve per patch (no inner loop).
    beta can be scalar or array of length n_outer."""
    if free_mask is None:
        free_mask = np.ones(n_nodes, dtype=bool)
    if D_global_inv is None:
        D_global = compute_diagonal_3x3(ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim)
        D_global_inv = invert_3x3_blocks(D_global)
    if np.isscalar(beta):
        betas = np.full(n_outer, beta)
        betas[0] = 0.0
    else:
        betas = np.asarray(beta)
    x = x0.copy()
    v = np.zeros_like(x)
    residuals = []
    b_norm = np.linalg.norm(b[free_mask]) + 1e-30
    r0 = b - matvec_A(x, ei, ej, k_eff, n_dirs, mass_dt2)
    residuals.append(np.linalg.norm(r0[free_mask]) / b_norm)
    for outer in range(n_outer):
        preds = []
        for pat in patches:
            preds.append(relax_patch_direct(pat, x))
        x_pre = x.copy()
        dx_all = np.zeros_like(x)
        if use_scalar_weight:
            for pi, pat in enumerate(patches):
                verts = pat['vertices']
                n_loc = pat['n_vertices']
                corr = pat['W'][:n_loc, None] * (preds[pi][:n_loc] - x_pre[verts])
                np.add.at(dx_all, verts, corr)
        else:
            for pi, pat in enumerate(patches):
                verts = pat['vertices']
                n_loc = pat['n_vertices']
                Dp = pat['D_loc_3x3']
                dxi = preds[pi][:n_loc] - x_pre[verts]
                for a, vv in enumerate(verts):
                    dx_all[vv] += D_global_inv[vv] @ (Dp[a] @ dxi[a])
        v[free_mask] = betas[outer] * v[free_mask] + omega * dx_all[free_mask]
        x = x + v
        x[~free_mask] = x0[~free_mask]
        r = b - matvec_A(x, ei, ej, k_eff, n_dirs, mass_dt2)
        residuals.append(np.linalg.norm(r[free_mask]) / b_norm)
    return x, residuals


def solve_alternating_patches_direct(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=3,
                                     patch_sets=None, free_mask=None,
                                     omega=0.8, beta=0.0, n_outer=20):
    """Alternating shifted patches with direct local solves (precomputed LU).
    Each outer iteration does 1 direct solve per patch (no inner loop).
    beta can be scalar or array of length n_outer."""
    if free_mask is None:
        free_mask = np.ones(n_nodes, dtype=bool)
    if np.isscalar(beta):
        betas = np.full(n_outer, beta)
        betas[0] = 0.0
    else:
        betas = np.asarray(beta)
    x = x0.copy()
    v = np.zeros_like(x)
    residuals = []
    b_norm = np.linalg.norm(b[free_mask]) + 1e-30
    r0 = b - matvec_A(x, ei, ej, k_eff, n_dirs, mass_dt2)
    residuals.append(np.linalg.norm(r0[free_mask]) / b_norm)
    for outer in range(n_outer):
        patches = patch_sets[outer % len(patch_sets)]
        x_candidate = x.copy()
        for pat in patches:
            x_loc = relax_patch_direct(pat, x)
            n_loc = pat['n_vertices']
            verts = pat['vertices']
            x_candidate[verts] = x_loc[:n_loc]
        dx = x_candidate - x
        v[free_mask] = betas[outer] * v[free_mask] + omega * dx[free_mask]
        x = x + v
        x[~free_mask] = x0[~free_mask]
        r = b - matvec_A(x, ei, ej, k_eff, n_dirs, mass_dt2)
        residuals.append(np.linalg.norm(r[free_mask]) / b_norm)
    return x, residuals


# ---------------------------------------------------------------------------
# Dynamic / frequency-domain solvers (from VibrationProbing)
# ---------------------------------------------------------------------------

def dynamic_stiffness(K, M, omega, eta=1e-3, stabilize=1e-6):
    """
    Dynamic stiffness: A = K - (omega + i*eta)^2 M.
    eta>0 pushes poles slightly into upper half-plane so we see modes sharply.
    stabilize>0 adds small diagonal real shift to keep system invertible.
    """
    z = omega + 1j * eta
    A = K - (z * z) * M
    if stabilize > 0:
        A = A + stabilize * np.eye(K.shape[0])
    return A


def cholesky_factor(A):
    """Dense Cholesky; may fail if A not HPD -> raise loudly."""
    L = np.linalg.cholesky(A)
    return L


def cholesky_solve(L, B):
    """Solve A X = B given A = L L^H."""
    Y = np.linalg.solve(L, B)
    X = np.linalg.solve(L.T.conj(), Y)
    return X


def solve_response(K, M, omega, eta, charges, direction_vec, dim=3, stabilize=1e-6):
    """Solve for displacement under dipole force at a single omega."""
    ndof = K.shape[0]
    n_nodes = ndof // dim
    A = dynamic_stiffness(K, M, omega, eta=eta, stabilize=stabilize)
    rhs = np.zeros((ndof, 1), dtype=np.complex128)
    for n in range(n_nodes):
        rhs[n * dim : n * dim + dim, 0] = charges[n] * direction_vec[:dim]
    U = np.linalg.solve(A, rhs)
    return U[:, 0].reshape(n_nodes, dim)


def mechanical_greens_probing(K, M, omegas, eta=1e-3, direction_vec=None, charges=None, dim=3, stabilize=1e-6):
    """
    Dipole-driven probing of mechanical Green's function.
    - K, M: reduced (Dirichlet) matrices
    - omegas: array of target frequencies (real)
    - eta: small damping to push poles above real axis
    - direction_vec: 3-vector direction of the homogeneous field
    - charges: per-node charges for dipole coupling (len = n_nodes)
    Returns dict with spectra and dipole couplings.
    """
    ndof = K.shape[0]
    if ndof % dim != 0:
        raise ValueError("DOF count not divisible by dim")
    n_nodes = ndof // dim
    if direction_vec is None:
        direction_vec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    direction_vec = np.asarray(direction_vec, dtype=np.float64)
    if charges is None:
        charges = np.ones(n_nodes)
    charges = np.asarray(charges)
    if charges.shape[0] != n_nodes:
        raise ValueError("charges size mismatch")

    spectrum_energy = np.zeros(len(omegas))
    spectrum_dipole = np.zeros((len(omegas), dim), dtype=np.complex128)

    for io, omega in enumerate(omegas):
        A = dynamic_stiffness(K, M, omega, eta=eta, stabilize=stabilize)
        rhs = np.zeros((ndof, 1), dtype=np.complex128)
        for n in range(n_nodes):
            rhs[n * dim : n * dim + dim, 0] = charges[n] * direction_vec[:dim]
        U = np.linalg.solve(A, rhs)
        spectrum_energy[io] = np.sum(np.abs(U) ** 2) / n_nodes
        disp_nodes = U[:, 0].reshape(n_nodes, dim)
        dip = (charges[:, None] * disp_nodes).sum(axis=0)
        spectrum_dipole[io] = dip
    return {
        "omega": np.asarray(omegas),
        "energy": spectrum_energy,
        "dipole": spectrum_dipole,
        "n_probes": len(omegas),
    }


def expand_displacement(disp_reduced, mask, dim=3):
    """Map reduced displacement (after Dirichlet) back to full node array."""
    ndof_full = mask.size
    n_nodes_full = ndof_full // dim
    disp_full = np.zeros((n_nodes_full, dim), dtype=disp_reduced.dtype)
    node_mask = mask.reshape(n_nodes_full, dim)[:, 0]
    disp_full[node_mask, :] = disp_reduced
    return disp_full


# ---------------------------------------------------------------------------
# Miscellaneous assembly utilities (from VibrationProbing)
# ---------------------------------------------------------------------------

def assemble_weighted_stiffness(pos, edges, k_edges, dim=3):
    """Assemble stiffness with per-edge spring constants k_edges (len==edges)."""
    n_nodes = pos.shape[0]
    ndof = dim * n_nodes
    K = np.zeros((ndof, ndof), dtype=np.float64)
    for (i, j), k_spring in zip(edges, k_edges):
        d = pos[j] - pos[i]
        L = np.linalg.norm(d)
        if L <= 1e-12:
            continue
        u = d / L
        k_fac = k_spring / (L * L)
        ia = i * dim
        ja = j * dim
        outer = k_fac * np.outer(u, u)
        K[ia:ia+dim, ia:ia+dim] += outer
        K[ja:ja+dim, ja:ja+dim] += outer
        K[ia:ia+dim, ja:ja+dim] -= outer
        K[ja:ja+dim, ia:ia+dim] -= outer
    return K


def classify_edge(pos, i, j, tol=1e-6):
    d = pos[j] - pos[i]
    dx, dy = abs(d[0]), abs(d[1])
    if dy < tol:
        return "x"
    if dx < tol:
        return "y"
    return "diag"


# ---------------------------------------------------------------------------
# Hex grid stiffness (from elasticity_benchmark)
# ---------------------------------------------------------------------------

def build_stiffness_hex(nnode, neighs, k0, k_sigma):
    """Build stiffness value matrix same shape as neighs.
    Base stiffness k0 with relative Gaussian perturbation k_sigma.
    Invalid slots (neigh == -1) remain 0.0.
    """
    k = np.zeros_like(neighs, dtype=np.float64)
    mask = neighs >= 0
    noise = np.random.normal(loc=0.0, scale=k_sigma, size=neighs.shape)
    vals = k0 * (1.0 + noise)
    k[mask] = vals[mask]
    k = np.clip(k, a_min=1e-6, a_max=None)
    k[~mask] = 0.0
    return k
