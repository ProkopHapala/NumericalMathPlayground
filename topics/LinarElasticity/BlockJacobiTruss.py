"""
BlockJacobiTruss.py — Truss solver: global Jacobi vs block Jacobi with
overlapping patches, stiffness-weighted averaging, and heavy-ball momentum.

See BlockJacobiGPU.md for full derivation. Key points:
  A = M/dt^2 + K,  D_i = m_i/dt^2 + sum_{j in N(i)} k_ij
  Global Jacobi+HB: v = beta*v + omega*(r/D), x += v
  Block Jacobi: overlapping patches do nInner local Jacobi steps,
    then corrections are blended by W_i^(p) = D_i^(p) / D_i.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from Truss import build_triangular_grid, grid_edges, boundary_nodes


# ---- graph utils ----

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


# ---- matrix-free operators (axial springs with rest length) ----
# Each edge (i,j) has rest direction n = (pos_j - pos_i)/L and stiffness
# k_eff = k_spring / L^2.  The stiffness contribution is k_eff * n⊗n (3x3),
# NOT k*I.  This resists only axial deformation and preserves rest length.

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


# ---- global Jacobi + heavy-ball ----

def solve_global_jacobi(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=3,
                        free_mask=None, omega=0.8, beta=0.5, n_iter=100):
    """Global Jacobi with heavy-ball momentum. beta can be scalar or array of length n_iter."""
    D = compute_diagonal_3x3(ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim)
    Dinv = invert_3x3_blocks(D)
    if free_mask is None:
        free_mask = np.ones(n_nodes, dtype=bool)
    if np.isscalar(beta):
        betas = np.full(n_iter, beta)
    else:
        betas = np.asarray(beta)
    x = x0.copy()
    v = np.zeros_like(x)
    residuals = []
    b_norm = np.linalg.norm(b[free_mask]) + 1e-30
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


# ---- patch construction ----

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
    # vertex -> incident patches
    vp = [[] for _ in range(n_nodes)]
    for pi, pat in enumerate(patches):
        for a in range(pat['n_vertices']):
            vp[pat['vertices'][a]].append((pi, a))
    D_global_inv = invert_3x3_blocks(D_global_3x3)
    return vp, D_global_3x3, D_global_inv


# ---- block Jacobi + heavy-ball ----

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
    else:
        betas = np.asarray(beta)
    x = x0.copy()
    v = np.zeros_like(x)
    residuals = []
    b_norm = np.linalg.norm(b[free_mask]) + 1e-30
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
    else:
        betas = np.asarray(beta)
    x = x0.copy()
    v = np.zeros_like(x)
    residuals = []
    b_norm = np.linalg.norm(b[free_mask]) + 1e-30
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


# ---- dense assembly for direct reference solve ----

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


# ---- benchmark / visualization ----

def main():
    parser = argparse.ArgumentParser(description="Block Jacobi truss solver benchmark.")
    parser.add_argument('--no-show', action='store_true', help='Do not call plt.show()')
    parser.add_argument('--strain', type=float, default=0.3,
                        help='Initial strain factor: x0 = pos * (1 + strain)')
    parser.add_argument('--jitter', type=float, default=0.1,
                        help='Random shift amplitude added to initial guess')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for jitter')
    parser.add_argument('--block-size', type=int, default=4, help='Grid patch block size')
    parser.add_argument('--overlap', type=int, default=1, help='Grid patch overlap')
    parser.add_argument('--n-outer', type=int, default=15, help='Outer iterations / kernel launches')
    args = parser.parse_args()

    nx, ny = 7, 7
    a = 1.0
    k_spring = 20000.0
    mass_val = 1.0
    dt = 0.02
    dim = 3

    pos = build_triangular_grid(nx, ny, a=a, jitter=0.0)
    edges = grid_edges(nx, ny, include_diag=True)
    n_nodes = nx * ny
    k_edges = [k_spring] * len(edges)
    masses = np.full(n_nodes, mass_val)
    mass_dt2 = masses / dt**2

    fixed = set(boundary_nodes(nx, ny, which="bottom"))
    free_mask = np.ones(n_nodes, dtype=bool)
    for f in fixed:
        free_mask[f] = False

    ei = np.array([e[0] for e in edges], dtype=np.int32)
    ej = np.array([e[1] for e in edges], dtype=np.int32)
    k_arr = np.array(k_edges)

    # Axial spring data: direction n and effective stiffness k/L^2
    n_dirs, k_eff = compute_edge_data(pos, ei, ej, k_arr, dim=dim)

    D_spr_trace = np.zeros(n_nodes)
    if len(ei) > 0:
        np.add.at(D_spr_trace, ei, k_eff)
        np.add.at(D_spr_trace, ej, k_eff)
    print(f"m/dt²={mass_dt2[0]:.1f}  avg spring diag(trace)={D_spr_trace[free_mask].mean():.1f}  "
          f"ratio inertial/total={mass_dt2[0]/(mass_dt2[0]+D_spr_trace[free_mask].mean()):.2%}")

    # RHS: b = M/dt^2 * y_rest + K * y_rest + f_gravity
    # The K*y_rest term is crucial: K acts on absolute positions, so without it
    # the springs see a phantom force pulling toward the origin (collapse bug).
    # Equivalently: solve (M/dt^2 + K) * u = f_gravity for displacement u = x - y_rest.
    gravity = np.zeros((n_nodes, dim))
    gravity[:, 1] = -9.81 * mass_val
    K_pos = matvec_A(pos, ei, ej, k_eff, n_dirs, np.zeros(n_nodes))  # K*y_rest (mass=0)
    b = mass_dt2[:, None] * pos + K_pos + gravity

    # Initial guess: strained + randomly perturbed (so solver has work to do)
    rng = np.random.default_rng(args.seed)
    x0 = pos * (1.0 + args.strain)  # uniform expansion
    x0[free_mask] += args.jitter * rng.standard_normal((n_nodes, dim))[free_mask]
    x0[~free_mask] = pos[~free_mask]  # keep fixed nodes at rest

    # --- direct solve ---
    A_dense = assemble_dense_A(ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim, fixed)
    b_flat = b.reshape(-1).copy()
    for f in fixed:
        for d in range(dim):
            b_flat[f*dim+d] = pos[f, d]
    x_direct = np.linalg.solve(A_dense, b_flat).reshape(n_nodes, dim)

    # --- global Jacobi + HB ---
    n_total = args.n_outer
    x_gj, res_gj = solve_global_jacobi(
        b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
        free_mask=free_mask, omega=0.8, beta=0.0, n_iter=n_total)

    # --- block Jacobi + HB ---
    patches = build_patches_grid(nx, ny, block_size=args.block_size, overlap=args.overlap, free_mask=free_mask)
    vp, D_global, D_global_inv = setup_patch_data(
        patches, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, free_mask, dim=dim,
        pos=pos, gravity=gravity)

    alt_shift = args.block_size // 2
    alt_patch_sets = [
        build_patches_grid_nonoverlap(nx, ny, block_size=args.block_size, shift_x=args.block_size, shift_y=args.block_size, free_mask=free_mask),
        build_patches_grid_nonoverlap(nx, ny, block_size=args.block_size, shift_x=alt_shift, shift_y=alt_shift, free_mask=free_mask),
    ]
    alt_patch_sets_data = []
    for pset in alt_patch_sets:
        setup_patch_data(pset, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, free_mask, dim=dim,
                         pos=pos, gravity=gravity, local_dirichlet=True)
        alt_patch_sets_data.append(pset)

    coverage = np.zeros(n_nodes, dtype=np.int32)
    core_coverage = np.zeros(n_nodes, dtype=np.int32)
    for p in patches:
        coverage[p['vertices']] += 1
        core_coverage[p['vertices'][p['core_mask_loc']]] += 1
    sizes = np.array([p['n_vertices'] for p in patches])
    core_sizes = np.array([p['n_core'] for p in patches])
    halo_sizes = np.array([p['n_halo'] for p in patches])
    print(f"Patches: {len(patches)}  block_size={args.block_size}  overlap={args.overlap}  grid={nx}x{ny}")
    print(f"  verts: min={sizes.min()} max={sizes.max()} mean={sizes.mean():.1f}")
    print(f"  overlap coverage free: min={coverage[free_mask].min()} max={coverage[free_mask].max()} mean={coverage[free_mask].mean():.1f}")
    print(f"Alternating sets: A={len(alt_patch_sets_data[0])} patches  B={len(alt_patch_sets_data[1])} patches  shift={alt_shift}")

    # Compare at same number of OUTER iterations (kernel launches).
    n_outer_fixed = args.n_outer
    configs = [
        (n_outer_fixed, 1,  0.8, 0.0, 1.0),
        (n_outer_fixed, 2,  0.8, 0.0, 1.0),
        (n_outer_fixed, 5,  0.8, 0.0, 1.0),
        (n_outer_fixed, 10, 0.8, 0.0, 1.0),
        (n_outer_fixed, 20, 0.8, 0.0, 1.0),
    ]
    colors = ['r', 'g', 'm', 'orange', 'cyan']

    err_gj = np.linalg.norm(x_gj[free_mask] - x_direct[free_mask]) / (
        np.linalg.norm(x_direct[free_mask]) + 1e-30)
    print(f"Global  Jacobi {n_total}x  = {n_total} total: res={res_gj[-1]:.4e}  err_vs_direct={err_gj:.4e}")

    straight_edges, diag_edges = classify_edges(edges, nx, ny)
    mass_dt2_val = mass_val / dt**2
    phys_str = f"k={k_spring:.0f}, m={mass_val:.0f}, dt={dt:.3f}, m/dt²={mass_dt2_val:.0f}, strain={args.strain}, jitter={args.jitter}"
    patch_str = (f"grid={nx}x{ny}, avg_patches={len(patches)}, alt_sets={len(alt_patch_sets_data[0])}+{len(alt_patch_sets_data[1])}, "
                 f"block={args.block_size}x{args.block_size}, overlap={args.overlap}, shift={alt_shift}")

    for mode_name, use_scalar in [("scalar", True), ("matrix", False)]:
        print(f"\n=== {mode_name} weighting ===")
        bj_results = {}
        alt_results = {}
        for n_outer, n_inner, om, bt, om_i in configs:
            x_bj, res_bj = solve_block_jacobi(
                b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
                patches=patches, vp=vp, free_mask=free_mask, D_global_inv=D_global_inv,
                omega=om, beta=bt, omega_inner=om_i,
                n_outer=n_outer, n_inner=n_inner, use_scalar_weight=use_scalar)
            bj_results[(n_outer, n_inner)] = (x_bj, res_bj)
            err = np.linalg.norm(x_bj[free_mask] - x_direct[free_mask]) / (
                np.linalg.norm(x_direct[free_mask]) + 1e-30)
            print(f"Block Jacobi {n_outer:3d}x{n_inner:2d} = {n_outer*n_inner:3d} total "
                  f"(ω={om:.2f},β={bt:.2f}): res={res_bj[-1]:.4e}  err={err:.4e}")
            x_alt, res_alt = solve_alternating_patches(
                b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
                patch_sets=alt_patch_sets_data, free_mask=free_mask,
                omega=om, beta=bt, omega_inner=om_i,
                n_outer=n_outer, n_inner=n_inner)
            alt_results[(n_outer, n_inner)] = (x_alt, res_alt)
            err_alt = np.linalg.norm(x_alt[free_mask] - x_direct[free_mask]) / (
                np.linalg.norm(x_direct[free_mask]) + 1e-30)
            print(f"Alt patches  {n_outer:3d}x{n_inner:2d} = {n_outer*n_inner:3d} total "
                  f"(ω={om:.2f},β={bt:.2f}): res={res_alt[-1]:.4e}  err={err_alt:.4e}")

        solver_str = f"{'scalar trace W' if use_scalar else 'exact matrix Dg⁻¹Dp'}  |  ω=0.8, β=0.0, ωi=1.0; dashed=alternating shifted no-average"

        fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
        fig.suptitle(f"{phys_str}\n{patch_str}\n{solver_str}", fontsize=8)

        # convergence
        ax = axes[0]
        ax.semilogy(np.arange(1, n_total + 1), res_gj, 'b-', lw=2,
                    label=f'Global Jacobi ({n_total})')
        for (no, ni, om, bt, om_i), col in zip(configs, colors):
            ax.semilogy(np.arange(1, no + 1), bj_results[(no, ni)][1], color=col, marker='o', ms=3,
                        label=f'BJ {no}x{ni}')
            ax.semilogy(np.arange(1, no + 1), alt_results[(no, ni)][1], color=col, ls='--', lw=1.4,
                        label=f'Alt {no}x{ni}')
        ax.set_xlabel('Outer iterations (kernel launches)')
        ax.set_ylabel('Relative residual')
        ax.set_title('Convergence')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # deformation + patch boxes
        ax = axes[1]
        x_bj_best = bj_results[(n_outer_fixed, 5)][0]
        patch_colors = ['r', 'g', 'm', 'orange', 'cyan', 'purple', 'brown']
        for i, j in straight_edges:
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 'k-', lw=0.3, alpha=0.2)
        for pi, pat in enumerate(patches):
            if 'grid_ix0' in pat:
                ix0, iy0 = pat['grid_ix0'], pat['grid_iy0']
                ix1, iy1 = pat['grid_ix1'], pat['grid_iy1']
                pad = 0.15
                rect = Rectangle((ix0 - pad, iy0 - pad), (ix1 - ix0) + 2 * pad, (iy1 - iy0) + 2 * pad,
                                 fill=False, ec=patch_colors[pi % len(patch_colors)], lw=1.0, alpha=0.45)
                ax.add_patch(rect)
                ax.text(ix0, iy1 + pad, f'P{pi}', color=patch_colors[pi % len(patch_colors)], fontsize=7)
            else:
                pts = pos[pat['vertices']]
                xmin, ymin = pts[:, :2].min(axis=0)
                xmax, ymax = pts[:, :2].max(axis=0)
                pad = 0.15
                rect = Rectangle((xmin - pad, ymin - pad), (xmax - xmin) + 2 * pad, (ymax - ymin) + 2 * pad,
                                 fill=False, ec=patch_colors[pi % len(patch_colors)], lw=1.0, alpha=0.45)
                ax.add_patch(rect)
        ax.plot(pos[:, 0], pos[:, 1], 'k.', ms=2, label='rest')
        ax.plot(x0[:, 0], x0[:, 1], 'k+', ms=3, alpha=0.4, label='initial (strained)')
        ax.plot(x_direct[:, 0], x_direct[:, 1], 'b.', ms=4, label='direct')
        ax.plot(x_bj_best[:, 0], x_bj_best[:, 1], 'r+', ms=4, alpha=0.7, label=f'BJ {n_outer_fixed}x5')
        ax.set_aspect('equal')
        ax.set_title('Deformation + patch boxes')
        ax.legend(fontsize=8)

        # error per vertex
        ax = axes[2]
        err_per_node = np.linalg.norm(x_bj_best - x_direct, axis=1)
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=err_per_node, s=30, cmap='hot_r')
        for i, j in straight_edges:
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 'k-', lw=0.2, alpha=0.15)
        ax.set_aspect('equal')
        ax.set_title('Per-node error')
        fig.colorbar(sc, ax=ax, label='||x_bj - x_direct||')

        plt.tight_layout(rect=[0, 0, 1, 0.88])
        fname = f'block_jacobi_{mode_name}.png'
        plt.savefig(fname, dpi=150)
        print(f"Saved {fname}")
        if not args.no_show:
            plt.show()
        plt.close(fig)

    # --- separate figure for alternating shifted patches with beta sweep ---
    print("\n=== alternating shifted patches: beta sweep (beta[0]=0) ===")
    beta_values = [0.0, 0.3, 0.5, 0.7]
    beta_labels = ['0.0', '0.3', '0.5', '0.7']
    n_inner_plot = [1, 2, 5, 10, 20]
    n_inner_colors = {1: 'r', 2: 'g', 5: 'm', 10: 'orange', 20: 'cyan'}
    alt_beta_results = {}  # (beta_label, n_inner) -> (x, res)
    for bt, blabel in zip(beta_values, beta_labels):
        for n_outer, n_inner, om, _, om_i in configs:
            betas_arr = np.full(n_outer, bt)
            betas_arr[0] = 0.0  # no momentum on first iteration
            x_alt, res_alt = solve_alternating_patches(
                b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
                patch_sets=alt_patch_sets_data, free_mask=free_mask,
                omega=om, beta=betas_arr, omega_inner=om_i,
                n_outer=n_outer, n_inner=n_inner)
            alt_beta_results[(blabel, n_inner)] = (x_alt, res_alt)
            err_alt = np.linalg.norm(x_alt[free_mask] - x_direct[free_mask]) / (
                np.linalg.norm(x_direct[free_mask]) + 1e-30)
            print(f"Alt β={bt:.1f} {n_outer:3d}x{n_inner:2d} = {n_outer*n_inner:3d} total: "
                  f"res={res_alt[-1]:.4e}  err={err_alt:.4e}")

    fig, axes = plt.subplots(1, len(beta_values) + 1, figsize=(20, 5.4))
    alt_solver_str = (f"alternating shifted no-average  |  ω=0.8, ωi=1.0, β[0]=0  |  "
                      f"sets: A={len(alt_patch_sets_data[0])} B={len(alt_patch_sets_data[1])} shift={alt_shift}")
    fig.suptitle(f"{phys_str}\n{patch_str}\n{alt_solver_str}", fontsize=8)

    # one subplot per beta, lines for different n_inner
    for ax_idx, blabel in enumerate(beta_labels):
        ax = axes[ax_idx]
        ax.semilogy(np.arange(1, n_total + 1), res_gj, 'b-', lw=2, label='GJ')
        for ni in n_inner_plot:
            key = (blabel, ni)
            if key not in alt_beta_results:
                continue
            col = n_inner_colors[ni]
            ax.semilogy(np.arange(1, n_outer_fixed + 1), alt_beta_results[key][1],
                        color=col, marker='s', ms=3,
                        label=f'{n_outer_fixed}x{ni}')
        ax.set_xlabel('Outer iterations')
        ax.set_ylabel('Relative residual')
        ax.set_title(f'β={blabel}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # deformation + patch boxes (best beta=0.5, n_inner=5)
    ax = axes[-1]
    x_alt_best = alt_beta_results[('0.5', 5)][0]
    patch_colors = ['r', 'g', 'm', 'orange', 'cyan', 'purple', 'brown']
    for i, j in straight_edges:
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 'k-', lw=0.3, alpha=0.2)
    for si, pset in enumerate(alt_patch_sets_data):
        for pi, pat in enumerate(pset):
            if 'grid_ix0' in pat:
                ix0, iy0 = pat['grid_ix0'], pat['grid_iy0']
                ix1, iy1 = pat['grid_ix1'], pat['grid_iy1']
                pad = 0.15
                ls = '-' if si == 0 else '--'
                rect = Rectangle((ix0 - pad, iy0 - pad), (ix1 - ix0) + 2 * pad, (iy1 - iy0) + 2 * pad,
                                 fill=False, ec=patch_colors[pi % len(patch_colors)], lw=1.0, alpha=0.5, ls=ls)
                ax.add_patch(rect)
    ax.plot(pos[:, 0], pos[:, 1], 'k.', ms=2, label='rest')
    ax.plot(x0[:, 0], x0[:, 1], 'k+', ms=3, alpha=0.4, label='initial')
    ax.plot(x_direct[:, 0], x_direct[:, 1], 'b.', ms=4, label='direct')
    ax.plot(x_alt_best[:, 0], x_alt_best[:, 1], 'r+', ms=4, alpha=0.7, label='Alt β=0.5 15x5')
    ax.set_aspect('equal')
    ax.set_title('Deformation (solid=A, dashed=B)')
    ax.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig('block_jacobi_alternating.png', dpi=150)
    print("Saved block_jacobi_alternating.png")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
