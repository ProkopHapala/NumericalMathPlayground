"""
MultiGrid.py — Multigrid solvers for truss elasticity.

This is a **core module** implementing prolongation operators, Galerkin coarse
operators, and two-grid V-cycle solvers.  It builds on TrussSolver.py's
matrix-free operators and uses global Jacobi as the fine-grid smoother.

Three strategies for building the prolongation P (coarse space):

  A. Spectral  — lowest eigenmodes of K via Lanczos (scipy.sparse.linalg.eigsh)
  B. Geometric — farthest-point pivot sampling + inverse-distance interpolation
  C. Beam      — coarse graph nodes + optional parabolic bending modes per edge

Responsibilities
----------------
1. **Flat matvec** — `matvec_A_flat` wraps `matvec_A` for flattened DOF vectors.
2. **Prolongation builders** — `build_spectral_prolongation`,
   `build_pivot_prolongation`, `build_beam_prolongation`.
3. **Beam membership** — `compute_fine_node_to_beam` maps fine nodes to
   coarse edges with parameter s.
4. **Galerkin coarse operator** — `galerkin_coarse_operator`.
5. **Two-grid solver** — `solve_two_grid` (V-cycle with Jacobi smoother +
   coarse correction).
6. **Multigrid driver** — `solve_multigrid` (iterate V-cycles to convergence).

Role in the system
------------------
- **Truss.py**: geometry, mesh, assembly, bookkeeping.
- **TrussSolver.py**: iterative and direct linear algebra solvers.
- **MultiGrid.py** (this file): multigrid-specific prolongation and solvers.
- **TrussPlotting.py**: all reusable plotting functions.
- Scripts: thin wrappers that combine the modules.
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from TrussSolver import (
    matvec_A, compute_diagonal_3x3, invert_3x3_blocks,
    build_adjacency, bfs_distances,
)


# ---------------------------------------------------------------------------
# Flat matvec helper
# ---------------------------------------------------------------------------

def matvec_A_flat(x_flat, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=3):
    """matvec_A for flattened DOF vector x_flat: (N*dim,) -> (N*dim,)."""
    x = x_flat.reshape(n_nodes, dim)
    Ax = matvec_A(x, ei, ej, k_eff, n_dirs, mass_dt2)
    return Ax.reshape(-1)


# ---------------------------------------------------------------------------
# Strategy A: Spectral prolongation (Lanczos lowest modes)
# ---------------------------------------------------------------------------

def build_spectral_prolongation(ei, ej, k_eff, n_dirs, mass_dt2,
                                n_nodes, dim=3, m=10, free_mask=None,
                                max_lanczos_iter=None, dense_threshold=500,
                                use_stiffness=True):
    """
    Build prolongation P from the m lowest eigenmodes.

    If use_stiffness=True (default), finds the lowest eigenmodes of the
    stiffness matrix K (the true 'soft modes').  This is preferred because
    A = M/dt^2 + K is mass-dominated and its eigenvalues are clustered,
    making the soft modes hard to distinguish.

    If use_stiffness=False, finds lowest eigenmodes of A = M/dt^2 + K directly.

    For small problems (ndof_free <= dense_threshold), uses dense eigendecomposition.
    For larger problems, uses scipy.sparse.linalg.eigsh with shift-invert mode.

    Parameters
    ----------
    m : number of lowest eigenmodes to extract
    free_mask : (N,) bool array, False = fixed nodes (excluded from eigenproblem)
    dense_threshold : if ndof_free <= this, use dense eigh (faster + more robust)
    use_stiffness : if True, eigen-decompose K (not A) to find soft modes

    Returns
    -------
    P : (N*dim, m) prolongation matrix
    eigvals : (m,) eigenvalues of K (or A if use_stiffness=False)
    """
    if free_mask is None:
        free_mask = np.ones(n_nodes, dtype=bool)

    n_free = int(free_mask.sum())
    ndof_free = n_free * dim
    free_idx = np.where(free_mask)[0]
    free_dof_idx = np.sort(np.concatenate([free_idx * dim + d for d in range(dim)]))

    k = min(m, ndof_free - 1)
    if k < 1:
        raise ValueError(f"Not enough free DOFs ({ndof_free}) for spectral prolongation with m={m}")

    from TrussSolver import assemble_dense_A

    if use_stiffness:
        # Assemble K only (mass_dt2 = 0)
        K_full = assemble_dense_A(ei, ej, k_eff, n_dirs, np.zeros(n_nodes), n_nodes, dim)
        K_free = K_full[np.ix_(free_dof_idx, free_dof_idx)]
        # Regularize to avoid exact zeros from rigid-body modes
        K_free = K_free + 1e-8 * np.eye(ndof_free) * np.trace(K_free) / max(ndof_free, 1)
        if ndof_free <= dense_threshold:
            eigvals, eigvecs = np.linalg.eigh(K_free)
            eigvals = eigvals[:k]
            eigvecs = eigvecs[:, :k]
        else:
            from scipy.sparse.linalg import eigsh
            if max_lanczos_iter is None:
                max_lanczos_iter = max(2 * m + 50, 100)
            sigma = 0.0
            eigvals, eigvecs = eigsh(K_free, k=k, sigma=sigma, which='LM',
                                     mode='normal', maxiter=max_lanczos_iter)
    else:
        # Use A = M/dt^2 + K directly
        A_full = assemble_dense_A(ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim)
        A_free = A_full[np.ix_(free_dof_idx, free_dof_idx)]
        if ndof_free <= dense_threshold:
            eigvals, eigvecs = np.linalg.eigh(A_free)
            eigvals = eigvals[:k]
            eigvecs = eigvecs[:, :k]
        else:
            from scipy.sparse.linalg import eigsh
            if max_lanczos_iter is None:
                max_lanczos_iter = max(2 * m + 50, 100)
            sigma = np.trace(A_free) / ndof_free * 0.5
            eigvals, eigvecs = eigsh(A_free, k=k, sigma=sigma, which='LM',
                                     mode='normal', maxiter=max_lanczos_iter)

    # Map back to full DOF space
    n_dof = n_nodes * dim
    P = np.zeros((n_dof, k))
    P[free_dof_idx, :] = eigvecs

    return P, eigvals


# ---------------------------------------------------------------------------
# Strategy B: Geometric pivot-based prolongation
# ---------------------------------------------------------------------------

def select_pivots_maximin(pos, adj, n_nodes, n_pivots, free_mask=None,
                          n_swap_iter=0, rng=None):
    """
    Farthest-point (maximin) sampling of pivot nodes.

    Phase 1: greedy maximin — pick farthest node from current set.
    Phase 2 (optional): swap optimization to maximize min pairwise distance.

    Returns
    -------
    pivots : (n_pivots,) array of node indices
    """
    if free_mask is None:
        free_mask = np.ones(n_nodes, dtype=bool)
    if rng is None:
        rng = np.random.default_rng()

    free_nodes = np.where(free_mask)[0]
    if len(free_nodes) == 0:
        raise ValueError("No free nodes for pivot selection")

    # Phase 1: greedy maximin using BFS distances
    first = int(free_nodes[rng.integers(len(free_nodes))])
    pivots = [first]
    min_dist = bfs_distances(adj, first, n_nodes).astype(float)
    min_dist[~free_mask] = -1

    while len(pivots) < n_pivots:
        nxt = int(np.argmax(min_dist))
        if not free_mask[nxt] or min_dist[nxt] <= 0:
            break
        pivots.append(nxt)
        d = bfs_distances(adj, nxt, n_nodes).astype(float)
        min_dist = np.minimum(min_dist, d)
        min_dist[~free_mask] = -1

    # Phase 2: swap optimization
    if n_swap_iter > 0 and len(pivots) > 1:
        all_dist = np.zeros((n_nodes, len(pivots)), dtype=np.int32)
        for p, s in enumerate(pivots):
            all_dist[:, p] = bfs_distances(adj, s, n_nodes)

        pivot_set = set(pivots)
        for _ in range(n_swap_iter):
            improved = False
            for pi in range(len(pivots)):
                old_p = pivots[pi]
                best_p = old_p
                best_min_dist = _min_pairwise_dist(all_dist, pi, pivots)
                for candidate in free_nodes:
                    if candidate in pivot_set:
                        continue
                    d_cand = bfs_distances(adj, int(candidate), n_nodes)
                    all_dist[:, pi] = d_cand
                    new_min = _min_pairwise_dist(all_dist, pi, pivots)
                    if new_min > best_min_dist:
                        best_min_dist = new_min
                        best_p = int(candidate)
                    all_dist[:, pi] = bfs_distances(adj, old_p, n_nodes)
                if best_p != old_p:
                    pivot_set.discard(old_p)
                    pivot_set.add(best_p)
                    pivots[pi] = best_p
                    all_dist[:, pi] = bfs_distances(adj, best_p, n_nodes)
                    improved = True
            if not improved:
                break

    return np.array(pivots)


def _min_pairwise_dist(all_dist, exclude_pi, pivots):
    """Min pairwise BFS distance between pivot exclude_pi and all other pivots."""
    k = len(pivots)
    min_d = float('inf')
    for pj in range(k):
        if pj == exclude_pi:
            continue
        d = all_dist[pivots[exclude_pi], pj]
        if d >= 0 and d < min_d:
            min_d = float(d)
    return min_d


def build_pivot_prolongation(pos, pivots, dim=3, power=2.0, r_cut=None,
                             free_mask=None):
    """
    Build prolongation P using inverse-distance interpolation from pivot nodes.

    P[i*dim+d, p*dim+d] = w_{i,p}  (same weight per spatial component)

    Parameters
    ----------
    pos : (N, dim) node positions
    pivots : (k,) indices of pivot nodes
    power : exponent for inverse distance weighting
    r_cut : cutoff radius (only pivots within r_cut contribute)
    free_mask : (N,) bool — fixed nodes get zero rows

    Returns
    -------
    P : (N*dim, k*dim) prolongation matrix
    """
    n_nodes = pos.shape[0]
    k = len(pivots)
    n_dof = n_nodes * dim
    n_coarse = k * dim

    if free_mask is None:
        free_mask = np.ones(n_nodes, dtype=bool)

    P = np.zeros((n_dof, n_coarse))
    pivot_pos = pos[pivots]

    for i in range(n_nodes):
        if not free_mask[i]:
            continue
        d = np.linalg.norm(pos[i] - pivot_pos, axis=1)
        if r_cut is not None:
            mask = d < r_cut
            if not mask.any():
                mask = d <= d.min() + 1e-12
        else:
            mask = np.ones(k, dtype=bool)

        d_safe = np.where(d > 1e-12, d, 1e-12)
        w = np.where(mask, 1.0 / d_safe**power, 0.0)
        w_sum = w.sum()
        if w_sum > 1e-30:
            w = w / w_sum
        else:
            w = np.zeros(k)
            w[np.argmin(d)] = 1.0

        for pd in range(dim):
            for d_idx in range(dim):
                P[i * dim + d_idx, pd * dim + d_idx] = w[pd]

    return P


# ---------------------------------------------------------------------------
# Strategy C: Coarse beam structure prolongation
# ---------------------------------------------------------------------------

def compute_fine_node_to_beam(fine_pos, coarse_pos, coarse_edges,
                              coarse_to_fine, thickness):
    """
    For each fine node, determine which beam (coarse edge) it belongs to
    and the parameter s in [0, 1] along that beam.

    Junction nodes (listed in coarse_to_fine) are assigned to their coarse node.
    Interior nodes are matched to the nearest beam axis by projection.

    Returns
    -------
    node_to_coarse : dict {fine_node_idx -> coarse_node_idx}  for junction nodes
    node_to_beam : dict {fine_node_idx -> (edge_idx, s)}  for beam interior nodes
    """
    n_fine = fine_pos.shape[0]
    n_coarse = coarse_pos.shape[0]

    junction_nodes = set()
    node_to_coarse = {}
    for c_idx, fine_list in coarse_to_fine.items():
        for fn in fine_list:
            node_to_coarse[fn] = c_idx
            junction_nodes.add(fn)

    node_to_beam = {}

    for i in range(n_fine):
        if i in junction_nodes:
            continue

        best_edge = -1
        best_s = 0.0
        best_dist = float('inf')

        for ei, (a, b) in enumerate(coarse_edges):
            pa = coarse_pos[a]
            pb = coarse_pos[b]
            ab = pb - pa
            L2 = np.dot(ab, ab)
            if L2 < 1e-20:
                continue
            t = np.dot(fine_pos[i] - pa, ab) / L2
            t = np.clip(t, 0.0, 1.0)
            proj = pa + t * ab
            dist = np.linalg.norm(fine_pos[i] - proj)
            if dist < best_dist:
                best_dist = dist
                best_edge = ei
                best_s = t

        if best_edge >= 0:
            node_to_beam[i] = (best_edge, best_s)

    return node_to_coarse, node_to_beam


def build_beam_prolongation(fine_pos, coarse_pos, coarse_edges,
                            coarse_to_fine, dim=3, bending=True,
                            n_bending_modes=1, free_mask=None):
    """
    Build prolongation P from coarse beam structure with optional bending modes.

    Coarse DOF layout:
      [0 .. dim*n_c - 1]                        = coarse node displacements
      [dim*n_c .. dim*n_c + n_bend*n_e - 1]    = bending amplitudes per edge

    For 2D (dim=2): 1 bending direction per edge (in-plane perpendicular)
    For 3D (dim=3): 2 bending directions per edge (two perpendiculars)

    Bending shape: phi_k(s) = sin((k+1) * pi * s)

    Parameters
    ----------
    fine_pos : (M, dim) fine node positions
    coarse_pos : (N, dim) coarse node positions
    coarse_edges : list of (i, j)
    coarse_to_fine : dict {coarse_node -> list of fine node indices}
    bending : if True, include bending DOFs
    n_bending_modes : number of sine bending modes per edge per direction

    Returns
    -------
    P : (M*dim, n_coarse_dof) prolongation matrix
    info : dict with metadata (n_coarse_nodes, n_edges, n_bend_per_edge, etc.)
    """
    n_fine = fine_pos.shape[0]
    n_coarse = len(coarse_pos)
    n_edges = len(coarse_edges)

    n_bend_dirs = (1 if dim == 2 else 2) if bending else 0
    n_bend_per_edge = n_bend_dirs * n_bending_modes
    n_coarse_dof = dim * n_coarse + n_bend_per_edge * n_edges
    n_fine_dof = dim * n_fine

    if free_mask is None:
        free_mask = np.ones(n_fine, dtype=bool)

    node_to_coarse, node_to_beam = compute_fine_node_to_beam(
        fine_pos, coarse_pos, coarse_edges, coarse_to_fine,
        thickness=0.3  # not used for projection, only for reference
    )

    P = np.zeros((n_fine_dof, n_coarse_dof))

    for i in range(n_fine):
        if not free_mask[i]:
            continue

        if i in node_to_coarse:
            c = node_to_coarse[i]
            for d in range(dim):
                P[i * dim + d, c * dim + d] = 1.0
        elif i in node_to_beam:
            ei, s = node_to_beam[i]
            a, b = coarse_edges[ei]
            u = coarse_pos[b] - coarse_pos[a]
            L = np.linalg.norm(u)
            if L > 1e-12:
                u = u / L
            else:
                u = np.array([1.0, 0.0, 0.0])[:dim]

            # Linear interpolation
            for d in range(dim):
                P[i * dim + d, a * dim + d] += (1.0 - s)
                P[i * dim + d, b * dim + d] += s

            # Bending modes
            if bending and n_bend_dirs > 0:
                perp_dirs = _perpendicular_directions(u, dim)
                for dir_idx, n_dir in enumerate(perp_dirs):
                    for mode_k in range(n_bending_modes):
                        phi = np.sin((mode_k + 1) * np.pi * s)
                        bend_dof = dim * n_coarse + ei * n_bend_per_edge + \
                                   dir_idx * n_bending_modes + mode_k
                        for d in range(dim):
                            P[i * dim + d, bend_dof] = phi * n_dir[d]

    info = {
        'n_coarse_nodes': n_coarse,
        'n_edges': n_edges,
        'n_bend_dirs': n_bend_dirs,
        'n_bending_modes': n_bending_modes,
        'n_bend_per_edge': n_bend_per_edge,
        'n_coarse_dof': n_coarse_dof,
        'node_to_coarse': node_to_coarse,
        'node_to_beam': node_to_beam,
    }
    return P, info


def _perpendicular_directions(u, dim=3):
    """Return perpendicular direction(s) to unit vector u."""
    if dim == 2:
        return [np.array([-u[1], u[0]])]
    elif dim == 3:
        if abs(u[2]) < 0.9:
            n1 = np.cross(u, np.array([0.0, 0.0, 1.0]))
        else:
            n1 = np.cross(u, np.array([1.0, 0.0, 0.0]))
        n1 = n1 / (np.linalg.norm(n1) + 1e-30)
        n2 = np.cross(u, n1)
        n2 = n2 / (np.linalg.norm(n2) + 1e-30)
        return [n1, n2]
    else:
        raise ValueError(f"Unsupported dim={dim}")


# ---------------------------------------------------------------------------
# Galerkin coarse operator
# ---------------------------------------------------------------------------

def galerkin_coarse_operator(P, ei, ej, k_eff, n_dirs, mass_dt2,
                             n_nodes, dim=3, free_mask=None):
    """
    Build Galerkin coarse operator A_c = P^T A P via matvec on each column of P.

    Parameters
    ----------
    P : (N*dim, n_coarse) prolongation matrix

    Returns
    -------
    A_c : (n_coarse, n_coarse) dense coarse operator
    """
    n_coarse = P.shape[1]
    n_dof = n_nodes * dim

    if free_mask is None:
        free_mask = np.ones(n_nodes, dtype=bool)

    # Apply A to each column of P
    AP = np.zeros((n_dof, n_coarse))
    for j in range(n_coarse):
        AP[:, j] = matvec_A_flat(P[:, j], ei, ej, k_eff, n_dirs,
                                 mass_dt2, n_nodes, dim)

    # A_c = P^T A P
    A_c = P.T @ AP

    # Ensure SPD by projecting to symmetric part
    A_c = 0.5 * (A_c + A_c.T)

    # Regularize: add small diagonal to prevent singularity from null-space modes
    n_coarse = A_c.shape[0]
    diag_max = np.max(np.diag(A_c)) + 1e-30
    A_c += 1e-6 * diag_max * np.eye(n_coarse)

    return A_c


# ---------------------------------------------------------------------------
# Two-grid V-cycle solver
# ---------------------------------------------------------------------------

def _jacobi_smooth(b, x, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim,
                   free_mask, Dinv, omega, n_steps, x0=None):
    """Perform n_steps of damped Jacobi smoothing in-place on x."""
    if x0 is None:
        x0 = x.copy()
    for _ in range(n_steps):
        r = b - matvec_A(x, ei, ej, k_eff, n_dirs, mass_dt2)
        dx = np.zeros_like(x)
        for n in np.where(free_mask)[0]:
            dx[n] = Dinv[n] @ r[n]
        x = x + omega * dx
        x[~free_mask] = x0[~free_mask]
    return x


def solve_two_grid(b, x0, P, A_c_lu, ei, ej, k_eff, n_dirs, mass_dt2,
                   n_nodes, dim=3, free_mask=None,
                   omega=0.8, n_pre_smooth=3, n_post_smooth=3,
                   n_outer=50, beta=0.0, tol=1e-8):
    """
    Two-grid V-cycle solver: Jacobi smoothing + Galerkin coarse correction.

    Each V-cycle:
      1. Pre-smooth (n_pre_smooth Jacobi steps)
      2. Restrict residual: r_c = P^T (b - A x)
      3. Coarse solve: e_c = A_c^{-1} r_c  (dense LU)
      4. Prolongate: x += P e_c
      5. Post-smooth (n_post_smooth Jacobi steps)

    Parameters
    ----------
    P : (N*dim, n_coarse) prolongation
    A_c_lu : LU factorization of coarse operator (from scipy.linalg.lu_factor)
    omega : Jacobi damping
    beta : heavy-ball momentum for outer iterations
    n_outer : max V-cycles

    Returns
    -------
    x : (N, dim) solution
    residuals : list of relative residuals (length n_outer+1)
    """
    if free_mask is None:
        free_mask = np.ones(n_nodes, dtype=bool)

    D = compute_diagonal_3x3(ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim)
    Dinv = invert_3x3_blocks(D)

    x = x0.copy()
    v = np.zeros_like(x)
    residuals = []
    b_norm = np.linalg.norm(b[free_mask]) + 1e-30
    r0 = b - matvec_A(x, ei, ej, k_eff, n_dirs, mass_dt2)
    residuals.append(np.linalg.norm(r0[free_mask]) / b_norm)

    for it in range(n_outer):
        # Pre-smooth
        x = _jacobi_smooth(b, x, ei, ej, k_eff, n_dirs, mass_dt2,
                           n_nodes, dim, free_mask, Dinv, omega, n_pre_smooth, x0=x0)

        # Coarse correction
        r = b - matvec_A(x, ei, ej, k_eff, n_dirs, mass_dt2)
        r_flat = r.reshape(-1)
        r_c = P.T @ r_flat
        e_c = lu_solve(A_c_lu, r_c)
        x_flat = x.reshape(-1) + P @ e_c
        x = x_flat.reshape(n_nodes, dim)
        x[~free_mask] = x0[~free_mask]

        # Post-smooth
        x = _jacobi_smooth(b, x, ei, ej, k_eff, n_dirs, mass_dt2,
                           n_nodes, dim, free_mask, Dinv, omega, n_post_smooth, x0=x0)

        # Residual
        r = b - matvec_A(x, ei, ej, k_eff, n_dirs, mass_dt2)
        res = np.linalg.norm(r[free_mask]) / b_norm
        residuals.append(res)

        if res < tol:
            break

    return x, residuals


def solve_multigrid(b, x0, P, ei, ej, k_eff, n_dirs, mass_dt2,
                    n_nodes, dim=3, free_mask=None,
                    omega=0.8, n_pre_smooth=3, n_post_smooth=3,
                    n_outer=100, tol=1e-8):
    """
    Convenience wrapper: build Galerkin operator, factorize, and run V-cycles.

    Returns
    -------
    x : (N, dim) solution
    residuals : list of relative residuals
    A_c : coarse operator (for inspection)
    """
    A_c = galerkin_coarse_operator(P, ei, ej, k_eff, n_dirs, mass_dt2,
                                   n_nodes, dim, free_mask)
    A_c_lu = lu_factor(A_c)

    x, residuals = solve_two_grid(
        b, x0, P, A_c_lu, ei, ej, k_eff, n_dirs, mass_dt2,
        n_nodes, dim, free_mask, omega, n_pre_smooth, n_post_smooth,
        n_outer, beta=0.0, tol=tol
    )
    return x, residuals, A_c
