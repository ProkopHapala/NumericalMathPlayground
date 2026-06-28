"""
WeightOptimizer.py — Quadrature weight optimization for CubeEmbededAtoms grid.

Core module: geometric weight computation, test function matrix assembly,
Tikhonov regularized least-squares optimization, validation against STO-nG.
No I/O, no plotting.
"""
import numpy as np
from scipy.linalg import lstsq

from CubeEmbededGrid import quad_area, collect_unique_points
from Symmetry import group_orbits
from BasisFunctions import (
    eval_test_func, build_test_function_set,
    gaussian_1s, gaussian_2p,
    analytic_1s_norm_sq, analytic_2p_norm_sq,
    STO3G_1S, STO3G_2P_C,
)


def geometric_weights(grid_dict):
    """
    Compute geometric weights w0 for each unique grid point.
    Each point gets 1/4 of the area of surrounding quads.

    Returns
    -------
    w0 : (Npts,) array
    grid_pts : (Npts, 2) array
    grid_idx : dict (i, j) -> point index
    """
    points = grid_dict['points']
    Nu, Nv = points.shape[:2]
    grid_pts, grid_idx = collect_unique_points(grid_dict)
    Npts = len(grid_pts)
    w0 = np.zeros(Npts)

    # Center point
    center_area = 0.0
    for i in range(Nu):
        i2 = (i + 1) % Nu
        a = quad_area(points[i, 0], points[i2, 0], points[i2, 1], points[i, 1])
        center_area += a
    w0[0] = center_area / 4.0

    # Shell points
    for j in range(1, Nv):
        for i in range(Nu):
            i2 = (i + 1) % Nu
            i1 = (i - 1) % Nu
            surrounding = []
            if j > 0:
                surrounding.append(quad_area(points[i1, j-1], points[i, j-1], points[i, j], points[i1, j]))
                surrounding.append(quad_area(points[i, j-1], points[i2, j-1], points[i2, j], points[i, j]))
            if j < Nv - 1:
                surrounding.append(quad_area(points[i1, j], points[i, j], points[i, j+1], points[i1, j+1]))
                surrounding.append(quad_area(points[i, j], points[i2, j], points[i2, j+1], points[i, j+1]))
            idx = grid_idx[(i, j)]
            w0[idx] = np.sum(surrounding) / 4.0

    return w0, grid_pts, grid_idx


def build_integration_matrix(grid_pts, orbit_members, test_funcs):
    """
    Build matrix A[k, oid] = Σ_{p in orbit oid} f_k(x_p).

    Returns
    -------
    A : (Ntest, N_orbits) array
    """
    Ntest = len(test_funcs)
    N_orbits = len(orbit_members)
    A = np.zeros((Ntest, N_orbits))

    for k, (label, params) in enumerate(test_funcs):
        for oid, members in enumerate(orbit_members):
            pts = grid_pts[members]
            if params is None:
                A[k, oid] = len(members)
            else:
                kind, zeta, x0, y0 = params
                vals = eval_test_func(pts, kind, zeta, x0, y0)
                A[k, oid] = np.sum(vals)
    return A


def compute_targets(test_funcs, integrals_total, outer_xy, outer_w, h):
    """
    Compute target vector b_inner = I_total - I_outer for each test function.

    Returns
    -------
    b_inner : (Ntest,) array
    I_outer : (Ntest,) array
    """
    Ntest = len(test_funcs)
    I_outer = np.zeros(Ntest)

    for k, (label, params) in enumerate(test_funcs):
        if params is None:
            I_outer[k] = outer_w.sum()
        else:
            kind, zeta, x0, y0 = params
            vals = eval_test_func(outer_xy, kind, zeta, x0, y0)
            I_outer[k] = np.sum(vals * outer_w)

    b_inner = np.array([iv if iv is not None else 0 for iv in integrals_total]) - I_outer
    b_inner[0] = 4 * h * h  # constant: only inner domain

    # Odd functions: set target to 0 exactly
    for k, (label, params) in enumerate(test_funcs):
        if params is not None and params[0] in ('px', 'xyg'):
            b_inner[k] = 0.0

    return b_inner, I_outer


def tikhonov_optimize(A, b, w0_orbit, lambdas, q=None):
    """
    Run Tikhonov-regularized least-squares for multiple λ values.

    min ||diag(q) (A @ w - b)||² + λ ||w - w0||²

    Returns
    -------
    results : dict λ -> {w_orbit, w_full, errs, max_err, mean_err}
    """
    Ntest, N_orbits = A.shape
    if q is None:
        q = np.ones(Ntest)

    A_w = A * q[:, None]
    b_w = b * q

    results = {}
    for lam in lambdas:
        A_aug = np.vstack([A_w, np.sqrt(lam) * np.eye(N_orbits)])
        b_aug = np.concatenate([b_w, np.sqrt(lam) * w0_orbit])
        w_orbit, _, _, _ = lstsq(A_aug, b_aug)

        predicted = A @ w_orbit
        nonzero = np.abs(b) > 1e-10
        errs = np.zeros(Ntest)
        errs[nonzero] = np.abs(predicted[nonzero] - b[nonzero]) / np.abs(b[nonzero]) * 100
        max_err = np.max(errs[nonzero]) if np.any(nonzero) else 0
        mean_err = np.mean(errs[nonzero]) if np.any(nonzero) else 0

        results[lam] = {
            'w_orbit': w_orbit.copy(),
            'errs': errs.copy(),
            'max_err': max_err,
            'mean_err': mean_err,
            'predicted': predicted,
        }

    return results


def expand_orbit_weights(w_orbit, orbit_members, Npts):
    """Expand per-orbit weights to per-point weights."""
    w_full = np.zeros(Npts)
    for oid, members in enumerate(orbit_members):
        for p in members:
            w_full[p] = w_orbit[oid]
    return w_full


def validate_sto3g(grid_pts, w_inner, outer_xy, outer_w,
                   zeta_1s=1.24, zeta_2p=1.72):
    """
    Validate weights against STO-3G orbital densities (not in training set).

    Returns
    -------
    dict with keys: err_1s, err_2p, exact_1s, exact_2p, num_1s, num_2p
    """
    r2_inner = grid_pts[:, 0]**2 + grid_pts[:, 1]**2
    rho_1s_inner = gaussian_1s(r2_inner, STO3G_1S, zeta_1s)**2
    rho_2p_inner = gaussian_2p(grid_pts[:, 0], grid_pts[:, 1], STO3G_2P_C, zeta_2p)**2

    r2_outer = outer_xy[:, 0]**2 + outer_xy[:, 1]**2
    rho_1s_outer = gaussian_1s(r2_outer, STO3G_1S, zeta_1s)**2
    rho_2p_outer = gaussian_2p(outer_xy[:, 0], outer_xy[:, 1], STO3G_2P_C, zeta_2p)**2

    exact_1s = analytic_1s_norm_sq(STO3G_1S, zeta_1s)
    exact_2p = analytic_2p_norm_sq(STO3G_2P_C, zeta_2p)

    num_1s = np.sum(rho_1s_inner * w_inner) + np.sum(rho_1s_outer * outer_w)
    num_2p = np.sum(rho_2p_inner * w_inner) + np.sum(rho_2p_outer * outer_w)

    return dict(
        err_1s=abs(num_1s - exact_1s) / exact_1s * 100,
        err_2p=abs(num_2p - exact_2p) / exact_2p * 100,
        exact_1s=exact_1s, exact_2p=exact_2p,
        num_1s=num_1s, num_2p=num_2p,
    )


def select_best_lambda(results, w0_orbit, min_w=-0.1, max_w=1.0):
    """Select best λ preferring stable weights (w_min > min_w, w_max < max_w)."""
    best_lam = None
    best_err = float('inf')
    for lam, r in results.items():
        w_o = r['w_orbit']
        if w_o.min() > min_w and w_o.max() < max_w:
            if r['mean_err'] < best_err:
                best_err = r['mean_err']
                best_lam = lam
    if best_lam is None:
        # fallback: just pick lowest mean_err
        for lam, r in results.items():
            if r['mean_err'] < best_err:
                best_err = r['mean_err']
                best_lam = lam
    return best_lam, best_err
