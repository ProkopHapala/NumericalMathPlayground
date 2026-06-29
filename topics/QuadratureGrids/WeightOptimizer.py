"""
WeightOptimizer.py — Quadrature weight optimization for CubeEmbededAtoms grid.

Core module: geometric weight computation, test function matrix assembly,
Tikhonov regularized least-squares optimization, validation against STO-nG.
No I/O, no plotting.
"""
import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import nnls, lsq_linear

from CubeEmbededGrid import (
    quad_area, collect_unique_points, build_grid, build_cartesian_outer,
    build_cartesian_fixed,
)
from Symmetry import group_orbits
from BasisFunctions import (
    eval_test_func, build_test_function_set,
    gaussian_1s, gaussian_2p,
    analytic_1s_norm_sq, analytic_2p_norm_sq,
    STO3G_1S, STO3G_2P_C,
    eval_trial_functions, trial_analytic_integrals,
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


def nn_tikhonov_optimize(A, b, w0_orbit, lambdas, q=None):
    """Non-negative Tikhonov-regularized least-squares.

    min ||diag(q) (A @ w - b)||² + λ ||w - w0||²   s.t. w >= 0

    Uses scipy.optimize.lsq_linear with bounds [0, inf).

    Returns
    -------
    results : dict λ -> {w_orbit, errs, max_err, mean_err, predicted}
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

        res = lsq_linear(A_aug, b_aug, bounds=(0, np.inf),
                         method='bvls', max_iter=200)
        w_orbit = res.x

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


def select_best_lambda(results, w0_orbit, min_w=-0.1, max_w=1.0,
                       A=None, b=None, q=None, val_ratio=0.2, rng=None,
                       fixed_lambda=None, non_negative=False):
    """Select best λ using train/validation split.

    Splits the trial-function equations (rows 1..N) into train/validation.
    Row 0 (area constraint) is ALWAYS kept in training.
    For each λ, re-optimizes on train only, then evaluates on validation.
    Picks λ with lowest validation mean_err.

    If fixed_lambda is given, skips the split and uses that λ directly.

    Parameters
    ----------
    results : dict λ -> {w_orbit, errs, max_err, mean_err, predicted}
    w0_orbit : (N_orbits,) — geometric weights
    A, b, q : the full system (needed for train/val split)
    val_ratio : float — fraction of trial equations for validation
    rng : np.random.Generator or None (default: fixed seed 42 for reproducibility)
    fixed_lambda : float or None — if given, use this λ directly (no selection)
    """
    if fixed_lambda is not None:
        # Use fixed lambda for weights, but still compute validation error
        # to detect overfitting
        if fixed_lambda not in results:
            fixed_lambda = min(results.keys(), key=lambda l: abs(l - fixed_lambda))

        # Run train/val split for validation error reporting
        Ntest, N_orbits = A.shape
        if q is None:
            q = np.ones(Ntest)
        if rng is None:
            rng = np.random.default_rng(42)

        n_trial = Ntest - 1
        n_val = max(1, int(n_trial * val_ratio))
        idx = rng.permutation(n_trial)
        val_idx = idx[:n_val] + 1
        train_idx = idx[n_val:] + 1
        train_rows = np.concatenate([[0], train_idx])

        A_train = A[train_rows] * q[train_rows, None]
        b_train = b[train_rows] * q[train_rows]
        A_val = A[val_idx]
        b_val = b[val_idx]

        A_aug = np.vstack([A_train, np.sqrt(fixed_lambda) * np.eye(N_orbits)])
        b_aug = np.concatenate([b_train, np.sqrt(fixed_lambda) * w0_orbit])
        if non_negative:
            res = lsq_linear(A_aug, b_aug, bounds=(0, np.inf),
                             method='bvls', max_iter=200)
            w_train = res.x
        else:
            w_train, _, _, _ = lstsq(A_aug, b_aug)

        pred_val = A_val @ w_train
        nonzero_val = np.abs(b_val) > 1e-10
        if np.any(nonzero_val):
            val_errs = np.abs(pred_val[nonzero_val] - b_val[nonzero_val]) / np.abs(b_val[nonzero_val]) * 100
            val_mean = np.mean(val_errs)
            val_max = np.max(val_errs)
        else:
            val_mean = 0.0
            val_max = 0.0

        val_results = {fixed_lambda: {
            'w_orbit': w_train,
            'val_mean_err': val_mean,
            'val_max_err': val_max,
        }}
        return fixed_lambda, results[fixed_lambda]['mean_err'], val_results

    if A is None or b is None:
        # Fallback: old behavior
        best_lam = None
        best_err = float('inf')
        for lam, r in results.items():
            w_o = r['w_orbit']
            if w_o.min() > min_w and w_o.max() < max_w:
                if r['mean_err'] < best_err:
                    best_err = r['mean_err']
                    best_lam = lam
        if best_lam is None:
            for lam, r in results.items():
                if r['mean_err'] < best_err:
                    best_err = r['mean_err']
                    best_lam = lam
        return best_lam, best_err, {}

    Ntest, N_orbits = A.shape
    if q is None:
        q = np.ones(Ntest)

    if rng is None:
        rng = np.random.default_rng(42)  # FIXED seed — same split for all configs

    # Split: row 0 (area constraint) always in training
    # Rows 1..Ntest-1 are trial equations, split those
    n_trial = Ntest - 1
    n_val = max(1, int(n_trial * val_ratio))
    idx = rng.permutation(n_trial)  # indices 0..n_trial-1 → shift by +1
    val_idx = idx[:n_val] + 1      # +1 to skip area constraint row
    train_idx = idx[n_val:] + 1

    # Training set: area constraint (row 0) + train trial equations
    train_rows = np.concatenate([[0], train_idx])
    val_rows = val_idx

    A_train = A[train_rows]
    b_train = b[train_rows]
    q_train = q[train_rows]
    A_val = A[val_rows]
    b_val = b[val_rows]

    A_w_train = A_train * q_train[:, None]
    b_w_train = b_train * q_train

    best_lam = None
    best_val_err = float('inf')
    val_results = {}

    for lam in results:
        # Re-optimize on train only
        A_aug = np.vstack([A_w_train, np.sqrt(lam) * np.eye(N_orbits)])
        b_aug = np.concatenate([b_w_train, np.sqrt(lam) * w0_orbit])
        if non_negative:
            res = lsq_linear(A_aug, b_aug, bounds=(0, np.inf),
                             method='bvls', max_iter=200)
            w_orbit = res.x
        else:
            w_orbit, _, _, _ = lstsq(A_aug, b_aug)

        # Evaluate on validation
        pred_val = A_val @ w_orbit
        nonzero_val = np.abs(b_val) > 1e-10
        if np.any(nonzero_val):
            val_errs = np.abs(pred_val[nonzero_val] - b_val[nonzero_val]) / np.abs(b_val[nonzero_val]) * 100
            val_mean = np.mean(val_errs)
            val_max = np.max(val_errs)
        else:
            val_mean = 0.0
            val_max = 0.0

        val_results[lam] = {
            'w_orbit': w_orbit,
            'val_mean_err': val_mean,
            'val_max_err': val_max,
        }

        if val_mean < best_val_err:
            best_val_err = val_mean
            best_lam = lam

    return best_lam, best_val_err, val_results


# ── Trial-based optimization (using precomputed function values) ──────────────

def build_trial_matrix(F_sq, orbit_members):
    """Build A[k, orbit] = Σ_{p in orbit} F_sq[k, p] from precomputed values.

    Parameters
    ----------
    F_sq : (n_combos, Npts) — squared trial function values at grid points
    orbit_members : list of lists — point indices per orbit

    Returns
    -------
    A : (n_combos, n_orbits) array
    """
    n_combos, Npts = F_sq.shape
    n_orbits = len(orbit_members)
    A = np.zeros((n_combos, n_orbits))
    for oid, members in enumerate(orbit_members):
        A[:, oid] = F_sq[:, members].sum(axis=1)
    return A


def compute_trial_targets(F_sq_outer, outer_w, I_total):
    """Compute b_inner = I_total - I_outer for trial functions.

    Parameters
    ----------
    F_sq_outer : (n_combos, N_outer) — squared trial function values at outer points
    outer_w : (N_outer,) — outer grid weights
    I_total : (n_combos,) — analytic integrals over all space

    Returns
    -------
    b_inner : (n_combos,) — target for inner grid
    I_outer : (n_combos,) — numerical integral over outer grid
    """
    I_outer = F_sq_outer @ outer_w
    b_inner = I_total - I_outer
    return b_inner, I_outer


def compute_R_cut(trial_set, threshold=1e-10):
    """Compute R_cut — the radius beyond which all trial functions are negligible.

    Finds the smallest Gaussian exponent across all basis functions, then
    determines R where exp(-2*alpha_min*R²) < threshold (factor 2 because
    trial functions are squared).

    Returns
    -------
    R_cut : float — safe cutoff radius
    """
    alpha_min = float('inf')
    for desc in trial_set['basis_descs']:
        a_min = desc['exps'].min()
        if a_min < alpha_min:
            alpha_min = a_min
    # Trial functions are squared: f ~ exp(-2*alpha*r²)
    R_cut = np.sqrt(-np.log(threshold) / (2 * alpha_min))
    return R_cut


def precompute_outer_cache(d, h, trial_set, R_cut=None, margin=5):
    """Precompute outer Cartesian grid and trial function values.

    The outer grid covers a FIXED [-R_cut, R_cut]² area (independent of d),
    with step size d. Points inside [-h, h]² are removed (replaced by inner grid).
    This ensures the embedding always covers the full support of trial functions.

    Parameters
    ----------
    d : float — grid step
    h : float — cutout half-size (inner grid boundary)
    trial_set : dict from build_atomic_trial_set
    R_cut : float or None — if None, auto-computed from trial_set
    margin : int — unused (kept for backward compat)

    Returns
    -------
    dict with keys: outer_xy, outer_w, F_sq_outer, I_total, d, h, R_cut
    """
    if R_cut is None:
        R_cut = compute_R_cut(trial_set)
    outer_xy, outer_w = build_cartesian_fixed(R_cut, d, h_cut=h)
    F_sq_outer = eval_trial_functions(outer_xy, trial_set)
    I_total = trial_analytic_integrals(trial_set)
    return dict(outer_xy=outer_xy, outer_w=outer_w,
                F_sq_outer=F_sq_outer, I_total=I_total,
                d=d, h=h, R_cut=R_cut)


def optimize_single_config(d, n, alpha, n_blend, trial_set, lambdas, margin=5,
                           outer_cache=None, fixed_lambda=None, non_negative=False):
    """Full optimization pipeline for a single grid configuration.

    Parameters
    ----------
    outer_cache : dict or None — precomputed outer grid data with keys:
        'outer_xy', 'outer_w', 'F_sq_outer', 'I_total'
        If None, these are computed from scratch.
    fixed_lambda : float or None — if given, use this λ for all configs
        (no train/val selection). If None, use train/val split.
    non_negative : bool — if True, enforce w >= 0 using lsq_linear.

    Returns dict with all results including errors and grid info.
    """
    grid_dict = build_grid(d=d, n=n, alpha=alpha, n_blend=n_blend)
    p = grid_dict['params']
    h = p['h']

    w0, grid_pts, grid_idx = geometric_weights(grid_dict)
    Npts = len(grid_pts)

    orbit_id, orbit_members, orbit_rep = group_orbits(grid_pts)
    N_orbits = len(orbit_rep)
    w0_orbit = np.array([np.mean(w0[m]) for m in orbit_members])

    # Use precomputed outer grid if available, otherwise compute
    if outer_cache is not None:
        outer_xy = outer_cache['outer_xy']
        outer_w = outer_cache['outer_w']
        F_sq_outer = outer_cache['F_sq_outer']
        I_total = outer_cache['I_total']
    else:
        R_cut = compute_R_cut(trial_set)
        outer_xy, outer_w = build_cartesian_fixed(R_cut, d, h_cut=h)
        F_sq_outer = eval_trial_functions(outer_xy, trial_set)
        I_total = trial_analytic_integrals(trial_set)

    # Evaluate trial functions at inner grid points
    F_sq = eval_trial_functions(grid_pts, trial_set)

    # Targets: b_inner = I_total - I_outer (embedding grid subtraction)
    b_inner, I_outer = compute_trial_targets(F_sq_outer, outer_w, I_total)

    # Build A matrix
    A = build_trial_matrix(F_sq, orbit_members)

    # Add area constraint as first equation: Σ w_orbit * orbit_size = 4h²
    orbit_sizes = np.array([len(m) for m in orbit_members])
    A = np.vstack([orbit_sizes.reshape(1, -1), A])
    b_inner = np.concatenate([[4 * h * h], b_inner])

    # Equation weights: area constraint gets high weight
    q = np.ones(len(b_inner))
    q[0] = 100.0

    # Optimize
    if non_negative:
        results = nn_tikhonov_optimize(A, b_inner, w0_orbit, lambdas, q=q)
    else:
        results = tikhonov_optimize(A, b_inner, w0_orbit, lambdas, q=q)

    # Select best lambda using train/validation split (or fixed lambda)
    n_eq = A.shape[0] - 1  # exclude area constraint row
    best_lam, best_err, val_results = select_best_lambda(
        results, w0_orbit, A=A, b=b_inner, q=q, fixed_lambda=fixed_lambda,
        non_negative=non_negative)

    # Per-function errors at best lambda (on FULL system, not train split)
    w_opt = expand_orbit_weights(results[best_lam]['w_orbit'], orbit_members, Npts)
    pred = A @ results[best_lam]['w_orbit']
    nonzero = np.abs(b_inner) > 1e-10
    errs_inner = np.abs(pred - b_inner) / (np.abs(b_inner) + 1e-30) * 100

    # Total integral error: (I_inner + I_outer - I_total) / I_total
    I_inner_opt = pred[1:]
    I_total_errs = np.abs(I_inner_opt + I_outer - I_total) / (np.abs(I_total) + 1e-30) * 100

    # Extract validation error (always computed now)
    val_info = list(val_results.values())[0] if val_results else {}
    val_mean = val_info.get('val_mean_err', 0.0)
    val_max = val_info.get('val_max_err', 0.0)

    return dict(
        d=d, n=n, alpha=alpha, n_blend=n_blend,
        grid_dict=grid_dict, grid_pts=grid_pts,
        Npts=Npts, N_orbits=N_orbits, N_eq=n_eq,
        w0=w0, w0_orbit=w0_orbit, w_opt=w_opt,
        best_lam=best_lam, best_err=best_err,
        results=results, val_results=val_results,
        A=A, b_inner=b_inner,
        errs=errs_inner, pred=pred,
        F_sq=F_sq, outer_xy=outer_xy, outer_w=outer_w,
        orbit_members=orbit_members,
        I_outer=I_outer, I_total=I_total,
        total_errs=I_total_errs,
        mean_err=np.mean(I_total_errs),
        max_err=np.max(I_total_errs),
        max_err_idx=np.argmax(I_total_errs),
        val_mean_err=val_mean,
        val_max_err=val_max,
    )


def baseline_cartesian(d, trial_set, R_cut=None):
    """Pure Cartesian grid baseline — no inner insert, just uniform d² weights.

    This shows how error decreases with finer d on a plain Cartesian grid
    covering [-R_cut, R_cut]². The optimized inner grid should beat this.

    Returns
    -------
    dict with mean_err, max_err, Npts, d, errs, grid_xy, grid_w, F_sq
    """
    if R_cut is None:
        R_cut = compute_R_cut(trial_set)

    xy, w = build_cartesian_fixed(R_cut, d, h_cut=None)
    F_sq = eval_trial_functions(xy, trial_set)
    I_total = trial_analytic_integrals(trial_set)

    # Numerical integrals: I_num = F_sq @ w
    I_num = F_sq @ w
    errs = np.abs(I_num - I_total) / (np.abs(I_total) + 1e-30) * 100

    return dict(
        d=d, R_cut=R_cut, Npts=len(xy),
        grid_xy=xy, grid_w=w, F_sq=F_sq,
        I_total=I_total, I_num=I_num,
        errs=errs,
        mean_err=np.mean(errs),
        max_err=np.max(errs),
        max_err_idx=np.argmax(errs),
    )
