"""
Core iterative solver methods for spectral filtering and eigenvalue computation.
"""

import numpy as np


def generate_test_matrix(n: int):
    """1D Laplacian, scaled to [-0.95, 0.95]"""
    diag = 2.0 * np.ones(n)
    off_diag = -1.0 * np.ones(n - 1)
    H = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    H = H / 2.0 - np.eye(n)
    H *= 0.95
    return H


def get_exact_eigenvalues(H: np.ndarray):
    return np.sort(np.linalg.eigvalsh(H))


def jackson_kernel(m: int, N: int):
    """Jackson damping coefficient to eliminate Gibbs oscillations."""
    if N == 0: return 1.0
    M = N + 1
    term1 = (M - m) * np.cos(np.pi * m / M)
    term2 = np.sin(np.pi * m / M) / np.tan(np.pi / M)
    return (term1 + term2) / M


def cheb_rect_coeffs(f_lo, f_hi, deg, use_jackson=True):
    """
    Chebyshev polynomial coefficients for band-pass filter [f_lo, f_hi].
    Approximates a rectangle function using the Fourier-Chebyshev expansion.
    """
    f_lo = float(np.clip(f_lo, -1.0, 1.0))
    f_hi = float(np.clip(f_hi, -1.0, 1.0))
    if f_lo > f_hi: f_lo, f_hi = f_hi, f_lo
    th_lo = np.arccos(np.clip(f_lo, -1+1e-12, 1-1e-12))
    th_hi = np.arccos(np.clip(f_hi, -1+1e-12, 1-1e-12))
    c = np.zeros(deg + 1, dtype=float)
    c[0] = (th_lo - th_hi) / np.pi
    m = np.arange(1, deg + 1)
    c[1:] = (2.0 / np.pi) * (np.sin(m * th_hi) - np.sin(m * th_lo)) / m
    if use_jackson:
        for i in range(1, deg + 1):
            c[i] *= jackson_kernel(i, deg)
    # Rescale so p(x_mid)≈1 inside the band
    x_mid = 0.5 * (f_lo + f_hi)
    p_mid = float(eval_cheb_series(np.array([x_mid]), c)[0])
    if abs(p_mid) > 1e-12: c /= p_mid
    return c


def eval_cheb_series(x, c):
    """Evaluate Chebyshev series with coefficients c at points x using Clenshaw's algorithm."""
    x = np.asarray(x, dtype=float)
    deg = len(c) - 1
    if deg < 0: return np.zeros_like(x)
    if deg == 0: return c[0] * np.ones_like(x)
    b_kp1 = np.zeros_like(x)
    b_kp2 = np.zeros_like(x)
    for k in range(deg, 0, -1):
        b_k = 2.0 * x * b_kp1 - b_kp2 + c[k]
        b_kp2, b_kp1 = b_kp1, b_k
    return x * b_kp1 - b_kp2 + c[0]


def apply_cheb_poly(H, V, c):
    """
    Apply Chebyshev polynomial p(H) to vectors V using three-term recurrence.
    Returns p(H) @ V.
    """
    deg = len(c) - 1
    if deg < 0: return np.zeros_like(V)
    T0 = V
    Y = c[0] * T0
    if deg == 0: return Y
    T1 = H @ V
    Y = Y + c[1] * T1
    for m in range(2, deg + 1):
        T2 = 2.0 * (H @ T1) - T0
        Y = Y + c[m] * T2
        T0, T1 = T1, T2
    return Y


def rayleigh_ritz(H, V):
    """
    Rayleigh-Ritz extraction from subspace V.
    Returns (eigenvalues w, eigenvectors U, residuals r).
    """
    Q, _ = np.linalg.qr(V)
    T = Q.T @ (H @ Q)
    w, Z = np.linalg.eigh(T)
    U = Q @ Z
    R = H @ U - U * w[None, :]
    r = np.linalg.norm(R, axis=0)
    return w, U, r


def lanczos_band(H, V0, steps):
    """
    Single-vector Lanczos starting from the first column of V0.
    Builds a Krylov subspace of size 'steps' and returns Ritz values/vectors.
    No matrix inversion needed. Cheap: one SpMV per step.
    """
    N = H.shape[0]
    Q, _ = np.linalg.qr(V0)
    q = Q[:, 0].copy()
    q = q / np.linalg.norm(q)

    Q_all = np.zeros((N, steps))
    Q_all[:, 0] = q
    alphas = np.zeros(steps)
    betas = np.zeros(steps)

    for j in range(steps):
        z = H @ q
        alpha = np.dot(q, z)
        alphas[j] = alpha
        z = z - alpha * q
        if j > 0:
            z = z - betas[j-1] * Q_all[:, j-1]
        beta = np.linalg.norm(z)
        if beta < 1e-14:
            break
        betas[j] = beta
        if j + 1 < steps:
            q = z / beta
            Q_all[:, j+1] = q

    m = j + 1
    T = np.zeros((m, m))
    for i in range(m):
        T[i, i] = alphas[i]
        if i > 0:
            T[i, i-1] = betas[i-1]
            T[i-1, i] = betas[i-1]

    w, Z = np.linalg.eigh(T)
    U = Q_all[:, :m] @ Z
    R = H @ U - U * w[None, :]
    r = np.linalg.norm(R, axis=0)
    return w, U, r


def cluster_prune_mask(V_filt, H, cluster_tol=0.85, eig_tol=0.05):
    """
    Identify redundant vectors chasing the same eigenvalue based on:
    1. Collinearity of raw filtered vectors (dot product)
    2. Eigenvalue proximity (cheap Rayleigh quotient on raw vectors)
    Returns a boolean mask where True = keep.
    """
    m = V_filt.shape[1]
    if m < 2:
        return np.ones(m, dtype=bool)

    # Normalize columns for collinearity check
    Vn = V_filt / np.linalg.norm(V_filt, axis=0, keepdims=True)

    # Cheap Rayleigh quotients on raw filtered vectors
    w_s = np.array([np.dot(Vn[:, i], H @ Vn[:, i]) for i in range(m)])

    # Sort by eigenvalue for neighbor check
    order = np.argsort(w_s)
    w_s = w_s[order]

    # Residuals estimated from Rayleigh quotient: r ≈ ||(H - w_i) v_i||
    r_s = np.array([np.linalg.norm(H @ Vn[:, order[i]] - w_s[i] * Vn[:, order[i]]) for i in range(m)])

    keep_sorted = np.ones(m, dtype=bool)

    # Check only adjacent pairs — conservative & cheap (O(m) not O(m^2))
    for i in range(m - 1):
        if not keep_sorted[i]:
            continue
        j = i + 1
        # Eigenvalue proximity check
        if abs(w_s[j] - w_s[i]) > eig_tol:
            continue
        # Collinearity check on the filtered vectors
        overlap = abs(np.dot(Vn[:, order[i]], Vn[:, order[j]]))
        if overlap > cluster_tol:
            # Same eigenvalue being chased — prune the worse one
            if r_s[i] <= r_s[j]:
                keep_sorted[j] = False
            else:
                keep_sorted[i] = False

    # Unsort back to original column ordering
    keep = np.zeros(m, dtype=bool)
    keep[order] = keep_sorted
    return keep


def cheap_rank_estimate(V, rel_thresh=0.1):
    """
    Estimate rank of V via SVD singular values.
    Returns number of singular values > rel_thresh * sigma_max.
    """
    if V.shape[1] == 0:
        return 0
    s = np.linalg.svd(V, compute_uv=False)
    if len(s) == 0:
        return 0
    sigma_max = s[0]
    if sigma_max < 1e-14:
        return 0
    return int(np.sum(s > rel_thresh * sigma_max))


def match_trajectories(w_in, prev_w, prev_ids, final_traj, bi):
    """
    Greedy trajectory matching by nearest eigenvalue.
    Assigns trajectory IDs to current Ritz values based on previous iteration.
    
    Returns curr_ids (array of trajectory IDs for each w_in).
    """
    curr_ids = np.full(len(w_in), -1, dtype=int)
    used = set()
    if prev_w is not None and len(prev_w) > 0:
        for j, wc in enumerate(w_in):
            best = None; best_d = 1e9
            for k, wp in enumerate(prev_w):
                if k in used: continue
                d = abs(wc - wp)
                if d < best_d:
                    best_d = d; best = k
            if best is not None and best_d < 0.15:
                curr_ids[j] = prev_ids[best]
                used.add(best)
    next_id = int(max(final_traj[bi].keys(), default=-1)) + 1
    for j in range(len(w_in)):
        if curr_ids[j] < 0:
            curr_ids[j] = next_id
            next_id += 1
    return curr_ids


def apply_pruning(r_in, curr_ids, final_traj, bi, it, prune_mode, prune_tol, 
                  cluster_tol, cluster_eig_tol, gradual_factor, V_filt, H, mask, order):
    """
    Apply pruning strategy to determine which vectors to keep.
    
    Returns keep (boolean mask for r_in).
    """
    keep = np.ones(len(r_in), dtype=bool)
    
    if prune_mode == 'cluster' and it >= 2:
        cluster_keep = cluster_prune_mask(
            V_filt, H,
            cluster_tol=cluster_tol,
            eig_tol=cluster_eig_tol
        )
        keep = keep & cluster_keep[mask][order]
        
    elif prune_mode == 'residual' and it >= 2 and prune_tol > 0:
        keep = keep & (r_in < prune_tol)
        
    elif prune_mode == 'gradual' and it >= 2 and prune_tol > 0:
        # Drop only the single worst vector if it is:
        #   1. Above prune_tol
        #   2. Significantly worse than median (outlier)
        #   3. NOT improving (residual stagnating or growing)
        if len(r_in) > 1:
            worst = np.argmax(r_in)
            median_r = float(np.median(r_in))
            is_outlier = r_in[worst] > gradual_factor * max(median_r, prune_tol * 0.1)
            # Improvement check: compare to previous residual of same trajectory
            tid = curr_ids[worst]
            prev_r = None
            if tid in final_traj[bi] and len(final_traj[bi][tid]) > 0:
                prev_r = final_traj[bi][tid][-1][2]
            is_improving = False
            if prev_r is not None and prev_r > 0:
                is_improving = r_in[worst] < 0.9 * prev_r  # at least 10% drop
            if r_in[worst] > prune_tol and is_outlier and not is_improving:
                keep[worst] = False
                
    elif prune_mode == 'hybrid' and it >= 2:
        # Step 1: cluster prune (remove redundant pairs)
        if cluster_tol > 0:
            cluster_keep = cluster_prune_mask(
                V_filt, H,
                cluster_tol=cluster_tol,
                eig_tol=cluster_eig_tol
            )
            keep = keep & cluster_keep[mask][order]
        # Step 2: if still too many, drop the worst above tol
        if prune_tol > 0 and keep.sum() > 1:
            r_masked = r_in.copy()
            r_masked[~keep] = -1.0
            worst = np.argmax(r_masked)
            if r_masked[worst] > prune_tol:
                keep[worst] = False
    
    return keep


def solve_spectrum(H, band_lo, band_hi, nvec, coarse_iters, conv_iters,
                   filter_reps, square_filter, method, prune_mode,
                   prune_tol, cluster_tol, cluster_eig_tol, gradual_factor,
                   rank_est=False, res_tol=1e-6):
    """
    Band-by-band spectral solver: Chebyshev filter + Ritz/Lanczos iteration
    with trajectory tracking and pruning.

    Parameters
    ----------
    H : (N, N) array - Hamiltonian matrix
    band_lo, band_hi : (nbands,) arrays - band edges
    nvec : int - initial probe vectors per band
    coarse_iters : int - Chebyshev degree
    conv_iters : int - number of filter->QR->Ritz refinement iterations
    filter_reps : int - filter applications per subspace iteration
    square_filter : bool - use p(H)^2
    method : 'ritz' or 'lanczos'
    prune_mode : 'none', 'residual', 'cluster', 'gradual', 'hybrid'
    prune_tol, cluster_tol, cluster_eig_tol, gradual_factor : pruning parameters
    rank_est : bool - print cheap rank estimate
    res_tol : float - residual tolerance for convergence report

    Returns
    -------
    dict with keys:
        'exact_w': exact eigenvalues (sorted)
        'band_raw_w': {bi: [w_in_it0, w_in_it1, ...]}
        'band_raw_r': {bi: [r_in_it0, r_in_it1, ...]}
        'band_keep': {bi: [keep_it0, keep_it1, ...]}
        'final_traj': {bi: {tid: [(it, w, r, kept), ...]}}
        'total_spmv': int
        'spmv_per_band': {bi: int}
    """
    exact_w = np.sort(np.linalg.eigvalsh(H))
    nbands = len(band_lo)

    band_raw_w = {bi: [] for bi in range(nbands)}
    band_raw_r = {bi: [] for bi in range(nbands)}
    band_keep  = {bi: [] for bi in range(nbands)}
    final_traj = {bi: {} for bi in range(nbands)}

    total_spmv = 0
    spmv_per_band = {}

    for bi in range(nbands):
        f_lo, f_hi = float(band_lo[bi]), float(band_hi[bi])
        c = cheb_rect_coeffs(f_lo, f_hi, coarse_iters, use_jackson=True)

        n_eig_est = max(2, int(nvec * 2))
        V = np.random.randn(H.shape[0], n_eig_est)

        prev_ids = None
        prev_w   = None
        for it in range(max(1, int(conv_iters))):
            n_reps = filter_reps if it > 0 else max(1, filter_reps)
            V_filt = V.copy()
            k_curr = V.shape[1]
            spmv_per_app = coarse_iters * (2 if square_filter else 1)
            for _ in range(n_reps):
                V_filt = apply_cheb_poly(H, V_filt, c)
                if square_filter:
                    V_filt = apply_cheb_poly(H, V_filt, c)
            spmv_this = n_reps * spmv_per_app * k_curr
            total_spmv += spmv_this
            spmv_per_band[bi] = spmv_per_band.get(bi, 0) + spmv_this

            if it == 0 and rank_est:
                rank_est_val = cheap_rank_estimate(V_filt, rel_thresh=0.1)
                true_count = int(np.sum((exact_w >= f_lo) & (exact_w <= f_hi)))
                print(f"  band={bi}  rank_est={rank_est_val}  true_eigs={true_count}")

            V_qr, _ = np.linalg.qr(V_filt)

            if method == 'ritz':
                w, U, r = rayleigh_ritz(H, V_qr)
                spmv_r = k_curr
                total_spmv += spmv_r
                spmv_per_band[bi] = spmv_per_band.get(bi, 0) + spmv_r
            else:  # lanczos
                steps = max(2, V.shape[1])
                w, U, r = lanczos_band(H, V_qr, steps=steps)
                spmv_r = steps + steps
                total_spmv += spmv_r
                spmv_per_band[bi] = spmv_per_band.get(bi, 0) + spmv_r

            mask = (w >= f_lo) & (w <= f_hi)
            w_in = w[mask]
            r_in = r[mask]
            order = np.argsort(w_in)
            w_in = w_in[order]
            r_in = r_in[order]

            curr_ids = match_trajectories(w_in, prev_w, prev_ids, final_traj, bi)

            keep = apply_pruning(r_in, curr_ids, final_traj, bi, it, prune_mode,
                                 prune_tol, cluster_tol, cluster_eig_tol,
                                 gradual_factor, V_filt, H, mask, order)

            for j in range(len(w_in)):
                tid = curr_ids[j]
                if tid not in final_traj[bi]:
                    final_traj[bi][tid] = []
                final_traj[bi][tid].append((it, float(w_in[j]), float(r_in[j]), bool(keep[j])))

            band_raw_w[bi].append(w_in.copy())
            band_raw_r[bi].append(r_in.copy())
            band_keep[bi].append(keep.copy())

            if keep.sum() > 0:
                V = U[:, mask][:, order[keep]]
                prev_ids = curr_ids[keep]
                prev_w   = w_in[keep]
            else:
                best = np.argmin(r_in)
                V = U[:, mask][:, order[best:best+1]]
                prev_ids = curr_ids[best:best+1]
                prev_w   = w_in[best:best+1]

    return {
        'exact_w': exact_w,
        'band_raw_w': band_raw_w,
        'band_raw_r': band_raw_r,
        'band_keep': band_keep,
        'final_traj': final_traj,
        'total_spmv': total_spmv,
        'spmv_per_band': spmv_per_band,
    }


def solve_band(H, V0, f_lo, f_hi, coarse_iters, solve_iters, square_filter):
    """
    Extract eigenpairs in a single band by repeated Chebyshev filter + Rayleigh-Ritz.

    Parameters
    ----------
    H : (N, N) array - Hamiltonian matrix
    V0 : (N, k) array - initial probe vectors
    f_lo, f_hi : float - band edges
    coarse_iters : int - Chebyshev degree
    solve_iters : int - number of filter->QR->Ritz rounds
    square_filter : bool - use p(H)^2

    Returns
    -------
    dict with keys:
        'w_in': eigenvalues inside band
        'r_in': residuals inside band
        'solve_spmv': SpMV count
        'k': number of probe vectors
    """
    c = cheb_rect_coeffs(f_lo, f_hi, coarse_iters, use_jackson=True)
    V = V0.copy()
    solve_spmv = 0
    k = V.shape[1]
    spmv_per_app = coarse_iters * (2 if square_filter else 1)
    for _ in range(max(1, int(solve_iters))):
        V = apply_cheb_poly(H, V, c)
        solve_spmv += spmv_per_app * k
        if square_filter:
            V = apply_cheb_poly(H, V, c)
        V, _ = np.linalg.qr(V)
    w, U, r = rayleigh_ritz(H, V)
    solve_spmv += k  # H @ Q in rayleigh_ritz
    mask = (w >= f_lo) & (w <= f_hi)
    w_in = w[mask]
    r_in = r[mask]
    return {
        'w_in': w_in,
        'r_in': r_in,
        'solve_spmv': solve_spmv,
        'k': k,
    }


def chebyshev_filter(H, V0, freqs, iters, use_jackson=True):
    """
    KPM Chebyshev filter: builds basis once, evaluates at multiple frequencies.
    Uses Jackson damping to eliminate Gibbs oscillations.
    
    Parameters
    ----------
    H : (N, N) array - Hamiltonian matrix
    V0 : (N, k) array - initial probe vectors
    freqs : (nfreq,) array - frequency points in [-1, 1]
    iters : list of int - iteration counts to evaluate
    use_jackson : bool - apply Jackson damping
    
    Returns
    -------
    total_amps : {n: (nfreq,) array} - total amplitude per frequency
    vec_amps : {n: (nfreq, k) array} - per-vector amplitudes
    """
    max_iter = max(iters)
    nfreq = len(freqs)
    N, k = V0.shape

    # 1. Frequency-independent step: Build the Chebyshev basis
    basis = np.zeros((max_iter + 1, N, k))
    basis[0] = V0
    if max_iter > 0:
        basis[1] = H @ V0
    for m in range(2, max_iter + 1):
        basis[m] = 2.0 * (H @ basis[m-1]) - basis[m-2]

    # 2. Frequency-dependent step: Cheap scalar accumulation
    theta = np.arccos(freqs)
    total_amps = {}
    vec_amps = {}

    for n in iters:
        v_amps_n = np.zeros((nfreq, k))
        for i, f in enumerate(freqs):
            th = theta[i]
            V_f = np.copy(basis[0])
            for m in range(1, n + 1):
                weight = 2.0 * np.cos(m * th)
                if use_jackson:
                    weight *= jackson_kernel(m, n)
                V_f += weight * basis[m]
            v_amps_n[i, :] = np.linalg.norm(V_f, axis=0)

        vec_amps[n] = v_amps_n
        total_amps[n] = np.sum(v_amps_n, axis=1)

    return total_amps, vec_amps


def power_iteration_filter(H, V0, freqs, iters):
    """
    Power iteration filter: evaluate (fI-H)^n V0 norm at each frequency.
    
    Parameters
    ----------
    H : (N, N) array - Hamiltonian matrix
    V0 : (N, k) array - initial probe vectors
    freqs : (nfreq,) array - frequency points
    iters : list of int - iteration counts to evaluate
    
    Returns
    -------
    total_amps : {n: (nfreq,) array}
    vec_amps : {n: (nfreq, k) array}
    """
    max_iter = max(iters)
    nfreq = len(freqs)
    N, k = V0.shape

    total_amps = {}
    vec_amps = {n: np.zeros((nfreq, k)) for n in iters}

    for i, f in enumerate(freqs):
        V = np.copy(V0)
        A = f * np.eye(N) - H
        for n in range(1, max_iter + 1):
            V = A @ V
            if n in iters:
                norms = np.linalg.norm(V, axis=0)
                vec_amps[n][i, :] = norms

    for n in iters:
        total_amps[n] = np.sum(vec_amps[n], axis=1)

    return total_amps, vec_amps
