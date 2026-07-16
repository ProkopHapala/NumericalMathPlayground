#!/usr/bin/env python3
"""Fit a BRBFFF stiffness in one coordinate system: ``eta = S log(D1^-1 D2)``.

Design:
- Every reported 6x6 stiffness acts on the scaled relative twist ``eta``; QR bases are
  used only for rank diagnostics, never as interchangeable physical coordinates.
- Nonlinear sample forces are obtained from the exact Jacobian transpose of the
  sampled reconstruction, evaluated by finite differences of geometry only.
- Galerkin samples are clamped to the frame manifold; ``Relaxed`` is the separate
  harmonic static-condensation model.

Open issues:
- Finite-amplitude constrained atomistic relaxation is not implemented here.
- The UFF evaluator is float32, so Hessian and stationarity diagnostics have a noise
  floor that must be checked before trusting inverse-Hessian condensation.
- The fitted cubic/quartic potential is local.  Before use for reduced relaxation it
  is radially stabilized from a sampled quartic-direction scan and minimized only in
  its fitted coordinate box; a boundary hit is an out-of-domain result, not a valid
  mechanical prediction.
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import eigvalsh

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from py.AtomicSystem import AtomicSystem
from py.FFs.Vibrations import (
    hessian_from_ff, atomic_masses, rigid_body_basis, project_rigid_modes,
    hessian_to_modes, ReducedPolynomialPotential, fit_reduced_polynomial,
    stabilize_reduced_quartic_sampled, relax_reduced_potential,
    relax_reduced_load_path, reduced_potential_hessian, exp_se3,
    reconstruct_blended_frame_positions, BlendedFrameDynamics,
    project_atomic_forces_to_frames,
)
from py.FFs.UFF_cl import make_ff_eval_fn

# ============================================================
# Core geometry
# ============================================================

def rigid_basis_geometric(pos):
    N = pos.shape[0]; com = pos.mean(axis=0)
    G = np.zeros((3*N, 6))
    for i in range(N):
        ri = pos[i] - com
        G[3*i:3*i+3, 0] = [1,0,0]; G[3*i:3*i+3, 1] = [0,1,0]; G[3*i:3*i+3, 2] = [0,0,1]
        G[3*i:3*i+3, 3] = np.cross([1,0,0], ri)
        G[3*i:3*i+3, 4] = np.cross([0,1,0], ri)
        G[3*i:3*i+3, 5] = np.cross([0,0,1], ri)
    Q, _ = np.linalg.qr(G)
    return Q

def build_frame_basis(pos, centers, weights):
    """Build frame basis B (3N x 6K) using ABSOLUTE positions for rotation.
    This ensures rigid body invariance: when all frames have same twist ξ,
    δpos = (sum_a w[i,a]) * (v + ω×pos[i]) = v + ω×pos[i]  (rigid body).
    Centers are only used for weight interpolation, not for rotation arm."""
    N, K = pos.shape[0], centers.shape[0]
    B = np.zeros((3*N, 6*K))
    for a in range(K):
        for i in range(N):
            r = pos[i]  # ABSOLUTE position (not pos - center)
            skew = np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])
            B[3*i:3*i+3, 6*a:6*a+6] = weights[i,a] * np.hstack([np.eye(3), -skew])
    return B

def build_internal_prolongation(B, pos, ell, masses=None, bMassOrthogonal=False):
    """Map scaled relative twist ``eta`` to Cartesian displacement.

    The symmetric lift ``q1=-xi/2, q2=+xi/2`` has exactly ``q2-q1=xi``.
    Subtracting a common rigid displacement only fixes the global-motion gauge and
    therefore leaves the relative twist unchanged.
    """
    if B.shape[1] != 12:
        raise ValueError(f"Two-frame prolongation requires B.shape[1] == 12, got {B.shape}")
    S_inv = np.diag([1.0, 1.0, 1.0, 1.0/ell, 1.0/ell, 1.0/ell])
    P_eta = 0.5 * (B[:, 6:12] - B[:, 0:6]) @ S_inv
    if bMassOrthogonal:
        if masses is None:
            raise ValueError("masses are required for mass-orthogonal projection")
        sqrt_m = np.repeat(np.sqrt(masses), 3)
        Q_mw = rigid_body_basis(pos, masses)
        P_mw = np.eye(B.shape[0]) - Q_mw @ Q_mw.T
        return (P_mw @ (sqrt_m[:, None] * P_eta)) / sqrt_m[:, None]
    G = rigid_basis_geometric(pos)
    return P_eta - G @ (G.T @ P_eta)

# ============================================================
# Method 1: Galerkin
# ============================================================

def galerkin_stiffness(H, P_eta):
    K = P_eta.T @ H @ P_eta
    return 0.5 * (K + K.T)

# ============================================================
# Method 2: Spectral (dynamics only)
# ============================================================

def spectral_stiffness(P_eta, masses, target_modes, target_eigenvalues, rcond=1e-10):
    """Build a spectrally calibrated ``K_eta`` and report representable rank."""
    mass3 = np.repeat(masses, 3)
    M_eta = P_eta.T @ (mass3[:, None] * P_eta)
    A = np.linalg.solve(M_eta, P_eta.T @ (mass3[:, None] * target_modes))
    gram = 0.5 * (A.T @ M_eta @ A + (A.T @ M_eta @ A).T)
    w, V = np.linalg.eigh(gram)
    tol = max(w.max(), 1.0) * rcond
    keep = w > tol
    gram_inv_sqrt = (V[:, keep] / np.sqrt(w[keep])) @ V[:, keep].T
    Z = A @ gram_inv_sqrt
    K_spec = M_eta @ Z @ np.diag(target_eigenvalues) @ Z.T @ M_eta
    return 0.5 * (K_spec + K_spec.T), int(np.count_nonzero(keep)), w

# ============================================================
# Method 3: Edge-LS
# ============================================================

def fit_edge_stiffness_ls(K_f_12x12, C, lambda_reg=1e-6):
    basis = []
    for a in range(6):
        for b in range(a, 6):
            E = np.zeros((6,6)); E[a,b] = E[b,a] = 1.0; basis.append(E)
    X = np.zeros((144, 21))
    for k, E in enumerate(basis):
        X[:, k] = (C.T @ E @ C).ravel()
    y = K_f_12x12.ravel()
    k_vec = np.linalg.solve(X.T @ X + lambda_reg * np.eye(21), X.T @ y)
    K_edge = sum(k_vec[k] * E for k, E in enumerate(basis))
    w, V = np.linalg.eigh(K_edge)
    K_psd = V @ np.diag(np.maximum(w, 0)) @ V.T
    return 0.5*(K_psd+K_psd.T), 0.5*(K_edge+K_edge.T)

# ============================================================
# Method 4: Energy/Force fitting
# ============================================================

def generate_training_samples(
    eval_fn, pos0, c1, c2, s, E0, n_samples=30,
    amp_t=0.1, amp_r=0.01, seed=42, paired=False,
):
    rng = np.random.RandomState(seed); I4 = np.eye(4); samples = []
    ell = np.linalg.norm(c2-c1)
    S = np.diag([1,1,1,ell,ell,ell]); S_inv = np.linalg.inv(S)
    _, F0 = eval_fn(pos0.astype(np.float32))
    B0 = build_frame_basis(pos0, np.vstack([c1,c2]), np.column_stack([1-s,s]))
    Q0_eta = (B0[:, 6:12] @ S_inv).T @ F0.ravel()
    xis = []
    while len(xis) < n_samples:
        xi = np.concatenate([
            rng.uniform(-amp_t, amp_t, 3),
            rng.uniform(-amp_r, amp_r, 3),
        ])
        xis.append(xi)
        if paired and len(xis) < n_samples:
            xis.append(-xi)
    for xi in xis:
        D2 = exp_se3(xi)
        weights = np.column_stack([1.0-s, s])
        pos = reconstruct_blended_frame_positions(pos0, np.stack([I4, D2]), weights)
        E, F = eval_fn(pos.astype(np.float32))
        # Exact virtual work for the sampled log coordinate: Q_xi = J_xi^T F.
        # Only geometry is differenced; no additional force-field evaluations are needed.
        J_xi = np.zeros((pos0.size, 6))
        for g in range(6):
            h = 1e-6
            xi_p = xi.copy(); xi_p[g] += h
            xi_m = xi.copy(); xi_m[g] -= h
            pos_p = reconstruct_blended_frame_positions(pos0, np.stack([I4, exp_se3(xi_p)]), weights)
            pos_m = reconstruct_blended_frame_positions(pos0, np.stack([I4, exp_se3(xi_m)]), weights)
            J_xi[:, g] = ((pos_p-pos_m)/(2*h)).ravel()
        Q_xi = project_atomic_forces_to_frames(F, J_xi)
        eta = S @ xi
        Q_eta = S_inv.T @ Q_xi
        # The float32 UFF relaxation leaves a small reference force.  Centering removes
        # its linear energy term so the fitted model represents the local curvature.
        dE_quadratic = E - E0 + Q0_eta @ eta
        samples.append((eta, dE_quadratic, Q_eta-Q0_eta))
    return samples


def select_anharmonic_model(train_samples, validation_samples, K_anchor, ridge_values):
    """Choose cubic/quartic ridge strength on held-out UFF samples."""
    e_ref = max(np.sqrt(np.mean([dE*dE for _, dE, _ in validation_samples])), 1e-12)
    q_ref = max(np.sqrt(np.mean([np.mean(Q*Q) for _, _, Q in validation_samples])), 1e-12)
    candidates = []
    coordinate_scale = np.max(np.abs(np.asarray([s[0] for s in train_samples])), axis=0)
    for ridge in ridge_values:
        model, diagnostics = fit_reduced_polynomial(
            train_samples, K_anchor, min_degree=3, max_degree=4,
            coordinate_scale=coordinate_scale, ridge=ridge,
        )
        metrics = model.error_metrics(validation_samples)
        score = (metrics['energy_rmse']/e_ref)**2 + (metrics['force_component_rmse']/q_ref)**2
        candidates.append((score, ridge, model, diagnostics, metrics))
    return min(candidates, key=lambda item: item[0]), candidates

def _sym_basis():
    basis = []
    for a in range(6):
        for b in range(a, 6):
            E = np.zeros((6,6)); E[a,b]=E[b,a]=1.0; basis.append(E)
    return basis

def fit_stiffness_from_samples(samples, lambda_reg=1e-8):
    basis = _sym_basis()
    e_scale = max(np.sqrt(np.mean([dE*dE for _,dE,_ in samples])), 1e-8)
    q_scale = max(np.sqrt(np.mean([np.dot(Q,Q)/6 for _,_,Q in samples])), 1e-8)
    rows_A = []; rows_b = []
    for eta, dE, Q in samples:
        rows_A.append([0.5 * eta @ E @ eta / e_scale for E in basis]); rows_b.append(dE/e_scale)
        for g in range(6):
            rows_A.append([-(E @ eta)[g] / q_scale for E in basis]); rows_b.append(Q[g]/q_scale)
    A = np.array(rows_A); b = np.array(rows_b)
    k_vec = np.linalg.solve(A.T @ A + lambda_reg * np.eye(21), A.T @ b)
    K = sum(k_vec[k] * E for k, E in enumerate(basis))
    return 0.5 * (K + K.T)

def fit_stiffness_nonlinear_psd(samples, eps=1e-8, x0=None):
    e_scale = max(np.sqrt(np.mean([dE*dE for _,dE,_ in samples])), 1e-8)
    q_scale = max(np.sqrt(np.mean([np.dot(Q,Q)/6 for _,_,Q in samples])), 1e-8)
    def unpack(p):
        L = np.zeros((6,6)); idx = 0
        for i in range(6):
            for j in range(i+1): L[i,j] = p[idx]; idx += 1
        return L
    def loss(p):
        K = unpack(p) @ unpack(p).T + eps * np.eye(6); total = 0.0
        for eta, dE, Q in samples:
            total += ((0.5 * eta @ K @ eta - dE)/e_scale)**2
            total += np.sum((-K @ eta - Q)**2)/q_scale**2
        return total
    if x0 is None: x0 = np.zeros(21)
    res = minimize(loss, x0=x0, method='L-BFGS-B', options={'maxiter':500,'ftol':1e-12})
    L = unpack(res.x); K = L @ L.T + eps * np.eye(6)
    return 0.5*(K+K.T), res

# ============================================================
# Method 5: Relaxed (Schur complement)
# ============================================================

def relaxed_stiffness(H, pos, P_eta):
    """Static condensation for the same ``eta`` extracted by ``pinv(P_eta)``."""
    ndof = 3 * pos.shape[0]
    G = rigid_basis_geometric(pos)
    P = np.eye(ndof) - G @ G.T  # projector to non-rigid space
    H_perp = P @ H @ P  # H projected to non-rigid space (still ndof×ndof but rank ndof-6)
    # Use pseudo-inverse to handle the zero eigenvalues from projection
    H_perp_inv = np.linalg.pinv(H_perp, rcond=1e-10)
    C_eta = np.linalg.pinv(P_eta, rcond=1e-12)
    compliance = C_eta @ H_perp_inv @ C_eta.T
    K_relax = np.linalg.inv(compliance)
    return 0.5 * (K_relax + K_relax.T)

# ============================================================
# Validation
# ============================================================

def compare_mechanics(Ks):
    results = {}
    for name, K in Ks.items():
        w = np.linalg.eigvalsh(K)
        results[name] = w
    return results

def compare_frequencies(Ks, masses, P_eta):
    mass3 = np.repeat(masses, 3)
    M_eta = P_eta.T @ (mass3[:, None] * P_eta)
    results = {}
    for name, K in Ks.items():
        w = eigvalsh(K, M_eta)
        eV_to_J = 1.602176634e-19; amu_to_kg = 1.66053906660e-27; ang_to_m = 1e-10; c_cm = 2.99792458e10
        freqs = np.sqrt(np.maximum(w, 0) * eV_to_J / (amu_to_kg * ang_to_m**2)) / (2*np.pi*c_cm)
        results[name] = freqs
    return results

def force_residual(K, samples):
    total = 0.0
    for eta, dE, Q in samples:
        Q_pred = -K @ eta
        total += np.sum((Q_pred - Q)**2)
    return np.sqrt(total / len(samples))

# ============================================================
# Energy scan along individual DOFs
# ============================================================

DOF_LABELS = ['tx (Å)', 'ty (Å)', 'tz (Å)', 'rx (rad)', 'ry (rad)', 'rz (rad)']

def scan_dof_energies(eval_fn, pos0, c1, c2, s, E0, ell, dof_idx, n_pts=41, amp_t=0.5, amp_r=0.05):
    """Scan one DOF of frame 2 (frame 1 fixed). Return (amplitudes, dE_uff).
    dof_idx: 0-2 = translation, 3-5 = rotation."""
    I4 = np.eye(4)
    is_trans = dof_idx < 3
    amp = amp_t if is_trans else amp_r
    amps = np.linspace(-amp, amp, n_pts)
    dE_arr = np.zeros(n_pts)
    for k, a in enumerate(amps):
        xi = np.zeros(6); xi[dof_idx] = a
        D2 = exp_se3(xi)
        pos = reconstruct_blended_frame_positions(pos0, np.stack([I4, D2]), np.column_stack([1.0-s, s]))
        E, _ = eval_fn(pos.astype(np.float32))
        dE_arr[k] = E - E0
    return amps, dE_arr

def scan_rigid_energies(eval_fn, pos0, c1, c2, s, E0, dof_idx, n_pts=21, amp_t=0.5, amp_r=0.05):
    """Apply TRUE rigid body motion directly to atoms (bypass frame blending).
    This verifies UFF invariance. The frame model should also give V=0 since ξ=0.
    dof_idx: 0-2 = translation, 3-5 = rotation about COM."""
    com = pos0.mean(axis=0)
    is_trans = dof_idx < 3
    amp = amp_t if is_trans else amp_r
    amps = np.linspace(-amp, amp, n_pts)
    dE_arr = np.zeros(n_pts)
    for k, a in enumerate(amps):
        xi = np.zeros(6); xi[dof_idx] = a
        D = exp_se3(xi); R = D[:3,:3]; t = D[:3,3]
        # Rigid body motion: pos = R @ (pos0 - com) + t + com
        pos = (R @ (pos0 - com).T).T + t + com
        E, _ = eval_fn(pos.astype(np.float32))
        dE_arr[k] = E - E0
    return amps, dE_arr

def scan_rigid_frame_energies(eval_fn, pos0, c1, c2, s, E0, dof_idx, n_pts=21, amp_t=0.5, amp_r=0.05):
    """Move both absolute deformation transforms together; this must be rigid."""
    is_trans = dof_idx < 3
    amp = amp_t if is_trans else amp_r
    amps = np.linspace(-amp, amp, n_pts)
    dE_arr = np.zeros(n_pts)
    I4 = np.eye(4)
    for k, a in enumerate(amps):
        xi = np.zeros(6); xi[dof_idx] = a
        D = exp_se3(xi)
        pos = reconstruct_blended_frame_positions(pos0, np.stack([D, D]), np.column_stack([1.0-s, s]))
        E, _ = eval_fn(pos.astype(np.float32))
        dE_arr[k] = E - E0
    return amps, dE_arr

def harmonic_energy_1dof(K, ell, dof_idx, amps):
    """Harmonic prediction V = ½ η^T K η for single-DOF scan.
    η = S ξ, ξ = amps * e_dof → V = ½ * S[dof,dof]² * K[dof,dof] * amps²"""
    S_diag = np.array([1,1,1,ell,ell,ell])
    s_dof = S_diag[dof_idx]
    return 0.5 * s_dof**2 * K[dof_idx, dof_idx] * amps**2


def polynomial_energy_1dof(model, ell, dof_idx, amps):
    eta = np.zeros((len(amps), 6))
    eta[:, dof_idx] = np.array([1,1,1,ell,ell,ell])[dof_idx] * amps
    energy, _ = model.energy_force(eta)
    return energy


def plot_energy_scans(eval_fn, pos0, c1, c2, s, E0, ell, Ks, out_dir, anharmonic_model=None):
    """Plot UFF energy vs harmonic prediction for each of 6 DOFs.
    Also plot rigid mode checks (true rigid vs frame-blended rigid)."""
    n_dof = 6
    colors = {'UFF': 'black', 'Galerkin': '#2060ff', 'Relaxed': '#ff4020',
              'E/F-fit': '#20a040', 'Edge-LS': '#ff8000', 'Anharmonic': '#9b2fae'}
    # --- Internal DOF scans ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for d in range(n_dof):
        ax = axes[d // 3, d % 3]
        amps, dE_uff = scan_dof_energies(eval_fn, pos0, c1, c2, s, E0, ell, d)
        ax.plot(amps, dE_uff, 'o:', ms=3, color=colors['UFF'], lw=1.5, label='UFF restricted')
        plotted = [dE_uff]
        for name, K in Ks.items():
            dE_pred = harmonic_energy_1dof(K, ell, d, amps)
            plotted.append(dE_pred)
            ax.plot(amps, dE_pred, '-', color=colors.get(name, 'gray'), lw=0.8, label=name)
        if anharmonic_model is not None:
            dE_anh = polynomial_energy_1dof(anharmonic_model, ell, d, amps)
            plotted.append(dE_anh)
            ax.plot(amps, dE_anh, '-', color=colors['Anharmonic'], lw=1.0, label='Anharmonic')
        # Include every model in the visible range; clipping hid failed fits previously.
        y_min = min(y.min() for y in plotted); y_max = max(y.max() for y in plotted)
        margin = 0.1 * max(y_max - y_min, 1e-10)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_xlabel(DOF_LABELS[d]); ax.set_ylabel('ΔE (eV)')
        ax.set_title(f'DOF {d}: {DOF_LABELS[d]}')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.suptitle('Restricted Frame Energy — Frame 1 Fixed, Frame 2 Deformed\n'
                 '(harmonic and local anharmonic models vs unrelaxed UFF manifold)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out1 = os.path.join(out_dir, 'energy_scans_internal.png')
    fig.savefig(out1, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Energy scan plot saved to {out1}")

    # --- Rigid mode check: TRUE rigid (direct atom transform) ---
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
    for d in range(n_dof):
        ax = axes2[d // 3, d % 3]
        amps, dE_rigid = scan_rigid_energies(eval_fn, pos0, c1, c2, s, E0, d)
        ax.plot(amps, dE_rigid, 'o-', ms=3, color='black', lw=2, label='UFF (true rigid)')
        ax.axhline(0, color='red', ls='--', lw=1, alpha=0.5)
        ax.set_xlabel(DOF_LABELS[d]); ax.set_ylabel('ΔE (eV)')
        max_dE = max(np.abs(dE_rigid).max(), 1e-10)
        margin = 0.1 * max_dE + 1e-12
        ax.set_ylim(-max_dE - margin, max_dE + margin)
        ax.set_title(f'True rigid: {DOF_LABELS[d]}  (max|ΔE|={max_dE:.2e} eV)')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig2.suptitle('Rigid Mode Check — TRUE rigid body motion (atoms transformed directly)\nΔE should be ≈0 (UFF invariance)', fontsize=13, fontweight='bold')
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    out2 = os.path.join(out_dir, 'energy_scans_rigid_true.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight'); plt.close(fig2)
    print(f"  True rigid mode check plot saved to {out2}")

    # --- Rigid mode check: FRAME-BLENDED rigid (D1=D2=D, fixed centers) ---
    fig3, axes3 = plt.subplots(2, 3, figsize=(16, 10))
    for d in range(n_dof):
        ax = axes3[d // 3, d % 3]
        amps, dE_frame = scan_rigid_frame_energies(eval_fn, pos0, c1, c2, s, E0, d)
        ax.plot(amps, dE_frame, 'o-', ms=3, color='red', lw=2, label='UFF (frame rigid)')
        ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.5)
        ax.set_xlabel(DOF_LABELS[d]); ax.set_ylabel('ΔE (eV)')
        max_dE = max(np.abs(dE_frame).max(), 1e-10)
        margin = 0.1 * max_dE + 1e-12
        ax.set_ylim(-max_dE - margin, max_dE + margin)
        ax.set_title(f'Frame rigid: {DOF_LABELS[d]}  (max|ΔE|={max_dE:.2e} eV)')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig3.suptitle('Rigid Mode Check — D1=D2=D with ABSOLUTE transforms\nFrame blending now preserves rigid body invariance (should be ≈0)', fontsize=13, fontweight='bold')
    fig3.tight_layout(rect=[0, 0, 1, 0.95])
    out3 = os.path.join(out_dir, 'energy_scans_rigid_frame.png')
    fig3.savefig(out3, dpi=150, bbox_inches='tight'); plt.close(fig3)
    print(f"  Frame rigid check plot saved to {out3}")

    # --- Print anharmonicity summary ---
    print("\n=== LOCAL CURVATURE SUMMARY (restricted UFF vs Galerkin) ===")
    for d in range(n_dof):
        amps, dE_uff = scan_dof_energies(eval_fn, pos0, c1, c2, s, E0, ell, d)
        dE_harm = harmonic_energy_1dof(Ks['Galerkin'], ell, d, amps)
        c2_eff = np.sum(amps**2 * dE_uff) / np.sum(amps**4)
        c2_harm = 0.5 * np.array([1,1,1,ell,ell,ell])[d]**2 * Ks['Galerkin'][d, d]
        A_fit = np.column_stack([amps**2, amps**4])
        coeffs, *_ = np.linalg.lstsq(A_fit, dE_uff, rcond=None)
        c2_fit, c4_fit = coeffs
        anh_ratio = abs(c4_fit / c2_fit) if abs(c2_fit) > 1e-10 else float('inf')
        quartic_fraction = anh_ratio * np.max(np.abs(amps))**2
        print(f"  DOF {d} ({DOF_LABELS[d]:10s}): c2_model={c2_harm:.4f}, "
              f"c2_UFF={c2_fit:.4f}, quartic_fraction@scan_edge={quartic_fraction:.4f}")

    # --- Print rigid mode summary ---
    print("\n=== RIGID MODE CHECK SUMMARY ===")
    print("  TRUE rigid (atoms transformed directly):")
    for d in range(n_dof):
        amps, dE = scan_rigid_energies(eval_fn, pos0, c1, c2, s, E0, d)
        print(f"    DOF {d} ({DOF_LABELS[d]:10s}): max|ΔE|={np.abs(dE).max():.6e} eV")
    print("  FRAME-BLENDED rigid (D1=D2=D, absolute transforms — should be ~0 now):")
    for d in range(n_dof):
        amps, dE = scan_rigid_frame_energies(eval_fn, pos0, c1, c2, s, E0, d)
        print(f"    DOF {d} ({DOF_LABELS[d]:10s}): max|ΔE|={np.abs(dE).max():.6e} eV")
    print("  MODEL prediction for rigid (ξ=0 → V=0 by construction): exactly 0")

    plt.close('all')


def plot_anharmonic_validation(harmonic_model, anharmonic_model, samples, out_dir):
    """Held-out reference/model scatter and residual diagnostics."""
    eta = np.asarray([x[0] for x in samples])
    energy_ref = np.asarray([x[1] for x in samples])
    force_ref = np.asarray([x[2] for x in samples])
    energy_h, force_h = harmonic_model.energy_force(eta)
    energy_a, force_a = anharmonic_model.energy_force(eta)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    comparisons = [
        (axes[0], energy_ref, energy_h, energy_a, 'Energy', 'eV'),
        (axes[1], force_ref.ravel(), force_h.ravel(), force_a.ravel(), 'Generalized force', 'eV/Å'),
    ]
    for ax, reference, harmonic, anharmonic, label, unit in comparisons:
        lo = min(reference.min(), harmonic.min(), anharmonic.min())
        hi = max(reference.max(), harmonic.max(), anharmonic.max())
        margin = 0.05 * max(hi-lo, 1e-12)
        ax.plot([lo-margin, hi+margin], [lo-margin, hi+margin], ':', color='black', lw=1.5,
                label='exact parity')
        ax.scatter(reference, harmonic, s=12, alpha=0.45, label='Harmonic')
        ax.scatter(reference, anharmonic, s=12, alpha=0.45, label='Cubic+quartic')
        rmse_h = np.sqrt(np.mean((harmonic-reference)**2))
        rmse_a = np.sqrt(np.mean((anharmonic-reference)**2))
        max_a = np.max(np.abs(anharmonic-reference))
        ax.text(0.97, 0.03, f'harm RMSE {rmse_h:.3e}\nanh  RMSE {rmse_a:.3e}\nanh max  {max_a:.3e}',
                transform=ax.transAxes, ha='right', va='bottom', family='monospace', fontsize=8,
                bbox={'facecolor':'white', 'alpha':0.8, 'edgecolor':'none'})
        ax.set_xlabel(f'UFF {label} ({unit})')
        ax.set_ylabel(f'Model {label} ({unit})')
        ax.set_title(f'Held-out {label} parity')
        ax.set_xlim(lo-margin, hi+margin); ax.set_ylim(lo-margin, hi+margin)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = os.path.join(out_dir, 'anharmonic_validation.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Anharmonic validation plot saved to {path}")


def plot_reduced_load_paths(load_paths, coordinate_scale, out_dir):
    """Plot reversible bounded reduced relaxations under one proportional load per DOF."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    for d, record in enumerate(load_paths):
        ax = axes[d // 3, d % 3]
        fractions = record['fractions']
        eta = np.asarray([r.eta for r in record['results']])
        boundary = np.asarray([r.hit_boundary for r in record['results']])
        split = record['n_loading']
        target = record['target_eta'][d] / coordinate_scale[d]
        ax.plot(fractions[:split], eta[:split, d] / coordinate_scale[d], 'o-', ms=3,
                color='#2060ff', label='load')
        ax.plot(fractions[split-1:], eta[split-1:, d] / coordinate_scale[d], 's-', ms=3,
                color='#ff6020', label='unload')
        ax.plot(fractions, fractions * target, ':', color='black', lw=1.2,
                label='harmonic target')
        if np.any(boundary):
            ax.scatter(fractions[boundary], eta[boundary, d] / coordinate_scale[d],
                       marker='x', s=45, color='red', label='trust-boundary hit')
        ax.set_title(DOF_LABELS[d])
        ax.set_xlabel('load fraction')
        ax.set_ylabel(r'$\eta_i / s_i$')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle('Quasistatic reduced-potential relaxation under external load\n'
                 'solid: stabilized cubic+quartic; dotted: harmonic reference',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(out_dir, 'load_relaxation.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Load-relaxation plot saved to {path}")


def validate_blended_frame_dynamics(pos0, weights, masses, potential, ell):
    """PTCDA-specific checks for frame skinning, force transfer, and damped motion."""
    rng = np.random.RandomState(721)
    dynamics = BlendedFrameDynamics(pos0, weights, masses, potential, ell)
    evaluation = dynamics.evaluate()

    # Virtual work is the non-negotiable projection invariant: J.T @ F must agree
    # with the Cartesian work of an arbitrary atomic force under a frame displacement.
    atom_force = rng.normal(size=pos0.shape)
    frame_force = project_atomic_forces_to_frames(atom_force, evaluation.position_jacobian)
    direction = rng.normal(size=12)
    direction /= np.linalg.norm(direction)
    h = 1e-6
    frames0 = dynamics.frame_transforms.copy()
    dynamics.apply_increment(h * direction); pos_plus = dynamics.positions()
    dynamics.frame_transforms = frames0.copy()
    dynamics.apply_increment(-h * direction); pos_minus = dynamics.positions()
    dynamics.frame_transforms = frames0
    work_fd = np.dot(atom_force.ravel(), ((pos_plus-pos_minus)/(2*h)).ravel())
    work_projected = np.dot(frame_force, direction)
    work_relative_error = abs(work_projected-work_fd) / max(abs(work_fd), 1e-12)
    if work_relative_error > 1e-7:
        raise RuntimeError(f"full-frame virtual-work parity failed: {work_relative_error:.3e}")

    # Damped velocity-Verlet is tested on an initially bent/twisted frame state.  The
    # test deliberately checks bounded energy rather than UFF-frequency accuracy.
    eta0 = np.zeros(6); eta0[2] = 0.5 * potential.coordinate_scale[2]; eta0[3] = 0.2 * potential.coordinate_scale[3]
    xi0 = np.diag([1, 1, 1, 1/ell, 1/ell, 1/ell]) @ eta0
    frames = np.stack([np.eye(4), exp_se3(xi0)])
    dynamics = BlendedFrameDynamics(pos0, weights, masses, potential, ell, frame_transforms=frames)
    initial_energy = dynamics.evaluate().potential_energy
    energies = []
    for _ in range(200):
        energies.append(dynamics.step_velocity_verlet(dt=0.2, damping=0.5).potential_energy)
    final_energy = energies[-1]
    max_energy = max(energies)
    if not np.isfinite(energies).all() or final_energy >= initial_energy or max_energy > initial_energy * 1.01:
        raise RuntimeError("damped reduced velocity-Verlet stability check failed")

    # An atom--fixed-environment harmonic pair potential exercises the actual external
    # force callback.  Coulomb/Morse evaluators use the same (energy, atom-force) API.
    target = pos0 + np.array([0.10, -0.03, 0.02])
    k_env = 0.05
    def anchored_pairwise_external(positions):
        displacement = positions - target
        return 0.5 * k_env * np.sum(displacement * displacement), -k_env * displacement

    driven = BlendedFrameDynamics(pos0, weights, masses, potential, ell)
    com0 = np.average(driven.positions(), axis=0, weights=masses)
    for _ in range(16):
        driven.relax_step(anchored_pairwise_external, mobility=1.0, max_atom_step=0.02)
    com1 = np.average(driven.positions(), axis=0, weights=masses)
    requested_shift = target.mean(axis=0) - pos0.mean(axis=0)
    achieved_shift = com1 - com0
    if np.dot(achieved_shift, requested_shift) <= 0:
        raise RuntimeError("external atomic forces did not drive the projected frame state")
    return {
        'virtual_work_relative_error': work_relative_error,
        'damped_initial_energy_eV': initial_energy,
        'damped_final_energy_eV': final_energy,
        'damped_max_energy_eV': max_energy,
        'external_com_shift_A': np.linalg.norm(achieved_shift),
    }

# ============================================================
# UFF relaxation
# ============================================================

def relax_uff(eval_fn, pos0, max_iter=500, force_tol=2e-3):
    """Relax molecule to UFF minimum using L-BFGS-B."""
    def objective(x):
        pos = x.reshape(-1, 3).astype(np.float32)
        E, _ = eval_fn(pos)
        return float(E)
    def gradient(x):
        pos = x.reshape(-1, 3).astype(np.float32)
        _, F = eval_fn(pos)
        return -F.ravel().astype(np.float64)
    x0 = pos0.ravel().astype(np.float64)
    # Energy is float32, so an ftol stopping condition can report success well before
    # forces are small. Disable relative-energy termination and report force convergence.
    res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': 0.0, 'gtol': force_tol,
                            'maxls': 50, 'maxfun': 50000})
    pos_relaxed = res.x.reshape(-1, 3)
    force_norm = np.linalg.norm(res.jac)
    force_max = np.max(np.abs(res.jac))
    converged = force_max <= force_tol
    print(f"  Relaxed: E={res.fun:.6f} eV, |grad|={force_norm:.2e}, "
          f"max|grad|={force_max:.2e}, force_converged={converged}, scipy={res.message}")
    return pos_relaxed, res.fun

# ============================================================
# Logging & CSV output
# ============================================================

class Tee:
    """Redirect stdout to both console and log file."""
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, 'w')
    def write(self, msg):
        self.terminal.write(msg); self.log.write(msg)
    def flush(self):
        self.terminal.flush(); self.log.flush()

def write_csv(path, header, rows):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def analyze_edge_ls(K_f_12, C_inc, K_edge_xi, B, H, ell):
    """Check the edge fit against the direct relative-twist Hessian."""
    print("\n=== EDGE-LS ANALYSIS ===")
    w_f = np.linalg.eigvalsh(K_f_12)
    print(f"  K_f (12x12) eigenvalues: {np.array2string(w_f, precision=4)}")
    B1, B2 = B[:, :6], B[:, 6:]
    K_xi_direct = 0.25 * (B2-B1).T @ H @ (B2-B1)
    # Because C C^T = 2I, the right inverse is T=C^T/2 and T^T K_f T
    # equals (1/4) C K_f C^T, not C K_f C^T.
    K_xi_extract = 0.25 * C_inc @ K_f_12 @ C_inc.T
    rel_fit = np.linalg.norm(K_edge_xi-K_xi_direct) / max(np.linalg.norm(K_xi_direct), 1e-30)
    rel_extract = np.linalg.norm(K_xi_extract-K_xi_direct) / max(np.linalg.norm(K_xi_direct), 1e-30)
    rigid_leak = np.linalg.norm((B1+B2).T @ H @ (B1+B2))
    S_inv = np.diag([1,1,1,1/ell,1/ell,1/ell])
    K_eta_direct = S_inv.T @ K_xi_direct @ S_inv
    print(f"  direct K_eta eigenvalues: {np.array2string(np.linalg.eigvalsh(K_eta_direct), precision=4)}")
    print(f"  Edge-LS/direct relative error: {rel_fit:.3e}")
    print(f"  right-inverse extraction/direct relative error: {rel_extract:.3e}")
    print(f"  common-rigid Hessian leakage Frobenius norm: {rigid_leak:.3e}")
    return K_eta_direct

# ============================================================
# Main
# ============================================================

def main():
    out_dir = os.path.join(REPO_ROOT, 'debug', 'fit_stiffness')
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'fit_stiffness.log')
    sys.stdout = Tee(log_path)
    PTCDA_FILE = os.path.join(REPO_ROOT, 'data', 'xyz', 'PTCDA.xyz')
    print(f"Loading {PTCDA_FILE}")
    mol = AtomicSystem(fname=PTCDA_FILE)
    mol.findBonds(byRvdW=False, Rcut=1.6, RvdwCut=0.5)
    mol.neighs()
    print(f"  {len(mol.apos)} atoms, {len(mol.bonds)} bonds")

    pos = np.asarray(mol.apos[:, :3], dtype=np.float64)
    natoms = len(pos)
    masses = atomic_masses(list(mol.enames))

    # Frame centers from bridge oxygens
    o_idx = [i for i in range(natoms) if mol.enames[i] == 'O']
    bridge_O = [i for i in o_idx if sum(1 for j in mol.ngs[i] if mol.enames[j] == 'C') == 2]
    left_O = [i for i in bridge_O if pos[i, 0] < 0]
    right_O = [i for i in bridge_O if pos[i, 0] > 0]
    c1 = pos[left_O].mean(axis=0); c2 = pos[right_O].mean(axis=0)
    ell = np.linalg.norm(c2 - c1)
    print(f"Frame centers: c1={c1}, c2={c2}, ℓ={ell:.3f} Å")

    # Weights: smoothstep
    axis = c2 - c1; e = axis / np.linalg.norm(axis)
    t = np.dot(pos - c1[None,:], e) / np.linalg.norm(axis)
    t = np.clip(t, 0, 1); s = t * t * (3 - 2 * t)
    weights = np.column_stack([1 - s, s])  # (N, 2)
    print(f"Weight range: s=[{s.min():.3f}, {s.max():.3f}]")

    # UFF relaxation to minimum
    print("\nRelaxing molecule to UFF minimum...")
    eval_fn_relax, _, _, _, _ = make_ff_eval_fn(mol, do_nonbond=False)
    pos, E_min = relax_uff(eval_fn_relax, pos, max_iter=500)
    mol.apos[:, :3] = pos  # update molecule positions
    # Recompute centers/weights from relaxed positions
    left_O = [i for i in bridge_O if pos[i, 0] < 0]
    right_O = [i for i in bridge_O if pos[i, 0] > 0]
    c1 = pos[left_O].mean(axis=0); c2 = pos[right_O].mean(axis=0)
    ell = np.linalg.norm(c2 - c1)
    print(f"  Relaxed centers: c1={c1}, c2={c2}, ℓ={ell:.3f} Å")
    axis = c2 - c1; e = axis / np.linalg.norm(axis)
    t = np.dot(pos - c1[None,:], e) / np.linalg.norm(axis)
    t = np.clip(t, 0, 1); s = t * t * (3 - 2 * t)
    weights = np.column_stack([1 - s, s])

    # UFF Hessian (at relaxed minimum)
    print("\nComputing UFF Hessian via finite differences...")
    H, pos0, masses, enames, _ = hessian_from_ff(mol, ff='uff', delta=1e-4, do_nonbond=False)
    print(f"  H shape={H.shape}, max|H|={np.abs(H).max():.4f} eV/Å²")
    _, F_stationary = eval_fn_relax(pos0.astype(np.float32))
    print(f"  stationarity: |F|={np.linalg.norm(F_stationary):.3e} eV/Å, "
          f"max|F|={np.max(np.abs(F_stationary)):.3e} eV/Å, "
          f"|sum F|={np.linalg.norm(F_stationary.sum(axis=0)):.3e}")

    # Build frame basis
    centers = np.vstack([c1, c2])
    B = build_frame_basis(pos0, centers, weights)
    print(f"  B shape={B.shape}")

    # One physical coordinate contract for all methods: eta=S*xi.
    P_eta = build_internal_prolongation(B, pos0, ell, bMassOrthogonal=False)
    P_eta_dyn = build_internal_prolongation(B, pos0, ell, masses=masses, bMassOrthogonal=True)
    sv_eta = np.linalg.svd(P_eta, compute_uv=False)
    print(f"  P_eta shape={P_eta.shape}, singular values={np.array2string(sv_eta, precision=4)}")

    # Incidence matrix for single edge
    C_inc = np.hstack([-np.eye(6), np.eye(6)])  # (6, 12)

    # --- Method 1: Galerkin ---
    print("\n--- Method 1: Galerkin (Ritz) ---")
    K_gal = galerkin_stiffness(H, P_eta)
    print(f"  eigenvalues: {np.linalg.eigvalsh(K_gal)}")

    # --- Method 3: Edge-LS ---
    print("\n--- Method 3: Edge-sparse LS ---")
    K_f_12 = B.T @ H @ B  # 12×12 full frame stiffness
    K_edge_xi_psd, K_edge_xi_raw = fit_edge_stiffness_ls(K_f_12, C_inc)
    S_inv = np.diag([1,1,1,1/ell,1/ell,1/ell])
    K_edge_raw = S_inv.T @ K_edge_xi_raw @ S_inv
    K_edge_psd = S_inv.T @ K_edge_xi_psd @ S_inv
    print(f"  K_edge_eta eigenvalues (raw): {np.linalg.eigvalsh(K_edge_raw)}")
    print(f"  K_edge_eta eigenvalues (PSD): {np.linalg.eigvalsh(K_edge_psd)}")
    K_eta_direct = analyze_edge_ls(K_f_12, C_inc, K_edge_xi_raw, B, H, ell)
    print(f"  Galerkin/direct-coordinate relative error: "
          f"{np.linalg.norm(K_gal-K_eta_direct)/np.linalg.norm(K_eta_direct):.3e}")

    # --- Method 4: Energy/Force fitting ---
    print("\n--- Method 4: Energy/Force fitting ---")
    eval_fn, pos0_eval, _, _, _ = make_ff_eval_fn(mol, do_nonbond=False)
    E0, F0_eval = eval_fn(pos0_eval.astype(np.float32))
    print(f"  E0 = {E0:.6f} eV")
    samples = generate_training_samples(eval_fn, pos0, c1, c2, s, E0, n_samples=50, amp_t=0.05, amp_r=0.005)
    print(f"  Generated {len(samples)} samples")
    # Independent energy-gradient parity check for the first sampled eta.
    eta_check, _, Q_check = samples[0]
    S = np.diag([1,1,1,ell,ell,ell]); S_inv_check = np.linalg.inv(S)
    Q_fd = np.zeros(6); h_eta = 1e-3
    for g in range(6):
        eta_p = eta_check.copy(); eta_p[g] += h_eta
        eta_m = eta_check.copy(); eta_m[g] -= h_eta
        xi_p = S_inv_check @ eta_p; xi_m = S_inv_check @ eta_m
        Ep, _ = eval_fn(reconstruct_blended_frame_positions(
            pos0, np.stack([np.eye(4), exp_se3(xi_p)]), np.column_stack([1.0-s, s]),
        ).astype(np.float32))
        Em, _ = eval_fn(reconstruct_blended_frame_positions(
            pos0, np.stack([np.eye(4), exp_se3(xi_m)]), np.column_stack([1.0-s, s]),
        ).astype(np.float32))
        Q_fd[g] = -(Ep-Em)/(2*h_eta)
    Q0_eta = (B[:, 6:12] @ S_inv_check).T @ F0_eval.ravel()
    Q_fd -= Q0_eta
    vw_abs = np.max(np.abs(Q_check-Q_fd))
    vw_rel = np.linalg.norm(Q_check-Q_fd)/max(np.linalg.norm(Q_fd), 1e-12)
    print(f"  virtual-work parity: max_abs={vw_abs:.3e} eV/Å, rel_L2={vw_rel:.3e}")
    K_ef = fit_stiffness_from_samples(samples)
    print(f"  K_ef eigenvalues: {np.linalg.eigvalsh(K_ef)}")
    # PSD fit: use the nearest positive Galerkin matrix as initial guess.
    w0, V0 = np.linalg.eigh(K_gal)
    K0_psd = (V0 * np.maximum(w0, 1e-8)) @ V0.T
    L0 = np.linalg.cholesky(K0_psd)
    x0_psd = L0[np.tril_indices(6)]
    K_ef_psd, res_psd = fit_stiffness_nonlinear_psd(samples, x0=x0_psd)
    print(f"  K_ef_psd eigenvalues: {np.linalg.eigvalsh(K_ef_psd)}")
    print(f"  PSD fit converged: {res_psd.success}, loss={res_psd.fun:.6e}")

    # --- Local cubic/quartic potential anchored to the exact Galerkin Hessian ---
    print("\n--- Local anharmonic fit: cubic + quartic ---")
    anh_train = generate_training_samples(
        eval_fn, pos0, c1, c2, s, E0, n_samples=240,
        amp_t=0.20, amp_r=0.020, seed=137, paired=True,
    )
    anh_valid = generate_training_samples(
        eval_fn, pos0, c1, c2, s, E0, n_samples=96,
        amp_t=0.20, amp_r=0.020, seed=911, paired=True,
    )
    ridge_values = [0.0, 1e2, 1e4, 1e6, 1e8, 1e10]
    best, candidates = select_anharmonic_model(anh_train, anh_valid, K_gal, ridge_values)
    anh_score, anh_ridge, anh_model, anh_diag, anh_valid_metrics = best
    empty_powers = np.zeros((0, 6), dtype=np.int16)
    harmonic_anchor = ReducedPolynomialPotential(
        K=K_gal,
        coordinate_scale=anh_model.coordinate_scale,
        powers=empty_powers,
        coefficients=np.zeros(0),
    )
    harmonic_valid_metrics = harmonic_anchor.error_metrics(anh_valid)
    for score, ridge, _, diag, metrics in candidates:
        print(f"  ridge={ridge:8.1e} score={score:.4e} rank={diag['rank']:3d}/{diag['n_terms']} "
              f"E_RMSE={metrics['energy_rmse']:.3e} Qcomp_RMSE={metrics['force_component_rmse']:.3e}")
    print(f"  selected ridge={anh_ridge:.1e}, terms={anh_diag['n_terms']}, "
          f"condition={anh_diag['condition']:.3e}")
    print(f"  eta coordinate scale (fitted trust box): "
          f"{np.array2string(anh_model.coordinate_scale, precision=4)} Å")
    print(f"  held-out harmonic:   E_RMSE={harmonic_valid_metrics['energy_rmse']:.3e} eV, "
          f"Qcomp_RMSE={harmonic_valid_metrics['force_component_rmse']:.3e} eV/Å")
    print(f"  held-out anharmonic: E_RMSE={anh_valid_metrics['energy_rmse']:.3e} eV, "
          f"Qcomp_RMSE={anh_valid_metrics['force_component_rmse']:.3e} eV/Å")

    # Exact derivative invariant of the fitted analytic polynomial.
    eta_grad = anh_valid[0][0]
    _, Q_analytic = anh_model.energy_force(eta_grad)
    Q_fd_model = np.zeros(6)
    for g in range(6):
        h = 1e-6 * anh_model.coordinate_scale[g]
        ep = eta_grad.copy(); ep[g] += h
        em = eta_grad.copy(); em[g] -= h
        Ep, _ = anh_model.energy_force(ep)
        Em, _ = anh_model.energy_force(em)
        Q_fd_model[g] = -(Ep-Em)/(2*h)
    print(f"  analytic-force/energy-gradient max error: {np.max(np.abs(Q_analytic-Q_fd_model)):.3e} eV/Å")
    quartic_min, quartic_max = anh_model.homogeneous_direction_range(degree=4)
    print(f"  dimensionless unit-sphere quartic range: [{quartic_min:.3e}, {quartic_max:.3e}] eV")
    if quartic_min < 0:
        print("  Local fit has a negative sampled quartic direction; applying radial stabilization for relaxation.")

    # The raw Taylor fit is valuable for regression diagnostics, but is not an
    # acceptable relaxation model if its quartic term turns down in some direction.
    # Add a small isotropic ||eta/scale||^4 term, then retain a hard trust box at
    # runtime.  The sampled scan is an operational check, not a positivity proof.
    stable_anh_model, stabilization = stabilize_reduced_quartic_sampled(
        anh_model, margin=1e-7, n_directions=32768, safety_factor=1.10, seed=451,
    )
    stable_valid_metrics = stable_anh_model.error_metrics(anh_valid)
    print(f"  radial quartic stabilization: lambda={stabilization['radial_quartic_lambda']:.3e} eV, "
          f"sampled range [{stabilization['quartic_min_before']:.3e}, "
          f"{stabilization['quartic_max_before']:.3e}] -> "
          f"[{stabilization['quartic_min_after']:.3e}, {stabilization['quartic_max_after']:.3e}] eV")
    print(f"  held-out stabilized: E_RMSE={stable_valid_metrics['energy_rmse']:.3e} eV, "
          f"Qcomp_RMSE={stable_valid_metrics['force_component_rmse']:.3e} eV/Å")

    # Mechanical invariants for the model that will actually be serialized and used.
    K_origin = reduced_potential_hessian(stable_anh_model)
    origin_eigs = np.linalg.eigvalsh(K_origin)
    if origin_eigs[0] <= 0:
        raise RuntimeError(f"stable reduced model is not a local minimum: min eigenvalue={origin_eigs[0]}")
    zero_relax = relax_reduced_potential(
        stable_anh_model, np.zeros(6), trust_radius=1.0, gtol=1e-11,
    )
    print(f"  zero-load relaxation: |eta|={np.linalg.norm(zero_relax.eta):.3e} Å, "
          f"residual={zero_relax.residual_norm:.3e} eV/Å, "
          f"success={zero_relax.success}, boundary={zero_relax.hit_boundary}")
    if not zero_relax.success or zero_relax.hit_boundary or zero_relax.residual_norm > 1e-8:
        raise RuntimeError("zero-load reduced relaxation failed its local-minimum invariant")

    # A quasistatic load/unload cycle in every generalized direction is the relevant
    # validation for this relaxation-focused model.  A half-scale target remains
    # deliberately inside the fitted coordinate box; a boundary hit is invalid.
    load_fractions = np.concatenate([np.linspace(0.0, 1.0, 17), np.linspace(1.0, 0.0, 17)[1:]])
    load_paths = []
    for d in range(6):
        target_eta = np.zeros(6)
        target_eta[d] = 0.5 * stable_anh_model.coordinate_scale[d]
        external_force = K_gal @ target_eta
        fractions, results = relax_reduced_load_path(
            stable_anh_model, external_force, load_fractions=load_fractions,
            trust_radius=1.0, gtol=1e-10,
        )
        max_residual = max(r.residual_norm for r in results)
        returned_scaled = np.linalg.norm(results[-1].eta / stable_anh_model.coordinate_scale)
        boundary_hit = any(r.hit_boundary for r in results)
        solver_failure = any(not r.success for r in results)
        print(f"  load path {d} ({DOF_LABELS[d]:8s}): max_residual={max_residual:.3e} eV/Å, "
              f"return |eta/scale|={returned_scaled:.3e}, boundary={boundary_hit}, success={not solver_failure}")
        if boundary_hit or solver_failure or max_residual > 1e-7 or returned_scaled > 1e-5:
            raise RuntimeError(f"reduced load-path invariant failed for {DOF_LABELS[d]}")
        load_paths.append({
            'dof': d, 'target_eta': target_eta, 'external_force': external_force,
            'fractions': fractions, 'results': results, 'n_loading': 17,
        })

    print("\n--- Blended-frame force transfer and damped dynamics ---")
    frame_dynamics_diag = validate_blended_frame_dynamics(
        pos0, weights, masses, stable_anh_model, ell,
    )
    print(f"  full-frame virtual-work relative error: "
          f"{frame_dynamics_diag['virtual_work_relative_error']:.3e}")
    print(f"  damped dynamics energy: {frame_dynamics_diag['damped_initial_energy_eV']:.3e} -> "
          f"{frame_dynamics_diag['damped_final_energy_eV']:.3e} eV "
          f"(max {frame_dynamics_diag['damped_max_energy_eV']:.3e} eV)")
    print(f"  projected external-force COM shift: "
          f"{frame_dynamics_diag['external_com_shift_A']:.3e} Å")

    # --- Method 5: Relaxed (Schur) ---
    print("\n--- Method 5: Relaxed (Schur complement) ---")
    K_relax = relaxed_stiffness(H, pos0, P_eta)
    print(f"  eigenvalues: {np.linalg.eigvalsh(K_relax)}")

    # --- Method 2: Spectral (dynamics only) ---
    print("\n--- Method 2: Spectral (dynamics) ---")
    Hp, _ = project_rigid_modes(H, pos0, masses)
    freq_all, modes_all = hessian_to_modes(Hp, masses)
    # Get 6 lowest non-rigid modes
    n_fit = min(6, np.sum(np.abs(freq_all) >= 20.0))
    mask = np.abs(freq_all) >= 20.0
    freq_vib = freq_all[mask]; modes_vib = modes_all[:, mask]
    order = np.argsort(np.abs(freq_vib))[:n_fit]
    target_freqs = freq_vib[order]; target_modes = modes_vib[:, order]
    # Convert freq to eigenvalues (ω² in eV/(amu·Å²))
    eV_to_J = 1.602176634e-19; amu_to_kg = 1.66053906660e-27; ang_to_m = 1e-10; c_cm = 2.99792458e10
    target_omega2 = np.sign(target_freqs) * (2*np.pi*np.abs(target_freqs)*c_cm)**2 \
                    * amu_to_kg * ang_to_m**2 / eV_to_J
    K_spec, spec_rank, spec_overlap_eigs = spectral_stiffness(
        P_eta_dyn, masses, target_modes, target_omega2)
    print(f"  eigenvalues: {np.linalg.eigvalsh(K_spec)}")
    print(f"  representable target rank: {spec_rank}/{n_fit}, overlap Gram eigenvalues: "
          f"{np.array2string(spec_overlap_eigs, precision=4)}")

    # --- Comparison ---
    print("\n=== MECHANICS: Stiffness eigenvalues (eV/Å²) ===")
    Ks = {'Galerkin': K_gal, 'Edge-LS': K_edge_psd, 'E/F-fit': K_ef_psd, 'Relaxed': K_relax}
    mech = compare_mechanics(Ks)
    for name, w in mech.items():
        print(f"  {name:12s}: {np.array2string(w, precision=4)}")

    print("\n=== DYNAMICS: Frequencies (cm⁻¹) ===")
    freqs = compare_frequencies(Ks, masses, P_eta_dyn)
    freqs['Spectral'] = compare_frequencies({'Spectral': K_spec}, masses, P_eta_dyn)['Spectral']
    # UFF target
    print(f"  {'UFF target':12s}: {np.array2string(target_freqs[:6], precision=2)}")
    for name, f in freqs.items():
        print(f"  {name:12s}: {np.array2string(f[:6], precision=2)}")

    # Only restricted/clamped methods target these unrelaxed nonlinear samples.
    # Relaxed static condensation has a different target and must not be scored here.
    Ks_restricted = {'Galerkin': K_gal, 'Edge-LS': K_edge_psd, 'E/F-fit': K_ef_psd}
    print("\n=== FORCE RESIDUALS ON RESTRICTED SAMPLES (RMS per sample) ===")
    force_res = {}
    for name, K in Ks_restricted.items():
        r = force_residual(K, samples)
        force_res[name] = r
        print(f"  {name:12s}: {r:.6e}")
    anh_small_metrics = stable_anh_model.error_metrics(samples)
    force_res['Stable anharmonic'] = anh_small_metrics['force_vector_rms']
    print(f"  {'Stable anharmonic':18s}: {force_res['Stable anharmonic']:.6e}")

    # --- Write CSV files ---
    csv_mech = os.path.join(out_dir, 'stiffness_eigenvalues.csv')
    write_csv(csv_mech, ['method'] + [f'lambda_{i+1}' for i in range(6)],
              [[name] + list(w) for name, w in mech.items()])
    csv_freq = os.path.join(out_dir, 'frequencies.csv')
    freq_rows = [['UFF target'] + list(target_freqs[:6])]
    for name, f in freqs.items(): freq_rows.append([name] + list(f[:6]))
    write_csv(csv_freq, ['method'] + [f'freq_{i+1}_cm1' for i in range(6)], freq_rows)
    csv_force = os.path.join(out_dir, 'force_residuals.csv')
    write_csv(csv_force, ['method', 'rms_force_residual'],
              [[name, r] for name, r in force_res.items()])
    csv_anh = os.path.join(out_dir, 'anharmonic_coefficients.csv')
    write_csv(
        csv_anh,
        ['degree'] + [f'power_eta_{i}' for i in range(6)] + ['coefficient_eV'],
        [[int(np.sum(p))] + list(map(int, p)) + [c]
         for p, c in zip(stable_anh_model.powers, stable_anh_model.coefficients)],
    )
    model_anh = os.path.join(out_dir, 'anharmonic_model.npz')
    model_anh_local = os.path.join(out_dir, 'anharmonic_model_local.npz')
    stable_anh_model.save_npz(model_anh)
    anh_model.save_npz(model_anh_local)
    anh_loaded = ReducedPolynomialPotential.load_npz(model_anh)
    E_saved, Q_saved = anh_loaded.energy_force(anh_valid[0][0])
    E_live, Q_live = stable_anh_model.energy_force(anh_valid[0][0])
    serialization_error = max(abs(E_saved-E_live), np.max(np.abs(Q_saved-Q_live)))
    if serialization_error > 1e-12:
        raise RuntimeError(f"anharmonic model serialization parity failed: {serialization_error}")
    print(f"\nCSV files written: {csv_mech}, {csv_freq}, {csv_force}")
    print(f"Anharmonic coefficients written: {csv_anh}")
    print(f"Stable anharmonic model written: {model_anh} (round-trip error {serialization_error:.1e})")
    print(f"Raw local regression model written: {model_anh_local}")

    csv_load = os.path.join(out_dir, 'load_relaxation.csv')
    load_rows = []
    for record in load_paths:
        for phase_index, (fraction, result) in enumerate(zip(record['fractions'], record['results'])):
            phase = 'load' if phase_index < record['n_loading'] else 'unload'
            load_rows.append([
                record['dof'], DOF_LABELS[record['dof']], phase, fraction,
                *result.eta, result.internal_energy, result.total_energy,
                result.residual_norm, int(result.hit_boundary), int(result.success),
            ])
    write_csv(
        csv_load,
        ['dof_idx', 'dof_label', 'phase', 'load_fraction'] + [f'eta_{i}' for i in range(6)]
        + ['internal_energy_eV', 'total_energy_eV', 'residual_norm_eV_per_A', 'hit_boundary', 'success'],
        load_rows,
    )
    print(f"Load-relaxation CSV written: {csv_load}")

    csv_frame_dynamics = os.path.join(out_dir, 'frame_dynamics_validation.csv')
    write_csv(
        csv_frame_dynamics, ['metric', 'value'],
        [[name, value] for name, value in frame_dynamics_diag.items()],
    )
    print(f"Frame-dynamics validation CSV written: {csv_frame_dynamics}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Stiffness eigenvalues bar chart
    ax = axes[0]
    methods = list(mech.keys()); n_meth = len(methods)
    x = np.arange(6); width = 0.8 / n_meth
    for i, (name, w) in enumerate(mech.items()):
        ax.bar(x + i * width - 0.4 + width/2, w, width, label=name)
    ax.set_xlabel('Mode index'); ax.set_ylabel('Stiffness eigenvalue (eV/Å²)')
    ax.set_yscale('log')
    ax.set_title('Frame Stiffness Eigenvalues — Mechanics'); ax.legend(); ax.set_xticks(x)
    # Frequency comparison
    ax = axes[1]
    n_freq_series = len(freqs) + 1
    width_freq = 0.8 / n_freq_series
    for i, (name, f) in enumerate(freqs.items()):
        ax.bar(x + i*width_freq - 0.4 + width_freq/2, f[:6], width_freq, label=name)
    ax.bar(x + len(freqs)*width_freq - 0.4 + width_freq/2, target_freqs[:6],
           width_freq, label='UFF target', color='black', alpha=0.5)
    ax.set_xlabel('Mode index'); ax.set_ylabel('Frequency (cm⁻¹)')
    ax.set_title('Frame Mode Frequencies — Dynamics'); ax.legend(); ax.set_xticks(x)
    fig.tight_layout()
    out = os.path.join(out_dir, 'stiffness_comparison.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {out}")
    plt.close(fig)

    # --- Energy scan plots ---
    print("\n=== ENERGY SCANS: Frame 1 fixed, frame 2 deformed along each DOF ===")
    plot_energy_scans(
        eval_fn, pos0, c1, c2, s, E0, ell, Ks_restricted, out_dir,
        anharmonic_model=stable_anh_model,
    )
    plot_anharmonic_validation(harmonic_anchor, stable_anh_model, anh_valid, out_dir)
    plot_reduced_load_paths(load_paths, stable_anh_model.coordinate_scale, out_dir)

    # --- Write energy scan CSV ---
    csv_energy = os.path.join(out_dir, 'energy_scans.csv')
    energy_rows = []
    for d in range(6):
        amps, dE_uff = scan_dof_energies(eval_fn, pos0, c1, c2, s, E0, ell, d)
        for k in range(len(amps)):
            row = [d, DOF_LABELS[d], amps[k], dE_uff[k]]
            for name, K in Ks_restricted.items():
                row.append(harmonic_energy_1dof(K, ell, d, np.array([amps[k]]))[0])
            row.append(polynomial_energy_1dof(stable_anh_model, ell, d, np.array([amps[k]]))[0])
            energy_rows.append(row)
    energy_header = (
        ['dof_idx', 'dof_label', 'amplitude', 'dE_uff']
        + [f'dE_{name}' for name in Ks_restricted]
        + ['dE_Anharmonic']
    )
    write_csv(csv_energy, energy_header, energy_rows)
    print(f"\nEnergy scan CSV written: {csv_energy}")

    # --- Write rigid mode CSV ---
    csv_rigid = os.path.join(out_dir, 'rigid_mode_check.csv')
    rigid_rows = []
    for d in range(6):
        _, dE_true = scan_rigid_energies(eval_fn, pos0, c1, c2, s, E0, d)
        _, dE_frame = scan_rigid_frame_energies(eval_fn, pos0, c1, c2, s, E0, d)
        rigid_rows.append([d, DOF_LABELS[d], np.abs(dE_true).max(), np.abs(dE_frame).max(), 0.0])
    write_csv(csv_rigid, ['dof_idx', 'dof_label', 'max_dE_true_rigid', 'max_dE_frame_rigid', 'max_dE_model'],
              rigid_rows)
    print(f"Rigid mode CSV written: {csv_rigid}")

    print(f"\nLog file written to: {log_path}")
    sys.stdout.log.close()
    sys.stdout = sys.__stdout__
    print(f"Done. Log: {log_path}, CSVs in {out_dir}")


if __name__ == '__main__':
    main()
