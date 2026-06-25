"""
Test resolvent iterative solvers on nanocrystal vibration benchmarks.

Plots exact eigenvalues (vertical red lines) vs estimated spectrum from
resolvent solve:  ||(K - omega^2 M)^{-1} b||.

Usage:
    python test_resolvent_spectrum.py --system nc_C_R5 --solver minres --n_freq 200
    python test_resolvent_spectrum.py --system nc_C_R4 --solver cocr --eta 0.5 --n_freq 200
    python test_resolvent_spectrum.py --system nc_C_R5 --solver block_jacobi --max_iter 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spectral_solvers import load_vibration_benchmark, resolve_benchmark_path
from resolvent_solvers import (
    ResolventOperator,
    BlockJacobiPreconditioner,
    batched_resolvent_sweep,
    stochastic_dos_sweep,
)


def parse_args():
    p = argparse.ArgumentParser(description="Resolvent spectrum scan on nanocrystal benchmarks")
    p.add_argument("--system", default="nc_C_R5",
                   choices=["adamantane", "nc_C_R4", "nc_C_R5", "nc_C_R6", "nc_C_R7", "nc_C_R8"])
    p.add_argument("--solver", default="minres",
                   choices=["block_jacobi", "minres", "cocr", "bicgstab", "gmres"])
    p.add_argument("--n_freq", type=int, default=200, help="Number of frequency samples")
    p.add_argument("--eta", type=float, default=0.01, help="Damping (imaginary part of frequency). Must be << eigenvalue spacing to see peaks.")
    p.add_argument("--tol", type=float, default=1e-5, help="Solver relative tolerance")
    p.add_argument("--max_iter", type=int, default=500, help="Max iterations per solve")
    p.add_argument("--restart", type=int, default=30, help="GMRES restart")
    p.add_argument("--backend", default="cpu", choices=["cpu", "opencl"])
    p.add_argument("--no_prec", action="store_true", help="Disable block-Jacobi preconditioner")
    p.add_argument("--no_warm", action="store_true", help="Disable warm-start across frequencies")
    p.add_argument("--n_probes", type=int, default=1, help="Number of random probes to average (smooths noise)")
    p.add_argument("--mass_weighted", action="store_true", help="Use mass-weighted random probe b = M^{1/2} * randn")
    p.add_argument("--response_mode", default="proj_im", 
                   choices=["norm", "proj_im", "proj_abs", "proj_real"],
                   help="Response observable: norm=||x||, proj_im=Im(b^H x) for DOS")
    p.add_argument("--verify_exact", action="store_true",
                   help="Adamantane only: compare iterative response against dense solve and exact eig projection")
    p.add_argument("--save", default="resolvent_spectrum.png")
    p.add_argument("--fixtures_dir", default=None)
    return p.parse_args()


def gaussian_blur_spectrum(omegas_grid, omegas_exact, sigma=None, eta=0.01):
    """Return blurred exact DOS: sum_j exp(-(omega - omega_j)^2 / (2*sigma^2)).
    If sigma is None, use eta as blur width."""
    if sigma is None:
        sigma = max(eta * 0.5, 1e-6)
    blur = np.zeros_like(omegas_grid, dtype=float)
    for wj in omegas_exact:
        blur += np.exp(-0.5 * ((omegas_grid - wj) / sigma) ** 2)
    blur /= (sigma * np.sqrt(2 * np.pi))  # normalize so integral ~ num_modes
    return blur


def exact_lorentzian_dos(omegas, omegas_exact, eta, omega_density=True):
    """Return exact Lorentzian DOS: sum_j Im[1/(lambda_j - z)] where z=(omega+i*eta)^2."""
    lambdas = omegas_exact**2
    out = np.zeros_like(omegas, dtype=float)
    for i, w in enumerate(omegas):
        z = (w + 1j*eta)**2
        vals = 1.0 / (lambdas - z)
        out[i] = np.imag(np.sum(vals))
        if omega_density:
            out[i] *= 2*w  # Jacobian from lambda to omega
    return out


def plot_resolvent_spectrum(omegas, responses, omegas_exact, solver, eta, max_iter,
                            info_list, save_path, blur_sigma=None, response_mode="norm"):
    """Plot spectrum estimate with exact DOS and residual error diagnostics."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1.5, 1]})

    # Top: Impulse response comparison
    ax = axes[0]
    # Exact Lorentzian DOS
    exact_dos = exact_lorentzian_dos(omegas, omegas_exact, eta, omega_density=False)
    # Normalize both to max=1 for easy visual comparison
    if np.max(np.abs(responses)) > 1e-30:
        resp_norm = responses / (np.max(np.abs(responses)))
    else:
        resp_norm = responses
    if np.max(exact_dos) > 1e-30:
        dos_norm = exact_dos / np.max(exact_dos)
    else:
        dos_norm = exact_dos

    response_label = f"Impulse response ({response_mode})"
    ax.plot(omegas, resp_norm, "b-", linewidth=1.2, label=f"{response_label} - {solver}")
    ax.plot(omegas, dos_norm, "g--", linewidth=1.2, label="Exact Lorentzian (Im Tr[(H-zI)^{-1}])")
    for w in omegas_exact:
        ax.axvline(w, color="red", linestyle=":", alpha=0.3, linewidth=0.5)
    ax.plot([], [], color="red", linestyle=":", alpha=0.3, label="Exact eigenvalues")

    ax.set_xlabel("Frequency omega")
    ax.set_ylabel("Normalized response")
    ax.set_title(f"Impulse Response Spectrum: {solver}, eta={eta}, max_iter={max_iter}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Middle: Residual error diagnostics
    ax2 = axes[1]
    iters = [inf.get("iter", max_iter) for inf in info_list]
    residuals = [inf.get("residual", None) for inf in info_list]
    true_residuals = [inf.get("true_residual", None) for inf in info_list]
    converged = [inf.get("converged", False) for inf in info_list]

    ax2.scatter(omegas, iters, c=["green" if c else "orange" for c in converged],
                s=8, alpha=0.6, label="Iterations")
    ax2.set_xlabel("Frequency omega")
    ax2.set_ylabel("Iterations to converge")
    ax2.set_title(f"Solver Convergence: converged={sum(converged)}/{len(converged)}")
    ax2.grid(True, alpha=0.3)

    # Residual scatter on secondary y
    if any(r is not None for r in residuals):
        ax2r = ax2.twinx()
        mask = [r is not None for r in residuals]
        o_m = omegas[mask]
        r_m = np.array([r for r in residuals if r is not None])
        ax2r.scatter(o_m, r_m, c="red", s=3, alpha=0.4, label="Preconditioned residual")
        
        # True residual if available
        if any(tr is not None for tr in true_residuals):
            mask_tr = [tr is not None for tr in true_residuals]
            o_tr = omegas[mask_tr]
            tr_m = np.array([tr for tr in true_residuals if tr is not None])
            ax2r.scatter(o_tr, tr_m, c="blue", s=3, alpha=0.4, label="True residual")
        
        ax2r.set_ylabel("Relative residual", color="red")
        ax2r.set_yscale("log")
        ax2r.legend(loc="upper right")

    # Bottom: Raw response (not normalized)
    ax3 = axes[2]
    ax3.plot(omegas, responses, "b-", linewidth=1.2, label=f"Raw response ({response_mode})")
    ax3.set_xlabel("Frequency omega")
    ax3.set_ylabel("Response magnitude")
    ax3.set_title("Raw impulse response (no normalization)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")


def plot_solver_comparison(omegas, all_responses, solver_names, omegas_exact, eta, save_path):
    """Overlay multiple solver spectra."""
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(solver_names)))
    for resp, name, color in zip(all_responses, solver_names, colors):
        ax.plot(omegas, resp, "-", color=color, linewidth=1.2, label=name)
    for w in omegas_exact:
        ax.axvline(w, color="black", linestyle="--", alpha=0.3, linewidth=0.6)
    ax.set_xlabel("Frequency omega")
    ax.set_ylabel("Response ||x(omega)||")
    ax.set_title(f"Solver comparison (eta={eta})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved comparison to {save_path}")


def run_single(args):
    print(f"=== Loading {args.system} ===")
    path = resolve_benchmark_path(args.system, fixtures_dir=args.fixtures_dir)
    bench = load_vibration_benchmark(path)
    K = bench["K"]
    mass = bench["mass"]
    ndof = bench["ndof"]
    natoms = ndof // 3
    print(f"System: {bench['name']}, atoms={natoms}, DOF={ndof}, nnz={K.nnz}")

    # Exact eigenvalues from benchmark (avoid double mass-weighting)
    print("\n--- Exact eigenvalues from benchmark ---")
    t0 = time.time()
    omegas_exact = bench["omegas_vib"].copy()
    print(f"Exact: {len(omegas_exact)} vibrational eigenvalues in {time.time()-t0:.3f}s")
    print(f"Omega range: {omegas_exact.min():.4f} .. {omegas_exact.max():.4f}")
    
    # Optional: verify against recomputed for debugging
    if False:  # Set to True to verify mass-weighting is correct
        print("\n--- Verifying mass-weighting ---")
        m_inv_sqrt = 1.0 / np.sqrt(mass)
        if bench.get("K_dense") is not None:
            Kd = bench["K_dense"]
        else:
            Kd = K.toarray()
        H = (m_inv_sqrt[:, None] * Kd) * m_inv_sqrt[None, :]
        H = 0.5 * (H + H.T)
        eigs = np.linalg.eigvalsh(H)
        omegas_recomputed = np.sqrt(np.clip(eigs, 0, None))
        vib_mask = omegas_recomputed < 100.0
        omegas_recomputed = omegas_recomputed[vib_mask]
        print(f"Recomputed: {len(omegas_recomputed)} eigenvalues")
        print(f"Max diff: {np.max(np.abs(np.sort(omegas_exact) - np.sort(omegas_recomputed))):.6f}")

    # Frequency sweep
    omega_min = omegas_exact.min() * 0.9
    omega_max = omegas_exact.max() * 1.1
    omegas = np.linspace(omega_min, omega_max, args.n_freq)
    print(f"Sweep: {args.n_freq} frequencies from {omega_min:.4f} to {omega_max:.4f}")

    # Resolvent operator
    print(f"\n--- Building resolvent operator (backend={args.backend}) ---")
    resolvent = ResolventOperator(K, mass, backend=args.backend)

    # Preconditioner
    prec = None if args.no_prec else BlockJacobiPreconditioner(resolvent, eps_rel=1e-8)

    # Multiple probes for averaging
    responses_acc = np.zeros(args.n_freq)
    info_list = None
    print(f"\n--- Running {args.solver} sweep with {args.n_probes} probe(s) ---")
    t0 = time.time()
    for p in range(args.n_probes):
        np.random.seed(42 + p)
        b = np.random.randn(ndof)
        if args.mass_weighted:
            b = b * np.sqrt(mass)
        if args.eta > 0:
            b = b.astype(complex)
        b /= np.linalg.norm(b)

        resp, x_all, info_list = batched_resolvent_sweep(
            omegas, resolvent, b,
            solver=args.solver,
            eta=args.eta,
            tol=args.tol,
            max_iter=args.max_iter,
            restart=args.restart,
            preconditioner=prec,
            warm_start=not args.no_warm,
            response_mode=args.response_mode,
        )
        responses_acc += resp
    responses = responses_acc / args.n_probes
    t_solve = time.time() - t0
    print(f"Sweep done in {t_solve:.2f}s ({t_solve/(args.n_freq*args.n_probes):.3f}s per solve)")
    print(f"Total SpMV count: {resolvent.spmv_count}")

    conv = sum(inf.get("converged", False) for inf in info_list)
    print(f"Converged: {conv}/{args.n_freq}")
    avg_iter = np.mean([inf.get("iter", args.max_iter) for inf in info_list])
    print(f"Avg iterations: {avg_iter:.1f}")

    if args.verify_exact:
        if args.system != "adamantane":
            raise RuntimeError("--verify_exact is only supported for adamantane")
        if args.response_mode != "proj_im":
            raise RuntimeError("--verify_exact expects --response_mode proj_im")
        if args.eta <= 0:
            raise RuntimeError("--verify_exact expects eta > 0")

        print("\n--- Exact verification (dense + eig projection) ---")
        Kd = K.toarray()
        A_dense_blocks = []
        s_dense = np.zeros(args.n_freq, dtype=float)
        s_exact = np.zeros(args.n_freq, dtype=float)

        m_inv_sqrt = 1.0 / np.sqrt(mass)
        H = (m_inv_sqrt[:, None] * Kd) * m_inv_sqrt[None, :]
        H = 0.5 * (H + H.T)
        lam, V = np.linalg.eigh(H)
        q = (b / np.sqrt(mass)).astype(complex)
        coeff = V.T @ q

        for i, w in enumerate(omegas):
            z = (w + 1j * args.eta) ** 2
            A = Kd - z * np.diag(mass)
            x = np.linalg.solve(A, b)
            s_dense[i] = np.imag(np.vdot(b, x))
            s_exact[i] = np.imag(np.sum((np.abs(coeff) ** 2) / (lam - z)))

        resp_norm = responses / (np.max(np.abs(responses)) + 1e-30)
        dense_norm = s_dense / (np.max(np.abs(s_dense)) + 1e-30)
        exact_norm = s_exact / (np.max(np.abs(s_exact)) + 1e-30)
        cos_dense_exact = float(np.dot(dense_norm, exact_norm) / ((np.linalg.norm(dense_norm) * np.linalg.norm(exact_norm)) + 1e-30))
        cos_iter_dense = float(np.dot(resp_norm, dense_norm) / ((np.linalg.norm(resp_norm) * np.linalg.norm(dense_norm)) + 1e-30))
        print(f"Cosine(dense, eig-proj)={cos_dense_exact:.4f}  Cosine(iter, dense)={cos_iter_dense:.4f}")

        fig, axes = plt.subplots(4, 1, figsize=(12, 12), gridspec_kw={"height_ratios": [2, 2, 1.5, 1]})
        ax = axes[0]
        ax.plot(omegas, exact_norm, "k-", linewidth=1.5, label="Exact eig-proj Im(b^H x)")
        ax.plot(omegas, dense_norm, "g--", linewidth=1.2, label="Dense solve Im(b^H x)")
        ax.plot(omegas, resp_norm, "b-", linewidth=1.2, label=f"Iterative ({args.solver}) Im(b^H x)")
        for w in omegas_exact:
            ax.axvline(w, color="red", linestyle=":", alpha=0.2, linewidth=0.5)
        ax.set_title(f"Exact verification: eta={args.eta}, solver={args.solver}")
        ax.set_ylabel("Normalized")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        ax1 = axes[1]
        ax1.plot(omegas, s_dense, "g--", linewidth=1.2, label="Dense solve")
        ax1.plot(omegas, responses, "b-", linewidth=1.2, label="Iterative")
        ax1.set_ylabel("Im(b^H x)")
        ax1.set_title("Raw (no normalization)")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        ax2 = axes[2]
        err = np.abs(responses - s_dense) / (np.max(np.abs(s_dense)) + 1e-30)
        ax2.plot(omegas, err, "r-", linewidth=1.0)
        ax2.set_yscale("log")
        ax2.set_ylabel("Rel err vs dense")
        ax2.grid(True, alpha=0.3)

        ax3 = axes[3]
        tr = np.array([inf.get("true_residual", np.nan) for inf in info_list], dtype=float)
        ax3.plot(omegas, tr, "m-", linewidth=1.0)
        ax3.set_yscale("log")
        ax3.set_xlabel("Frequency omega")
        ax3.set_ylabel("True residual")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(args.save, dpi=150)
        print(f"Saved verification plot to {args.save}")
    else:
        # Plot
        print("\n--- Plotting ---")
        plot_resolvent_spectrum(omegas, responses, omegas_exact, args.solver,
                               args.eta, args.max_iter, info_list, args.save,
                               blur_sigma=args.eta * 2.0, response_mode=args.response_mode)


def run_comparison(args):
    """Compare all solvers on the same sweep."""
    print(f"=== Comparison on {args.system} ===")
    path = resolve_benchmark_path(args.system, fixtures_dir=args.fixtures_dir)
    bench = load_vibration_benchmark(path)
    K = bench["K"]; mass = bench["mass"]; ndof = bench["ndof"]

    omegas_exact = bench["omegas_vib"].copy()

    omega_min = omegas_exact.min() * 0.9
    omega_max = omegas_exact.max() * 1.1
    omegas = np.linspace(omega_min, omega_max, args.n_freq)

    np.random.seed(42)
    b = np.random.randn(ndof)
    if args.eta > 0:
        b = b.astype(complex)
    b /= np.linalg.norm(b)

    resolvent = ResolventOperator(K, mass, backend="cpu")
    prec = None if args.no_prec else BlockJacobiPreconditioner(resolvent)

    all_resp = []
    names = []
    for solver in ["block_jacobi", "minres", "cocr", "bicgstab", "gmres"]:
        if solver == "cocr" and args.eta == 0:
            # COCR needs complex matrix; skip in real mode or add tiny eta
            continue
        print(f"\n--- {solver} ---")
        t0 = time.time()
        resp, _, info_list = batched_resolvent_sweep(
            omegas, resolvent, b, solver=solver, eta=args.eta,
            tol=args.tol, max_iter=args.max_iter, restart=args.restart,
            preconditioner=prec, warm_start=not args.no_warm,
        )
        t = time.time() - t0
        conv = sum(inf.get("converged", False) for inf in info_list)
        avg_iter = np.mean([inf.get("iter", args.max_iter) for inf in info_list])
        print(f"  {t:.2f}s, converged={conv}/{args.n_freq}, avg_iter={avg_iter:.1f}")
        all_resp.append(resp)
        names.append(solver)

    save = args.save.replace(".png", "_comparison.png")
    plot_solver_comparison(omegas, all_resp, names, omegas_exact, args.eta, save)


def main():
    args = parse_args()
    run_single(args)
    # Optionally also run comparison
    # run_comparison(args)


if __name__ == "__main__":
    main()
