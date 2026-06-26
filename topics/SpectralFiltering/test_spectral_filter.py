"""
Test spectral filtering (KPM Chebyshev) and sub-interval eigenvector solvers
on nanocrystal vibration benchmarks.

Usage:
    python test_spectral_filter.py --system nc_C_R5
    python test_spectral_filter.py --system nc_C_R6 --bands 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))

from spectral_solvers import (
    build_vibration_operator,
    chebyshev_filter_omega,
    solve_band,
    solve_spectrum,
    omega_band_edges,
    scaled_bands_from_omega,
    rayleigh_ritz,
    op_matmul,
    load_vibration_benchmark,
    resolve_benchmark_path,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="nc_C_R5",
                   choices=["adamantane", "nc_C_R4", "nc_C_R5", "nc_C_R6", "nc_C_R7", "nc_C_R8"])
    p.add_argument("--fixtures_dir", default=None)
    p.add_argument("--n_probe", type=int, default=32,
                   help="Number of random probe vectors for KPM filter")
    p.add_argument("--cheb_degrees", type=int, default=100,
                   help="Chebyshev polynomial degree for KPM filter")
    p.add_argument("--bands", type=int, default=4,
                   help="Number of omega sub-bands for interval solver")
    p.add_argument("--nvec_per_band", type=int, default=16,
                   help="Probe vectors per band for interval solver")
    p.add_argument("--solve_iters", type=int, default=5,
                   help="Filter->Ritz refinement rounds per band")
    p.add_argument("--save", default="/tmp/spectral_filter_test.png")
    return p.parse_args()


def test_chebyshev_filter(op, bench, args):
    """KPM Chebyshev spectral density estimation."""
    print("\n--- Chebyshev spectral filter (KPM) ---")
    omegas = bench["omegas_vib"]
    omega_lo, omega_hi = float(omegas.min()), float(omegas.max())
    nfreq = 200
    omega_grid = np.linspace(omega_lo, omega_hi, nfreq)

    V0 = np.random.randn(op.ndim, args.n_probe)
    t0 = time.time()
    total_amps, vec_amps = chebyshev_filter_omega(
        op, V0, omega_grid, iters=[args.cheb_degrees], use_jackson=True
    )
    t_kpm = time.time() - t0
    print(f"KPM filter: {t_kpm:.3f}s  (degree={args.cheb_degrees}, probes={args.n_probe})")

    # Compare to exact histogram
    amps = total_amps[args.cheb_degrees]
    # Normalize
    amps = amps / (amps.sum() + 1e-30)
    exact_hist, bins = np.histogram(omegas, bins=nfreq, range=(omega_lo, omega_hi), density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # KPM amplitudes are not a direct density; we just show shape convergence
    print(f"  KPM max amp at omega={omega_grid[np.argmax(amps)]:.3f}")
    print(f"  Exact max density at omega={bin_centers[np.argmax(exact_hist)]:.3f}")
    return omega_grid, amps, bin_centers, exact_hist


def test_subband_solver(op, bench, args):
    """Band-by-band eigenvector extraction via Chebyshev filter + Rayleigh-Ritz."""
    print("\n--- Sub-band eigenvector solver ---")
    omegas = bench["omegas_vib"]
    omega_lo, omega_hi = float(omegas.min()), float(omegas.max())
    band_width = (omega_hi - omega_lo) / args.bands
    lo_edges, hi_edges = omega_band_edges(omegas, band_width)

    # Convert to scaled coordinates
    band_lo_s = op.omega_to_scaled(lo_edges)
    band_hi_s = op.omega_to_scaled(hi_edges)

    t0 = time.time()
    result = solve_spectrum(
        op,
        band_lo=band_lo_s,
        band_hi=band_hi_s,
        nvec=args.nvec_per_band,
        coarse_iters=60,
        conv_iters=args.solve_iters,
        filter_reps=1,
        square_filter=False,
        method="ritz",
        prune_mode="hybrid",
        prune_tol=0.5,
        cluster_tol=0.85,
        cluster_eig_tol=0.05,
        gradual_factor=3.0,
        rank_est=False,
        res_tol=1e-3,
        exact_w=op.exact_scaled_eigs(),
    )
    t_solve = time.time() - t0
    print(f"Sub-band solver: {t_solve:.3f}s  ({args.bands} bands, {args.nvec_per_band} probes/band)")
    print(f"  Total SpMV: {result['total_spmv']}")

    # Collect converged eigenvalues from all bands
    found_omegas = []
    found_residuals = []
    for bi in range(args.bands):
        for tid, traj in result["final_traj"][bi].items():
            # Take the last entry of each trajectory
            it, w, r, kept = traj[-1]
            if kept and r < 1e-2:
                # Convert scaled omega back to physical omega
                found_omegas.append(float(w))
                found_residuals.append(float(r))

    found_omegas = np.array(sorted(found_omegas))
    found_residuals = np.array(found_residuals)
    print(f"  Converged eigenvalues: {len(found_omegas)}")
    if len(found_residuals) > 0:
        print(f"  Residual range: {found_residuals.min():.2e} .. {found_residuals.max():.2e}")

    # Compare to exact
    exact = np.sort(omegas)
    n = min(len(found_omegas), len(exact))
    if n > 0:
        err = np.abs(found_omegas[:n] - exact[:n])
        print(f"  Match error (first {n}): max={err.max():.4f}  mean={err.mean():.4f}")

    return result, found_omegas, found_residuals


def plot_results(omega_grid, kpm_amps, bin_centers, exact_hist,
                 found_omegas, found_residuals, omegas_exact, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: KPM amplitude vs exact density
    ax = axes[0]
    ax.plot(omega_grid, kpm_amps / (kpm_amps.max() + 1e-30), label="KPM amplitude", color="steelblue")
    ax.plot(bin_centers, exact_hist / (exact_hist.max() + 1e-30), label="Exact DOS", color="crimson", lw=1)
    ax.set_xlabel("Omega")
    ax.set_ylabel("Normalized amplitude / DOS")
    ax.set_title("Chebyshev KPM spectral density")
    ax.legend()

    # Panel 2: Sub-band solver found eigenvalues vs exact
    ax = axes[1]
    ax.scatter(range(len(omegas_exact)), omegas_exact, s=5, color="black", alpha=0.5, label="Exact")
    if len(found_omegas) > 0:
        ax.scatter(range(len(found_omegas)), found_omegas, s=15, color="crimson", edgecolors="none", label=f"Found ({len(found_omegas)})")
    ax.set_xlabel("Mode index (sorted)")
    ax.set_ylabel("Omega")
    ax.set_title("Sub-band eigenvalue extraction")
    ax.legend()

    # Panel 3: Residuals of found eigenvalues
    ax = axes[2]
    if len(found_residuals) > 0:
        ax.semilogy(range(len(found_residuals)), np.sort(found_residuals), "o-", color="forestgreen")
    ax.set_xlabel("Found eigenvalue index")
    ax.set_ylabel("Residual")
    ax.set_title("Sub-band solver residuals")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")


def run(args):
    print(f"=== Loading {args.system} ===")
    op, bench = build_vibration_operator(
        args.system, fixtures_dir=args.fixtures_dir, backend="cpu", use_spectral_basis=True
    )
    ndof = op.ndim
    omegas = bench["omegas_vib"]
    print(f"System: {args.system}, DOF={ndof}, vibrational modes={len(omegas)}")
    print(f"Omega range: {omegas.min():.4f} .. {omegas.max():.4f}")

    # 1) Chebyshev KPM filter
    omega_grid, kpm_amps, bin_centers, exact_hist = test_chebyshev_filter(op, bench, args)

    # 2) Sub-band eigenvector solver
    result, found_omegas, found_residuals = test_subband_solver(op, bench, args)

    # 3) Plot
    plot_results(omega_grid, kpm_amps, bin_centers, exact_hist,
                 found_omegas, found_residuals, np.sort(omegas), args.save)

    print("\nDone.")


if __name__ == "__main__":
    run(parse_args())
