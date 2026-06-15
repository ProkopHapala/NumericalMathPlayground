import os
import time
import numpy as np
import argparse

from spectral_solvers import (
    generate_test_matrix,
    get_exact_eigenvalues,
    cheb_rect_coeffs,
    eval_cheb_series,
    apply_cheb_poly,
    rayleigh_ritz,
    lanczos_band,
    cluster_prune_mask,
    cheap_rank_estimate,
    match_trajectories,
    apply_pruning,
    solve_spectrum,
    solve_band,
    chebyshev_filter,
    chebyshev_filter_omega,
    power_iteration_filter,
    build_vibration_operator,
    BENCHMARK_SYSTEMS,
    DEFAULT_FIXTURES_DIR,
    omega_band_edges,
    scaled_bands_from_omega,
)

from spectral_plotting import (
    save_or_show,
    plot_debug_rectangle,
    plot_convergence_3panel,
    plot_spectral_evolution,
)


def _setup_problem(args):
    """Return (H, V0, exact_eigs_plot, bench_meta, op_or_none)."""
    if args.benchmark:
        backend = "opencl" if args.opencl else "cpu"
        op, bench = build_vibration_operator(
            args.benchmark, fixtures_dir=args.fixtures_dir, backend=backend,
            use_spectral_basis=not args.sparse_k,
        )
        n = op.ndim
        nvec = args.nvec
        V0 = np.random.randn(n, nvec)
        exact_plot = bench["omegas_vib"]
        meta = {
            "name": bench["name"],
            "ndof": bench["ndof"],
            "omega_range": (op.omega_min, op.omega_max),
            "backend": op.backend,
            "spmv_count": 0,
        }
        return op, V0, exact_plot, meta, op

    H = generate_test_matrix(args.size)
    V0 = np.random.randn(args.size, args.nvec)
    exact_plot = get_exact_eigenvalues(H)
    return H, V0, exact_plot, {"name": f"laplacian_N{args.size}"}, None


def _band_edges(args, exact_plot, op):
    """Band edges for filtering: omega windows (benchmark) or scaled [-0.95,0.95]."""
    if op is not None:
        width = args.band_width
        omega_lo, omega_hi = omega_band_edges(exact_plot, width)
        band_lo, band_hi = scaled_bands_from_omega(op, omega_lo, omega_hi)
        band_lo = np.clip(band_lo, -0.999, 0.999)
        band_hi = np.clip(band_hi, -0.999, 0.999)
        return band_lo, band_hi, omega_lo, omega_hi
    edges = np.linspace(-0.95, 0.95, args.nbands + 1)
    return edges[:-1], edges[1:], None, None


def _run_spectral_filter(H, V0, exact_plot, args, op, meta, save_path):
    if op is not None:
        omegas = np.linspace(op.omega_min, op.omega_max, args.spectral_nfreq)
        iters_to_plot = [int(s) for s in args.spectral_iters.split(",") if s.strip()]
        use_jackson = args.spectral_jackson and not args.spectral_no_jackson
        normalize = args.spectral_normalize and not args.spectral_no_normalize
        t0 = time.perf_counter()
        if args.spectral_method == "chebyshev":
            total_amps, vec_amps = chebyshev_filter_omega(
                H, V0, omegas, iters_to_plot, use_jackson=use_jackson,
            )
        else:
            freqs_scaled = np.clip(op.omega_to_scaled(omegas), -0.999, 0.999)
            total_amps, vec_amps = power_iteration_filter(H, V0, freqs_scaled, iters_to_plot)
        elapsed = time.perf_counter() - t0
        freqs_plot = omegas
        x_label = "ω (internal MMFF units)"
        title_suffix = meta["name"]
        print(f"  KPM spectral filter: {elapsed:.2f}s, SpMVs={H.spmv_count:,}, backend={H.backend}")
    else:
        freqs_plot = np.linspace(-0.99, 0.99, args.spectral_nfreq)
        iters_to_plot = [int(s) for s in args.spectral_iters.split(",") if s.strip()]
        use_jackson = args.spectral_jackson and not args.spectral_no_jackson
        normalize = args.spectral_normalize and not args.spectral_no_normalize
        if args.spectral_method == "chebyshev":
            total_amps, vec_amps = chebyshev_filter(H, V0, freqs_plot, iters_to_plot, use_jackson=use_jackson)
        else:
            total_amps, vec_amps = power_iteration_filter(H, V0, freqs_plot, iters_to_plot)
        x_label = "scaled eigenvalue"
        title_suffix = meta["name"]

    plot_spectral_evolution(
        freqs_plot, total_amps, vec_amps, iters_to_plot,
        exact_eigs=exact_plot, method=args.spectral_method, normalize=normalize,
        save_path=save_path, x_label=x_label, title_suffix=title_suffix,
    )


def _run_converge_spectrum(H, V0, exact_plot, args, op, meta, save_path):
    band_lo, band_hi, omega_lo, omega_hi = _band_edges(args, exact_plot, op)
    nbands = len(band_lo)
    exact_w = get_exact_eigenvalues(H)

    result = solve_spectrum(
        H, band_lo, band_hi, args.nvec, args.coarse_iters, args.conv_iters,
        args.filter_reps, args.square_filter, args.method, args.prune_mode,
        args.prune_tol, args.cluster_tol, args.cluster_eig_tol, args.gradual_factor,
        rank_est=args.rank_est, res_tol=args.res_tol, exact_w=exact_w,
    )

    band_raw_w = result["band_raw_w"]
    band_raw_r = result["band_raw_r"]
    final_traj = result["final_traj"]
    total_spmv = result["total_spmv"]
    if op is not None:
        total_spmv = H.spmv_count
        plot_lo = omega_lo if omega_lo is not None else band_lo
        plot_hi = omega_hi if omega_hi is not None else band_hi
        exact_w_plot = exact_plot
        x_label = "ω (internal MMFF units)"
        title_suffix = meta["name"]
    else:
        plot_lo, plot_hi = band_lo, band_hi
        exact_w_plot = exact_w
        x_label = "scaled eigenvalue"
        title_suffix = meta["name"]

    print(f"\n=== SpMV cost summary ({meta['name']}) ===")
    print(f"  nbands={nbands}, TOTAL SpMVs={total_spmv:,}")
    if op is not None:
        print(f"  backend={op.backend}, band_width={args.band_width}")

    print(f"\nConverged (last-iter, residual < {args.res_tol})")
    for bi in range(nbands):
        for tid, pts in final_traj[bi].items():
            if not pts:
                continue
            last = pts[-1]
            if last[3] and last[2] < args.res_tol:
                print(f"  band={bi}  omega={last[1]:+.6f}  resid={last[2]:.3e}")

    plot_convergence_3panel(
        plot_lo, plot_hi, exact_w_plot, band_raw_w, band_raw_r,
        result["band_keep"], final_traj,
        args.coarse_iters, args.square_filter, args.filter_reps, args.conv_iters,
        args.method, args.prune_mode, args.nvec, save_path,
        x_label=x_label, title_suffix=title_suffix,
        band_lo_filt=band_lo if op is not None else None,
        band_hi_filt=band_hi if op is not None else None,
    )


def _run_benchmark_suite(args):
    """Run spectral_filter + converge_spectrum for R4–R8 (or --benchmark_suite list)."""
    systems = BENCHMARK_SYSTEMS if args.benchmark_suite == "all" else [
        s.strip() for s in args.benchmark_suite.split(",") if s.strip()
    ]
    out_dir = args.save_dir or "."
    os.makedirs(out_dir, exist_ok=True)

    for system in systems:
        if system in ("adamantane",) and args.skip_adamantane:
            continue
        args.benchmark = system
        print(f"\n{'='*60}\nBenchmark: {system}\n{'='*60}")
        H, V0, exact_plot, meta, op = _setup_problem(args)

        if args.spectral_filter or args.run_all:
            save = os.path.join(out_dir, f"{system}_kpm.png")
            print(f"--- spectral filter -> {save}")
            _run_spectral_filter(H, V0, exact_plot, args, op, meta, save)

        if args.converge_spectrum or args.run_all:
            save = os.path.join(out_dir, f"{system}_converge.png")
            print(f"--- converge spectrum -> {save}")
            _run_converge_spectrum(H, V0, exact_plot, args, op, meta, save)


def main():
    parser = argparse.ArgumentParser(description="Spectral filtering demos (toy or MMFF vibration benchmarks)")
    parser.add_argument("--size", type=int, default=20, help="Laplacian matrix size (non-benchmark mode)")
    parser.add_argument("--nvec", type=int, default=8, help="Random probe vectors")
    parser.add_argument("--coarse_iters", type=int, default=60, help="Chebyshev degree for band-pass filter")
    parser.add_argument("--fine_iters", type=int, default=30, help="(unused) legacy Jacobi iterations")
    parser.add_argument("--nbands", type=int, default=5, help="Coarse bands (Laplacian mode only)")
    parser.add_argument("--band_width", type=float, default=0.15,
                        help="Omega window width for benchmark band solver")
    parser.add_argument("--nfine", type=int, default=200, help="(unused)")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory for benchmark suite")
    parser.add_argument("--debug_rect", action="store_true")
    parser.add_argument("--rect_deg_max", type=int, default=128)
    parser.add_argument("--rect_degs", type=str, default="4,8,16,32,64,128")
    parser.add_argument("--rect_plot", type=str, default="square", choices=["signed", "abs", "square"])
    parser.add_argument("--solve_band", type=int, default=-1)
    parser.add_argument("--solve_iters", type=int, default=2)
    parser.add_argument("--square_filter", action="store_true")
    parser.add_argument("--filter_reps", type=int, default=1)
    parser.add_argument("--converge_spectrum", action="store_true")
    parser.add_argument("--conv_iters", type=int, default=6)
    parser.add_argument("--res_tol", type=float, default=1e-3)
    parser.add_argument("--method", type=str, default="ritz", choices=["ritz", "lanczos"])
    parser.add_argument("--prune_mode", type=str, default="gradual",
                        choices=["none", "residual", "cluster", "gradual", "hybrid"])
    parser.add_argument("--prune_tol", type=float, default=1e-2)
    parser.add_argument("--cluster_tol", type=float, default=0.85)
    parser.add_argument("--cluster_eig_tol", type=float, default=0.05)
    parser.add_argument("--gradual_factor", type=float, default=3.0)
    parser.add_argument("--rank_est", action="store_true")
    parser.add_argument("--spectral_filter", action="store_true")
    parser.add_argument("--spectral_method", choices=["chebyshev", "power"], default="chebyshev")
    parser.add_argument("--spectral_iters", type=str, default="4,8,16,32,64,128")
    parser.add_argument("--spectral_nfreq", type=int, default=800)
    parser.add_argument("--spectral_normalize", type=int, default=1)
    parser.add_argument("--spectral_no_normalize", type=int, default=0)
    parser.add_argument("--spectral_jackson", type=int, default=1)
    parser.add_argument("--spectral_no_jackson", type=int, default=0)

    # Vibration benchmark options
    parser.add_argument("--benchmark", type=str, default=None,
                        help="System name (nc_C_R5) or path to .npz")
    parser.add_argument("--fixtures_dir", type=str, default=DEFAULT_FIXTURES_DIR)
    parser.add_argument("--benchmark_suite", type=str, default=None,
                        help="Comma-separated systems or 'all' for R4-R8 + adamantane")
    parser.add_argument("--run_all", action="store_true",
                        help="With --benchmark_suite: run KPM + converge for each system")
    parser.add_argument("--skip_adamantane", action="store_true")
    parser.add_argument("--opencl", action="store_true",
                        help="Use OpenCL CSR SpMM for K @ V (requires pyopencl; use with --sparse_k)")
    parser.add_argument("--sparse_k", action="store_true",
                        help="Sparse K matvec instead of vibrational spectral basis (experimental)")

    args = parser.parse_args()

    if args.benchmark_suite:
        _run_benchmark_suite(args)
        return

    H, V0, exact_plot, meta, op = _setup_problem(args)
    print(f"\n--- Spectral solver: {meta['name']}  nvec={args.nvec}  prune={args.prune_mode} ---")
    if op is not None:
        print(f"  ndof={meta['ndof']}, ω∈[{meta['omega_range'][0]:.4f}, {meta['omega_range'][1]:.4f}], backend={meta['backend']}")

    band_lo, band_hi, _, _ = _band_edges(args, exact_plot, op)

    if args.debug_rect:
        exact_w = get_exact_eigenvalues(H)
        plot_debug_rectangle(band_lo, band_hi, exact_w, args.rect_degs, args.rect_deg_max,
                             args.rect_degs, args.rect_plot, args.save)
        return

    if args.converge_spectrum:
        _run_converge_spectrum(H, V0, exact_plot, args, op, meta, args.save)
        return

    if args.solve_band >= 0:
        bi = int(args.solve_band) % len(band_lo)
        f_lo, f_hi = float(band_lo[bi]), float(band_hi[bi])
        result = solve_band(H, V0, f_lo, f_hi, args.coarse_iters, args.solve_iters, args.square_filter)
        w_in, r_in = result["w_in"], result["r_in"]
        print(f"Band {bi} range [{f_lo:.6f},{f_hi:.6f}]")
        for wi, ri in zip(w_in, r_in):
            val = op.scaled_to_omega(wi) if op is not None else wi
            print(f"  ritz {val:+.6f}  resid {ri:.3e}")
        if op is None and args.size <= 200:
            exact_w = get_exact_eigenvalues(H)
            w_exact = exact_w[(exact_w >= f_lo) & (exact_w <= f_hi)]
            print("Exact in band:", " ".join(f"{x:+.6f}" for x in w_exact))
        return

    if args.spectral_filter:
        _run_spectral_filter(H, V0, exact_plot, args, op, meta, args.save)
        return

    print("No mode selected. Examples:")
    print("  python spectral_demos.py --benchmark adamantane --spectral_filter --save out.png")
    print("  python spectral_demos.py --benchmark nc_C_R5 --converge_spectrum --save out.png")
    print("  python spectral_demos.py --benchmark_suite nc_C_R4,nc_C_R5,nc_C_R6 --run_all --save_dir /tmp/bench")


if __name__ == "__main__":
    main()
