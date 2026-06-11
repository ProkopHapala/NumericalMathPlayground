import numpy as np
import argparse

# Import core solver functions
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
    power_iteration_filter,
)

# Import plotting utilities
from spectral_plotting import (
    save_or_show,
    plot_debug_rectangle,
    plot_convergence_3panel,
    plot_spectral_evolution,
)


# =====================================================================
# FRONTEND / MAIN EXECUTION
# =====================================================================

'''
# Example CLI commands:

1) Debug: show Chebyshev rectangle convergence for each band

python chebyshev_prefiltered_jacobi_ocl.py --debug_rect --size 60 --nbands 6 --coarse_iters 40 --save /tmp/rect.png

2) Converge spectrum with Ritz + gradual pruning, save 3-panel plot

python chebyshev_prefiltered_jacobi_ocl.py --converge_spectrum --size 60 --nvec 20 --nbands 6 --coarse_iters 40 --conv_iters 8 \
     --square_filter --filter_reps 1 
     --prune_mode gradual --prune_tol 1e-3 --gradual_factor 3.0 \
     --method ritz --save /tmp/conv.png

3) Same but with Lanczos instead of Ritz

python chebyshev_prefiltered_jacobi_ocl.py --converge_spectrum \
    --size 60 --nvec 20 --nbands 6 --coarse_iters 40 --conv_iters 8 \
    --method lanczos --save /tmp/conv_lanczos.png

4) Solve single band (no convergence plot, just eigenpairs)

python chebyshev_prefiltered_jacobi_ocl.py --solve_band 2 --size 60 \
    --nvec 10 --coarse_iters 40 --solve_iters 4 --square_filter

5) No pruning baseline (all vectors kept)

python chebyshev_prefiltered_jacobi_ocl.py --converge_spectrum \
    --size 60 --nvec 20 --nbands 6 --coarse_iters 40 --conv_iters 8 \
    --prune_mode none --method ritz --save /tmp/no_prune.png

6) Aggressive residual pruning (drop all above tol after it>=2)

python chebyshev_prefiltered_jacobi_ocl.py --converge_spectrum \
    --size 60 --nvec 20 --nbands 6 --coarse_iters 40 --conv_iters 8 \
    --prune_mode residual --prune_tol 1e-3 --method ritz --save /tmp/residual_prune.png

7) Hybrid pruning (cluster + gradual)

python chebyshev_prefiltered_jacobi_ocl.py --converge_spectrum \
    --size 60 --nvec 20 --nbands 6 --coarse_iters 40 --conv_iters 8 \
    --prune_mode hybrid --cluster_tol 0.85 --prune_tol 1e-3 --method ritz

8) KPM Chebyshev spectral filter (Jackson damping, spectrum convergence)

python chebyshev_prefiltered_jacobi_ocl.py --spectral_filter --size 60 --nvec 8 --save /tmp/spectral_cheb.png

9) KPM Chebyshev without Jackson damping (observe Gibbs ripples)

python chebyshev_prefiltered_jacobi_ocl.py --spectral_filter --spectral_no_jackson 1 --size 60 --nvec 8 --save /tmp/spectral_gibbs.png

10) Power iteration spectral filter

python chebyshev_prefiltered_jacobi_ocl.py --spectral_filter --spectral_method power --size 60 --nvec 8 --save /tmp/spectral_power.png
'''

def main():
    parser = argparse.ArgumentParser(description="Coarse-Fine Spectral Solver")
    parser.add_argument('--size',         type=int,   default=20,  help='Matrix size N')
    parser.add_argument('--nvec',         type=int,   default=8,   help='Random probe vectors')
    parser.add_argument('--coarse_iters', type=int,   default=32,  help='Chebyshev degree for band-pass filter')
    parser.add_argument('--fine_iters',   type=int,   default=30,  help='Jacobi iterations inside each band')
    parser.add_argument('--alpha',        type=float, default=0.3, help='Jacobi step size')
    parser.add_argument('--beta',         type=float, default=0.0, help='Heavy-ball momentum')
    parser.add_argument('--nbands',       type=int,   default=5,   help='Number of coarse frequency bands')
    parser.add_argument('--nfine',        type=int,   default=200, help='Fine frequency points per band')
    parser.add_argument('--n_quad',       type=int,   default=16,  help='Quadrature pts for band-pass integration')
    parser.add_argument('--save',         type=str,   default=None)
    parser.add_argument('--log',          action='store_true', default=True, help='Log scale on amplitude plot (default on)')
    parser.add_argument('--no_log',       action='store_true', help='Disable log scale')
    parser.add_argument('--debug_filter', action='store_true', help='Debug: plot band-pass filter shape for each band at multiple Chebyshev degrees')
    parser.add_argument('--debug_rect',   action='store_true',  help='Debug: plot scalar Chebyshev band-pass polynomial p(x) converging to a rectangle')
    parser.add_argument('--rect_deg_max', type=int, default=128)
    parser.add_argument('--rect_degs',    type=str, default='4,8,16,32,64,128')
    parser.add_argument('--rect_plot',    type=str, default='square', choices=['signed', 'abs', 'square'],  help='How to visualize the band-pass polynomial: signed p(x), |p(x)|, or p(x)^2 (nonnegative)')
    parser.add_argument('--solve_band',   type=int, default=-1, help='If >=0, run Chebyshev band filtering + Rayleigh-Ritz to extract eigenpairs in that band index')
    parser.add_argument('--solve_iters',  type=int, default=2,  help='Number of (filter->QR) refinement rounds for solve_band')
    parser.add_argument('--square_filter', action='store_true', help='Use p(H)^2 (apply band-pass filter twice). Makes filter nonnegative and sharper.')
    parser.add_argument('--filter_reps',  type=int, default=1, help='How many times to apply the Chebyshev filter per subspace iteration (default 1). Each application costs ~coarse_iters SpMVs. Set to 0 to skip filtering after the first step.')
    parser.add_argument('--converge_spectrum', action='store_true', help='Compute approximate eigenvalues by band-pass filter + repeated Rayleigh-Ritz, and plot convergence vs exact spectrum')
    parser.add_argument('--conv_iters',   type=int, default=6, help='Number of filter->QR->Ritz refinement iterations per band')
    parser.add_argument('--res_tol',      type=float, default=1e-6, help='Residual tolerance for reporting converged Ritz values')
    parser.add_argument('--method',       type=str, default='ritz', choices=['ritz','lanczos'], help='Subspace method after band-pass filtering: ritz = subspace iteration, lanczos = Lanczos on filtered basis')
    parser.add_argument('--prune_mode',   type=str, default='none', choices=['none','residual','cluster','gradual','hybrid'], help='Pruning strategy for redundant trial vectors in converge_spectrum mode')
    parser.add_argument('--prune_tol',    type=float, default=-1.0, help='Residual tolerance for pruning (used by residual, gradual, hybrid modes). If >0 and prune_mode==none, auto-switches to residual mode.')
    parser.add_argument('--cluster_tol',  type=float, default=-1.0, help='Dot-product threshold for cluster mode (e.g. 0.85). If >0 and prune_mode==none, auto-switches to cluster mode.')
    parser.add_argument('--cluster_eig_tol', type=float, default=0.05, help='Max eigenvalue separation for cluster pruning (default 0.05)')
    parser.add_argument('--gradual_factor',  type=float, default=3.0, help='Outlier factor for gradual pruning: prune worst only if r_worst > factor * median_r (default 3.0)')
    parser.add_argument('--rank_est',        action='store_true', help='Print cheap rank estimate (subspace dimension) per band after first filter')
    parser.add_argument('--spectral_filter', action='store_true', help='KPM spectral filter: evaluate Chebyshev point filter at many frequencies to show spectrum convergence')
    parser.add_argument('--spectral_method', choices=['chebyshev', 'power'], default='chebyshev', help='Filter method: chebyshev (KPM with Jackson) or power (shifted power iteration)')
    parser.add_argument('--spectral_iters',        type=str, default='4,8,16,32,64,128', help='Iteration counts to plot (comma-separated)')
    parser.add_argument('--spectral_nfreq',        type=int, default=1000, help='Number of frequency points for spectral filter')
    parser.add_argument('--spectral_normalize',    type=int, default=1, help='Normalize each iteration line to max=1')
    parser.add_argument('--spectral_no_normalize', type=int, default=0, help='Disable normalization')
    parser.add_argument('--spectral_jackson',      type=int, default=1, help='Use Jackson damping (chebyshev method only)')
    parser.add_argument('--spectral_no_jackson',   type=int, default=0, help='Disable Jackson damping to observe Gibbs ripples')

    args = parser.parse_args()

    # Backward-compat: old --prune_tol or --cluster_tol auto-set prune_mode
    if args.prune_mode == 'none':
        if args.prune_tol > 0 and args.cluster_tol > 0:
            args.prune_mode = 'hybrid'
        elif args.cluster_tol > 0:
            args.prune_mode = 'cluster'
        elif args.prune_tol > 0:
            args.prune_mode = 'residual'

    H          = generate_test_matrix(args.size)
    V0         = np.random.randn(args.size, args.nvec)
    exact_eigs = get_exact_eigenvalues(H)

    print(f"\n--- Coarse-Fine Spectral Solver  (N={args.size}, nvec={args.nvec}, prune_mode={args.prune_mode}) ---")

    # Band edges covering the spectrum
    edges     = np.linspace(-0.95, 0.95, args.nbands + 1)
    band_lo   = edges[:-1]
    band_hi   = edges[1:]
    band_cent = 0.5 * (band_lo + band_hi)

    if args.debug_rect:
        plot_debug_rectangle(band_lo, band_hi, exact_eigs, args.rect_degs, args.rect_deg_max,  args.rect_degs, args.rect_plot, args.save)
        return

    if args.converge_spectrum:
        # Run the core band-by-band spectral solver
        result = solve_spectrum(
            H, band_lo, band_hi, args.nvec, args.coarse_iters, args.conv_iters,
            args.filter_reps, args.square_filter, args.method, args.prune_mode,
            args.prune_tol, args.cluster_tol, args.cluster_eig_tol, args.gradual_factor,
            rank_est=args.rank_est, res_tol=args.res_tol
        )

        exact_w = result['exact_w']
        band_raw_w = result['band_raw_w']
        band_raw_r = result['band_raw_r']
        band_keep = result['band_keep']
        final_traj = result['final_traj']
        total_spmv = result['total_spmv']
        spmv_per_band = result['spmv_per_band']

        # ---- SpMV summary ----
        print(f'\n=== SpMV cost summary ===')
        spmv_per_filter_app = args.coarse_iters * (2 if args.square_filter else 1)
        for bi in range(args.nbands):
            print(f'  band={bi}  SpMVs={spmv_per_band.get(bi, 0):,}')
        print(f'  TOTAL  SpMVs={total_spmv:,}')
        print(f'  (degree={args.coarse_iters}, square={args.square_filter}, '
              f'filter_reps={args.filter_reps}, conv_iters={args.conv_iters}, '
              f'method={args.method})')
        print(f'  Each filter application costs {spmv_per_filter_app} SpMVs per vector')
        # Dense comparison
        N = args.size
        n_bands = args.nbands
        dense_flops = int(4.0/3.0 * N**3)
        nnz_est = max(N, 5 * N)
        our_flops_sparse = total_spmv * nnz_est
        our_flops_dense  = total_spmv * N * N
        print(f'\n--- Comparison to dense linear algebra (N={N}) ---')
        print(f'  Full diagonalization (dense): ~{dense_flops:,} flops')
        print(f'  Our method if H is dense:     ~{our_flops_dense:,} flops')
        print(f'  Our method if H is sparse (nnz~{nnz_est}): ~{our_flops_sparse:,} flops')
        if our_flops_dense > 0:
            ratio = our_flops_dense / max(1, dense_flops)
            print(f'  Ratio (our_dense / full_diag): {ratio:.1f}x')
            print(f'  NOTE: for N={N} dense, direct solve is far cheaper.')
            print(f'        Our method is designed for large sparse N where full diag is impossible.')

        # Print last-iter converged values
        print(f'\nConverged (last-iter, residual < {args.res_tol})')
        for bi in range(args.nbands):
            for tid, pts in final_traj[bi].items():
                if len(pts) == 0: continue
                last = pts[-1]
                if last[3] and last[2] < args.res_tol:
                    print(f"  band={bi}  ritz={last[1]:+.10f}  resid={last[2]:.3e}")

        # Call plotting function from spectral_plotting
        plot_convergence_3panel(band_lo, band_hi, exact_w, band_raw_w, band_raw_r, band_keep, final_traj, args.coarse_iters, args.square_filter, args.filter_reps, args.conv_iters, args.method, args.prune_mode, args.nvec, args.save)
        return

    if args.solve_band >= 0:
        bi = int(args.solve_band) % args.nbands
        f_lo, f_hi = float(band_lo[bi]), float(band_hi[bi])
        result = solve_band(H, V0, f_lo, f_hi, args.coarse_iters,
                            args.solve_iters, args.square_filter)
        w_in = result['w_in']
        r_in = result['r_in']
        solve_spmv = result['solve_spmv']
        k = result['k']
        print(f'Band {bi} range [{f_lo:.6f},{f_hi:.6f}]')
        for wi, ri in zip(w_in, r_in):
            print(f'  ritz_eval {wi:+.10f}  resid {ri:.3e}')
        print(f'  SpMVs={solve_spmv:,}  (degree={args.coarse_iters}, square={args.square_filter}, solve_iters={args.solve_iters}, k={k})')
        if args.size <= 200:
            w_exact = exact_eigs[(exact_eigs >= f_lo) & (exact_eigs <= f_hi)]
            print('Exact eigs in band:', ' '.join([f'{x:+.10f}' for x in w_exact]))
        return

    if args.spectral_filter:
        freqs = np.linspace(-0.99, 0.99, args.spectral_nfreq)
        iters_to_plot = [int(s) for s in args.spectral_iters.split(',') if s.strip()]
        use_jackson = args.spectral_jackson and not args.spectral_no_jackson
        normalize = args.spectral_normalize and not args.spectral_no_normalize

        print(f"\n--- Spectral Filter: {args.spectral_method.upper()} ---")
        if args.spectral_method == 'chebyshev':
            total_amps, vec_amps = chebyshev_filter(H, V0, freqs, iters_to_plot, use_jackson=use_jackson)
        else:
            total_amps, vec_amps = power_iteration_filter(H, V0, freqs, iters_to_plot)

        plot_spectral_evolution(freqs, total_amps, vec_amps, iters_to_plot,  exact_eigs=exact_eigs, method=args.spectral_method, normalize=normalize, save_path=args.save)
        return

    # Note: debug_filter and main mode removed - they used deprecated KPM-based functions
    # Use --debug_rect, --converge_spectrum, --solve_band, or --spectral_filter instead
    print("No mode selected. Use one of: --debug_rect, --converge_spectrum, --solve_band, --spectral_filter")

if __name__ == "__main__":
    main()