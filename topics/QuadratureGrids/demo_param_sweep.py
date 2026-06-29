"""
demo_param_sweep.py — Parameter sweep for CubeEmbededAtoms weight optimization.

Sweeps over 4 grid parameters, optimizes weights using atomic orbital trial
functions, and plots error trends + pathological functions.

Usage:
    python demo_param_sweep.py
    python demo_param_sweep.py --n-combos 100 --n-angular-dirs 4
    python demo_param_sweep.py --jobs 4
"""
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from BasisFunctions import build_atomic_trial_set
from WeightOptimizer import (
    optimize_single_config, precompute_outer_cache, compute_R_cut,
    baseline_cartesian,
)
from GridPlotting import (
    plot_param_sweep, plot_pathological_functions, plot_grid_overview,
    plot_dscan_grids_weights,
)

IMG_DIR = "Quadrature_Images"

# Default grid parameters
DEFAULTS = dict(d=0.5, n=10, alpha=1.8, n_blend=4)

# Fixed cutout half-size for d sweep (so cutout area doesn't change with d)
H_FIXED = DEFAULTS['d'] * DEFAULTS['n'] / 2  # = 2.5

# Sweep ranges
SWEEPS = dict(
    d       = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
    n       = [4, 6, 8, 10, 12, 16, 20],
    alpha   = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
    n_blend = [1, 2, 4, 6, 8, 10],
)

LAMBDAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]


def _worker(args):
    """Worker function for multiprocessing."""
    params, trial_set, lambdas, margin, outer_cache, fixed_lambda, non_negative = args
    return optimize_single_config(
        d=params['d'], n=params['n'], alpha=params['alpha'],
        n_blend=params['n_blend'], trial_set=trial_set,
        lambdas=lambdas, margin=margin, outer_cache=outer_cache,
        fixed_lambda=fixed_lambda, non_negative=non_negative,
    )


def sweep_parameter(param_name, values, trial_set, lambdas, defaults, R_cut,
                    margin=5, jobs=None, h_fixed=None, fixed_lambda=None,
                    non_negative=False):
    """Sweep a single parameter, keeping others at default.

    Precomputes outer grid caches per unique (d, h) pair for efficiency.
    Uses fixed R_cut for the embedding grid extent.

    If param_name='d' and h_fixed is given, keeps cutout size h=h_fixed
    and adjusts n = round(2*h/d) so the cutout area stays constant.
    """
    configs = []
    for val in values:
        params = dict(defaults)
        params[param_name] = val
        if param_name == 'd' and h_fixed is not None:
            params['n'] = max(4, int(round(2 * h_fixed / val)))
        configs.append(params)

    # Precompute outer caches per unique (d, h) pair
    # h = n * d / 2
    outer_caches = {}
    for c in configs:
        h = c['n'] * c['d'] / 2
        key = (c['d'], round(h, 10))
        if key not in outer_caches:
            outer_caches[key] = precompute_outer_cache(c['d'], h, trial_set, R_cut=R_cut)

    worker_args = []
    for c in configs:
        h = c['n'] * c['d'] / 2
        key = (c['d'], round(h, 10))
        worker_args.append((c, trial_set, lambdas, margin, outer_caches[key], fixed_lambda, non_negative))

    if jobs is None:
        jobs = min(cpu_count(), len(configs))

    print(f"\n── Sweeping {param_name}: {values} ({len(configs)} configs, {jobs} jobs) ──")
    t0 = time.time()

    if jobs <= 1:
        results = [_worker(a) for a in worker_args]
    else:
        with Pool(jobs) as pool:
            results = pool.map(_worker, worker_args)

    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s")

    for i, (val, r) in enumerate(zip(values, results)):
        n_outer = len(r['outer_xy'])
        overdetermined = "✓" if r['N_eq'] > r['N_orbits'] else "✗ UNDERDET"
        print(f"  {param_name}={val:8.3f}  n={r['n']:3d}  Npts={r['Npts']:5d}  "
              f"eq/orbits={r['N_eq']:4d}/{r['N_orbits']:3d} {overdetermined}  "
              f"λ={r['best_lam']:10.6f}  "
              f"train={r['mean_err']:.4f}%  val={r['val_mean_err']:.4f}%  "
              f"max_val={r['val_max_err']:.4f}%")

    return results


def sweep_baseline(d_values, trial_set, R_cut):
    """Compute pure Cartesian baseline for each d value."""
    print(f"\n── Baseline (pure Cartesian, no inner insert) ──")
    print(f"  R_cut = {R_cut:.2f}")
    results = []
    for d in d_values:
        r = baseline_cartesian(d, trial_set, R_cut=R_cut)
        results.append(r)
        print(f"  d={d:8.3f}  Npts={r['Npts']:5d}  "
              f"mean_err={r['mean_err']:.4f}%  max_err={r['max_err']:.4f}%")
    return results


def plot_baseline_comparison(opt_results, baseline_results, d_values):
    """Compare optimized inner+outer grid vs pure Cartesian baseline.

    Shows how error decreases with finer d for both approaches.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Mean error
    opt_mean = [r['mean_err'] for r in opt_results]
    base_mean = [r['mean_err'] for r in baseline_results]
    opt_npts = [r['Npts'] + len(r['outer_xy']) for r in opt_results]
    base_npts = [r['Npts'] for r in baseline_results]

    ax1.loglog(d_values, base_mean, 's--', label='Cartesian baseline', color='gray', linewidth=2)
    ax1.loglog(d_values, opt_mean, 'o-', label='Optimized (inner+outer)', color='steelblue', linewidth=2)
    ax1.set_xlabel('Grid step d (Bohr)', fontsize=12)
    ax1.set_ylabel('Mean integration error (%)', fontsize=12)
    ax1.set_title('Mean error vs grid step', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # finer d on the right

    # Max error
    opt_max = [r['max_err'] for r in opt_results]
    base_max = [r['max_err'] for r in baseline_results]

    ax2.loglog(d_values, base_max, 's--', label='Cartesian baseline', color='gray', linewidth=2)
    ax2.loglog(d_values, opt_max, 'o-', label='Optimized (inner+outer)', color='crimson', linewidth=2)
    ax2.set_xlabel('Grid step d (Bohr)', fontsize=12)
    ax2.set_ylabel('Max integration error (%)', fontsize=12)
    ax2.set_title('Max error vs grid step', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    # Add Npts annotations
    for ax, npts_list in [(ax1, base_npts), (ax2, opt_npts)]:
        for i, (d, n) in enumerate(zip(d_values, npts_list)):
            if i % 2 == 0:
                ax.annotate(f'N={n}', (d, ax.get_ylim()[1] * 0.5),
                           fontsize=7, ha='center', alpha=0.5)

    fig.suptitle('Baseline (pure Cartesian) vs Optimized grid — '
                 'embedding covers fixed [-R_cut, R_cut]²', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Parameter sweep for weight optimization')
    parser.add_argument('--basis', default='cc-pVDZ', help='Basis set name')
    parser.add_argument('--elements', default='H,C,N,O', help='Comma-separated elements')
    parser.add_argument('--n-angular-dirs', type=int, default=8, help='Angular directions per p/d shell')
    parser.add_argument('--n-combos', type=int, default=1000, help='Number of random trial combinations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--jobs', type=int, default=None, help='Number of parallel jobs')
    parser.add_argument('--margin', type=int, default=5, help='Outer grid margin in grid steps')
    parser.add_argument('--fixed-lambda', type=float, default=None,
                        help='Use fixed lambda for all configs (skip train/val selection). E.g. 0.01')
    parser.add_argument('--non-negative', action='store_true',
                        help='Enforce non-negative weights (w >= 0) using bounded least-squares')
    args = parser.parse_args()

    os.makedirs(IMG_DIR, exist_ok=True)

    elements = args.elements.split(',')

    # Build trial function set (done once, shared across all configs)
    trial_set = build_atomic_trial_set(
        elements=elements, basis_name=args.basis,
        n_angular_dirs=args.n_angular_dirs,
        n_combos=args.n_combos, seed=args.seed,
    )

    # Compute R_cut — fixed extent of embedding grid (covers full trial function support)
    R_cut = compute_R_cut(trial_set)
    print(f"\n  R_cut (embedding grid extent) = {R_cut:.2f} Bohr")

    # Run baseline: pure Cartesian grid, no inner insert
    baseline_results = sweep_baseline(SWEEPS['d'], trial_set, R_cut)

    # Run sweeps
    sweep_results = {}
    param_values = {}

    for pname in SWEEPS:
        h_fix = H_FIXED if pname == 'd' else None
        sweep_results[pname] = sweep_parameter(
            pname, SWEEPS[pname], trial_set, LAMBDAS, DEFAULTS, R_cut,
            margin=args.margin, jobs=args.jobs, h_fixed=h_fix,
            fixed_lambda=args.fixed_lambda, non_negative=args.non_negative,
        )
        param_values[pname] = SWEEPS[pname]

    # Plot 1D trends
    fig1 = plot_param_sweep(sweep_results, list(SWEEPS.keys()), param_values, DEFAULTS)
    fig1.savefig(f"{IMG_DIR}/param_sweep_trends.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {IMG_DIR}/param_sweep_trends.png")
    plt.close(fig1)

    # Plot baseline vs optimized comparison (d sweep)
    fig_cmp = plot_baseline_comparison(sweep_results['d'], baseline_results, SWEEPS['d'])
    fig_cmp.savefig(f"{IMG_DIR}/param_sweep_baseline.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {IMG_DIR}/param_sweep_baseline.png")
    plt.close(fig_cmp)

    # Plot d-scan grid geometries + weight ratios
    fig_dscan = plot_dscan_grids_weights(sweep_results['d'], SWEEPS['d'])
    fig_dscan.savefig(f"{IMG_DIR}/param_sweep_dscan_weights.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {IMG_DIR}/param_sweep_dscan_weights.png")
    plt.close(fig_dscan)

    # Plot grid geometry overview (all tried configs with embedding)
    fig_grid = plot_grid_overview(sweep_results, list(SWEEPS.keys()), param_values, DEFAULTS)
    fig_grid.savefig(f"{IMG_DIR}/param_sweep_grids.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {IMG_DIR}/param_sweep_grids.png")
    plt.close(fig_grid)

    # Plot pathological functions for the default config
    default_result = None
    for pname in SWEEPS:
        for i, val in enumerate(SWEEPS[pname]):
            if val == DEFAULTS[pname]:
                r = sweep_results[pname][i]
                if default_result is None:
                    default_result = r
                break

    if default_result is not None:
        fig2 = plot_pathological_functions(
            default_result, default_result['grid_pts'], trial_set, top_n=6)
        fig2.savefig(f"{IMG_DIR}/param_sweep_pathological.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {IMG_DIR}/param_sweep_pathological.png")
        plt.close(fig2)

    # Also plot pathological functions for the worst config in each sweep
    for pname in SWEEPS:
        worst = max(sweep_results[pname], key=lambda r: r['max_err'])
        fig = plot_pathological_functions(
            worst, worst['grid_pts'], trial_set, top_n=6)
        fname = f"{IMG_DIR}/param_sweep_pathological_{pname}.png"
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
        plt.close(fig)

    print("\nDone.")


if __name__ == '__main__':
    main()
