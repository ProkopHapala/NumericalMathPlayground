"""
demo_cube_embeded.py — Demo for CubeEmbededAtoms integration grid.

Usage:
    python demo_cube_embeded.py --mode grid
    python demo_cube_embeded.py --mode integration
    python demo_cube_embeded.py --mode weights
    python demo_cube_embeded.py --mode convergence
    python demo_cube_embeded.py --mode all
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from CubeEmbededGrid import build_grid, build_combined_grid, build_cartesian_outer
from BasisFunctions import (
    gaussian_1s, gaussian_2p, gaussian_2py,
    analytic_1s_norm_sq, analytic_2p_norm_sq,
    STO3G_1S, STO6G_1S, STO3G_2P_C,
    build_test_function_set, eval_test_func,
)
from Symmetry import group_orbits
from WeightOptimizer import (
    geometric_weights, build_integration_matrix, compute_targets,
    tikhonov_optimize, expand_orbit_weights, validate_sto3g, select_best_lambda,
)
from GridPlotting import (
    plot_cube_full_view, plot_cube_cell_areas, plot_cube_radial_profile,
    plot_cube_three_grids, plot_blend_weight,
    plot_orbitals_on_grid, plot_density_overlay, plot_convergence,
    plot_weight_comparison, plot_weight_errors, plot_lambda_sweep,
)

IMG_DIR = "Quadrature_Images"


def mode_grid(d=0.5, n=10, alpha=1.8, n_blend=4):
    grid_dict = build_grid(d=d, n=n, alpha=alpha, n_blend=n_blend)
    p = grid_dict['params']

    print(f"Angular spacing at boundary: {p['angular_spacing']:.3f} Bohr")
    print(f"Derived Nv for corner isotropy: {p['Nv']} (α={alpha})")
    print(f"Boundary snap error: {np.max(np.abs(grid_dict['points'][:, -1, :] - grid_dict['edge_pts'])):.2e}")
    print(f"Cell areas: min={grid_dict['cell_areas'].min():.4f}, max={grid_dict['cell_areas'].max():.4f}")
    print(f"Cell aspect ratios: mean={grid_dict['cell_aspect'].mean():.3f} (1.0=isotropic)")
    print(f"Total grid points: {p['Nu'] * p['Nv']}")

    fig1 = plot_cube_full_view(grid_dict)
    fig1.savefig(f"{IMG_DIR}/cube_embeded_atoms_2d.png", dpi=150, bbox_inches='tight')
    plt.show()

    fig2 = plot_cube_cell_areas(grid_dict)
    fig2.savefig(f"{IMG_DIR}/cube_embeded_atoms_cells.png", dpi=150, bbox_inches='tight')
    plt.show()

    fig3 = plot_cube_radial_profile(grid_dict)
    fig3.savefig(f"{IMG_DIR}/cube_embeded_atoms_radial.png", dpi=150, bbox_inches='tight')
    plt.show()

    fig4 = plot_cube_three_grids(grid_dict)
    fig4.savefig(f"{IMG_DIR}/cube_embeded_atoms_compare.png", dpi=150, bbox_inches='tight')
    plt.show()

    fig5 = plot_blend_weight(grid_dict)
    fig5.savefig(f"{IMG_DIR}/cube_embeded_atoms_blend.png", dpi=150, bbox_inches='tight')
    plt.show()


def mode_integration(d=0.5, n=10, alpha=1.8, n_blend=4, margin=5):
    grid_dict = build_grid(d=d, n=n, alpha=alpha, n_blend=n_blend)
    combined = build_combined_grid(grid_dict, margin=margin)
    p = grid_dict['params']
    h = p['h']

    print(f"Inner grid points: {len(combined['inner_xy'])}")
    print(f"Outer Cartesian points: {len(combined['outer_xy'])}")
    print(f"Combined grid points: {len(combined['combined_xy'])}")
    print(f"Total weight: {combined['combined_w'].sum():.4f}")

    zeta_1s = 1.24
    zeta_2p = 1.72
    xy = combined['combined_xy']
    x, y = xy[:, 0], xy[:, 1]
    r2 = x**2 + y**2

    phi_1s = gaussian_1s(r2, STO3G_1S, zeta_1s)
    rho_1s = phi_1s**2
    phi_2px = gaussian_2p(x, y, STO3G_2P_C, zeta_2p)
    rho_2px = phi_2px**2
    rho_sp = phi_1s**2 + 0.25 * phi_2px**2

    phi_1s_6g = gaussian_1s(r2, STO6G_1S, zeta_1s)
    rho_1s_6g = phi_1s_6g**2

    w = combined['combined_w']
    num_1s = np.sum(rho_1s * w)
    num_2p = np.sum(rho_2px * w)
    num_1s_6g = np.sum(rho_1s_6g * w)
    num_rho_sp = np.sum(rho_sp * w)

    exact_1s = analytic_1s_norm_sq(STO3G_1S, zeta_1s)
    exact_2p = analytic_2p_norm_sq(STO3G_2P_C, zeta_2p)
    exact_1s_6g = analytic_1s_norm_sq(STO6G_1S, zeta_1s)
    exact_rho_sp = exact_1s + 0.25 * exact_2p

    print(f"\n── Integration test ──")
    print(f"1s (STO-3G):  exact={exact_1s:.8f},  numerical={num_1s:.8f},  error={abs(num_1s-exact_1s)/exact_1s*100:.4f}%")
    print(f"2p (STO-3G):  exact={exact_2p:.8f},  numerical={num_2p:.8f},  error={abs(num_2p-exact_2p)/exact_2p*100:.4f}%")
    print(f"1s (STO-6G):  exact={exact_1s_6g:.8f},  numerical={num_1s_6g:.8f},  error={abs(num_1s_6g-exact_1s_6g)/exact_1s_6g*100:.4f}%")
    print(f"ρ=1s²+0.25·2p²:  exact={exact_rho_sp:.8f},  numerical={num_rho_sp:.8f},  error={abs(num_rho_sp-exact_rho_sp)/exact_rho_sp*100:.4f}%")

    fig1 = plot_orbitals_on_grid(xy, phi_1s, rho_1s, phi_2px, rho_2px, rho_sp,
                                 num_1s, exact_1s, num_2p, exact_2p, num_rho_sp, exact_rho_sp,
                                 zeta_1s, zeta_2p, h)
    fig1.savefig(f"{IMG_DIR}/cube_embeded_atoms_orbitals.png", dpi=150, bbox_inches='tight')
    plt.show()

    fig2 = plot_density_overlay(grid_dict, combined['inner_xy'],
                                STO3G_1S, zeta_1s, STO3G_2P_C, zeta_2p)
    fig2.savefig(f"{IMG_DIR}/cube_embeded_atoms_density.png", dpi=150, bbox_inches='tight')
    plt.show()


def mode_weights(d=0.5, n=10, alpha=1.8, n_blend=4, margin=5):
    grid_dict = build_grid(d=d, n=n, alpha=alpha, n_blend=n_blend)
    p = grid_dict['params']
    h = p['h']

    w0, grid_pts, grid_idx = geometric_weights(grid_dict)
    Npts = len(grid_pts)
    print(f"Grid points: {Npts}")
    print(f"Geometric weights w0: min={w0.min():.6f}, max={w0.max():.6f}, sum={w0.sum():.4f}")

    orbit_id, orbit_members, orbit_rep = group_orbits(grid_pts)
    N_orbits = len(orbit_rep)
    print(f"Symmetry orbits: {N_orbits} (from {Npts} points, compression {Npts/N_orbits:.1f}×)")

    w0_orbit = np.array([np.mean(w0[members]) for members in orbit_members])

    outer_xy, outer_w = build_cartesian_outer(h, d, margin)
    test_funcs, integrals_total = build_test_function_set()
    Ntest = len(test_funcs)
    print(f"Test functions: {Ntest}")

    b_inner, I_outer = compute_targets(test_funcs, integrals_total, outer_xy, outer_w, h)

    A = build_integration_matrix(grid_pts, orbit_members, test_funcs)
    U, svals, Vt = np.linalg.svd(A, full_matrices=False)
    rank = np.sum(svals > 1e-10 * svals[0])
    print(f"A shape: {A.shape}, rank={rank}, condition={svals[0]/svals[-1]:.1e}")

    # Equation weights q
    q = np.ones(Ntest)
    q[0] = 100.0
    for k, (label, params) in enumerate(test_funcs):
        if params is not None:
            kind, zeta = params[0], params[1]
            if kind == 'gauss':
                q[k] = 1.0 / zeta
            elif kind == 'px':
                q[k] = 0.1
            elif kind == 'x2g':
                q[k] = 1.0 / zeta
            elif kind == 'xyg':
                q[k] = 0.1

    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    results = tikhonov_optimize(A, b_inner, w0_orbit, lambdas, q=q)

    print("\n── Tikhonov optimization ──")
    print(f"{'λ':>10s}  {'max err%':>10s}  {'mean err%':>10s}  {'w min':>10s}  {'w max':>10s}")
    for lam in lambdas:
        r = results[lam]
        w_o = r['w_orbit']
        print(f"{lam:10.6f}  {r['max_err']:10.4f}  {r['mean_err']:10.4f}  "
              f"{w_o.min():10.6f}  {w_o.max():10.6f}")

    best_lam, best_err = select_best_lambda(results, w0_orbit)
    print(f"\nBest λ = {best_lam} (mean err = {best_err:.4f}%)")

    # STO-3G validation for all λ
    print("\n── STO-3G integration for all λ ──")
    print(f"{'λ':>10s}  {'1s err%':>10s}  {'2p err%':>10s}  {'combined':>10s}")
    best_lam_sto3g = None
    best_combined = float('inf')
    for lam in sorted(results.keys()):
        w_full = expand_orbit_weights(results[lam]['w_orbit'], orbit_members, Npts)
        val = validate_sto3g(grid_pts, w_full, outer_xy, outer_w)
        combined_err = val['err_1s'] + val['err_2p']
        print(f"{lam:10.6f}  {val['err_1s']:10.4f}  {val['err_2p']:10.4f}  {combined_err:10.4f}")
        if combined_err < best_combined:
            best_combined = combined_err
            best_lam_sto3g = lam

    print(f"\nBest λ for STO-3G: {best_lam_sto3g} (combined err = {best_combined:.4f}%)")
    lam_opt = best_lam_sto3g
    w_opt = expand_orbit_weights(results[lam_opt]['w_orbit'], orbit_members, Npts)
    w_geom = expand_orbit_weights(w0_orbit, orbit_members, Npts)

    # Per-function error breakdown
    print("\n── Per-function errors (best λ) ──")
    pred_best = results[lam_opt]['predicted']
    nonzero = np.abs(b_inner) > 1e-10
    test_labels = [label for label, _ in test_funcs]
    for k in range(Ntest):
        if nonzero[k]:
            err = abs(pred_best[k] - b_inner[k]) / abs(b_inner[k]) * 100
        else:
            err = abs(pred_best[k] - b_inner[k])
        flag = '  <<<' if err > 100 else ''
        print(f"{k:3d}  {test_labels[k]:>20s}  {b_inner[k]:12.6f}  {pred_best[k]:12.6f}  {err:12.4f}{flag}")

    # Plots
    fig1 = plot_weight_comparison(grid_pts, w_geom, w_opt, h, lam_opt)
    fig1.savefig(f"{IMG_DIR}/cube_embeded_atoms_weights.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Error bars
    pred_geom = A @ w0_orbit
    errs_geom = np.zeros(Ntest)
    errs_opt = np.zeros(Ntest)
    for k in range(Ntest):
        if nonzero[k]:
            errs_geom[k] = abs(pred_geom[k] - b_inner[k]) / abs(b_inner[k]) * 100
            errs_opt[k] = abs(pred_best[k] - b_inner[k]) / abs(b_inner[k]) * 100
        else:
            errs_geom[k] = abs(pred_geom[k] - b_inner[k])
            errs_opt[k] = abs(pred_best[k] - b_inner[k])
    fig2 = plot_weight_errors(test_labels, errs_geom, errs_opt, lam_opt)
    fig2.savefig(f"{IMG_DIR}/cube_embeded_atoms_weight_errors.png", dpi=150, bbox_inches='tight')
    plt.show()

    fig3 = plot_lambda_sweep(results, lambdas, w0_orbit)
    fig3.savefig(f"{IMG_DIR}/cube_embeded_atoms_lambda_sweep.png", dpi=150, bbox_inches='tight')
    plt.show()


def mode_convergence(d=0.5, alpha=1.8, n_blend=4, margin=5):
    zeta_1s = 1.24
    zeta_2p = 1.72
    exact_1s = analytic_1s_norm_sq(STO3G_1S, zeta_1s)
    exact_2p = analytic_2p_norm_sq(STO3G_2P_C, zeta_2p)

    ns_test = [6, 8, 10, 12, 16, 20]
    errors_1s = []
    errors_2p = []
    npts_list = []

    print("\n── Convergence test ──")
    for n_test in ns_test:
        grid_dict = build_grid(d=d, n=n_test, alpha=alpha, n_blend=n_blend)
        combined = build_combined_grid(grid_dict, margin=margin)
        xy = combined['combined_xy']
        w = combined['combined_w']

        r2 = xy[:, 0]**2 + xy[:, 1]**2
        rho_1s = gaussian_1s(r2, STO3G_1S, zeta_1s)**2
        num_1s = np.sum(rho_1s * w)
        err_1s = abs(num_1s - exact_1s) / exact_1s * 100
        errors_1s.append(err_1s)

        rho_2p = gaussian_2p(xy[:, 0], xy[:, 1], STO3G_2P_C, zeta_2p)**2
        num_2p = np.sum(rho_2p * w)
        err_2p = abs(num_2p - exact_2p) / exact_2p * 100
        errors_2p.append(err_2p)

        npts_list.append(len(xy))
        print(f"  n={n_test:2d}: Npts={len(xy):5d}, err_1s={err_1s:.4f}%, err_2p={err_2p:.4f}%")

    fig = plot_convergence(ns_test, errors_1s, errors_2p, npts_list)
    fig.savefig(f"{IMG_DIR}/cube_embeded_atoms_convergence.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CubeEmbededAtoms grid demos")
    parser.add_argument('--mode', default='all',
                        choices=['grid', 'integration', 'weights', 'convergence', 'all'])
    parser.add_argument('--d', type=float, default=0.5)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=1.8)
    parser.add_argument('--n-blend', type=int, default=4)
    args = parser.parse_args()

    if args.mode in ('grid', 'all'):
        mode_grid(d=args.d, n=args.n, alpha=args.alpha, n_blend=args.n_blend)
    if args.mode in ('integration', 'all'):
        mode_integration(d=args.d, n=args.n, alpha=args.alpha, n_blend=args.n_blend)
    if args.mode in ('weights', 'all'):
        mode_weights(d=args.d, n=args.n, alpha=args.alpha, n_blend=args.n_blend)
    if args.mode in ('convergence', 'all'):
        mode_convergence(d=args.d, alpha=args.alpha, n_blend=args.n_blend)
