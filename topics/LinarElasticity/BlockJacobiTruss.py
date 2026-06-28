"""
BlockJacobiTruss.py — Benchmark script: global Jacobi vs block Jacobi with
overlapping patches, stiffness-weighted averaging, and heavy-ball momentum.

This is a thin wrapper that imports solver functions from TrussSolver and
geometry functions from Truss.  See BlockJacobiGPU.md for full derivation.

Key points:
  A = M/dt^2 + K,  D_i = m_i/dt^2 + sum_{j in N(i)} k_ij
  Global Jacobi+HB: v = beta*v + omega*(r/D), x += v
  Block Jacobi: overlapping patches do nInner local Jacobi steps,
    then corrections are blended by W_i^(p) = D_i^(p) / D_i.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from Truss import build_triangular_grid, grid_edges, boundary_nodes, node_index
from TrussSolver import (
    build_adjacency, classify_edges, bfs_distances,
    compute_edge_data, matvec_A, compute_diagonal_3x3, invert_3x3_blocks,
    assemble_dense_A,
    solve_global_jacobi,
    make_grid_patch, build_patches_grid, build_patches_grid_nonoverlap,
    build_patches_1d, build_patches_1d_nonoverlap, build_patches,
    setup_patch_data, relax_patch_from_global, relax_patch_direct,
    solve_block_jacobi, solve_alternating_patches,
    solve_block_jacobi_direct, solve_alternating_patches_direct,
)
from TrussPlotting import (
    plot_convergence, plot_beta_sweep,
    plot_deformation_with_patches, plot_per_node_error,
    plot_1d_partitioning, plot_1d_partitioning_alt,
)


# ---- helpers ----

def _err(x, x_ref, free_mask):
    """Relative error of x vs x_ref on free DOFs."""
    return np.linalg.norm(x[free_mask] - x_ref[free_mask]) / (
        np.linalg.norm(x_ref[free_mask]) + 1e-30)

def _print_coverage(patches, alt_patch_sets_data, free_mask, n_nodes):
    """Print patch coverage statistics."""
    coverage = np.zeros(n_nodes, dtype=np.int32)
    for p in patches:
        coverage[p['vertices']] += 1
    print(f"  coverage: min={coverage[free_mask].min()} max={coverage[free_mask].max()} mean={coverage[free_mask].mean():.1f}")
    print(f"Alternating sets: A={len(alt_patch_sets_data[0])} B={len(alt_patch_sets_data[1])}")
    for si, pset in enumerate(alt_patch_sets_data):
        cov = np.zeros(n_nodes, dtype=np.int32)
        for p in pset:
            cov[p['vertices']] += 1
        print(f"  Set {'AB'[si]}: coverage free min={cov[free_mask].min()} max={cov[free_mask].max()} mean={cov[free_mask].mean():.1f}")
    cov_both = np.zeros(n_nodes, dtype=np.int32)
    for pset in alt_patch_sets_data:
        for p in pset:
            cov_both[p['vertices']] += 1
    print(f"  Combined: coverage free min={cov_both[free_mask].min()} max={cov_both[free_mask].max()} mean={cov_both[free_mask].mean():.1f}")

def _run_solver_betas(solver_fn, b, x0, common_kw, beta_values, n_outer, prefix=''):
    """Run a solver for multiple beta values, return {beta: (x, res)}."""
    results = {}
    for bt in beta_values:
        betas_arr = np.full(n_outer, bt)
        betas_arr[0] = 0.0
        x, res = solver_fn(b, x0, beta=betas_arr, n_outer=n_outer, **common_kw)
        results[bt] = (x, res)
        print(f"  {prefix}β={bt:.1f} {n_outer:3d} outer: res={res[-1]:.4e}")
    return results


# ---- beam benchmark ----

def main_beam(args):
    """Beam test: 17-box cantilever, anchored by large inertia, parabolic initial bend."""
    nx, ny = 19, 2
    a, k_spring, mass_val, dt, dim = 1.0, 20000.0, 1.0, 0.02, 3
    block_size, overlap = 4, 1
    anchor_factor, bend_amplitude = 1000.0, 3.0

    pos = build_triangular_grid(nx, ny, a=a, jitter=0.0)
    edges = grid_edges(nx, ny, include_diag=True)
    n_nodes = nx * ny
    mass_dt2 = np.full(n_nodes, mass_val) / dt**2
    anchor_nodes = [node_index(0, iy, nx) for iy in range(ny)]
    mass_dt2[anchor_nodes] *= anchor_factor
    free_mask = np.ones(n_nodes, dtype=bool)

    ei = np.array([e[0] for e in edges], dtype=np.int32)
    ej = np.array([e[1] for e in edges], dtype=np.int32)
    k_arr = np.array([k_spring] * len(edges))
    n_dirs, k_eff = compute_edge_data(pos, ei, ej, k_arr, dim=dim)

    D_spr_trace = np.zeros(n_nodes)
    np.add.at(D_spr_trace, ei, k_eff); np.add.at(D_spr_trace, ej, k_eff)
    print(f"Beam: {nx}x{ny} = {n_nodes} nodes, {len(edges)} edges, {nx-1} boxes")
    print(f"m/dt²={mass_dt2[0]:.1f}  anchor m/dt²={mass_dt2[anchor_nodes[0]]:.1f}  "
          f"avg spring diag={D_spr_trace[free_mask].mean():.1f}")

    gravity = np.zeros((n_nodes, dim))
    b = mass_dt2[:, None] * pos + matvec_A(pos, ei, ej, k_eff, n_dirs, np.zeros(n_nodes)) + gravity

    x0 = pos.copy()
    tn = pos[:, 0] / (pos[:, 0].max() - pos[:, 0].min())
    x0[:, 1] += bend_amplitude * tn * tn
    x0 += 0.01 * np.random.default_rng(args.seed).standard_normal((n_nodes, dim))

    A_dense = assemble_dense_A(ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim)
    x_direct = np.linalg.solve(A_dense, b.reshape(-1)).reshape(n_nodes, dim)
    print(f"Direct solve: max displacement = {np.linalg.norm(x_direct[free_mask] - pos[free_mask], axis=1).max():.6f}")

    MAX_BUDGET = 32
    inner_options = [2, 4, 8]
    n_gj = MAX_BUDGET
    omega_val, beta_val, omega_inner_val = 0.8, 0.0, 1.0

    x_gj, res_gj = solve_global_jacobi(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
                                        free_mask=free_mask, omega=0.8, beta=0.0, n_iter=n_gj)
    x_gj05, res_gj05 = solve_global_jacobi(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
                                            free_mask=free_mask, omega=0.8, beta=0.5, n_iter=n_gj)

    patches = build_patches_1d(nx, ny, block_size=block_size, overlap=overlap, free_mask=free_mask)
    vp, D_global, D_global_inv = setup_patch_data(patches, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, free_mask, dim=dim, pos=pos, gravity=gravity)

    alt_shift = block_size // 2
    alt_patch_sets_data = []
    for pset in [build_patches_1d_nonoverlap(nx, ny, block_size=block_size, shift_x=0, free_mask=free_mask),
                 build_patches_1d_nonoverlap(nx, ny, block_size=block_size, shift_x=alt_shift, free_mask=free_mask)]:
        setup_patch_data(pset, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, free_mask, dim=dim, pos=pos, gravity=gravity, local_dirichlet=True)
        alt_patch_sets_data.append(pset)

    print(f"\nOverlapping patches: {len(patches)}  block={block_size}  overlap={overlap}")
    _print_coverage(patches, alt_patch_sets_data, free_mask, n_nodes)

    bj_configs = [(MAX_BUDGET, ni) for ni in inner_options]
    alt_configs = [(MAX_BUDGET, ni) for ni in inner_options]
    colors = ['r', 'g', 'm']
    straight_edges, diag_edges = classify_edges(edges, nx, ny)

    print(f"\n{'='*80}\nBudget: {MAX_BUDGET} total iterations\n{'='*80}")
    print(f"Global Jacobi  {n_gj} iter:  res={res_gj[-1]:.4e}  err={_err(x_gj, x_direct, free_mask):.4e}")

    # Direct solvers
    print(f"\n--- direct local solve (LU) ---")
    direct_bj_results, direct_alt_results = {}, {}
    for n_outer_dir in [1, 2, 4, 8, 16, 32]:
        x_dbj, res_dbj = solve_block_jacobi_direct(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
            patches=patches, vp=vp, free_mask=free_mask, D_global_inv=D_global_inv,
            omega=omega_val, beta=beta_val, use_scalar_weight=True, n_outer=n_outer_dir)
        direct_bj_results[n_outer_dir] = (x_dbj, res_dbj)
        print(f"  DirBJ  {n_outer_dir:3d} outer: res={res_dbj[-1]:.4e}  err={_err(x_dbj, x_direct, free_mask):.4e}")

    betas05 = np.full(MAX_BUDGET, 0.5); betas05[0] = 0.0
    _, res_dbj05 = solve_block_jacobi_direct(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
        patches=patches, vp=vp, free_mask=free_mask, D_global_inv=D_global_inv,
        omega=omega_val, beta=betas05, use_scalar_weight=True, n_outer=MAX_BUDGET)

    for n_outer_dir in [1, 2, 4, 8, 16, 32]:
        x_dalt, res_dalt = solve_alternating_patches_direct(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
            patch_sets=alt_patch_sets_data, free_mask=free_mask, omega=omega_val, beta=beta_val, n_outer=n_outer_dir)
        direct_alt_results[n_outer_dir] = (x_dalt, res_dalt)
        print(f"  DirAlt {n_outer_dir:3d} outer = {n_outer_dir*2:3d} total (2x): res={res_dalt[-1]:.4e}  err={_err(x_dalt, x_direct, free_mask):.4e}")

    betas05_alt = np.full(MAX_BUDGET, 0.5); betas05_alt[0] = 0.0
    _, res_dalt05 = solve_alternating_patches_direct(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
        patch_sets=alt_patch_sets_data, free_mask=free_mask, omega=omega_val, beta=betas05_alt, n_outer=MAX_BUDGET)

    print(f"\n--- direct Alt beta sweep ---")
    direct_alt_beta_results = _run_solver_betas(
        solve_alternating_patches_direct, b, x0,
        dict(ei=ei, ej=ej, k_eff=k_eff, n_dirs=n_dirs, mass_dt2=mass_dt2, n_nodes=n_nodes, dim=dim,
             patch_sets=alt_patch_sets_data, free_mask=free_mask, omega=omega_val),
        [0.0, 0.3, 0.5, 0.7], MAX_BUDGET, prefix='DirAlt ')

    # Jacobi-based solvers (both weightings)
    bj_results, alt_results = {}, {}
    bj_results_beta05, alt_results_beta05 = {}, {}
    for mode_name, use_scalar in [("scalar", True), ("matrix", False)]:
        print(f"\n--- {mode_name} weighting ---")
        for (no, ni) in bj_configs:
            x_bj, res_bj = solve_block_jacobi(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
                patches=patches, vp=vp, free_mask=free_mask, D_global_inv=D_global_inv,
                omega=omega_val, beta=beta_val, omega_inner=omega_inner_val, n_outer=no, n_inner=ni, use_scalar_weight=use_scalar)
            bj_results[(no, ni)] = (x_bj, res_bj)
            print(f"  BJ  {no:3d}x{ni:2d} = {no*ni:3d} total: res={res_bj[-1]:.4e}  err={_err(x_bj, x_direct, free_mask):.4e}")
        for (no, ni) in alt_configs:
            x_alt, res_alt = solve_alternating_patches(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
                patch_sets=alt_patch_sets_data, free_mask=free_mask,
                omega=omega_val, beta=beta_val, omega_inner=omega_inner_val, n_outer=no, n_inner=ni)
            alt_results[(no, ni)] = (x_alt, res_alt)
            print(f"  Alt {no:3d}x{ni:2d} = {no*ni*2:3d} total (2x): res={res_alt[-1]:.4e}  err={_err(x_alt, x_direct, free_mask):.4e}")

    for (no, ni) in bj_configs:
        x_bj, res_bj = solve_block_jacobi(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
            patches=patches, vp=vp, free_mask=free_mask, D_global_inv=D_global_inv,
            omega=omega_val, beta=0.5, omega_inner=omega_inner_val, n_outer=no, n_inner=ni, use_scalar_weight=False)
        bj_results_beta05[(no, ni)] = (x_bj, res_bj)
    for (no, ni) in alt_configs:
        x_alt, res_alt = solve_alternating_patches(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
            patch_sets=alt_patch_sets_data, free_mask=free_mask,
            omega=omega_val, beta=0.5, omega_inner=omega_inner_val, n_outer=no, n_inner=ni)
        alt_results_beta05[(no, ni)] = (x_alt, res_alt)

    # --- Plotting ---
    phys_str = f"k={k_spring:.0f}, m={mass_val:.0f}, dt={dt:.3f}, anchor×{anchor_factor:.0f}, bend={bend_amplitude}"
    patch_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999']
    snap_iters = [1, 2, 4, 8, 16, 32]
    snap_colors = ['red', 'orange', 'gold', 'green', 'cyan', 'blue']

    # ======= Figure 1: Overlapping (DirBJ) =======
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    ax_beam = fig.add_subplot(gs[0, :])
    ax_conv = fig.add_subplot(gs[1, 0])
    ax_beta = fig.add_subplot(gs[1, 1])
    fig.suptitle(f"Overlapping patches (DirBJ) — {len(patches)} patches, block={block_size}, overlap={overlap}\n{phys_str}", fontsize=10)

    ax = ax_beam
    plot_1d_partitioning(ax, patches, patch_colors)
    for i, j in edges:
        ax.plot([x0[i, 0], x0[j, 0]], [x0[i, 1], x0[j, 1]], 'k--', lw=0.8, alpha=0.4)
    for i, j in edges:
        ax.plot([x_direct[i, 0], x_direct[j, 0]], [x_direct[i, 1], x_direct[j, 1]], 'b-', lw=1.5, alpha=0.8)
    ax.plot([], [], 'b-', lw=2, label='direct')
    for n_iter, col in zip(snap_iters, snap_colors):
        if n_iter in direct_bj_results:
            x_snap, res_snap = direct_bj_results[n_iter]
            if res_snap[-1] > 1.0:
                ax.plot([], [], color=col, lw=1, alpha=0.3, label=f'DirBJ {n_iter} (diverged)')
                continue
            for i, j in edges:
                ax.plot([x_snap[i, 0], x_snap[j, 0]], [x_snap[i, 1], x_snap[j, 1]], color=col, lw=0.8, alpha=0.6)
            ax.plot([], [], color=col, lw=1, label=f'DirBJ {n_iter}')
    ax.set_aspect('equal'); ax.set_title('Beam shape snapshots + 1D partitioning')
    ax.set_ylim(-3.0, 4.0); ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # Convergence
    ax = ax_conv
    conv_data = [
        (res_gj, 'Glob.Jac β=0.0', 'k', '-'),
        (res_gj05, 'Glob.Jac β=0.5', 'k', ':'),
    ]
    if 32 in direct_bj_results:
        conv_data.append((direct_bj_results[32][1], 'DirBJ β=0.0', 'blue', '-'))
        conv_data.append((res_dbj05, 'DirBJ β=0.5', 'blue', ':'))
    for (no, ni), col in zip(bj_configs, colors):
        conv_data.append((bj_results[(no, ni)][1], f'Block.Jac {ni}sub β=0.0', col, '-'))
        conv_data.append((bj_results_beta05[(no, ni)][1], f'Block.Jac {ni}sub β=0.5', col, ':'))
    plot_convergence(ax, [d[0] for d in conv_data], [d[1] for d in conv_data],
                     [d[2] for d in conv_data], [d[3] for d in conv_data],
                     title=f'Convergence (budget={MAX_BUDGET})', xlabel='Outer iterations')

    # Beta sweep
    ax = ax_beta
    ax.semilogy(np.arange(0, n_gj + 1), res_gj, 'k-', lw=2, label='Glob.Jac β=0.0')
    beta_colors = ['blue', 'red', 'green', 'purple']
    for bt, bcol in zip([0.0, 0.3, 0.5, 0.7], beta_colors):
        betas_arr = np.full(MAX_BUDGET, bt); betas_arr[0] = 0.0
        _, res_dbj = solve_block_jacobi_direct(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
            patches=patches, vp=vp, free_mask=free_mask, D_global_inv=D_global_inv,
            omega=omega_val, beta=betas_arr, use_scalar_weight=True, n_outer=MAX_BUDGET)
        ax.semilogy(np.arange(0, MAX_BUDGET + 1), res_dbj, color=bcol, marker='D', ms=3, label=f'DirBJ β={bt:.1f}')
    ax.set_xlabel('Outer iterations'); ax.set_ylabel('Relative residual')
    ax.set_title('DirBJ beta sweep'); ax.set_ylim(bottom=1e-6)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('beam_overlap.png', dpi=150)
    print("\nSaved beam_overlap.png")
    if not args.no_show: plt.show()
    plt.close(fig)

    # ======= Figure 2: Alternating (DirAlt) =======
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    ax_beam = fig.add_subplot(gs[0, :])
    ax_conv = fig.add_subplot(gs[1, 0])
    ax_beta = fig.add_subplot(gs[1, 1])
    fig.suptitle(f"Alternating patches (DirAlt) — sets A={len(alt_patch_sets_data[0])} B={len(alt_patch_sets_data[1])}, shift={alt_shift}\n{phys_str}", fontsize=10)

    ax = ax_beam
    set_colors_a = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']
    set_colors_b = ['#a65628', '#f781bf', '#999999', '#ffd92f', '#a6d854']
    plot_1d_partitioning_alt(ax, alt_patch_sets_data, [set_colors_a, set_colors_b])
    for i, j in edges:
        ax.plot([x0[i, 0], x0[j, 0]], [x0[i, 1], x0[j, 1]], 'k--', lw=0.8, alpha=0.4)
    for i, j in edges:
        ax.plot([x_direct[i, 0], x_direct[j, 0]], [x_direct[i, 1], x_direct[j, 1]], 'b-', lw=1.5, alpha=0.8)
    ax.plot([], [], 'b-', lw=2, label='direct')
    alt_snap_totals = [2, 4, 8, 16, 32, 64]
    for n_outer_dir, total, col in zip(snap_iters, alt_snap_totals, snap_colors[:6]):
        if n_outer_dir in direct_alt_results:
            x_snap = direct_alt_results[n_outer_dir][0]
            for i, j in edges:
                ax.plot([x_snap[i, 0], x_snap[j, 0]], [x_snap[i, 1], x_snap[j, 1]], color=col, lw=0.8, alpha=0.6)
            ax.plot([], [], color=col, lw=1, label=f'DirAlt {total}')
    ax.set_aspect('equal'); ax.set_title('Beam shape snapshots + 1D partitioning (A=red/orange, B=brown/pink)')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # Convergence
    ax = ax_conv
    conv_data = [
        (res_gj, 'Glob.Jac β=0.0', 'k', '-'),
        (res_gj05, 'Glob.Jac β=0.5', 'k', ':'),
    ]
    if 32 in direct_alt_results:
        conv_data.append((direct_alt_results[32][1], 'DirAlt β=0.0', 'blue', '-'))
        conv_data.append((res_dalt05, 'DirAlt β=0.5', 'blue', ':'))
    for (no, ni), col in zip(alt_configs, colors):
        conv_data.append((alt_results[(no, ni)][1], f'Alt.Block.Jac {ni}sub β=0.0', col, '-'))
        conv_data.append((alt_results_beta05[(no, ni)][1], f'Alt.Block.Jac {ni}sub β=0.5', col, ':'))
    plot_convergence(ax, [d[0] for d in conv_data], [d[1] for d in conv_data],
                     [d[2] for d in conv_data], [d[3] for d in conv_data],
                     title=f'Convergence (budget={MAX_BUDGET})', xlabel='Outer iterations')

    # Beta sweep
    ax = ax_beta
    ax.semilogy(np.arange(0, n_gj + 1), res_gj, 'k-', lw=2, label='Glob.Jac β=0.0')
    for bt, bcol in zip([0.0, 0.3, 0.5, 0.7], beta_colors):
        if bt in direct_alt_beta_results:
            ax.semilogy(np.arange(0, MAX_BUDGET + 1), direct_alt_beta_results[bt][1],
                        color=bcol, marker='s', ms=3, label=f'DirAlt β={bt:.1f}')
    ax.set_xlabel('Outer iterations'); ax.set_ylabel('Relative residual')
    ax.set_title('DirAlt beta sweep'); ax.set_ylim(bottom=1e-6)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('beam_alternating.png', dpi=150)
    print("Saved beam_alternating.png")
    if not args.no_show: plt.show()
    plt.close(fig)


# ---- benchmark / visualization ----

def main():
    parser = argparse.ArgumentParser(description="Block Jacobi truss solver benchmark.")
    parser.add_argument('--mode', choices=['grid', 'beam'], default='grid',
                        help='Test mode: 2D grid or 1D beam')
    parser.add_argument('--no-show', action='store_true', help='Do not call plt.show()')
    parser.add_argument('--strain', type=float, default=0.3,
                        help='Initial strain factor: x0 = pos * (1 + strain)')
    parser.add_argument('--jitter', type=float, default=0.1,
                        help='Random shift amplitude added to initial guess')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for jitter')
    parser.add_argument('--block-size', type=int, default=8, help='Grid patch block size')
    parser.add_argument('--overlap', type=int, default=1, help='Grid patch overlap')
    parser.add_argument('--n-outer', type=int, default=15, help='Outer iterations / kernel launches')
    args = parser.parse_args()

    if args.mode == 'beam':
        main_beam(args)
        return

    nx, ny = 32, 32
    a, k_spring, mass_val, dt, dim = 1.0, 20000.0, 1.0, 0.02, 3

    pos = build_triangular_grid(nx, ny, a=a, jitter=0.0)
    edges = grid_edges(nx, ny, include_diag=True)
    n_nodes = nx * ny
    mass_dt2 = np.full(n_nodes, mass_val) / dt**2

    fixed = set(boundary_nodes(nx, ny, which="bottom"))
    free_mask = np.ones(n_nodes, dtype=bool)
    for f in fixed:
        free_mask[f] = False

    ei = np.array([e[0] for e in edges], dtype=np.int32)
    ej = np.array([e[1] for e in edges], dtype=np.int32)
    k_arr = np.array([k_spring] * len(edges))
    n_dirs, k_eff = compute_edge_data(pos, ei, ej, k_arr, dim=dim)

    D_spr_trace = np.zeros(n_nodes)
    np.add.at(D_spr_trace, ei, k_eff); np.add.at(D_spr_trace, ej, k_eff)
    print(f"m/dt²={mass_dt2[0]:.1f}  avg spring diag(trace)={D_spr_trace[free_mask].mean():.1f}  "
          f"ratio inertial/total={mass_dt2[0]/(mass_dt2[0]+D_spr_trace[free_mask].mean()):.2%}")

    gravity = np.zeros((n_nodes, dim))
    gravity[:, 1] = -9.81 * mass_val
    b = mass_dt2[:, None] * pos + matvec_A(pos, ei, ej, k_eff, n_dirs, np.zeros(n_nodes)) + gravity

    rng = np.random.default_rng(args.seed)
    x0 = pos * (1.0 + args.strain)
    x0[free_mask] += args.jitter * rng.standard_normal((n_nodes, dim))[free_mask]
    x0[~free_mask] = pos[~free_mask]

    A_dense = assemble_dense_A(ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim, fixed)
    b_flat = b.reshape(-1).copy()
    for f in fixed:
        for d in range(dim):
            b_flat[f*dim+d] = pos[f, d]
    x_direct = np.linalg.solve(A_dense, b_flat).reshape(n_nodes, dim)

    MAX_BUDGET = 64
    inner_options = [1, 2, 4, 8]
    n_gj = MAX_BUDGET
    omega_val, beta_val, omega_inner_val = 0.8, 0.0, 1.0

    x_gj, res_gj = solve_global_jacobi(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
                                        free_mask=free_mask, omega=0.8, beta=0.0, n_iter=n_gj)

    patches = build_patches_grid(nx, ny, block_size=args.block_size, overlap=args.overlap, free_mask=free_mask)
    vp, D_global, D_global_inv = setup_patch_data(patches, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, free_mask, dim=dim, pos=pos, gravity=gravity)

    alt_shift = args.block_size // 2
    alt_patch_sets_data = []
    for pset in [build_patches_grid_nonoverlap(nx, ny, block_size=args.block_size, shift_x=0, shift_y=0, free_mask=free_mask),
                 build_patches_grid_nonoverlap(nx, ny, block_size=args.block_size, shift_x=alt_shift, shift_y=alt_shift, free_mask=free_mask)]:
        setup_patch_data(pset, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, free_mask, dim=dim, pos=pos, gravity=gravity, local_dirichlet=True)
        alt_patch_sets_data.append(pset)

    sizes = np.array([p['n_vertices'] for p in patches])
    print(f"Patches: {len(patches)}  block_size={args.block_size}  overlap={args.overlap}  grid={nx}x{ny}")
    print(f"  verts: min={sizes.min()} max={sizes.max()} mean={sizes.mean():.1f}")
    _print_coverage(patches, alt_patch_sets_data, free_mask, n_nodes)

    colors = ['r', 'g', 'm', 'orange']
    bj_configs = [(MAX_BUDGET // ni, ni) for ni in inner_options]
    alt_configs = [(MAX_BUDGET // (2 * ni), ni) for ni in inner_options]
    straight_edges, diag_edges = classify_edges(edges, nx, ny)

    err_gj = _err(x_gj, x_direct, free_mask)
    phys_str = f"k={k_spring:.0f}, m={mass_val:.0f}, dt={dt:.3f}, m/dt²={mass_val/dt**2:.0f}, strain={args.strain}, jitter={args.jitter}"
    patch_str = (f"grid={nx}x{ny}, patches={len(patches)}, alt={len(alt_patch_sets_data[0])}+{len(alt_patch_sets_data[1])}, "
                 f"block={args.block_size}, overlap={args.overlap}, shift={alt_shift}")

    print(f"\n{'='*80}\nBudget: {MAX_BUDGET} total iterations\n{'='*80}")
    print(f"Global Jacobi  {n_gj} iter:  res={res_gj[-1]:.4e}  err={err_gj:.4e}")

    # Direct solvers
    print(f"\n--- direct local solve (LU) ---")
    direct_bj_results, direct_alt_results = {}, {}
    for n_outer_dir in [MAX_BUDGET, MAX_BUDGET // 2, MAX_BUDGET // 4]:
        x_dbj, res_dbj = solve_block_jacobi_direct(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
            patches=patches, vp=vp, free_mask=free_mask, D_global_inv=D_global_inv,
            omega=omega_val, beta=beta_val, use_scalar_weight=True, n_outer=n_outer_dir)
        direct_bj_results[n_outer_dir] = (x_dbj, res_dbj)
        print(f"  DirBJ  {n_outer_dir:3d} outer: res={res_dbj[-1]:.4e}  err={_err(x_dbj, x_direct, free_mask):.4e}")

    for n_outer_dir in [MAX_BUDGET // 2, MAX_BUDGET // 4, MAX_BUDGET // 8]:
        x_dalt, res_dalt = solve_alternating_patches_direct(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
            patch_sets=alt_patch_sets_data, free_mask=free_mask, omega=omega_val, beta=beta_val, n_outer=n_outer_dir)
        direct_alt_results[n_outer_dir] = (x_dalt, res_dalt)
        print(f"  DirAlt {n_outer_dir:3d} outer = {n_outer_dir*2:3d} total (2x): res={res_dalt[-1]:.4e}  err={_err(x_dalt, x_direct, free_mask):.4e}")

    print(f"\n--- direct Alt beta sweep ---")
    direct_alt_beta_results = _run_solver_betas(
        solve_alternating_patches_direct, b, x0,
        dict(ei=ei, ej=ej, k_eff=k_eff, n_dirs=n_dirs, mass_dt2=mass_dt2, n_nodes=n_nodes, dim=dim,
             patch_sets=alt_patch_sets_data, free_mask=free_mask, omega=omega_val),
        [0.0, 0.3, 0.5], MAX_BUDGET // 2, prefix='DirAlt ')

    # Jacobi-based solvers + plots per weighting mode
    for mode_name, use_scalar in [("scalar", True), ("matrix", False)]:
        print(f"\n--- {mode_name} weighting ---")
        bj_results, alt_results = {}, {}
        for (no, ni) in bj_configs:
            x_bj, res_bj = solve_block_jacobi(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
                patches=patches, vp=vp, free_mask=free_mask, D_global_inv=D_global_inv,
                omega=omega_val, beta=beta_val, omega_inner=omega_inner_val, n_outer=no, n_inner=ni, use_scalar_weight=use_scalar)
            bj_results[(no, ni)] = (x_bj, res_bj)
            print(f"  BJ  {no:3d}x{ni:2d} = {no*ni:3d} total: res={res_bj[-1]:.4e}  err={_err(x_bj, x_direct, free_mask):.4e}")
        for (no, ni) in alt_configs:
            x_alt, res_alt = solve_alternating_patches(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
                patch_sets=alt_patch_sets_data, free_mask=free_mask,
                omega=omega_val, beta=beta_val, omega_inner=omega_inner_val, n_outer=no, n_inner=ni)
            alt_results[(no, ni)] = (x_alt, res_alt)
            print(f"  Alt {no:3d}x{ni:2d} = {no*ni*2:3d} total (2x): res={res_alt[-1]:.4e}  err={_err(x_alt, x_direct, free_mask):.4e}")

        solver_str = f"{'scalar W' if use_scalar else 'matrix Dg⁻¹Dp'}  |  ω={omega_val}, β={beta_val}, ωi={omega_inner_val}"
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
        fig.suptitle(f"{phys_str}\n{patch_str}\n{solver_str}", fontsize=8)

        # convergence
        ax = axes[0]
        conv_data = [(res_gj, f'GJ ({n_gj})', 'b', '-')]
        for (no, ni), col in zip(bj_configs, colors):
            conv_data.append((bj_results[(no, ni)][1], f'BJ {no}x{ni} ({no*ni})', col, '-'))
        for (no, ni), col in zip(alt_configs, colors):
            conv_data.append((alt_results[(no, ni)][1], f'Alt {no}x{ni} ({no*ni*2})', col, '--'))
        for n_outer_dir, col in [(MAX_BUDGET, '#888888'), (MAX_BUDGET//2, '#444444'), (MAX_BUDGET//4, '#222222')]:
            if n_outer_dir in direct_bj_results:
                conv_data.append((direct_bj_results[n_outer_dir][1], f'DirBJ ({n_outer_dir})', col, '-'))
        for n_outer_dir, col in [(MAX_BUDGET//2, 'lime'), (MAX_BUDGET//4, 'green'), (MAX_BUDGET//8, 'darkgreen')]:
            if n_outer_dir in direct_alt_results:
                conv_data.append((direct_alt_results[n_outer_dir][1], f'DirAlt ({n_outer_dir*2})', col, '-.'))
        plot_convergence(ax, [d[0] for d in conv_data], [d[1] for d in conv_data],
                         [d[2] for d in conv_data], [d[3] for d in conv_data],
                         title=f'Convergence (budget={MAX_BUDGET})', xlabel='Total iterations')

        # deformation + patch boxes
        x_bj_best = bj_results[bj_configs[2]][0]
        plot_deformation_with_patches(axes[1], pos, edges, x_direct, x_bj_best, patches,
                                      straight_edges=straight_edges,
                                      solved_label=f'BJ {bj_configs[2][0]}x{bj_configs[2][1]}')

        # per-node error
        sc, _ = plot_per_node_error(axes[2], pos, edges, x_direct, x_bj_best, straight_edges=straight_edges)
        fig.colorbar(sc, ax=axes[2], label='||x_bj - x_direct||')

        plt.tight_layout(rect=[0, 0, 1, 0.88])
        fname = f'block_jacobi_{mode_name}.png'
        plt.savefig(fname, dpi=150)
        print(f"Saved {fname}")
        if not args.no_show: plt.show()
        plt.close(fig)

    # --- alternating patches: beta sweep ---
    print(f"\n=== alternating shifted patches: beta sweep (budget={MAX_BUDGET}) ===")
    beta_values = [0.0, 0.3, 0.5]
    n_inner_colors = {1: 'r', 2: 'g', 4: 'm', 8: 'orange'}
    alt_beta_results = {}
    for bt in beta_values:
        for (no, ni) in alt_configs:
            betas_arr = np.full(no, bt); betas_arr[0] = 0.0
            x_alt, res_alt = solve_alternating_patches(b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
                patch_sets=alt_patch_sets_data, free_mask=free_mask,
                omega=omega_val, beta=betas_arr, omega_inner=omega_inner_val, n_outer=no, n_inner=ni)
            alt_beta_results[(f'{bt:.1f}', ni)] = (x_alt, res_alt)
            print(f"  Alt β={bt:.1f} {no:3d}x{ni:2d} = {no*ni*2:3d} total: res={res_alt[-1]:.4e}  err={_err(x_alt, x_direct, free_mask):.4e}")

    fig, axes = plt.subplots(1, len(beta_values) + 1, figsize=(20, 5.4))
    alt_solver_str = (f"alternating shifted  |  ω={omega_val}, ωi={omega_inner_val}, β[0]=0  |  "
                      f"sets: A={len(alt_patch_sets_data[0])} B={len(alt_patch_sets_data[1])} shift={alt_shift}")
    fig.suptitle(f"{phys_str}\n{patch_str}\n{alt_solver_str}", fontsize=8)

    for ax_idx, bt in enumerate(beta_values):
        ax = axes[ax_idx]
        blabel = f'{bt:.1f}'
        conv_data = [(res_gj, 'GJ', 'b', '-')]
        for ni in inner_options:
            key = (blabel, ni)
            if key in alt_beta_results:
                no = MAX_BUDGET // (2 * ni)
                conv_data.append((alt_beta_results[key][1], f'{no}x{ni} ({no*ni*2})', n_inner_colors[ni], '-'))
        plot_convergence(ax, [d[0] for d in conv_data], [d[1] for d in conv_data],
                         [d[2] for d in conv_data], [d[3] for d in conv_data],
                         title=f'β={blabel}', xlabel='Total iterations')

    # deformation + patch boxes (best: beta=0.3, n_inner=4)
    x_alt_best = alt_beta_results[('0.3', 4)][0]
    patch_colors = ['r', 'g', 'm', 'orange', 'cyan', 'purple', 'brown']
    all_patches_flat = [p for pset in alt_patch_sets_data for p in pset]
    plot_deformation_with_patches(axes[-1], pos, edges, x_direct, x_alt_best, all_patches_flat,
                                  patch_colors=patch_colors, straight_edges=straight_edges,
                                  solved_label='Alt β=0.3', title='Deformation (solid=A, dashed=B)')

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig('block_jacobi_alternating.png', dpi=150)
    print("Saved block_jacobi_alternating.png")
    if not args.no_show: plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
