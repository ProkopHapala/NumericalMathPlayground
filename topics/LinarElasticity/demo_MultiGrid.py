"""
demo_MultiGrid.py — Demonstrate multigrid solvers for truss elasticity.

Compares three prolongation strategies (spectral, geometric pivots, coarse beam)
against plain Jacobi on a refined truss beam.

Usage:
    python demo_MultiGrid.py [options]

Options:
    --mode {beam,grid}    Test geometry (default: beam)
    --no-show             Do not call plt.show()
    --seed N              Random seed (default: 42)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from Truss import (
    build_triangular_grid, grid_edges, boundary_nodes, node_index,
    Truss, assemble_stiffness_typed, mass_matrix, apply_dirichlet,
)
from TrussSolver import (
    build_adjacency, compute_edge_data, matvec_A,
    compute_diagonal_3x3, invert_3x3_blocks, assemble_dense_A,
    solve_global_jacobi,
)
from MultiGrid import (
    build_spectral_prolongation, select_pivots_maximin,
    build_pivot_prolongation, build_beam_prolongation,
    compute_fine_node_to_beam, galerkin_coarse_operator,
    solve_multigrid, matvec_A_flat,
)
from TrussPlotting import (
    plot_truss, plot_convergence, plot_multigrid_convergence,
    plot_coarse_overlay, plot_pivot_selection, plot_prolongation_modes,
    plot_per_node_error,
)


def _err(x, x_ref, free_mask):
    return np.linalg.norm(x[free_mask] - x_ref[free_mask]) / (
        np.linalg.norm(x_ref[free_mask]) + 1e-30)


def main_beam(args):
    """Beam test: refined truss from coarse graph, compare multigrid strategies."""
    # --- Build coarse truss and refine ---
    coarse_pos = np.array([
        [0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [8.0, 0.0, 0.0],
        [12.0, 0.0, 0.0],
        [16.0, 0.0, 0.0],
    ])
    coarse_edges = [(0, 1), (1, 2), (2, 3), (3, 4)]

    t = Truss(coarse_pos, coarse_edges, thickness=0.5)
    t.refine(method='regular')
    t.assemble(k_long=20000.0, k_perp=5000.0, k_diag=10000.0,
               mass_value=1.0, fixed_coarse_nodes=[0], dim=3)

    fine_pos = t.fine_pos
    fine_edges = t.fine_edges
    n_nodes = fine_pos.shape[0]
    dim = 3

    # t.mask is per-DOF (length n_nodes*dim); derive per-node mask
    mask_dof = t.mask
    free_mask = mask_dof.reshape(n_nodes, dim)[:, 0].copy()
    n_free = int(free_mask.sum())
    print(f"Fine mesh: {n_nodes} nodes, {len(fine_edges)} edges, {n_free} free nodes")

    # --- Setup solver data ---
    ei = np.array([e[0] for e in fine_edges], dtype=np.int32)
    ej = np.array([e[1] for e in fine_edges], dtype=np.int32)
    k_arr = np.array([20000.0 if (len(e) > 2 and e[2] == 'long') else
                      5000.0 if (len(e) > 2 and e[2] == 'perp') else
                      10000.0 for e in fine_edges])
    n_dirs, k_eff = compute_edge_data(fine_pos, ei, ej, k_arr, dim=dim)

    dt = 0.02
    mass_dt2 = np.ones(n_nodes) / dt**2
    # Apply Dirichlet: fixed nodes get huge mass
    fixed_nodes = np.where(~free_mask)[0].tolist()
    mass_dt2[fixed_nodes] *= 1000.0

    # --- RHS: gravity-like load ---
    gravity = np.zeros((n_nodes, dim))
    gravity[:, 1] = -9.81
    b = mass_dt2[:, None] * fine_pos + matvec_A(fine_pos, ei, ej, k_eff, n_dirs, np.zeros(n_nodes)) + gravity

    # --- Initial guess: parabolic bend + noise ---
    rng = np.random.default_rng(args.seed)
    x0 = fine_pos.copy()
    tn = fine_pos[:, 0] / (fine_pos[:, 0].max() - fine_pos[:, 0].min() + 1e-30)
    x0[:, 1] += 3.0 * tn * tn
    x0[free_mask] += 0.01 * rng.standard_normal((n_nodes, dim))[free_mask]
    x0[~free_mask] = fine_pos[~free_mask]

    # --- Direct solve ---
    A_dense = assemble_dense_A(ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim,
                               fixed_nodes=fixed_nodes)
    b_flat = b.reshape(-1).copy()
    for f in fixed_nodes:
        for d in range(dim):
            b_flat[f * dim + d] = fine_pos[f, d]
    x_direct = np.linalg.solve(A_dense, b_flat).reshape(n_nodes, dim)
    print(f"Direct solve: max disp = {np.linalg.norm(x_direct[free_mask] - fine_pos[free_mask], axis=1).max():.6f}")

    # --- Plain Jacobi baseline ---
    n_jacobi = 200
    x_jac, res_jac = solve_global_jacobi(b, x0, ei, ej, k_eff, n_dirs, mass_dt2,
                                         n_nodes, dim=dim, free_mask=free_mask,
                                         omega=0.8, beta=0.5, n_iter=n_jacobi)
    print(f"Jacobi {n_jacobi} iter: res={res_jac[-1]:.4e}  err={_err(x_jac, x_direct, free_mask):.4e}")

    # --- Strategy A: Spectral prolongation ---
    print("\n--- Spectral prolongation (Lanczos) ---")
    m_spectral = 12
    P_spectral, eigvals = build_spectral_prolongation(
        ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
        m=m_spectral, free_mask=free_mask)
    print(f"  P shape: {P_spectral.shape}, lowest eigenvalues: {eigvals[:5]}")

    x_mg_s, res_mg_s, A_c_s = solve_multigrid(
        b, x0, P_spectral, ei, ej, k_eff, n_dirs, mass_dt2,
        n_nodes, dim=dim, free_mask=free_mask,
        omega=0.8, n_pre_smooth=3, n_post_smooth=3,
        n_outer=100, tol=1e-8)
    print(f"  Spectral MG: {len(res_mg_s)-1} V-cycles, res={res_mg_s[-1]:.4e}, err={_err(x_mg_s, x_direct, free_mask):.4e}")

    # --- Strategy B: Geometric pivot prolongation ---
    print("\n--- Geometric pivot prolongation ---")
    adj = build_adjacency([(int(e[0]), int(e[1])) for e in fine_edges], n_nodes)
    n_pivots = 8
    pivots = select_pivots_maximin(fine_pos, adj, n_nodes, n_pivots,
                                   free_mask=free_mask, n_swap_iter=3, rng=rng)
    print(f"  Pivots: {pivots}")
    P_pivot = build_pivot_prolongation(fine_pos, pivots, dim=dim, power=2.0,
                                       r_cut=None, free_mask=free_mask)
    print(f"  P shape: {P_pivot.shape}")

    x_mg_p, res_mg_p, A_c_p = solve_multigrid(
        b, x0, P_pivot, ei, ej, k_eff, n_dirs, mass_dt2,
        n_nodes, dim=dim, free_mask=free_mask,
        omega=0.8, n_pre_smooth=3, n_post_smooth=3,
        n_outer=100, tol=1e-8)
    print(f"  Pivot MG: {len(res_mg_p)-1} V-cycles, res={res_mg_p[-1]:.4e}, err={_err(x_mg_p, x_direct, free_mask):.4e}")

    # --- Strategy C: Coarse beam prolongation ---
    print("\n--- Coarse beam prolongation ---")
    P_beam, beam_info = build_beam_prolongation(
        fine_pos, coarse_pos, coarse_edges, t.coarse_to_fine,
        dim=dim, bending=True, n_bending_modes=1, free_mask=free_mask)
    print(f"  P shape: {P_beam.shape}, coarse DOFs: {beam_info['n_coarse_dof']}")
    print(f"  Coarse nodes: {beam_info['n_coarse_nodes']}, edges: {beam_info['n_edges']}, "
          f"bend dirs: {beam_info['n_bend_dirs']}, bend modes: {beam_info['n_bending_modes']}")

    x_mg_b, res_mg_b, A_c_b = solve_multigrid(
        b, x0, P_beam, ei, ej, k_eff, n_dirs, mass_dt2,
        n_nodes, dim=dim, free_mask=free_mask,
        omega=0.8, n_pre_smooth=3, n_post_smooth=3,
        n_outer=100, tol=1e-8)
    print(f"  Beam MG: {len(res_mg_b)-1} V-cycles, res={res_mg_b[-1]:.4e}, err={_err(x_mg_b, x_direct, free_mask):.4e}")

    # --- Strategy C2: Coarse beam WITHOUT bending ---
    print("\n--- Coarse beam prolongation (no bending) ---")
    P_beam_nb, beam_info_nb = build_beam_prolongation(
        fine_pos, coarse_pos, coarse_edges, t.coarse_to_fine,
        dim=dim, bending=False, free_mask=free_mask)
    print(f"  P shape: {P_beam_nb.shape}, coarse DOFs: {beam_info_nb['n_coarse_dof']}")

    x_mg_bnb, res_mg_bnb, A_c_bnb = solve_multigrid(
        b, x0, P_beam_nb, ei, ej, k_eff, n_dirs, mass_dt2,
        n_nodes, dim=dim, free_mask=free_mask,
        omega=0.8, n_pre_smooth=3, n_post_smooth=3,
        n_outer=100, tol=1e-8)
    print(f"  Beam MG (no bend): {len(res_mg_bnb)-1} V-cycles, res={res_mg_bnb[-1]:.4e}, err={_err(x_mg_bnb, x_direct, free_mask):.4e}")

    # ===================== PLOTTING =====================

    # --- Figure 1: Convergence comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"Multigrid convergence — refined beam ({n_nodes} fine nodes, {n_free} free DOFs)\n"
                 f"k_long=20000, dt={dt}, anchor×1000, bend=3.0", fontsize=10)

    # Convergence: V-cycle iterations
    ax = axes[0]
    mg_data = [
        (res_mg_s, 'Spectral MG (m=12)', 'tab:blue', '-'),
        (res_mg_p, 'Pivot MG (k=8)', 'tab:green', '-'),
        (res_mg_b, 'Beam MG (bending)', 'tab:red', '-'),
        (res_mg_bnb, 'Beam MG (no bend)', 'tab:orange', '--'),
    ]
    plot_multigrid_convergence(ax, [d[0] for d in mg_data],
                               [d[1] for d in mg_data], [d[2] for d in mg_data],
                               [d[3] for d in mg_data],
                               title='V-cycle convergence', xlabel='V-cycle iteration')

    # Convergence: total work (pre+post smooth steps)
    ax = axes[1]
    work_per_cycle = 3 + 3  # n_pre + n_post
    mg_work = [
        (np.array(res_mg_s) , 'Spectral MG', 'tab:blue', '-'),
        (np.array(res_mg_p) , 'Pivot MG', 'tab:green', '-'),
        (np.array(res_mg_b) , 'Beam MG (bend)', 'tab:red', '-'),
        (np.array(res_mg_bnb), 'Beam MG (no bend)', 'tab:orange', '--'),
    ]
    for res, label, col, ls in mg_work:
        iters = np.arange(len(res)) * work_per_cycle
        ax.semilogy(iters, res, color=col, ls=ls, lw=1.5, label=label)
    # Jacobi for comparison
    ax.semilogy(np.arange(len(res_jac)), res_jac, 'k--', lw=1.5, label='Plain Jacobi')
    ax.set_xlabel('Total Jacobi-equivalent steps')
    ax.set_ylabel('Relative residual')
    ax.set_title('Convergence vs total work')
    ax.set_ylim(bottom=1e-8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig('multigrid_convergence.png', dpi=150)
    print("\nSaved multigrid_convergence.png")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    # --- Figure 2: Prolongation mode shapes ---
    fig = plot_prolongation_modes(fine_pos, fine_edges, P_beam, dim=dim, n_show=6,
                                  scale=0.4, coarse_pos=coarse_pos,
                                  coarse_edges=coarse_edges,
                                  title='Beam prolongation modes (first 6 DOFs)')
    plt.savefig('multigrid_beam_modes.png', dpi=150)
    print("Saved multigrid_beam_modes.png")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    # --- Figure 3: Spectral mode shapes ---
    fig = plot_prolongation_modes(fine_pos, fine_edges, P_spectral, dim=dim, n_show=6,
                                  scale=0.4,
                                  title='Spectral prolongation modes (6 lowest)')
    plt.savefig('multigrid_spectral_modes.png', dpi=150)
    print("Saved multigrid_spectral_modes.png")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    # --- Figure 4: Coarse overlay + pivot selection ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Coarse space visualization", fontsize=10)

    plot_coarse_overlay(axes[0], fine_pos, fine_edges, coarse_pos, coarse_edges,
                        title='Coarse beam overlay')

    plot_pivot_selection(axes[1], fine_pos, [(int(e[0]), int(e[1])) for e in fine_edges],
                         pivots, free_mask=free_mask, title='Pivot selection (maximin)')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('multigrid_coarse_spaces.png', dpi=150)
    print("Saved multigrid_coarse_spaces.png")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    # --- Figure 5: Deformation comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Deformation comparison — direct vs multigrid", fontsize=10)

    solutions = [
        (x_direct, 'Direct solve', 'b'),
        (x_mg_b, 'Beam MG (bending)', 'r'),
        (x_mg_s, 'Spectral MG', 'g'),
        (x_mg_p, 'Pivot MG', 'm'),
    ]
    for ax, (x_sol, label, col) in zip(axes.ravel(), solutions):
        for e in fine_edges:
            i, j = e[0], e[1]
            ax.plot([fine_pos[i, 0], fine_pos[j, 0]],
                    [fine_pos[i, 1], fine_pos[j, 1]],
                    'k-', lw=0.2, alpha=0.15)
        for e in fine_edges:
            i, j = e[0], e[1]
            ax.plot([x_sol[i, 0], x_sol[j, 0]],
                    [x_sol[i, 1], x_sol[j, 1]],
                    color=col, lw=0.5, alpha=0.7)
        ax.set_aspect('equal')
        ax.set_title(f'{label} (err={_err(x_sol, x_direct, free_mask):.2e})', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('multigrid_deformation.png', dpi=150)
    print("Saved multigrid_deformation.png")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    # --- Figure 6: Per-node error maps ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Per-node error vs direct solve", fontsize=10)

    err_solutions = [
        (x_mg_b, 'Beam MG (bend)', axes[0]),
        (x_mg_s, 'Spectral MG', axes[1]),
        (x_mg_p, 'Pivot MG', axes[2]),
    ]
    fine_edges_2t = [(e[0], e[1]) for e in fine_edges]
    for x_sol, label, ax in err_solutions:
        sc, _ = plot_per_node_error(ax, fine_pos, fine_edges_2t, x_direct, x_sol,
                                    title=label)
        fig.colorbar(sc, ax=ax, label='error')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig('multigrid_error.png', dpi=150)
    print("Saved multigrid_error.png")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    # --- Summary table ---
    print(f"\n{'='*80}")
    print(f"{'Method':<25} {'V-cycles':<10} {'Residual':<12} {'Error':<12}")
    print(f"{'='*80}")
    print(f"{'Plain Jacobi':<25} {n_jacobi:<10} {res_jac[-1]:<12.4e} {_err(x_jac, x_direct, free_mask):<12.4e}")
    print(f"{'Spectral MG (m=12)':<25} {len(res_mg_s)-1:<10} {res_mg_s[-1]:<12.4e} {_err(x_mg_s, x_direct, free_mask):<12.4e}")
    print(f"{'Pivot MG (k=8)':<25} {len(res_mg_p)-1:<10} {res_mg_p[-1]:<12.4e} {_err(x_mg_p, x_direct, free_mask):<12.4e}")
    print(f"{'Beam MG (bending)':<25} {len(res_mg_b)-1:<10} {res_mg_b[-1]:<12.4e} {_err(x_mg_b, x_direct, free_mask):<12.4e}")
    print(f"{'Beam MG (no bending)':<25} {len(res_mg_bnb)-1:<10} {res_mg_bnb[-1]:<12.4e} {_err(x_mg_bnb, x_direct, free_mask):<12.4e}")
    print(f"{'='*80}")


def main_grid(args):
    """Grid test: triangular grid with bottom-fixed boundary."""
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

    gravity = np.zeros((n_nodes, dim))
    gravity[:, 1] = -9.81 * mass_val
    b = mass_dt2[:, None] * pos + matvec_A(pos, ei, ej, k_eff, n_dirs, np.zeros(n_nodes)) + gravity

    rng = np.random.default_rng(args.seed)
    x0 = pos * (1.0 + 0.3)
    x0[free_mask] += 0.1 * rng.standard_normal((n_nodes, dim))[free_mask]
    x0[~free_mask] = pos[~free_mask]

    A_dense = assemble_dense_A(ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim, fixed)
    b_flat = b.reshape(-1).copy()
    for f in fixed:
        for d in range(dim):
            b_flat[f * dim + d] = pos[f, d]
    x_direct = np.linalg.solve(A_dense, b_flat).reshape(n_nodes, dim)

    n_jacobi = 200
    x_jac, res_jac = solve_global_jacobi(b, x0, ei, ej, k_eff, n_dirs, mass_dt2,
                                         n_nodes, dim=dim, free_mask=free_mask,
                                         omega=0.8, beta=0.5, n_iter=n_jacobi)
    print(f"Grid: {nx}x{ny} = {n_nodes} nodes, {len(edges)} edges")
    print(f"Jacobi {n_jacobi} iter: res={res_jac[-1]:.4e}")

    # Spectral MG
    print("\n--- Spectral prolongation ---")
    P_spectral, eigvals = build_spectral_prolongation(
        ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
        m=20, free_mask=free_mask)
    print(f"  P shape: {P_spectral.shape}")
    x_mg_s, res_mg_s, _ = solve_multigrid(
        b, x0, P_spectral, ei, ej, k_eff, n_dirs, mass_dt2,
        n_nodes, dim=dim, free_mask=free_mask,
        omega=0.8, n_pre_smooth=3, n_post_smooth=3,
        n_outer=100, tol=1e-8)
    print(f"  Spectral MG: {len(res_mg_s)-1} V-cycles, res={res_mg_s[-1]:.4e}")

    # Pivot MG
    print("\n--- Pivot prolongation ---")
    adj = build_adjacency([(int(e[0]), int(e[1])) for e in edges], n_nodes)
    n_pivots = 16
    pivots = select_pivots_maximin(pos, adj, n_nodes, n_pivots,
                                   free_mask=free_mask, n_swap_iter=0, rng=rng)
    P_pivot = build_pivot_prolongation(pos, pivots, dim=dim, power=2.0,
                                       r_cut=None, free_mask=free_mask)
    print(f"  P shape: {P_pivot.shape}")
    x_mg_p, res_mg_p, _ = solve_multigrid(
        b, x0, P_pivot, ei, ej, k_eff, n_dirs, mass_dt2,
        n_nodes, dim=dim, free_mask=free_mask,
        omega=0.8, n_pre_smooth=3, n_post_smooth=3,
        n_outer=100, tol=1e-8)
    print(f"  Pivot MG: {len(res_mg_p)-1} V-cycles, res={res_mg_p[-1]:.4e}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"Multigrid convergence — {nx}x{ny} grid ({n_nodes} nodes)", fontsize=10)

    ax = axes[0]
    mg_data = [
        (res_mg_s, 'Spectral MG (m=20)', 'tab:blue', '-'),
        (res_mg_p, 'Pivot MG (k=16)', 'tab:green', '-'),
    ]
    plot_multigrid_convergence(ax, [d[0] for d in mg_data],
                               [d[1] for d in mg_data], [d[2] for d in mg_data],
                               [d[3] for d in mg_data],
                               title='V-cycle convergence', xlabel='V-cycle iteration')

    ax = axes[1]
    work_per_cycle = 6
    for res, label, col, ls in mg_data:
        iters = np.arange(len(res)) * work_per_cycle
        ax.semilogy(iters, res, color=col, ls=ls, lw=1.5, label=label)
    ax.semilogy(np.arange(len(res_jac)), res_jac, 'k--', lw=1.5, label='Plain Jacobi')
    ax.set_xlabel('Total Jacobi-equivalent steps')
    ax.set_ylabel('Relative residual')
    ax.set_title('Convergence vs total work')
    ax.set_ylim(bottom=1e-8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig('multigrid_grid_convergence.png', dpi=150)
    print("\nSaved multigrid_grid_convergence.png")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    # Pivot selection
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_pivot_selection(ax, pos, [(int(e[0]), int(e[1])) for e in edges],
                         pivots, free_mask=free_mask, title='Pivot selection')
    plt.tight_layout()
    plt.savefig('multigrid_grid_pivots.png', dpi=150)
    print("Saved multigrid_grid_pivots.png")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    print(f"\n{'='*60}")
    print(f"{'Method':<25} {'Residual':<12}")
    print(f"{'='*60}")
    print(f"{'Plain Jacobi':<25} {res_jac[-1]:<12.4e}")
    print(f"{'Spectral MG (m=20)':<25} {res_mg_s[-1]:<12.4e}")
    print(f"{'Pivot MG (k=16)':<25} {res_mg_p[-1]:<12.4e}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Multigrid truss solver demo.")
    parser.add_argument('--mode', choices=['beam', 'grid'], default='beam',
                        help='Test mode: refined beam or 2D grid (default: beam)')
    parser.add_argument('--no-show', action='store_true', help='Do not call plt.show()')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    if args.mode == 'beam':
        main_beam(args)
    else:
        main_grid(args)


if __name__ == '__main__':
    main()
