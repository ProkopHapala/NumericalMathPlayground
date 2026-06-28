#!/usr/bin/env python3
"""
demo_GPU.py — Test and benchmark GPU kernels for truss elasticity solvers.

Tests:
1. Block Jacobi smoothing on GPU vs CPU
2. Multigrid V-cycle on GPU vs CPU
3. Convergence comparison: GPU block-Jacobi MG vs CPU Jacobi MG vs plain Jacobi
4. Timing benchmarks

Usage:
    python demo_GPU.py [--nx N] [--ny N] [--wg-size W] [--cluster-size C]
                       [--nmax-neigh K] [--n-v-cycles V] [--no-show]
"""

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from TrussSolver import (
    compute_edge_data, matvec_A, compute_diagonal_3x3, invert_3x3_blocks,
    build_adjacency, build_patches, setup_patch_data,
    solve_global_jacobi, assemble_dense_A,
)
from Truss import build_triangular_grid, grid_edges, boundary_nodes, apply_dirichlet
from MultiGrid import (
    build_spectral_prolongation, galerkin_coarse_operator,
    solve_multigrid, matvec_A_flat,
)
from SparseTruss_ocl import (
    SparseTrussOCL, edges_to_csr, build_cluster_map, prolongation_to_tiled,
)
from TrussPlotting import plot_multigrid_convergence, plot_truss


def main():
    parser = argparse.ArgumentParser(description="GPU truss elasticity benchmark")
    parser.add_argument('--nx', type=int, default=16, help="Grid nodes in x")
    parser.add_argument('--ny', type=int, default=16, help="Grid nodes in y")
    parser.add_argument('--wg-size', type=int, default=32, help="Workgroup size")
    parser.add_argument('--cluster-size', type=int, default=32, help="Max cluster size")
    parser.add_argument('--nmax-neigh', type=int, default=8, help="Max neighbors")
    parser.add_argument('--n-coarse', type=int, default=12, help="Number of coarse modes")
    parser.add_argument('--n-v-cycles', type=int, default=50, help="Max V-cycles")
    parser.add_argument('--n-jacobi', type=int, default=200, help="Max Jacobi iterations")
    parser.add_argument('--omega', type=float, default=0.8, help="Jacobi damping")
    parser.add_argument('--n-pre-smooth', type=int, default=3, help="Pre-smooth steps")
    parser.add_argument('--n-post-smooth', type=int, default=3, help="Post-smooth steps")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--no-show', action='store_true', help="Don't display plots")
    args = parser.parse_args()

    dim = 2  # 2D for grid demo
    nmax_neigh = args.nmax_neigh
    cluster_size = args.cluster_size
    wg_size = args.wg_size

    # --- Generate grid truss ---
    print(f"\n=== Generating {args.nx}x{args.ny} grid ===")
    pos = build_triangular_grid(args.nx, args.ny, a=1.0)[:, :dim]
    n_nodes = pos.shape[0]
    edges = grid_edges(args.nx, args.ny, include_diag=True)
    ei = np.array([e[0] for e in edges], dtype=np.int32)
    ej = np.array([e[1] for e in edges], dtype=np.int32)
    k_arr = np.ones(len(edges), dtype=np.float64) * 1000.0

    # Boundary: fix bottom row
    fixed = boundary_nodes(args.nx, args.ny, which='bottom')
    free_mask = np.ones(n_nodes, dtype=bool)
    free_mask[fixed] = False

    # Compute edge data
    n_dirs, k_eff = compute_edge_data(pos, ei, ej, k_arr, dim=dim)

    # Mass/dt^2 — moderate stiffness so displacement is visible
    dt = 0.05
    mass_dt2 = np.ones(n_nodes) / dt**2
    mass_dt2[fixed] *= 1e6  # pin fixed nodes

    # RHS: b = A*y + f_ext for free nodes, b = y for fixed nodes
    y = pos.copy()
    Ay = matvec_A(y, ei, ej, k_eff, n_dirs, mass_dt2)
    f_ext = np.zeros((n_nodes, dim))
    f_ext[:, 1] = -0.5  # downward force on free nodes
    b = Ay + f_ext
    b[fixed] = pos[fixed]  # Dirichlet: x[fixed] = rest position

    print(f"Nodes: {n_nodes}, Edges: {len(edges)}, Free: {free_mask.sum()}")

    # --- Direct solve (reference) ---
    print("\n--- Direct solve ---")
    A_dense = assemble_dense_A(ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim,
                               fixed_nodes=fixed)
    b_flat = b.ravel()
    t0 = time.perf_counter()
    x_direct = np.linalg.solve(A_dense, b_flat).reshape(n_nodes, dim)
    t_direct = time.perf_counter() - t0
    print(f"Direct solve: {t_direct*1000:.1f} ms, max disp = {np.max(np.linalg.norm(x_direct - pos, axis=1)):.6f}")

    # --- CPU plain Jacobi ---
    print("\n--- CPU plain Jacobi ---")
    x0 = pos.copy()
    t0 = time.perf_counter()
    x_jacobi, res_jacobi = solve_global_jacobi(
        b, x0, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim,
        free_mask, omega=args.omega, n_iter=args.n_jacobi
    )
    t_jacobi = time.perf_counter() - t0
    err_jacobi = np.linalg.norm(x_jacobi - x_direct) / np.linalg.norm(x_direct)
    print(f"Jacobi {len(res_jacobi)-1} iter: res={res_jacobi[-1]:.4e}, err={err_jacobi:.4e}, time={t_jacobi*1000:.1f} ms")

    # --- CPU spectral MG ---
    print("\n--- CPU spectral multigrid ---")
    P, eigvals = build_spectral_prolongation(
        ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim=dim,
        m=args.n_coarse, free_mask=free_mask
    )
    print(f"  P shape: ({n_nodes*dim}, {P.shape[1]}), lowest eigenvalues: {eigvals[:5]}")

    t0 = time.perf_counter()
    x_mg_cpu, res_mg_cpu, A_c = solve_multigrid(
        b, x0, P, ei, ej, k_eff, n_dirs, mass_dt2, n_nodes, dim,
        free_mask=free_mask, omega=args.omega,
        n_pre_smooth=args.n_pre_smooth, n_post_smooth=args.n_post_smooth,
        n_outer=args.n_v_cycles, tol=1e-8
    )
    t_mg_cpu = time.perf_counter() - t0
    err_mg_cpu = np.linalg.norm(x_mg_cpu - x_direct) / np.linalg.norm(x_direct)
    print(f"CPU MG {len(res_mg_cpu)-1} V-cycles: res={res_mg_cpu[-1]:.4e}, err={err_mg_cpu:.4e}, time={t_mg_cpu*1000:.1f} ms")

    # --- Build patches for GPU ---
    # Use non-overlapping patches (halo_radius=0) so each node is in exactly one patch.
    # This is required for correct restriction/prolongation with the tiled layout.
    print("\n--- Building patches for GPU ---")
    adj = build_adjacency(edges, n_nodes)
    patches = build_patches(adj, n_nodes, free_mask, target_core=cluster_size // 4,
                            halo_radius=0)
    # Filter patches to fit cluster_size
    patches = [p for p in patches if p['n_vertices'] <= cluster_size]
    # Verify full coverage of free nodes
    covered = set()
    for p in patches:
        covered.update(p['vertices'].tolist())
    free_set = set(np.where(free_mask)[0].tolist())
    missing = free_set - covered
    if missing:
        print(f"  WARNING: {len(missing)} free nodes not covered by patches!")
    print(f"  {len(patches)} patches, sizes: {[p['n_vertices'] for p in patches[:10]]}...")

    # Build CSR data
    neighs, k_vals_csr, n_dirs_csr, nneigh = edges_to_csr(
        ei, ej, k_eff, n_dirs, n_nodes, nmax_neigh=nmax_neigh, dim=dim
    )

    # Build cluster map
    patch_node_ids, patch_nneigh, patch_core_mask = build_cluster_map(
        patches, neighs, nneigh, n_nodes, nmax_neigh=nmax_neigh,
        cluster_size=cluster_size
    )

    # Convert P to tiled layout
    P_tiled = prolongation_to_tiled(
        P, patch_node_ids, cluster_size, dim, P.shape[1]
    )

    # --- GPU setup ---
    print("\n=== GPU initialization ===")
    ocl = SparseTrussOCL(
        wg_size=wg_size, cluster_size=cluster_size,
        nmax_neigh=nmax_neigh, dim=dim
    )

    ocl.upload_truss(neighs, k_vals_csr, n_dirs_csr, nneigh, mass_dt2)
    ocl.upload_patches(patch_node_ids, patch_nneigh, patch_core_mask)
    ocl.upload_prolongation(P_tiled, P.shape[1])
    ocl.upload_coarse_operator(A_c)

    # --- GPU block Jacobi test ---
    print("\n--- GPU block Jacobi (single step) ---")
    x0_gpu = pos.astype(np.float32).copy()
    t0 = time.perf_counter()
    x_gpu_jac = ocl.run_block_jacobi(x0_gpu, b.astype(np.float32), n_steps=1, omega=args.omega)
    t_gpu_jac = time.perf_counter() - t0
    print(f"  GPU block Jacobi 1 step: {t_gpu_jac*1000:.2f} ms")

    # --- GPU residual test ---
    print("\n--- GPU residual computation ---")
    t0 = time.perf_counter()
    r_gpu = ocl.compute_residual(x0_gpu, b.astype(np.float32))
    t_gpu_res = time.perf_counter() - t0
    r_cpu = b - matvec_A(x0, ei, ej, k_eff, n_dirs, mass_dt2)
    r_err = np.linalg.norm(r_gpu - r_cpu) / (np.linalg.norm(r_cpu) + 1e-30)
    print(f"  GPU residual: {t_gpu_res*1000:.2f} ms, error vs CPU: {r_err:.4e}")

    # --- GPU restriction test ---
    print("\n--- GPU restriction ---")
    t0 = time.perf_counter()
    r_c_gpu = ocl.restrict(r_gpu)
    t_gpu_rest = time.perf_counter() - t0
    r_c_cpu = P.T @ r_cpu.ravel()
    r_c_err = np.linalg.norm(r_c_gpu - r_c_cpu) / (np.linalg.norm(r_c_cpu) + 1e-30)
    print(f"  GPU restrict: {t_gpu_rest*1000:.2f} ms, error vs CPU: {r_c_err:.4e}")

    # --- GPU coarse solve test ---
    print("\n--- GPU coarse solve ---")
    t0 = time.perf_counter()
    e_c_gpu = ocl.coarse_solve(r_c_gpu)
    t_gpu_csolve = time.perf_counter() - t0
    from scipy.linalg import cho_factor, cho_solve
    L_c, low = cho_factor(A_c)
    e_c_cpu = cho_solve((L_c, low), r_c_cpu)
    e_c_err = np.linalg.norm(e_c_gpu - e_c_cpu) / (np.linalg.norm(e_c_cpu) + 1e-30)
    print(f"  GPU coarse solve: {t_gpu_csolve*1000:.2f} ms, error vs CPU: {e_c_err:.4e}")

    # --- GPU prolongation test ---
    print("\n--- GPU prolongation ---")
    x_test = x0_gpu.copy()
    t0 = time.perf_counter()
    ocl.prolongate(x=x_test, e_c=e_c_gpu, in_place=True)
    x_prolonged = ocl.download_x()
    t_gpu_prol = time.perf_counter() - t0
    x_cpu_prol = x0 + (P @ e_c_cpu).reshape(n_nodes, dim)
    prol_err = np.linalg.norm(x_prolonged - x_cpu_prol.astype(np.float32)) / (
        np.linalg.norm(x_cpu_prol) + 1e-30)
    print(f"  GPU prolongate: {t_gpu_prol*1000:.2f} ms, error vs CPU: {prol_err:.4e}")

    # --- GPU full V-cycle solve ---
    print("\n--- GPU multigrid V-cycle solve ---")
    t0 = time.perf_counter()
    x_gpu_mg, res_gpu_mg = ocl.solve(
        b.astype(np.float32), x0=pos.astype(np.float32),
        n_v_cycles=args.n_v_cycles,
        n_pre_smooth=args.n_pre_smooth, n_post_smooth=args.n_post_smooth,
        omega=args.omega, tol=1e-6, free_mask=free_mask, fixed_nodes=fixed
    )
    t_gpu_mg = time.perf_counter() - t0
    err_gpu_mg = np.linalg.norm(x_gpu_mg - x_direct) / np.linalg.norm(x_direct)
    print(f"GPU MG {len(res_gpu_mg)-1} V-cycles: res={res_gpu_mg[-1]:.4e}, err={err_gpu_mg:.4e}, time={t_gpu_mg*1000:.1f} ms")

    # --- Summary table ---
    print("\n" + "=" * 70)
    print(f"{'Method':<30} {'Iters':>8} {'Residual':>12} {'Error':>12} {'Time(ms)':>10}")
    print("=" * 70)
    print(f"{'Direct solve':<30} {'1':>8} {'0':>12} {'0':>12} {t_direct*1000:>10.1f}")
    print(f"{'CPU plain Jacobi':<30} {len(res_jacobi)-1:>8} {res_jacobi[-1]:>12.4e} {err_jacobi:>12.4e} {t_jacobi*1000:>10.1f}")
    print(f"{'CPU spectral MG':<30} {len(res_mg_cpu)-1:>8} {res_mg_cpu[-1]:>12.4e} {err_mg_cpu:>12.4e} {t_mg_cpu*1000:>10.1f}")
    print(f"{'GPU spectral MG':<30} {len(res_gpu_mg)-1:>8} {res_gpu_mg[-1]:>12.4e} {err_gpu_mg:>12.4e} {t_gpu_mg*1000:>10.1f}")
    print("=" * 70)

    # --- Timing breakdown ---
    print(f"\nGPU timing breakdown (per operation):")
    print(f"  Block Jacobi 1 step: {t_gpu_jac*1000:.2f} ms")
    print(f"  Residual:            {t_gpu_res*1000:.2f} ms")
    print(f"  Restrict:            {t_gpu_rest*1000:.2f} ms")
    print(f"  Coarse solve:        {t_gpu_csolve*1000:.2f} ms")
    print(f"  Prolongate:          {t_gpu_prol*1000:.2f} ms")

    # --- Plot convergence ---
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = [
        (res_jacobi, 'CPU Jacobi', 'b-'),
        (res_mg_cpu, 'CPU spectral MG', 'r-o'),
        (res_gpu_mg, 'GPU spectral MG', 'g-s'),
    ]
    for res, label, style in methods:
        marker = style[2] if len(style) > 2 else None
        color = style[0]
        ls = style[1]
        ax.semilogy(range(len(res)), res, ls, color=color, label=label,
                    marker=marker, markersize=4, markevery=max(1, len(res)//20))
    ax.set_xlabel('Iteration / V-cycle')
    ax.set_ylabel('Relative residual')
    ax.set_title(f'Convergence comparison ({args.nx}x{args.ny} grid, {n_nodes} nodes)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=1e-10)
    plt.tight_layout()
    plt.savefig('gpu_convergence.png', dpi=150)
    print("\nSaved gpu_convergence.png")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    # --- Plot deformation ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    disp_scale = 5.0
    for ax, x_sol, title in zip(axes,
        [x_direct, x_mg_cpu, x_gpu_mg],
        ['Direct', 'CPU MG', 'GPU MG']):
        deformed = pos + disp_scale * (x_sol - pos)
        ax.scatter(deformed[:, 0], deformed[:, 1], c='blue', s=10, zorder=2)
        ax.scatter(pos[:, 0], pos[:, 1], c='gray', s=5, alpha=0.3, zorder=1)
        for i, j in edges:
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    'k-', alpha=0.1, lw=0.5)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xlim(pos[:, 0].min() - 1, pos[:, 0].max() + 1)
        ax.set_ylim(pos[:, 1].min() - 1, pos[:, 1].max() + 1)
    plt.suptitle(f'Deformation comparison (scale={disp_scale}x)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig('gpu_deformation.png', dpi=150)
    print("Saved gpu_deformation.png")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
