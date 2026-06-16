"""
Test nested dissection + block diagonalization on nanocrystal benchmarks.

Runs:
  1. Exact numpy eigh (reference)
  2. Geometric clustering (RCB) -> block-diagonal approximation
  3. Nested dissection -> bordered block-diagonal
  4. Block diagonalization via Python Jacobi and OpenCL batched Jacobi
  5. Spectrum comparison plots

Usage:
    python test_nested_solver.py --system nc_C_R5 --n_clusters 8
    python test_nested_solver.py --system nc_C_R4 --gpu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# ensure imports from same directory work when run directly
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nested_solver import (
    load_system_dense,
    rcb_cluster_atoms,
    reorder_by_clusters,
    extract_diagonal_blocks,
    build_block_diagonal_approximation,
    diagonalize_blocks_python,
    diagonalize_blocks_jacobi,
    nested_dissection_reorder,
    rcm_reorder,
    ritz_correction_from_blocks,
    recursive_exact_amls,
    static_condensation_spectrum,
    OpenCLBlockJacobi,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="nc_C_R5",
                   choices=["adamantane", "nc_C_R4", "nc_C_R5", "nc_C_R6", "nc_C_R7", "nc_C_R8"])
    p.add_argument("--n_clusters", type=int, default=8,
                   help="Number of geometric clusters (power of 2)")
    p.add_argument("--max_leaf_atoms", type=int, default=8,
                   help="Max atoms per leaf in nested dissection")
    p.add_argument("--gpu", action="store_true", help="Run OpenCL block-Jacobi")
    p.add_argument("--save", default="nested_solver_plots.png")
    p.add_argument("--fixtures_dir", default=None)
    p.add_argument("--rigid_threshold", type=float, default=100.0,
                   help="Discard modes above this omega as shifted rigid modes")
    return p.parse_args()


def _cumulative_boundaries(block_sizes):
    """Return list of boundary indices from block sizes."""
    bounds = np.cumsum(block_sizes)
    return bounds[:-1]  # exclude final total size


def _draw_split_lines(ax, boundaries, color="red", lw=0.8, alpha=0.9):
    for b in boundaries:
        ax.axvline(b - 0.5, color=color, lw=lw, alpha=alpha)
        ax.axhline(b - 0.5, color=color, lw=lw, alpha=alpha)


def plot_matrix_log(H, block_sizes, title, save_path, cmap="viridis"):
    """Plot log10(abs(H_ij)) with block-split lines. Separate figure."""
    fig, ax = plt.subplots(figsize=(7, 6))
    logH = np.log10(np.abs(H) + 1e-12)
    vmax = np.percentile(logH[logH > -12], 99)
    vmin = max(logH.min(), -12)
    im = ax.imshow(logH, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    if block_sizes is not None and len(block_sizes) > 1:
        _draw_split_lines(ax, _cumulative_boundaries(block_sizes))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("DOF")
    ax.set_ylabel("DOF")
    fig.colorbar(im, ax=ax, label="log10(|A_ij|)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved matrix plot to {save_path}")


def plot_scatter_comparison(exact_vals, method_vals, method_name, save_path,
                            unit="omega"):
    """Scatter: exact eigenvalues (x) vs method eigenvalues (y). Separate figure."""
    fig, ax = plt.subplots(figsize=(6, 6))
    exact_sorted = np.sort(exact_vals)
    n_exact = len(exact_sorted)
    n_method = len(method_vals)
    if n_method >= n_exact:
        method_sorted = np.sort(method_vals)[:n_exact]
    else:
        method_sorted = np.sort(method_vals)
        exact_sorted = exact_sorted[:n_method]
    ax.scatter(exact_sorted, method_sorted, s=8, alpha=0.6, edgecolors="none", color="steelblue")
    # y=x diagonal
    lo = min(exact_sorted.min(), method_sorted.min())
    hi = max(exact_sorted.max(), method_sorted.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x")
    ax.set_xlabel(f"Exact {unit}")
    ax.set_ylabel(f"{method_name} {unit}")
    ax.set_title(f"{method_name} vs Exact ({len(method_sorted)} points)")
    ax.legend()
    # error stats in corner
    err = np.abs(method_sorted - exact_sorted)
    ax.text(0.05, 0.95, f"max err = {err.max():.3f}\nmean err = {err.mean():.3f}",
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved scatter plot to {save_path}")


def _gather_tree_boundaries(node, level=0, boundaries_by_level=None):
    """Gather cumulative boundaries at each tree level."""
    if boundaries_by_level is None:
        boundaries_by_level = {}
    if node["type"] == "leaf":
        return boundaries_by_level
    # For a separator node, boundaries are between children and after children (before separator)
    # We need the actual DOF layout from tree_to_dofs to know exact positions.
    # Instead, we'll use block_sizes which already reflect the final ordering.
    return boundaries_by_level


def plot_amls_steps(H_nd, tree, block_sizes, save_path):
    """
    Multi-panel figure illustrating nested dissection / AMLS steps:
    - original reordered matrix with all block boundaries
    - matrix with only top-level split boundaries
    - block-diagonal approximation
    """
    from nested_solver import _annotate_subtree_ranges
    tree = tree.copy()
    _annotate_subtree_ranges(tree)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    logH = np.log10(np.abs(H_nd) + 1e-12)
    vmax = np.percentile(logH[logH > -12], 99)
    vmin = max(logH.min(), -12)

    def im(ax, title):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("DOF")
        ax.set_ylabel("DOF")

    # Panel 1: full reordered matrix with all leaf/separator boundaries
    ax = axes[0]
    ax.imshow(logH, aspect="equal", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    _draw_split_lines(ax, _cumulative_boundaries(block_sizes), color="red", lw=0.8)
    im(ax, "ND reordered H (all block boundaries)")

    # Panel 2: top-level split boundaries only (recursion depth 0 and 1)
    ax = axes[1]
    ax.imshow(logH, aspect="equal", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    # Top level: root separator splits left and right subtrees
    # We can infer top-level boundaries from tree structure
    def _top_level_bounds(node):
        bounds = []
        if node["type"] == "separator" and node.get("children"):
            child_end = node["children"][0]["subtree_end"]
            bounds.append(child_end)
        return bounds
    top_bounds = _top_level_bounds(tree)
    _draw_split_lines(ax, top_bounds, color="yellow", lw=2.0)
    im(ax, "Top-level split (yellow) + all (red faint)")
    _draw_split_lines(ax, _cumulative_boundaries(block_sizes), color="red", lw=0.3, alpha=0.3)

    # Panel 3: block diagonal approximation
    ax = axes[2]
    H_bd = H_nd.copy()
    idx = 0
    for bs in block_sizes:
        H_bd[idx:idx + bs, idx + bs:] = 0.0
        H_bd[idx + bs:, idx:idx + bs] = 0.0
        idx += bs
    log_bd = np.log10(np.abs(H_bd) + 1e-12)
    ax.imshow(log_bd, aspect="equal", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    _draw_split_lines(ax, _cumulative_boundaries(block_sizes), color="red", lw=0.8)
    im(ax, "Block-diagonal approximation")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved AMLS steps plot to {save_path}")


def _tree_max_depth(node):
    if node["type"] == "leaf":
        return 0
    return 1 + max(_tree_max_depth(ch) for ch in node.get("children", []))


def _collect_separators_at_depth(node, depth, out):
    if depth == 0:
        if node["type"] == "separator":
            out.append(node)
        return
    if node["type"] == "separator":
        for ch in node.get("children", []):
            _collect_separators_at_depth(ch, depth - 1, out)


def _condense_one_separator(H, node):
    si, sj = int(node["dof_start"]), int(node["dof_end"])
    if sj <= si:
        return
    for child in node.get("children", []):
        ci, cj = int(child["subtree_start"]), int(child["subtree_end"])
        if cj <= ci:
            continue
        B = H[ci:cj, si:sj]
        if np.max(np.abs(B)) < 1e-16:
            H[ci:cj, si:sj] = 0.0
            H[si:sj, ci:cj] = 0.0
            continue
        A = H[ci:cj, ci:cj]
        X = np.linalg.solve(A, B)
        H[si:sj, si:sj] -= B.T @ X
        H[ci:cj, si:sj] = 0.0
        H[si:sj, ci:cj] = 0.0


def plot_condensation_sequence(H_nd, tree_nd, block_sizes_nd, save_prefix):
    import nested_solver as ns
    tree = tree_nd.copy()
    ns._annotate_subtree_ranges(tree)
    max_d = _tree_max_depth(tree)
    H = H_nd.copy()
    for depth in range(max_d, -1, -1):
        nodes = []
        _collect_separators_at_depth(tree, depth, nodes)
        for node in nodes:
            _condense_one_separator(H, node)
        plot_matrix_log(H, block_sizes_nd, f"ND/AMLS condensation (depth >= {depth})",
                        f"{save_prefix}_condense_d{depth}.png")


def _plot_exact_methods_overlay(exact_vals, amls_f64, ritz_rcb_f64, ritz_nd_f64,
                                amls_f32, ritz_rcb_f32, ritz_nd_f32, save_path):
    """Overlay scatter: all 3 exact methods (f64 + f32) on one figure."""
    fig, ax = plt.subplots(figsize=(7, 7))
    exact_sorted = np.sort(exact_vals)
    n = len(exact_sorted)

    def _prep(vals):
        v = np.sort(vals)
        return v[:n] if len(v) >= n else np.concatenate([v, np.full(n - len(v), v.max())])

    data = [
        ("AMLS f64", _prep(amls_f64), "C0", "o"),
        ("RCB Ritz f64", _prep(ritz_rcb_f64), "C1", "s"),
        ("ND Ritz f64", _prep(ritz_nd_f64), "C2", "^"),
        ("AMLS f32", _prep(amls_f32), "C0", "."),
        ("RCB Ritz f32", _prep(ritz_rcb_f32), "C1", "."),
        ("ND Ritz f32", _prep(ritz_nd_f32), "C2", "."),
    ]
    for label, v, color, marker in data:
        err = np.abs(v - exact_sorted)
        ax.scatter(exact_sorted, err, s=10 if marker == "." else 20,
                   alpha=0.5, color=color, marker=marker, label=label, edgecolors="none")

    ax.set_yscale("log")
    ax.set_xlabel("Exact omega")
    ax.set_ylabel("Absolute error |omega - omega_exact|")
    ax.set_title("Exact methods: eigenvalue error vs reference (f64 + f32)")
    ax.legend(loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved overlay plot to {save_path}")


def _filter_vib(omegas, threshold):
    """Discard modes above threshold (shifted rigid modes)."""
    return omegas[omegas < threshold]


def run(args):
    print(f"=== Loading {args.system} ===")
    sys_data = load_system_dense(args.system, fixtures_dir=args.fixtures_dir)
    H = sys_data["H"]
    pos = sys_data["pos"]
    natoms = sys_data["natoms"]
    ndof = sys_data["ndof"]
    print(f"System: {sys_data['name']}, atoms={natoms}, DOF={ndof}")

    th = args.rigid_threshold
    print(f"Rigid-mode threshold: omega < {th} (only vibration modes plotted/compared)")

    # ------------------------------------------------------------------
    # 1) Brute-force exact spectrum
    # ------------------------------------------------------------------
    print("\n--- Exact numpy eigh ---")
    t0 = time.time()
    eigs_exact, _ = np.linalg.eigh(H)
    t_exact = time.time() - t0
    omegas_exact = np.sqrt(np.clip(eigs_exact, 0, None))
    omegas_exact = _filter_vib(omegas_exact, th)
    print(f"Exact: {len(eigs_exact)} eigenvalues in {t_exact:.3f}s")
    print(f"  Full omega range: {np.sqrt(np.clip(eigs_exact,0,None)).min():.4f} .. {np.sqrt(np.clip(eigs_exact,0,None)).max():.4f}")
    print(f"  Vibration modes (<{th}): {len(omegas_exact)}  range: {omegas_exact.min():.4f} .. {omegas_exact.max():.4f}")

    # ------------------------------------------------------------------
    # 2) RCB clustering and block diagonalization
    # ------------------------------------------------------------------
    print(f"\n--- RCB clustering into {args.n_clusters} clusters ---")
    atom_to_cluster, centers = rcb_cluster_atoms(pos, args.n_clusters)
    H_rcb, perm_rcb, block_sizes_rcb = reorder_by_clusters(H, atom_to_cluster)
    H_bd_rcb = build_block_diagonal_approximation(H_rcb, block_sizes_rcb)

    blocks_rcb = extract_diagonal_blocks(H_rcb, block_sizes_rcb)
    print(f"RCB block sizes (DOF): {block_sizes_rcb}")
    print(f"RCB total DOF: {block_sizes_rcb.sum()} (expected {ndof})")

    t0 = time.time()
    eigs_rcb, _ = diagonalize_blocks_python(blocks_rcb)
    t_rcb = time.time() - t0
    omegas_rcb = np.sqrt(np.clip(eigs_rcb, 0, None))
    omegas_rcb = _filter_vib(omegas_rcb, th)
    print(f"RCB block-diag: {len(eigs_rcb)} eigenvalues in {t_rcb:.3f}s")

    # ------------------------------------------------------------------
    # 3) Nested dissection reordering and block diagonalization
    # ------------------------------------------------------------------
    print(f"\n--- Nested dissection (max_leaf_atoms={args.max_leaf_atoms}) ---")
    H_nd, perm_nd, block_sizes_nd, tree_nd = nested_dissection_reorder(H, pos, args.max_leaf_atoms)
    H_bd_nd = build_block_diagonal_approximation(H_nd, block_sizes_nd)

    blocks_nd = extract_diagonal_blocks(H_nd, block_sizes_nd)
    print(f"ND block sizes (DOF): {block_sizes_nd}")
    print(f"ND total DOF: {block_sizes_nd.sum()} (expected {ndof})")

    t0 = time.time()
    eigs_nd, _ = diagonalize_blocks_python(blocks_nd)
    t_nd = time.time() - t0
    omegas_nd = np.sqrt(np.clip(eigs_nd, 0, None))
    omegas_nd = _filter_vib(omegas_nd, th)
    print(f"ND block-diag: {len(eigs_nd)} eigenvalues in {t_nd:.3f}s")

    # ------------------------------------------------------------------
    # 4) Python Jacobi reference on one block (skip if too large)
    # ------------------------------------------------------------------
    print("\n--- Python Jacobi rotation test (largest block) ---")
    largest_block_idx = int(np.argmax(block_sizes_nd))
    B_big = blocks_nd[largest_block_idx]
    print(f"Largest ND block size: {B_big.shape}")
    if B_big.shape[0] > 100:
        print(f"Skipping Python Jacobi for block > 100x100 (would take too long)")
    else:
        t0 = time.time()
        from nested_solver import jacobi_rotation_block
        eigs_jac, vecs_jac, nsweeps = jacobi_rotation_block(B_big, tol=1e-10, max_sweeps=100)
        t_jac = time.time() - t0
        w_ref, _ = np.linalg.eigh(B_big)
        err_jac = np.max(np.abs(np.sort(eigs_jac) - np.sort(w_ref)))
        print(f"Jacobi on largest block: {t_jac:.4f}s, sweeps={nsweeps}, max eig error={err_jac:.2e}")

    # ------------------------------------------------------------------
    # 5) OpenCL batched block Jacobi (optional)
    # ------------------------------------------------------------------
    if args.gpu:
        print("\n--- OpenCL batched block Jacobi ---")
        try:
            cl_solver = OpenCLBlockJacobi()
            size_counts = {}
            for bs in block_sizes_nd:
                size_counts.setdefault(bs, 0)
                size_counts[bs] += 1
            print(f"ND block size distribution: {size_counts}")

            # (a) Uniform-size batch (old kernel)
            most_common_size = max(size_counts, key=lambda k: size_counts[k])
            same_size_blocks = [b for b in blocks_nd if b.shape[0] == most_common_size]
            print(f"Running GPU Jacobi (uniform) on {len(same_size_blocks)} blocks of size {most_common_size}x{most_common_size}")
            t0 = time.time()
            cl_eigvals, cl_eigvecs = cl_solver.diagonalize(same_size_blocks, tol=1e-6, max_sweeps=50)
            t_cl = time.time() - t0
            ref_vals = [np.linalg.eigh(b)[0] for b in same_size_blocks]
            max_rel_err = 0.0
            for i in range(len(same_size_blocks)):
                ref = np.sort(ref_vals[i])
                gpu = np.sort(cl_eigvals[i])
                abs_err = np.max(np.abs(gpu - ref))
                rel_err = abs_err / max(1.0, np.abs(ref).max())
                max_rel_err = max(max_rel_err, rel_err)
            print(f"GPU uniform: {t_cl:.4f}s for {len(same_size_blocks)} blocks, max rel eig error={max_rel_err:.2e}")

            # (b) Padded batch on ALL blocks with varying sizes
            print(f"\nRunning GPU Jacobi (padded) on ALL {len(blocks_nd)} blocks")
            t0 = time.time()
            cl_eigvals_pad, cl_eigvecs_pad = cl_solver.diagonalize_padded(blocks_nd, tol=1e-6, max_sweeps=50)
            t_cl_pad = time.time() - t0
            max_rel_err_pad = 0.0
            for i, B in enumerate(blocks_nd):
                ref = np.sort(np.linalg.eigh(B)[0])
                gpu = np.sort(cl_eigvals_pad[i])
                abs_err = np.max(np.abs(gpu - ref))
                rel_err = abs_err / max(1.0, np.abs(ref).max())
                max_rel_err_pad = max(max_rel_err_pad, rel_err)
            print(f"GPU padded: {t_cl_pad:.4f}s for {len(blocks_nd)} blocks, max rel eig error={max_rel_err_pad:.2e}")

            # (c) GPU-accelerated truncated Ritz (similarity transform on GPU)
            print("\n--- GPU-accelerated truncated Ritz (similarity transform) ---")
            H_rcb_gpu = H_rcb.astype(np.float32)
            blocks_rcb_gpu = [b.astype(np.float32) for b in blocks_rcb]
            H_nd_gpu = H_nd.astype(np.float32)
            blocks_nd_gpu = [b.astype(np.float32) for b in blocks_nd]
            for n_keep in [8, 16]:
                m_total_rcb = sum(min(n_keep, bs) for bs in block_sizes_rcb)
                m_total_nd = sum(min(n_keep, bs) for bs in block_sizes_nd)
                print(f"n_keep={n_keep}: RCB m={m_total_rcb}, ND m={m_total_nd}")

                # CPU truncated Ritz
                t0 = time.time()
                eigs_tr_rcb_cpu, _, _ = ritz_correction_from_blocks(H_rcb_gpu, blocks_rcb_gpu, n_modes_per_block=n_keep)
                t_cpu = time.time() - t0

                # GPU truncated Ritz
                t0 = time.time()
                eigs_tr_rcb_gpu, _, _ = ritz_correction_from_blocks(H_rcb_gpu, blocks_rcb_gpu, n_modes_per_block=n_keep, gpu_solver=cl_solver)
                t_gpu = time.time() - t0

                err = np.max(np.abs(eigs_tr_rcb_gpu - eigs_tr_rcb_cpu))
                rel = err / (np.max(np.abs(eigs_tr_rcb_cpu)) + 1e-12)
                print(f"  RCB truncated ({n_keep}/block): CPU {t_cpu*1000:.2f}ms, GPU {t_gpu*1000:.2f}ms, rel_err={rel:.2e}")

                # ND truncated Ritz
                t0 = time.time()
                eigs_tr_nd_cpu, _, _ = ritz_correction_from_blocks(H_nd_gpu, blocks_nd_gpu, n_modes_per_block=n_keep)
                t_cpu = time.time() - t0

                t0 = time.time()
                eigs_tr_nd_gpu, _, _ = ritz_correction_from_blocks(H_nd_gpu, blocks_nd_gpu, n_modes_per_block=n_keep, gpu_solver=cl_solver)
                t_gpu = time.time() - t0

                err = np.max(np.abs(eigs_tr_nd_gpu - eigs_tr_nd_cpu))
                rel = err / (np.max(np.abs(eigs_tr_nd_cpu)) + 1e-12)
                print(f"  ND truncated ({n_keep}/block):  CPU {t_cpu*1000:.2f}ms, GPU {t_gpu*1000:.2f}ms, rel_err={rel:.2e}")

        except Exception as e:
            print(f"OpenCL failed: {e}")

    # ------------------------------------------------------------------
    # 4b) RCM reordering to show narrow bandwidth
    # ------------------------------------------------------------------
    print("\n--- RCM reordering ---")
    H_rcm, perm_rcm, bw_rcm = rcm_reorder(H)
    print(f"RCM bandwidth: {bw_rcm} (DOF={ndof})")

    # ------------------------------------------------------------------
    # 5) Recursive exact AMLS  (returns eigenvectors too)
    # ------------------------------------------------------------------
    print("\n--- Recursive exact AMLS ---")
    t0 = time.time()
    eigs_amls, vecs_amls = recursive_exact_amls(H_nd, tree_nd)
    t_amls = time.time() - t0
    omegas_amls = np.sqrt(np.clip(eigs_amls, 0, None))
    omegas_amls = _filter_vib(omegas_amls, th)
    print(f"AMLS exact: {len(eigs_amls)} eigenvalues in {t_amls:.3f}s")

    # ------------------------------------------------------------------
    # 6) Static condensation (Guyan reduction)
    # ------------------------------------------------------------------
    print("\n--- Static condensation (Guyan) ---")
    t0 = time.time()
    eigs_guyan = static_condensation_spectrum(H_nd, block_sizes_nd)
    t_guyan = time.time() - t0
    omegas_guyan = np.sqrt(np.clip(eigs_guyan, 0, None))
    omegas_guyan = _filter_vib(omegas_guyan, th)
    print(f"Static condensation: {len(eigs_guyan)} eigenvalues (separator only) in {t_guyan:.3f}s")

    # ------------------------------------------------------------------
    # 7) Ritz projection correction (two-level)  — exact when keeping all modes
    # ------------------------------------------------------------------
    print("\n--- Ritz projection correction ---")
    t0 = time.time()
    eigs_ritz_rcb, vecs_ritz_rcb, _ = ritz_correction_from_blocks(H_rcb, blocks_rcb)
    t_ritz_rcb = time.time() - t0
    omegas_ritz_rcb = np.sqrt(np.clip(eigs_ritz_rcb, 0, None))
    omegas_ritz_rcb = _filter_vib(omegas_ritz_rcb, th)
    print(f"RCB Ritz correction (all modes): {len(eigs_ritz_rcb)} eigenvalues in {t_ritz_rcb:.3f}s")

    t0 = time.time()
    eigs_ritz_nd, vecs_ritz_nd, _ = ritz_correction_from_blocks(H_nd, blocks_nd)
    t_ritz_nd = time.time() - t0
    omegas_ritz_nd = np.sqrt(np.clip(eigs_ritz_nd, 0, None))
    omegas_ritz_nd = _filter_vib(omegas_ritz_nd, th)
    print(f"ND Ritz correction (all modes): {len(eigs_ritz_nd)} eigenvalues in {t_ritz_nd:.3f}s")

    # Truncated Ritz (reduced model) - keep only a few modes per block
    omegas_tr_rcb_8 = None
    omegas_tr_nd_8 = None
    for n_keep in [4, 8]:
        print(f"\n--- Truncated Ritz ({n_keep} modes/block) ---")
        t0 = time.time()
        eigs_tr_rcb, _, _ = ritz_correction_from_blocks(H_rcb, blocks_rcb, n_modes_per_block=n_keep)
        t_tr = time.time() - t0
        omegas_tr_rcb = np.sqrt(np.clip(eigs_tr_rcb, 0, None))
        omegas_tr_rcb = _filter_vib(omegas_tr_rcb, th)
        print(f"RCB truncated ({n_keep}/block): {len(eigs_tr_rcb)} eigenvalues in {t_tr:.3f}s")

        t0 = time.time()
        eigs_tr_nd, _, _ = ritz_correction_from_blocks(H_nd, blocks_nd, n_modes_per_block=n_keep)
        t_tr = time.time() - t0
        omegas_tr_nd = np.sqrt(np.clip(eigs_tr_nd, 0, None))
        omegas_tr_nd = _filter_vib(omegas_tr_nd, th)
        print(f"ND truncated ({n_keep}/block): {len(eigs_tr_nd)} eigenvalues in {t_tr:.3f}s")

        if n_keep == 8:
            omegas_tr_rcb_8 = omegas_tr_rcb
            omegas_tr_nd_8 = omegas_tr_nd

    # ------------------------------------------------------------------
    # 8) Spectrum error analysis
    # ------------------------------------------------------------------
    print("\n--- Spectrum error analysis ---")
    exact_sorted = np.sort(omegas_exact)

    def report_error(name, vals):
        v_sorted = np.sort(vals)
        n = len(exact_sorted)
        if len(v_sorted) < n:
            v_sorted = np.concatenate([v_sorted, np.full(n - len(v_sorted), v_sorted.max())])
        else:
            v_sorted = v_sorted[:n]
        err = np.abs(exact_sorted - v_sorted)
        print(f"{name:20s}: mean={err.mean():.2e}, max={err.max():.2e}, rmse={np.sqrt((err**2).mean()):.2e}")

    report_error("RCB block-diag", omegas_rcb)
    report_error("ND block-diag", omegas_nd)
    report_error("AMLS exact", omegas_amls)
    report_error("Guyan (sep only)", omegas_guyan)
    report_error("RCB Ritz (exact)", omegas_ritz_rcb)
    report_error("ND Ritz (exact)", omegas_ritz_nd)
    if omegas_tr_rcb_8 is not None:
        report_error("RCB trunc 8/blk", omegas_tr_rcb_8)
    if omegas_tr_nd_8 is not None:
        report_error("ND trunc 8/blk", omegas_tr_nd_8)

    # ------------------------------------------------------------------
    # 9) Float32 exact methods (single precision, CPU)
    # ------------------------------------------------------------------
    print("\n--- Float32 exact methods ---")
    H_nd_f32 = H_nd.astype(np.float32)
    H_rcb_f32 = H_rcb.astype(np.float32)
    blocks_rcb_f32 = [b.astype(np.float32) for b in blocks_rcb]
    blocks_nd_f32 = [b.astype(np.float32) for b in blocks_nd]

    t0 = time.time()
    eigs_amls_f32, vecs_amls_f32 = recursive_exact_amls(H_nd_f32, tree_nd)
    t_amls_f32 = time.time() - t0
    omegas_amls_f32 = np.sqrt(np.clip(eigs_amls_f32, 0, None))
    omegas_amls_f32 = _filter_vib(omegas_amls_f32, th)

    t0 = time.time()
    eigs_ritz_rcb_f32, vecs_ritz_rcb_f32, _ = ritz_correction_from_blocks(H_rcb_f32, blocks_rcb_f32)
    t_ritz_rcb_f32 = time.time() - t0
    omegas_ritz_rcb_f32 = np.sqrt(np.clip(eigs_ritz_rcb_f32, 0, None))
    omegas_ritz_rcb_f32 = _filter_vib(omegas_ritz_rcb_f32, th)

    t0 = time.time()
    eigs_ritz_nd_f32, vecs_ritz_nd_f32, _ = ritz_correction_from_blocks(H_nd_f32, blocks_nd_f32)
    t_ritz_nd_f32 = time.time() - t0
    omegas_ritz_nd_f32 = np.sqrt(np.clip(eigs_ritz_nd_f32, 0, None))
    omegas_ritz_nd_f32 = _filter_vib(omegas_ritz_nd_f32, th)

    print(f"AMLS exact float32 : {len(eigs_amls_f32)} eigvals in {t_amls_f32:.3f}s")
    print(f"RCB Ritz float32   : {len(eigs_ritz_rcb_f32)} eigvals in {t_ritz_rcb_f32:.3f}s")
    print(f"ND Ritz float32    : {len(eigs_ritz_nd_f32)} eigvals in {t_ritz_nd_f32:.3f}s")

    exact_sorted_f64 = np.sort(omegas_exact)
    def report_error_f32(name, vals):
        v_sorted = np.sort(vals)
        n = len(exact_sorted_f64)
        if len(v_sorted) < n:
            v_sorted = np.concatenate([v_sorted, np.full(n - len(v_sorted), v_sorted.max())])
        else:
            v_sorted = v_sorted[:n]
        err = np.abs(exact_sorted_f64 - v_sorted)
        print(f"  {name:18s}: mean={err.mean():.2e}, max={err.max():.2e}, rmse={np.sqrt((err**2).mean()):.2e}")

    report_error_f32("AMLS f32", omegas_amls_f32)
    report_error_f32("RCB Ritz f32", omegas_ritz_rcb_f32)
    report_error_f32("ND Ritz f32", omegas_ritz_nd_f32)

    # ------------------------------------------------------------------
    # 10) Eigenvector correctness check (exact methods)
    # ------------------------------------------------------------------
    def _eigvec_residual(Hmat, w, V):
        """Max relative residual ||H v - λ v|| / |λ| over all modes."""
        residuals = np.linalg.norm(Hmat @ V - V @ np.diag(w), axis=0)
        denom = np.abs(w) + 1e-12
        rel = residuals / denom
        return rel.max(), rel.mean()

    print("\n--- Eigenvector residual check (exact methods, float64) ---")
    rmax, rmean = _eigvec_residual(H_nd, eigs_amls, vecs_amls)
    print(f"AMLS exact      : max_rel_resid={rmax:.2e}, mean={rmean:.2e}")
    rmax, rmean = _eigvec_residual(H_rcb, eigs_ritz_rcb, vecs_ritz_rcb)
    print(f"RCB Ritz exact  : max_rel_resid={rmax:.2e}, mean={rmean:.2e}")
    rmax, rmean = _eigvec_residual(H_nd, eigs_ritz_nd, vecs_ritz_nd)
    print(f"ND Ritz exact   : max_rel_resid={rmax:.2e}, mean={rmean:.2e}")

    print("\n--- Eigenvector residual check (exact methods, float32) ---")
    rmax, rmean = _eigvec_residual(H_nd_f32, eigs_amls_f32, vecs_amls_f32)
    print(f"AMLS f32        : max_rel_resid={rmax:.2e}, mean={rmean:.2e}")
    rmax, rmean = _eigvec_residual(H_rcb_f32, eigs_ritz_rcb_f32, vecs_ritz_rcb_f32)
    print(f"RCB Ritz f32    : max_rel_resid={rmax:.2e}, mean={rmean:.2e}")
    rmax, rmean = _eigvec_residual(H_nd_f32, eigs_ritz_nd_f32, vecs_ritz_nd_f32)
    print(f"ND Ritz f32     : max_rel_resid={rmax:.2e}, mean={rmean:.2e}")

    # ------------------------------------------------------------------
    # 10) Plots — organized into separate folders
    # ------------------------------------------------------------------
    save_path = Path(args.save)
    save_dir = save_path.parent if save_path.parent.name else Path(".")
    base = save_path.stem

    # Organized folders
    dir_matrix = save_dir / "matrix_structure"
    dir_exact = save_dir / "exact_methods"
    dir_approx = save_dir / "approx_methods"
    for d in (dir_matrix, dir_exact, dir_approx):
        d.mkdir(parents=True, exist_ok=True)

    # Matrix structure plots
    plot_matrix_log(H, None, f"Original H  ({args.system})",
                    str(dir_matrix / f"{base}_00_input_original.png"))
    plot_matrix_log(H_rcb, block_sizes_rcb,
                    f"RCB reordered input ({args.n_clusters} clusters) -> RCB Ritz exact",
                    str(dir_matrix / f"{base}_01_RCB_Ritz_input.png"))
    plot_amls_steps(H_nd, tree_nd, block_sizes_nd,
                    str(dir_matrix / f"{base}_02_ND_AMLS_steps.png"))
    plot_condensation_sequence(H_nd, tree_nd, block_sizes_nd,
                              str(dir_matrix / f"{base}_03_ND_AMLS"))

    # Exact methods scatter plots (float64)
    exact_methods = [
        ("AMLS exact", omegas_amls),
        ("RCB Ritz (exact)", omegas_ritz_rcb),
        ("ND Ritz (exact)", omegas_ritz_nd),
    ]
    for name, vals in exact_methods:
        safe = name.replace(" ", "_").replace("/", "_")
        plot_scatter_comparison(omegas_exact, vals, name,
                                str(dir_exact / f"{base}_scatter_{safe}_f64.png"))

    # Float32 exact methods scatter plots
    exact_methods_f32 = [
        ("AMLS exact f32", omegas_amls_f32),
        ("RCB Ritz exact f32", omegas_ritz_rcb_f32),
        ("ND Ritz exact f32", omegas_ritz_nd_f32),
    ]
    for name, vals in exact_methods_f32:
        safe = name.replace(" ", "_").replace("/", "_")
        plot_scatter_comparison(omegas_exact, vals, name,
                                str(dir_exact / f"{base}_scatter_{safe}.png"))

    # Combined overlay plot for all exact methods
    _plot_exact_methods_overlay(omegas_exact, omegas_amls, omegas_ritz_rcb, omegas_ritz_nd,
                               omegas_amls_f32, omegas_ritz_rcb_f32, omegas_ritz_nd_f32,
                               str(dir_exact / f"{base}_scatter_exact_overlay.png"))

    # Approximate methods scatter plots
    approx_methods = [
        ("RCB block-diag", omegas_rcb),
        ("ND block-diag", omegas_nd),
        ("Guyan (sep only)", omegas_guyan),
    ]
    if omegas_tr_rcb_8 is not None:
        approx_methods.append(("RCB trunc 8/blk", omegas_tr_rcb_8))
    if omegas_tr_nd_8 is not None:
        approx_methods.append(("ND trunc 8/blk", omegas_tr_nd_8))

    for name, vals in approx_methods:
        safe = name.replace(" ", "_").replace("/", "_")
        plot_scatter_comparison(omegas_exact, vals, name,
                                str(dir_approx / f"{base}_scatter_{safe}.png"))

    print("\nDone.")


if __name__ == "__main__":
    run(parse_args())
