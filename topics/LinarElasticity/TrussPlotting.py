"""
TrussPlotting.py — Reusable plotting/visualization functions for truss systems.

This is the **visualization layer** of the LinearElasticity module system.
It provides plotting functions for truss meshes, convergence curves,
beam snapshots, partitioning diagrams, spectra, and mode shapes.
All functions are pure visualization — no solver logic.

Responsibilities
----------------
1. **Truss mesh visualization** — `plot_truss` (edge-colored by type),
   `plot_truss_charge` (nodes colored by charge + displacement vectors).
2. **Convergence plots** — `plot_convergence` (generic semilogy residual
   curves with configurable labels, colors, line styles).
3. **Beta sweep plots** — `plot_beta_sweep` (residual vs iteration for
   multiple beta values, each with distinct color).
4. **Beam snapshot plots** — `plot_beam_snapshots` (overlay deformed beam
   shapes at selected iteration counts).
5. **1D partitioning plots** — `plot_1d_patches` (colored horizontal segments
   for overlapping patches), `plot_1d_patches_alt` (alternating sets A/B).
6. **Spectrum plots** — `plot_spectrum` (response magnitude vs frequency).
7. **Mode shape plots** — `plot_modes_with_response` (eigenvectors + response
   vectors overlaid on mesh).
8. **Hex grid plots** — `plot_hex_mesh`, `plot_matrix` (from elasticity_benchmark).

Role in the system
------------------
- **Truss.py**: geometry, mesh, assembly, bookkeeping.
- **TrussSolver.py**: iterative and direct linear algebra solvers.
- **TrussPlotting.py** (this file): all reusable plotting functions.
- Scripts: thin wrappers that combine the three modules.

Design principles
-----------------
- Functions accept matplotlib axes when possible (caller manages figure layout).
- No `plt.show()` inside library functions (caller decides when to show).
- No solver logic — pure visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------------
# Truss mesh visualization
# ---------------------------------------------------------------------------

def plot_truss(pos, edges, ax=None, color_by_type=True, node_size=3, linewidth=0.6):
    """
    Quick 2D visualisation of a truss.

    Edge colours: blue=longitudinal, green=perpendicular, red=diagonal.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    cmap = {'long': '#2196F3', 'perp': '#4CAF50', 'diag': '#F44336'}
    for e in edges:
        i, j = e[0], e[1]
        et = e[2] if len(e) > 2 else None
        c = cmap.get(et, '#999')
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                color=c, linewidth=linewidth)

    ax.scatter(pos[:, 0], pos[:, 1], s=node_size, c='black', zorder=5)
    ax.set_aspect('equal')
    return ax


def plot_truss_charge(pos, edges, charges, disp=None, scale=0.2, title="truss",
                      ax=None):
    """Plot nodes colored by charge and optional displacement vectors."""
    charges = np.asarray(charges)
    if disp is not None:
        disp = np.real(disp)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=charges, cmap="coolwarm", s=40, edgecolors="k")
    for i, j in edges:
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], "k-", lw=0.5, alpha=0.6)
    if disp is not None:
        ax.quiver(pos[:, 0], pos[:, 1], disp[:, 0], disp[:, 1],
                  angles="xy", scale_units="xy", scale=1.0 / scale,
                  color="tab:green", width=0.005, alpha=0.8)
    ax.set_aspect("equal")
    ax.set_title(title)
    return ax, sc


# ---------------------------------------------------------------------------
# Convergence plots
# ---------------------------------------------------------------------------

def plot_convergence(ax, residuals_list, labels, colors, line_styles=None,
                     markers=None, title='Convergence', xlabel='Outer iterations',
                     ylabel='Relative residual', ylim_bottom=1e-6, fontsize=6):
    """
    Generic convergence plot on a semilogy axis.

    Parameters
    ----------
    ax : matplotlib axes
    residuals_list : list of 1D arrays (each length n_iter+1, starting from iter 0)
    labels : list of str
    colors : list of color specs
    line_styles : list of line style strings (default all '-')
    markers : list of marker specs or None
    """
    if line_styles is None:
        line_styles = ['-'] * len(residuals_list)
    if markers is None:
        markers = [None] * len(residuals_list)
    for res, label, col, ls, mk in zip(residuals_list, labels, colors, line_styles, markers):
        n = len(res)
        kw = dict(color=col, ls=ls, lw=1.5, label=label)
        if mk is not None:
            kw['marker'] = mk
            kw['ms'] = 3
        ax.semilogy(np.arange(0, n), res, **kw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    ax.legend(fontsize=fontsize)
    ax.grid(True, alpha=0.3)


def plot_beta_sweep(ax, residuals_list, labels, colors, markers=None,
                    title='Beta sweep', xlabel='Outer iterations',
                    ylabel='Relative residual', ylim_bottom=1e-6, fontsize=7):
    """
    Beta sweep plot — each line has a distinct color.

    Parameters
    ----------
    ax : matplotlib axes
    residuals_list : list of 1D arrays
    labels : list of str (e.g. 'DirBJ β=0.0', 'DirBJ β=0.3', ...)
    colors : list of distinct color specs
    """
    if markers is None:
        markers = [None] * len(residuals_list)
    for res, label, col, mk in zip(residuals_list, labels, colors, markers):
        n = len(res)
        kw = dict(color=col, lw=1.5, label=label)
        if mk is not None:
            kw['marker'] = mk
            kw['ms'] = 3
        ax.semilogy(np.arange(0, n), res, **kw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    ax.legend(fontsize=fontsize)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Beam snapshot plots
# ---------------------------------------------------------------------------

def plot_beam_snapshots(ax, x0, edges, snapshot_results, snapshot_iters,
                        snapshot_colors, method_label='DirBJ'):
    """
    Overlay deformed beam shapes at selected iteration counts.

    Parameters
    ----------
    ax : matplotlib axes
    x0 : (N, dim) initial positions
    edges : list of (i, j)
    snapshot_results : dict {n_iter: (x_final, residuals)}
    snapshot_iters : list of int (iteration counts to plot)
    snapshot_colors : list of color specs
    method_label : str for legend prefix
    """
    # Plot initial shape (light gray)
    for i, j in edges:
        ax.plot([x0[i, 0], x0[j, 0]], [x0[i, 1], x0[j, 1]],
                color='gray', lw=0.5, alpha=0.3)
    for n_iter, col in zip(snapshot_iters, snapshot_colors):
        if n_iter not in snapshot_results:
            continue
        x_snap = snapshot_results[n_iter][0]
        for i, j in edges:
            ax.plot([x_snap[i, 0], x_snap[j, 0]], [x_snap[i, 1], x_snap[j, 1]],
                    color=col, lw=0.8, alpha=0.6)
        ax.plot([], [], color=col, lw=1, label=f'{method_label} {n_iter}')
    ax.set_aspect('equal')


# ---------------------------------------------------------------------------
# 1D partitioning plots
# ---------------------------------------------------------------------------

def plot_1d_patches(ax, patches, patch_colors, ny=2, y_offset=0,
                    title='1D partitioning'):
    """
    Plot 1D overlapping patches as colored horizontal line segments.

    Parameters
    ----------
    ax : matplotlib axes
    patches : list of patch dicts with 'grid_ix0', 'grid_ix1'
    patch_colors : list of color specs
    ny : number of rows (for y positioning)
    y_offset : vertical offset for the segments
    """
    for pi, pat in enumerate(patches):
        col = patch_colors[pi % len(patch_colors)]
        ix0 = pat.get('grid_ix0', 0)
        ix1 = pat.get('grid_ix1', 0)
        y = y_offset - pi * 0.3
        ax.plot([ix0, ix1], [y, y], color=col, lw=3, solid_capstyle='round')
        ax.plot([ix0, ix1], [y, y], color=col, lw=3, solid_capstyle='round',
                label=f'patch {pi}' if pi < len(patch_colors) else '')
    ax.set_ylim(y_offset - len(patches) * 0.3 - 0.5, y_offset + 0.5)
    ax.set_title(title)


def plot_1d_patches_alt(ax, patch_sets, set_colors, ny=2, y_offset=0,
                        title='1D partitioning (alternating)'):
    """
    Plot alternating 1D patch sets as colored horizontal segments.

    Parameters
    ----------
    ax : matplotlib axes
    patch_sets : list of two lists of patches (set A, set B)
    set_colors : list of 2 color specs (for sets A and B)
    """
    for si, (patches, col) in enumerate(zip(patch_sets, set_colors)):
        for pi, pat in enumerate(patches):
            ix0 = pat.get('grid_ix0', 0)
            ix1 = pat.get('grid_ix1', 0)
            y = y_offset - (si * len(patch_sets[0]) + pi) * 0.3
            ax.plot([ix0, ix1], [y, y], color=col, lw=3, solid_capstyle='round')
    ax.set_title(title)


# ---------------------------------------------------------------------------
# Spectrum and mode shape plots (from VibrationProbing)
# ---------------------------------------------------------------------------

def plot_spectrum(omegas, res, eigfreq=None, sel=None, ax=None):
    """Plot response magnitude vs frequency with eigenvalue markers."""
    dip = res["dipole"]
    mag = np.linalg.norm(dip, axis=1)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(omegas, mag, "-", ms=3, lw=1)
    if eigfreq is not None:
        eigfreq = np.asarray(eigfreq)
        sel = set(sel) if sel is not None else set()
        for i, f in enumerate(eigfreq):
            color = "r" if i in sel else "0.6"
            ax.axvline(f, color=color, lw=1, alpha=0.8)
    ax.set_xlabel("omega")
    ax.set_ylabel("|dipole response|")
    ax.set_title("Dipole-coupled spectrum")
    return ax


def plot_modes_with_response(pos, edges, charges, eigvecs_full, respvecs_full,
                             freqs, sel_idx, scale=0.15):
    """
    Plot eigenmode shapes and response vectors overlaid on mesh.

    Returns a figure (not an axes) since it creates a subplot grid.
    """
    nsel = len(sel_idx)
    if nsel == 0:
        return None
    cols = min(3, nsel)
    rows = (nsel + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    charges = np.asarray(charges)
    for ax, idx, eig, resp in zip(axes, sel_idx, eigvecs_full, respvecs_full):
        eig = np.real(eig)
        resp = np.real(resp)
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=charges, cmap="coolwarm", s=30, edgecolors="k")
        for i, j in edges:
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], "k-", lw=0.5, alpha=0.5)
        ax.quiver(pos[:, 0], pos[:, 1], eig[:, 0], eig[:, 1],
                  angles="xy", scale_units="xy", scale=1.0 / scale,
                  color="tab:blue", width=0.004, alpha=0.8, label="eig")
        ax.quiver(pos[:, 0], pos[:, 1], resp[:, 0], resp[:, 1],
                  angles="xy", scale_units="xy", scale=1.0 / scale,
                  color="tab:orange", width=0.004, alpha=0.8, label="resp")
        ax.set_aspect("equal")
        ax.set_title(f"mode {idx} f={freqs[idx]:.3f}")
    for ax in axes[nsel:]:
        ax.axis("off")
    fig.colorbar(sc, ax=axes.tolist(), fraction=0.025, pad=0.02, label="charge")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Hex grid plots (from elasticity_benchmark)
# ---------------------------------------------------------------------------

def plot_hex_mesh(pos, neighs, title="Mesh", ax=None):
    """Plot nodes and edges of a hex/triangular grid."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    nnode = pos.shape[0]
    for i in range(nnode):
        for j in range(neighs.shape[1]):
            nb = neighs[i, j]
            if nb >= 0:
                ax.plot([pos[i, 0], pos[nb, 0]], [pos[i, 1], pos[nb, 1]],
                        'k-', lw=0.5, alpha=0.4)
    ax.scatter(pos[:, 0], pos[:, 1], c='royalblue', s=20, zorder=5)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return ax


def plot_matrix(mat, title="Matrix", cmap='viridis', invalid_color='white', ax=None):
    """
    Plot matrix with nearest-neighbor interpolation.
    For integer matrices (neighbor indices), -1 is treated as invalid (NaN)
    and rendered in invalid_color.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    plot_mat = mat.astype(np.float64).copy()
    plot_mat[plot_mat == -1] = np.nan
    im = ax.imshow(plot_mat, aspect='auto', interpolation='nearest', cmap=cmap)
    im.cmap.set_bad(color=invalid_color)
    ax.set_title(title)
    return ax, im


# ---------------------------------------------------------------------------
# Deformation + patch overlay plots
# ---------------------------------------------------------------------------

def plot_deformation_with_patches(ax, pos, edges, x_direct, x_solved, patches,
                                  patch_colors=None, straight_edges=None,
                                  solved_label='BJ', solved_color='r+',
                                  linestyle='-', linewidth=0.3, alpha=0.2,
                                  title='Deformation + patch boxes'):
    """
    Plot rest shape (gray), direct solve (blue), iterative solve (red),
    and patch boundary rectangles overlaid on the mesh.

    Parameters
    ----------
    ax : matplotlib axes
    pos : (N, dim) rest positions
    edges : list of (i, j) — only straight_edges are drawn if provided
    x_direct : (N, dim) direct solve positions
    x_solved : (N, dim) iterative solver positions
    patches : list of patch dicts with 'grid_ix0', 'grid_iy0', etc.
    patch_colors : list of color specs for rectangles
    straight_edges : subset of edges to draw (optional)
    """
    from matplotlib.patches import Rectangle
    if patch_colors is None:
        patch_colors = ['r', 'g', 'm', 'orange', 'cyan', 'purple', 'brown']
    draw_edges = straight_edges if straight_edges is not None else edges
    for i, j in draw_edges:
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                'k-', lw=linewidth, alpha=alpha)
    for pi, pat in enumerate(patches):
        if 'grid_ix0' in pat:
            ix0, iy0 = pat['grid_ix0'], pat['grid_iy0']
            ix1, iy1 = pat['grid_ix1'], pat['grid_iy1']
            pad = 0.15
            rect = Rectangle((ix0 - pad, iy0 - pad),
                             (ix1 - ix0) + 2 * pad, (iy1 - iy0) + 2 * pad,
                             fill=False, ec=patch_colors[pi % len(patch_colors)],
                             lw=1.0, alpha=0.45, ls=linestyle)
            ax.add_patch(rect)
    ax.plot(pos[:, 0], pos[:, 1], 'k.', ms=2, label='rest')
    ax.plot(x_direct[:, 0], x_direct[:, 1], 'b.', ms=4, label='direct')
    ax.plot(x_solved[:, 0], x_solved[:, 1], solved_color, ms=4, alpha=0.7,
            label=solved_label)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend(fontsize=8)


def plot_per_node_error(ax, pos, edges, x_ref, x_approx,
                        straight_edges=None, cmap='hot_r', title='Per-node error'):
    """
    Scatter plot of per-node error ||x_approx - x_ref|| colored by magnitude.

    Parameters
    ----------
    ax : matplotlib axes
    pos : (N, dim) rest positions for scatter coordinates
    edges : list of (i, j) — only straight_edges drawn if provided
    x_ref : (N, dim) reference solution (e.g. direct solve)
    x_approx : (N, dim) approximate solution
    """
    draw_edges = straight_edges if straight_edges is not None else edges
    err_per_node = np.linalg.norm(x_approx - x_ref, axis=1)
    for i, j in draw_edges:
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                'k-', lw=0.2, alpha=0.15)
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=err_per_node, s=30, cmap=cmap)
    ax.set_aspect('equal')
    ax.set_title(title)
    return sc, err_per_node


def plot_1d_partitioning(ax, patches, patch_colors, seg_y0=-1.0, seg_dy=0.3,
                         label_prefix='P', title=None):
    """
    Plot 1D patch segments as colored horizontal bars below a beam plot.

    Parameters
    ----------
    ax : matplotlib axes
    patches : list of patch dicts with 'grid_ix0', 'grid_ix1'
    patch_colors : list of color specs
    seg_y0 : y-position of first segment
    seg_dy : vertical spacing between segments
    label_prefix : prefix for segment labels
    """
    for pi, pat in enumerate(patches):
        if 'grid_ix0' in pat:
            ix0, ix1 = pat['grid_ix0'], pat['grid_ix1']
            y_seg = seg_y0 - pi * seg_dy
            col = patch_colors[pi % len(patch_colors)]
            ax.plot([ix0 - 0.5, ix1 - 0.5], [y_seg, y_seg], color=col,
                    lw=5, alpha=0.6, solid_capstyle='butt')
            ax.text(-1.0, y_seg, f'{label_prefix}{pi}', fontsize=6,
                    ha='right', va='center', color=col)


def plot_1d_partitioning_alt(ax, patch_sets, set_colors_list,
                             seg_y0=-1.0, seg_dy=0.3):
    """
    Plot alternating 1D patch sets (A, B) as colored horizontal segments.

    Parameters
    ----------
    ax : matplotlib axes
    patch_sets : list of two lists of patches (set A, set B)
    set_colors_list : list of two color lists (for sets A and B)
    """
    for si, (pset, set_cols) in enumerate(zip(patch_sets, set_colors_list)):
        y_base = seg_y0 - si * (len(pset) + 1) * seg_dy
        set_label = f'set {"AB"[si]}'
        ax.text(-1.0, y_base + seg_dy / 2, set_label, fontsize=7,
                ha='right', va='center', fontweight='bold')
        for pi, pat in enumerate(pset):
            if 'grid_ix0' in pat:
                ix0, ix1 = pat['grid_ix0'], pat['grid_ix1']
                y_seg = y_base - pi * seg_dy
                col = set_cols[pi % len(set_cols)]
                ax.plot([ix0 - 0.5, ix1 - 0.5], [y_seg, y_seg], color=col,
                        lw=5, alpha=0.6, solid_capstyle='butt')


# ---------------------------------------------------------------------------
# Multigrid visualization
# ---------------------------------------------------------------------------

def plot_prolongation_modes(pos, edges, P, dim=3, n_show=6, scale=0.3,
                            coarse_pos=None, coarse_edges=None,
                            title='Prolongation basis modes'):
    """
    Visualize selected columns of the prolongation matrix P as displacement
    fields overlaid on the fine mesh.

    Each subplot shows one coarse basis vector: the fine-mesh displacement
    pattern that a unit coarse DOF induces.

    Parameters
    ----------
    pos : (N, dim) fine node positions
    edges : list of (i, j) or (i, j, type)
    P : (N*dim, n_coarse) prolongation matrix
    n_show : number of modes to display
    scale : displacement arrow scale
    coarse_pos : optional (N_c, dim) for overlay
    coarse_edges : optional list of (i, j) for overlay

    Returns
    -------
    fig : matplotlib Figure
    """
    n_nodes = pos.shape[0]
    n_coarse = P.shape[1]
    n_show = min(n_show, n_coarse)
    cols = min(3, n_show)
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for idx in range(n_show):
        ax = axes[idx]
        mode = P[:, idx].reshape(n_nodes, dim)
        mode_mag = np.linalg.norm(mode, axis=1)
        mode_max = mode_mag.max() + 1e-30
        mode_normalized = mode / mode_max

        for e in edges:
            i, j = e[0], e[1]
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    'k-', lw=0.3, alpha=0.2)

        ax.quiver(pos[:, 0], pos[:, 1],
                  mode_normalized[:, 0], mode_normalized[:, 1],
                  color='tab:blue', width=0.004, alpha=0.8,
                  angles="xy", scale_units="xy", scale=1.0 / scale)

        if coarse_pos is not None:
            ax.scatter(coarse_pos[:, 0], coarse_pos[:, 1], c='red', s=50,
                       zorder=5, edgecolors='k')
            if coarse_edges is not None:
                for a, b in coarse_edges:
                    ax.plot([coarse_pos[a, 0], coarse_pos[b, 0]],
                            [coarse_pos[a, 1], coarse_pos[b, 1]],
                            'r-', lw=1.5, alpha=0.5)

        ax.set_aspect('equal')
        ax.set_title(f'mode {idx}', fontsize=9)

    for ax in axes[n_show:]:
        ax.axis('off')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_coarse_overlay(ax, fine_pos, fine_edges, coarse_pos, coarse_edges,
                        fine_alpha=0.2, coarse_alpha=0.8,
                        fine_color='gray', coarse_color='red',
                        coarse_node_size=50, title='Coarse overlay'):
    """
    Overlay coarse graph on fine mesh.

    Parameters
    ----------
    ax : matplotlib axes
    fine_pos : (N, dim) fine node positions
    fine_edges : list of (i, j) or (i, j, type)
    coarse_pos : (N_c, dim) coarse node positions
    coarse_edges : list of (i, j)
    """
    for e in fine_edges:
        i, j = e[0], e[1]
        ax.plot([fine_pos[i, 0], fine_pos[j, 0]],
                [fine_pos[i, 1], fine_pos[j, 1]],
                color=fine_color, lw=0.3, alpha=fine_alpha)

    for a, b in coarse_edges:
        ax.plot([coarse_pos[a, 0], coarse_pos[b, 0]],
                [coarse_pos[a, 1], coarse_pos[b, 1]],
                color=coarse_color, lw=1.5, alpha=coarse_alpha)

    ax.scatter(coarse_pos[:, 0], coarse_pos[:, 1], c=coarse_color,
               s=coarse_node_size, zorder=5, edgecolors='k')
    ax.set_aspect('equal')
    ax.set_title(title)


def plot_pivot_selection(ax, pos, edges, pivots, free_mask=None,
                         title='Pivot selection', node_size=10,
                         pivot_size=80):
    """
    Visualize pivot nodes selected by farthest-point sampling.

    Parameters
    ----------
    ax : matplotlib axes
    pos : (N, dim) node positions
    edges : list of (i, j)
    pivots : (k,) indices of pivot nodes
    free_mask : optional (N,) bool
    """
    for i, j in edges:
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                'k-', lw=0.3, alpha=0.2)

    if free_mask is not None:
        fixed = ~free_mask
        ax.scatter(pos[fixed, 0], pos[fixed, 1], c='gray', s=node_size,
                   marker='x', zorder=3, label='fixed')

    free = free_mask if free_mask is not None else np.ones(len(pos), dtype=bool)
    ax.scatter(pos[free, 0], pos[free, 1], c='blue', s=node_size,
               zorder=4, label='free')

    ax.scatter(pos[pivots, 0], pos[pivots, 1], c='red', s=pivot_size,
               marker='*', zorder=6, edgecolors='k', label='pivots')

    for pi, p in enumerate(pivots):
        ax.annotate(str(pi), pos[p, :2], textcoords="offset points",
                    xytext=(8, 8), fontsize=7, color='red')

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend(fontsize=7)


def plot_multigrid_convergence(ax, residuals_list, labels, colors,
                               line_styles=None, title='Multigrid convergence',
                               xlabel='V-cycle iterations',
                               ylabel='Relative residual', ylim_bottom=1e-8,
                               fontsize=7):
    """
    Convergence plot specifically for multigrid V-cycle comparison.
    Same interface as plot_convergence but with multigrid defaults.
    """
    if line_styles is None:
        line_styles = ['-'] * len(residuals_list)
    for res, label, col, ls in zip(residuals_list, labels, colors, line_styles):
        n = len(res)
        ax.semilogy(np.arange(0, n), res, color=col, ls=ls, lw=1.5, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    ax.legend(fontsize=fontsize)
    ax.grid(True, alpha=0.3)
