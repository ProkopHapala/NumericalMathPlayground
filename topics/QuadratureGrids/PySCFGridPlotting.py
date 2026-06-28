"""
PySCFGridPlotting.py — Visualization for PySCF DFT integration grids.

Plotting module: 3D scatter, 2D projections, radial distributions,
Lebedev shell visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_3d_scatter(coords, atm_idx, atom_labels, atom_colors, atom_coords,
                    level, total_pts, max_pts=5000):
    """3D scatter of grid points colored by atom assignment."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    natm = len(atom_labels)
    for ia in range(natm):
        mask = atm_idx == ia
        pts = coords[mask]
        if len(pts) > max_pts:
            idx = np.random.choice(len(pts), max_pts, replace=False)
            pts = pts[idx]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=atom_colors[ia], s=1, alpha=0.3,
                   label=f'{atom_labels[ia]} ({mask.sum()} pts)')
    for ia in range(natm):
        ax.scatter(*atom_coords[ia], c='black', s=100, marker='o',
                   edgecolors='white', linewidths=1.5, zorder=5)
        ax.text(*atom_coords[ia], f'  {atom_labels[ia]}', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (Bohr)')
    ax.set_ylabel('Y (Bohr)')
    ax.set_zlabel('Z (Bohr)')
    ax.set_title(f'PySCF DFT Grid Points\n(Level {level}, {total_pts} total points)')
    ax.legend(markerscale=5)
    plt.tight_layout()
    return fig


def plot_2d_projections(coords, atm_idx, atom_labels, atom_colors, atom_coords, level):
    """2D projections (XY, XZ, YZ planes)."""
    natm = len(atom_labels)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    planes = [('X', 'Y', 0, 1), ('X', 'Z', 0, 2), ('Y', 'Z', 1, 2)]
    for ax, (xl, yl, ix, iy) in zip(axes, planes):
        for ia in range(natm):
            mask = atm_idx == ia
            pts = coords[mask]
            if len(pts) > 5000:
                idx = np.random.choice(len(pts), 5000, replace=False)
                pts = pts[idx]
            ax.scatter(pts[:, ix], pts[:, iy], c=atom_colors[ia], s=1, alpha=0.3,
                       label=atom_labels[ia])
        for ia in range(natm):
            ax.scatter(atom_coords[ia, ix], atom_coords[ia, iy],
                       c='black', s=100, marker='o', edgecolors='white', linewidths=1.5, zorder=5)
            ax.text(atom_coords[ia, ix] + 0.05, atom_coords[ia, iy] + 0.05,
                    atom_labels[ia], fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{xl} (Bohr)')
        ax.set_ylabel(f'{yl} (Bohr)')
        ax.set_aspect('equal')
        ax.legend(markerscale=5)
        ax.set_title(f'{xl}{yl} plane')
    fig.suptitle(f'PySCF DFT Grid Projections (Level {level})', fontsize=14)
    plt.tight_layout()
    return fig


def plot_radial_distribution(coords, atom_coords, atm_idx, atom_labels, atom_colors, level):
    """Radial distribution histogram per atom."""
    natm = len(atom_labels)
    fig, ax = plt.subplots(figsize=(10, 6))
    for ia in range(natm):
        mask = atm_idx == ia
        r = np.linalg.norm(coords[mask] - atom_coords[ia], axis=1)
        ax.hist(r, bins=80, alpha=0.5, color=atom_colors[ia],
                label=f'{atom_labels[ia]} ({mask.sum()} pts)')
    ax.set_xlabel('Distance from atom center (Bohr)')
    ax.set_ylabel('Number of grid points')
    ax.set_title(f'Radial distribution of DFT grid points (Level {level})')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_grid_count_vs_level(mol, levels=range(0, 8)):
    """Grid count vs integration level."""
    from pyscf import dft
    counts = []
    for lvl in levels:
        g = dft.gen_grid.Grids(mol)
        g.level = lvl
        g.build()
        counts.append(len(g.weights))
        print(f"  Level {lvl}: {len(g.weights)} grid points")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(levels, counts, color='steelblue')
    ax.set_xlabel('Grid level')
    ax.set_ylabel('Total grid points')
    ax.set_title('Grid size vs. integration level')
    for lvl, c in zip(levels, counts):
        ax.text(lvl, c + 200, str(c), ha='center', fontsize=9)
    plt.tight_layout()
    return fig


def plot_single_atom_3d(coords, n_rad, n_ang):
    """3D scatter of single-atom grid colored by radial distance."""
    r_pts = np.linalg.norm(coords, axis=1)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                         c=r_pts, cmap='viridis', s=2, alpha=0.4, edgecolors='none')
    plt.colorbar(scatter, ax=ax, label='Radial distance (Bohr)', shrink=0.6)
    ax.scatter(0, 0, 0, c='red', s=200, marker='*', zorder=5, label='nucleus')
    ax.set_xlabel('X (Bohr)')
    ax.set_ylabel('Y (Bohr)')
    ax.set_zlabel('Z (Bohr)')
    ax.set_title(f'Atomic DFT grid (level 3)\n{len(coords)} points, {n_rad} radial × Lebedev angular')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_single_atom_2d(coords):
    """2D projections of single-atom grid colored by radius."""
    r_pts = np.linalg.norm(coords, axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    planes = [('X', 'Y', 0, 1), ('X', 'Z', 0, 2), ('Y', 'Z', 1, 2)]
    for ax, (xl, yl, ix, iy) in zip(axes, planes):
        sc = ax.scatter(coords[:, ix], coords[:, iy], c=r_pts, cmap='viridis',
                        s=2, alpha=0.3, edgecolors='none')
        ax.scatter(0, 0, c='red', s=150, marker='*', zorder=5)
        ax.set_xlabel(f'{xl} (Bohr)')
        ax.set_ylabel(f'{yl} (Bohr)')
        ax.set_aspect('equal')
        ax.set_title(f'{xl}{yl} plane')
    fig.colorbar(sc, ax=axes, label='Radial distance (Bohr)', shrink=0.8)
    fig.suptitle(f'Atomic DFT grid projections (level 3, {len(coords)} points)', fontsize=14)
    plt.tight_layout()
    return fig


def plot_single_atom_radial(rad, angs, n_rad):
    """Radial points, angular pruning, and cumulative count."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    shell_idx = np.arange(n_rad)
    ax1.plot(shell_idx, rad, 'o-', markersize=3, color='steelblue')
    ax1.set_ylabel('Radius (Bohr)')
    ax1.set_title(f'Treutler-Ahlrichs radial grid ({n_rad} shells)')
    colors_map = {50: 'tab:red', 86: 'tab:orange', 350: 'tab:green', 434: 'tab:blue'}
    for n_leb in sorted(set(angs)):
        mask = angs == n_leb
        ax2.scatter(shell_idx[mask], angs[mask], s=20,
                    color=colors_map.get(n_leb, 'gray'),
                    label=f'{n_leb} Lebedev pts ({mask.sum()} shells)')
    ax2.set_ylabel('Number of angular points')
    ax2.legend(fontsize=9)
    ax2.set_title('Angular grid size per shell (after pruning)')
    cumulative = np.cumsum(angs)
    ax3.plot(shell_idx, cumulative, 'o-', markersize=3, color='darkgreen')
    ax3.set_xlabel('Shell index')
    ax3.set_ylabel('Cumulative number of points')
    ax3.set_title(f'Cumulative grid points (total = {cumulative[-1]})')
    plt.tight_layout()
    return fig


def plot_lebedev_shells(rad, angs, n_rad):
    """Individual Lebedev angular shells at different radii."""
    from pyscf.dft.LebedevGrid import MakeAngularGrid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    shell_indices = [0, 5, 15, 30, 50, n_rad - 1]
    for ax, si in zip(axes.flat, shell_indices):
        if si >= n_rad:
            ax.set_visible(False)
            continue
        n_leb = int(angs[si])
        grid = MakeAngularGrid(n_leb)
        pts = rad[si] * grid[:, :3]
        w = grid[:, 3]
        ax.scatter(pts[:, 0], pts[:, 1], c=w, cmap='coolwarm', s=10, edgecolors='none')
        ax.scatter(0, 0, c='red', s=50, marker='*', zorder=5)
        ax.set_aspect('equal')
        ax.set_title(f'Shell {si}: r={rad[si]:.3f} Bohr\n{n_leb} Lebedev points')
        ax.set_xlabel('X (Bohr)')
        ax.set_ylabel('Y (Bohr)')
    fig.suptitle('Individual Lebedev angular shells at different radii', fontsize=14)
    plt.tight_layout()
    return fig


# ── Atomic radial wavefunction plotting ───────────────────────────────────────

def plot_radial_wavefunctions(elem_shells, r_grid, basis, rmax):
    """
    Plot raw radial wavefunctions R(r) for multiple elements.

    Parameters
    ----------
    elem_shells : list of (elem, shells) where shells is list of (name, l, R)
    r_grid : (N,) array
    basis : str — basis set name
    rmax : float — x-axis limit

    Returns
    -------
    fig : matplotlib Figure
    """
    n_elem = len(elem_shells)
    fig, axes = plt.subplots(1, n_elem, figsize=(5 * n_elem, 5), squeeze=False)
    axes = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for ielem, (elem, shells) in enumerate(elem_shells):
        ax = axes[ielem]
        shells_sorted = sorted(shells, key=lambda x: (x[1], x[0]))
        for ish, (name, l, R) in enumerate(shells_sorted):
            color = colors[ish % 10]
            lw = 2.0 if l == 0 else 1.5
            ls = ['-', '--', ':'][l] if l < 3 else '-.'
            ax.plot(r_grid, R, label=name, color=color, linewidth=lw, linestyle=ls)
        ax.set_title(f'{elem} ({basis})', fontsize=14)
        ax.set_xlabel('r (Bohr)', fontsize=12)
        ax.set_ylabel('R(r)', fontsize=12)
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(0, rmax)
        r_nonzero = r_grid > 0.1
        all_R = np.concatenate([s[2][r_nonzero] for s in shells_sorted])
        ymax = np.max(np.abs(all_R)) * 1.2
        ax.set_ylim(-ymax, ymax)
        for ish, (name, l, R) in enumerate(shells_sorted):
            R_abs = np.abs(R)
            if R_abs.max() > 0:
                threshold = 0.01 * R_abs.max()
                beyond = np.where(R_abs[r_grid > 0.1] < threshold)[0]
                if len(beyond) > 0:
                    r_extent = r_grid[r_grid > 0.1][beyond[0]]
                else:
                    r_extent = rmax
                ax.axvline(r_extent, color=colors[ish % 10], linewidth=0.5, alpha=0.3, linestyle=':')

    fig.suptitle(f'Radial wavefunctions R(r) — {basis}', fontsize=16, y=1.02)
    plt.tight_layout()
    return fig


def plot_radial_wavefunctions_normalized(elem_shells, r_grid, basis, rmax):
    """Plot normalized radial wavefunctions R(r)/R_max for multiple elements."""
    n_elem = len(elem_shells)
    fig, axes = plt.subplots(1, n_elem, figsize=(5 * n_elem, 5), squeeze=False)
    axes = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for ielem, (elem, shells) in enumerate(elem_shells):
        ax = axes[ielem]
        shells_sorted = sorted(shells, key=lambda x: (x[1], x[0]))
        for ish, (name, l, R) in enumerate(shells_sorted):
            color = colors[ish % 10]
            lw = 2.0 if l == 0 else 1.5
            ls = ['-', '--', ':'][l] if l < 3 else '-.'
            R_norm = R / (np.abs(R).max() + 1e-30)
            ax.plot(r_grid, R_norm, label=name, color=color, linewidth=lw, linestyle=ls)
        ax.set_title(f'{elem} ({basis})', fontsize=14)
        ax.set_xlabel('r (Bohr)', fontsize=12)
        ax.set_ylabel('R(r) / R_max', fontsize=12)
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(0, rmax)
        ax.set_ylim(-1.2, 1.2)

    fig.suptitle(f'Normalized radial wavefunctions — {basis}', fontsize=16, y=1.02)
    plt.tight_layout()
    return fig


def plot_radial_density(elem_shells, r_grid, basis, rmax):
    """Plot radial density r²|R(r)|² (log scale) for multiple elements."""
    n_elem = len(elem_shells)
    fig, axes = plt.subplots(1, n_elem, figsize=(5 * n_elem, 5), squeeze=False)
    axes = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for ielem, (elem, shells) in enumerate(elem_shells):
        ax = axes[ielem]
        shells_sorted = sorted(shells, key=lambda x: (x[1], x[0]))
        for ish, (name, l, R) in enumerate(shells_sorted):
            color = colors[ish % 10]
            lw = 2.0 if l == 0 else 1.5
            ls = ['-', '--', ':'][l] if l < 3 else '-.'
            r2_R2 = r_grid**2 * R**2
            ax.semilogy(r_grid, r2_R2 + 1e-30, label=name, color=color, linewidth=lw, linestyle=ls)
        ax.set_title(f'{elem} ({basis})', fontsize=14)
        ax.set_xlabel('r (Bohr)', fontsize=12)
        ax.set_ylabel(r'$r^2 |R(r)|^2$', fontsize=12)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(0, rmax)
        all_vals = np.concatenate([r_grid**2 * s[2]**2 + 1e-30 for s in shells_sorted])
        ax.set_ylim(1e-10, all_vals.max() * 2)

    fig.suptitle(f'Radial density $r^2|R(r)|^2$ — {basis}', fontsize=16, y=1.02)
    plt.tight_layout()
    return fig
