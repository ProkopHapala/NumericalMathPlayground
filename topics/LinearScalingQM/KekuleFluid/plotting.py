#!/usr/bin/env python3
"""plotting.py — Common visualization layer for the 4-component Dirac family (legacy).

Physical background
-------------------
Visualization helpers for the 4-component Dirac solver (Dirac4_ocl.py).  Renders
honeycomb graphs, Dirac spinor fields (density, Kekulé mass, bilinear response
S_K), vortex/antivortex detection on hexagonal rings, and nodal lines.

Role in the system
------------------
Legacy visualization module used by run_dirac.py and run_combined.py.  Redundant
with visualize.py — the current codebase uses visualize.py (HexVisualizer /
LiveViewer) for all plotting.  This module is kept for compatibility with the
4-component Dirac solver family.

Provides functions to visualize:
  - Honeycomb graph with atoms, bonds, and bond orders
  - Dirac solver fields (density, Kekule mass, bilinear response)
  - Vortex/antivortex detection on hexagonal rings
  - Nodal lines (Re=0, Im=0 crossings)

This module is shared between the Dirac solver (Model B) and the
fluid/dimer solver (Model A). Each solver provides its own data;
this module only handles rendering.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
from typing import Optional, List, Tuple


# ---- Color helpers ----

def phase_cmap():
    """Cyclic colormap for phase fields (hue cycle)."""
    return plt.cm.hsv


def make_phase_norm(phase_arr):
    """Normalization that maps [-pi, pi] to [0, 1] for cyclic colormap."""
    return mcolors.Normalize(vmin=-np.pi, vmax=np.pi)


# ---- Honeycomb graph plotting ----

def plot_honeycomb(ax, graph, bond_values=None, atom_values=None,
                   bond_cmap='RdYlBu', atom_cmap='viridis',
                   bond_vmin=None, bond_vmax=None,
                   atom_size=30, bond_width=2.0, show_rings=False,
                   ring_winding=None, title=""):
    """Plot a honeycomb graph with atoms and bonds colored by values.

    Parameters:
      graph: HoneycombGraph from hexgrid.py
      bond_values: (n_bonds,) array, e.g. bond orders x_ij
      atom_values: (n_atoms,) array, e.g. charge or |z|
      ring_winding: (n_rings,) array of winding numbers for ring coloring
    """
    # Draw bonds as LineCollection
    bond_segs = []
    for ia, ib in graph.bonds:
        bond_segs.append([graph.pos[ia], graph.pos[ib]])

    lc = LineCollection(bond_segs, linewidths=bond_width)
    if bond_values is not None:
        lc.set_array(bond_values)
        if bond_cmap:
            lc.set_cmap(bond_cmap)
        if bond_vmin is not None:
            lc.set_clim(bond_vmin, bond_vmax)
        elif bond_values is not None:
            vabs = max(abs(bond_values.min()), abs(bond_values.max()))
            lc.set_clim(-vabs, vabs)
    ax.add_collection(lc)

    # Draw atoms
    if atom_values is not None:
        sc = ax.scatter(graph.pos[:, 0], graph.pos[:, 1],
                        c=atom_values, cmap=atom_cmap, s=atom_size,
                        zorder=5, edgecolors='black', linewidths=0.3)
    else:
        sc = ax.scatter(graph.pos[:, 0], graph.pos[:, 1],
                        c='black', s=atom_size, zorder=5)

    # Draw ring winding markers
    if show_rings and ring_winding is not None:
        for ir, w in enumerate(ring_winding):
            if abs(w) > 0.5:
                c = graph.ring_centers[ir]
                color = 'red' if w > 0 else 'blue'
                ax.plot(c[0], c[1], 'o', color=color, markersize=8,
                        markerfacecolor='none', markeredgewidth=2, zorder=10)
                ax.text(c[0], c[1], f'{w:+.0f}', ha='center', va='center',
                        fontsize=7, color=color, fontweight='bold', zorder=11)

    ax.set_aspect('equal')
    ax.set_title(title)
    margin = 1.5
    if len(graph.pos) > 0:
        ax.set_xlim(graph.pos[:, 0].min() - margin, graph.pos[:, 0].max() + margin)
        ax.set_ylim(graph.pos[:, 1].min() - margin, graph.pos[:, 1].max() + margin)

    return lc, sc


# ---- Dirac field plotting ----

def plot_dirac_fields(diag, delta=None, grid_extent=None,
                      fig=None, figsize=(14, 10)):
    """Plot Dirac solver diagnostics in a multi-panel figure.

    diag: dict from Dirac4Solver.compute_diagnostics()
    delta: optional complex Delta field (Nx, Ny)
    """
    if fig is not None:
        fig.clear()
    fig = fig or plt.figure(figsize=figsize)

    rho = diag['rho']
    SK = diag['SK']
    arg_SK = diag['arg_SK']
    abs_SK = diag['abs_SK']

    # --- Panel 1: Density ---
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(rho.T, origin='lower', cmap='hot', interpolation='bilinear')
    ax1.set_title(r'$\rho = \Psi^\dagger \Psi$')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # --- Panel 2: |SK| ---
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(abs_SK.T, origin='lower', cmap='magma', interpolation='bilinear')
    ax2.set_title(r'$|S_K|$')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    # --- Panel 3: arg(SK) ---
    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.imshow(arg_SK.T, origin='lower', cmap='hsv', interpolation='bilinear',
                     vmin=-np.pi, vmax=np.pi)
    ax3.set_title(r'$\arg(S_K)$')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # --- Panel 4: Delta mass (if provided) ---
    if delta is not None:
        ax4 = fig.add_subplot(2, 3, 4)
        abs_D = np.abs(delta)
        im4 = ax4.imshow(abs_D.T, origin='lower', cmap='viridis', interpolation='bilinear')
        ax4.set_title(r'$|\Delta|$')
        plt.colorbar(im4, ax=ax4, fraction=0.046)

        ax5 = fig.add_subplot(2, 3, 5)
        arg_D = np.angle(delta)
        im5 = ax5.imshow(arg_D.T, origin='lower', cmap='hsv', interpolation='bilinear',
                         vmin=-np.pi, vmax=np.pi)
        ax5.set_title(r'$\arg(\Delta)$')
        plt.colorbar(im5, ax=ax5, fraction=0.046)

        # --- Panel 6: Overlay rho contour on |Delta| ---
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.imshow(abs_D.T, origin='lower', cmap='gray', alpha=0.5, interpolation='bilinear')
        ax6.contour(rho.T, levels=5, colors='red', linewidths=0.5)
        ax6.set_title(r'$\rho$ contours on $|\Delta|$')
    else:
        ax6 = fig.add_subplot(2, 3, 4)
        im6 = ax6.imshow(rho.T, origin='lower', cmap='hot', interpolation='bilinear')
        ax6.set_title(r'$\rho$ (zoom)')
        plt.colorbar(im6, ax=ax6, fraction=0.046)

    fig.tight_layout()
    return fig


# ---- Combined plot: hexagonal graph + Dirac grid overlay ----

def plot_combined(graph, bond_values=None, atom_values=None,
                  dirac_diag=None, delta=None,
                  grid_origin=None, grid_dx=1.0,
                  fig=None, figsize=(14, 6)):
    """Plot honeycomb graph side by side with Dirac solver fields.

    This is the common visualization that shows both the atomistic
    picture (atoms + bonds) and the continuum Dirac response.
    """
    if fig is not None:
        fig.clear()
    fig = fig or plt.figure(figsize=figsize)

    # Left: honeycomb graph
    ax1 = fig.add_subplot(1, 2, 1)
    plot_honeycomb(ax1, graph, bond_values=bond_values, atom_values=atom_values,
                   title="Honeycomb graph")

    # Right: Dirac density overlaid with graph
    ax2 = fig.add_subplot(1, 2, 2)
    if dirac_diag is not None:
        rho = dirac_diag['rho']
        extent = (0, rho.shape[0] * grid_dx, 0, rho.shape[1] * grid_dx)
        im = ax2.imshow(rho.T, origin='lower', extent=extent,
                        cmap='hot', interpolation='bilinear', alpha=0.7)
        plt.colorbar(im, ax=ax2, fraction=0.046, label=r'$\rho$')

        # Overlay honeycomb graph
        if delta is not None:
            # Show Delta phase as contour
            arg_D = np.angle(delta)
            ax2.contour(arg_D.T, levels=[-np.pi, 0, np.pi],
                        extent=extent, colors='cyan', linewidths=0.3, alpha=0.5)

    # Draw graph on top
    bond_segs = [[graph.pos[ia], graph.pos[ib]] for ia, ib in graph.bonds]
    lc = LineCollection(bond_segs, linewidths=1.0, colors='white', alpha=0.5)
    ax2.add_collection(lc)
    ax2.scatter(graph.pos[:, 0], graph.pos[:, 1], c='white', s=10, zorder=5, edgecolors='gray')

    ax2.set_aspect('equal')
    ax2.set_title("Dirac response on grid")
    margin = 1.5
    ax2.set_xlim(graph.pos[:, 0].min() - margin, graph.pos[:, 0].max() + margin)
    ax2.set_ylim(graph.pos[:, 1].min() - margin, graph.pos[:, 1].max() + margin)

    fig.tight_layout()
    return fig


# ---- Animation helper ----

class DiracAnimator:
    """Simple animation helper for Dirac solver propagation.

    Usage:
        anim = DiracAnimator(solver, delta, fig, ax)
        anim.run(n_steps=200, interval=50)
    """

    def __init__(self, solver, fig=None, ax=None, figsize=(8, 8)):
        self.solver = solver
        if fig is None or ax is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)
        else:
            self.fig, self.ax = fig, ax

        self.diag = solver.compute_diagnostics()
        self.im = self.ax.imshow(self.diag['rho'].T, origin='lower',
                                 cmap='hot', interpolation='bilinear',
                                 animated=True)
        self.ax.set_title("Step 0")
        self.fig.colorbar(self.im, ax=self.ax, fraction=0.046)
        self.frames = []

    def update(self, frame):
        for _ in range(3):  # 3 steps per frame for speed
            self.solver.step()
        self.diag = self.solver.compute_diagnostics()
        self.im.set_array(self.diag['rho'].T)
        self.im.set_clim(self.diag['rho'].min(), max(self.diag['rho'].max(), 1e-10))
        self.ax.set_title(f"Step {frame * 3}")
        return [self.im]

    def run(self, n_steps=200, interval=50, save_path=None):
        frames = n_steps // 3
        anim = FuncAnimation(self.fig, self.update, frames=frames,
                             interval=interval, blit=True, repeat=False)
        if save_path:
            anim.save(save_path, writer='pillow', fps=20)
        plt.show()
        return anim
