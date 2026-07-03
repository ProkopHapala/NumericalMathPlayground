#!/usr/bin/env python3
"""run_combined.py — Side-by-side comparison of Model A (Kekulé fluid) and Model B (4-component Dirac solver).

Physical background
-------------------
This script bridges the two solver families by running Model A on a honeycomb
graph and Model B on a Cartesian grid, interpolating the Kekulé order parameter
z_i from the lattice to the grid to construct the Dirac mass field Δ(x,y).

Causal chain (from the KekuleFluid spec):
  chemical defects / edge pins
    -> x_ij (bond orders, Model A)
    -> z_i (complex Kekulé order, Model A)
    -> Delta(x,y) (interpolated to grid)
    -> Psi(x,y,t) (Dirac quasiparticle, Model B)
    -> rho(x,y,t), S_K(x,y,t) (Dirac diagnostics)

Comparison:
  Model A vortex cores  <->  Model B rho localization
  Model A arg(z)         <->  Model B arg(S_K)
  Model A nodal crossing <->  Model B density/mass core

Role in the system
------------------
Comparison script that shows the correspondence between the lattice-based
Kekulé fluid and the continuum Dirac equation.  Uses the legacy 4-component
Dirac solver (Dirac4_ocl.py).  For the current lattice-based approach, see
run_dirac_lattice.py instead.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # change to 'TkAgg' for interactive
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Graph import HoneycombGraph
from ModelA import KekuleFluidSolver
from visualize import HexVisualizer
from Dirac4_ocl import Dirac4Solver


def experiment_vortex_pair_combined():
    """Run both models on the same vortex-antivortex pair and compare."""

    # --- Model A: honeycomb graph ---
    nx, ny = 10, 10
    graph = HoneycombGraph(aCC=1.0)
    graph.build_rect_patch(nx, ny)
    print(f"Model A graph: {graph.natom} atoms, {graph.nbond} bonds, {len(graph.rings)} rings")

    pos = np.array([[a.pos[0], a.pos[1]] for a in graph.atoms])
    xmin, xmax = pos[:, 0].min(), pos[:, 0].max()
    ymin, ymax = pos[:, 1].min(), pos[:, 1].max()
    xc = 0.5 * (xmin + xmax)
    yc = 0.5 * (ymin + ymax)
    sep = 0.25 * (xmax - xmin)

    vortices = [(xc - sep, yc, +1), (xc + sep, yc, -1)]
    print(f"Vortex positions: {vortices}")

    # --- Run Model A ---
    print("Running Model A (Kekulé fluid)...")
    solverA = KekuleFluidSolver(graph)
    solverA.init_vortices(vortices, r_core=2.0)
    solverA.run(300)
    solverA.pull_to_graph()

    zr, zi = solverA.get_z()
    z_A = zr + 1j * zi
    bx_A = solverA.get_bond_x()
    print(f"  Model A: |z| range=[{np.abs(z_A).min():.3f},{np.abs(z_A).max():.3f}]")

    # --- Model B: Dirac solver on Cartesian grid ---
    # Grid covers the same spatial region as the honeycomb graph
    margin = 2.0
    grid_xmin, grid_xmax = xmin - margin, xmax + margin
    grid_ymin, grid_ymax = ymin - margin, ymax + margin
    grid_extent = max(grid_xmax - grid_xmin, grid_ymax - grid_ymin)
    Nx = Ny = 96
    dx = grid_extent / Nx

    print(f"Model B grid: {Nx}x{Ny}, dx={dx:.3f}, origin=({grid_xmin:.2f},{grid_ymin:.2f})")
    solverB = Dirac4Solver(Nx=Nx, Ny=Ny, dx=dx, dy=dx, dt=0.08 * dx, vF=1.0,
                           x0=grid_xmin, y0=grid_ymin)

    # Interpolate Model A's z field to the Dirac grid
    solverB.init_delta_from_modelA(solverA, Delta0=0.3, sigma=0.75)
    print("  Interpolated Delta from Model A")

    # Initialize Gaussian wavepacket at center
    solverB.init_gaussian_packet(
        r0=(xc, yc),
        sigma=5.0,
        k0=(0.0, 0.0),
        chi=(1.0, 0.0, 0.0, 0.0)
    )

    # Run Model B
    print("Running Model B (Dirac solver)...")
    solverB.run(200)
    diagB = solverB.compute_diagnostics()
    deltaB = solverB.get_delta()
    print(f"  Model B: rho max={diagB['rho'].max():.6f}, sum={diagB['rho'].sum():.6f}")

    # --- Visualization: side-by-side comparison ---
    viz = HexVisualizer(graph)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Model A
    # (a) Bond order
    viz.plot_bond_order(bx_A, ax=axes[0, 0], title="Model A: Bond order $x_{ij}$",
                        cmap='RdYlBu_r', vmin=0, vmax=1)

    # (b) |z| on atoms
    viz.plot_z_field(zr, zi, mode='amplitude', ax=axes[0, 1],
                     title="Model A: $|z_i|$")

    # (c) arg(z) on atoms
    viz.plot_z_field(zr, zi, mode='phase', ax=axes[0, 2],
                     title="Model A: $\\arg(z_i)$")

    # Row 2: Model B
    # (d) Dirac density
    extent = (grid_xmin, grid_xmax, grid_ymin, grid_ymax)
    im_rho = axes[1, 0].imshow(diagB['rho'].T, origin='lower', extent=extent,
                                cmap='hot', interpolation='bilinear')
    axes[1, 0].set_title("Model B: $\\rho = \\Psi^\\dagger \\Psi$")
    plt.colorbar(im_rho, ax=axes[1, 0], fraction=0.046)

    # (e) |Delta|
    im_d = axes[1, 1].imshow(np.abs(deltaB).T, origin='lower', extent=extent,
                              cmap='viridis', interpolation='bilinear')
    axes[1, 1].set_title("Model B: $|\\Delta|$ (from Model A)")
    plt.colorbar(im_d, ax=axes[1, 1], fraction=0.046)

    # (f) arg(S_K) vs arg(Delta)
    im_sk = axes[1, 2].imshow(diagB['arg_SK'].T, origin='lower', extent=extent,
                              cmap='hsv', interpolation='bilinear',
                              vmin=-np.pi, vmax=np.pi)
    axes[1, 2].set_title("Model B: $\\arg(S_K)$")
    plt.colorbar(im_sk, ax=axes[1, 2], fraction=0.046)

    # Overlay graph on Dirac panels
    for ax in axes[1]:
        bond_segs = [[pos[b.iA], pos[b.iB]] for b in graph.bonds]
        from matplotlib.collections import LineCollection
        lc = LineCollection(bond_segs, linewidths=0.5, colors='white', alpha=0.3)
        ax.add_collection(lc)
        ax.set_aspect('equal')

    fig.suptitle("Comparison: Model A (Kekulé fluid) vs Model B (Dirac solver)\n"
                 "Vortex-antivortex pair", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("comparison_vortex_pair.png", dpi=150)
    print("Saved comparison_vortex_pair.png")

    # --- Evolution comparison ---
    print("Generating evolution comparison...")
    fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
    snapshots = [0, 50, 100, 200]

    # Re-initialize Model B for clean evolution
    solverB2 = Dirac4Solver(Nx=Nx, Ny=Ny, dx=dx, dy=dx, dt=0.08 * dx, vF=1.0,
                            x0=grid_xmin, y0=grid_ymin)
    solverB2.init_delta_from_modelA(solverA, Delta0=0.3, sigma=0.75)
    solverB2.init_gaussian_packet(r0=(xc, yc), sigma=5.0)

    for step in range(201):
        if step in snapshots:
            idx = snapshots.index(step)
            diag = solverB2.compute_diagnostics()

            # Top row: Model A |z| (static — already relaxed)
            viz.plot_z_field(zr, zi, mode='amplitude', ax=axes2[0, idx],
                             title=f"Model A: $|z|$ (step {step})")

            # Bottom row: Model B rho
            im = axes2[1, idx].imshow(diag['rho'].T, origin='lower', extent=extent,
                                       cmap='hot', interpolation='bilinear')
            axes2[1, idx].set_title(f"Model B: $\\rho$ (step {step})")
            plt.colorbar(im, ax=axes2[1, idx], fraction=0.046)

        if step < 200:
            solverB2.step()

    fig2.suptitle("Evolution comparison: Model A (static) vs Model B (propagating)\n"
                  "Vortex-antivortex pair", fontsize=14, fontweight='bold')
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    fig2.savefig("comparison_evolution.png", dpi=150)
    print("Saved comparison_evolution.png")

    plt.show()
    print("Done.")


def experiment_edge_defect_combined():
    """Edge-pinned protonated defect: compare both models.

    Model A: pin an edge atom's Kekulé phase.
    Model B: interpolate the resulting Delta and propagate a wavepacket.
    """
    nx, ny = 10, 8
    graph = HoneycombGraph(aCC=1.0)
    graph.build_rect_patch(nx, ny)
    print(f"Graph: {graph.natom} atoms, {graph.nbond} bonds, {len(graph.rings)} rings")

    pos = np.array([[a.pos[0], a.pos[1]] for a in graph.atoms])
    xmin, xmax = pos[:, 0].min(), pos[:, 0].max()
    ymin, ymax = pos[:, 1].min(), pos[:, 1].max()

    # Find leftmost and rightmost atoms
    left_idx = np.argmin(pos[:, 0])
    right_idx = np.argmax(pos[:, 0])
    print(f"Left edge atom {left_idx} at {pos[left_idx]}")
    print(f"Right edge atom {right_idx} at {pos[right_idx]}")

    # Run Model A with edge pins
    print("Running Model A with edge pins...")
    solverA = KekuleFluidSolver(graph)

    # Pin left atom with phase 0, right atom with phase 2*pi/3
    solverA.set_pin(left_idx, pinStrength=1.0, dir=0)
    solverA.set_pin(right_idx, pinStrength=1.0, dir=1)

    # Also set a defect on the left atom
    solverA.set_defect(left_idx, defect=0.5, targetVal=0.5)

    solverA.init_uniform_kekule(phi0=0.0)
    solverA.run(300)
    solverA.pull_to_graph()

    zr, zi = solverA.get_z()
    z_A = zr + 1j * zi
    bx_A = solverA.get_bond_x()
    print(f"  Model A: |z| range=[{np.abs(z_A).min():.3f},{np.abs(z_A).max():.3f}]")

    # Run Model B with interpolated Delta
    margin = 2.0
    grid_xmin, grid_xmax = xmin - margin, xmax + margin
    grid_ymin, grid_ymax = ymin - margin, ymax + margin
    grid_extent = max(grid_xmax - grid_xmin, grid_ymax - grid_ymin)
    Nx = Ny = 96
    dx = grid_extent / Nx

    print("Running Model B...")
    solverB = Dirac4Solver(Nx=Nx, Ny=Ny, dx=dx, dy=dx, dt=0.08 * dx, vF=1.0,
                           x0=grid_xmin, y0=grid_ymin)
    solverB.init_delta_from_modelA(solverA, Delta0=0.3, sigma=0.75)

    # Place wavepacket near the defect
    solverB.init_gaussian_packet(r0=(pos[left_idx, 0] + 2, pos[left_idx, 1]),
                                 sigma=4.0, k0=(0.1, 0.0))
    solverB.run(200)

    diagB = solverB.compute_diagnostics()
    deltaB = solverB.get_delta()

    # Plot
    viz = HexVisualizer(graph)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    viz.plot_bond_order(bx_A, ax=axes[0, 0], title="Model A: Bond order",
                        cmap='RdYlBu_r', vmin=0, vmax=1)
    viz.plot_z_field(zr, zi, mode='amplitude', ax=axes[0, 1],
                     title="Model A: $|z|$ (defect = suppressed)")

    extent = (xmin - margin, xmax + margin, ymin - margin, ymax + margin)
    im_rho = axes[1, 0].imshow(diagB['rho'].T, origin='lower', extent=extent,
                                cmap='hot', interpolation='bilinear')
    axes[1, 0].set_title("Model B: $\\rho$")
    plt.colorbar(im_rho, ax=axes[1, 0], fraction=0.046)

    im_d = axes[1, 1].imshow(np.abs(deltaB).T, origin='lower', extent=extent,
                              cmap='viridis', interpolation='bilinear')
    axes[1, 1].set_title("Model B: $|\\Delta|$ (from Model A)")
    plt.colorbar(im_d, ax=axes[1, 1], fraction=0.046)

    # Overlay graph
    from matplotlib.collections import LineCollection
    for ax in axes[1]:
        bond_segs = [[pos[b.iA], pos[b.iB]] for b in graph.bonds]
        lc = LineCollection(bond_segs, linewidths=0.5, colors='white', alpha=0.3)
        ax.add_collection(lc)
        ax.set_aspect('equal')

    fig.suptitle("Edge defect comparison: Model A vs Model B", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("comparison_edge_defect.png", dpi=150)
    print("Saved comparison_edge_defect.png")

    plt.show()
    print("Done.")


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 1: Vortex-antivortex pair (combined)")
    print("=" * 60)
    experiment_vortex_pair_combined()

    print("\n" + "=" * 60)
    print("Experiment 2: Edge defect (combined)")
    print("=" * 60)
    experiment_edge_defect_combined()
