#!/usr/bin/env python3
"""run_dirac.py — Demo script for the 4-component Dirac-Kekule solver (legacy).

Physical background
-------------------
Demonstrates the 4-component Dirac solver with a prescribed vortex-antivortex
pair in the Kekulé mass field.  A Gaussian wavepacket is propagated through the
vortex texture, showing:
  1. The Kekulé mass field |Δ| and arg(Δ) with vortex/antivortex
  2. The Dirac quasiparticle density ρ evolving over time
  3. The Kekulé bilinear response S_K = Ψ†(M₁ + iM₂)Ψ
  4. Overlay of the honeycomb graph with the Dirac density

Role in the system
------------------
Legacy demo for the 4-component Dirac family (hexgrid.py / Dirac4_ocl.py /
plotting.py).  Alternative to run_dirac_lattice.py which uses the lattice
tight-binding solver instead.  Uses a Cartesian grid, not the honeycomb lattice
directly.

Runs Experiment 1 from the KekuleFluid spec:
  Vortex-antivortex pair in the Kekule mass field.
  A Gaussian wavepacket is propagated through the vortex texture.
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # interactive backend; change to 'Agg' for headless
import matplotlib.pyplot as plt
import sys
import os

# Add parent dirs to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hexgrid import build_honeycomb_patch, Vortex, init_vortex_phase, init_vortex_phase_grid
from Dirac4_ocl import Dirac4Solver
from plotting import plot_honeycomb, plot_dirac_fields, plot_combined, DiracAnimator


def experiment_vortex_pair():
    """Vortex-antivortex pair: Dirac quasiparticle in Kekule mass texture."""

    # --- Grid parameters ---
    Nx, Ny = 128, 128
    dx = 1.0
    dt = 0.08  # stability: dt <= 0.2 * dx / vF = 0.2

    # --- Honeycomb graph (for visualization overlay) ---
    print("Building honeycomb graph...")
    graph = build_honeycomb_patch(nx=12, ny=12, aCC=1.0)
    print(f"  {graph.n_atoms} atoms, {graph.n_bonds} bonds, {graph.n_rings} rings")

    # --- Vortex positions ---
    # Place vortex pair in the grid
    x_center = Nx * dx * 0.5
    y_center = Ny * dx * 0.5
    separation = Nx * dx * 0.15

    vortices = [
        Vortex(pos=np.array([x_center - separation, y_center]), winding=+1),
        Vortex(pos=np.array([x_center + separation, y_center]), winding=-1),
    ]

    # --- Initialize Dirac solver ---
    print("Initializing Dirac solver...")
    solver = Dirac4Solver(Nx=Nx, Ny=Ny, dx=dx, dy=dx, dt=dt, vF=1.0)

    # Initialize Kekule mass field with vortices
    solver.init_delta_vortices(vortices, Delta0=0.3, r_core=3.0)

    # Initialize Gaussian wavepacket centered between the vortices
    solver.init_gaussian_packet(
        r0=(x_center, y_center),
        sigma=8.0,
        k0=(0.0, 0.0),
        chi=(1.0, 0.0, 0.0, 0.0)  # start in A,K component
    )

    # --- Initial diagnostics ---
    print("Computing initial diagnostics...")
    diag0 = solver.compute_diagnostics()
    delta0 = solver.get_delta()

    # --- Plot initial state ---
    fig1 = plot_dirac_fields(diag0, delta=delta0)
    fig1.suptitle("Dirac solver: initial state (vortex-antivortex pair)", fontsize=14)
    fig1.savefig("dirac_initial.png", dpi=150)
    print("  Saved dirac_initial.png")

    # --- Honeycomb graph with vortex phase overlay ---
    # Initialize atom z field with same vortices for visualization
    atom_z = init_vortex_phase(graph.pos, vortices, r_core=2.0)

    fig2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: honeycomb with |z| as atom colors
    plot_honeycomb(ax_a, graph,
                   atom_values=np.abs(atom_z),
                   atom_cmap='viridis',
                   bond_values=np.ones(graph.n_bonds) * 0.5,
                   bond_cmap='gray',
                   title="Honeycomb: |z| (vortex cores = dark)")

    # Right: honeycomb with arg(z) as atom colors
    plot_honeycomb(ax_b, graph,
                   atom_values=np.angle(atom_z),
                   atom_cmap='hsv',
                   bond_values=np.ones(graph.n_bonds) * 0.5,
                   bond_cmap='gray',
                   title="Honeycomb: arg(z) (phase winding)")

    fig2.suptitle("Honeycomb graph with Kekule vortex texture", fontsize=14)
    fig2.savefig("hexgrid_vortex.png", dpi=150)
    print("  Saved hexgrid_vortex.png")

    # --- Run propagation and capture snapshots ---
    print("Running Dirac propagation...")
    n_total = 300
    snapshots = [0, 50, 100, 200, 300]

    fig3, axes = plt.subplots(1, len(snapshots), figsize=(4 * len(snapshots), 4))

    for step in range(n_total + 1):
        if step in snapshots:
            diag = solver.compute_diagnostics()
            idx = snapshots.index(step)
            im = axes[idx].imshow(diag['rho'].T, origin='lower',
                                  cmap='hot', interpolation='bilinear')
            axes[idx].set_title(f"Step {step}")
            axes[idx].set_xlabel('x')
            axes[idx].set_ylabel('y')
            plt.colorbar(im, ax=axes[idx], fraction=0.046)

        if step < n_total:
            solver.step()

    fig3.suptitle(r"Dirac density $\rho$ evolution in vortex-antivortex texture", fontsize=14)
    fig3.tight_layout()
    fig3.savefig("dirac_evolution.png", dpi=150)
    print("  Saved dirac_evolution.png")

    # --- Final combined plot: graph + Dirac density ---
    diag_final = solver.compute_diagnostics()
    delta_final = solver.get_delta()

    fig4 = plot_combined(graph,
                         bond_values=np.ones(graph.n_bonds) * 0.5,
                         atom_values=np.abs(atom_z),
                         dirac_diag=diag_final,
                         delta=delta_final,
                         grid_dx=dx)
    fig4.suptitle("Combined: honeycomb graph + Dirac density response", fontsize=14)
    fig4.savefig("combined_final.png", dpi=150)
    print("  Saved combined_final.png")

    # --- Show all plots ---
    plt.show()

    print("Done.")


def experiment_uniform_kekule():
    """Simple test: uniform Kekule mass, wavepacket propagation.

    This is a sanity check — the wavepacket should propagate
    like a Dirac quasiparticle with mass gap 2*Delta0.
    """
    Nx, Ny = 128, 128
    dx = 1.0
    dt = 0.08

    print("Initializing Dirac solver (uniform Kekule)...")
    solver = Dirac4Solver(Nx=Nx, Ny=Ny, dx=dx, dy=dx, dt=dt, vF=1.0)

    # Uniform Kekule mass
    delta = np.ones((Nx, Ny), dtype=np.complex64) * 0.3
    solver.upload_delta(delta)

    # Wavepacket with momentum
    solver.init_gaussian_packet(
        r0=(Nx * 0.3, Ny * 0.5),
        sigma=6.0,
        k0=(0.3, 0.0),
        chi=(1.0, 0.0, 0.0, 0.0)
    )

    # Run and animate
    print("Running animation...")
    anim = DiracAnimator(solver, figsize=(8, 8))
    anim.run(n_steps=300, interval=30, save_path="dirac_uniform.gif")
    print("Done.")


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 1: Vortex-antivortex pair")
    print("=" * 60)
    experiment_vortex_pair()

    # Uncomment for the uniform Kekule sanity check:
    # print("\n" + "=" * 60)
    # print("Experiment 2: Uniform Kekule (sanity check)")
    # print("=" * 60)
    # experiment_uniform_kekule()
