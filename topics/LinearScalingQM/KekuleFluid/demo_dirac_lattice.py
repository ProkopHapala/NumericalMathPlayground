#!/usr/bin/env python3
"""demo_dirac_lattice.py — Interactive live animation of the lattice Dirac solver.

Physical background
-------------------
Shows the full causal chain in real time: Model A (Kekulé fluid) relaxes to a
defect-pinned texture, then Model B (tight-binding quasiparticle) propagates a
Gaussian wavepacket through the resulting Kekulé-modulated hopping landscape.
The wavepacket scatters off the defect and the nodal line, revealing the
influence of the Kekulé mass gap on quasiparticle dynamics.

Role in the system
------------------
Interactive demo that runs both Model A and Model B on the same honeycomb flake
with live side-by-side animation.  Uses build_flake (circular/hexagonal) rather
than build_pah.  Good for visualizing the coupling between the two models.

Key functions
-------------
- `run_interactive(radius, pin_dir, nsteps_A, nsteps_B)` — main function.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Graph import HoneycombGraph
from ModelA import KekuleFluidSolver
from DiracLattice_ocl import DiracLatticeSolver
from visualize import HexVisualizer
from matplotlib.collections import LineCollection


def run_interactive(radius=7, pin_dir=1, nsteps_A=300, nsteps_B=600):
    g = HoneycombGraph(aCC=1.0)
    g.build_flake(radius=radius, shape='hex')
    print(f"Flake: {g.natom} atoms, {g.nbond} bonds, {len(g.rings)} rings")

    edge = g.find_edge_atoms()
    pos = np.array([[a.pos[0], a.pos[1]] for a in g.atoms])
    angles = np.arctan2(pos[:, 1], pos[:, 0])
    edge_arr = np.array(edge)
    best = np.argmin(np.abs(angles[edge_arr] - 0.0))
    defect_idx = edge_arr[best]
    dp = pos[defect_idx]
    print(f"H+ defect on atom {defect_idx} at ({dp[0]:.2f}, {dp[1]:.2f})")

    # --- Model A ---
    solverA = KekuleFluidSolver(g)
    solverA.init_uniform_kekule(phi0=0.0)
    solverA.add_proton_defect(defect_idx, pin_dir=pin_dir, pin_strength=1.0)

    # --- Model B ---
    solverB = DiracLatticeSolver(g, t0=1.0, dt_kek=0.5, dt=0.05)

    viz = HexVisualizer(g)
    bond_segs = [[pos[b.iA], pos[b.iB]] for b in g.bonds]

    xc = pos[:, 0].mean()
    yc = pos[:, 1].mean()

    plt.ion()
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Lattice Dirac solver — live", fontsize=14)
    fig.canvas.draw()

    phase = 'A'  # 'A' = Model A relaxing, 'B' = Model B propagating
    step = 0

    for frame in range(nsteps_A + nsteps_B):
        if phase == 'A':
            solverA.step()
            step += 1
            if step >= nsteps_A:
                # Switch to Model B
                phase = 'B'
                step = 0
                zr, zi = solverA.get_z()
                bx = solverA.get_bond_x()
                solverB.update_bond_x(bx)
                defect_arr = np.array([a.defect for a in g.atoms], dtype=np.float32)
                solverB.update_defect(defect_arr * 2.0)
                solverB.init_gaussian_at_pos(xc, yc, sigma=3.0)
                print("Switching to Model B (quasiparticle propagation)")
        else:
            solverB.step()
            step += 1

        if frame % 3 == 0 or frame == nsteps_A + nsteps_B - 1:
            # Clear all axes
            for ax in fig.get_axes():
                ax.remove()
            axes = fig.subplots(2, 3)

            if phase == 'A':
                zr, zi = solverA.get_z()
                bx = solverA.get_bond_x()
                z = zr + 1j * zi
                title_prefix = f"Model A step {step}/{nsteps_A}"
            else:
                zr, zi = solverA.get_z()
                bx = solverA.get_bond_x()
                z = zr + 1j * zi
                title_prefix = f"Model B step {step}/{nsteps_B}"

            # Row 0: Model A fields (static once phase B starts)
            viz.plot_z_field(zr, zi, mode='amplitude', ax=axes[0, 0],
                             bond_x=bx, title=f"|z| (Kekule)", add_colorbar=True)
            axes[0, 0].plot(dp[0], dp[1], 'r*', markersize=15, zorder=10)

            viz.plot_z_field(zr, zi, mode='phase', ax=axes[0, 1],
                             bond_x=bx, title=f"arg(z)", add_colorbar=True)
            axes[0, 1].plot(dp[0], dp[1], 'r*', markersize=15, zorder=10)

            viz.plot_bond_order(bx, ax=axes[0, 2],
                                title="Bond order", add_colorbar=True)

            # Row 1: Model B quasiparticle
            if phase == 'B':
                pr, pi = solverB.get_psi()
                density = pr**2 + pi**2
                norm = density.sum()
            else:
                pr = np.zeros(g.natom, dtype=np.float32)
                pi = np.zeros(g.natom, dtype=np.float32)
                density = np.zeros(g.natom, dtype=np.float32)
                norm = 0.0

            viz.plot_z_field(pr, pi, mode='amplitude', ax=axes[1, 0],
                             bond_x=bx, title=f"|psi|", add_colorbar=True)
            axes[1, 0].plot(dp[0], dp[1], 'r*', markersize=15, zorder=10)

            viz.plot_z_field(pr, pi, mode='phase', ax=axes[1, 1],
                             bond_x=bx, title=f"arg(psi)", add_colorbar=True)
            axes[1, 1].plot(dp[0], dp[1], 'r*', markersize=15, zorder=10)

            # Density
            sc = axes[1, 2].scatter(pos[:, 0], pos[:, 1], c=density,
                                    cmap='hot', s=80, edgecolors='black',
                                    linewidths=0.3, vmin=0, vmax=max(density.max(), 0.001))
            lc = LineCollection(bond_segs, colors='gray', linewidths=0.5, alpha=0.3, zorder=0)
            axes[1, 2].add_collection(lc)
            axes[1, 2].set_aspect('equal')
            axes[1, 2].set_title(f"|psi|^2  (norm={norm:.4f})")
            axes[1, 2].plot(dp[0], dp[1], 'r*', markersize=15, zorder=10)
            fig.colorbar(sc, ax=axes[1, 2], label='density')

            fig.suptitle(f"{title_prefix}  |  pin_dir={pin_dir}", fontsize=12)
            fig.canvas.draw_idle()
            plt.pause(0.01)

            if step % 50 == 0:
                if phase == 'A':
                    print(f"  A step {step:4d}  |z|_mean={np.abs(z).mean():.4f}")
                else:
                    print(f"  B step {step:4d}  norm={norm:.6f}  rho_max={density.max():.6f}")

    plt.ioff()
    print("\nDone. Close window to exit.")
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Interactive lattice Dirac demo')
    parser.add_argument('--radius', type=float, default=7, help='Flake radius')
    parser.add_argument('--pin-dir', type=int, choices=[0, 1, 2], default=1)
    parser.add_argument('--steps-a', type=int, default=300, help='Model A steps')
    parser.add_argument('--steps-b', type=int, default=600, help='Model B steps')
    args = parser.parse_args()

    run_interactive(radius=args.radius, pin_dir=args.pin_dir,
                    nsteps_A=args.steps_a, nsteps_B=args.steps_b)
