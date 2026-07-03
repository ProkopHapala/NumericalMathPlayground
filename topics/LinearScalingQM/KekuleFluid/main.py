"""Main entry point: run Kekulé fluid experiments (batch mode, no animation).

Physical background
-------------------
Batch experiments exploring Model A dynamics on rectangular honeycomb patches:
  1. Vortex-antivortex pair relaxation — shows annihilation and phase healing.
  2. Edge-pinned phase gradient — shows domain wall formation.
  3. Spontaneous relaxation from noise — shows Z₃ symmetry breaking.

Role in the system
------------------
Older batch experiment script.  Produces saved figures but no live animation.
Superseded by demo.py for interactive use and run_dirac_lattice.py for
production-quality output with PAH flakes and Model B coupling.

Key functions
-------------
- `experiment_vortex_pair(...)` — vortex-antivortex pair.
- `experiment_edge_pin(...)` — edge-pinned phase gradient.
- `experiment_relax(...)` — spontaneous relaxation from noise.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from Graph import HoneycombGraph
from ModelA import KekuleFluidSolver
from visualize import HexVisualizer


def experiment_vortex_pair(nx=8, ny=8, nsteps=500, save=False):
    """Experiment 1: vortex–antivortex pair relaxation."""
    print("=== Experiment 1: Vortex-Antivortex Pair ===")

    graph = HoneycombGraph(aCC=1.0)
    graph.build_rect_patch(nx, ny)
    print(f"Graph: {graph.natom} atoms, {graph.nbond} bonds, {len(graph.rings)} rings")

    solver = KekuleFluidSolver(graph)

    # Find approximate center and extents
    pos = np.array([[a.pos[0], a.pos[1]] for a in graph.atoms])
    xmin, xmax = pos[:, 0].min(), pos[:, 0].max()
    ymin, ymax = pos[:, 1].min(), pos[:, 1].max()
    xc = 0.5 * (xmin + xmax)
    yc = 0.5 * (ymin + ymax)
    dx = 0.25 * (xmax - xmin)

    vortices = [
        (xc - dx, yc, +1),
        (xc + dx, yc, -1),
    ]
    print(f"Vortex positions: {vortices}")

    solver.init_vortices(vortices)

    viz = HexVisualizer(graph)

    # Capture frames
    frames = []

    def callback(slv, step):
        state = slv.get_state()
        frames.append((step, state))
        if step % 50 == 0:
            z = state['z_real'] + 1j * state['z_imag']
            amp = np.abs(z)
            print(f"  step {step:4d}  |z|_mean={amp.mean():.4f}  |z|_max={amp.max():.4f}")

    solver.run(nsteps, callback=callback, callback_interval=5)

    # Final plots
    state = solver.get_state()
    z_real = state['z_real']
    z_imag = state['z_imag']
    bond_x = state['bond_x']

    fig, axes = viz.plot_combined(z_real, z_imag, bond_x)
    fig.suptitle(f'Vortex-Antivortex Pair after {nsteps} steps', fontsize=14)
    if save:
        fig.savefig('vortex_pair_final.png', dpi=150)
    plt.show()

    # Animation
    print("Creating animation...")
    anim = viz._animate_combined(frames, interval=80, save_path='vortex_pair.gif' if save else None)

    return solver, viz, frames


def experiment_edge_pin(nx=8, ny=8, nsteps=500, save=False):
    """Experiment 2: edge-pinned protonated defect."""
    print("\n=== Experiment 2: Edge-Pinned Defect ===")

    graph = HoneycombGraph(aCC=1.0)
    graph.build_rect_patch(nx, ny)
    print(f"Graph: {graph.natom} atoms, {graph.nbond} bonds, {len(graph.rings)} rings")

    pos = np.array([[a.pos[0], a.pos[1]] for a in graph.atoms])
    xmin, xmax = pos[:, 0].min(), pos[:, 0].max()
    ymin, ymax = pos[:, 1].min(), pos[:, 1].max()

    # Find leftmost edge atom
    left_mask = pos[:, 0] < xmin + 0.3
    left_atoms = np.where(left_mask)[0]
    p = left_atoms[np.argmin(pos[left_atoms, 1])]

    # Find rightmost edge atom
    right_mask = pos[:, 0] > xmax - 0.3
    right_atoms = np.where(right_mask)[0]
    q = right_atoms[np.argmax(pos[right_atoms, 1])]

    print(f"Defect atom p={p} at {pos[p]}, pin atom q={q} at {pos[q]}")

    solver = KekuleFluidSolver(graph)

    # Set up defects on graph before init
    solver.set_defect(p, defect=0.5, targetVal=0.5)
    solver.set_pin(p, pinStrength=1.0, dir=0)
    solver.set_pin(q, pinStrength=1.0, dir=1)

    solver.init_uniform_kekule(phi0=0.0)

    # Re-apply pins/defects after init (init may reset)
    solver.set_defect(p, defect=0.5, targetVal=0.5)
    solver.set_pin(p, pinStrength=1.0, dir=0)
    solver.set_pin(q, pinStrength=1.0, dir=1)

    viz = HexVisualizer(graph)

    frames = []

    def callback(slv, step):
        state = slv.get_state()
        frames.append((step, state))
        if step % 50 == 0:
            z = state['z_real'] + 1j * state['z_imag']
            amp = np.abs(z)
            print(f"  step {step:4d}  |z|_mean={amp.mean():.4f}  |z|_max={amp.max():.4f}")

    solver.run(nsteps, callback=callback, callback_interval=5)

    state = solver.get_state()
    fig, axes = viz.plot_combined(state['z_real'], state['z_imag'], state['bond_x'])
    fig.suptitle(f'Edge-Pinned Defect after {nsteps} steps', fontsize=14)
    if save:
        fig.savefig('edge_pin_final.png', dpi=150)
    plt.show()

    print("Creating animation...")
    anim = viz._animate_combined(frames, interval=80, save_path='edge_pin.gif' if save else None)

    return solver, viz, frames


def experiment_relaxation(nx=8, ny=8, nsteps=300, save=False):
    """Experiment 3: random noise relaxation to Kekulé ground state."""
    print("\n=== Experiment 3: Random Noise Relaxation ===")

    graph = HoneycombGraph(aCC=1.0)
    graph.build_rect_patch(nx, ny)
    print(f"Graph: {graph.natom} atoms, {graph.nbond} bonds, {len(graph.rings)} rings")

    solver = KekuleFluidSolver(graph)

    # Start from random noise
    z = 0.1 * (np.random.randn(graph.natom) + 1j * np.random.randn(graph.natom))
    solver.set_z(z.real.astype(np.float32), z.imag.astype(np.float32))
    solver.zToRawBonds()
    solver.copyRawToX()
    solver.projectBondOrders()
    solver.bondsToZ()

    viz = HexVisualizer(graph)
    frames = []

    def callback(slv, step):
        state = slv.get_state()
        frames.append((step, state))
        if step % 30 == 0:
            z = state['z_real'] + 1j * state['z_imag']
            amp = np.abs(z)
            print(f"  step {step:4d}  |z|_mean={amp.mean():.4f}  |z|_max={amp.max():.4f}")

    solver.run(nsteps, callback=callback, callback_interval=5)

    state = solver.get_state()
    fig, axes = viz.plot_combined(state['z_real'], state['z_imag'], state['bond_x'])
    fig.suptitle(f'Random Noise Relaxation after {nsteps} steps', fontsize=14)
    if save:
        fig.savefig('relaxation_final.png', dpi=150)
    plt.show()

    print("Creating animation...")
    anim = viz._animate_combined(frames, interval=80, save_path='relaxation.gif' if save else None)

    return solver, viz, frames


if __name__ == '__main__':
    # Run all experiments
    solver1, viz1, frames1 = experiment_vortex_pair(nx=8, ny=8, nsteps=400, save=True)
    solver2, viz2, frames2 = experiment_edge_pin(nx=8, ny=8, nsteps=400, save=True)
    solver3, viz3, frames3 = experiment_relaxation(nx=8, ny=8, nsteps=300, save=True)
