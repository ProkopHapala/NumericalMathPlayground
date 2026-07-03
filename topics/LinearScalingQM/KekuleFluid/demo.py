"""Interactive demo: run the Kekulé fluid solver and watch it evolve live.

Physical background
-------------------
This demo shows the spontaneous relaxation of the Kekulé order parameter z_i
from various initial conditions on a rectangular honeycomb patch.  Experiments:
  - 'vortex': vortex-antivortex pair initial condition — shows annihilation.
  - 'edge':   phase gradient from edge pinning — shows domain wall formation.
  - 'relax':  random noise — shows spontaneous symmetry breaking and ordering.

Role in the system
------------------
Simple interactive demo with a 4-panel live animation (amplitude, phase, bond
order, HSV phase + nodal lines).  Uses rectangular patches (build_rect_patch)
rather than PAH flakes.  Good for quick visual exploration of Model A dynamics.

Key functions
-------------
- `run_interactive(nx, ny, nsteps, experiment)` — main function.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from Graph import HoneycombGraph
from ModelA import KekuleFluidSolver
from visualize import HexVisualizer, LiveViewer


def run_interactive(nx=8, ny=8, nsteps=500, experiment='vortex'):
    """Run solver with live plotting. Close the window to exit.

    experiment: 'vortex', 'edge', 'relax'
    """
    g = HoneycombGraph(aCC=1.0)
    g.build_rect_patch(nx, ny)
    print(f"Graph: {g.natom} atoms, {g.nbond} bonds, {len(g.rings)} rings")

    solver = KekuleFluidSolver(g)
    viz = HexVisualizer(g)

    # --- Initialize ---
    pos = np.array([[a.pos[0], a.pos[1]] for a in g.atoms])
    xc = 0.5 * (pos[:, 0].min() + pos[:, 0].max())
    yc = 0.5 * (pos[:, 1].min() + pos[:, 1].max())
    dx = 0.25 * (pos[:, 0].max() - pos[:, 0].min())

    if experiment == 'vortex':
        solver.init_vortices([(xc - dx, yc, +1), (xc + dx, yc, -1)])
        title = "Vortex-Antivortex Pair"
    elif experiment == 'edge':
        left = np.where(pos[:, 0] < pos[:, 0].min() + 0.3)[0]
        p = left[np.argmin(pos[left, 1])]
        right = np.where(pos[:, 0] > pos[:, 0].max() - 0.3)[0]
        q = right[np.argmax(pos[right, 1])]
        solver.set_defect(p, defect=0.5, targetVal=0.5)
        solver.set_pin(p, pinStrength=1.0, dir=0)
        solver.set_pin(q, pinStrength=1.0, dir=1)
        solver.init_uniform_kekule(phi0=0.0)
        solver.set_defect(p, defect=0.5, targetVal=0.5)
        solver.set_pin(p, pinStrength=1.0, dir=0)
        solver.set_pin(q, pinStrength=1.0, dir=1)
        title = "Edge-Pinned Defect"
    else:
        z = 0.1 * (np.random.randn(g.natom) + 1j * np.random.randn(g.natom))
        solver.set_z(z.real.astype(np.float32), z.imag.astype(np.float32))
        solver.zToRawBonds()
        solver.copyRawToX()
        solver.projectBondOrders()
        solver.bondsToZ()
        title = "Random Noise Relaxation"

    # --- Live viewer with persistent artists ---
    plt.ion()
    viewer = LiveViewer(viz, n_panels=4)
    viewer.fig.suptitle(title, fontsize=14)

    for step in range(nsteps):
        solver.step()

        if step % 5 == 0 or step == nsteps - 1:
            state = solver.get_state()
            zr, zi, bx = state['z_real'], state['z_imag'], state['bond_x']
            z = zr + 1j * zi

            viewer.update(zr, zi, bx, step=step,
                         suptitle=f'{title}  step {step}/{nsteps}  |z|_mean={np.abs(z).mean():.3f}')
            plt.pause(0.01)

            if step % 50 == 0:
                print(f"  step {step:4d}  |z|_mean={np.abs(z).mean():.4f}  "
                      f"|z|_max={np.abs(z).max():.4f}")

    plt.ioff()
    print(f"\nDone. Close window to exit.")
    plt.show()
    return solver, viz


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Interactive Kekulé fluid demo')
    parser.add_argument('--exp', choices=['vortex', 'edge', 'relax'],
                        default='vortex', help='Experiment type')
    parser.add_argument('--nx', type=int, default=8, help='Grid cells in x')
    parser.add_argument('--ny', type=int, default=8, help='Grid cells in y')
    parser.add_argument('--steps', type=int, default=500, help='Number of timesteps')
    args = parser.parse_args()

    run_interactive(nx=args.nx, ny=args.ny, nsteps=args.steps, experiment=args.exp)
