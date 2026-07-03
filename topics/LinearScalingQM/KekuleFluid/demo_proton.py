"""Interactive demo: H+ protonated defect on a finite graphene flake.

Physical background
-------------------
Protonation of a carbon atom in a PAH molecule (adding H+) removes its p_z
electron from the π system.  In the model this is represented by setting
targetVal=0 and defect=1 on that atom, which suppresses the Kekulé order
parameter locally (defect core) and pins the phase to a chosen Kekulé pattern.
The pinned phase forces a nodal line (where |z|→0) to emanate from the defect,
creating a domain wall between regions of different Kekulé phase.

Role in the system
-------------------
This is an interactive demo script that runs Model A (KekuleFluidSolver) with
one or two H+ defects on a PAH flake and displays a live 6-panel animation:
amplitude |z|, phase arg(z), bond order, HSV phase field + nodal lines, ring
winding numbers, and a closeup of the defect region.

Key functions
-------------
- `run_proton_defect(n_shells, nsteps, defects, save, outdir)` — main function:
    builds PAH, places defects, runs solver with live animation, saves figure.
- CLI arguments: `--n-shells`, `--steps`, `--save`, `--two-defects`,
    `--pin1`, `--pin2`, `--pin-dir`, `--dx`, `--dy`.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from Graph import HoneycombGraph
from ModelA import KekuleFluidSolver
from visualize import HexVisualizer, LiveViewer


def run_proton_defect(n_shells=2, nsteps=600,
                      defects=None, save=False, outdir='output'):
    """Run a finite PAH with one or more H+ defects.

    n_shells: number of ring shells (0=benzene, 1=coronene, 2=circumcoronene, ...)
    defects: list of dicts, each with keys:
        - 'pos': (x, y) or None for auto-select edge atom at angle 'angle'
        - 'angle': angle in radians for auto-select (default 0)
        - 'pin_dir': 0, 1, or 2 — Kekulé pattern to pin
        - 'pin_strength': float (default 1.0)
      If None, uses a single defect at angle 0, pin_dir=0.
      If a single dict, wraps it in a list.
    """
    if defects is None:
        defects = [{'pos': None, 'angle': 0.0, 'pin_dir': 0, 'pin_strength': 1.0}]
    elif isinstance(defects, dict):
        defects = [defects]

    g = HoneycombGraph(aCC=1.0)
    g.build_pah(n_shells=n_shells)
    print(f"PAH (n_shells={n_shells}): {g.natom} atoms, {g.nbond} bonds, {len(g.rings)} rings")

    edge = g.find_edge_atoms()
    print(f"Edge atoms (undercoordinated): {len(edge)}")

    # Find defect atoms
    pos = np.array([[a.pos[0], a.pos[1]] for a in g.atoms])
    angles = np.arctan2(pos[:, 1], pos[:, 0])
    edge_arr = np.array(edge)

    defect_indices = []
    defect_positions = []
    for d_idx, ddef in enumerate(defects):
        if ddef.get('pos') is not None:
            di = g.find_nearest_atom(ddef['pos'][0], ddef['pos'][1])
        else:
            target_angle = ddef.get('angle', 0.0)
            dist = np.abs(np.angle(np.exp(1j * (angles[edge_arr] - target_angle))))
            best = np.argmin(dist)
            di = edge_arr[best]

        dp = g.atoms[di].pos
        pin_dir = ddef.get('pin_dir', 0)
        pin_str = ddef.get('pin_strength', 1.0)
        defect_indices.append(di)
        defect_positions.append(dp.copy())
        print(f"H+ defect {d_idx} on atom {di} at ({dp[0]:.2f}, {dp[1]:.2f})  "
              f"pin_dir={pin_dir} -> pinPhase={g.thetaDir[pin_dir]:.4f} rad "
              f"({np.degrees(g.thetaDir[pin_dir]):.1f} deg)")

    solver = KekuleFluidSolver(g)

    # Start from uniform Kekulé
    solver.init_uniform_kekule(phi0=0.0)

    # Apply H+ defects
    for d_idx, ddef in enumerate(defects):
        pin_dir = ddef.get('pin_dir', 0)
        pin_str = ddef.get('pin_strength', 1.0)
        solver.add_proton_defect(defect_indices[d_idx], pin_dir=pin_dir,
                                  pin_strength=pin_str)

    viz = HexVisualizer(g)

    # --- Live viewer with persistent artists (6 panels) ---
    plt.ion()
    viewer = LiveViewer(viz, n_panels=6)
    viewer.mark_defect(defect_positions)
    pin_dirs = [d.get('pin_dir', 0) for d in defects]
    viewer.fig.suptitle(f"H+ Defects  pin_dirs={pin_dirs}", fontsize=14)

    # Use midpoint of defects for closeup
    defect_center = np.mean(defect_positions, axis=0)

    for step in range(nsteps):
        solver.step()

        if step % 5 == 0 or step == nsteps - 1:
            state = solver.get_state()
            zr, zi, bx = state['z_real'], state['z_imag'], state['bond_x']
            z = zr + 1j * zi

            viewer.update(zr, zi, bx, step=step, defect_pos=defect_center,
                         suptitle=f"H+ Defects  step={step}/{nsteps}  "
                                  f"|z|_mean={np.abs(z).mean():.3f}  pins={pin_dirs}")
            plt.pause(0.01)

            if step % 50 == 0:
                print(f"  step {step:4d}  |z|_mean={np.abs(z).mean():.4f}  "
                      f"|z|_max={np.abs(z).max():.4f}")

    plt.ioff()

    if save:
        os.makedirs(outdir, exist_ok=True)
        nd = len(defects)
        fname = f'proton_defect_pah{g.natom}_n{nd}_pin{"-".join(str(p) for p in pin_dirs)}.png'
        path = os.path.join(outdir, fname)
        viewer.fig.savefig(path, dpi=150)
        print(f"Saved {path}")

    # Print diagnostics
    solver.pull_to_graph()
    print("\n--- Diagnostics ---")
    amp = np.abs(z)
    for d_idx, di in enumerate(defect_indices):
        print(f"Defect {d_idx} atom |z| = {amp[di]:.4f} (should be ~0)")
    print(f"Mean |z| = {amp.mean():.4f}")
    print(f"Max |z| = {amp.max():.4f}")

    # Check valence satisfaction
    val = np.zeros(g.natom)
    for b, bond in enumerate(g.bonds):
        val[bond.iA] += bx[b]
        val[bond.iB] += bx[b]
    for d_idx, di in enumerate(defect_indices):
        print(f"Defect {d_idx} valence = {val[di]:.4f} (target=0.0)")
    print(f"Max valence error = {np.max(np.abs(val - np.array([a.targetVal for a in g.atoms]))):.6f}")

    print("\nDone. Close window to exit.")
    plt.show()
    return solver, viz


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='H+ protonated defect(s) on PAH')
    parser.add_argument('--n-shells', type=int, default=2,
                        help='Ring shells (0=benzene, 1=coronene, 2=circumcoronene, ...)')
    parser.add_argument('--steps', type=int, default=600, help='Number of timesteps')
    parser.add_argument('--save', action='store_true', help='Save final figure')
    parser.add_argument('--outdir', default='output', help='Output directory for images')
    # Two-defect mode: place defects at opposite edges
    parser.add_argument('--two-defects', action='store_true',
                        help='Place two defects at opposite edges (angle 0 and pi)')
    parser.add_argument('--pin1', type=int, choices=[0, 1, 2], default=0,
                        help='Kekulé pattern to pin at defect 1')
    parser.add_argument('--pin2', type=int, choices=[0, 1, 2], default=0,
                        help='Kekulé pattern to pin at defect 2')
    # Single defect options
    parser.add_argument('--pin-dir', type=int, choices=[0, 1, 2], default=0,
                        help='Kekulé pattern to pin at single defect')
    parser.add_argument('--dx', type=float, default=None, help='Defect x position')
    parser.add_argument('--dy', type=float, default=None, help='Defect y position')
    args = parser.parse_args()

    if args.two_defects:
        defects = [
            {'angle': 0.0, 'pin_dir': args.pin1, 'pin_strength': 1.0},
            {'angle': np.pi, 'pin_dir': args.pin2, 'pin_strength': 1.0},
        ]
        run_proton_defect(n_shells=args.n_shells, nsteps=args.steps,
                          defects=defects, save=args.save, outdir=args.outdir)
    else:
        if args.dx is not None:
            defects = [{'pos': (args.dx, args.dy or 0.0), 'pin_dir': args.pin_dir,
                        'pin_strength': 1.0}]
        else:
            defects = None  # auto single defect at angle 0
        run_proton_defect(n_shells=args.n_shells, nsteps=args.steps,
                          defects=defects, save=args.save, outdir=args.outdir)
