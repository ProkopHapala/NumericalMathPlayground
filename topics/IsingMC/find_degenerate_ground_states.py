#!/usr/bin/env python
"""
Search for Ising clusters with a small degenerate ground state manifold (2-4 states)
that is well-separated from all higher excited states.

NO gate voltage is applied. The cluster is evaluated by itself.
We scan the on-site energy E0 (chemical potential for charging) vs coupling W1,
with W2 as a parameter. At certain E0 the ground state becomes degenerate,
making the cluster highly susceptible to small external perturbations.

Output: plots showing regions with good degenerate ground states.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from IsingExactSolver import IsingExactSolver
from Ising_utils import CLUSTERS_POS, scan_E0_W1
from IsingPlotting import plot_scan, plot_best_states, plot_energy_spectrum, plot_energy_spectrum_E0

parser = argparse.ArgumentParser(description='Find degenerate ground states in Ising clusters')
parser.add_argument('--outdir', type=str, default='./results', help='Output directory for plots (default: ./results)')
parser.add_argument('--clusters', type=str, nargs='*', default=None, help='Subset of clusters to scan (default: all)')
parser.add_argument('--W2', type=float, nargs='*', default=[0.0, 0.5, 1.0], help='W2 values to scan')
parser.add_argument('--nE', type=int, default=80, help='Number of E0 scan points')
parser.add_argument('--nW', type=int, default=80, help='Number of W1 scan points')
args = parser.parse_args()

outdir = args.outdir
os.makedirs(outdir, exist_ok=True)
print(f"Output directory: {outdir}")

print("Creating solver...")
solver = IsingExactSolver(preferred_vendor='nvidia', bPrint=False)

nE = args.nE
nW = args.nW
E0_vals = np.linspace(-3.0, 0.5, nE)
W1_vals = np.linspace(0.0, 4.0, nW)
W2_vals_to_scan = args.W2

clusters_to_scan = args.clusters if args.clusters else list(CLUSTERS_POS.keys())

for name in clusters_to_scan:
    if name not in CLUSTERS_POS:
        print(f"  WARNING: Unknown cluster '{name}', skipping")
        continue
    pos = CLUSTERS_POS[name]()
    nSite = len(pos)
    print(f"\n{'='*70}")
    print(f"Cluster: {name} ({nSite} sites)")
    print(f"  positions = {pos.tolist()}")
    print(f"{'='*70}")

    for W2 in W2_vals_to_scan:
        print(f"\n  --- W2 = {W2:.2f} ---")
        n_degen_map, gap_map = scan_E0_W1(solver, pos, E0_vals, W1_vals, W2)

        tag = f'{name}_W2_{W2:.1f}'.replace('.', 'p')
        fname_scan = os.path.join(outdir, f'degeneracy_scan_{tag}.png')
        good_mask = plot_scan(E0_vals, W1_vals, n_degen_map, gap_map, name, W2, fname_scan, pos=pos)

        good_points = np.argwhere(good_mask)
        if len(good_points) > 0:
            scored = [(gap_map[iy, ix], iy, ix) for iy, ix in good_points]
            scored.sort(reverse=True)
            gap_val, iy, ix = scored[0]
            E0_best, W1_best = E0_vals[iy], W1_vals[ix]
            print(f"\n  Best point: E0={E0_best:.4f}, W1={W1_best:.4f}, gap={gap_val:.4f}")
            plot_best_states(solver, pos, E0_best, W1_best, W2,
                             os.path.join(outdir, f'best_degenerate_{tag}.png'))

            W1_fine = np.linspace(max(0, W1_best - 1.0), W1_best + 1.0, 100)
            plot_energy_spectrum(solver, pos, E0_best, W1_fine, W2,
                                os.path.join(outdir, f'spectrum_W1_{tag}.png'))
            E0_fine = np.linspace(E0_best - 1.0, E0_best + 1.0, 100)
            plot_energy_spectrum_E0(solver, pos, E0_fine, W1_best, W2,
                                   os.path.join(outdir, f'spectrum_E0_{tag}.png'))

        plt.close('all')

print(f"\n\nDone! All plots saved to {outdir}")
