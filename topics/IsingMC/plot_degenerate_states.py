#!/usr/bin/env python
"""Plot degenerate ground states for T_extended_output at W1=2.0, W2=1.0."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from IsingExactSolver import IsingExactSolver, sq_lattice_sparse, apply_input_bias, INPUT_COMBOS
from Ising_utils import CLUSTERS_FULL
from IsingPlotting import plot_state_scatter

print("Creating solver...")
solver = IsingExactSolver(preferred_vendor='nvidia', bPrint=False)

pos, inp_pos, inp_neigh, out_site = CLUSTERS_FULL['T_extended_output'](input_pos_type='side')
nSite = len(pos)
Eb = -np.ones(nSite, dtype=np.float32)

W1, W2 = 2.0, 1.0

Esite_batch = np.zeros((4, nSite), dtype=np.float32)
for k, (A, B) in enumerate(INPUT_COMBOS):
    Esite_batch[k] = apply_input_bias(Eb, pos, inp_pos, inp_neigh, [A, B], W1, W2)

W_val, W_idx, nNeigh = sq_lattice_sparse(positions=pos, W1=W1, W2=W2, nSite=nSite)
W_val_batch = np.tile(W_val, (4, 1, 1))
W_idx_batch = np.tile(W_idx, (4, 1, 1))
nNeigh_batch = np.tile(nNeigh, (4, 1))

E_top8, occ_top8 = solver.solve_batch_W_top8(Esite_batch, W_val_batch, W_idx_batch, nNeigh_batch, nSite)

fig, axes = plt.subplots(2, 6, figsize=(24, 8))
fig.suptitle(f'T_extended_output at W1={W1}, W2={W2} - Degenerate Ground States (E=-3.0)', fontsize=16)

for row, k in enumerate([1, 2]):  # input (0,1) and (1,0)
    A, B = INPUT_COMBOS[k]
    for i in range(6):
        ax = axes[row, i]
        occ_mask = occ_top8[k, i]
        occ_array = [(occ_mask >> s) & 1 for s in range(nSite)]
        output_val = (occ_mask >> out_site) & 1
        ax.set_title(f'State {i+1}: out={output_val}\nocc={occ_array}', fontsize=9)
        plot_state_scatter(ax, pos, occ_mask, out_site, inp_pos, inp_neigh)
        ax.set_xlabel(f'Input ({A},{B}): A={A}, B={B}', fontsize=8)

plt.tight_layout()
os.makedirs('results', exist_ok=True)
outpath = os.path.join('results', 'plot_degenerate_states.png')
plt.savefig(outpath, dpi=150)
print(f"Saved: {outpath}")
