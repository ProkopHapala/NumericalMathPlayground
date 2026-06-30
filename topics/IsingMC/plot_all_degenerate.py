#!/usr/bin/env python
"""Plot all 4 input combinations with degenerate states using Tetris view mode."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from IsingExactSolver import IsingExactSolver, sq_lattice_sparse, apply_input_bias, INPUT_COMBOS
from Ising_utils import CLUSTERS_FULL
from IsingPlotting import plot_tetris_state

print("Creating solver...")
solver = IsingExactSolver(preferred_vendor='nvidia', bPrint=False)

pos, inp_pos, inp_neigh, out_site = CLUSTERS_FULL['T_extended_output'](input_pos_type='above')
nSite = len(pos)
Eb = -np.ones(nSite, dtype=np.float32)

W1, W2 = 0.0, 0.0

Esite_batch = np.zeros((4, nSite), dtype=np.float32)
for k, (A, B) in enumerate(INPUT_COMBOS):
    Esite_batch[k] = apply_input_bias(Eb, pos, inp_pos, inp_neigh, [A, B], W1, W2)

W_val, W_idx, nNeigh = sq_lattice_sparse(positions=pos, W1=W1, W2=W2, nSite=nSite)
W_val_batch = np.tile(W_val, (4, 1, 1))
W_idx_batch = np.tile(W_idx, (4, 1, 1))
nNeigh_batch = np.tile(nNeigh, (4, 1))

E_top8, occ_top8 = solver.solve_batch_W_top8(Esite_batch, W_val_batch, W_idx_batch, nNeigh_batch, nSite)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle(f'T_extended_output at W1={W1}, W2={W2} - All Input Combinations with Degenerate States (Tetris view)', fontsize=14)

for k, (A, B) in enumerate(INPUT_COMBOS):
    ground_gap = E_top8[k, 1] - E_top8[k, 0]
    if ground_gap < 0.01:
        n_states_to_plot = 2
        degenerate_str = " (DEGENERATE)"
    else:
        n_states_to_plot = 1
        degenerate_str = ""
    
    for i in range(n_states_to_plot):
        ax = axes[i, k]
        occ_mask = occ_top8[k, i]
        output_val = (occ_mask >> out_site) & 1
        ax.set_title(f'In({A},{B}) State {i+1}: E={E_top8[k,i]:.3f}, out={output_val}{degenerate_str}', 
                    fontsize=9)
        plot_tetris_state(ax, pos, occ_mask, out_site, inp_pos, inp_neigh)
    
    if n_states_to_plot == 1:
        axes[1, k].axis('off')

plt.tight_layout()
os.makedirs('results', exist_ok=True)
outpath = os.path.join('results', 'plot_all_degenerate_tetris.png')
plt.savefig(outpath, dpi=150)
print(f"Saved: {outpath}")
