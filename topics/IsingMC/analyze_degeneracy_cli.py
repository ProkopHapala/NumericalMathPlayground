#!/usr/bin/env python
"""
CLI tool to analyze MQCA degeneracy for different cluster geometries and parameters.

Usage:
    python analyze_degeneracy_cli.py --cluster T_extended_output --W1 2.0 --W2 0.0
    python analyze_degeneracy_cli.py --cluster T_extended_inputs --W1 2.0 --W2 1.0 --input-pos side
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from IsingExactSolver import IsingExactSolver, sq_lattice_sparse, apply_input_bias, INPUT_COMBOS
from Ising_utils import CLUSTERS_FULL
from IsingPlotting import plot_tetris_state

# Subset of clusters that support input_pos_type parameter
CLUSTER_CHOICES = ['T_extended_output', 'T_extended_inputs', 'T_simple']

def main():
    parser = argparse.ArgumentParser(description='Analyze MQCA degeneracy')
    parser.add_argument('--cluster', type=str, default='T_extended_output',
                       choices=CLUSTER_CHOICES,
                       help='Cluster geometry')
    parser.add_argument('--W1', type=float, default=2.0, help='Cartesian coupling')
    parser.add_argument('--W2', type=float, default=0.0, help='Diagonal coupling')
    parser.add_argument('--input-pos', type=str, default='above',
                       choices=['above', 'side'],
                       help='Input position: above sites or from sides')
    parser.add_argument('--shift', type=float, default=0.0,
                       help='Shift applied to input values (0.0: 0→0,1→1; -0.5: 0→-0.5,1→+0.5; -1.0: 0→-1,1→0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename (default: results/<cluster>_degenerate.png)')
    args = parser.parse_args()
    
    print(f"Creating solver...")
    solver = IsingExactSolver(preferred_vendor='nvidia', bPrint=False)
    
    print(f"\n=== Configuration ===")
    print(f"Cluster: {args.cluster}")
    print(f"W1: {args.W1}, W2: {args.W2}")
    print(f"Input positions: {args.input_pos}")
    print(f"Shift: {args.shift}")
    
    pos, inp_pos, inp_neigh, out_site = CLUSTERS_FULL[args.cluster](input_pos_type=args.input_pos)
    nSite = len(pos)
    Eb = -np.ones(nSite, dtype=np.float32)
    
    print(f"Number of sites: {nSite}")
    print(f"Positions:\n{pos}")
    print(f"Input positions: {inp_pos}")
    print(f"Input neighbors: {inp_neigh}")
    print(f"Output site: {out_site}")
    
    Esite_batch = np.zeros((4, nSite), dtype=np.float32)
    for k, (A, B) in enumerate(INPUT_COMBOS):
        Esite_batch[k] = apply_input_bias(Eb, pos, inp_pos, inp_neigh, [A, B], args.W1, args.W2, args.shift)
    
    W_val, W_idx, nNeigh = sq_lattice_sparse(positions=pos, W1=args.W1, W2=args.W2, nSite=nSite)
    W_val_batch = np.tile(W_val, (4, 1, 1))
    W_idx_batch = np.tile(W_idx, (4, 1, 1))
    nNeigh_batch = np.tile(nNeigh, (4, 1))
    
    E_top8, occ_top8 = solver.solve_batch_W_top8(Esite_batch, W_val_batch, W_idx_batch, nNeigh_batch, nSite)
    
    print(f"\n=== Degeneracy Analysis ===")
    for k, (A, B) in enumerate(INPUT_COMBOS):
        ground_gap = E_top8[k, 1] - E_top8[k, 0]
        status = "DEGENERATE" if ground_gap < 0.01 else "OK"
        print(f"Input ({A},{B}): gap={ground_gap:.4f} [{status}]")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'{args.cluster} at W1={args.W1}, W2={args.W2} (inputs {args.input_pos})', fontsize=14)
    
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
            ax.set_title(f'In({A},{B}) State {i+1}: E={E_top8[k,i]:.3f}, out={output_val}{degenerate_str}', fontsize=9)
            plot_tetris_state(ax, pos, occ_mask, out_site, inp_pos)
        
        if n_states_to_plot == 1:
            axes[1, k].axis('off')
    
    plt.tight_layout()
    outpath = args.output
    if outpath is None:
        os.makedirs('results', exist_ok=True)
        outpath = os.path.join('results', f'{args.cluster}_degenerate.png')
    plt.savefig(outpath, dpi=150)
    print(f"\nSaved: {outpath}")

if __name__ == '__main__':
    main()
