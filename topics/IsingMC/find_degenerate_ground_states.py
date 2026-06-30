#!/usr/bin/env python
"""
Search for Ising clusters with a small degenerate ground state manifold (2-4 states)
that is well-separated from all higher excited states.

NO gate voltage is applied. The cluster is evaluated by itself.
We scan the on-site energy E0 (chemical potential for charging) vs coupling W1,
with W2 as a parameter. At certain E0 the ground state becomes degenerate,
making the cluster highly susceptible to small external perturbations.

The Hamiltonian for a single instance (no gating) is:
  H = sum_i n_i * E0 + sum_{i<j} n_i * n_j * W_ij

Output: plots showing regions with good degenerate ground states.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from IsingExactSolver import IsingExactSolver, sq_lattice_sparse

# ==============================================================================
#  Cluster geometry definitions (positions only, no input/output needed)
# ==============================================================================
def make_cluster_T_extended_output():
    return np.array([[0,3],[2,3],[0,2],[1,2],[2,2],[1,1],[1,0],[1,-1]], dtype=int)

def make_cluster_T_extended_inputs():
    return np.array([[0,3],[2,3],[0,2],[2,2],[0,1],[1,1],[2,1],[1,0],[1,-1]], dtype=int)

def make_cluster_cross():
    return np.array([[2,0],[1,1],[2,1],[3,1],[2,2]], dtype=int)

def make_cluster_zigzag():
    return np.array([[0,1],[1,0],[2,1],[3,0],[4,1]], dtype=int)

def make_cluster_T_simple():
    return np.array([[0,1],[1,1],[2,1],[1,0],[1,-1]], dtype=int)

def make_cluster_chain4():
    return np.array([[0,0],[1,0],[2,0],[3,0]], dtype=int)

def make_cluster_chain5():
    return np.array([[0,0],[1,0],[2,0],[3,0],[4,0]], dtype=int)

def make_cluster_L():
    return np.array([[0,0],[1,0],[2,0],[0,1],[1,1],[2,1]], dtype=int)

def make_cluster_S():
    return np.array([[1,2],[2,2],[0,1],[1,1],[0,0],[1,0]], dtype=int)

CLUSTERS = {
    'T_extended_output': make_cluster_T_extended_output,
    'T_extended_inputs': make_cluster_T_extended_inputs,
    'Cross': make_cluster_cross,
    'Zigzag': make_cluster_zigzag,
    'T_simple': make_cluster_T_simple,
    'Chain4': make_cluster_chain4,
    'Chain5': make_cluster_chain5,
    'L': make_cluster_L,
    'S': make_cluster_S,
}

# ==============================================================================
#  Core: evaluate degeneracy of a cluster at given (E0, W1, W2) — no gating
# ==============================================================================
def analyze_degeneracy_no_gate(solver, pos, E0, W1, W2, degeneracy_threshold=0.01):
    """Evaluate the cluster by itself (no gate voltage).
    
    Returns:
        n_degen: number of states within threshold of E0_ground
        gap_to_rest: energy gap from last degenerate state to first excited state
        E_top8: (8,) energies
        occ_top8: (8,) occupancy masks
    """
    nSite = len(pos)
    # All sites have the same on-site energy E0, no input bias
    Esite = np.full((1, nSite), E0, dtype=np.float32)
    W_val, W_idx, nNeigh = sq_lattice_sparse(positions=pos, W1=W1, W2=W2, nSite=nSite)
    W_val_batch = W_val[np.newaxis]
    W_idx_batch = W_idx[np.newaxis]
    nNeigh_batch = nNeigh[np.newaxis]
    E_top8, occ_top8 = solver.solve_batch_W_top8(Esite, W_val_batch, W_idx_batch, nNeigh_batch, nSite)
    E_top8 = E_top8[0]      # shape (8,)
    occ_top8 = occ_top8[0]  # shape (8,)
    
    E0_ground = E_top8[0]
    n_degen = int(np.sum(np.abs(E_top8 - E0_ground) < degeneracy_threshold))
    if n_degen < 8:
        gap_to_rest = float(E_top8[n_degen] - E_top8[n_degen - 1])
    else:
        gap_to_rest = 0.0
    return n_degen, gap_to_rest, E_top8, occ_top8

# ==============================================================================
#  Scan (E0, W1) for fixed W2 values
# ==============================================================================
def scan_E0_W1(solver, pos, E0_vals, W1_vals, W2, degeneracy_threshold=0.01):
    """Scan E0 vs W1 at fixed W2. Returns n_degen_map[iE, iW], gap_map[iE, iW]."""
    nE = len(E0_vals)
    nW = len(W1_vals)
    n_degen_map = np.zeros((nE, nW), dtype=int)
    gap_map = np.zeros((nE, nW), dtype=float)
    for iE, E0 in enumerate(E0_vals):
        for iW, W1 in enumerate(W1_vals):
            n_d, g_r, _, _ = analyze_degeneracy_no_gate(
                solver, pos, E0, W1, W2, degeneracy_threshold)
            n_degen_map[iE, iW] = n_d
            gap_map[iE, iW] = g_r
    return n_degen_map, gap_map

# ==============================================================================
#  Plotting
# ==============================================================================
def plot_scan(E0_vals, W1_vals, n_degen_map, gap_map, cluster_name, W2, fname, pos=None, gate_sites=None, gate_labels=None):
    """Plot the (E0, W1) degeneracy scan with a cluster geometry panel on the side."""
    nE, nW = n_degen_map.shape
    nSite = len(pos) if pos is not None else 0
    
    # Good: 2 <= n_degen <= 4 and gap > 0.3
    good_mask = (n_degen_map >= 2) & (n_degen_map <= 4) & (gap_map > 0.3)
    # Partial: 2 <= n_degen <= 4 and gap > 0.1
    partial_mask = (n_degen_map >= 2) & (n_degen_map <= 4) & (gap_map > 0.1) & ~good_mask
    
    if pos is not None:
        fig = plt.figure(figsize=(20, 6))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.35])
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        ax_cluster = fig.add_subplot(gs[0, 3])
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        ax_cluster = None
    gate_str = f' | gates: {gate_labels}' if gate_sites else ' | No gate voltage'
    fig.suptitle(f'{cluster_name} ({nSite} sites) | W2={W2:.2f}{gate_str}\nScan: E0 (chemical potential) vs W1 (coupling)', fontsize=13)
    
    # Plot 1: number of degenerate states
    ax = axes[0]
    im = ax.imshow(n_degen_map, origin='lower', extent=[W1_vals[0], W1_vals[-1], E0_vals[0], E0_vals[-1]],
                   cmap='YlOrRd', vmin=0, vmax=8, interpolation='nearest', aspect='auto')
    ax.set_title('# degenerate ground states')
    ax.set_xlabel('W1 (nearest-neighbor coupling)')
    ax.set_ylabel('E0 (on-site energy / chemical potential)')
    plt.colorbar(im, ax=ax, label='# degenerate')
    
    # Plot 2: gap to excited states
    ax = axes[1]
    im = ax.imshow(gap_map, origin='lower', extent=[W1_vals[0], W1_vals[-1], E0_vals[0], E0_vals[-1]],
                   cmap='RdYlGn', vmin=0, vmax=2, interpolation='nearest', aspect='auto')
    ax.set_title('Gap to excited states')
    ax.set_xlabel('W1')
    ax.set_ylabel('E0')
    plt.colorbar(im, ax=ax, label='gap')
    
    # Plot 3: good regions
    ax = axes[2]
    img = np.ones((nE, nW, 3)) * 0.9
    # Color by n_degen where good
    for nd in range(2, 5):
        mask = good_mask & (n_degen_map == nd)
        if nd == 2:   img[mask] = [0.0, 0.8, 0.0]  # green
        elif nd == 3: img[mask] = [0.0, 0.6, 1.0]  # blue
        elif nd == 4: img[mask] = [0.8, 0.4, 0.0]  # orange
    img[partial_mask] = [1.0, 0.9, 0.0]  # yellow = partial
    ax.imshow(img, origin='lower', extent=[W1_vals[0], W1_vals[-1], E0_vals[0], E0_vals[-1]],
              interpolation='nearest', aspect='auto')
    ax.set_title('Good regions (2-4 degen, gap>0.3)\nGreen=2, Blue=3, Orange=4, Yellow=partial')
    ax.set_xlabel('W1')
    ax.set_ylabel('E0')
    
    # Cluster geometry panel (one per figure)
    if ax_cluster is not None:
        draw_cluster_panel(ax_cluster, pos, nSite, gate_sites, gate_labels)
    
    good_points = np.argwhere(good_mask)
    if len(good_points) > 0:
        # Sort by gap (largest first)
        scored = [(gap_map[iy, ix], iy, ix) for iy, ix in good_points]
        scored.sort(reverse=True)
        print(f"  Found {len(good_points)} good (E0,W1) points (showing top 10 by gap):")
        for gap_val, iy, ix in scored[:10]:
            print(f"    E0={E0_vals[iy]:.4f}, W1={W1_vals[ix]:.4f}: n_degen={n_degen_map[iy,ix]}, gap={gap_val:.4f}")
    else:
        print(f"  No good points found at W2={W2:.2f}")
    
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")
    return good_mask

def plot_best_states(solver, pos, E0, W1, W2, fname, degeneracy_threshold=0.01):
    """Plot the degenerate ground states for a specific (E0, W1, W2) point."""
    nSite = len(pos)
    n_d, g_r, E_top8, occ_top8 = analyze_degeneracy_no_gate(
        solver, pos, E0, W1, W2, degeneracy_threshold)
    
    print(f"\n  Detailed analysis at E0={E0:.4f}, W1={W1:.4f}, W2={W2:.4f}:")
    print(f"    {n_d} degenerate ground states, gap to rest = {g_r:.4f}")
    for i in range(min(n_d + 3, 8)):
        occ_mask = occ_top8[i]
        occ_array = [(occ_mask >> s) & 1 for s in range(nSite)]
        n_occ = sum(occ_array)
        marker = " *" if i < n_d else ""
        print(f"      State {i+1}: E={E_top8[i]:.6f}, n_occ={n_occ}, occ=0b{occ_mask:0{nSite}b} {occ_array}{marker}")
    
    # Plot: show all degenerate states + 2 excited states
    n_show = min(n_d + 2, 8)
    n_cols = min(n_show, 4)
    n_rows = (n_show + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), squeeze=False)
    fig.suptitle(f'{nSite}-site cluster | E0={E0:.3f}, W1={W1:.3f}, W2={W2:.3f}\n'
                 f'{n_d} degenerate ground states, gap={g_r:.4f}', fontsize=13)
    
    for i in range(n_show):
        ax = axes[i // n_cols, i % n_cols]
        occ_mask = occ_top8[i]
        occ_array = [(occ_mask >> s) & 1 for s in range(nSite)]
        is_ground = i < n_d
        color = 'blue' if is_ground else 'orange'
        label = f'GS {i+1}' if is_ground else f'ES {i+1}'
        ax.set_title(f'{label}: E={E_top8[i]:.4f}, n_occ={sum(occ_array)}', fontsize=10)
        
        for s, (x, y) in enumerate(pos):
            c = color if occ_array[s] else 'lightgray'
            rect = plt.Rectangle((x-0.45, y-0.45), 0.9, 0.9, facecolor=c, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x, y, str(s), ha='center', va='center', fontsize=9,
                   color='white' if occ_array[s] else 'gray', fontweight='bold')
        ax.set_aspect('equal')
        margin = 1.5
        xs, ys = pos[:, 0].min(), pos[:, 0].max()
        ys_, yys = pos[:, 1].min(), pos[:, 1].max()
        ax.set_xlim(xs - margin, xs + yys + margin)
        ax.set_ylim(ys_ - margin, ys_ + yys + margin)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(True, alpha=0.2)
    
    # Hide unused axes
    for i in range(n_show, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis('off')
    
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    print(f"  Saved: {fname}")

def draw_cluster_panel(ax, pos, nSite, gate_sites=None, gate_labels=None):
    """Draw the cluster geometry on a dedicated axes (not an inset).
    
    Args:
        ax: matplotlib axes to draw on
        pos: (nSite, 2) array of site positions
        nSite: number of sites
        gate_sites: list of site indices where gate voltages are applied (optional)
        gate_labels: list of labels for gate sites (e.g. ['A', 'B'])
    """
    gate_set = set(gate_sites) if gate_sites is not None else set()
    for s, (x, y) in enumerate(pos):
        if s in gate_set:
            fc = '#ff4444'  # red for gate sites
        else:
            fc = 'lightblue'
        rect = plt.Rectangle((x-0.4, y-0.4), 0.8, 0.8, facecolor=fc,
                             edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, str(s), ha='center', va='center', fontsize=11, fontweight='bold')
    # Draw gate labels
    if gate_sites is not None and gate_labels is not None:
        for s, label in zip(gate_sites, gate_labels):
            x, y = pos[s]
            ax.annotate(label, (x, y + 0.6), fontsize=12, fontweight='bold',
                       color='red', ha='center', va='bottom')
    ax.set_aspect('equal')
    margin = 1.0
    xs0, xs1 = pos[:, 0].min() - margin, pos[:, 0].max() + margin
    ys0, ys1 = pos[:, 1].min() - margin, pos[:, 1].max() + margin
    ax.set_xlim(xs0, xs1)
    ax.set_ylim(ys0, ys1)
    ax.set_xticks([]); ax.set_yticks([])
    title = f'{nSite} sites'
    if gate_set:
        title += f'\ngates: {gate_labels}'
    ax.set_title(title, fontsize=11)

def plot_energy_spectrum(solver, pos, E0_fixed, W1_vals, W2, fname):
    """Plot how the energy spectrum (top 8 states) changes with W1.
    Cluster geometry shown as a side panel."""
    nSite = len(pos)
    nW = len(W1_vals)
    E_all = np.zeros((nW, 8), dtype=float)
    for iW, W1 in enumerate(W1_vals):
        _, _, E_top8, _ = analyze_degeneracy_no_gate(solver, pos, E0_fixed, W1, W2)
        E_all[iW] = E_top8
    
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.2])
    ax = fig.add_subplot(gs[0, 0])
    ax_cluster = fig.add_subplot(gs[0, 1])
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    for k in range(8):
        ax.plot(W1_vals, E_all[:, k], '-', color=colors[k], label=f'State {k+1}',
                linewidth=1.8 if k < 4 else 1.0, alpha=0.9 if k < 4 else 0.6)
    ax.set_xlabel('W1 (nearest-neighbor coupling)')
    ax.set_ylabel('Energy')
    ax.set_title(f'Energy spectrum vs W1 | E0={E0_fixed:.3f}, W2={W2:.3f} | {nSite} sites')
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)
    draw_cluster_panel(ax_cluster, pos, nSite)
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")

def plot_energy_spectrum_E0(solver, pos, E0_vals, W1, W2, fname):
    """Plot how the energy spectrum changes with E0 (chemical potential).
    Cluster geometry shown as a side panel."""
    nSite = len(pos)
    nE = len(E0_vals)
    E_all = np.zeros((nE, 8), dtype=float)
    for iE, E0 in enumerate(E0_vals):
        _, _, E_top8, _ = analyze_degeneracy_no_gate(solver, pos, E0, W1, W2)
        E_all[iE] = E_top8
    
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.2])
    ax = fig.add_subplot(gs[0, 0])
    ax_cluster = fig.add_subplot(gs[0, 1])
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    for k in range(8):
        ax.plot(E0_vals, E_all[:, k], '-', color=colors[k], label=f'State {k+1}',
                linewidth=1.8 if k < 4 else 1.0, alpha=0.9 if k < 4 else 0.6)
    ax.set_xlabel('E0 (on-site energy / chemical potential)')
    ax.set_ylabel('Energy')
    ax.set_title(f'Energy spectrum vs E0 | W1={W1:.3f}, W2={W2:.3f} | {nSite} sites')
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)
    draw_cluster_panel(ax_cluster, pos, nSite)
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")

# ==============================================================================
#  Main
# ==============================================================================
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

# Scan parameters
nE = args.nE
nW = args.nW
E0_vals = np.linspace(-3.0, 0.5, nE)   # on-site energy (chemical potential)
W1_vals = np.linspace(0.0, 4.0, nW)    # nearest-neighbor coupling
W2_vals_to_scan = args.W2

clusters_to_scan = args.clusters if args.clusters else list(CLUSTERS.keys())

for name in clusters_to_scan:
    if name not in CLUSTERS:
        print(f"  WARNING: Unknown cluster '{name}', skipping")
        continue
    pos = CLUSTERS[name]()
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
        
        # Plot best degenerate state in detail
        good_points = np.argwhere(good_mask)
        if len(good_points) > 0:
            scored = [(gap_map[iy, ix], iy, ix) for iy, ix in good_points]
            scored.sort(reverse=True)
            gap_val, iy, ix = scored[0]
            E0_best, W1_best = E0_vals[iy], W1_vals[ix]
            print(f"\n  Best point: E0={E0_best:.4f}, W1={W1_best:.4f}, gap={gap_val:.4f}")
            plot_best_states(solver, pos, E0_best, W1_best, W2,
                             os.path.join(outdir, f'best_degenerate_{tag}.png'))
            
            # Also plot energy spectrum near this point
            W1_fine = np.linspace(max(0, W1_best - 1.0), W1_best + 1.0, 100)
            plot_energy_spectrum(solver, pos, E0_best, W1_fine, W2,
                                os.path.join(outdir, f'spectrum_W1_{tag}.png'))
            E0_fine = np.linspace(E0_best - 1.0, E0_best + 1.0, 100)
            plot_energy_spectrum_E0(solver, pos, E0_fine, W1_best, W2,
                                   os.path.join(outdir, f'spectrum_E0_{tag}.png'))
        
        plt.close('all')  # prevent memory buildup

print(f"\n\nDone! All plots saved to {outdir}")
