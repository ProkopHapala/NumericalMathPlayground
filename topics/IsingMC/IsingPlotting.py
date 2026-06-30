"""
IsingPlotting.py — All matplotlib visualization functions for IsingMC.

Functions:
  - draw_cluster_panel: cluster geometry side panel (site indices + optional gate sites)
  - plot_cluster: single cluster state (scatter or tetris style)
  - plot_ground_states: 2x2 ground states for 4 input combinations
  - plot_logic_map: 2D logic phase diagram
  - plot_logic_fraction_map: useful logic regions map
  - plot_scan: 3-panel degeneracy scan map + cluster panel
  - plot_best_states: degenerate state grid visualization
  - plot_energy_spectrum: energy spectrum vs W1 with cluster panel
  - plot_energy_spectrum_E0: energy spectrum vs E0 with cluster panel
  - plot_tetris_state: Tetris-style single state visualization
  - plot_state_scatter: scatter-based state visualization
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from IsingExactSolver import (
    INPUT_COMBOS, LOGIC_NAMES, USEFUL_LOGIC,
    occ_mask_to_array,
)

# Assign a distinct colour to each of the 16 logic codes
_LOGIC_CMAP = matplotlib.colormaps.get_cmap('tab20')
LOGIC_COLORS = {code: _LOGIC_CMAP(code / 15.0) for code in range(16)}


# =====================================================================
#  Cluster geometry panel (side panel, not inset)
# =====================================================================

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
            fc = '#ff4444'
        else:
            fc = 'lightblue'
        rect = plt.Rectangle((x-0.4, y-0.4), 0.8, 0.8, facecolor=fc,
                             edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, str(s), ha='center', va='center', fontsize=11, fontweight='bold')
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


# =====================================================================
#  Single cluster state visualization
# =====================================================================

def plot_cluster(ax, positions, occ, input_positions, output_site,
                 W_val=None, W_idx=None, nNeigh=None, title='', tetris_style=False,
                 input_values=None, Esite=None):
    """Draw one cluster configuration.

    positions       : (nSite,2) active site grid coords
    occ             : (nSite,) occupancy 0/1
    input_positions : list of (x,y) for input pads (not active)
    output_site     : int index of output site
    tetris_style    : if True, use large adjacent squares like Tetris blocks
    input_values    : (2,) int array [A, B] values 0 or 1 for each input pad
    """
    pos = np.array(positions, dtype=float)
    nSite = len(pos)
    if input_values is None:
        input_values = [0, 0]

    if tetris_style:
        if len(pos) > 1:
            dxs = np.abs(pos[:,0:1] - pos[:,0:1].T)
            dys = np.abs(pos[:,1:2] - pos[:,1:2].T)
            dxs = dxs[dxs > 0.1]
            dys = dys[dys > 0.1]
            cell = min(np.median(dxs) if len(dxs) > 0 else 1.0,
                       np.median(dys) if len(dys) > 0 else 1.0)
        else:
            cell = 1.0

        color_on = '#c44e4e'
        color_off = '#8cb3d9'

        for i, (x, y) in enumerate(pos):
            is_on = occ[i] if occ is not None else False
            facecolor = color_on if is_on else color_off
            is_output = (i == output_site)
            is_input_adj = i in getattr(plot_cluster, '_input_neighbor_sites', [])

            rect = mpatches.Rectangle((x - cell*0.4, y - cell*0.4), cell*0.8, cell*0.8,
                                       linewidth=2, edgecolor='black', facecolor=facecolor, zorder=3)
            ax.add_patch(rect)

            if is_input_adj:
                dot = plt.Circle((x, y), cell*0.12, fill=True, facecolor='black', edgecolor='none', zorder=4)
                ax.add_patch(dot)
            elif is_output:
                circ = plt.Circle((x, y), cell*0.25, fill=False, edgecolor='black', linewidth=2, zorder=4)
                ax.add_patch(circ)

            if Esite is not None:
                label_text = f'{i}\nε={Esite[i]:.2f}'
                fs = 7
            else:
                label_text = str(i)
                fs = 8
            ax.text(x, y, label_text, ha='center', va='center', fontsize=fs, color='white', fontweight='bold', zorder=5)

        for idx, (x, y) in enumerate(input_positions):
            val = input_values[idx] if idx < len(input_values) else 0
            pad_color = color_on if val else color_off
            rect = mpatches.Rectangle((x - cell*0.35, y - cell*0.35), cell*0.7, cell*0.7,
                                       linewidth=2, edgecolor='black', facecolor=pad_color, zorder=3)
            ax.add_patch(rect)
            label = 'A' if idx == 0 else 'B'
            ax.text(x, y, label, ha='center', va='center', fontsize=10, color='white', fontweight='bold', zorder=5)

        if len(pos) > 0:
            x_min, x_max = pos[:,0].min(), pos[:,0].max()
            y_min, y_max = pos[:,1].min(), pos[:,1].max()
            if input_positions:
                inp = np.array(input_positions)
                x_min, x_max = min(x_min, inp[:,0].min()), max(x_max, inp[:,0].max())
                y_min, y_max = min(y_min, inp[:,1].min()), max(y_max, inp[:,1].max())
            ax.set_xlim(x_min - cell, x_max + cell)
            ax.set_ylim(y_min - cell, y_max + cell)

    else:
        if W_val is not None and W_idx is not None and nNeigh is not None:
            for i in range(nSite):
                for k in range(nNeigh[i]):
                    j = W_idx[i, k]
                    if j > i:
                        ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]],
                                'k-', lw=1.0, alpha=0.3, zorder=1)

        for i, (x, y) in enumerate(pos):
            c = 'red' if occ[i] else 'steelblue'
            mk = 's' if i == output_site else 'o'
            sz = 220 if i == output_site else 160
            ax.scatter(x, y, c=c, marker=mk, s=sz, edgecolors='black', linewidths=1.2, zorder=3)
            ax.text(x, y, str(i), ha='center', va='center', fontsize=7, color='white', zorder=4)

        for x, y in input_positions:
            ax.scatter(x, y, c='limegreen', marker='^', s=260, edgecolors='black', linewidths=1.2, zorder=3)

    ax.set_title(title, fontsize=9)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_ground_states(positions, occ_4, E_4, outputs_4, input_positions,
                       output_site, logic_name,
                       W_val=None, W_idx=None, nNeigh=None,
                       input_neighbors=None, Esite_4=None,
                       W1=None, W2=None, eps0=None,
                       fname=None, show=False, tetris_style=False):
    """2x2 subplot showing ground state for each of the 4 input combinations."""
    if input_neighbors is not None:
        input_neighbor_sites = set()
        for sites in input_neighbors:
            input_neighbor_sites.update(sites)
        plot_cluster._input_neighbor_sites = list(input_neighbor_sites)

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    param_str = ""
    if W1 is not None: param_str += f"  W1={W1:.2f}"
    if W2 is not None: param_str += f"  W2={W2:.2f}"
    if eps0 is not None: param_str += f"  ε0={eps0:.2f}"
    fig.suptitle(f'Cluster: {logic_name}{param_str}', fontsize=11)

    for k, ax in enumerate(axes.flat):
        A, B = INPUT_COMBOS[k]
        title = f'In({A},{B}) → Out={int(outputs_4[k])}  E={E_4[k]:.3f}'
        Esite_k = Esite_4[k] if Esite_4 is not None else None
        plot_cluster(ax, positions, occ_4[k], input_positions, output_site,
                     W_val, W_idx, nNeigh, title=title, tetris_style=tetris_style,
                     input_values=[A, B], Esite=Esite_k)

    if tetris_style:
        handles = [
            mpatches.Rectangle((0,0), 1, 1, facecolor='#c44e4e', edgecolor='black', label='ON (n=1)'),
            mpatches.Rectangle((0,0), 1, 1, facecolor='#8cb3d9', edgecolor='black', label='OFF (n=0)'),
            mpatches.Patch(facecolor='white', edgecolor='black', label='Output: ○  In-bias: ●'),
        ]
    else:
        handles = [
            mpatches.Patch(color='red',       label='Occupied (n=1)'),
            mpatches.Patch(color='steelblue', label='Empty    (n=0)'),
            mpatches.Patch(color='limegreen', label='Input pad (fixed)'),
            plt.scatter([], [], marker='s', c='white', edgecolors='black', s=80, label='Output site'),
        ]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8, frameon=False)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    if fname: plt.savefig(fname, dpi=150)
    if show:  plt.show()
    plt.close()

    plot_cluster._input_neighbor_sites = []


# =====================================================================
#  Logic phase diagrams
# =====================================================================

def plot_logic_map(W1_vals, W2_vals, logic_map, title='Logic phase diagram',
                   fname=None, show=False):
    """imshow of logic_map with annotated logic names."""
    nW2, nW1 = logic_map.shape
    img = np.zeros((nW2, nW1, 3))
    for code in range(16):
        mask = logic_map == code
        c = LOGIC_COLORS[code][:3]
        img[mask] = c

    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [W1_vals[0], W1_vals[-1], W2_vals[0], W2_vals[-1]]
    ax.imshow(img, origin='lower', extent=extent, aspect='auto', interpolation='nearest')
    ax.set_xlabel('W1 (Cartesian coupling)', fontsize=11)
    ax.set_ylabel('W2 (Diagonal coupling)',  fontsize=11)
    ax.set_title(title, fontsize=12)

    present = np.unique(logic_map)
    patches = [mpatches.Patch(color=LOGIC_COLORS[c], label=LOGIC_NAMES[c]) for c in present]
    ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8, frameon=True)
    plt.tight_layout()
    if fname: plt.savefig(fname, dpi=150, bbox_inches='tight')
    if show:  plt.show()
    plt.close()


def plot_logic_fraction_map(W1_vals, W2_vals, logic_map, target_set=None,
                             title='Useful logic fraction', fname=None, show=False):
    """Show which (W1,W2) regions produce any useful logic function."""
    if target_set is None: target_set = USEFUL_LOGIC
    target_codes = {c for c, n in LOGIC_NAMES.items() if n in target_set}
    useful_mask = np.isin(logic_map, list(target_codes))

    fig, ax = plt.subplots(figsize=(7, 5))
    extent = [W1_vals[0], W1_vals[-1], W2_vals[0], W2_vals[-1]]
    ax.imshow(useful_mask.astype(float), origin='lower', extent=extent,
              aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xlabel('W1', fontsize=11)
    ax.set_ylabel('W2', fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    if fname: plt.savefig(fname, dpi=150)
    if show:  plt.show()
    plt.close()


# =====================================================================
#  Degeneracy scan plots
# =====================================================================

def plot_scan(E0_vals, W1_vals, n_degen_map, gap_map, cluster_name, W2, fname,
              pos=None, gate_sites=None, gate_labels=None):
    """Plot the (E0, W1) degeneracy scan with a cluster geometry panel on the side."""
    nE, nW = n_degen_map.shape
    nSite = len(pos) if pos is not None else 0

    good_mask = (n_degen_map >= 2) & (n_degen_map <= 4) & (gap_map > 0.3)
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

    ax = axes[0]
    im = ax.imshow(n_degen_map, origin='lower', extent=[W1_vals[0], W1_vals[-1], E0_vals[0], E0_vals[-1]],
                   cmap='YlOrRd', vmin=0, vmax=8, interpolation='nearest', aspect='auto')
    ax.set_title('# degenerate ground states')
    ax.set_xlabel('W1 (nearest-neighbor coupling)')
    ax.set_ylabel('E0 (on-site energy / chemical potential)')
    plt.colorbar(im, ax=ax, label='# degenerate')

    ax = axes[1]
    im = ax.imshow(gap_map, origin='lower', extent=[W1_vals[0], W1_vals[-1], E0_vals[0], E0_vals[-1]],
                   cmap='RdYlGn', vmin=0, vmax=2, interpolation='nearest', aspect='auto')
    ax.set_title('Gap to excited states')
    ax.set_xlabel('W1')
    ax.set_ylabel('E0')
    plt.colorbar(im, ax=ax, label='gap')

    ax = axes[2]
    img = np.ones((nE, nW, 3)) * 0.9
    for nd in range(2, 5):
        mask = good_mask & (n_degen_map == nd)
        if nd == 2:   img[mask] = [0.0, 0.8, 0.0]
        elif nd == 3: img[mask] = [0.0, 0.6, 1.0]
        elif nd == 4: img[mask] = [0.8, 0.4, 0.0]
    img[partial_mask] = [1.0, 0.9, 0.0]
    ax.imshow(img, origin='lower', extent=[W1_vals[0], W1_vals[-1], E0_vals[0], E0_vals[-1]],
              interpolation='nearest', aspect='auto')
    ax.set_title('Good regions (2-4 degen, gap>0.3)\nGreen=2, Blue=3, Orange=4, Yellow=partial')
    ax.set_xlabel('W1')
    ax.set_ylabel('E0')

    if ax_cluster is not None:
        draw_cluster_panel(ax_cluster, pos, nSite, gate_sites, gate_labels)

    good_points = np.argwhere(good_mask)
    if len(good_points) > 0:
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


def plot_best_states(solver, pos, E0, W1, W2, fname, degeneracy_threshold=0.01,
                     analyze_fn=None):
    """Plot the degenerate ground states for a specific (E0, W1, W2) point.

    analyze_fn: callable(solver, pos, E0, W1, W2, threshold) -> (n_degen, gap, E_top8, occ_top8)
                If None, uses Ising_utils.analyze_degeneracy_no_gate.
    """
    from Ising_utils import analyze_degeneracy_no_gate
    if analyze_fn is None:
        analyze_fn = analyze_degeneracy_no_gate

    nSite = len(pos)
    n_d, g_r, E_top8, occ_top8 = analyze_fn(solver, pos, E0, W1, W2, degeneracy_threshold)

    print(f"\n  Detailed analysis at E0={E0:.4f}, W1={W1:.4f}, W2={W2:.4f}:")
    print(f"    {n_d} degenerate ground states, gap to rest = {g_r:.4f}")
    for i in range(min(n_d + 3, 8)):
        occ_mask = occ_top8[i]
        occ_array = [(occ_mask >> s) & 1 for s in range(nSite)]
        n_occ = sum(occ_array)
        marker = " *" if i < n_d else ""
        print(f"      State {i+1}: E={E_top8[i]:.6f}, n_occ={n_occ}, occ=0b{occ_mask:0{nSite}b} {occ_array}{marker}")

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

    for i in range(n_show, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis('off')

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_energy_spectrum(solver, pos, E0_fixed, W1_vals, W2, fname,
                         analyze_fn=None):
    """Plot how the energy spectrum (top 8 states) changes with W1.
    Cluster geometry shown as a side panel."""
    from Ising_utils import analyze_degeneracy_no_gate
    if analyze_fn is None:
        analyze_fn = analyze_degeneracy_no_gate

    nSite = len(pos)
    nW = len(W1_vals)
    E_all = np.zeros((nW, 8), dtype=float)
    for iW, W1 in enumerate(W1_vals):
        _, _, E_top8, _ = analyze_fn(solver, pos, E0_fixed, W1, W2)
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


def plot_energy_spectrum_E0(solver, pos, E0_vals, W1, W2, fname,
                            analyze_fn=None):
    """Plot how the energy spectrum changes with E0 (chemical potential).
    Cluster geometry shown as a side panel."""
    from Ising_utils import analyze_degeneracy_no_gate
    if analyze_fn is None:
        analyze_fn = analyze_degeneracy_no_gate

    nSite = len(pos)
    nE = len(E0_vals)
    E_all = np.zeros((nE, 8), dtype=float)
    for iE, E0 in enumerate(E0_vals):
        _, _, E_top8, _ = analyze_fn(solver, pos, E0, W1, W2)
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


# =====================================================================
#  Tetris-style and scatter-style state visualization
# =====================================================================

def plot_tetris_state(ax, positions, occ_mask, output_site, input_positions,
                      input_neighbors=None):
    """Plot a single state in Tetris style.

    Args:
        ax: matplotlib axes
        positions: (nSite, 2) site coordinates
        occ_mask: int bitmask of occupancy
        output_site: index of output site
        input_positions: list of (x,y) for input pads A and B
        input_neighbors: list of lists (e.g. [[0], [2]]) — sites adjacent to input pads
    """
    nSite = len(positions)
    occ_array = [(occ_mask >> s) & 1 for s in range(nSite)]

    ax.set_xlim(-2, 5)
    ax.set_ylim(-2, 5)
    ax.grid(True, alpha=0.3)

    for s, (x, y) in enumerate(positions):
        if occ_array[s]:
            color = 'red' if s == output_site else 'blue'
            rect = plt.Rectangle((x-0.45, y-0.45), 0.9, 0.9,
                                facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, str(s), ha='center', va='center',
                   color='white', fontsize=10, fontweight='bold')
        else:
            rect = plt.Rectangle((x-0.45, y-0.45), 0.9, 0.9,
                                facecolor='lightgray', edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(x, y, str(s), ha='center', va='center',
                   color='gray', fontsize=8)

    ax.scatter(input_positions[0][0], input_positions[0][1], s=200, c='green', marker='^',
              edgecolors='black', linewidth=2, zorder=10)
    ax.scatter(input_positions[1][0], input_positions[1][1], s=200, c='green', marker='^',
              edgecolors='black', linewidth=2, zorder=10)
    ax.text(input_positions[0][0], input_positions[0][1]+0.3, 'A', ha='center', fontsize=10, fontweight='bold')
    ax.text(input_positions[1][0], input_positions[1][1]+0.3, 'B', ha='center', fontsize=10, fontweight='bold')

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def plot_state_scatter(ax, positions, occ_mask, output_site, input_positions,
                       input_neighbors=None):
    """Plot a single state using scatter markers.

    Args:
        ax: matplotlib axes
        positions: (nSite, 2) site coordinates
        occ_mask: int bitmask of occupancy
        output_site: index of output site
        input_positions: list of (x,y) for input pads A and B
        input_neighbors: list of lists (e.g. [[0], [2]]) — sites adjacent to input pads
    """
    nSite = len(positions)
    occ_array = [(occ_mask >> s) & 1 for s in range(nSite)]
    input_sites = []
    if input_neighbors:
        for sites in input_neighbors:
            input_sites.extend(sites)

    for s, (x, y) in enumerate(positions):
        color = 'red' if s == output_site else ('blue' if occ_array[s] else 'lightgray')
        marker = 's' if s in input_sites else 'o'
        ax.scatter(x, y, s=200, c=color, marker=marker, edgecolors='black', linewidth=2)
        ax.text(x, y+0.2, str(s), ha='center', fontsize=8)

    ax.scatter(input_positions[0][0], input_positions[0][1], s=150, c='green', marker='^', edgecolors='black', linewidth=2)
    ax.scatter(input_positions[1][0], input_positions[1][1], s=150, c='green', marker='^', edgecolors='black', linewidth=2)
    ax.text(input_positions[0][0], input_positions[0][1]+0.2, 'A', ha='center', fontsize=8)
    ax.text(input_positions[1][0], input_positions[1][1]+0.2, 'B', ha='center', fontsize=8)

    ax.set_aspect('equal')
    ax.set_xlim(-2, 4)
    ax.set_ylim(-2, 4)
    ax.grid(True, alpha=0.3)
