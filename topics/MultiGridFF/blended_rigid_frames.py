#!/usr/bin/env python3
"""
Blended Rigid Body Frames — Step 1: Load PTCDA, place two frame centers,
compute interpolation weights from geometry, and plot the weight map.

Two weight methods:
  1) Radial exponential: w_a ~ exp(-|r_i - c_a| / decay), renormalized so sum=1
  2) Linear smoothstep: project atom position onto the line between the two
     frame pivots, then apply smoothstep to the fractional coordinate.

Run:
  python blended_rigid_frames.py
"""

import sys, os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- make the ChemicalGraphs package importable ---
_topic_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_topic_root))
from ChemicalGraphs import AtomicSystem, plotUtils as pu, elements

# ================================================================
# 1. Load PTCDA and assign bonds / atom types
# ================================================================
xyz_path = Path(__file__).resolve().parent.parent.parent / "data" / "xyz" / "PTCDA.xyz"
print(f"Loading {xyz_path}")
mol = AtomicSystem(fname=str(xyz_path), bPreinit=True)

mol.findBonds(byRvdW=False, Rcut=1.6, RvdwCut=0.5)  # effective cutoff = 1.6*2*0.5 = 1.6 Å
mol.neighs()

natoms = len(mol.apos)
print(f"Loaded {natoms} atoms, {len(mol.bonds)} bonds")

# Print atom summary
for i in range(natoms):
    e = mol.enames[i]
    r = mol.apos[i]
    n_bonds = len(mol.ngs[i])
    print(f"  [{i:2d}] {e:>2s}  ({r[0]:8.4f},{r[1]:8.4f},{r[2]:8.4f})  nbonds={n_bonds}")

# ================================================================
# 2. Identify bridge oxygens (-O-) vs carbonyl oxygens (=O)
# ================================================================
# In PTCDA anhydride groups: bridge O has 2 C neighbors, carbonyl O has 1 C neighbor
o_indices = [i for i in range(natoms) if mol.enames[i] == 'O']
bridge_O = []
carbonyl_O = []
for i in o_indices:
    nC = sum(1 for j in mol.ngs[i] if mol.enames[j] == 'C')
    if nC == 2:
        bridge_O.append(i)
    else:
        carbonyl_O.append(i)

print(f"\nBridge oxygens (-O-):  {bridge_O}  at x={[f'{mol.apos[i,0]:.3f}' for i in bridge_O]}")
print(f"Carbonyl oxygens (=O): {carbonyl_O} at x={[f'{mol.apos[i,0]:.3f}' for i in carbonyl_O]}")

# ================================================================
# 3. Place two rigid-body frame centers at molecular ends along x
# ================================================================
# Use the bridge oxygens as the natural end points
# PTCDA has 2 bridge O per end; take the centroid of each end's bridge O's
# Left end (negative x): bridge O at index 26 (x=-5.73)
# Right end (positive x): bridge O at index 27 (x=+5.73)
# Group by x-sign
left_O = [i for i in bridge_O if mol.apos[i, 0] < 0]
right_O = [i for i in bridge_O if mol.apos[i, 0] > 0]

c1 = mol.apos[left_O].mean(axis=0)
c2 = mol.apos[right_O].mean(axis=0)

print(f"\nFrame center 1 (left):  {c1}")
print(f"Frame center 2 (right): {c2}")

# ================================================================
# 4. Compute interpolation weights
# ================================================================

def smoothstep(t):
    """Hermite smoothstep: 3t^2 - 2t^3, clamped to [0,1]."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def weights_radial_exp(apos, c1, c2, decay=2.0):
    """Method 1: radial exponential decay from each center, renormalized.
    w1 = exp(-d1/decay), w2 = exp(-d2/decay), then s = w2/(w1+w2).
    Returns s in [0,1] where s=0 -> fully frame1, s=1 -> fully frame2.
    """
    d1 = np.linalg.norm(apos - c1[None, :], axis=1)
    d2 = np.linalg.norm(apos - c2[None, :], axis=1)
    w1 = np.exp(-d1 / decay)
    w2 = np.exp(-d2 / decay)
    s = w2 / (w1 + w2)
    return s

def weights_linear_smoothstep(apos, c1, c2):
    """Method 2: project onto the c1->c2 line, use smoothstep on the
    fractional coordinate along that line.
    Returns s in [0,1] where s=0 -> fully frame1, s=1 -> fully frame2.
    """
    axis = c2 - c1
    L = np.linalg.norm(axis)
    e = axis / L
    # project each atom onto the axis relative to c1
    t = np.dot(apos - c1[None, :], e) / L
    s = smoothstep(t)
    return s

s_radial = weights_radial_exp(mol.apos, c1, c2, decay=2.0)
s_linear = weights_linear_smoothstep(mol.apos, c1, c2)

print("\n--- Interpolation weights s (0=frame1/left, 1=frame2/right) ---")
print(f"{'atom':>4s} {'elem':>4s} {'s_radial':>10s} {'s_linear':>10s}")
for i in range(natoms):
    print(f"  {i:3d} {mol.enames[i]:>3s}  {s_radial[i]:10.4f}  {s_linear[i]:10.4f}")

# ================================================================
# 5. Plot the weight map
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# custom colormap: blue (frame1) -> white (middle) -> red (frame2)
cmap = LinearSegmentedColormap.from_list('frame_blend', ['#2060ff', '#ffffff', '#ff4020'])

elem_colors = {'H': 'gray', 'C': 'black', 'O': 'red'}
elem_sizes  = {'H': 30, 'C': 60, 'O': 80}

for ax, s_vals, title in zip(axes, [s_radial, s_linear],
                              ['Method 1: Radial exponential (decay=2.0 Å)',
                               'Method 2: Linear smoothstep']):
    ax.set_aspect('equal')

    # draw bonds
    for (ia, ib) in mol.bonds:
        ax.plot([mol.apos[ia, 0], mol.apos[ib, 0]],
                [mol.apos[ia, 1], mol.apos[ib, 1]],
                'k-', lw=0.8, alpha=0.5, zorder=1)

    # draw atoms colored by weight
    colors = [cmap(s_vals[i]) for i in range(natoms)]
    sizes  = [elem_sizes.get(mol.enames[i], 40) for i in range(natoms)]
    ax.scatter(mol.apos[:, 0], mol.apos[:, 1], c=colors, s=sizes,
               edgecolors='black', linewidths=0.5, zorder=5)

    # annotate atom indices
    for i in range(natoms):
        ax.annotate(str(i), (mol.apos[i, 0], mol.apos[i, 1]),
                    fontsize=5, ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points', zorder=6)

    # mark frame centers
    for ci, label, color in [(c1, 'Frame 1', '#2060ff'), (c2, 'Frame 2', '#ff4020')]:
        ax.plot(ci[0], ci[1], marker='X', color=color, markersize=15,
                markeredgecolor='black', markeredgewidth=1.5, zorder=10)
        ax.annotate(label, (ci[0], ci[1]), fontsize=9, fontweight='bold',
                    color=color, xytext=(0, -15), textcoords='offset points',
                    ha='center', zorder=11)

    # mark bridge oxygens with circles
    for i in bridge_O:
        ax.plot(mol.apos[i, 0], mol.apos[i, 1], 'o', ms=14,
                mfc='none', mec='green', mew=2, zorder=8)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('x (Å)')
    ax.set_ylabel('y (Å)')

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('s (0=Frame1, 1=Frame2)')

fig.suptitle('PTCDA — Blended Rigid Body Frame Interpolation Weights', fontsize=14, fontweight='bold')
fig.tight_layout()

out_path = Path(__file__).resolve().parent / 'ptcda_weights.png'
fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
print(f"\nSaved plot to {out_path}")
plt.show()
