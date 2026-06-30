"""
Ising_utils.py — Cluster definitions, analysis functions, and CPU verification.

Cluster registries:
  CLUSTERS_POS: positions-only (for degeneracy scans without gating)
  CLUSTERS_FULL: (positions, input_positions, input_neighbors, output_site)

Analysis functions:
  analyze_degeneracy_no_gate: evaluate cluster degeneracy at (E0, W1, W2)
  scan_E0_W1: scan E0 vs W1 at fixed W2
  compute_energy_py: Python brute-force energy for verification
  check_ground_state_uniqueness: check if ground state is well-separated
"""

import numpy as np
from IsingExactSolver import (
    sq_lattice_sparse, apply_input_bias, INPUT_COMBOS,
    occ_mask_to_array,
)


# =====================================================================
#  Cluster definitions — positions only (for degeneracy scans)
# =====================================================================

def _pos_T_extended_output():
    return np.array([[0,3],[2,3],[0,2],[1,2],[2,2],[1,1],[1,0],[1,-1]], dtype=int)

def _pos_T_extended_inputs():
    return np.array([[0,3],[2,3],[0,2],[2,2],[0,1],[1,1],[2,1],[1,0],[1,-1]], dtype=int)

def _pos_cross():
    return np.array([[2,0],[1,1],[2,1],[3,1],[2,2]], dtype=int)

def _pos_zigzag():
    return np.array([[0,1],[1,0],[2,1],[3,0],[4,1]], dtype=int)

def _pos_T_simple():
    return np.array([[0,1],[1,1],[2,1],[1,0],[1,-1]], dtype=int)

def _pos_chain4():
    return np.array([[0,0],[1,0],[2,0],[3,0]], dtype=int)

def _pos_chain5():
    return np.array([[0,0],[1,0],[2,0],[3,0],[4,0]], dtype=int)

def _pos_L():
    return np.array([[0,0],[1,0],[2,0],[0,1],[1,1],[2,1]], dtype=int)

def _pos_S():
    return np.array([[1,2],[2,2],[0,1],[1,1],[0,0],[1,0]], dtype=int)

CLUSTERS_POS = {
    'T_extended_output': _pos_T_extended_output,
    'T_extended_inputs': _pos_T_extended_inputs,
    'Cross': _pos_cross,
    'Zigzag': _pos_zigzag,
    'T_simple': _pos_T_simple,
    'Chain4': _pos_chain4,
    'Chain5': _pos_chain5,
    'L': _pos_L,
    'S': _pos_S,
}


# =====================================================================
#  Cluster definitions — full (positions + inputs + output)
# =====================================================================

def make_cluster_T():
    """T-shaped 5-site cluster.
       *  *  *     (0,1) (1,1) (2,1)
          *           (1,0)
          *           (1,-1)
    input A: pad at (-1,1), neighbors site 0
    input B: pad at (3, 1), neighbors site 2
    output : site 4 (bottom stem)
    """
    positions = np.array([[0,1],[1,1],[2,1],[1,0],[1,-1]], dtype=int)
    input_positions = [(-1, 1), (3, 1)]
    input_neighbors = [[0], [2]]
    output_site = 4
    return positions, input_positions, input_neighbors, output_site


def make_cluster_T_extended_inputs(input_pos_type='side'):
    """T-shape with longer input lines (active cells on input arms)."""
    positions = np.array([
        [0,3], [2,3], [0,2], [2,2], [0,1], [1,1], [2,1], [1,0], [1,-1]
    ], dtype=int)
    if input_pos_type == 'above':
        input_positions = [(0, 4), (2, 4)]
    elif input_pos_type == 'side':
        input_positions = [(-1, 3), (3, 3)]
    else:
        raise ValueError(f"Unknown input_pos_type: {input_pos_type}")
    input_neighbors = [[0], [1]]
    output_site = 8
    return positions, input_positions, input_neighbors, output_site


def make_cluster_T_extended_output(input_pos_type='side'):
    """T-shape with longer output line (2-site stem)."""
    positions = np.array([
        [0,3], [2,3], [0,2], [1,2], [2,2], [1,1], [1,0], [1,-1]
    ], dtype=int)
    if input_pos_type == 'above':
        input_positions = [(0, 4), (2, 4)]
    elif input_pos_type == 'side':
        input_positions = [(-1, 3), (3, 3)]
    else:
        raise ValueError(f"Unknown input_pos_type: {input_pos_type}")
    input_neighbors = [[0], [1]]
    output_site = 7
    return positions, input_positions, input_neighbors, output_site


def make_cluster_T_no_center():
    """T-shape with central site removed (no site at cross-junction)."""
    positions = np.array([
        [0,2], [2,2], [0,1], [2,1], [1,0], [1,-1]
    ], dtype=int)
    input_positions = [(-1, 2), (3, 2)]
    input_neighbors = [[0], [1]]
    output_site = 5
    return positions, input_positions, input_neighbors, output_site


def make_cluster_T_fork():
    """Fork: separate left/right arms that join at the stem."""
    positions = np.array([
        [0,3], [2,3], [0,2], [2,2], [1,1], [1,0]
    ], dtype=int)
    input_positions = [(0, 4), (2, 4)]
    input_neighbors = [[0], [1]]
    output_site = 5
    return positions, input_positions, input_neighbors, output_site


def make_cluster_T_simple(input_pos_type='side'):
    """Simple T-shape (5 sites)."""
    positions = np.array([[0,1],[1,1],[2,1],[1,0],[1,-1]], dtype=int)
    if input_pos_type == 'above':
        input_positions = [(0, 2), (2, 2)]
    elif input_pos_type == 'side':
        input_positions = [(-1, 1), (3, 1)]
    else:
        raise ValueError(f"Unknown input_pos_type: {input_pos_type}")
    input_neighbors = [[0], [2]]
    output_site = 4
    return positions, input_positions, input_neighbors, output_site


def make_cluster_S():
    """S/Z-shaped 6-site cluster – asymmetric."""
    positions = np.array([[1,2],[2,2],[0,1],[1,1],[0,0],[1,0]], dtype=int)
    input_positions = [(3, 2), (-1, 0)]
    input_neighbors = [[1], [4]]
    output_site = 2
    return positions, input_positions, input_neighbors, output_site


def make_cluster_zigzag():
    """Zigzag 5-site cluster."""
    positions = np.array([[0,1],[1,0],[2,1],[3,0],[4,1]], dtype=int)
    input_positions = [(-1, 1), (5, 1)]
    input_neighbors = [[0], [4]]
    output_site = 2
    return positions, input_positions, input_neighbors, output_site


def make_cluster_L():
    """L-shaped 6-site cluster on a square grid."""
    positions = np.array([[0,0],[1,0],[2,0],[0,1],[1,1],[2,1]], dtype=int)
    input_positions = [(-1, 0), (3, 1)]
    input_neighbors = [[0], [5]]
    output_site = 4
    return positions, input_positions, input_neighbors, output_site


def make_cluster_cross():
    """Cross / plus shaped 5-site cluster."""
    positions = np.array([[2,0],[1,1],[2,1],[3,1],[2,2]], dtype=int)
    input_positions = [(2, 3), (0, 1)]
    input_neighbors = [[4], [1]]
    output_site = 3
    return positions, input_positions, input_neighbors, output_site


def make_cluster_chain():
    """Simple 4-site horizontal chain."""
    positions = np.array([[i,0] for i in range(4)], dtype=int)
    input_positions = [(-1, 0), (4, 0)]
    input_neighbors = [[0], [3]]
    output_site = 2
    return positions, input_positions, input_neighbors, output_site


def make_cluster_straight_line_center():
    """5-site straight line with inputs on ends, output in center."""
    positions = np.array([[i,0] for i in range(5)], dtype=int)
    input_positions = [(-1, 0), (5, 0)]
    input_neighbors = [[0], [4]]
    output_site = 2
    return positions, input_positions, input_neighbors, output_site


CLUSTERS_FULL = {
    'T': make_cluster_T,
    'T_extended_inputs': make_cluster_T_extended_inputs,
    'T_extended_output': make_cluster_T_extended_output,
    'T_no_center': make_cluster_T_no_center,
    'T_fork': make_cluster_T_fork,
    'T_simple': make_cluster_T_simple,
    'S': make_cluster_S,
    'Zigzag': make_cluster_zigzag,
    'L': make_cluster_L,
    'Cross': make_cluster_cross,
    'Chain': make_cluster_chain,
    'Straight_center': make_cluster_straight_line_center,
}


# =====================================================================
#  Analysis functions
# =====================================================================

def analyze_degeneracy_no_gate(solver, pos, E0, W1, W2, degeneracy_threshold=0.01):
    """Evaluate the cluster by itself (no gate voltage).

    Returns:
        n_degen: number of states within threshold of E0_ground
        gap_to_rest: energy gap from last degenerate state to first excited state
        E_top8: (8,) energies
        occ_top8: (8,) occupancy masks
    """
    nSite = len(pos)
    Esite = np.full((1, nSite), E0, dtype=np.float32)
    W_val, W_idx, nNeigh = sq_lattice_sparse(positions=pos, W1=W1, W2=W2, nSite=nSite)
    W_val_batch = W_val[np.newaxis]
    W_idx_batch = W_idx[np.newaxis]
    nNeigh_batch = nNeigh[np.newaxis]
    E_top8, occ_top8 = solver.solve_batch_W_top8(Esite, W_val_batch, W_idx_batch, nNeigh_batch, nSite)
    E_top8 = E_top8[0]
    occ_top8 = occ_top8[0]

    E0_ground = E_top8[0]
    n_degen = int(np.sum(np.abs(E_top8 - E0_ground) < degeneracy_threshold))
    if n_degen < 8:
        gap_to_rest = float(E_top8[n_degen] - E_top8[n_degen - 1])
    else:
        gap_to_rest = 0.0
    return n_degen, gap_to_rest, E_top8, occ_top8


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


def compute_energy_py(occ_mask, Esite, W_val, W_idx, nNeigh, nSite):
    """Compute energy for a given occupancy mask (Python version for verification)."""
    E = 0.0
    for i in range(nSite):
        if not ((occ_mask >> i) & 1):
            continue
        E += Esite[i]
        for k in range(nNeigh[i]):
            j = W_idx[i, k]
            if j < i and ((occ_mask >> j) & 1):
                E += W_val[i, k]
    return E


def check_ground_state_uniqueness(E_top8, threshold=0.01):
    """Check if ground state is well-separated from first excited state.

    Parameters
    ----------
    E_top8 : (nInst, 8) float32
        Energies of 8 lowest states from solve_batch_W_top8
    threshold : float
        Minimum energy gap to consider ground state well-defined

    Returns
    -------
    unique : (nInst,) bool
        True if ground state gap >= threshold (ground state is unique)
    ground_gaps : (nInst,) float
        Energy gaps between ground and first excited state
    """
    ground_gaps = E_top8[:, 1] - E_top8[:, 0]
    unique = ground_gaps >= threshold
    return unique, ground_gaps
