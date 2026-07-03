"""
hexgrid.py — Honeycomb lattice builder and shared data structures (legacy).

Physical background
-------------------
Same honeycomb lattice as Graph.py, but using a numpy-dataclass representation
instead of Python objects.  Provides vortex initialization helpers used by the
4-component Dirac solver (Dirac4_ocl.py).

Role in the system
------------------
Legacy module used only by Dirac4_ocl.py and run_dirac.py.  Redundant with
Graph.py — the current codebase uses Graph.py for all honeycomb graph building.

Provides:
  - build_honeycomb_patch(nx, ny, aCC)  -> atoms, bonds, rings
  - Vortex dataclass
  - init_vortex_phase(positions, vortices, r_core)
  - init_vortex_phase_grid(...)

Atoms and bonds follow the convention from the KekuleFluid spec:
  - sublattice A (+1) and B (-1)
  - bond directions dir in {0,1,2} with thetaDir = {0, 2*pi/3, 4*pi/3}
  - bonds oriented from A to B
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# Bond direction phases
THETA_DIR = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
OMEGA = np.exp(1j * THETA_DIR)  # omega[0], omega[1], omega[2]


@dataclass
class Vortex:
    pos: np.ndarray   # (x, y)
    winding: int      # +1 or -1


@dataclass
class HoneycombGraph:
    n_atoms: int
    n_bonds: int
    n_rings: int
    pos: np.ndarray           # (n_atoms, 2)
    sub: np.ndarray           # (n_atoms,) +1=A, -1=B
    bonds: np.ndarray         # (n_bonds, 2) int32 [iA, iB]
    bond_dir: np.ndarray      # (n_bonds,) int32 0,1,2
    atom_bonds: np.ndarray    # (n_atoms, 3) int32, bond indices or -1
    rings: np.ndarray         # (n_rings, 6) int32, atom indices around each hexagon
    ring_centers: np.ndarray  # (n_rings, 2)


def build_honeycomb_patch(nx: int, ny: int, aCC: float = 1.0) -> HoneycombGraph:
    """Build a rectangular honeycomb patch with open boundaries.

    The lattice is generated using two sublattice basis vectors:
      a1 = (3*aCC, 0)
      a2 = (3*aCC/2, sqrt(3)*aCC/2)  ... actually use standard convention

    We use a unit-cell approach: each unit cell has 2 atoms (A, B)
    and the lattice vectors are:
      a1 = (sqrt(3)*aCC, 0)
      a2 = (sqrt(3)*aCC/2, 3*aCC/2)

    Atom A at (0, 0), atom B at (0, aCC) within the cell.
    Nearest-neighbor vectors from A to B:
      delta_0 = (0, aCC)          — intra-cell
      delta_1 = (sqrt(3)*aCC/2, -aCC/2)  — down-right
      delta_2 = (-sqrt(3)*aCC/2, -aCC/2) — down-left
    """
    s3 = np.sqrt(3.0)
    a1 = np.array([s3 * aCC, 0.0])
    a2 = np.array([s3 * aCC / 2.0, 3.0 * aCC / 2.0])
    b_offset = np.array([0.0, aCC])

    # Delta vectors from A to B
    deltas = np.array([
        [0.0, aCC],                      # dir 0: intra-cell (up)
        [s3 * aCC / 2.0, -aCC / 2.0],    # dir 1: down-right
        [-s3 * aCC / 2.0, -aCC / 2.0],   # dir 2: down-left
    ])

    # Generate atoms
    pos_list = []
    sub_list = []
    cell_map = {}  # (i, j, sub) -> atom index

    for i in range(nx):
        for j in range(ny):
            base = i * a1 + j * a2
            # A atom
            posA = base.copy()
            pos_list.append(posA)
            sub_list.append(+1)
            cell_map[(i, j, 0)] = len(pos_list) - 1
            # B atom
            posB = base + b_offset
            pos_list.append(posB)
            sub_list.append(-1)
            cell_map[(i, j, 1)] = len(pos_list) - 1

    n_atoms = len(pos_list)
    pos = np.array(pos_list, dtype=np.float64)
    sub = np.array(sub_list, dtype=np.int32)

    # Build bonds: for each A atom, find B neighbors
    # A at (i,j) connects to:
    #   dir 0: B at (i,j)       — intra-cell
    #   dir 1: B at (i+1, j-1)  — down-right (actually need to check geometry)
    #   dir 2: B at (i-1, j-1)  — down-left

    # Let's use distance-based bond finding for robustness
    bond_list = []
    bond_dir_list = []

    # For each A atom, find 3 nearest B atoms
    a_indices = np.where(sub == +1)[0]
    b_indices = np.where(sub == -1)[0]

    posA = pos[a_indices]
    posB = pos[b_indices]

    for ia_local, ia in enumerate(a_indices):
        pa = pos[ia]
        dists = np.linalg.norm(posB - pa, axis=1)
        # Sort by distance, take up to 3 within cutoff
        sorted_idx = np.argsort(dists)
        count = 0
        for ib_local in sorted_idx:
            d = dists[ib_local]
            if d > aCC * 1.5:
                break
            ib = b_indices[ib_local]
            # Determine direction
            diff = pos[ib] - pa
            angle = np.arctan2(diff[1], diff[0])
            # Match to nearest theta direction
            # Our deltas: dir0 = (0, aCC) -> angle pi/2
            #            dir1 = (s3/2, -1/2)*aCC -> angle -pi/6
            #            dir2 = (-s3/2, -1/2)*aCC -> angle -5pi/6
            # But thetaDir = {0, 2pi/3, 4pi/3} is the Kekule phase, not the geometric angle
            # We need to map geometric direction to dir index
            dir_angles = np.array([np.pi / 2, -np.pi / 6, -5 * np.pi / 6])
            dir_idx = np.argmin(np.abs(np.angle(np.exp(1j * (angle - dir_angles)))))
            bond_list.append([ia, ib])
            bond_dir_list.append(dir_idx)
            count += 1
            if count >= 3:
                break

    n_bonds = len(bond_list)
    bonds = np.array(bond_list, dtype=np.int32)
    bond_dir = np.array(bond_dir_list, dtype=np.int32)

    # Build atom_bonds: for each atom, which bonds are incident
    atom_bonds = np.full((n_atoms, 3), -1, dtype=np.int32)
    for ibond, (ia, ib) in enumerate(bonds):
        for k in range(3):
            if atom_bonds[ia, k] == -1:
                atom_bonds[ia, k] = ibond
                break
        for k in range(3):
            if atom_bonds[ib, k] == -1:
                atom_bonds[ib, k] = ibond
                break

    # Detect hexagonal rings: find cycles of length 6
    rings = _detect_hex_rings(n_atoms, bonds, pos, aCC)
    if len(rings) > 0:
        ring_centers = np.array([pos[r].mean(axis=0) for r in rings], dtype=np.float64)
    else:
        ring_centers = np.zeros((0, 2), dtype=np.float64)

    return HoneycombGraph(
        n_atoms=n_atoms,
        n_bonds=n_bonds,
        n_rings=len(rings),
        pos=pos,
        sub=sub,
        bonds=bonds,
        bond_dir=bond_dir,
        atom_bonds=atom_bonds,
        rings=np.array(rings, dtype=np.int32) if rings else np.zeros((0, 6), dtype=np.int32),
        ring_centers=ring_centers,
    )


def _detect_hex_rings(n_atoms, bonds, pos, aCC):
    """Detect hexagonal rings by finding 6-cycles in the graph."""
    # Build adjacency
    adj = [[] for _ in range(n_atoms)]
    for ia, ib in bonds:
        adj[ia].append(ib)
        adj[ib].append(ia)

    # For each bond, try to find a hexagon starting from it
    ring_set = set()
    rings = []

    for ia, ib in bonds:
        # Walk: ia -> ib -> c -> d -> e -> f -> back to ia
        for c in adj[ib]:
            if c == ia:
                continue
            for d in adj[c]:
                if d == ib or d == ia:
                    continue
                for e in adj[d]:
                    if e == c or e == ib or e == ia:
                        continue
                    for f in adj[e]:
                        if f == d or f == c or f == ib or f == ia:
                            continue
                        if f in adj[ia]:
                            ring = tuple(sorted([ia, ib, c, d, e, f]))
                            if ring not in ring_set:
                                ring_set.add(ring)
                                # Order the ring atoms properly
                                ordered = _order_ring([ia, ib, c, d, e, f], adj)
                                rings.append(ordered)

    return rings


def _order_ring(atoms, adj):
    """Order ring atoms so they form a cycle."""
    ordered = [atoms[0]]
    remaining = set(atoms[1:])
    while remaining:
        for a in remaining:
            if a in adj[ordered[-1]]:
                ordered.append(a)
                remaining.remove(a)
                break
        else:
            break
    return ordered


def init_vortex_phase(positions: np.ndarray, vortices: List[Vortex],
                      r_core: float = 2.0) -> np.ndarray:
    """Initialize complex phase field with vortex/antivortex winding.

    Returns complex array of shape (n_points,).
    """
    n = len(positions)
    phi = np.zeros(n)
    A = np.ones(n)

    for v in vortices:
        d = positions - v.pos
        r = np.linalg.norm(d, axis=1)
        phi += v.winding * np.arctan2(d[:, 1], d[:, 0])
        A *= np.tanh(r / r_core)

    return A * np.exp(1j * phi)


def init_vortex_phase_grid(grid_x: np.ndarray, grid_y: np.ndarray,
                           vortices: List[Vortex], r_core: float = 2.0) -> np.ndarray:
    """Initialize complex phase field on a 2D Cartesian grid.

    grid_x, grid_y are 2D arrays from meshgrid.
    Returns complex 2D array.
    """
    shape = grid_x.shape
    phi = np.zeros(shape)
    A = np.ones(shape)

    for v in vortices:
        dx = grid_x - v.pos[0]
        dy = grid_y - v.pos[1]
        r = np.sqrt(dx**2 + dy**2)
        phi += v.winding * np.arctan2(dy, dx)
        A *= np.tanh(r / r_core)

    return A * np.exp(1j * phi)
