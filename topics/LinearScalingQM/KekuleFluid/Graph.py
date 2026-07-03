"""Honeycomb graph builder with bond-direction labeling and ring detection.

Physical background
-------------------
Graphene and polycyclic aromatic hydrocarbons (PAHs) are honeycomb lattices of
carbon atoms, each contributing one p_z electron to the π system.  The lattice
is bipartite (sublattices A and B) with three nearest-neighbor bond directions
labelled 0, 1, 2 at angles 0, 2π/3, 4π/3.  These three directions correspond to
the three possible Kekulé dimer coverings.

Role in the system
------------------
This module provides the shared graph data structure used by both Model A
(Kekulé fluid solver) and Model B (Dirac / tight-binding solver).  It builds
the lattice, labels bonds by direction, detects hexagonal rings, computes the
aromatic bond baseline (bondBase), and exports all arrays needed for OpenCL
buffer allocation.

Key classes and functions
-------------------------
- `Atom` — dataclass: position, sublattice, bond indices, targetVal, defect,
  pinStrength, pinPhase, complex order parameter z.
- `Bond` — dataclass: endpoints iA/iB, direction dir, bond order x, raw xRaw.
- `HoneycombGraph` — main class:
    - `build_pah(n_shells)` — build hexagonal PAH (benzene, coronene, ...).
    - `build_rect_patch(nx, ny)` — build rectangular patch.
    - `build_flake(radius, shape)` — build circular/hexagonal flake.
    - `find_edge_atoms()` — return undercoordinated atoms (degree < 3).
    - `set_defect(idx)` — mark atom as H⁺ defect (targetVal=0, defect=1).
    - `get_arrays()` — export all numpy arrays for OpenCL.
    - `pull_from_arrays(arrs)` — sync GPU state back to Python objects.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Atom:
    pos: np.ndarray
    sub: int            # +1 A, -1 B
    bonds: list         # 3 bond indices, -1 if absent
    targetVal: float = 1.0
    defect: float = 0.0
    pinStrength: float = 0.0
    pinPhase: float = 0.0
    z: complex = 0 + 0j


@dataclass
class Bond:
    iA: int
    iB: int
    dir: int            # 0, 1, 2
    x: float = 0.0
    xRaw: float = 0.0


class HoneycombGraph:
    def __init__(self, aCC=1.0):
        self.atoms: List[Atom] = []
        self.bonds: List[Bond] = []
        self.rings: List[List[int]] = []
        self.bondBase: np.ndarray = None
        self.aCC = aCC
        self.thetaDir = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
        self.omega = np.exp(1j * self.thetaDir)

    @property
    def natom(self):
        return len(self.atoms)

    @property
    def nbond(self):
        return len(self.bonds)

    def _key(self, pos):
        return (round(float(pos[0]), 4), round(float(pos[1]), 4))

    def build_rect_patch(self, nx, ny):
        """Build a rectangular honeycomb patch with nx*ny unit cells."""
        a = self.aCC
        a1 = np.array([1.5, np.sqrt(3) / 2]) * a
        a2 = np.array([1.5, -np.sqrt(3) / 2]) * a
        delta = [
            np.array([1.0, 0.0]) * a,
            np.array([-0.5, np.sqrt(3) / 2]) * a,
            np.array([-0.5, -np.sqrt(3) / 2]) * a,
        ]

        pos_to_idx = {}
        atom_list = []

        for n1 in range(nx):
            for n2 in range(ny):
                a_pos = n1 * a1 + n2 * a2
                b_pos = a_pos + delta[0]

                ka = self._key(a_pos)
                if ka not in pos_to_idx:
                    pos_to_idx[ka] = len(atom_list)
                    atom_list.append(Atom(pos=a_pos.copy(), sub=+1, bonds=[-1, -1, -1]))

                kb = self._key(b_pos)
                if kb not in pos_to_idx:
                    pos_to_idx[kb] = len(atom_list)
                    atom_list.append(Atom(pos=b_pos.copy(), sub=-1, bonds=[-1, -1, -1]))

        self.atoms = atom_list

        for ia, atom in enumerate(self.atoms):
            if atom.sub != +1:
                continue
            for d in range(3):
                b_pos = atom.pos + delta[d]
                kb = self._key(b_pos)
                if kb in pos_to_idx:
                    ib = pos_to_idx[kb]
                    bid = len(self.bonds)
                    self.bonds.append(Bond(iA=ia, iB=ib, dir=d))
                    atom.bonds[d] = bid
                    self.atoms[ib].bonds[d] = bid

        self._compute_bond_base()
        self._find_rings()

    def build_pah(self, n_shells=2):
        """Build a proper hexagonal PAH (polycyclic aromatic hydrocarbon).

        Grows the molecule ring-by-ring from a central hexagon, adding complete
        rings in concentric shells.  This guarantees:
          - Every atom is part of at least one complete hexagonal ring
          - No dangling ethylene groups or undercoordinated fragments
          - The boundary is a regular hexagon with zigzag edges (all degree-2)

        n_shells: number of ring shells around the central ring.
          n_shells=0 → benzene  (6 atoms, 1 ring)
          n_shells=1 → coronene (24 atoms, 7 rings)
          n_shells=2 → circumcoronene (54 atoms, 19 rings)
          n_shells=3 → 96 atoms, 37 rings

        Total rings = 1 + 3*n_shells*(n_shells+1)
        """
        a = self.aCC
        a1 = np.array([1.5, np.sqrt(3) / 2]) * a
        a2 = np.array([1.5, -np.sqrt(3) / 2]) * a
        delta = [
            np.array([1.0, 0.0]) * a,
            np.array([-0.5, np.sqrt(3) / 2]) * a,
            np.array([-0.5, -np.sqrt(3) / 2]) * a,
        ]

        # Step 1: Build a large enough rect patch to contain the PAH
        # The PAH with n_shells has radius ~ (2*n_shells+1)*a in the longest direction
        extent = 2 * n_shells + 3
        pos_to_idx = {}
        atom_list = []

        for n1 in range(-extent, extent + 1):
            for n2 in range(-extent, extent + 1):
                a_pos = n1 * a1 + n2 * a2
                b_pos = a_pos + delta[0]

                for p, sub in [(a_pos, +1), (b_pos, -1)]:
                    k = self._key(p)
                    if k not in pos_to_idx:
                        pos_to_idx[k] = len(atom_list)
                        atom_list.append(Atom(pos=p.copy(), sub=sub, bonds=[-1, -1, -1]))

        self.atoms = atom_list

        # Build all bonds
        self.bonds = []
        for ia, atom in enumerate(self.atoms):
            if atom.sub != +1:
                continue
            for d in range(3):
                b_pos = atom.pos + delta[d]
                kb = self._key(b_pos)
                if kb in pos_to_idx:
                    ib = pos_to_idx[kb]
                    bid = len(self.bonds)
                    self.bonds.append(Bond(iA=ia, iB=ib, dir=d))
                    atom.bonds[d] = bid
                    self.atoms[ib].bonds[d] = bid

        # Step 2: Find all complete rings
        self._find_rings()
        if not self.rings:
            return

        # Step 3: Compute ring centers and select rings within n_shells
        ring_centers = []
        for ring in self.rings:
            cx = np.mean([self.atoms[i].pos[0] for i in ring])
            cy = np.mean([self.atoms[i].pos[1] for i in ring])
            ring_centers.append((cx, cy))

        # Find the central ring (closest to origin)
        origin_dist = [cx**2 + cy**2 for cx, cy in ring_centers]
        central_idx = np.argmin(origin_dist)
        central_pos = ring_centers[central_idx]

        # Ring spacing: distance between adjacent ring centers = sqrt(3)*a
        ring_spacing = np.sqrt(3) * a

        # Select rings within n_shells * ring_spacing of central ring
        selected_ring_ids = set()
        selected_atoms = set()
        for ri, (cx, cy) in enumerate(ring_centers):
            dx = cx - central_pos[0]
            dy = cy - central_pos[1]
            dist = np.sqrt(dx**2 + dy**2)
            # Allow small tolerance for floating point
            if dist <= n_shells * ring_spacing + 0.01 * a:
                selected_ring_ids.add(ri)
                for atom_idx in self.rings[ri]:
                    selected_atoms.add(atom_idx)

        # Step 4: Rebuild graph with only selected atoms
        old_to_new = {}
        new_atoms = []
        new_pos_to_idx = {}
        for i, atom in enumerate(self.atoms):
            if i not in selected_atoms:
                continue
            old_to_new[i] = len(new_atoms)
            new_atoms.append(Atom(
                pos=atom.pos.copy(), sub=atom.sub, bonds=[-1, -1, -1],
                targetVal=atom.targetVal, defect=atom.defect,
                pinStrength=atom.pinStrength, pinPhase=atom.pinPhase, z=atom.z,
            ))
            new_pos_to_idx[self._key(atom.pos)] = old_to_new[i]

        # Rebuild bonds among selected atoms
        new_bonds = []
        for ia, atom in enumerate(new_atoms):
            if atom.sub != +1:
                continue
            for d in range(3):
                b_pos = atom.pos + delta[d]
                kb = self._key(b_pos)
                if kb in new_pos_to_idx:
                    ib = new_pos_to_idx[kb]
                    bid = len(new_bonds)
                    new_bonds.append(Bond(iA=ia, iB=ib, dir=d))
                    new_atoms[ia].bonds[d] = bid
                    new_atoms[ib].bonds[d] = bid

        # Step 5: Shift all positions so central ring is at origin
        # This ensures the PAH is symmetric about both x and y axes
        shift = np.array(central_pos)
        for atom in new_atoms:
            atom.pos -= shift
        new_pos_to_idx.clear()
        for i, atom in enumerate(new_atoms):
            new_pos_to_idx[self._key(atom.pos)] = i

        self.atoms = new_atoms
        self.bonds = new_bonds
        self._compute_bond_base()
        self._find_rings()

    def build_flake(self, radius, shape='circle'):
        """Build a finite graphene flake (molecule) within a given radius.

        shape: 'circle' or 'hex'
        radius: in units of aCC
        """
        a = self.aCC
        a1 = np.array([1.5, np.sqrt(3) / 2]) * a
        a2 = np.array([1.5, -np.sqrt(3) / 2]) * a
        delta = [
            np.array([1.0, 0.0]) * a,
            np.array([-0.5, np.sqrt(3) / 2]) * a,
            np.array([-0.5, -np.sqrt(3) / 2]) * a,
        ]

        # Generate a large enough grid then filter by shape
        extent = int(radius / a) + 3
        pos_to_idx = {}
        atom_list = []

        for n1 in range(-extent, extent + 1):
            for n2 in range(-extent, extent + 1):
                a_pos = n1 * a1 + n2 * a2
                b_pos = a_pos + delta[0]

                for p, sub in [(a_pos, +1), (b_pos, -1)]:
                    r = np.sqrt(p[0]**2 + p[1]**2)
                    if shape == 'circle':
                        inside = r <= radius
                    elif shape == 'hex':
                        # hexagonal boundary: |x| <= R, |x/2 + sqrt(3)*y/2| <= R, |x/2 - sqrt(3)*y/2| <= R
                        inside = (abs(p[0]) <= radius and
                                  abs(0.5 * p[0] + np.sqrt(3) / 2 * p[1]) <= radius and
                                  abs(0.5 * p[0] - np.sqrt(3) / 2 * p[1]) <= radius)
                    else:
                        inside = r <= radius

                    if inside:
                        k = self._key(p)
                        if k not in pos_to_idx:
                            pos_to_idx[k] = len(atom_list)
                            atom_list.append(Atom(pos=p.copy(), sub=sub, bonds=[-1, -1, -1]))

        self.atoms = atom_list

        for ia, atom in enumerate(self.atoms):
            if atom.sub != +1:
                continue
            for d in range(3):
                b_pos = atom.pos + delta[d]
                kb = self._key(b_pos)
                if kb in pos_to_idx:
                    ib = pos_to_idx[kb]
                    bid = len(self.bonds)
                    self.bonds.append(Bond(iA=ia, iB=ib, dir=d))
                    atom.bonds[d] = bid
                    self.atoms[ib].bonds[d] = bid

        # Trim to proper PAH: iteratively remove atoms with <2 bonds
        # (dangling atoms not part of any complete hexagon)
        self._trim_dangling(pos_to_idx, delta)

        self._compute_bond_base()
        self._find_rings()

    def _trim_dangling(self, pos_to_idx, delta):
        """Iteratively remove atoms with <2 bonds until all have >=2.
        Rebuilds atom list, bonds, and pos_to_idx mapping."""
        changed = True
        while changed:
            changed = False
            # Find atoms to remove (fewer than 2 bonds)
            remove = set()
            for i, atom in enumerate(self.atoms):
                nb = sum(1 for b in atom.bonds if b >= 0)
                if nb < 2:
                    remove.add(i)

            if not remove:
                break
            changed = True

            # Rebuild without removed atoms
            old_to_new = {}
            new_atoms = []
            for i, atom in enumerate(self.atoms):
                if i in remove:
                    continue
                old_to_new[i] = len(new_atoms)
                new_atoms.append(Atom(
                    pos=atom.pos.copy(), sub=atom.sub,
                    bonds=[-1, -1, -1],
                    targetVal=atom.targetVal,
                    defect=atom.defect,
                    pinStrength=atom.pinStrength,
                    pinPhase=atom.pinPhase,
                    z=atom.z,
                ))

            # Rebuild pos_to_idx
            new_pos_to_idx = {}
            for i, atom in enumerate(new_atoms):
                new_pos_to_idx[self._key(atom.pos)] = i

            # Rebuild bonds (same logic as build_flake: A atoms seek B neighbors)
            new_bonds = []
            for ia, atom in enumerate(new_atoms):
                if atom.sub != +1:
                    continue
                for d in range(3):
                    b_pos = atom.pos + delta[d]
                    kb = self._key(b_pos)
                    if kb in new_pos_to_idx:
                        ib = new_pos_to_idx[kb]
                        bid = len(new_bonds)
                        new_bonds.append(Bond(iA=ia, iB=ib, dir=d))
                        new_atoms[ia].bonds[d] = bid
                        new_atoms[ib].bonds[d] = bid

            self.atoms = new_atoms
            self.bonds = new_bonds
            pos_to_idx.clear()
            pos_to_idx.update(new_pos_to_idx)

    def find_edge_atoms(self):
        """Return indices of atoms with fewer than 3 bonds (edge/undercoordinated)."""
        return [i for i, a in enumerate(self.atoms)
                if sum(1 for b in a.bonds if b >= 0) < 3]

    def find_nearest_atom(self, x, y):
        """Return index of atom nearest to (x, y)."""
        best_i, best_d = -1, 1e18
        for i, a in enumerate(self.atoms):
            d = (a.pos[0] - x)**2 + (a.pos[1] - y)**2
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def _compute_bond_base(self):
        deg = np.array([sum(1 for b in a.bonds if b >= 0) for a in self.atoms], dtype=np.float32)
        self._deg = deg
        self.bondBase = np.zeros(len(self.bonds), dtype=np.float32)
        for ib, bond in enumerate(self.bonds):
            di = deg[bond.iA]
            dj = deg[bond.iB]
            if di > 0 and dj > 0:
                self.bondBase[ib] = 0.5 * (self.atoms[bond.iA].targetVal / di +
                                           self.atoms[bond.iB].targetVal / dj)

    def _find_rings(self):
        self.rings = []
        seen = set()
        for ia, atom in enumerate(self.atoms):
            if atom.sub != +1:
                continue
            for d0 in range(3):
                if atom.bonds[d0] < 0:
                    continue
                ring = self._walk_ring(ia, d0)
                if ring is not None:
                    key = frozenset(ring)
                    if key not in seen:
                        seen.add(key)
                        self.rings.append(ring)

    def _walk_ring(self, ia, d0):
        """Walk hexagonal ring from A atom ia, exit via dir d0, always turn right."""
        ring = [ia]
        current = ia
        d = d0
        for step in range(6):
            b = self.atoms[current].bonds[d]
            if b < 0:
                return None
            bond = self.bonds[b]
            nxt = bond.iB if bond.iA == current else bond.iA
            ring.append(nxt)
            current = nxt
            d = (d - 1) % 3
        if current == ia:
            return ring[:6]
        return None

    def get_neighbor_list(self):
        """Return (natom, 3) int32 array of neighbor atom indices (-1 if absent)."""
        neigh = np.full((self.natom, 3), -1, dtype=np.int32)
        for ia, atom in enumerate(self.atoms):
            for d in range(3):
                b = atom.bonds[d]
                if b >= 0:
                    bond = self.bonds[b]
                    neigh[ia, d] = bond.iB if bond.iA == ia else bond.iA
        return neigh

    def get_atom_bonds(self):
        """Return (natom, 3) int32 array of bond indices (-1 if absent)."""
        ab = np.full((self.natom, 3), -1, dtype=np.int32)
        for ia, atom in enumerate(self.atoms):
            for d in range(3):
                ab[ia, d] = atom.bonds[d]
        return ab

    def get_arrays(self):
        """Return all arrays needed for OpenCL buffers."""
        n = self.natom
        nb = self.nbond
        pos = np.array([[a.pos[0], a.pos[1]] for a in self.atoms], dtype=np.float32)
        sub = np.array([a.sub for a in self.atoms], dtype=np.int32)
        targetVal = np.array([a.targetVal for a in self.atoms], dtype=np.float32)
        defect = np.array([a.defect for a in self.atoms], dtype=np.float32)
        pinStrength = np.array([a.pinStrength for a in self.atoms], dtype=np.float32)
        pinPhase = np.array([a.pinPhase for a in self.atoms], dtype=np.float32)
        neighbors = self.get_neighbor_list()
        atom_bonds = self.get_atom_bonds()
        bond_iA = np.array([b.iA for b in self.bonds], dtype=np.int32)
        bond_iB = np.array([b.iB for b in self.bonds], dtype=np.int32)
        bond_dir = np.array([b.dir for b in self.bonds], dtype=np.int32)
        bond_x = np.array([b.x for b in self.bonds], dtype=np.float32)
        bond_xRaw = np.array([b.xRaw for b in self.bonds], dtype=np.float32)
        return dict(
            pos=pos, sub=sub, targetVal=targetVal, defect=defect,
            pinStrength=pinStrength, pinPhase=pinPhase,
            neighbors=neighbors, atom_bonds=atom_bonds,
            bond_iA=bond_iA, bond_iB=bond_iB, bond_dir=bond_dir,
            bond_x=bond_x, bond_xRaw=bond_xRaw, bondBase=self.bondBase,
        )

    def pull_from_arrays(self, arrs):
        """Update Python atom/bond objects from numpy arrays (for diagnostics)."""
        for i, a in enumerate(self.atoms):
            a.z = complex(arrs['z_real'][i], arrs['z_imag'][i])
            a.targetVal = float(arrs['targetVal'][i])
            a.defect = float(arrs['defect'][i])
            a.pinStrength = float(arrs['pinStrength'][i])
            a.pinPhase = float(arrs['pinPhase'][i])
        for b, bond in enumerate(self.bonds):
            bond.x = float(arrs['bond_x'][b])
            bond.xRaw = float(arrs['bond_xRaw'][b])
