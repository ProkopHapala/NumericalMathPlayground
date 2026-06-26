#!/usr/bin/env python3
"""
KekuleQM_ocl.py — pyOpenCL harness for the KekuleQM.cl kernels.

Loads a molecule from .xyz (or generates a graphene flake), builds the
fixed-size neighbor lists with reverse-slot mapping, uploads the half-bond
electron allocation state to the GPU, runs the 3-kernel iteration loop
(local bonding gather → Coulomb gather → DOF update), and plots the
resulting bond orders and bond lengths.

The persistent electronic DOF is y[i,spin,channel] where channel 0 is
localized site density and channels 1..3 are half-bond donations to neighbors.
The bond order between atoms i and j is B_ij = 2*sum_s sqrt(y[i,s,i->j]*y[j,s,j->i]).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path
import os

try:
    import pyopencl as cl
    HAS_PYOPENCL = True
except ImportError:
    HAS_PYOPENCL = False

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.realpath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
    from ChemicalGraphs import parse_ascii_art, ASCII_EXAMPLES
    from ChemicalGraphs import make_n_pi
    HAS_CHEMICAL_GRAPHS = True
except ImportError:
    HAS_CHEMICAL_GRAPHS = False

# =====================================================================
# 1. MOLECULE LOADING AND GENERATION
# =====================================================================

def read_xyz(fname):
    """Parse a standard .xyz file. Returns (enames, positions)."""
    with open(fname, 'r') as f:
        lines = f.readlines()
    natoms = int(lines[0].strip())
    enames = []
    positions = []
    for line in lines[2:2+natoms]:
        parts = line.strip().split()
        enames.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(enames), np.array(positions, dtype=np.float64)


def generate_graphene_flake(radius=5.0, a=1.42, noise=0.0):
    """Generate a honeycomb flake and return (enames, positions)."""
    pts = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            x_base = 1.5 * a * i
            y_base = np.sqrt(3) * a * (j + 0.5 * (i % 2))
            pts.append([x_base, y_base, 0.0])
            pts.append([x_base + a, y_base, 0.0])
    pts = np.array(pts)
    dists = np.linalg.norm(pts[:, :2], axis=1)
    mask = dists < radius
    pts = pts[mask]
    if noise > 0:
        pts += np.random.normal(0, noise, pts.shape)
    enames = np.array(['C'] * len(pts))
    return enames, pts


# =====================================================================
# 1b. LOAD FROM ASCII ART (ChemicalGraphs)
# =====================================================================

def from_ascii_art(art_text, aCC=1.42, bHydrogens=True):
    """Build (enames, positions, bonds) from ASCII art using ChemicalGraphs.

    Returns heavy-atom-only arrays suitable for the OpenCL solver.
    """
    if not HAS_CHEMICAL_GRAPHS:
        raise RuntimeError("ChemicalGraphs package not available")
    atoms = parse_ascii_art(art_text, hbond_length=3.0)
    atoms.neighs()
    n_pi = make_n_pi(atoms)
    if bHydrogens:
        from ChemicalGraphs.KekuleBackend import KekuleBackend
        kb = KekuleBackend()
        tv = {}
        for ia, e in enumerate(atoms.enames):
            eu = e.upper()
            if eu in ('C', 'N', 'O'):
                npi = 1 if float(n_pi[ia]) > 0.5 else 0
                tv[ia] = kb._target_sigma(eu, npi)
        atoms.add_capping_h_sp2(target_valence=tv)
        atoms.neighs()
    # Filter to heavy atoms only for the pi-system solver
    heavy_mask = np.array([e not in ('H', 'E') for e in atoms.enames], dtype=bool)
    heavy_idx = np.nonzero(heavy_mask)[0]
    idx_map = {old: new for new, old in enumerate(heavy_idx)}
    positions = atoms.apos[heavy_idx]
    enames = np.array([atoms.enames[i] for i in heavy_idx])
    # Remap bonds to heavy-atom indices
    bonds = []
    for i, j in atoms.bonds:
        if heavy_mask[i] and heavy_mask[j]:
            bonds.append((idx_map[i], idx_map[j]))
    bonds = np.array(bonds, dtype=np.int32) if bonds else np.zeros((0, 2), dtype=np.int32)
    return enames, positions, bonds


def from_atoms(atoms, bHeavyOnly=True):
    """Convert a ChemicalGraphs AtomicSystem to (enames, positions, bonds).

    If bHeavyOnly, filter out H atoms and remap bond indices.
    """
    if bHeavyOnly:
        heavy_mask = np.array([e not in ('H', 'E') for e in atoms.enames], dtype=bool)
        heavy_idx = np.nonzero(heavy_mask)[0]
        idx_map = {old: new for new, old in enumerate(heavy_idx)}
        positions = atoms.apos[heavy_idx]
        enames = np.array([atoms.enames[i] for i in heavy_idx])
        bonds = []
        for i, j in atoms.bonds:
            if heavy_mask[i] and heavy_mask[j]:
                bonds.append((idx_map[i], idx_map[j]))
        bonds = np.array(bonds, dtype=np.int32) if bonds else np.zeros((0, 2), dtype=np.int32)
    else:
        positions = atoms.apos
        enames = np.array(atoms.enames)
        bonds = np.asarray(atoms.bonds, dtype=np.int32) if atoms.bonds is not None else np.zeros((0, 2), dtype=np.int32)
    return enames, positions, bonds


def build_neighbor_lists_from_bonds(positions, bonds, max_neigh=3):
    """Build neigh/rev arrays from an explicit bond list instead of distance cutoff.

    This preserves the bond topology from the ASCII art parser exactly.
    """
    natoms = len(positions)
    neigh = np.full((natoms, max_neigh), -1, dtype=np.int32)
    nneigh = np.zeros(natoms, dtype=np.int32)
    for i, j in bonds:
        if nneigh[i] < max_neigh:
            neigh[i, nneigh[i]] = j
            nneigh[i] += 1
        if nneigh[j] < max_neigh:
            neigh[j, nneigh[j]] = i
            nneigh[j] += 1
    # Build reverse mapping
    rev = np.full((natoms, max_neigh), -1, dtype=np.int32)
    for i in range(natoms):
        for k in range(max_neigh):
            j = neigh[i, k]
            if j < 0:
                continue
            for kk in range(max_neigh):
                if neigh[j, kk] == i:
                    rev[i, k] = kk
                    break
    return neigh, rev


# =====================================================================
# 2. NEIGHBOR LIST AND REVERSE-SLOT BUILDING
# =====================================================================

def build_neighbor_lists(positions, cutoff=1.8, max_neigh=3):
    """Build fixed-size neigh[i][k] and rev[i][k] arrays.

    neigh[i][k] = global index of k-th neighbor of atom i (or -1)
    rev[i][k]   = slot of i inside neighbor j=neigh[i][k]'s list

    This is the critical data structure: each atom stores up to max_neigh
    neighbors, and the reverse slot lets atom i find its own half-bond
    variable inside neighbor j's local array during the gather.
    """
    natoms = len(positions)
    neigh = np.full((natoms, max_neigh), -1, dtype=np.int32)
    nneigh = np.zeros(natoms, dtype=np.int32)

    # First pass: find neighbors by distance cutoff
    for i in range(natoms):
        for j in range(natoms):
            if i == j:
                continue
            r = np.linalg.norm(positions[i] - positions[j])
            if r < cutoff:
                k = nneigh[i]
                if k < max_neigh:
                    neigh[i, k] = j
                    nneigh[i] += 1

    # Build reverse mapping: if j = neigh[i][k], find slot of i in j's list
    rev = np.full((natoms, max_neigh), -1, dtype=np.int32)
    for i in range(natoms):
        for k in range(max_neigh):
            j = neigh[i, k]
            if j < 0:
                continue
            for kk in range(max_neigh):
                if neigh[j, kk] == i:
                    rev[i, k] = kk
                    break

    return neigh, rev


def build_bond_list(neigh, max_neigh=3):
    """Extract unique (i,j) bond pairs from the neighbor list."""
    natoms = neigh.shape[0]
    bonds = []
    seen = set()
    for i in range(natoms):
        for k in range(max_neigh):
            j = neigh[i, k]
            if j < 0:
                continue
            pair = (min(i, j), max(i, j))
            if pair not in seen:
                seen.add(pair)
                bonds.append(pair)
    return np.array(bonds, dtype=np.int32)


# =====================================================================
# 3. PHYSICAL PARAMETERS
# =====================================================================

def default_params(natoms, neigh, enames=None, r0_bond=1.42,
                    eps0_val=2.7, beta_val=3.0, Kr_val=20.0,
                    chi_C=0.0, eta_C=0.0, U_C=0.0, Zpi_C=1.0,
                    chi_N=-1.5, eta_N=0.0, U_N=0.0, Zpi_N=1.0,
                    chi_O=-2.5, eta_O=0.0, U_O=0.0, Zpi_O=1.0):
    """Build per-slot parameter arrays for the kernels.

    Returns eps0, beta, r0, Kr arrays of shape (natoms, max_neigh),
    and atomParams of shape (natoms, 4) = (chi, eta, U, Zpi).
    """
    max_neigh = neigh.shape[1]
    eps0 = np.full((natoms, max_neigh), eps0_val, dtype=np.float32)
    beta = np.full((natoms, max_neigh), beta_val, dtype=np.float32)
    r0   = np.full((natoms, max_neigh), r0_bond, dtype=np.float32)
    Kr   = np.full((natoms, max_neigh), Kr_val, dtype=np.float32)

    # Per-atom QEq/Hubbard parameters: (chi, eta, U, Zpi)
    atomParams = np.zeros((natoms, 4), dtype=np.float32)
    atomParams[:, 0] = chi_C
    atomParams[:, 1] = eta_C
    atomParams[:, 2] = U_C
    atomParams[:, 3] = Zpi_C

    if enames is not None:
        for i, e in enumerate(enames):
            if e == 'N':
                atomParams[i, 0] = chi_N
                atomParams[i, 1] = eta_N
                atomParams[i, 2] = U_N
                atomParams[i, 3] = Zpi_N
            elif e == 'O':
                atomParams[i, 0] = chi_O
                atomParams[i, 1] = eta_O
                atomParams[i, 2] = U_O
                atomParams[i, 3] = Zpi_O

    return eps0, beta, r0, Kr, atomParams


# =====================================================================
# 4. OPENCL SOLVER
# =====================================================================

class KekuleQM_OCL:
    """pyOpenCL harness for the KekuleQM 3-kernel iteration."""

    KQ_MAX_NEIGH = 3
    KQ_NSPIN = 2
    KQ_NCHAN = 4   # 1 localized + 3 bond channels

    # Flag constants matching the .cl file
    KQ_FLAG_LOCAL_SIMPLEX        = 1
    KQ_FLAG_UPDATE_R             = 2
    KQ_FLAG_USE_COULOMB_ELECTRON = 4
    KQ_FLAG_USE_COULOMB_FORCE    = 8

    def __init__(self, ctx=None, queue=None):
        if not HAS_PYOPENCL:
            raise RuntimeError("pyopencl not installed")
        if ctx is None:
            # Pick first available platform/device
            platforms = cl.get_platforms()
            ctx = None
            for plat in platforms:
                devs = plat.get_devices()
                if devs:
                    ctx = cl.Context(devs)
                    break
            if ctx is None:
                raise RuntimeError("No OpenCL device found")
        self.ctx = ctx
        self.queue = cl.CommandQueue(ctx) if queue is None else queue
        self._load_kernels()

    def _load_kernels(self):
        cl_path = os.path.join(os.path.dirname(__file__), 'KekuleQM.cl')
        with open(cl_path, 'r') as f:
            source = f.read()
        self.program = cl.Program(self.ctx, source).build()
        self.k_gather_local  = self.program.KekuleQM_gatherLocalBonding
        self.k_gather_coulomb = self.program.KekuleQM_gatherCoulombDirect
        self.k_update_dofs    = self.program.KekuleQM_updateDOFs

    def setup(self, positions, neigh, rev, eps0, beta, r0, Kr, atomParams,
              rhoTarget=None, bUseCoulomb=False, bUpdateR=False):
        """Allocate GPU buffers and upload constant data + initial state."""
        natoms = len(positions)
        self.natoms = natoms
        self.bUseCoulomb = bUseCoulomb
        self.bUpdateR = bUpdateR

        mf = cl.mem_flags
        NS = self.KQ_NSPIN
        NCH = self.KQ_NCHAN
        MNE = self.KQ_MAX_NEIGH

        # --- Constant data ---
        self.d_R       = cl.Buffer(self.ctx, mf.READ_WRITE, natoms * 16)
        self.d_neigh   = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                   hostbuf=neigh.astype(np.int32))
        self.d_rev     = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                   hostbuf=rev.astype(np.int32))
        self.d_eps0    = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                   hostbuf=eps0.astype(np.float32).ravel())
        self.d_beta    = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                   hostbuf=beta.astype(np.float32).ravel())
        self.d_r0      = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                   hostbuf=r0.astype(np.float32).ravel())
        self.d_Kr      = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                   hostbuf=Kr.astype(np.float32).ravel())
        self.d_atomParams = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                      hostbuf=atomParams.astype(np.float32))

        # --- Persistent electronic state: y[natoms, NS, NCH] ---
        y_host = np.random.rand(natoms * NS * NCH).astype(np.float32) * 0.1
        # Distribute roughly equally among active channels per atom
        for i in range(natoms):
            nchan = 1
            for k in range(MNE):
                if neigh[i, k] >= 0:
                    nchan += 1
            target_per = 0.5 / nchan  # rho_target/2 per spin for carbon
            for s in range(NS):
                y_host[((i * NS + s) * NCH + 0)] = target_per
                for k in range(MNE):
                    if neigh[i, k] >= 0:
                        y_host[((i * NS + s) * NCH + 1 + k)] = target_per
                    else:
                        y_host[((i * NS + s) * NCH + 1 + k)] = 0.0
        self.d_y = cl.Buffer(self.ctx, mf.READ_WRITE, natoms * NS * NCH * 4)
        cl.enqueue_copy(self.queue, self.d_y, y_host)

        # Upload positions
        R_host = np.zeros((natoms, 4), dtype=np.float32)
        R_host[:, :3] = positions.astype(np.float32)
        cl.enqueue_copy(self.queue, self.d_R, R_host)

        # --- rhoTarget: (rho_up, rho_dn) per atom ---
        if rhoTarget is None:
            rhoTarget = np.zeros((natoms, 2), dtype=np.float32)
            rhoTarget[:, 0] = atomParams[:, 3] / 2.0  # Zpi/2 per spin
            rhoTarget[:, 1] = atomParams[:, 3] / 2.0
        self.d_rhoTarget = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                     hostbuf=rhoTarget.astype(np.float32))

        # --- Auxiliary buffers ---
        self.d_rhoQ    = cl.Buffer(self.ctx, mf.READ_WRITE, natoms * 16)
        self.d_gyLocal = cl.Buffer(self.ctx, mf.READ_WRITE, natoms * NS * NCH * 4)
        self.d_FLocal  = cl.Buffer(self.ctx, mf.READ_WRITE, natoms * 16)
        self.d_bondData = cl.Buffer(self.ctx, mf.READ_WRITE, natoms * MNE * 16)
        self.d_vel     = cl.Buffer(self.ctx, mf.READ_WRITE, natoms * 16)  # heavy-ball velocity
        cl.enqueue_copy(self.queue, self.d_vel, np.zeros((natoms, 4), dtype=np.float32))

        if bUseCoulomb:
            self.d_phi   = cl.Buffer(self.ctx, mf.READ_WRITE, natoms * 4)
            self.d_FCoul = cl.Buffer(self.ctx, mf.READ_WRITE, natoms * 16)
        else:
            self.d_phi   = cl.Buffer(self.ctx, mf.READ_WRITE, natoms * 4)
            self.d_FCoul = cl.Buffer(self.ctx, mf.READ_WRITE, natoms * 16)

    def step(self, dtY=0.05, dtR=0.0, momentum=0.0, lambdaGlobal=0.0, bLocalSimplex=True):
        """Run one full 3-kernel iteration.

        Parameters
        ----------
        dtY : float
            Electronic step size.
        dtR : float
            Geometry step size (0 = no position update).
        momentum : float
            Heavy-ball damping coefficient (0 = plain Euler, ~0.9 = damped).
        """
        flags = 0
        if bLocalSimplex:
            flags |= self.KQ_FLAG_LOCAL_SIMPLEX
        if self.bUpdateR:
            flags |= self.KQ_FLAG_UPDATE_R
        if self.bUseCoulomb:
            flags |= self.KQ_FLAG_USE_COULOMB_ELECTRON
            flags |= self.KQ_FLAG_USE_COULOMB_FORCE

        gs = (self.natoms,)

        # Kernel 1: gather local bonding forces and gradients
        self.k_gather_local(
            self.queue, gs, None,
            np.int32(self.natoms),
            self.d_y, self.d_R, self.d_neigh, self.d_rev,
            self.d_eps0, self.d_beta, self.d_r0, self.d_Kr,
            self.d_atomParams,
            self.d_rhoQ, self.d_gyLocal, self.d_FLocal, self.d_bondData
        )

        # Kernel 2: gather nonlocal Coulomb (always run, but flags control usage)
        kCoul = np.float32(14.3996)
        soft2 = np.float32(0.0)
        cut2  = np.float32(0.0)  # no cutoff
        coul_wg = (128,)
        coul_gs = (max(128, ((self.natoms + 127) // 128) * 128),)
        self.k_gather_coulomb(
            self.queue, coul_gs, coul_wg,
            np.int32(self.natoms),
            self.d_R, self.d_rhoQ,
            self.d_phi, self.d_FCoul,
            kCoul, soft2, cut2
        )

        # Kernel 3: update DOFs
        self.k_update_dofs(
            self.queue, gs, None,
            np.int32(self.natoms),
            self.d_y, self.d_R, self.d_vel, self.d_neigh,
            self.d_gyLocal, self.d_FLocal, self.d_FCoul,
            self.d_phi, self.d_rhoTarget,
            np.float32(dtY), np.float32(dtR),
            np.float32(momentum),
            np.float32(lambdaGlobal), np.int32(flags)
        )

    def relax(self, nsteps=500, dtY=0.05, dtR=0.0, momentum=0.0, verbose=False, check_every=50,
              n_elec_substeps=1):
        """Run many iterations and optionally print convergence.

        Parameters
        ----------
        n_elec_substeps : int
            Number of electronic-only steps per geometry step. The electronic
            system relaxes faster, so sub-stepping keeps it close to equilibrium
            while geometry slowly evolves.
        """
        for it in range(nsteps):
            if dtR != 0.0 and n_elec_substeps > 1:
                # Electronic sub-steps without geometry update
                for _ in range(n_elec_substeps - 1):
                    self.step(dtY=dtY, dtR=0.0, momentum=0.0)
                # Final step with geometry update
                self.step(dtY=dtY, dtR=dtR, momentum=momentum)
            else:
                self.step(dtY=dtY, dtR=dtR, momentum=momentum)
            if verbose and (it % check_every == 0 or it == nsteps - 1):
                rhoQ, bondData = self.download_state()
                Q = rhoQ[:, 2]
                maxQ = np.max(np.abs(Q))
                print(f"  iter {it:4d}  max|Q|={maxQ:.6e}")

    def download_state(self):
        """Download current rhoQ and bondData from GPU."""
        rhoQ = np.empty((self.natoms, 4), dtype=np.float32)
        cl.enqueue_copy(self.queue, rhoQ, self.d_rhoQ)
        bondData = np.empty((self.natoms, self.KQ_MAX_NEIGH, 4), dtype=np.float32)
        cl.enqueue_copy(self.queue, bondData, self.d_bondData)
        return rhoQ, bondData

    def download_y(self):
        """Download the persistent electronic state y."""
        y = np.empty(self.natoms * self.KQ_NSPIN * self.KQ_NCHAN, dtype=np.float32)
        cl.enqueue_copy(self.queue, y, self.d_y)
        return y.reshape(self.natoms, self.KQ_NSPIN, self.KQ_NCHAN)

    def download_R(self):
        """Download current positions."""
        R = np.empty((self.natoms, 4), dtype=np.float32)
        cl.enqueue_copy(self.queue, R, self.d_R)
        return R[:, :3]

    def compute_bond_orders(self, neigh, rev):
        """Compute B_ij = 2*sum_s sqrt(y[i,s,i->j]*y[j,s,j->i]) for each bond."""
        y = self.download_y()
        bonds = build_bond_list(neigh, self.KQ_MAX_NEIGH)
        B = np.zeros(len(bonds), dtype=np.float64)
        for idx, (i, j) in enumerate(bonds):
            # Find slot of j in i's neighbor list and vice versa
            ki = -1
            for k in range(self.KQ_MAX_NEIGH):
                if neigh[i, k] == j:
                    ki = k
                    break
            kj = -1
            for k in range(self.KQ_MAX_NEIGH):
                if neigh[j, k] == i:
                    kj = k
                    break
            if ki < 0 or kj < 0:
                continue
            for s in range(self.KQ_NSPIN):
                yi = y[i, s, 1 + ki]
                yj = y[j, s, 1 + kj]
                B[idx] += 2.0 * np.sqrt(max(yi * yj, 0.0))
        return bonds, B

    def compute_bond_lengths(self, bonds):
        """Compute current bond lengths from positions."""
        R = self.download_R()
        lengths = np.zeros(len(bonds), dtype=np.float64)
        for idx, (i, j) in enumerate(bonds):
            lengths[idx] = np.linalg.norm(R[i] - R[j])
        return lengths


# =====================================================================
# 5. PLOTTING
# =====================================================================

def plot_molecule(positions, bonds, bond_orders, bond_lengths=None,
                  enames=None, n_pi=None, ax=None, title="Kekulé QM Result",
                  show_bond_lengths=True, atom_label_mode='element'):
    """Plot the molecule with bond widths proportional to bond order.

    Bond color encodes localization:
      - red    = double bond (B ~ 1)
      - blue   = single bond (B ~ 0)
      - green  = aromatic    (B ~ 0.5)

    atom_label_mode : str
      'none'     — no atom labels
      'element'  — element symbol at each vertex
      'index'    — atom index number at each vertex
      'element_pi' — element + n_pi (e.g. "C1", "N1")
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Classify bonds — use lighter colors so black labels are visible
    colors = []
    for B in bond_orders:
        if B > 0.75:
            colors.append('#e74c3c')    # light red for double
        elif B < 0.25:
            colors.append('#3498db')    # light blue for single
        else:
            colors.append('#2ecc71')    # light green for aromatic

    # Line widths proportional to bond order
    lws = 1.0 + bond_orders * 4.0

    # Draw bonds
    for idx, (i, j) in enumerate(bonds):
        ax.plot([positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                color=colors[idx], linewidth=lws[idx], alpha=0.7, zorder=1)
        if bond_lengths is not None and show_bond_lengths:
            mx = (positions[i, 0] + positions[j, 0]) / 2
            my = (positions[i, 1] + positions[j, 1]) / 2
            ax.text(mx, my, f'{bond_lengths[idx]:.2f}', fontsize=6,
                    ha='center', va='center', color='#555555', zorder=2)

    # Draw atoms — light-colored circles with dark edge so black text is readable
    atom_colors = {'C': '#bbbbbb', 'N': '#aaccff', 'O': '#ffaaaa', 'H': '#dddddd'}
    if enames is not None:
        for i, e in enumerate(enames):
            c = atom_colors.get(e, '#bbbbbb')
            ax.scatter(positions[i, 0], positions[i, 1], c=c, s=200, zorder=3,
                       edgecolors='black', linewidths=0.8)
    else:
        ax.scatter(positions[:, 0], positions[:, 1], c='#bbbbbb', s=200, zorder=3,
                   edgecolors='black', linewidths=0.8)

    # Atom labels — black text on light atom circles
    if atom_label_mode != 'none' and enames is not None:
        for i in range(len(positions)):
            x, y = positions[i, 0], positions[i, 1]
            if atom_label_mode == 'element':
                label = enames[i] if i < len(enames) else '?'
            elif atom_label_mode == 'index':
                label = str(i)
            elif atom_label_mode == 'element_pi':
                el = enames[i] if i < len(enames) else '?'
                np_str = str(int(n_pi[i])) if n_pi is not None and i < len(n_pi) else '?'
                label = f"{el}{np_str}"
            else:
                label = ''
            if label:
                ax.text(x, y, label, fontsize=8, ha='center', va='center',
                        color='black', fontweight='bold', zorder=4)

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color='#e74c3c', lw=3, label='double (B>0.75)'),
        Line2D([0], [0], color='#2ecc71', lw=2, label='aromatic (0.25-0.75)'),
        Line2D([0], [0], color='#3498db', lw=1, label='single (B<0.25)'),
    ]
    ax.legend(handles=handles, loc='lower left', fontsize=8)

    return ax


def plot_convergence(solver, neigh, rev, nsteps=500, dtY=0.05, dtR=0.0,
                     momentum=0.0, n_elec_substeps=1,
                     check_every=10, enames=None, n_pi=None,
                     atom_label_mode='element', show_bond_lengths=True,
                     R_init=None):
    """Run relaxation and plot bond order convergence + initial vs final state.

    If R_init is provided and positions changed, displacement arrows are drawn
    on the final plot showing how each atom moved from initial to final.
    """
    # Capture initial bond orders before any steps
    bonds, B_init = solver.compute_bond_orders(neigh, rev)
    R_before = solver.download_R()

    bond_hist = []
    for it in range(nsteps):
        if dtR != 0.0 and n_elec_substeps > 1:
            for _ in range(n_elec_substeps - 1):
                solver.step(dtY=dtY, dtR=0.0, momentum=0.0)
            solver.step(dtY=dtY, dtR=dtR, momentum=momentum)
        else:
            solver.step(dtY=dtY, dtR=dtR, momentum=momentum)
        if it % check_every == 0 or it == nsteps - 1:
            bonds, B = solver.compute_bond_orders(neigh, rev)
            bond_hist.append(B.copy())

    R_final = solver.download_R()
    bond_lengths = solver.compute_bond_lengths(bonds)
    B_final = bond_hist[-1]

    # Check if positions actually changed
    max_disp = np.max(np.linalg.norm(R_final - R_before, axis=1)) if R_init is not None else 0.0
    bMoved = (dtR != 0.0) and (max_disp > 1e-6)

    # 3-panel layout: convergence | initial | final
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: bond order convergence
    ax1 = axes[0]
    bond_hist_arr = np.array(bond_hist)
    for idx in range(len(bonds)):
        ax1.plot(np.arange(len(bond_hist_arr)) * check_every, bond_hist_arr[:, idx],
                 alpha=0.5, linewidth=0.8)
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.3, label='aromatic')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Bond order B_ij')
    ax1.set_title('Bond order convergence')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.5)
    ax1.legend(fontsize=7, loc='lower right')

    # Plot 2: initial state
    ax2 = axes[1]
    init_lengths = np.zeros(len(bonds))
    for idx, (i, j) in enumerate(bonds):
        init_lengths[idx] = np.linalg.norm(R_before[i] - R_before[j])
    plot_molecule(R_before, bonds, B_init, bond_lengths=init_lengths,
                  enames=enames, n_pi=n_pi, ax=ax2,
                  title=f'Initial guess (iter 0)',
                  show_bond_lengths=show_bond_lengths,
                  atom_label_mode=atom_label_mode)

    # Plot 3: final state (with displacement arrows if atoms moved)
    ax3 = axes[2]
    plot_molecule(R_final, bonds, B_final, bond_lengths=bond_lengths,
                  enames=enames, n_pi=n_pi, ax=ax3,
                  title=f'Final state (iter {nsteps})',
                  show_bond_lengths=show_bond_lengths,
                  atom_label_mode=atom_label_mode)

    if bMoved and R_init is not None:
        # Draw displacement arrows from initial to final position
        disp = R_final - R_before
        scale = 1.0
        # Auto-scale arrows to be visible but not overwhelming
        max_d = np.max(np.linalg.norm(disp[:, :2], axis=1))
        if max_d > 0:
            scale = min(0.3 / max_d, 5.0)  # cap at 5x
        for i in range(len(R_final)):
            dx = disp[i, 0] * scale
            dy = disp[i, 1] * scale
            if abs(dx) > 1e-5 or abs(dy) > 1e-5:
                ax3.annotate('', xy=(R_final[i, 0] + dx, R_final[i, 1] + dy),
                             xytext=(R_final[i, 0], R_final[i, 1]),
                             arrowprops=dict(arrowstyle='->', color='magenta', lw=1.5, alpha=0.8))
        ax3.set_title(f'Final state (iter {nsteps})\nmax disp={max_disp:.4f} Å (arrows {scale:.1f}x)')

    plt.tight_layout()
    return fig, bonds, B_final, bond_lengths, B_init


# =====================================================================
# 6. MAIN DEMO
# =====================================================================

def main(example='naphthalene', nsteps=500, dtY=0.05, bUseCoulomb=False, bUpdateR=False,
         atom_label_mode='element', show_bond_lengths=True,
         eps0_val=2.7, beta_val=3.0, Kr_val=20.0, r0_bond=1.42,
         chi_C=0.0, eta_C=0.0, U_C=0.0, Zpi_C=1.0,
         chi_N=-1.5, eta_N=0.0, U_N=0.0, Zpi_N=1.0,
         chi_O=-2.5, eta_O=0.0, U_O=0.0, Zpi_O=1.0,
         dtR=0.0, kCoul=14.3996, momentum=0.9, n_elec_substeps=10):
    """Demo: load molecule from ASCII art, solve on GPU, and plot.

    Parameters
    ----------
    example : str
        Key into ASCII_EXAMPLES (e.g. 'naphthalene', 'purin', 'uracil')
        or 'flake' for a generated graphene flake.
    atom_label_mode : str
        'none', 'element', 'index', or 'element_pi'
    """
    print(f"=== KekuleQM OpenCL Demo: {example} ===")

    n_pi = None
    if example == 'flake':
        enames, positions = generate_graphene_flake(radius=5.0, a=1.42, noise=0.01)
        neigh, rev = build_neighbor_lists(positions, cutoff=1.8, max_neigh=3)
        bonds = build_bond_list(neigh, max_neigh=3)
    else:
        if not HAS_CHEMICAL_GRAPHS:
            raise RuntimeError("ChemicalGraphs package required for ASCII art examples")
        art = ASCII_EXAMPLES[example]
        enames, positions, bonds = from_ascii_art(art, bHydrogens=False)
        neigh, rev = build_neighbor_lists_from_bonds(positions, bonds, max_neigh=3)
        # Get n_pi for labeling
        from ChemicalGraphs import make_n_pi as _make_n_pi
        from ChemicalGraphs import parse_ascii_art as _parse
        _atoms = _parse(art, hbond_length=3.0)
        _atoms.neighs()
        n_pi_full = _make_n_pi(_atoms)
        heavy_mask = np.array([e not in ('H', 'E') for e in _atoms.enames], dtype=bool)
        n_pi = n_pi_full[heavy_mask]

    natoms = len(positions)
    print(f"System: {natoms} atoms, {len(bonds)} bonds")
    print(f"Elements: {np.unique(enames)}")

    # Build parameters
    eps0, beta, r0, Kr, atomParams = default_params(
        natoms, neigh, enames, r0_bond=r0_bond,
        eps0_val=eps0_val, beta_val=beta_val, Kr_val=Kr_val,
        chi_C=chi_C, eta_C=eta_C, U_C=U_C, Zpi_C=Zpi_C,
        chi_N=chi_N, eta_N=eta_N, U_N=U_N, Zpi_N=Zpi_N,
        chi_O=chi_O, eta_O=eta_O, U_O=U_O, Zpi_O=Zpi_O,
    )

    # Initialize solver
    solver = KekuleQM_OCL()
    solver.setup(positions, neigh, rev, eps0, beta, r0, Kr, atomParams,
                 bUseCoulomb=bUseCoulomb, bUpdateR=bUpdateR)

    # Relax
    print("Relaxing electronic DOFs...")
    if bUpdateR and dtR == 0.0:
        dtR = 0.0005  # auto-set a small geometry step
        print(f"  (auto-set dtR={dtR} for geometry relaxation)")
    fig, bonds_out, B_final, lengths, B_init = plot_convergence(
        solver, neigh, rev, nsteps=nsteps, dtY=dtY, dtR=dtR,
        momentum=momentum, n_elec_substeps=n_elec_substeps,
        check_every=max(1, nsteps//50),
        enames=enames, n_pi=n_pi, atom_label_mode=atom_label_mode,
        show_bond_lengths=show_bond_lengths,
        R_init=positions.copy(),
    )

    print(f"\nInitial bond orders: min={B_init.min():.3f} max={B_init.max():.3f} mean={B_init.mean():.3f}")
    print(f"Final bond orders:   min={B_final.min():.3f} max={B_final.max():.3f} mean={B_final.mean():.3f}")
    print(f"Final bond lengths:  min={lengths.min():.3f} max={lengths.max():.3f} mean={lengths.mean():.3f}")

    # Print per-bond results with initial->final comparison
    for idx, (i, j) in enumerate(bonds_out):
        print(f"  bond {i:2d}-{j:2d}  B: {B_init[idx]:.3f} -> {B_final[idx]:.3f}  L: {lengths[idx]:.3f}")

    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='KekuleQM OpenCL solver',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # System selection
    parser.add_argument('--example', '-e', default='naphthalene',
                        help='ASCII art example name (or "flake")')
    # Solver parameters
    parser.add_argument('--nsteps', type=int, default=500, help='Number of iterations')
    parser.add_argument('--dtY', type=float, default=0.05, help='Electronic step size')
    parser.add_argument('--dtR', type=float, default=0.0, help='Geometry step size (0=no geometry relaxation)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Heavy-ball momentum for geometry (0=plain Euler, 0.9=damped)')
    parser.add_argument('--elec_substeps', type=int, default=10, help='Electronic sub-steps per geometry step')
    parser.add_argument('--coulomb', type=int, default=0, help='Use Coulomb electrostatics (1=yes)')
    parser.add_argument('--updateR', type=int, default=0, help='Update atomic positions (1=yes)')
    parser.add_argument('--kCoul', type=float, default=14.3996, help='Coulomb constant (eV*A/e^2)')
    # Plotting
    parser.add_argument('--atom_labels', default='element',
                        choices=['none', 'element', 'index', 'element_pi'],
                        help='Atom label mode')
    parser.add_argument('--no_bond_lengths', action='store_true',
                        help='Hide bond length labels on plot')
    # Model parameters - bonding
    parser.add_argument('--eps0', type=float, default=2.7, help='Hopping strength (eV)')
    parser.add_argument('--beta', type=float, default=3.0, help='Peierls decay rate (1/Angstrom)')
    parser.add_argument('--Kr', type=float, default=20.0, help='Sigma bond stiffness (eV/Angstrom^2)')
    parser.add_argument('--r0', type=float, default=1.42, help='Equilibrium bond length (Angstrom)')
    # Model parameters - per-element QEq/Hubbard
    parser.add_argument('--chi_C', type=float, default=0.0, help='Carbon electronegativity chi')
    parser.add_argument('--eta_C', type=float, default=0.0, help='Carbon hardness eta')
    parser.add_argument('--U_C', type=float, default=0.0, help='Carbon Hubbard U')
    parser.add_argument('--Zpi_C', type=float, default=1.0, help='Carbon pi electron count')
    parser.add_argument('--chi_N', type=float, default=-1.5, help='Nitrogen electronegativity chi')
    parser.add_argument('--eta_N', type=float, default=0.0, help='Nitrogen hardness eta')
    parser.add_argument('--U_N', type=float, default=0.0, help='Nitrogen Hubbard U')
    parser.add_argument('--Zpi_N', type=float, default=1.0, help='Nitrogen pi electron count')
    parser.add_argument('--chi_O', type=float, default=-2.5, help='Oxygen electronegativity chi')
    parser.add_argument('--eta_O', type=float, default=0.0, help='Oxygen hardness eta')
    parser.add_argument('--U_O', type=float, default=0.0, help='Oxygen Hubbard U')
    parser.add_argument('--Zpi_O', type=float, default=1.0, help='Oxygen pi electron count')
    args = parser.parse_args()
    main(example=args.example, nsteps=args.nsteps, dtY=args.dtY, dtR=args.dtR,
         bUseCoulomb=bool(args.coulomb), bUpdateR=bool(args.updateR),
         atom_label_mode=args.atom_labels, show_bond_lengths=not args.no_bond_lengths,
         eps0_val=args.eps0, beta_val=args.beta, Kr_val=args.Kr, r0_bond=args.r0,
         chi_C=args.chi_C, eta_C=args.eta_C, U_C=args.U_C, Zpi_C=args.Zpi_C,
         chi_N=args.chi_N, eta_N=args.eta_N, U_N=args.U_N, Zpi_N=args.Zpi_N,
         chi_O=args.chi_O, eta_O=args.eta_O, U_O=args.U_O, Zpi_O=args.Zpi_O,
         kCoul=args.kCoul, momentum=args.momentum, n_elec_substeps=args.elec_substeps)
