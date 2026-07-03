"""DiracLattice_ocl.py — Tight-binding quasiparticle propagation on honeycomb lattice.

Physical background
-------------------
On a honeycomb lattice (graphene), the π-electron band structure has two Dirac
cones (at K and K' points) that cross at zero energy.  A Kekulé bond-order
distortion opens a mass gap at these Dirac points, with the gap proportional to
the amplitude of the dimerization |z|.  Vortices in the Kekulé texture (where
|z|→0) host topologically protected in-gap states.

The tight-binding Hamiltonian for a single π electron is:

    H_{ij} = t₀ + δt·(x_ij − x₀_ij)    (nearest-neighbor hopping)
    H_{ii} = V_def · defect_i            (onsite potential at H⁺ defects)

where x_ij are the bond orders from Model A, x₀_ij is the aromatic baseline,
t₀ is the bare hopping, and δt is the Kekulé modulation strength.  H⁺ defects
are modeled by a large onsite potential V_def >> bandwidth, effectively removing
that atom from the π system.

Role in the system
------------------
Model B is the second link in the causal chain:
  bond orders x_ij (from Model A) → hopping t_ij → Hamiltonian H → eigenstates
It can be used in two modes:
1. **Real-time propagation**: RK4 integration of i∂ψ/∂t = Hψ on the GPU.
2. **Exact diagonalization**: numpy `eigh` for spectrum, eigenstates, and LDOS.

Key class
---------
- `DiracLatticeSolver` — main solver class:
    - `update_bond_x(bond_x)` — upload Model A bond orders to GPU.
    - `update_defect(defect_arr)` — upload onsite potentials (scaled by V_def).
    - `step()` — one RK4 timestep of i∂ψ/∂t = Hψ (GPU).
    - `run(nsteps, callback)` — run nsteps with optional callback.
    - `normalize()` — normalize ψ so Σ|ψ_i|² = 1.
    - `init_gaussian_on_atom(center_idx, sigma, kx, ky)` — Gaussian wavepacket.
    - `init_delta_atom(atom_idx)` — delta function on one atom.
    - `init_random(seed)` — random noise initialization.
    - `build_hamiltonian(bond_x, defect_arr)` — dense numpy H matrix.
    - `diagonalize(bond_x, defect_arr)` — exact eigenvalues and eigenvectors.
    - `compute_ldos(bond_x, defect_arr, sigma)` — local density of states.
    - `compute_diagnostics()` — density, sublattice polarization, norm.
"""
import numpy as np
import pyopencl as cl
import os
from typing import Optional, Tuple, List


class DiracLatticeSolver:
    """Tight-binding Dirac solver on a HoneycombGraph."""

    def __init__(self, graph, ctx=None, queue=None,
                 t0=1.0, dt_kek=0.5, dt=0.05, V_def=20.0):
        self.graph = graph
        if ctx is None:
            # Auto-select NVIDIA platform if available (same logic as ModelA)
            for plat in cl.get_platforms():
                if 'NVIDIA' in plat.name.upper():
                    ctx = cl.Context(properties=[(cl.context_properties.PLATFORM, plat)])
                    break
            if ctx is None:
                ctx = cl.create_some_context()
        self.ctx = ctx
        self.queue = queue or cl.CommandQueue(self.ctx)

        # Load and build kernel
        cl_path = os.path.join(os.path.dirname(__file__), 'DiracLattice.cl')
        with open(cl_path, 'r') as f:
            src = f.read()
        src = f"#define NATOM {graph.natom}\n#define NBOND {graph.nbond}\n" + src
        self.prg = cl.Program(self.ctx, src).build()

        # Cache kernels
        self._k_rhs = cl.Kernel(self.prg, 'rhs_psi')
        self._k_combine = cl.Kernel(self.prg, 'rk4_combine')
        self._k_half = cl.Kernel(self.prg, 'rk4_intermediate_half')
        self._k_full = cl.Kernel(self.prg, 'rk4_intermediate_full')

        # Pack params (must match DiracParams struct in DiracLattice.cl)
        self.params_arr = np.array([t0, dt_kek, dt], dtype=np.float32)
        self.params_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                    hostbuf=self.params_arr)
        self.t0 = t0
        self.dt_kek = dt_kek
        self.dt = dt
        self.V_def = V_def  # defect onsite potential (must be >> bandwidth ~6*t0)

        self._alloc_buffers()

    def _alloc_buffers(self):
        ctx = self.ctx
        arrs = self.graph.get_arrays()
        na = self.graph.natom

        def mk_ro(arr):
            return cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=arr)
        def mk_rw(arr):
            return cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=arr)

        # Graph structure (read-only)
        self.b_neighbors = mk_ro(arrs['neighbors'])
        self.b_atom_bonds = mk_ro(arrs['atom_bonds'])
        self.b_bondBase = mk_ro(arrs['bondBase'])
        self.b_defect = mk_ro(arrs['defect'] * self.V_def)

        # Bond orders — will be updated from Model A
        self.b_bond_x = mk_rw(arrs['bond_x'].copy())

        # Wavefunction state (read-write)
        self.b_psi_r = mk_rw(np.zeros(na, dtype=np.float32))
        self.b_psi_i = mk_rw(np.zeros(na, dtype=np.float32))

        # RK4 temp buffers
        self.b_k1r = mk_rw(np.zeros(na, dtype=np.float32))
        self.b_k1i = mk_rw(np.zeros(na, dtype=np.float32))
        self.b_k2r = mk_rw(np.zeros(na, dtype=np.float32))
        self.b_k2i = mk_rw(np.zeros(na, dtype=np.float32))
        self.b_k3r = mk_rw(np.zeros(na, dtype=np.float32))
        self.b_k3i = mk_rw(np.zeros(na, dtype=np.float32))
        self.b_k4r = mk_rw(np.zeros(na, dtype=np.float32))
        self.b_k4i = mk_rw(np.zeros(na, dtype=np.float32))

        self.b_tmp_r = mk_rw(np.zeros(na, dtype=np.float32))
        self.b_tmp_i = mk_rw(np.zeros(na, dtype=np.float32))

    def update_bond_x(self, bond_x):
        """Upload bond orders from Model A (or any source)."""
        cl.enqueue_copy(self.queue, self.b_bond_x, bond_x.astype(np.float32))

    def update_defect(self, defect_arr):
        """Upload onsite potentials (raw defect values, will be scaled by V_def in kernel).

        The defect array contains values in [0,1].  The kernel reads defect[i]
        directly as the onsite potential epsilon_i.  We pre-scale by V_def here
        so the kernel stays simple.
        """
        cl.enqueue_copy(self.queue, self.b_defect,
                        (defect_arr * self.V_def).astype(np.float32))

    def set_psi(self, psi_r, psi_i):
        cl.enqueue_copy(self.queue, self.b_psi_r, psi_r.astype(np.float32))
        cl.enqueue_copy(self.queue, self.b_psi_i, psi_i.astype(np.float32))

    def get_psi(self):
        pr = np.empty(self.graph.natom, dtype=np.float32)
        pi = np.empty(self.graph.natom, dtype=np.float32)
        cl.enqueue_copy(self.queue, pr, self.b_psi_r)
        cl.enqueue_copy(self.queue, pi, self.b_psi_i)
        return pr, pi

    def get_density(self):
        """Return |psi_i|^2 per atom."""
        pr, pi = self.get_psi()
        return pr**2 + pi**2

    def _rhs(self, psi_r, psi_i, out_r, out_i):
        self._k_rhs(self.queue, (self.graph.natom,), None,
                    psi_r, psi_i, out_r, out_i,
                    self.b_neighbors, self.b_atom_bonds,
                    self.b_bond_x, self.b_bondBase, self.b_defect,
                    self.params_buf)

    def step(self):
        """One RK4 timestep of i dpsi/dt = H psi."""
        na = self.graph.natom

        # Save psi0 to tmp
        cl.enqueue_copy(self.queue, self.b_tmp_r, self.b_psi_r)
        cl.enqueue_copy(self.queue, self.b_tmp_i, self.b_psi_i)

        # k1 = rhs(psi0)
        self._rhs(self.b_psi_r, self.b_psi_i, self.b_k1r, self.b_k1i)

        # psi_temp = psi0 + 0.5*dt*k1
        self._k_half(self.queue, (na,), None,
                     self.b_tmp_r, self.b_tmp_i, self.b_k1r, self.b_k1i,
                     self.b_psi_r, self.b_psi_i, self.params_buf)

        # k2 = rhs(psi_temp)
        self._rhs(self.b_psi_r, self.b_psi_i, self.b_k2r, self.b_k2i)

        # psi_temp = psi0 + 0.5*dt*k2
        self._k_half(self.queue, (na,), None,
                     self.b_tmp_r, self.b_tmp_i, self.b_k2r, self.b_k2i,
                     self.b_psi_r, self.b_psi_i, self.params_buf)

        # k3 = rhs(psi_temp)
        self._rhs(self.b_psi_r, self.b_psi_i, self.b_k3r, self.b_k3i)

        # psi_temp = psi0 + dt*k3
        self._k_full(self.queue, (na,), None,
                     self.b_tmp_r, self.b_tmp_i, self.b_k3r, self.b_k3i,
                     self.b_psi_r, self.b_psi_i, self.params_buf)

        # k4 = rhs(psi_temp)
        self._rhs(self.b_psi_r, self.b_psi_i, self.b_k4r, self.b_k4i)

        # psi_new = psi0 + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        self._k_combine(self.queue, (na,), None,
                        self.b_tmp_r, self.b_tmp_i,
                        self.b_k1r, self.b_k1i, self.b_k2r, self.b_k2i,
                        self.b_k3r, self.b_k3i, self.b_k4r, self.b_k4i,
                        self.b_psi_r, self.b_psi_i, self.params_buf)

    def run(self, nsteps, callback=None, callback_interval=10):
        for step in range(nsteps):
            self.step()
            if callback is not None and (step % callback_interval == 0 or step == nsteps - 1):
                callback(self, step)

    def normalize(self):
        """Normalize psi so that sum |psi_i|^2 = 1."""
        pr, pi = self.get_psi()
        norm = np.sqrt(np.sum(pr**2 + pi**2))
        if norm > 0:
            self.set_psi(pr / norm, pi / norm)

    # --- Initialization ---

    def init_gaussian_on_atom(self, center_idx, sigma=2.0,
                               kx=0.0, ky=0.0):
        """Initialize psi as a Gaussian envelope centered on an atom,
        with plane-wave phase."""
        pos = np.array([[a.pos[0], a.pos[1]] for a in self.graph.atoms])
        cx, cy = pos[center_idx]

        r2 = (pos[:, 0] - cx)**2 + (pos[:, 1] - cy)**2
        envelope = np.exp(-r2 / (2.0 * sigma**2))
        phase = np.exp(1j * (kx * pos[:, 0] + ky * pos[:, 1]))

        psi = envelope * phase
        self.set_psi(psi.real.astype(np.float32), psi.imag.astype(np.float32))
        self.normalize()

    def init_gaussian_at_pos(self, cx, cy, sigma=2.0, kx=0.0, ky=0.0):
        """Initialize psi as a Gaussian centered at world position (cx, cy)."""
        pos = np.array([[a.pos[0], a.pos[1]] for a in self.graph.atoms])
        r2 = (pos[:, 0] - cx)**2 + (pos[:, 1] - cy)**2
        envelope = np.exp(-r2 / (2.0 * sigma**2))
        phase = np.exp(1j * (kx * pos[:, 0] + ky * pos[:, 1]))

        psi = envelope * phase
        self.set_psi(psi.real.astype(np.float32), psi.imag.astype(np.float32))
        self.normalize()

    def init_delta_atom(self, atom_idx):
        """Initialize psi as a delta function on a single atom."""
        pr = np.zeros(self.graph.natom, dtype=np.float32)
        pi = np.zeros(self.graph.natom, dtype=np.float32)
        pr[atom_idx] = 1.0
        self.set_psi(pr, pi)

    def init_random(self, seed=42):
        """Initialize with random noise (for relaxation/spectrum studies)."""
        rng = np.random.RandomState(seed)
        psi = rng.randn(self.graph.natom) + 1j * rng.randn(self.graph.natom)
        self.set_psi(psi.real.astype(np.float32), psi.imag.astype(np.float32))
        self.normalize()

    # --- Diagnostics ---

    def compute_diagnostics(self):
        """Return density, sublattice polarization, total norm."""
        pr, pi = self.get_psi()
        psi = pr + 1j * pi
        density = np.abs(psi)**2

        # Sublattice polarization: rho_A - rho_B
        sub = np.array([a.sub for a in self.graph.atoms])
        rho_A = density[sub > 0].sum()
        rho_B = density[sub < 0].sum()
        pol = rho_A - rho_B

        norm = density.sum()

        return dict(
            psi_real=pr,
            psi_imag=pi,
            psi=psi,
            density=density,
            norm=norm,
            sublattice_polarization=pol,
            rho_A=rho_A,
            rho_B=rho_B,
        )

    # --- Eigenstate analysis (numpy, not OpenCL) ---

    def build_hamiltonian(self, bond_x=None, defect_arr=None):
        """Build the tight-binding Hamiltonian matrix (natom × natom).

        H_{ij} = t_ij for nearest neighbors
        H_{ii} = ε_i  (onsite potential)

        Kekulé-modulated hopping:
            t_ij = t₀ + δt · (x_ij - x₀_ij)

        where x₀_ij = bondBase is the aromatic baseline.

        Onsite potential:
            ε_i = V_def · defect_i

        where V_def is set at construction (default 20.0, >> bandwidth ~6t₀).
        For a protonated carbon (defect=1), this removes the atom from
        the π system by pushing its level far outside the band.

        Returns dense numpy array (real, since t_ij and ε_i are real).
        """
        n = self.graph.natom
        H = np.zeros((n, n), dtype=np.float64)

        if bond_x is None:
            bond_x = np.array([b.x for b in self.graph.bonds], dtype=np.float64)
        if defect_arr is None:
            defect_arr = np.array([a.defect for a in self.graph.atoms], dtype=np.float64)

        bondBase = self.graph.bondBase

        # Onsite terms: ε_i = V_def · defect_i
        for i in range(n):
            H[i, i] = self.V_def * defect_arr[i]

        # Hopping terms: t_ij = t₀ + δt · (x_ij - x₀_ij)
        for b_idx, bond in enumerate(self.graph.bonds):
            i, j = bond.iA, bond.iB
            t_ij = self.t0 + self.dt_kek * (bond_x[b_idx] - bondBase[b_idx])
            H[i, j] = t_ij
            H[j, i] = t_ij

        return H

    def diagonalize(self, bond_x=None, defect_arr=None):
        """Diagonalize the Hamiltonian. Returns (eigenvalues, eigenvectors).

        eigenvectors[:, k] is the k-th eigenstate (complex amplitudes per atom).
        """
        H = self.build_hamiltonian(bond_x, defect_arr)
        evals, evecs = np.linalg.eigh(H)  # H is Hermitian
        return evals, evecs

    def compute_ldos(self, bond_x=None, defect_arr=None, sigma=0.05):
        """Compute local density of states on each atom.

        LDOS_i(E) = Σ_k |ψ_k(i)|² · G(E - E_k, σ)

        where G(x, σ) = exp(-x²/2σ²) / (σ√2π) is a Gaussian broadening.

        Returns (energies, ldos, evals, evecs) where ldos has shape (natom, n_energies).
        """
        evals, evecs = self.diagonalize(bond_x, defect_arr)
        n = self.graph.natom

        Emin = evals.min() - 0.3
        Emax = evals.max() + 0.3
        nE = 500
        energies = np.linspace(Emin, Emax, nE)

        # Vectorized: |ψ_k(i)|² has shape (natom, nstates)
        weights = np.abs(evecs)**2  # (natom, nstates)

        # Gaussian: G[i,k,E] = weights[i,k] * exp(-(E - E_k)²/2σ²) / (σ√2π)
        # Use broadcasting: (1, nstates, 1) × (nstates, 1) → (natom, nstates, nE)
        dE = energies[np.newaxis, np.newaxis, :] - evals[np.newaxis, :, np.newaxis]  # (1, nstates, nE)
        gauss = np.exp(-dE**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))  # (1, nstates, nE)
        ldos = np.sum(weights[:, :, np.newaxis] * gauss, axis=1)  # (natom, nE)

        return energies, ldos, evals, evecs
