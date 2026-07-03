"""Model A: Kekulé fluid / dimer-smoke solver using pyOpenCL.

Physical background
-------------------
The Kekulé order parameter z_i is a complex number defined on each atom that
encodes the local bond-order alternation (dimerization) pattern.  Its amplitude
|z_i| measures the strength of alternation; its phase arg(z_i) selects which of
the three Kekulé patterns is locally preferred (phases 0, 2π/3, 4π/3 correspond
to the three bond directions).

z_i evolves according to a modified complex Ginzburg–Landau equation:

    ∂z/∂t = κ∇²z + r·z − u·|z|²·z − λ·z*²  −  η·g  −  iΩ·g

where g = ∂F/∂z* is the derivative of the free energy, κ is the spatial
stiffness (graph Laplacian), r<0 drives spontaneous ordering, u saturates the
amplitude, λ introduces Z₃ anisotropy, η provides dissipation, and Ω gives
conservative phase rotation.

Role in the system
------------------
Model A is the first link in the causal chain:
  chemical defects / edge pins → bond orders x_ij → complex order parameter z_i
The bond orders x_ij produced here are fed into Model B (DiracLatticeSolver) to
modulate the hopping amplitudes and open a Kekulé mass gap.

Main algorithm (per timestep)
-----------------------------
1. `bondsToZ` — reconstruct z_i from bond orders x_ij (Fourier synthesis).
2. `applyPinsToZ` — enforce hard pinning at defect/pin sites.
3. `evolveZ_RK4` — 4th-order Runge–Kutta integration of the Ginzburg–Landau ODE.
4. `applyPinsToZ` — re-apply pins after RK4 drift.
5. `zToRawBonds` — project z_i back to raw bond orders x_ij^raw.
6. `copyRawToX` — copy raw bond orders to working buffer.
7. `projectBondOrders` — enforce valence constraint Σ_j x_ij = targetVal_i
   via red-black Gauss-Seidel with ABBA sweep ordering for mirror symmetry.
8. `bondsToZ` — final reconstruction for diagnostics.

Key class
---------
- `KekuleFluidSolver` — main solver class:
    - `init_uniform_kekule(phi0)` — initialize z with uniform phase phi0.
    - `add_proton_defect(atom_idx, pin_dir, pin_strength)` — set up H⁺ defect.
    - `step()` — one complete timestep (all 8 stages above).
    - `run(nsteps, callback)` — run nsteps with optional callback.
    - `get_z()`, `get_bond_x()`, `get_state()` — read GPU state to numpy.
    - `compute_diagnostics()` — amplitude stats, valence residuals, bond stats.
"""
import numpy as np
import pyopencl as cl
import os

kernel_source = None

def _load_kernels():
    global kernel_source
    if kernel_source is None:
        cl_path = os.path.join(os.path.dirname(__file__), 'kekule_fluid.cl')
        with open(cl_path, 'r') as f:
            kernel_source = f.read()
    return kernel_source


class KekuleFluidSolver:
    """OpenCL solver for Model A: atom-bond Kekulé fluid."""

    DEFAULT_PARAMS = dict(
        kappa=0.20, r=-1.00, u=1.00, lambda_=0.15,
        eta=0.08, Omega=1.00, k_pin=2.00, k_core=5.00,
        dtA=0.02, A_pin=1.0, nProj=10,
        kekule_amp=0.8, omega_proj=0.5,
    )

    def __init__(self, graph, ctx=None, queue=None, params=None):
        self.graph = graph
        if ctx is None:
            # Auto-select NVIDIA platform if available
            for plat in cl.get_platforms():
                if 'NVIDIA' in plat.name.upper():
                    ctx = cl.Context(properties=[(cl.context_properties.PLATFORM, plat)])
                    break
            if ctx is None:
                ctx = cl.create_some_context()
        self.ctx = ctx
        self.queue = queue or cl.CommandQueue(self.ctx)

        src = _load_kernels()
        # Inject natom/nbond as preprocessor defines for the kernels that use them directly
        src_with_defs = f"#define NATOM {graph.natom}\n#define NBOND {graph.nbond}\n" + src
        self.prg = cl.Program(self.ctx, src_with_defs).build()

        # Cache kernel objects to avoid repeated retrieval overhead
        self._kernels = {}
        for name in ['bondsToZ', 'applyPinsToZ', 'rhsZ', 'rk4Combine',
                      'rk4Intermediate', 'rk4IntermediateHalf',
                      'zToRawBonds', 'copyRawToX', 'copyBonds', 'projectBondOrdersSub']:
            self._kernels[name] = cl.Kernel(self.prg, name)

        # Merge params
        p = dict(self.DEFAULT_PARAMS)
        if params:
            p.update(params)

        # Pack params struct: must match the C struct in the .cl file
        # struct: kappa, r, u, lambda, eta, Omega, k_pin, k_core, dtA, A_pin, kekule_amp, omega_proj
        self.params_arr = np.array([
            p['kappa'], p['r'], p['u'], p['lambda_'],
            p['eta'], p['Omega'], p['k_pin'], p['k_core'],
            p['dtA'], p['A_pin'],
            p['kekule_amp'], p['omega_proj']
        ], dtype=np.float32)
        self.params_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                    hostbuf=self.params_arr)

        self._alloc_buffers()
        self.natom = graph.natom
        self.nbond = graph.nbond
        self.nProj = int(p['nProj'])
        self.dtA = p['dtA']
        self.omega_proj = float(p['omega_proj'])

    def _alloc_buffers(self):
        ctx = self.ctx
        arrs = self.graph.get_arrays()

        def mk_rw(arr):
            return cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=arr)

        def mk_ro(arr):
            return cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=arr)

        # Read-only graph structure
        self.b_neighbors = mk_ro(arrs['neighbors'])
        self.b_atom_bonds = mk_ro(arrs['atom_bonds'])
        self.b_bond_iA = mk_ro(arrs['bond_iA'])
        self.b_bond_iB = mk_ro(arrs['bond_iB'])
        self.b_bond_dir = mk_ro(arrs['bond_dir'])
        self.b_bondBase = mk_ro(arrs['bondBase'])
        self.b_sub = mk_ro(arrs['sub'])

        # Read-only atom properties (can be updated by host)
        self.b_defect = mk_ro(arrs['defect'])
        self.b_pinStrength = mk_ro(arrs['pinStrength'])
        self.b_pinPhase = mk_ro(arrs['pinPhase'])
        self.b_targetVal = mk_ro(arrs['targetVal'])

        # Read-write state
        self.b_z_real = mk_rw(np.zeros(self.graph.natom, dtype=np.float32))
        self.b_z_imag = mk_rw(np.zeros(self.graph.natom, dtype=np.float32))
        self.b_bond_x = mk_rw(arrs['bond_x'].copy())
        self.b_bond_x2 = mk_rw(arrs['bond_x'].copy())  # ping-pong buffer for projection
        self.b_bond_xRaw = mk_rw(arrs['bond_xRaw'].copy())

        # Temp buffers for RK4
        self.b_k1r = mk_rw(np.zeros(self.graph.natom, dtype=np.float32))
        self.b_k1i = mk_rw(np.zeros(self.graph.natom, dtype=np.float32))
        self.b_k2r = mk_rw(np.zeros(self.graph.natom, dtype=np.float32))
        self.b_k2i = mk_rw(np.zeros(self.graph.natom, dtype=np.float32))
        self.b_k3r = mk_rw(np.zeros(self.graph.natom, dtype=np.float32))
        self.b_k3i = mk_rw(np.zeros(self.graph.natom, dtype=np.float32))
        self.b_k4r = mk_rw(np.zeros(self.graph.natom, dtype=np.float32))
        self.b_k4i = mk_rw(np.zeros(self.graph.natom, dtype=np.float32))

        self.b_z_temp_r = mk_rw(np.zeros(self.graph.natom, dtype=np.float32))
        self.b_z_temp_i = mk_rw(np.zeros(self.graph.natom, dtype=np.float32))

    def _update_atom_props(self):
        """Re-upload defect/pin/targetVal buffers after host-side changes."""
        arrs = self.graph.get_arrays()
        cl.enqueue_copy(self.queue, self.b_defect, arrs['defect'])
        cl.enqueue_copy(self.queue, self.b_pinStrength, arrs['pinStrength'])
        cl.enqueue_copy(self.queue, self.b_pinPhase, arrs['pinPhase'])
        cl.enqueue_copy(self.queue, self.b_targetVal, arrs['targetVal'])
        cl.enqueue_copy(self.queue, self.b_bondBase, arrs['bondBase'])

    def set_z(self, z_real, z_imag):
        cl.enqueue_copy(self.queue, self.b_z_real, z_real.astype(np.float32))
        cl.enqueue_copy(self.queue, self.b_z_imag, z_imag.astype(np.float32))

    def set_bond_x(self, bond_x):
        cl.enqueue_copy(self.queue, self.b_bond_x, bond_x.astype(np.float32))

    def get_z(self):
        zr = np.empty(self.graph.natom, dtype=np.float32)
        zi = np.empty(self.graph.natom, dtype=np.float32)
        cl.enqueue_copy(self.queue, zr, self.b_z_real)
        cl.enqueue_copy(self.queue, zi, self.b_z_imag)
        return zr, zi

    def get_bond_x(self):
        bx = np.empty(self.graph.nbond, dtype=np.float32)
        cl.enqueue_copy(self.queue, bx, self.b_bond_x)
        return bx

    def get_state(self):
        """Return full state as dict of numpy arrays."""
        zr, zi = self.get_z()
        bx = self.get_bond_x()
        bxRaw = np.empty(self.graph.nbond, dtype=np.float32)
        cl.enqueue_copy(self.queue, bxRaw, self.b_bond_xRaw)
        return dict(z_real=zr, z_imag=zi, bond_x=bx, bond_xRaw=bxRaw)

    def pull_to_graph(self):
        """Sync GPU state back to Python graph objects for diagnostics."""
        state = self.get_state()
        arrs = self.graph.get_arrays()
        arrs.update(state)
        self.graph.pull_from_arrays(arrs)

    # --- Core steps ---

    def bondsToZ(self):
        self._kernels['bondsToZ'](self.queue, (self.graph.natom,), None,
                          self.b_atom_bonds, self.b_bond_dir, self.b_bond_x,
                          self.b_z_real, self.b_z_imag)

    def applyPinsToZ(self):
        self._kernels['applyPinsToZ'](self.queue, (self.graph.natom,), None,
                              self.b_z_real, self.b_z_imag,
                              self.b_defect, self.b_pinStrength, self.b_pinPhase,
                              self.params_buf)

    def _rhsZ(self, z_r, z_i, out_r, out_i):
        self._kernels['rhsZ'](self.queue, (self.graph.natom,), None,
                      z_r, z_i, out_r, out_i,
                      self.b_neighbors, self.b_defect, self.b_pinStrength,
                      self.b_pinPhase, self.params_buf)

    def evolveZ_RK4(self):
        """RK4 integration of the complex Ginzburg-Landau equation."""
        na = self.graph.natom

        # Save z0 (device-to-device copy, no host round-trip)
        cl.enqueue_copy(self.queue, self.b_z_temp_r, self.b_z_real)
        cl.enqueue_copy(self.queue, self.b_z_temp_i, self.b_z_imag)

        # k1 = rhs(z0)
        self._rhsZ(self.b_z_real, self.b_z_imag, self.b_k1r, self.b_k1i)

        # z_temp = z0 + 0.5*dt*k1
        self._kernels['rk4IntermediateHalf'](self.queue, (na,), None,
            self.b_z_temp_r, self.b_z_temp_i, self.b_k1r, self.b_k1i,
            self.b_z_real, self.b_z_imag, self.params_buf)

        # k2 = rhs(z_temp)
        self._rhsZ(self.b_z_real, self.b_z_imag, self.b_k2r, self.b_k2i)

        # z_temp = z0 + 0.5*dt*k2
        self._kernels['rk4IntermediateHalf'](self.queue, (na,), None,
            self.b_z_temp_r, self.b_z_temp_i, self.b_k2r, self.b_k2i,
            self.b_z_real, self.b_z_imag, self.params_buf)

        # k3 = rhs(z_temp)
        self._rhsZ(self.b_z_real, self.b_z_imag, self.b_k3r, self.b_k3i)

        # z_temp = z0 + dt*k3
        self._kernels['rk4Intermediate'](self.queue, (na,), None,
            self.b_z_temp_r, self.b_z_temp_i, self.b_k3r, self.b_k3i,
            self.b_z_real, self.b_z_imag, self.params_buf)

        # k4 = rhs(z_temp)
        self._rhsZ(self.b_z_real, self.b_z_imag, self.b_k4r, self.b_k4i)

        # z_new = z0 + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        self._kernels['rk4Combine'](self.queue, (na,), None,
            self.b_z_temp_r, self.b_z_temp_i,
            self.b_k1r, self.b_k1i, self.b_k2r, self.b_k2i,
            self.b_k3r, self.b_k3i, self.b_k4r, self.b_k4i,
            self.b_z_real, self.b_z_imag, self.params_buf)

    def zToRawBonds(self):
        self._kernels['zToRawBonds'](self.queue, (self.graph.nbond,), None,
                             self.b_bond_iA, self.b_bond_iB, self.b_bond_dir,
                             self.b_z_real, self.b_z_imag, self.b_bondBase,
                             self.b_bond_xRaw, self.params_buf)

    def copyRawToX(self):
        self._kernels['copyRawToX'](self.queue, (self.graph.nbond,), None,
                            self.b_bond_x, self.b_bond_xRaw)

    def projectBondOrders(self):
        """Race-free ping-pong valence projection using bipartite red-black sweeps."""
        na = self.graph.natom
        nb = self.graph.nbond
        for _ in range(self.nProj):
            for target_sub in (1, -1, -1, 1):
                self._kernels['copyBonds'](self.queue, (nb,), None,
                                           self.b_bond_x, self.b_bond_x2)
                self._kernels['projectBondOrdersSub'](
                    self.queue, (na,), None,
                    self.b_bond_x, self.b_bond_x2,
                    self.b_atom_bonds, self.b_targetVal, self.b_sub,
                    np.int32(target_sub), np.float32(self.omega_proj))
                self.b_bond_x, self.b_bond_x2 = self.b_bond_x2, self.b_bond_x

    def step(self):
        """One complete Model A timestep."""
        self.bondsToZ()
        self.applyPinsToZ()
        self.evolveZ_RK4()
        self.applyPinsToZ()  # re-apply hard pins after RK4 drift
        self.zToRawBonds()
        self.copyRawToX()
        self.projectBondOrders()
        self.bondsToZ()  # final reconstruction for diagnostics

    def run(self, nsteps, callback=None, callback_interval=10):
        """Run nsteps timesteps. Optional callback called every callback_interval."""
        for step in range(nsteps):
            self.step()
            if callback is not None and (step % callback_interval == 0 or step == nsteps - 1):
                callback(self, step)

    def compute_diagnostics(self):
        """Return a dict of physical diagnostics computed from current GPU state."""
        zr, zi = self.get_z()
        bx = self.get_bond_x()
        z = zr + 1j * zi
        amp = np.abs(z)

        # Valence residual
        atom_bonds = self.graph.get_atom_bonds()
        target_val = np.array([a.targetVal for a in self.graph.atoms])
        valence = np.zeros(self.graph.natom)
        for i in range(self.graph.natom):
            for d in range(3):
                b = atom_bonds[i, d]
                if b >= 0:
                    valence[i] += bx[b]
        val_err = np.abs(valence - target_val)

        return dict(
            z_abs_min=amp.min(),
            z_abs_max=amp.max(),
            z_abs_mean=amp.mean(),
            valence_max_err=val_err.max(),
            valence_mean_err=val_err.mean(),
            n_low_amp=int((amp < 0.3 * amp.max()).sum()),
            bond_x_min=bx.min(),
            bond_x_max=bx.max(),
            bond_x_mean=bx.mean(),
        )

    # --- Initialization ---

    def init_uniform_kekule(self, phi0=0.0):
        z = np.ones(self.graph.natom, dtype=np.complex64) * np.exp(1j * phi0)
        self.set_z(z.real.astype(np.float32), z.imag.astype(np.float32))
        self.zToRawBonds()
        self.copyRawToX()
        self.projectBondOrders()
        self.bondsToZ()

    def init_vortices(self, vortices, r_core=None):
        """Initialize with vortex/antivortex pair(s).
        vortices: list of (pos_x, pos_y, winding)
        """
        if r_core is None:
            r_core = 2.0 * self.graph.aCC

        pos = np.array([[a.pos[0], a.pos[1]] for a in self.graph.atoms])
        phi = np.zeros(self.graph.natom)
        A = np.ones(self.graph.natom)

        for vx, vy, w in vortices:
            dx = pos[:, 0] - vx
            dy = pos[:, 1] - vy
            r = np.sqrt(dx**2 + dy**2)
            phi += w * np.arctan2(dy, dx)
            A *= np.tanh(r / r_core)

        z = A * np.exp(1j * phi)
        self.set_z(z.real.astype(np.float32), z.imag.astype(np.float32))
        self.zToRawBonds()
        self.copyRawToX()
        self.projectBondOrders()
        self.bondsToZ()

    def set_defect(self, atom_idx, defect=1.0, targetVal=None):
        """Set a defect on an atom and re-upload buffers."""
        self.graph.atoms[atom_idx].defect = defect
        if targetVal is not None:
            self.graph.atoms[atom_idx].targetVal = targetVal
        else:
            self.graph.atoms[atom_idx].targetVal = 1.0 - defect
        self.graph._compute_bond_base()
        self._update_atom_props()

    def set_pin(self, atom_idx, pinStrength=1.0, pinPhase=None, dir=None):
        """Pin an atom's Kekulé phase. If dir is given, pinPhase = thetaDir[dir]."""
        if dir is not None:
            pinPhase = self.graph.thetaDir[dir]
        if pinPhase is None:
            pinPhase = 0.0
        self.graph.atoms[atom_idx].pinStrength = pinStrength
        self.graph.atoms[atom_idx].pinPhase = pinPhase
        self._update_atom_props()

    def add_proton_defect(self, atom_idx, pin_dir=0, pin_strength=1.0,
                          pin_atom_idx=None):
        """Add an H+ (protonation) defect on an atom.

        Physically: H+ binds to the carbon, removing its pi electron.
        - defect = 1.0: full suppression of Kekulé order at this site
        - targetVal = 0.0: no pi electron budget
        - pinStrength = 0.0 on the defect atom (it has no pi orbital)

        Phase pinning is applied to a NEIGHBORING normal atom, not the
        protonated one.  This avoids the contradiction of simultaneously
        suppressing z→0 and pinning z→A_pin·exp(iφ) on the same atom.
        - pinPhase = thetaDir[pin_dir]: selects which of the 3 Kekulé patterns
          is favored near this defect, determining where nodal lines anchor

        pin_atom_idx: index of the atom to pin.  If None, auto-selects
            the nearest neighbor of the defect atom that is NOT itself
            a defect.
        """
        self.set_defect(atom_idx, defect=1.0, targetVal=0.0)
        # Do NOT pin the protonated atom — it has no pi orbital
        self.set_pin(atom_idx, pinStrength=0.0)

        # Pin a neighboring atom instead
        if pin_atom_idx is None:
            neighbors = self.graph.get_neighbor_list()
            best_d2 = 1e18
            for d in range(3):
                nb = neighbors[atom_idx, d]
                if nb < 0:
                    continue
                if self.graph.atoms[nb].defect > 0:
                    continue
                px, py = self.graph.atoms[nb].pos
                d2 = px*px + py*py
                if d2 < best_d2:
                    best_d2 = d2
                    pin_atom_idx = nb
        if pin_atom_idx is not None:
            self.set_pin(pin_atom_idx, pinStrength=pin_strength, dir=pin_dir)
