"""
SparseTruss_ocl.py — OpenCL GPU acceleration for truss elasticity solvers.

This module provides a SparseTrussOCL class that manages GPU buffers and
dispatches kernels for:

1. **Block Jacobi smoothing** — one workgroup per patch, local memory for
   x, b, Dinv.  Inner Jacobi iterations within the patch using CSR neighbor data.
2. **Multigrid restriction** — r_c = P^T * r (tall-skinny matvec with tree reduction)
3. **Multigrid prolongation** — x += P * e_c (short-wide matvec, one thread per node)
4. **Residual computation** — r = b - A*x (sparse CSR matvec)
5. **Coarse solve** — dense Cholesky solve on GPU (small L_COARSE x L_COARSE)

Data layout (CSR-like, per-node):
    neighs   : (N, NMAX_NEIGH) int32  — neighbor indices, -1 = void (sorted last)
    k_vals   : (N, NMAX_NEIGH) float32 — spring stiffness per neighbor slot
    n_dirs   : (N, NMAX_NEIGH, DIM) float32 — unit direction per edge
    nneigh   : (N) int32 — actual neighbor count per node
    mass_dt2 : (N) float32 — m_i / dt^2 per node

Patch mapping:
    patch_node_ids : (m, CLUSTER_SIZE) int32 — global node index per local slot
    patch_nneigh   : (m, CLUSTER_SIZE, NMAX_NEIGH) int32 — local neighbor indices
    patch_core_mask: (m, CLUSTER_SIZE) int8 — 1 = core (write back), 0 = halo

Prolongation P (tiled layout):
    P_global : (m * L_COARSE * CLUSTER_SIZE * DIM) float32
    Indexed as: P_global[p * L * CS * DIM + j * CS * DIM + a * DIM + d]

Compile-time parameters (substituted before compilation):
    WG_SIZE      — workgroup size (16, 32, or 64)
    CLUSTER_SIZE — max nodes per cluster (<= WG_SIZE)
    NMAX_NEIGH   — max neighbors per node (8 or 16)
    DIM          — spatial dimension (2 or 3)
    L_COARSE     — number of coarse modes
    N_STEPS      — inner Jacobi iterations (for block Jacobi kernel)
    OMEGA        — Jacobi damping factor

Role in the system
------------------
- **Truss.py**: geometry, mesh, assembly, bookkeeping.
- **TrussSolver.py**: CPU iterative and direct solvers.
- **MultiGrid.py**: CPU multigrid prolongation and solvers.
- **SparseTruss_ocl.py** (this file): GPU kernel management and dispatch.
- **kernels_block_jacobi.cl**: OpenCL C source for block Jacobi.
- **kernels_multigrid.cl**: OpenCL C source for restriction/prolongation.
- Scripts: thin wrappers that combine modules.
"""

import os
import time
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array


# ---------------------------------------------------------------------------
# Kernel source loading and template substitution
# ---------------------------------------------------------------------------

_KERNELS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_kernel_source(filename, defines):
    """Load a .cl file and substitute #define-style compile-time parameters.

    Parameters
    ----------
    filename : str — name of the .cl file in the same directory
    defines : dict — {name: value} to be injected as #define statements

    Returns
    -------
    str — the kernel source with defines prepended
    """
    path = os.path.join(_KERNELS_DIR, filename)
    with open(path, 'r') as f:
        source = f.read()

    define_lines = []
    for name, value in defines.items():
        define_lines.append(f"#define {name} {value}")
    header = "\n".join(define_lines) + "\n"
    return header + source


# ---------------------------------------------------------------------------
# SparseTrussOCL class
# ---------------------------------------------------------------------------

class SparseTrussOCL:
    """
    Manages OpenCL context, buffers, and kernel dispatch for truss elasticity.

    Usage
    -----
    >>> ocl = SparseTrussOCL(wg_size=32, cluster_size=32, nmax_neigh=8, dim=3)
    >>> ocl.upload_truss(neighs, k_vals, n_dirs, nneigh, mass_dt2)
    >>> ocl.upload_patches(patch_node_ids, patch_nneigh, patch_core_mask)
    >>> ocl.upload_prolongation(P_global, l_coarse)
    >>> ocl.upload_coarse_operator(A_c)
    >>> x_gpu = ocl.run_block_jacobi(x0, b, n_steps=5, omega=0.8)
    >>> r_c = ocl.restrict(r)
    >>> e_c = ocl.coarse_solve(r_c)
    >>> ocl.prolongate(x, e_c)
    """

    def __init__(self, wg_size=32, cluster_size=32, nmax_neigh=8, dim=3,
                 device_type=None, platform_idx=None, device_idx=None):
        """
        Parameters
        ----------
        wg_size : int — workgroup size (16, 32, or 64)
        cluster_size : int — max nodes per cluster (<= wg_size)
        nmax_neigh : int — max neighbors per node (8 or 16)
        dim : int — spatial dimension (2 or 3)
        device_type : cl.device_type — preferred device type
        platform_idx, device_idx : int — explicit platform/device selection
        """
        self.wg_size = wg_size
        self.cluster_size = cluster_size
        self.nmax_neigh = nmax_neigh
        self.dim = dim

        assert cluster_size <= wg_size, "cluster_size must be <= wg_size"
        assert wg_size in (16, 32, 64), "wg_size must be 16, 32, or 64"

        # --- Initialize OpenCL context ---
        platforms = cl.get_platforms()
        if platform_idx is not None:
            platform = platforms[platform_idx]
        else:
            # Prefer NVIDIA, then GPU, then first available
            platform = None
            for p in platforms:
                devs = p.get_devices()
                for d in devs:
                    if d.type == cl.device_type.GPU:
                        if 'NVIDIA' in d.name.upper() or 'CUDA' in p.name.upper():
                            platform = p
                            break
                if platform is not None:
                    break
            if platform is None:
                for p in platforms:
                    devs = p.get_devices()
                    for d in devs:
                        if d.type == cl.device_type.GPU:
                            platform = p
                            break
                    if platform is not None:
                        break
            if platform is None:
                platform = platforms[0]

        devices = platform.get_devices()
        if device_idx is not None:
            device = devices[device_idx]
        else:
            device = devices[0]

        self.platform = platform
        self.device = device
        self.ctx = cl.Context([device])
        self.queue = cl.CommandQueue(self.ctx)

        print(f"#OCL platform: {platform.name}")
        print(f"#OCL device:   {device.name}")
        print(f"#OCL wg_size={wg_size} cluster_size={cluster_size} nmax_neigh={nmax_neigh} dim={dim}")

        # Buffers (allocated on upload)
        self.buf_x = None
        self.buf_b = None
        self.buf_r = None
        self.buf_x_out = None
        self.buf_neighs = None
        self.buf_k_vals = None
        self.buf_n_dirs = None
        self.buf_nneigh = None
        self.buf_mass_dt2 = None
        self.buf_Dinv = None
        self.buf_patch_node_ids = None
        self.buf_patch_nneigh = None
        self.buf_patch_core_mask = None
        self.buf_P = None
        self.buf_r_c = None
        self.buf_e_c = None
        self.buf_L_c = None

        self.n_nodes = 0
        self.n_patches = 0
        self.l_coarse = 0

        # Programs (compiled lazily)
        self._prog_jacobi = None
        self._prog_multigrid = None

    # -------------------------------------------------------------------
    # Compilation
    # -------------------------------------------------------------------

    def _compile_jacobi(self, n_steps=5, omega=0.8):
        """Compile block Jacobi kernels with given parameters."""
        defines = {
            'WG_SIZE': self.wg_size,
            'CLUSTER_SIZE': self.cluster_size,
            'NMAX_NEIGH': self.nmax_neigh,
            'DIM': self.dim,
            'N_STEPS': n_steps,
            'OMEGA': f'{omega}f',
        }
        source = _load_kernel_source('kernels_block_jacobi.cl', defines)
        self._prog_jacobi = cl.Program(self.ctx, source).build()
        self._kernels_jacobi = {
            'block_jacobi_step': cl.Kernel(self._prog_jacobi, 'block_jacobi_step'),
            'compute_residual': cl.Kernel(self._prog_jacobi, 'compute_residual'),
            'compute_diagonal_dinv': cl.Kernel(self._prog_jacobi, 'compute_diagonal_dinv'),
        }
        self._jacobi_params = (n_steps, omega)

    def _compile_multigrid(self, l_coarse):
        """Compile multigrid kernels with given coarse space size."""
        defines = {
            'WG_SIZE': self.wg_size,
            'CLUSTER_SIZE': self.cluster_size,
            'DIM': self.dim,
            'L_COARSE': l_coarse,
        }
        source = _load_kernel_source('kernels_multigrid.cl', defines)
        self._prog_multigrid = cl.Program(self.ctx, source).build()
        self._kernels_mg = {
            'prolongate': cl.Kernel(self._prog_multigrid, 'prolongate'),
            'restrict_residual_tree': cl.Kernel(self._prog_multigrid, 'restrict_residual_tree'),
            'zero_buffer': cl.Kernel(self._prog_multigrid, 'zero_buffer'),
            'coarse_solve_cholesky': cl.Kernel(self._prog_multigrid, 'coarse_solve_cholesky'),
        }
        self._l_coarse_compiled = l_coarse

    # -------------------------------------------------------------------
    # Data upload
    # -------------------------------------------------------------------

    def upload_truss(self, neighs, k_vals, n_dirs, nneigh, mass_dt2):
        """Upload CSR-format truss data to GPU.

        Parameters
        ----------
        neighs   : (N, NMAX_NEIGH) int32 — neighbor indices, -1 = void
        k_vals   : (N, NMAX_NEIGH) float32 — stiffness per neighbor slot
        n_dirs   : (N, NMAX_NEIGH, DIM) float32 — edge directions
        nneigh   : (N,) int32 — neighbor count per node
        mass_dt2 : (N,) float32 — mass/dt^2 per node
        """
        self.n_nodes = neighs.shape[0]
        N, NN = neighs.shape
        assert NN == self.nmax_neigh, f"neighs has {NN} cols, expected {self.nmax_neigh}"
        assert n_dirs.shape == (N, NN, self.dim)

        mf = cl.mem_flags
        self.buf_neighs = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                    hostbuf=neighs.astype(np.int32))
        self.buf_k_vals = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                    hostbuf=k_vals.astype(np.float32))
        self.buf_n_dirs = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                    hostbuf=n_dirs.astype(np.float32).ravel())
        self.buf_nneigh = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                    hostbuf=nneigh.astype(np.int32))
        self.buf_mass_dt2 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                      hostbuf=mass_dt2.astype(np.float32))

        # Allocate x, b, r, x_out buffers
        ndof = self.n_nodes * self.dim
        self.buf_x = cl.Buffer(self.ctx, mf.READ_WRITE, ndof * 4)
        self.buf_b = cl.Buffer(self.ctx, mf.READ_ONLY, ndof * 4)
        self.buf_r = cl.Buffer(self.ctx, mf.READ_WRITE, ndof * 4)
        self.buf_x_out = cl.Buffer(self.ctx, mf.READ_WRITE, ndof * 4)
        self.buf_Dinv = cl.Buffer(self.ctx, mf.READ_WRITE,
                                  self.n_nodes * self.dim * self.dim * 4)

        # Compile and run Dinv computation
        if self._prog_jacobi is None:
            self._compile_jacobi()

        self._kernels_jacobi['compute_diagonal_dinv'](
            self.queue, (self.n_nodes,), None,
            self.buf_Dinv, self.buf_neighs, self.buf_k_vals,
            self.buf_n_dirs, self.buf_nneigh, self.buf_mass_dt2,
            np.int32(self.n_nodes)
        )
        self.queue.finish()

    def upload_vectors(self, x, b):
        """Upload solution x and RHS b to GPU.
        x, b : (N, DIM) float32 or float64 (will be cast to float32)
        """
        x32 = x.astype(np.float32).ravel()
        b32 = b.astype(np.float32).ravel()
        cl.enqueue_copy(self.queue, self.buf_x, x32)
        cl.enqueue_copy(self.queue, self.buf_b, b32)
        self.queue.finish()

    def upload_patches(self, patch_node_ids, patch_nneigh, patch_core_mask):
        """Upload patch/cluster mapping to GPU.

        Parameters
        ----------
        patch_node_ids : (m, CLUSTER_SIZE) int32 — global node index per local slot
        patch_nneigh   : (m, CLUSTER_SIZE, NMAX_NEIGH) int32 — local neigh indices
        patch_core_mask: (m, CLUSTER_SIZE) int8 — 1=core, 0=halo
        """
        m, cs = patch_node_ids.shape
        assert cs == self.cluster_size, f"cluster_size mismatch: {cs} vs {self.cluster_size}"
        self.n_patches = m

        mf = cl.mem_flags
        self.buf_patch_node_ids = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                            hostbuf=patch_node_ids.astype(np.int32))
        self.buf_patch_nneigh = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                          hostbuf=patch_nneigh.astype(np.int32))
        self.buf_patch_core_mask = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                             hostbuf=patch_core_mask.astype(np.int8))

    def upload_prolongation(self, P_global, l_coarse):
        """Upload prolongation matrix in tiled layout.

        Parameters
        ----------
        P_global : (m * L_COARSE * CLUSTER_SIZE * DIM) float32 — tiled P matrix
        l_coarse : int — number of coarse modes
        """
        self.l_coarse = l_coarse
        mf = cl.mem_flags
        self.buf_P = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                               hostbuf=P_global.astype(np.float32))

        # Allocate coarse vectors
        self.buf_r_c = cl.Buffer(self.ctx, mf.READ_WRITE, l_coarse * 4)
        self.buf_e_c = cl.Buffer(self.ctx, mf.READ_WRITE, l_coarse * 4)

        # Compile multigrid kernels
        self._compile_multigrid(l_coarse)

    def upload_coarse_operator(self, A_c):
        """Upload coarse operator and precompute Cholesky factorization.

        Parameters
        ----------
        A_c : (l_coarse, l_coarse) float64 — dense coarse operator (SPD)
        """
        from scipy.linalg import cho_factor, cho_solve
        l = A_c.shape[0]
        assert l == self.l_coarse

        # Cholesky factorization on host
        L_c, low = cho_factor(A_c)
        # Extract lower triangular L such that A_c = L * L^T
        L_lower = np.tril(L_c)

        mf = cl.mem_flags
        self.buf_L_c = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=L_lower.astype(np.float32).ravel())

    # -------------------------------------------------------------------
    # Kernel dispatch
    # -------------------------------------------------------------------

    def run_block_jacobi(self, x0, b, n_steps=5, omega=0.8):
        """Run block Jacobi smoothing on GPU.

        Parameters
        ----------
        x0 : (N, DIM) float — initial solution
        b  : (N, DIM) float — RHS
        n_steps : int — inner Jacobi iterations per patch
        omega : float — Jacobi damping

        Returns
        -------
        x_out : (N, DIM) float32 — smoothed solution
        """
        # Recompile if parameters changed
        if self._prog_jacobi is None or self._jacobi_params != (n_steps, omega):
            self._compile_jacobi(n_steps, omega)

        self.upload_vectors(x0, b)

        # Copy x -> x_out so nodes not written by any patch retain their values
        cl.enqueue_copy(self.queue, self.buf_x_out, self.buf_x)

        self._kernels_jacobi['block_jacobi_step'](
            self.queue, (self.n_patches * self.wg_size,), (self.wg_size,),
            self.buf_x, self.buf_b, self.buf_x_out,
            self.buf_patch_node_ids, self.buf_patch_nneigh, self.buf_patch_core_mask,
            self.buf_k_vals, self.buf_n_dirs, self.buf_nneigh, self.buf_mass_dt2,
            self.buf_Dinv, self.buf_neighs,
            np.int32(self.n_patches)
        )
        self.queue.finish()

        x_out = np.empty(self.n_nodes * self.dim, dtype=np.float32)
        cl.enqueue_copy(self.queue, x_out, self.buf_x_out)
        self.queue.finish()
        return x_out.reshape(self.n_nodes, self.dim)

    def compute_residual(self, x=None, b=None):
        """Compute r = b - A*x on GPU.

        If x or b is None, uses the currently uploaded buffer.
        Otherwise uploads the provided arrays first.

        Returns
        -------
        r : (N, DIM) float32 — residual
        """
        if x is not None:
            cl.enqueue_copy(self.queue, self.buf_x, x.astype(np.float32).ravel())
        if b is not None:
            cl.enqueue_copy(self.queue, self.buf_b, b.astype(np.float32).ravel())

        self._kernels_jacobi['compute_residual'](
            self.queue, (self.n_nodes,), None,
            self.buf_x, self.buf_b, self.buf_r,
            self.buf_neighs, self.buf_k_vals, self.buf_n_dirs,
            self.buf_nneigh, self.buf_mass_dt2,
            np.int32(self.n_nodes)
        )
        self.queue.finish()

        r = np.empty(self.n_nodes * self.dim, dtype=np.float32)
        cl.enqueue_copy(self.queue, r, self.buf_r)
        self.queue.finish()
        return r.reshape(self.n_nodes, self.dim)

    def restrict(self, r=None):
        """Compute r_c = P^T * r (restriction / projection to coarse space).

        Parameters
        ----------
        r : (N, DIM) float — fine residual (if None, uses uploaded buf_r)

        Returns
        -------
        r_c : (L_COARSE,) float32 — coarse residual
        """
        if r is not None:
            cl.enqueue_copy(self.queue, self.buf_r, r.astype(np.float32).ravel())

        # Zero the coarse residual buffer (atomic accumulation)
        self._kernels_mg['zero_buffer'](
            self.queue, (self.l_coarse,), None,
            self.buf_r_c, np.int32(self.l_coarse)
        )

        # Dispatch restriction kernel
        self._kernels_mg['restrict_residual_tree'](
            self.queue, (self.n_patches * self.wg_size,), (self.wg_size,),
            self.buf_r_c, self.buf_r, self.buf_P,
            self.buf_patch_node_ids,
            np.int32(self.n_patches)
        )
        self.queue.finish()

        r_c = np.empty(self.l_coarse, dtype=np.float32)
        cl.enqueue_copy(self.queue, r_c, self.buf_r_c)
        self.queue.finish()
        return r_c

    def coarse_solve(self, r_c=None):
        """Solve A_c * e_c = r_c on GPU using precomputed Cholesky.

        Parameters
        ----------
        r_c : (L_COARSE,) float — coarse residual (if None, uses uploaded buf_r_c)

        Returns
        -------
        e_c : (L_COARSE,) float32 — coarse correction
        """
        if r_c is not None:
            cl.enqueue_copy(self.queue, self.buf_r_c, r_c.astype(np.float32))

        wg = min(self.wg_size, max(self.l_coarse, 1))
        self._kernels_mg['coarse_solve_cholesky'](
            self.queue, (wg,), (wg,),
            self.buf_e_c, self.buf_r_c, self.buf_L_c,
            np.int32(self.l_coarse)
        )
        self.queue.finish()

        e_c = np.empty(self.l_coarse, dtype=np.float32)
        cl.enqueue_copy(self.queue, e_c, self.buf_e_c)
        self.queue.finish()
        return e_c

    def prolongate(self, x=None, e_c=None, in_place=True):
        """Apply x += P * e_c (prolongation / coarse correction).

        Parameters
        ----------
        x    : (N, DIM) float — fine solution (if None, uses uploaded buf_x)
        e_c  : (L_COARSE,) float — coarse correction (if None, uses uploaded buf_e_c)
        in_place : bool — if True, updates buf_x on GPU; if False, reads back

        Returns
        -------
        x_out : (N, DIM) float32 — updated solution (only if in_place=False)
        """
        if x is not None:
            cl.enqueue_copy(self.queue, self.buf_x, x.astype(np.float32).ravel())
        if e_c is not None:
            cl.enqueue_copy(self.queue, self.buf_e_c, e_c.astype(np.float32))

        self._kernels_mg['prolongate'](
            self.queue, (self.n_patches * self.wg_size,), (self.wg_size,),
            self.buf_x, self.buf_e_c, self.buf_P,
            self.buf_patch_node_ids,
            np.int32(self.n_patches)
        )
        self.queue.finish()

        if not in_place:
            x_out = np.empty(self.n_nodes * self.dim, dtype=np.float32)
            cl.enqueue_copy(self.queue, x_out, self.buf_x)
            self.queue.finish()
            return x_out.reshape(self.n_nodes, self.dim)
        return None

    def download_x(self):
        """Download current solution from GPU."""
        x = np.empty(self.n_nodes * self.dim, dtype=np.float32)
        cl.enqueue_copy(self.queue, x, self.buf_x)
        self.queue.finish()
        return x.reshape(self.n_nodes, self.dim)

    # -------------------------------------------------------------------
    # Full V-cycle
    # -------------------------------------------------------------------

    def v_cycle(self, x0, b, n_pre_smooth=3, n_post_smooth=3, omega=0.8,
                fixed_nodes=None):
        """Execute one multigrid V-cycle on GPU.

        Steps:
        1. Pre-smooth: n_pre_smooth block Jacobi steps
        2. Restrict: r_c = P^T * (b - A*x)
        3. Coarse solve: e_c = A_c^{-1} * r_c
        4. Prolongate: x += P * e_c
        5. Post-smooth: n_post_smooth block Jacobi steps

        Parameters
        ----------
        x0 : (N, DIM) float — initial solution
        b  : (N, DIM) float — RHS
        n_pre_smooth, n_post_smooth : int — Jacobi steps before/after coarse correction
        omega : float — Jacobi damping
        fixed_nodes : list of int — node indices to pin to x0 values

        Returns
        -------
        x : (N, DIM) float32 — solution after V-cycle
        """
        # Pre-smooth
        x = x0.copy()
        for _ in range(n_pre_smooth):
            x = self.run_block_jacobi(x, b, n_steps=1, omega=omega)
            if fixed_nodes is not None:
                x[fixed_nodes] = x0[fixed_nodes]

        # Compute residual
        self.upload_vectors(x, b)
        self.compute_residual()

        # Restrict
        r_c = self.restrict()

        # Coarse solve
        e_c = self.coarse_solve(r_c)

        # Prolongate (in-place on GPU buffer)
        self.prolongate(e_c=e_c, in_place=True)
        x = self.download_x()
        if fixed_nodes is not None:
            x[fixed_nodes] = x0[fixed_nodes]

        # Post-smooth
        for _ in range(n_post_smooth):
            x = self.run_block_jacobi(x, b, n_steps=1, omega=omega)
            if fixed_nodes is not None:
                x[fixed_nodes] = x0[fixed_nodes]

        return x

    def solve(self, b, x0=None, n_v_cycles=100, n_pre_smooth=3, n_post_smooth=3,
              omega=0.8, tol=1e-6, free_mask=None, fixed_nodes=None):
        """Run multigrid V-cycles until convergence.

        Parameters
        ----------
        b  : (N, DIM) float — RHS
        x0 : (N, DIM) float — initial guess (default: zeros)
        n_v_cycles : int — max V-cycles
        tol : float — relative residual tolerance
        free_mask : (N,) bool — if given, residual norm computed only over free nodes

        Returns
        -------
        x : (N, DIM) float32 — solution
        residuals : list of relative residuals
        """
        if x0 is None:
            x0 = np.zeros((self.n_nodes, self.dim), dtype=np.float32)

        x = x0.astype(np.float32).copy()
        residuals = []

        if free_mask is not None:
            free_flat = np.repeat(free_mask, self.dim)
        else:
            free_flat = np.ones(self.n_nodes * self.dim, dtype=bool)

        # Initial residual
        self.upload_vectors(x, b)
        r = self.compute_residual()
        b32 = b.astype(np.float32).ravel()
        b_norm = np.linalg.norm(b32[free_flat]) + 1e-30
        res = np.linalg.norm(r.ravel()[free_flat]) / b_norm
        residuals.append(res)
        print(f"#OCL V-cycle 0: res={res:.4e}")

        for it in range(n_v_cycles):
            x = self.v_cycle(x, b, n_pre_smooth, n_post_smooth, omega,
                             fixed_nodes=fixed_nodes)

            # Check residual
            self.upload_vectors(x, b)
            r = self.compute_residual()
            res = np.linalg.norm(r.ravel()[free_flat]) / b_norm
            residuals.append(res)

            if (it + 1) % 10 == 0 or res < tol:
                print(f"#OCL V-cycle {it+1}: res={res:.4e}")

            if res < tol:
                break

        return x, residuals


# ---------------------------------------------------------------------------
# Helper: build CSR data from edge list (ei, ej, k_eff, n_dirs)
# ---------------------------------------------------------------------------

def edges_to_csr(ei, ej, k_eff, n_dirs, n_nodes, nmax_neigh=8, dim=3):
    """Convert edge list to CSR-like per-node neighbor format.

    Parameters
    ----------
    ei, ej : (n_edges,) int — edge endpoints
    k_eff  : (n_edges,) float — effective stiffness
    n_dirs : (n_edges, dim) float — unit directions
    n_nodes : int
    nmax_neigh : int — max neighbors per node (padding)
    dim : int

    Returns
    -------
    neighs   : (N, nmax_neigh) int32 — neighbor indices, -1 = void (sorted last)
    k_vals   : (N, nmax_neigh) float32 — stiffness per neighbor slot
    n_dirs_out: (N, nmax_neigh, dim) float32 — direction per neighbor slot
    nneigh   : (N,) int32 — actual neighbor count per node
    """
    adj = [[] for _ in range(n_nodes)]
    for e in range(len(ei)):
        i, j = int(ei[e]), int(ej[e])
        adj[i].append((j, k_eff[e], n_dirs[e]))
        adj[j].append((i, k_eff[e], -n_dirs[e]))  # reverse direction

    neighs = np.full((n_nodes, nmax_neigh), -1, dtype=np.int32)
    k_vals = np.zeros((n_nodes, nmax_neigh), dtype=np.float32)
    n_dirs_out = np.zeros((n_nodes, nmax_neigh, dim), dtype=np.float32)
    nneigh = np.zeros(n_nodes, dtype=np.int32)

    for i in range(n_nodes):
        nbrs = adj[i]
        nneigh[i] = min(len(nbrs), nmax_neigh)
        for e, (j, k, d) in enumerate(nbrs[:nmax_neigh]):
            neighs[i, e] = j
            k_vals[i, e] = k
            n_dirs_out[i, e, :] = d

    return neighs, k_vals, n_dirs_out, nneigh


# ---------------------------------------------------------------------------
# Helper: build patch mapping from patch list
# ---------------------------------------------------------------------------

def patches_to_cluster_map(patches, n_nodes, nmax_neigh=8, cluster_size=32):
    """Convert patch list to GPU-friendly cluster mapping.

    Parameters
    ----------
    patches : list of dicts with 'vertices', 'v2l', 'n_vertices', 'core_mask_loc'
    n_nodes : int
    nmax_neigh : int
    cluster_size : int — padded cluster size (must be >= max patch size)

    Returns
    -------
    patch_node_ids : (m, cluster_size) int32 — global node index per local slot
    patch_nneigh   : (m, cluster_size, nmax_neigh) int32 — local neigh indices
    patch_core_mask: (m, cluster_size) int8 — 1=core, 0=halo/padding
    """
    m = len(patches)
    patch_node_ids = np.full((m, cluster_size), -1, dtype=np.int32)
    patch_nneigh = np.full((m, cluster_size, nmax_neigh), -1, dtype=np.int32)
    patch_core_mask = np.zeros((m, cluster_size), dtype=np.int8)

    for p, pat in enumerate(patches):
        verts = pat['vertices']
        n_loc = len(verts)
        v2l = pat['v2l']

        for a in range(min(n_loc, cluster_size)):
            patch_node_ids[p, a] = verts[a]
            patch_core_mask[p, a] = 1 if pat['core_mask_loc'][a] else 0

            # Map global neighbors to local indices
            node = verts[a]
            for e_idx, (nbr_global, _, _) in enumerate(
                [(v2l.get(v, -1), 0, 0) for v in []]  # placeholder
            ):
                pass  # This needs the actual adjacency

    # Second pass: fill in local neighbor indices using global neighs
    # We need the global neighs array to do this properly
    # This is done in build_cluster_map below

    return patch_node_ids, patch_nneigh, patch_core_mask


def build_cluster_map(patches, neighs, nneigh, n_nodes, nmax_neigh=8, cluster_size=32):
    """Build complete cluster mapping including local neighbor indices.

    Parameters
    ----------
    patches : list of dicts with 'vertices', 'v2l', 'core_mask_loc'
    neighs  : (N, nmax_neigh) int32 — global neighbor indices
    nneigh  : (N,) int32 — neighbor count per node
    n_nodes, nmax_neigh, cluster_size : int

    Returns
    -------
    patch_node_ids : (m, cluster_size) int32
    patch_nneigh   : (m, cluster_size, nmax_neigh) int32 — local neigh indices
    patch_core_mask: (m, cluster_size) int8
    """
    m = len(patches)
    patch_node_ids = np.full((m, cluster_size), -1, dtype=np.int32)
    patch_nneigh = np.full((m, cluster_size, nmax_neigh), -1, dtype=np.int32)
    patch_core_mask = np.zeros((m, cluster_size), dtype=np.int8)

    for p, pat in enumerate(patches):
        verts = pat['vertices']
        v2l = pat['v2l']
        n_loc = len(verts)

        for a in range(min(n_loc, cluster_size)):
            node = int(verts[a])
            patch_node_ids[p, a] = node
            patch_core_mask[p, a] = 1 if pat['core_mask_loc'][a] else 0

            # Map global neighbors to local indices
            nn = nneigh[node]
            for e in range(min(nn, nmax_neigh)):
                nbr_global = neighs[node, e]
                if nbr_global in v2l:
                    patch_nneigh[p, a, e] = v2l[nbr_global]
                else:
                    patch_nneigh[p, a, e] = -1  # ghost neighbor

    return patch_node_ids, patch_nneigh, patch_core_mask


# ---------------------------------------------------------------------------
# Helper: convert prolongation matrix to tiled layout
# ---------------------------------------------------------------------------

def prolongation_to_tiled(P, patch_node_ids, cluster_size, dim, l_coarse):
    """Convert a standard (N*dim, l_coarse) prolongation matrix to tiled layout.

    Tiled layout: P_global[p * L * CS * DIM + j * CS * DIM + a * DIM + d]

    Parameters
    ----------
    P : (N*dim, l_coarse) float — standard prolongation matrix
    patch_node_ids : (m, cluster_size) int32 — global node per local slot
    cluster_size, dim, l_coarse : int

    Returns
    -------
    P_tiled : (m * l_coarse * cluster_size * dim) float32
    """
    m = patch_node_ids.shape[0]
    P_tiled = np.zeros(m * l_coarse * cluster_size * dim, dtype=np.float32)

    for p in range(m):
        for a in range(cluster_size):
            node = patch_node_ids[p, a]
            if node < 0:
                continue
            for j in range(l_coarse):
                for d in range(dim):
                    idx = (p * l_coarse * cluster_size * dim
                           + j * cluster_size * dim
                           + a * dim + d)
                    P_tiled[idx] = P[node * dim + d, j]

    return P_tiled
