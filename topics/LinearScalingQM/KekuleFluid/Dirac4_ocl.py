#!/usr/bin/env python3
"""Dirac4_ocl.py — pyOpenCL harness for the 4-component Dirac-Kekule solver (legacy).

Physical background
-------------------
The low-energy physics of graphene near the Dirac points K, K' is described by
a 4-component Dirac equation with spinor Ψ = (ψ_AK, ψ_BK, ψ_AK', ψ_BK').
The Kekulé mass Δ = |Δ|e^{iφ} opens a gap in the spectrum:

    i ∂Ψ/∂t = [ -i vF (α_x ∂_x + α_y ∂_y) + Δ_R M₁ + Δ_I M₂ + V ] Ψ

where α_x, α_y are Dirac velocity matrices, M₁, M₂ are mass matrices, and V is
a scalar potential.  Vortices in Δ(x,y) bind topologically protected zero modes.

Role in the system
------------------
Legacy alternative to DiracLattice_ocl.py.  This solver works on a 2D Cartesian
grid with periodic boundaries, not on the honeycomb lattice directly.  The
Kekulé mass field Δ(x,y) must be interpolated from the honeycomb graph (from
Model A) or prescribed analytically (e.g. vortex-antivortex pair).  Used by
run_dirac.py and run_combined.py.

Key class
---------
- `Dirac4Solver` — 4-component Dirac solver on Cartesian grid:
    - `set_mass_field(...)` — prescribe Δ_R, Δ_I on the grid.
    - `step()` — RK4 timestep.
    - `run(nsteps, callback)` — run with optional callback.
    - `get_density()`, `get_bilinear()` — diagnostics.

Implements Model B from the KekuleFluid spec:
  i dPsi/dt = [ -i vF (alpha_x d_x + alpha_y d_y)
                 + Delta_R M1 + Delta_I M2 + V*I4 ] Psi

Uses RK4 time integration on a 2D Cartesian grid with periodic boundaries.
The Kekule mass field Delta(x,y) is prescribed (from vortices or interpolated
from a honeycomb graph).

This solver is SEPARATE from the fluid/dimer solver (Model A).
The common visualization layer in plotting.py works with both.
"""
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from pathlib import Path
from typing import List, Optional, Tuple

from hexgrid import Vortex, init_vortex_phase_grid

# Default physical parameters (dimensionless units)
DEFAULTS = dict(
    vF=1.0,
    dx=1.0,
    dy=1.0,
    dt=0.1,        # <= 0.2 * dx / vF for stability
    Delta0=0.3,
    r_core=2.0,
)


class Dirac4Solver:
    """4-component Dirac solver on a 2D periodic grid using pyOpenCL."""

    def __init__(self, Nx: int, Ny: int,
                 device: Optional[cl.Device] = None,
                 vF: float = 1.0, dx: float = 1.0, dy: float = 1.0,
                 dt: float = 0.1, x0: float = 0.0, y0: float = 0.0):
        self.Nx = Nx
        self.Ny = Ny
        self.vF = vF
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.x0 = x0  # grid origin (world coordinates of grid point [0,0])
        self.y0 = y0
        self.n_floats = Nx * Ny * 8  # 4 complex components = 8 floats per grid point

        # --- OpenCL setup ---
        if device is not None:
            self.ctx = cl.Context([device])
        else:
            # Pick first available GPU device
            platforms = cl.get_platforms()
            devices = []
            for p in platforms:
                for d in p.get_devices():
                    if d.type & cl.device_type.GPU:
                        devices.append(d)
            if not devices:
                # fallback to any device
                for p in platforms:
                    devices.extend(p.get_devices())
            self.ctx = cl.Context([devices[0]])

        self.queue = cl.CommandQueue(self.ctx)
        self.device = self.ctx.devices[0]
        print(f"Dirac4Solver: using device {self.device.name}")

        # --- Load and build kernel ---
        cl_path = Path(__file__).parent / "Dirac4.cl"
        with open(cl_path, 'r') as f:
            cl_source = f.read()

        # Inject grid dimensions as compile-time defines
        cl_source = cl_source.replace("#define D4_NX 128", f"#define D4_NX {Nx}")
        cl_source = cl_source.replace("#define D4_NY 128", f"#define D4_NY {Ny}")
        # Remove the ifndef guards since we're injecting directly
        cl_source = cl_source.replace(f"#ifndef D4_NX\n#define D4_NX {Nx}\n#endif\n", f"#define D4_NX {Nx}\n")
        cl_source = cl_source.replace(f"#ifndef D4_NY\n#define D4_NY {Ny}\n#endif\n", f"#define D4_NY {Ny}\n")

        self.prg = cl.Program(self.ctx, cl_source).build()

        # Cache kernel objects for performance
        self._k_rhs = cl.Kernel(self.prg, "dirac_rhs")
        self._k_axpy = cl.Kernel(self.prg, "dirac_axpy")
        self._k_rk4 = cl.Kernel(self.prg, "dirac_rk4_final")

        # --- Allocate buffers ---
        mf = cl.mem_flags
        self.psi_buf   = cl.Buffer(self.ctx, mf.READ_WRITE, self.n_floats * 4)  # float32
        self.k1_buf    = cl.Buffer(self.ctx, mf.READ_WRITE, self.n_floats * 4)
        self.k2_buf    = cl.Buffer(self.ctx, mf.READ_WRITE, self.n_floats * 4)
        self.k3_buf    = cl.Buffer(self.ctx, mf.READ_WRITE, self.n_floats * 4)
        self.k4_buf    = cl.Buffer(self.ctx, mf.READ_WRITE, self.n_floats * 4)
        self.tmp_buf   = cl.Buffer(self.ctx, mf.READ_WRITE, self.n_floats * 4)
        self.delta_buf = cl.Buffer(self.ctx, mf.READ_ONLY,  Nx * Ny * 2 * 4)  # complex per point
        self.pot_buf   = cl.Buffer(self.ctx, mf.READ_ONLY,  Nx * Ny * 4)      # float per point
        self.gamma_buf = cl.Buffer(self.ctx, mf.READ_ONLY,  Nx * Ny * 4)      # float per point

        # Initialize potential and damping to zero
        zero_pot = np.zeros(Nx * Ny, dtype=np.float32)
        zero_gamma = np.zeros(Nx * Ny, dtype=np.float32)
        cl.enqueue_copy(self.queue, self.pot_buf, zero_pot)
        cl.enqueue_copy(self.queue, self.gamma_buf, zero_gamma)

        # Work group sizes
        self.global_rhs = (Nx, Ny)
        self.local_rhs = None  # let OpenCL choose
        self.global_1d = (self.n_floats,)
        # Cap 1D work group to device max
        max_wg = self.device.max_work_group_size
        wg_1d = min(256, max_wg)
        self.local_1d = (wg_1d,)
        # Pad global to be multiple of local
        n_padded = ((self.n_floats + wg_1d - 1) // wg_1d) * wg_1d
        self.global_1d_padded = (n_padded,)

    # ---- Buffer management ----

    def upload_psi(self, psi: np.ndarray):
        """Upload spinor array. Shape (Nx, Ny, 4) complex64."""
        assert psi.shape == (self.Nx, self.Ny, 4), f"Expected ({self.Nx},{self.Ny},4), got {psi.shape}"
        flat = self._pack(psi)
        cl.enqueue_copy(self.queue, self.psi_buf, flat)

    def download_psi(self) -> np.ndarray:
        """Download spinor array. Shape (Nx, Ny, 4) complex64."""
        flat = np.empty(self.n_floats, dtype=np.float32)
        cl.enqueue_copy(self.queue, flat, self.psi_buf)
        return self._unpack(flat)

    def upload_delta(self, delta: np.ndarray):
        """Upload complex Kekule mass field. Shape (Nx, Ny) complex64."""
        assert delta.shape == (self.Nx, self.Ny), f"Expected ({self.Nx},{self.Ny}), got {delta.shape}"
        flat = np.ascontiguousarray(delta.T).astype(np.complex64).view(np.float32)
        # delta is (Nx, Ny) complex. We need (Ny*Nx) interleaved re,im
        # Layout in kernel: gbase = iy*Nx + ix, delta[gbase*2], delta[gbase*2+1]
        flat = np.ascontiguousarray(delta).astype(np.complex64).view(np.float32)
        cl.enqueue_copy(self.queue, self.delta_buf, flat)

    def upload_potential(self, V: np.ndarray):
        """Upload scalar potential. Shape (Nx, Ny) float32."""
        assert V.shape == (self.Nx, self.Ny)
        cl.enqueue_copy(self.queue, self.pot_buf, np.ascontiguousarray(V, dtype=np.float32))

    def upload_gamma(self, gamma: np.ndarray):
        """Upload absorbing damping field. Shape (Nx, Ny) float32."""
        assert gamma.shape == (self.Nx, self.Ny)
        cl.enqueue_copy(self.queue, self.gamma_buf, np.ascontiguousarray(gamma, dtype=np.float32))

    # ---- Packing / unpacking ----

    def _pack(self, psi: np.ndarray) -> np.ndarray:
        """Convert (Nx, Ny, 4) complex64 to flat float32 array.
        Layout: ((iy*Nx + ix)*4 + comp)*2 + ri
        """
        psi_c = np.ascontiguousarray(psi, dtype=np.complex64)
        # psi_c shape: (Nx, Ny, 4)
        # We want index: ((iy*Nx + ix)*4 + comp)*2 + ri
        # Transpose to (Ny, Nx, 4) then reshape
        psi_t = np.ascontiguousarray(psi_c.transpose(1, 0, 2))  # (Ny, Nx, 4)
        flat = psi_t.view(np.float32).ravel()
        return np.ascontiguousarray(flat)

    def _unpack(self, flat: np.ndarray) -> np.ndarray:
        """Convert flat float32 array back to (Nx, Ny, 4) complex64."""
        # flat shape: (Ny*Nx*8,)
        complex_flat = flat[0::2] + 1j * flat[1::2]  # interleave re, im -> complex
        # complex_flat shape: (Ny*Nx*4,)
        reshaped = complex_flat.reshape(self.Ny, self.Nx, 4)
        # Transpose back to (Nx, Ny, 4)
        return np.ascontiguousarray(reshaped.transpose(1, 0, 2))

    # ---- Time stepping ----

    def step(self):
        """One RK4 time step."""
        dt = np.float32(self.dt)
        vF = np.float32(self.vF)
        dx = np.float32(self.dx)
        dy = np.float32(self.dy)

        k_rhs = self._k_rhs
        k_axpy = self._k_axpy
        k_rk4 = self._k_rk4

        # k1 = rhs(psi)
        k_rhs(self.queue, self.global_rhs, self.local_rhs,
              self.psi_buf, self.k1_buf,
              self.delta_buf, self.pot_buf, self.gamma_buf,
              vF, dx, dy)

        # tmp = psi + 0.5*dt*k1
        k_axpy(self.queue, self.global_1d_padded, self.local_1d,
               self.tmp_buf, self.psi_buf, self.k1_buf,
               np.float32(0.5 * self.dt))

        # k2 = rhs(tmp)
        k_rhs(self.queue, self.global_rhs, self.local_rhs,
              self.tmp_buf, self.k2_buf,
              self.delta_buf, self.pot_buf, self.gamma_buf,
              vF, dx, dy)

        # tmp = psi + 0.5*dt*k2
        k_axpy(self.queue, self.global_1d_padded, self.local_1d,
               self.tmp_buf, self.psi_buf, self.k2_buf,
               np.float32(0.5 * self.dt))

        # k3 = rhs(tmp)
        k_rhs(self.queue, self.global_rhs, self.local_rhs,
              self.tmp_buf, self.k3_buf,
              self.delta_buf, self.pot_buf, self.gamma_buf,
              vF, dx, dy)

        # tmp = psi + dt*k3
        k_axpy(self.queue, self.global_1d_padded, self.local_1d,
               self.tmp_buf, self.psi_buf, self.k3_buf,
               np.float32(self.dt))

        # k4 = rhs(tmp)
        k_rhs(self.queue, self.global_rhs, self.local_rhs,
              self.tmp_buf, self.k4_buf,
              self.delta_buf, self.pot_buf, self.gamma_buf,
              vF, dx, dy)

        # psi += dt*(k1 + 2*k2 + 2*k3 + k4)/6
        k_rk4(self.queue, self.global_1d_padded, self.local_1d,
              self.psi_buf, self.k1_buf, self.k2_buf,
              self.k3_buf, self.k4_buf, dt)

    def run(self, n_steps: int, callback=None, callback_interval: int = 1):
        """Run n_steps RK4 steps. Optional callback called every callback_interval steps."""
        for step in range(n_steps):
            self.step()
            if callback is not None and (step % callback_interval == 0):
                callback(self, step)

    # ---- Normalization ----

    def normalize(self):
        """Normalize psi so that sum |psi|^2 * dx*dy = 1."""
        psi = self.download_psi()
        norm_sq = np.sum(np.abs(psi)**2) * self.dx * self.dy
        if norm_sq > 0:
            psi /= np.sqrt(norm_sq)
            self.upload_psi(psi)

    # ---- Initialization ----

    def init_gaussian_packet(self, r0: Tuple[float, float],
                             sigma: float = 5.0,
                             k0: Tuple[float, float] = (0.0, 0.0),
                             chi: Tuple[complex, complex, complex, complex] = (1, 0, 0, 0)):
        """Initialize a Gaussian wavepacket at position r0 with momentum k0."""
        x = self.x0 + np.arange(self.Nx, dtype=np.float32) * self.dx
        y = self.y0 + np.arange(self.Ny, dtype=np.float32) * self.dy
        X, Y = np.meshgrid(x, y, indexing='ij')

        dx_r = X - r0[0]
        dy_r = Y - r0[1]
        envelope = np.exp(-(dx_r**2 + dy_r**2) / (2.0 * sigma**2))
        phase = np.exp(1j * (k0[0] * X + k0[1] * Y))

        psi = np.zeros((self.Nx, self.Ny, 4), dtype=np.complex64)
        for c in range(4):
            psi[:, :, c] = envelope * phase * chi[c]

        self.upload_psi(psi)
        self.normalize()

    def init_delta_vortices(self, vortices: List[Vortex],
                            Delta0: float = 0.3, r_core: float = 2.0):
        """Initialize the Kekule mass field from vortex positions."""
        x = self.x0 + np.arange(self.Nx, dtype=np.float32) * self.dx
        y = self.y0 + np.arange(self.Ny, dtype=np.float32) * self.dy
        X, Y = np.meshgrid(x, y, indexing='ij')

        delta = init_vortex_phase_grid(X, Y, vortices, r_core) * Delta0
        self.upload_delta(delta)

    def init_delta_from_graph(self, graph, atom_z: np.ndarray,
                              Delta0: float = 0.3, sigma: float = 0.75):
        """Interpolate complex z from honeycomb atoms to the Cartesian grid.

        Uses Gaussian interpolation as specified in the KekuleFluid spec.
        """
        x = self.x0 + np.arange(self.Nx, dtype=np.float32) * self.dx
        y = self.y0 + np.arange(self.Ny, dtype=np.float32) * self.dy
        X, Y = np.meshgrid(x, y, indexing='ij')

        delta = np.zeros((self.Nx, self.Ny), dtype=np.complex64)
        weight = np.zeros((self.Nx, self.Ny), dtype=np.float32)

        for i in range(graph.n_atoms):
            ax, ay = graph.pos[i]
            d2 = (X - ax)**2 + (Y - ay)**2
            w = np.exp(-d2 / (2.0 * sigma**2))
            delta += w * atom_z[i]
            weight += w

        weight = np.maximum(weight, 1e-8)
        delta = Delta0 * delta / weight
        self.upload_delta(delta)

    def init_delta_from_modelA(self, graph, z_real=None, z_imag=None,
                               Delta0: float = 0.3, sigma: float = 0.75):
        """Interpolate complex z from a Model A HoneycombGraph (Graph.py) to the grid.

        Accepts either a KekuleFluidSolver (calls get_z()) or raw z_real/z_imag arrays.
        Uses Gaussian interpolation per the KekuleFluid spec.
        """
        # Extract z from solver if not given directly
        if z_real is None or z_imag is None:
            z_real, z_imag = graph.get_z()
            graph = graph.graph
        z = z_real + 1j * z_imag

        # Extract atom positions from Model A graph
        pos = np.array([[a.pos[0], a.pos[1]] for a in graph.atoms], dtype=np.float32)

        x = self.x0 + np.arange(self.Nx, dtype=np.float32) * self.dx
        y = self.y0 + np.arange(self.Ny, dtype=np.float32) * self.dy
        X, Y = np.meshgrid(x, y, indexing='ij')

        delta = np.zeros((self.Nx, self.Ny), dtype=np.complex64)
        weight = np.zeros((self.Nx, self.Ny), dtype=np.float32)

        for i in range(len(pos)):
            d2 = (X - pos[i, 0])**2 + (Y - pos[i, 1])**2
            w = np.exp(-d2 / (2.0 * sigma**2))
            delta += w * z[i]
            weight += w

        weight = np.maximum(weight, 1e-8)
        delta = Delta0 * delta / weight
        self.upload_delta(delta)

    # ---- Diagnostics ----

    def compute_diagnostics(self) -> dict:
        """Compute density, Kekule bilinear response, etc."""
        psi = self.download_psi()

        # Density: rho = Psi^dagger Psi = sum |psi_comp|^2
        rho = np.sum(np.abs(psi)**2, axis=2)  # (Nx, Ny)

        # Kekule bilinear: S1 = Re(Psi^dagger M1 Psi), S2 = Re(Psi^dagger M2 Psi)
        # M1 Psi = (p3, p2, p1, p0)  [real permutation]
        # Psi^dagger M1 Psi = conj(p0)*p3 + conj(p1)*p2 + conj(p2)*p1 + conj(p3)*p0
        #                   = 2*Re(conj(p0)*p3 + conj(p1)*p2)
        S1 = 2.0 * (np.real(np.conj(psi[:,:,0]) * psi[:,:,3]) +
                    np.real(np.conj(psi[:,:,1]) * psi[:,:,2]))

        # M2 Psi = (-i*p3, -i*p2, i*p1, i*p0)
        # Psi^dagger M2 Psi = conj(p0)*(-i*p3) + conj(p1)*(-i*p2) + conj(p2)*(i*p1) + conj(p3)*(i*p0)
        #                   = -i*conj(p0)*p3 - i*conj(p1)*p2 + i*conj(p2)*p1 + i*conj(p3)*p0
        #                   = 2*Im(conj(p2)*p1 + conj(p3)*p0)  ... let me compute
        # = -i*(conj(p0)*p3 + conj(p1)*p2) + i*(conj(p2)*p1 + conj(p3)*p0)
        # = -i*A + i*conj(A)  where A = conj(p0)*p3 + conj(p1)*p2
        # = -i*A + i*conj(A) = 2*Im(conj(A)) = -2*Im(A)
        # Actually: -i*A + i*conj(A) = i*(conj(A) - A) = i*(-2i*Im(A)) = 2*Im(A)
        S2 = 2.0 * (np.imag(np.conj(psi[:,:,0]) * psi[:,:,3]) +
                    np.imag(np.conj(psi[:,:,1]) * psi[:,:,2]))

        SK = S1 + 1j * S2

        return {
            'rho': rho,
            'S1': S1,
            'S2': S2,
            'SK': SK,
            'arg_SK': np.angle(SK),
            'abs_SK': np.abs(SK),
        }

    def get_delta(self) -> np.ndarray:
        """Download the current Delta field as complex array (Nx, Ny)."""
        flat = np.empty(self.Nx * self.Ny * 2, dtype=np.float32)
        cl.enqueue_copy(self.queue, flat, self.delta_buf)
        complex_flat = flat[0::2] + 1j * flat[1::2]
        return complex_flat.reshape(self.Nx, self.Ny)
