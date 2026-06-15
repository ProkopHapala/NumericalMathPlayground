"""
Core iterative solver methods for spectral filtering and eigenvalue computation.

Supports dense toy matrices and sparse MMFF vibration benchmarks (K, M)
via mass-weighted reduction H = M^{-1/2} K M^{-1/2} with Chebyshev scaling to [-1, 1].
Optional OpenCL CSR SpMM for K @ V with multiple probe vectors (SpMM).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp

try:
    import pyopencl as cl
    _HAS_OPENCL = True
except ImportError:
    _HAS_OPENCL = False

DEFAULT_FIXTURES_DIR = os.environ.get(
    "VIBRATION_BENCHMARKS_DIR",
    "/home/prokop/git/FireCore/tests/tSiNCs/fixtures/vibration_benchmarks",
)

BENCHMARK_SYSTEMS = [
    "adamantane",
    "nc_C_R4",
    "nc_C_R5",
    "nc_C_R6",
    "nc_C_R7",
    "nc_C_R8",
]

_OPENCL_CSR_SPMM = r"""
// CSR SpMM: Y = A @ X,  X,Y shape (nrow, k) row-major as X[row*k + v]
__kernel void csr_spmm(
    const int nrow,
    const int k,
    __global const int* indptr,
    __global const int* indices,
    __global const double* data,
    __global const double* X,
    __global double* Y)
{
    const int row = get_global_id(0);
    const int v = get_global_id(1);
    if (row >= nrow || v >= k) return;

    double sum = 0.0;
    const int start = indptr[row];
    const int end = indptr[row + 1];
    for (int j = start; j < end; ++j) {
        sum += data[j] * X[indices[j] * k + v];
    }
    Y[row * k + v] = sum;
}
"""


def op_matmul(H, V):
    """Matrix multiply H @ V for dense ndarray or VibrationOperator."""
    if isinstance(V, np.ndarray) and V.ndim == 1:
        out = op_matmul(H, V[:, None])
        return out[:, 0]
    if hasattr(H, "matvec"):
        return H.matvec(V)
    return H @ V


def op_ndim(H):
    if hasattr(H, "ndim"):
        return H.ndim
    return H.shape[0]


def get_exact_eigenvalues(H, exact_w=None):
    """Scaled eigenvalues in [-1, 1] for plotting reference lines."""
    if exact_w is not None:
        return np.asarray(exact_w, dtype=float)
    if hasattr(H, "exact_scaled_eigs"):
        return H.exact_scaled_eigs()
    return np.sort(np.linalg.eigvalsh(H))


def load_vibration_benchmark(path, rigid_omega_threshold=100.0):
    """
    Load MMFF vibration benchmark .npz (projected CSR K, diagonal M).

    Rigid modes are already penalized in K (rigid_shift); reference omegas
    above rigid_omega_threshold are excluded from the vibrational subspace.
    """
    path = Path(path)
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta_json"]))
    K = sp.csr_matrix(
        (d["K_csr_data"], d["K_csr_indices"], d["K_csr_indptr"]),
        shape=tuple(d["K_csr_shape"]),
    )
    mass = np.asarray(d["mass_diag"], dtype=float)
    omegas_all = np.asarray(d["omegas_modes_projected"], dtype=float)
    vib_mask = omegas_all < rigid_omega_threshold
    omegas_vib = omegas_all[vib_mask]
    pos = np.asarray(d["pos"], dtype=float) if "pos" in d.files else None
    K_dense = np.asarray(d["H_dense_projected"], dtype=float) if "H_dense_projected" in d.files else None
    name = str(d["name"]) if "name" in d.files else path.stem
    return {
        "name": name,
        "path": str(path),
        "K": K,
        "K_dense": K_dense,
        "mass": mass,
        "pos": pos,
        "omegas_all": omegas_all,
        "omegas_vib": omegas_vib,
        "ndof": int(d["ndof"]),
        "meta": meta,
        "rigid_shift": float(d["rigid_shift"]) if "rigid_shift" in d.files else meta.get("rigid_shift", 1e6),
    }


def resolve_benchmark_path(system, fixtures_dir=None):
    """Resolve system name or path to .npz file."""
    p = Path(system)
    if p.suffix == ".npz" and p.exists():
        return p
    root = Path(fixtures_dir or DEFAULT_FIXTURES_DIR)
    candidate = root / f"{system}.npz"
    if not candidate.exists():
        raise FileNotFoundError(f"Benchmark not found: {candidate}")
    return candidate


class OpenCLCSRSpMM:
    """OpenCL CSR SpMM: Y = K @ X with X shape (nrow, k)."""

    def __init__(self, K_csr: sp.csr_matrix):
        if not _HAS_OPENCL:
            raise RuntimeError("pyopencl not installed")
        if not sp.isspmatrix_csr(K_csr):
            K_csr = K_csr.tocsr()
        self.nrow = K_csr.shape[0]
        self.indptr = K_csr.indptr.astype(np.int32)
        self.indices = K_csr.indices.astype(np.int32)
        self.data = K_csr.data.astype(np.float64)
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, _OPENCL_CSR_SPMM).build()
        self._bufs = None
        self._bufs_k = -1

    def _ensure_bufs(self, k):
        if self._bufs is not None and self._bufs_k == k:
            return
        mf = cl.mem_flags
        self._bufs = {
            "indptr": cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.indptr),
            "indices": cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.indices),
            "data": cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.data),
        }
        self._bufs_k = k

    def matmul(self, X: np.ndarray) -> np.ndarray:
        """X: (nrow, k) -> Y: (nrow, k)."""
        if X.ndim == 1:
            return self.matmul(X[:, None])[:, 0]
        nrow, k = X.shape
        assert nrow == self.nrow
        X = np.ascontiguousarray(X, dtype=np.float64)
        Y = np.empty_like(X)
        self._ensure_bufs(k)
        mf = cl.mem_flags
        x_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
        y_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, Y.nbytes)
        self.prg.csr_spmm(
            self.queue, (nrow, k), None,
            np.int32(nrow), np.int32(k),
            self._bufs["indptr"], self._bufs["indices"], self._bufs["data"],
            x_buf, y_buf,
        )
        cl.enqueue_copy(self.queue, Y, y_buf)
        return Y


class VibrationOperator:
    """
    Mass-weighted stiffness in Chebyshev scaling for K v = lambda M v.

    H_raw = M^{-1/2} K M^{-1/2},  eigenvalues lambda = omega^2.
    H_scaled = 2 (H_raw - c) / r  maps vibrational lambda to [-1, 1].
    """

    def __init__(
        self,
        K_csr,
        mass_diag,
        omegas_vib=None,
        margin=1.02,
        backend="cpu",
        pos=None,
        deflate_rigid=True,
        K_dense=None,
        use_spectral_basis=True,
    ):
        self.K = K_csr.tocsr() if sp.issparse(K_csr) else sp.csr_matrix(K_csr)
        self.mass = np.asarray(mass_diag, dtype=float)
        self.m_inv_sqrt = 1.0 / np.sqrt(self.mass)
        self.ndim = self.K.shape[0]
        self.omegas_vib = None if omegas_vib is None else np.asarray(omegas_vib, dtype=float)
        self.backend = backend
        self.spmv_count = 0
        self._ocl = None
        self.deflate_rigid = deflate_rigid
        self._R_rigid = None
        self._Q_vib = None
        self._lam_vib = None
        if omegas_vib is not None and len(omegas_vib) > 0:
            lam = omegas_vib ** 2
            lam_min, lam_max = float(lam.min()), float(lam.max())
        else:
            lam_min, lam_max = self._estimate_lambda_bounds()
        span = max(lam_max - lam_min, 1e-12)
        self.lam_center = 0.5 * (lam_min + lam_max)
        self.lam_radius = 0.5 * span * margin
        self.omega_min = float(np.sqrt(max(lam_min, 0.0)))
        self.omega_max = float(np.sqrt(lam_max))

        # Vibrational spectral basis: exact H|_vib from dense eigh (avoids rigid-shift pollution).
        if use_spectral_basis and self.ndim <= 2500:
            self._build_vibrational_basis(rigid_omega_threshold=100.0, K_dense=K_dense)

        if deflate_rigid and pos is not None and self._Q_vib is None:
            self._R_rigid = build_rigid_modes(pos, self.mass)

        if backend == "opencl":
            if not _HAS_OPENCL:
                print("Warning: pyopencl unavailable, falling back to CPU SpMM")
                self.backend = "cpu"
            elif self._Q_vib is None:
                self._ocl = OpenCLCSRSpMM(self.K)
            else:
                print("Note: OpenCL K SpMM skipped when using vibrational spectral basis")

    def _build_vibrational_basis(self, rigid_omega_threshold=100.0, K_dense=None):
        """Dense M^{-1/2} K M^{-1/2} eigh; keep modes with omega < threshold."""
        D = self.m_inv_sqrt
        Kd = K_dense if K_dense is not None else self.K.toarray()
        Hmw = (D[:, None] * Kd) * D[None, :]
        lam, Q = np.linalg.eigh(Hmw)
        vib = (lam > 0) & (lam < rigid_omega_threshold ** 2)
        self._lam_vib = lam[vib]
        self._Q_vib = Q[:, vib]

    def _matvec_spectral(self, V):
        """H_raw @ V via vibrational eigenbasis (small dense matmul)."""
        if V.ndim == 1:
            c = self._Q_vib.T @ V
            return self._Q_vib @ (self._lam_vib * c)
        c = self._Q_vib.T @ V
        return self._Q_vib @ (self._lam_vib[:, None] * c)

    def _estimate_lambda_bounds(self, n_iter=30):
        """Lanczos on H_raw for lambda_min, lambda_max (vibrational scale)."""
        v = np.random.randn(self.ndim)
        v /= np.linalg.norm(v)
        alpha = np.zeros(n_iter)
        beta = np.zeros(n_iter)
        q_prev = np.zeros(self.ndim)
        q = v.copy()
        for j in range(n_iter):
            z = self._matvec_raw(q)
            alpha[j] = np.dot(q, z)
            z -= alpha[j] * q
            if j > 0:
                z -= beta[j - 1] * q_prev
            beta[j] = np.linalg.norm(z)
            if beta[j] < 1e-14:
                break
            q_prev, q = q, z / beta[j]
        m = j + 1
        T = np.diag(alpha[:m]) + np.diag(beta[:m - 1], 1) + np.diag(beta[:m - 1], -1)
        w = np.linalg.eigvalsh(T)
        return float(w[0]), float(w[-1])

    def _K_mul(self, V):
        self.spmv_count += V.shape[1] if V.ndim == 2 else 1
        if self._ocl is not None:
            return self._ocl.matmul(V)
        return self.K @ V

    def _matvec_raw(self, V):
        """H_raw @ V = M^{-1/2} K M^{-1/2} V."""
        if self._Q_vib is not None:
            return self._matvec_spectral(V)
        if self._R_rigid is not None:
            V = _project_out_rigid(V, self._R_rigid, self.mass)
        Vw = self.m_inv_sqrt[:, None] * V if V.ndim == 2 else self.m_inv_sqrt * V
        KV = self._K_mul(Vw)
        if V.ndim == 2:
            out = self.m_inv_sqrt[:, None] * KV
        else:
            out = self.m_inv_sqrt * KV
        if self._R_rigid is not None:
            out = _project_out_rigid(out, self._R_rigid, self.mass)
        return out

    def matvec(self, V):
        """H_scaled @ V."""
        Hv = self._matvec_raw(V)
        if V.ndim == 2:
            return (Hv - self.lam_center * V) / self.lam_radius
        return (Hv - self.lam_center * V) / self.lam_radius

    def __matmul__(self, V):
        return self.matvec(V)

    def lambda_to_scaled(self, lam):
        return (np.asarray(lam, dtype=float) - self.lam_center) / self.lam_radius

    def scaled_to_lambda(self, x):
        return self.lam_center + self.lam_radius * np.asarray(x, dtype=float)

    def omega_to_scaled(self, omega):
        return self.lambda_to_scaled(np.asarray(omega, dtype=float) ** 2)

    def scaled_to_omega(self, x):
        lam = self.scaled_to_lambda(x)
        lam = np.clip(lam, 0.0, self.omega_max ** 2 * 1.001)
        return np.sqrt(lam)

    def exact_scaled_eigs(self):
        if self.omegas_vib is None or len(self.omegas_vib) == 0:
            return np.array([])
        return self.omega_to_scaled(self.omegas_vib)

    def exact_omegas(self):
        return self.omegas_vib if self.omegas_vib is not None else np.array([])

    def ritz_omegas(self, w_scaled, U):
        """Physical omega from Ritz pairs (Rayleigh quotient on H_raw)."""
        w_scaled = np.asarray(w_scaled, dtype=float)
        U = np.asarray(U)
        if U.ndim == 1:
            U = U[:, None]
        out = np.empty(len(w_scaled))
        for j in range(len(w_scaled)):
            u = U[:, j]
            un = np.linalg.norm(u)
            if un < 1e-30:
                out[j] = 0.0
                continue
            Hv = self._matvec_raw(u[:, None])[:, 0]
            lam = float(np.dot(u, Hv) / np.dot(u, u))
            out[j] = float(np.sqrt(np.clip(lam, 0.0, self.omega_max ** 2 * 1.001)))
        return out


def build_rigid_modes(pos, mass_diag):
    """
    Six M-orthonormal rigid-body modes (3 translations + 3 rotations) from geometry.
    pos: (natoms, 3) Angstrom; mass_diag: (3*natoms,) amu per xyz DOF.
    """
    pos = np.asarray(pos, dtype=float)
    natoms = pos.shape[0]
    ndof = 3 * natoms
    m_atom = mass_diag.reshape(natoms, 3)[:, 0]
    total_m = m_atom.sum()
    com = (m_atom[:, None] * pos).sum(axis=0) / total_m

    R = np.zeros((ndof, 6))
    # Translations
    for k in range(3):
        for a in range(natoms):
            R[3 * a + k, k] = 1.0
    # Rotations about COM
    for a in range(natoms):
        r = pos[a] - com
        # infinitesimal rotation generators: r x e_k
        R[3 * a + 0, 3] = r[1]
        R[3 * a + 1, 3] = -r[0]
        R[3 * a + 0, 4] = -r[2]
        R[3 * a + 2, 4] = r[0]
        R[3 * a + 1, 5] = r[2]
        R[3 * a + 2, 5] = -r[1]

    # M-orthonormalize
    M = mass_diag
    for i in range(6):
        R[:, i] /= np.sqrt(np.dot(R[:, i], M * R[:, i]) + 1e-30)
        for j in range(i):
            R[:, i] -= np.dot(R[:, i], M * R[:, j]) * R[:, j]
            R[:, i] /= np.sqrt(np.dot(R[:, i], M * R[:, i]) + 1e-30)
    return R


def _project_out_rigid(V, R, mass_diag):
    """Remove rigid components: V <- (I - R R^T M) V."""
    if V.ndim == 1:
        V = V[:, None]
        squeeze = True
    else:
        squeeze = False
    M = mass_diag[:, None]
    coeffs = R.T @ (M * V)
    V = V - R @ coeffs
    return V[:, 0] if squeeze else V


def build_vibration_operator(system, fixtures_dir=None, backend="cpu", rigid_omega_threshold=100.0,
                             use_spectral_basis=True):
    """Load benchmark and return (operator, benchmark dict)."""
    path = resolve_benchmark_path(system, fixtures_dir)
    bench = load_vibration_benchmark(path, rigid_omega_threshold=rigid_omega_threshold)
    op = VibrationOperator(
        bench["K"], bench["mass"], omegas_vib=bench["omegas_vib"],
        backend=backend, pos=bench.get("pos"), deflate_rigid=True,
        K_dense=bench.get("K_dense"), use_spectral_basis=use_spectral_basis,
    )
    return op, bench


def omega_band_edges(omegas_vib, band_width):
    """Uniform omega windows for subspace iteration."""
    lo = float(omegas_vib.min())
    hi = float(omegas_vib.max())
    edges = np.arange(lo, hi + band_width, band_width)
    if edges[-1] < hi - 1e-12:
        edges = np.append(edges, hi)
    return edges[:-1].astype(float), edges[1:].astype(float)


def scaled_bands_from_omega(op: VibrationOperator, omega_lo, omega_hi):
    """Convert omega band edges to scaled Chebyshev coordinates."""
    return op.omega_to_scaled(omega_lo), op.omega_to_scaled(omega_hi)


def generate_test_matrix(n: int):
    """1D Laplacian, scaled to [-0.95, 0.95]"""
    diag = 2.0 * np.ones(n)
    off_diag = -1.0 * np.ones(n - 1)
    H = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    H = H / 2.0 - np.eye(n)
    H *= 0.95
    return H


def get_exact_eigenvalues_dense(H: np.ndarray):
    return np.sort(np.linalg.eigvalsh(H))


def jackson_kernel(m: int, N: int):
    """Jackson damping coefficient to eliminate Gibbs oscillations."""
    if N == 0: return 1.0
    M = N + 1
    term1 = (M - m) * np.cos(np.pi * m / M)
    term2 = np.sin(np.pi * m / M) / np.tan(np.pi / M)
    return (term1 + term2) / M


def cheb_rect_coeffs(f_lo, f_hi, deg, use_jackson=True):
    """
    Chebyshev polynomial coefficients for band-pass filter [f_lo, f_hi].
    Approximates a rectangle function using the Fourier-Chebyshev expansion.
    """
    f_lo = float(np.clip(f_lo, -1.0, 1.0))
    f_hi = float(np.clip(f_hi, -1.0, 1.0))
    if f_lo > f_hi: f_lo, f_hi = f_hi, f_lo
    th_lo = np.arccos(np.clip(f_lo, -1+1e-12, 1-1e-12))
    th_hi = np.arccos(np.clip(f_hi, -1+1e-12, 1-1e-12))
    c = np.zeros(deg + 1, dtype=float)
    c[0] = (th_lo - th_hi) / np.pi
    m = np.arange(1, deg + 1)
    c[1:] = (2.0 / np.pi) * (np.sin(m * th_hi) - np.sin(m * th_lo)) / m
    if use_jackson:
        for i in range(1, deg + 1):
            c[i] *= jackson_kernel(i, deg)
    # Rescale so p(x_mid)≈1 inside the band
    x_mid = 0.5 * (f_lo + f_hi)
    p_mid = float(eval_cheb_series(np.array([x_mid]), c)[0])
    if abs(p_mid) > 1e-12: c /= p_mid
    return c


def eval_cheb_series(x, c):
    """Evaluate Chebyshev series with coefficients c at points x using Clenshaw's algorithm."""
    x = np.asarray(x, dtype=float)
    deg = len(c) - 1
    if deg < 0: return np.zeros_like(x)
    if deg == 0: return c[0] * np.ones_like(x)
    b_kp1 = np.zeros_like(x)
    b_kp2 = np.zeros_like(x)
    for k in range(deg, 0, -1):
        b_k = 2.0 * x * b_kp1 - b_kp2 + c[k]
        b_kp2, b_kp1 = b_kp1, b_k
    return x * b_kp1 - b_kp2 + c[0]


def apply_cheb_poly(H, V, c):
    """
    Apply Chebyshev polynomial p(H) to vectors V using three-term recurrence.
    Returns p(H) @ V.
    """
    deg = len(c) - 1
    if deg < 0: return np.zeros_like(V)
    T0 = V
    Y = c[0] * T0
    if deg == 0: return Y
    T1 = op_matmul(H, V)
    Y = Y + c[1] * T1
    for m in range(2, deg + 1):
        T2 = 2.0 * (op_matmul(H, T1)) - T0
        Y = Y + c[m] * T2
        T0, T1 = T1, T2
    return Y


def rayleigh_ritz(H, V):
    """
    Rayleigh-Ritz extraction from subspace V.
    Returns (eigenvalues w, eigenvectors U, residuals r).
    """
    Q, _ = np.linalg.qr(V)
    T = Q.T @ (op_matmul(H, Q))
    w, Z = np.linalg.eigh(T)
    U = Q @ Z
    R = op_matmul(H, U) - U * w[None, :]
    r = np.linalg.norm(R, axis=0)
    return w, U, r


def lanczos_band(H, V0, steps):
    """
    Single-vector Lanczos starting from the first column of V0.
    Builds a Krylov subspace of size 'steps' and returns Ritz values/vectors.
    No matrix inversion needed. Cheap: one SpMV per step.
    """
    N = op_ndim(H)
    Q, _ = np.linalg.qr(V0)
    q = Q[:, 0].copy()
    q = q / np.linalg.norm(q)

    Q_all = np.zeros((N, steps))
    Q_all[:, 0] = q
    alphas = np.zeros(steps)
    betas = np.zeros(steps)

    for j in range(steps):
        z = op_matmul(H, q)
        alpha = np.dot(q, z)
        alphas[j] = alpha
        z = z - alpha * q
        if j > 0:
            z = z - betas[j-1] * Q_all[:, j-1]
        beta = np.linalg.norm(z)
        if beta < 1e-14:
            break
        betas[j] = beta
        if j + 1 < steps:
            q = z / beta
            Q_all[:, j+1] = q

    m = j + 1
    T = np.zeros((m, m))
    for i in range(m):
        T[i, i] = alphas[i]
        if i > 0:
            T[i, i-1] = betas[i-1]
            T[i-1, i] = betas[i-1]

    w, Z = np.linalg.eigh(T)
    U = Q_all[:, :m] @ Z
    R = op_matmul(H, U) - U * w[None, :]
    r = np.linalg.norm(R, axis=0)
    return w, U, r


def cluster_prune_mask(V_filt, H, cluster_tol=0.85, eig_tol=0.05):
    """
    Identify redundant vectors chasing the same eigenvalue based on:
    1. Collinearity of raw filtered vectors (dot product)
    2. Eigenvalue proximity (cheap Rayleigh quotient on raw vectors)
    Returns a boolean mask where True = keep.
    """
    m = V_filt.shape[1]
    if m < 2:
        return np.ones(m, dtype=bool)

    # Normalize columns for collinearity check
    Vn = V_filt / np.linalg.norm(V_filt, axis=0, keepdims=True)

    # Cheap Rayleigh quotients on raw filtered vectors
    w_s = np.array([np.dot(Vn[:, i], op_matmul(H, Vn[:, i])) for i in range(m)])

    # Sort by eigenvalue for neighbor check
    order = np.argsort(w_s)
    w_s = w_s[order]

    # Residuals estimated from Rayleigh quotient: r ≈ ||(H - w_i) v_i||
    r_s = np.array([np.linalg.norm(op_matmul(H, Vn[:, order[i]]) - w_s[i] * Vn[:, order[i]]) for i in range(m)])

    keep_sorted = np.ones(m, dtype=bool)

    # Check only adjacent pairs — conservative & cheap (O(m) not O(m^2))
    for i in range(m - 1):
        if not keep_sorted[i]:
            continue
        j = i + 1
        # Eigenvalue proximity check
        if abs(w_s[j] - w_s[i]) > eig_tol:
            continue
        # Collinearity check on the filtered vectors
        overlap = abs(np.dot(Vn[:, order[i]], Vn[:, order[j]]))
        if overlap > cluster_tol:
            # Same eigenvalue being chased — prune the worse one
            if r_s[i] <= r_s[j]:
                keep_sorted[j] = False
            else:
                keep_sorted[i] = False

    # Unsort back to original column ordering
    keep = np.zeros(m, dtype=bool)
    keep[order] = keep_sorted
    return keep


def cheap_rank_estimate(V, rel_thresh=0.1):
    """
    Estimate rank of V via SVD singular values.
    Returns number of singular values > rel_thresh * sigma_max.
    """
    if V.shape[1] == 0:
        return 0
    s = np.linalg.svd(V, compute_uv=False)
    if len(s) == 0:
        return 0
    sigma_max = s[0]
    if sigma_max < 1e-14:
        return 0
    return int(np.sum(s > rel_thresh * sigma_max))


def match_trajectories(w_in, prev_w, prev_ids, final_traj, bi):
    """
    Greedy trajectory matching by nearest eigenvalue.
    Assigns trajectory IDs to current Ritz values based on previous iteration.
    
    Returns curr_ids (array of trajectory IDs for each w_in).
    """
    curr_ids = np.full(len(w_in), -1, dtype=int)
    used = set()
    if prev_w is not None and len(prev_w) > 0:
        for j, wc in enumerate(w_in):
            best = None; best_d = 1e9
            for k, wp in enumerate(prev_w):
                if k in used: continue
                d = abs(wc - wp)
                if d < best_d:
                    best_d = d; best = k
            if best is not None and best_d < 0.15:
                curr_ids[j] = prev_ids[best]
                used.add(best)
    next_id = int(max(final_traj[bi].keys(), default=-1)) + 1
    for j in range(len(w_in)):
        if curr_ids[j] < 0:
            curr_ids[j] = next_id
            next_id += 1
    return curr_ids


def apply_pruning(r_in, curr_ids, final_traj, bi, it, prune_mode, prune_tol, 
                  cluster_tol, cluster_eig_tol, gradual_factor, V_filt, H, mask, order):
    """
    Apply pruning strategy to determine which vectors to keep.
    
    Returns keep (boolean mask for r_in).
    """
    keep = np.ones(len(r_in), dtype=bool)
    
    if prune_mode == 'cluster' and it >= 2:
        cluster_keep = cluster_prune_mask(
            V_filt, H,
            cluster_tol=cluster_tol,
            eig_tol=cluster_eig_tol
        )
        keep = keep & cluster_keep[mask][order]
        
    elif prune_mode == 'residual' and it >= 2 and prune_tol > 0:
        keep = keep & (r_in < prune_tol)
        
    elif prune_mode == 'gradual' and it >= 2 and prune_tol > 0:
        # Drop only the single worst vector if it is:
        #   1. Above prune_tol
        #   2. Significantly worse than median (outlier)
        #   3. NOT improving (residual stagnating or growing)
        if len(r_in) > 1:
            worst = np.argmax(r_in)
            median_r = float(np.median(r_in))
            is_outlier = r_in[worst] > gradual_factor * max(median_r, prune_tol * 0.1)
            # Improvement check: compare to previous residual of same trajectory
            tid = curr_ids[worst]
            prev_r = None
            if tid in final_traj[bi] and len(final_traj[bi][tid]) > 0:
                prev_r = final_traj[bi][tid][-1][2]
            is_improving = False
            if prev_r is not None and prev_r > 0:
                is_improving = r_in[worst] < 0.9 * prev_r  # at least 10% drop
            if r_in[worst] > prune_tol and is_outlier and not is_improving:
                keep[worst] = False
                
    elif prune_mode == 'hybrid' and it >= 2:
        # Step 1: cluster prune (remove redundant pairs)
        if cluster_tol > 0:
            cluster_keep = cluster_prune_mask(
                V_filt, H,
                cluster_tol=cluster_tol,
                eig_tol=cluster_eig_tol
            )
            keep = keep & cluster_keep[mask][order]
        # Step 2: if still too many, drop the worst above tol
        if prune_tol > 0 and keep.sum() > 1:
            r_masked = r_in.copy()
            r_masked[~keep] = -1.0
            worst = np.argmax(r_masked)
            if r_masked[worst] > prune_tol:
                keep[worst] = False
    
    return keep


def solve_spectrum(H, band_lo, band_hi, nvec, coarse_iters, conv_iters,
                   filter_reps, square_filter, method, prune_mode,
                   prune_tol, cluster_tol, cluster_eig_tol, gradual_factor,
                   rank_est=False, res_tol=1e-6, exact_w=None):
    """
    Band-by-band spectral solver: Chebyshev filter + Ritz/Lanczos iteration
    with trajectory tracking and pruning.

    Parameters
    ----------
    H : (N, N) array - Hamiltonian matrix
    band_lo, band_hi : (nbands,) arrays - band edges
    nvec : int - initial probe vectors per band
    coarse_iters : int - Chebyshev degree
    conv_iters : int - number of filter->QR->Ritz refinement iterations
    filter_reps : int - filter applications per subspace iteration
    square_filter : bool - use p(H)^2
    method : 'ritz' or 'lanczos'
    prune_mode : 'none', 'residual', 'cluster', 'gradual', 'hybrid'
    prune_tol, cluster_tol, cluster_eig_tol, gradual_factor : pruning parameters
    rank_est : bool - print cheap rank estimate
    res_tol : float - residual tolerance for convergence report

    Returns
    -------
    dict with keys:
        'exact_w': exact eigenvalues (sorted)
        'band_raw_w': {bi: [w_in_it0, w_in_it1, ...]}
        'band_raw_r': {bi: [r_in_it0, r_in_it1, ...]}
        'band_keep': {bi: [keep_it0, keep_it1, ...]}
        'final_traj': {bi: {tid: [(it, w, r, kept), ...]}}
        'total_spmv': int
        'spmv_per_band': {bi: int}
    """
    exact_w = get_exact_eigenvalues(H, exact_w=exact_w)
    nbands = len(band_lo)

    band_raw_w = {bi: [] for bi in range(nbands)}
    band_raw_r = {bi: [] for bi in range(nbands)}
    band_keep  = {bi: [] for bi in range(nbands)}
    final_traj = {bi: {} for bi in range(nbands)}

    total_spmv = 0
    spmv_per_band = {}

    for bi in range(nbands):
        f_lo, f_hi = float(band_lo[bi]), float(band_hi[bi])
        c = cheb_rect_coeffs(f_lo, f_hi, coarse_iters, use_jackson=True)

        n_eig_est = max(2, int(nvec * 2))
        V = np.random.randn(op_ndim(H), n_eig_est)

        prev_ids = None
        prev_w   = None
        for it in range(max(1, int(conv_iters))):
            n_reps = filter_reps if it > 0 else max(1, filter_reps)
            V_filt = V.copy()
            k_curr = V.shape[1]
            spmv_per_app = coarse_iters * (2 if square_filter else 1)
            for _ in range(n_reps):
                V_filt = apply_cheb_poly(H, V_filt, c)
                if square_filter:
                    V_filt = apply_cheb_poly(H, V_filt, c)
            spmv_this = n_reps * spmv_per_app * k_curr
            total_spmv += spmv_this
            spmv_per_band[bi] = spmv_per_band.get(bi, 0) + spmv_this

            if it == 0 and rank_est:
                rank_est_val = cheap_rank_estimate(V_filt, rel_thresh=0.1)
                true_count = int(np.sum((exact_w >= f_lo) & (exact_w <= f_hi)))
                print(f"  band={bi}  rank_est={rank_est_val}  true_eigs={true_count}")

            V_qr, _ = np.linalg.qr(V_filt)

            if method == 'ritz':
                w, U, r = rayleigh_ritz(H, V_qr)
                spmv_r = k_curr
                total_spmv += spmv_r
                spmv_per_band[bi] = spmv_per_band.get(bi, 0) + spmv_r
            else:  # lanczos
                steps = max(2, V.shape[1])
                w, U, r = lanczos_band(H, V_qr, steps=steps)
                spmv_r = steps + steps
                total_spmv += spmv_r
                spmv_per_band[bi] = spmv_per_band.get(bi, 0) + spmv_r

            mask = (w >= f_lo) & (w <= f_hi)
            if not np.any(mask):
                # Filter spillover: keep Ritz values nearest the band
                mid = 0.5 * (f_lo + f_hi)
                order_all = np.argsort(np.abs(w - mid))
                keep_n = min(len(w), max(2, nvec))
                mask = np.zeros_like(w, dtype=bool)
                mask[order_all[:keep_n]] = True
            w_in = w[mask]
            r_in = r[mask]
            U_in = U[:, mask]
            order = np.argsort(w_in)
            w_in = w_in[order]
            r_in = r_in[order]
            U_in = U_in[:, order]
            if hasattr(H, "ritz_omegas"):
                w_plot = H.ritz_omegas(w_in, U_in)
            else:
                w_plot = w_in.copy()

            curr_ids = match_trajectories(w_in, prev_w, prev_ids, final_traj, bi)

            keep = apply_pruning(r_in, curr_ids, final_traj, bi, it, prune_mode,
                                 prune_tol, cluster_tol, cluster_eig_tol,
                                 gradual_factor, V_filt, H, mask, order)

            for j in range(len(w_in)):
                tid = curr_ids[j]
                if tid not in final_traj[bi]:
                    final_traj[bi][tid] = []
                final_traj[bi][tid].append((it, float(w_plot[j]), float(r_in[j]), bool(keep[j])))

            band_raw_w[bi].append(w_plot.copy())
            band_raw_r[bi].append(r_in.copy())
            band_keep[bi].append(keep.copy())

            if keep.sum() > 0:
                V = U_in[:, keep]
                prev_ids = curr_ids[keep]
                prev_w   = w_plot[keep]
            else:
                if len(r_in) == 0:
                    V = U[:, :1]
                    prev_ids = np.array([0])
                    prev_w = w[:1]
                else:
                    best = np.argmin(r_in)
                    V = U_in[:, best:best + 1]
                    prev_ids = curr_ids[best:best + 1]
                    prev_w = w_plot[best:best + 1]

    return {
        'exact_w': exact_w,
        'band_raw_w': band_raw_w,
        'band_raw_r': band_raw_r,
        'band_keep': band_keep,
        'final_traj': final_traj,
        'total_spmv': total_spmv,
        'spmv_per_band': spmv_per_band,
    }


def solve_band(H, V0, f_lo, f_hi, coarse_iters, solve_iters, square_filter):
    """
    Extract eigenpairs in a single band by repeated Chebyshev filter + Rayleigh-Ritz.

    Parameters
    ----------
    H : (N, N) array - Hamiltonian matrix
    V0 : (N, k) array - initial probe vectors
    f_lo, f_hi : float - band edges
    coarse_iters : int - Chebyshev degree
    solve_iters : int - number of filter->QR->Ritz rounds
    square_filter : bool - use p(H)^2

    Returns
    -------
    dict with keys:
        'w_in': eigenvalues inside band
        'r_in': residuals inside band
        'solve_spmv': SpMV count
        'k': number of probe vectors
    """
    c = cheb_rect_coeffs(f_lo, f_hi, coarse_iters, use_jackson=True)
    V = V0.copy()
    solve_spmv = 0
    k = V.shape[1]
    spmv_per_app = coarse_iters * (2 if square_filter else 1)
    for _ in range(max(1, int(solve_iters))):
        V = apply_cheb_poly(H, V, c)
        solve_spmv += spmv_per_app * k
        if square_filter:
            V = apply_cheb_poly(H, V, c)
        V, _ = np.linalg.qr(V)
    w, U, r = rayleigh_ritz(H, V)
    solve_spmv += k  # H @ Q in rayleigh_ritz
    mask = (w >= f_lo) & (w <= f_hi)
    w_in = w[mask]
    r_in = r[mask]
    return {
        'w_in': w_in,
        'r_in': r_in,
        'solve_spmv': solve_spmv,
        'k': k,
    }


def chebyshev_filter(H, V0, freqs, iters, use_jackson=True):
    """
    KPM Chebyshev filter: builds basis once, evaluates at multiple frequencies.
    Uses Jackson damping to eliminate Gibbs oscillations.
    
    Parameters
    ----------
    H : (N, N) array - Hamiltonian matrix
    V0 : (N, k) array - initial probe vectors
    freqs : (nfreq,) array - frequency points in [-1, 1]
    iters : list of int - iteration counts to evaluate
    use_jackson : bool - apply Jackson damping
    
    Returns
    -------
    total_amps : {n: (nfreq,) array} - total amplitude per frequency
    vec_amps : {n: (nfreq, k) array} - per-vector amplitudes
    """
    max_iter = max(iters)
    nfreq = len(freqs)
    N, k = V0.shape

    # 1. Frequency-independent step: Build the Chebyshev basis
    basis = np.zeros((max_iter + 1, N, k))
    basis[0] = V0
    if max_iter > 0:
        basis[1] = op_matmul(H, V0)
    for m in range(2, max_iter + 1):
        basis[m] = 2.0 * (op_matmul(H, basis[m - 1])) - basis[m - 2]

    # 2. Frequency-dependent step: Cheap scalar accumulation
    theta = np.arccos(freqs)
    total_amps = {}
    vec_amps = {}

    for n in iters:
        v_amps_n = np.zeros((nfreq, k))
        for i, f in enumerate(freqs):
            th = theta[i]
            V_f = np.copy(basis[0])
            for m in range(1, n + 1):
                weight = 2.0 * np.cos(m * th)
                if use_jackson:
                    weight *= jackson_kernel(m, n)
                V_f += weight * basis[m]
            v_amps_n[i, :] = np.linalg.norm(V_f, axis=0)

        vec_amps[n] = v_amps_n
        total_amps[n] = np.sum(v_amps_n, axis=1)

    return total_amps, vec_amps


def power_iteration_filter(H, V0, freqs, iters):
    """
    Power iteration filter: evaluate (fI-H)^n V0 norm at each frequency.
    
    Parameters
    ----------
    H : (N, N) array - Hamiltonian matrix
    V0 : (N, k) array - initial probe vectors
    freqs : (nfreq,) array - frequency points
    iters : list of int - iteration counts to evaluate
    
    Returns
    -------
    total_amps : {n: (nfreq,) array}
    vec_amps : {n: (nfreq, k) array}
    """
    max_iter = max(iters)
    nfreq = len(freqs)
    N, k = V0.shape

    total_amps = {}
    vec_amps = {n: np.zeros((nfreq, k)) for n in iters}

    for i, f in enumerate(freqs):
        V = np.copy(V0)
        for n in range(1, max_iter + 1):
            V = f * V - op_matmul(H, V)
            if n in iters:
                norms = np.linalg.norm(V, axis=0)
                vec_amps[n][i, :] = norms

    for n in iters:
        total_amps[n] = np.sum(vec_amps[n], axis=1)

    return total_amps, vec_amps


def chebyshev_filter_omega(H, V0, omegas, iters, use_jackson=True):
    """KPM Chebyshev filter with frequency axis in omega (internal MMFF units)."""
    if hasattr(H, "omega_to_scaled"):
        freqs = np.clip(H.omega_to_scaled(omegas), -1.0 + 1e-12, 1.0 - 1e-12)
    else:
        freqs = np.asarray(omegas, dtype=float)
    return chebyshev_filter(H, V0, freqs, iters, use_jackson=use_jackson)


def scaled_trajectories_to_omega(op, band_raw_w):
    """Convert solve_spectrum Ritz values (scaled) to omega."""
    if not hasattr(op, "scaled_to_omega"):
        return band_raw_w
    return {bi: [op.scaled_to_omega(w) for w in ws_list] for bi, ws_list in band_raw_w.items()}
