"""
Iterative solvers for vibration resolvent: A(omega) x = b
where A(omega) = K - (omega + i*eta)^2 M   [physical coordinates]

Designed for interior spectrum scanning with batched [N, k] RHS.
Backend: CPU (scipy sparse) first, OpenCL GPU second.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from spectral_solvers import load_vibration_benchmark, resolve_benchmark_path

try:
    import pyopencl as cl
    _HAS_OPENCL = True
except ImportError:
    _HAS_OPENCL = False


def _auto_cl_context():
    """Auto-select NVIDIA GPU if available, else first GPU, else CPU. No prompts."""
    if not _HAS_OPENCL:
        raise RuntimeError("pyopencl not installed")
    platforms = cl.get_platforms()
    for platform in platforms:
        for device in platform.get_devices():
            if device.type & cl.device_type.GPU and "NVIDIA" in device.vendor.upper():
                return cl.Context([device])
    for platform in platforms:
        for device in platform.get_devices():
            if device.type & cl.device_type.GPU:
                return cl.Context([device])
    for platform in platforms:
        for device in platform.get_devices():
            if device.type & cl.device_type.CPU:
                return cl.Context([device])
    raise RuntimeError("No OpenCL device found")


# ---------------------------------------------------------------------------
# OpenCL kernels for batched SpMM [N, k]  (single precision)
# ---------------------------------------------------------------------------

_OPENCL_CSR_SPMM_FLOAT = r"""
__kernel void csr_spmm_float(
    const int nrow, const int k,
    __global const int* indptr,
    __global const int* indices,
    __global const float* data,
    __global const float* X,
    __global float* Y)
{
    const int row = get_global_id(0);
    const int v = get_global_id(1);
    if (row >= nrow || v >= k) return;
    float sum = 0.0f;
    const int start = indptr[row];
    const int end = indptr[row + 1];
    for (int j = start; j < end; ++j) {
        sum += data[j] * X[indices[j] * k + v];
    }
    Y[row * k + v] = sum;
}
"""

_OPENCL_CSR_SPMM_COMPLEX = r"""
__kernel void csr_spmm_complex(
    const int nrow, const int k,
    __global const int* indptr,
    __global const int* indices,
    __global const float* data,
    __global const float* X,
    __global float* Y)
{
    const int row = get_global_id(0);
    const int v = get_global_id(1);
    if (row >= nrow || v >= k) return;
    float sum_re = 0.0f, sum_im = 0.0f;
    const int start = indptr[row];
    const int end = indptr[row + 1];
    for (int j = start; j < end; ++j) {
        int col = indices[j];
        float Are = data[2*j], Aim = data[2*j+1];
        float Xre = X[2*(col*k+v)],   Xim = X[2*(col*k+v)+1];
        sum_re += Are*Xre - Aim*Xim;
        sum_im += Are*Xim + Aim*Xre;
    }
    Y[2*(row*k+v)]   = sum_re;
    Y[2*(row*k+v)+1] = sum_im;
}
"""


# ---------------------------------------------------------------------------
# Batched SpMM backends
# ---------------------------------------------------------------------------

class CPUBatchedSpMM:
    """CPU: scipy sparse matrix multiply with (N, k) dense matrix."""
    def __init__(self, K_csr: sp.csr_matrix):
        self.K = K_csr.tocsr()
        self.nrow = K_csr.shape[0]
        self.spmv_count = 0
    def matmul(self, X):
        if X.ndim == 1: return self.K @ X
        self.spmv_count += X.shape[1]
        return self.K @ X


class OpenCLBatchedSpMM:
    """OpenCL GPU: CSR SpMM with (N, k) dense matrix."""
    def __init__(self, K_csr: sp.csr_matrix):
        if not _HAS_OPENCL: raise RuntimeError("pyopencl not installed")
        self.K = K_csr.tocsr()
        self.nrow = K_csr.shape[0]
        self.is_complex = np.iscomplexobj(K_csr.data)
        self.indptr = K_csr.indptr.astype(np.int32)
        self.indices = K_csr.indices.astype(np.int32)
        self.spmv_count = 0
        self.ctx = _auto_cl_context()
        self.queue = cl.CommandQueue(self.ctx)
        if self.is_complex:
            d = np.empty(2*len(K_csr.data), np.float32)
            d[0::2] = K_csr.data.real.astype(np.float32)
            d[1::2] = K_csr.data.imag.astype(np.float32)
            self.data = d
            src = _OPENCL_CSR_SPMM_COMPLEX
            self._kname = "csr_spmm_complex"
        else:
            self.data = K_csr.data.astype(np.float32)
            src = _OPENCL_CSR_SPMM_FLOAT
            self._kname = "csr_spmm_float"
        self.prg = cl.Program(self.ctx, src).build()
        self._bufs = None; self._bufs_k = -1

    def _ensure_bufs(self, k):
        if self._bufs is not None and self._bufs_k == k: return
        mf = cl.mem_flags
        self._bufs = {
            "indptr":  cl.Buffer(self.ctx, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=self.indptr),
            "indices": cl.Buffer(self.ctx, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=self.indices),
            "data":    cl.Buffer(self.ctx, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=self.data),
        }
        self._bufs_k = k

    def matmul(self, X):
        if X.ndim == 1:
            X = X[:, None]
            return self.matmul(X)[:, 0]
        nrow, k = X.shape
        assert nrow == self.nrow
        self.spmv_count += k
        self._ensure_bufs(k)
        mf = cl.mem_flags
        if self.is_complex:
            Xf = np.ascontiguousarray(X, np.complex64)
            xi = np.empty(2*nrow*k, np.float32)
            xi[0::2] = Xf.real.ravel(); xi[1::2] = Xf.imag.ravel()
            yi = np.empty_like(xi)
            xb = cl.Buffer(self.ctx, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=xi)
            yb = cl.Buffer(self.ctx, mf.WRITE_ONLY, yi.nbytes)
            self.prg.__getattr__(self._kname)(self.queue, (nrow, k), None,
                np.int32(nrow), np.int32(k), self._bufs["indptr"], self._bufs["indices"], self._bufs["data"], xb, yb)
            cl.enqueue_copy(self.queue, yi, yb)
            Y = np.empty((nrow, k), np.complex64)
            Y.real = yi[0::2].reshape(nrow, k); Y.imag = yi[1::2].reshape(nrow, k)
            return Y
        else:
            Xf = np.ascontiguousarray(X, np.float32)
            Y = np.empty_like(Xf)
            xb = cl.Buffer(self.ctx, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=Xf)
            yb = cl.Buffer(self.ctx, mf.WRITE_ONLY, Y.nbytes)
            self.prg.__getattr__(self._kname)(self.queue, (nrow, k), None,
                np.int32(nrow), np.int32(k), self._bufs["indptr"], self._bufs["indices"], self._bufs["data"], xb, yb)
            cl.enqueue_copy(self.queue, Y, yb)
            return Y.astype(X.dtype)


# ---------------------------------------------------------------------------
# Resolvent Operator  A(omega) = K - (omega+i*eta)^2 M
# ---------------------------------------------------------------------------

class ResolventOperator:
    """
    Dynamic stiffness operator A(omega, eta) = K - (omega + i*eta)^2 * M.
    Applies A @ X for dense X (N, k) batched.
    """
    def __init__(self, K_csr, mass_diag, backend="cpu"):
        self.K = K_csr.tocsr() if sp.issparse(K_csr) else sp.csr_matrix(K_csr)
        self.mass = np.asarray(mass_diag, dtype=float)
        self.ndim = self.K.shape[0]
        self.natoms = self.ndim // 3
        self.backend = backend
        self.spmv_count = 0
        self.atom_dof = np.arange(self.ndim).reshape(self.natoms, 3)
        self.K_diag_blocks = self._extract_diag_blocks(self.K)
        if backend == "opencl":
            try:
                self._spmm = OpenCLBatchedSpMM(self.K)
            except Exception as e:
                print(f"OpenCL failed ({e}), CPU fallback")
                self._spmm = CPUBatchedSpMM(self.K)
                self.backend = "cpu"
        else:
            self._spmm = CPUBatchedSpMM(self.K)

    def _extract_diag_blocks(self, K):
        blocks = []
        for i in range(self.natoms):
            dofs = self.atom_dof[i]
            blocks.append(K[dofs[:,None], dofs[None,:]].toarray().astype(float))
        return np.array(blocks)  # (natoms, 3, 3)

    def get_diag_blocks(self, omega, eta=0.0):
        """A_ii = K_ii - (omega+i*eta)^2 * m_i * I_3, shape (natoms, 3, 3)."""
        if eta == 0.0:
            fac = omega ** 2
        else:
            fac = (omega + 1j * eta) ** 2
        m = self.mass.reshape(self.natoms, 3)[:, 0]
        I3 = np.eye(3)
        return self.K_diag_blocks - fac * m[:, None, None] * I3[None, :, :]

    def matvec(self, omega, X, eta=0.0):
        """A(omega, eta) @ X."""
        fac = (omega + 1j*eta)**2
        Y = self._spmm.matmul(X)
        if X.ndim == 1: Y -= fac * self.mass * X
        else:           Y -= fac * self.mass[:, None] * X
        self.spmv_count += 1
        return Y

    def matvec_real(self, omega, X):
        """Real A(omega) @ X = K@X - omega^2 * M @ X."""
        fac = omega**2
        Y = self._spmm.matmul(X)
        if X.ndim == 1: Y -= fac * self.mass * X
        else:           Y -= fac * self.mass[:, None] * X
        self.spmv_count += 1
        return Y

    def matvec_batch(self, omegas, X, eta=0.0):
        """Batched: A(omega_k) @ X[:,k] for k=0..m-1."""
        m = len(omegas); assert X.shape == (self.ndim, m)
        Y = self._spmm.matmul(X)
        for k in range(m):
            fac = (omegas[k] + 1j*eta)**2
            Y[:, k] -= fac * self.mass * X[:, k]
        self.spmv_count += 1
        return Y


# ---------------------------------------------------------------------------
# Block-Jacobi Preconditioner (3x3 per atom)
# ---------------------------------------------------------------------------

class BlockJacobiPreconditioner:
    """P = blockdiag(A_ii).  Eigendecompose 3x3, regularize, invert."""
    def __init__(self, resolvent, eps_rel=1e-8):
        self.resolvent = resolvent
        self.natoms = resolvent.natoms
        self.eps_rel = eps_rel

    def build(self, omega, eta=0.0):
        A_ii = self.resolvent.get_diag_blocks(omega, eta)
        self.preconds = []
        for i in range(self.natoms):
            Ai = A_ii[i]
            s = float(np.max(np.abs(Ai)))
            eps = self.eps_rel * max(s, 1.0)
            Ai = Ai + eps * np.eye(3, dtype=Ai.dtype)
            self.preconds.append(np.linalg.inv(Ai))

    def apply(self, R):
        """P^{-1} @ R.  R: (N,) or (N, k)."""
        if R.ndim == 1:
            Z = np.empty_like(R)
            for i in range(self.natoms):
                dofs = self.resolvent.atom_dof[i]
                P = self.preconds[i]
                Z[dofs] = P @ R[dofs]
            return Z
        Z = np.empty_like(R)
        for i in range(self.natoms):
            dofs = self.resolvent.atom_dof[i]
            P = self.preconds[i]
            Z[dofs, :] = P @ R[dofs, :]
        return Z


# ---------------------------------------------------------------------------
# Iterative Solvers
# ---------------------------------------------------------------------------

def solve_block_jacobi(A_fn, b, x0, P_inv_fn, max_iter=100, alpha=1.0, beta=0.2,
                       check_conv=False, tol=1e-4):
    """Block-Jacobi + heavy ball. Fixed iterations, optional conv check."""
    x_old = x0.copy()
    x_older = np.zeros_like(x0)
    b_norm = np.linalg.norm(b) + 1e-30
    info = {"iter": max_iter, "residual": None, "converged": False}
    for it in range(max_iter):
        r = b - A_fn(x_old)
        if check_conv and it % 10 == 0:
            res = np.linalg.norm(r) / b_norm
            if res < tol:
                info.update(iter=it, residual=res, converged=True)
                return x_old, info
        dx = P_inv_fn(r)
        x_new = x_old + alpha * dx + beta * (x_old - x_older)
        x_older, x_old = x_old, x_new
    if check_conv:
        info["residual"] = np.linalg.norm(b - A_fn(x_old)) / b_norm
    return x_old, info


def solve_minres(A_fn, b, x0, P_inv_fn=None, tol=1e-5, max_iter=200):
    """Preconditioned MINRES for real symmetric indefinite A."""
    n = len(b)
    b_norm = np.linalg.norm(b) + 1e-30
    x = x0.copy()
    r = b - A_fn(x)
    z = P_inv_fn(r) if P_inv_fn else r.copy()
    v_old = np.zeros(n)
    v = z.copy()
    beta = np.linalg.norm(v)
    if beta < 1e-14:
        return x, {"iter": 0, "residual": 0.0, "converged": True}
    v /= beta
    c_old, s_old = 1.0, 0.0
    c, s = 1.0, 0.0
    phi = beta
    w_old = np.zeros(n)
    w = np.zeros(n)
    residual = beta / b_norm
    info = {"iter": max_iter, "residual": residual, "converged": False}
    for it in range(1, max_iter + 1):
        Av = A_fn(v)
        alpha = np.dot(v, Av)
        z_new = P_inv_fn(Av - alpha * v - beta * v_old) if P_inv_fn else (Av - alpha * v - beta * v_old)
        beta_new = np.linalg.norm(z_new)
        if beta_new < 1e-14:
            info.update(iter=it, residual=residual, converged=residual < tol)
            return x, info
        delta = c * alpha - c_old * s * beta
        gamma = s * alpha + c_old * c * beta
        rho = np.sqrt(delta**2 + beta_new**2)
        c_new = delta / rho; s_new = beta_new / rho
        phi_new = -s_new * phi; phi = c_new * phi
        p = (z - delta * w - gamma * w_old) / rho
        x += phi * p
        v_old, v = v, z_new / beta_new
        w_old, w = w, p
        c_old, s_old = c, s; c, s = c_new, s_new
        beta = beta_new
        residual = abs(phi) / b_norm
        if residual < tol:
            info.update(iter=it, residual=residual, converged=True)
            return x, info
    info["residual"] = residual
    return x, info


def solve_cocr(A_fn, b, x0, P_inv_fn=None, tol=1e-5, max_iter=200, pivot_guard=1e-12):
    """Preconditioned COCR for complex-symmetric A = A^T.  1 SpMV/iter."""
    n = len(b)
    b_norm = np.linalg.norm(b) + 1e-30
    x = x0.copy()
    r = b - A_fn(x)
    z = P_inv_fn(r) if P_inv_fn else r.copy()
    p = z.copy()
    rz = np.dot(r, z)
    residual = np.linalg.norm(r) / b_norm
    info = {"iter": max_iter, "residual": residual, "converged": False}
    for it in range(max_iter):
        Ap = A_fn(p)
        pap = np.dot(p, Ap)
        # pivot guard
        if abs(pap) < pivot_guard * np.linalg.norm(p) * np.linalg.norm(Ap):
            r = b - A_fn(x)
            z = P_inv_fn(r) if P_inv_fn else r.copy()
            p = z.copy(); rz = np.dot(r, z)
            Ap = A_fn(p); pap = np.dot(p, Ap)
        alpha = rz / pap
        x += alpha * p
        r -= alpha * Ap
        res = np.linalg.norm(r) / b_norm
        if res < tol:
            info.update(iter=it, residual=res, converged=True)
            return x, info
        z_old = z.copy()
        z = P_inv_fn(r) if P_inv_fn else r.copy()
        rz_new = np.dot(r, z)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new
    info["residual"] = residual
    return x, info


def solve_bicgstab(A_fn, b, x0, P_inv_fn=None, tol=1e-5, max_iter=200):
    """Preconditioned BiCGSTAB for general non-symmetric/complex A.  2 SpMV/iter."""
    n = len(b)
    b_norm = np.linalg.norm(b) + 1e-30
    x = x0.copy()
    r = b - A_fn(x)
    r0 = r.copy()
    rho = np.dot(r0.conj(), r)
    p = r.copy()
    residual = np.linalg.norm(r) / b_norm
    info = {"iter": max_iter, "residual": residual, "converged": False}
    for it in range(max_iter):
        Ap = A_fn(p)
        alpha = rho / (np.dot(r0.conj(), Ap) + 1e-30)
        s = r - alpha * Ap
        if np.linalg.norm(s) / b_norm < tol:
            x += alpha * p
            info.update(iter=it, residual=np.linalg.norm(s)/b_norm, converged=True)
            return x, info
        As = A_fn(s)
        omega = np.dot(As.conj(), s) / (np.dot(As.conj(), As) + 1e-30)
        x += alpha * p + omega * s
        r = s - omega * As
        res = np.linalg.norm(r) / b_norm
        if res < tol:
            info.update(iter=it, residual=res, converged=True)
            return x, info
        rho_new = np.dot(r0.conj(), r)
        beta = (rho_new / rho) * (alpha / (omega + 1e-30))
        p = r + beta * (p - omega * Ap)
        rho = rho_new
    info["residual"] = residual
    return x, info


def solve_gmres(A_fn, b, x0, P_inv_fn=None, tol=1e-5, max_iter=200, restart=30):
    """Restarted GMRES with optional preconditioning."""
    n = len(b)
    b_norm = np.linalg.norm(b) + 1e-30
    x = x0.copy()
    r0 = b - A_fn(x)
    info = {"iter": 0, "residual": None, "true_residual": np.linalg.norm(r0) / b_norm, "converged": False}
    restart_n = int(restart)
    n_outer = max_iter // restart_n + 1

    for outer in range(n_outer):
        r = b - A_fn(x)
        beta_true = np.linalg.norm(r)
        if beta_true / b_norm < tol:
            info.update(iter=outer * restart_n, residual=info.get("residual", beta_true / b_norm), true_residual=beta_true / b_norm, converged=True)
            return x, info

        if P_inv_fn:
            r = P_inv_fn(r)
        beta = np.linalg.norm(r)
        if beta < 1e-30:
            info.update(iter=outer * restart_n, residual=0.0, true_residual=beta_true / b_norm, converged=(beta_true / b_norm) < tol)
            return x, info

        V = np.zeros((n, restart_n + 1), dtype=r.dtype)
        H = np.zeros((restart_n + 1, restart_n), dtype=np.result_type(r.dtype, np.float64))
        V[:, 0] = r / beta

        m_used = restart_n
        y = None
        residual_est = None

        for j in range(restart_n):
            w = A_fn(V[:, j])
            if P_inv_fn:
                w = P_inv_fn(w)
            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i].conj(), w)
                w -= H[i, j] * V[:, i]
            H[j+1, j] = np.linalg.norm(w)
            if H[j+1, j] < 1e-14:
                m_used = j + 1
                break
            V[:, j+1] = w / H[j+1, j]

            m = j + 1
            e1 = np.zeros(m + 1, dtype=H.dtype)
            e1[0] = beta
            y, res, _, _ = np.linalg.lstsq(H[:m+1, :m], e1, rcond=None)
            if len(res) > 0 and res[0] >= 0:
                residual_est = float(np.sqrt(res[0])) / b_norm
            else:
                residual_est = np.linalg.norm(H[:m+1, :m] @ y - e1) / b_norm
            info["iter"] = outer * restart_n + m
            info["residual"] = residual_est

            if residual_est < tol:
                x_trial = x + V[:, :m] @ y
                r_true = b - A_fn(x_trial)
                true_res = np.linalg.norm(r_true) / b_norm
                info["true_residual"] = true_res
                if true_res < tol:
                    info["converged"] = True
                    return x_trial, info

        if y is None:
            e1 = np.zeros(m_used + 1, dtype=H.dtype)
            e1[0] = beta
            y, _, _, _ = np.linalg.lstsq(H[:m_used+1, :m_used], e1, rcond=None)
        x = x + V[:, :m_used] @ y

        r_true = b - A_fn(x)
        true_res = np.linalg.norm(r_true) / b_norm
        info.update(iter=(outer + 1) * restart_n, true_residual=true_res, converged=true_res < tol)
        if info["converged"]:
            return x, info

    return x, info


# ---------------------------------------------------------------------------
# Batched resolvent sweep across frequencies
# ---------------------------------------------------------------------------

def batched_resolvent_sweep(omegas, resolvent, b, solver="minres", eta=0.0,
                            tol=1e-4, max_iter=200, restart=30,
                            preconditioner=None, warm_start=True, verbose=True,
                            response_mode="norm"):
    """
    Solve A(omega_k) x_k = b for each frequency in omegas.
    Returns spectrum response: (m,) array of ||x_k|| or x_k^H b.

    Parameters
    ----------
    omegas : (m,) array of frequencies
    resolvent : ResolventOperator
    b : (N,) probe vector
    solver : "block_jacobi", "minres", "cocr", "bicgstab", "gmres"
    eta : damping
    tol, max_iter, restart : solver params
    preconditioner : BlockJacobiPreconditioner or None
    warm_start : if True, use x_{k-1} as initial guess for x_k
    verbose : print progress

    Returns
    -------
    responses : (m,) array of ||x_k||
    x_all : (N, m) solution vectors
    info_list : list of per-solve info dicts
    """
    m = len(omegas)
    N = resolvent.ndim
    x_all = np.empty((N, m), dtype=complex if eta > 0 else float)
    responses = np.empty(m)
    info_list = []

    x0 = np.zeros(N, dtype=complex if eta > 0 else float)

    solvers = {
        "block_jacobi": solve_block_jacobi,
        "minres": solve_minres,
        "cocr": solve_cocr,
        "bicgstab": solve_bicgstab,
        "gmres": solve_gmres,
    }
    solve_fn = solvers.get(solver, solve_minres)

    for k in range(m):
        if verbose and k % 10 == 0:
            print(f"  freq {k}/{m}: omega={omegas[k]:.4f}")

        omega = omegas[k]

        # Build preconditioner for this omega
        P_inv_fn = None
        if preconditioner is not None:
            preconditioner.build(omega, eta)
            P_inv_fn = preconditioner.apply

        # Define A(x) for this omega
        if eta > 0 or solver == "cocr":
            A_fn = lambda x, o=omega: resolvent.matvec(o, x, eta)
        else:
            A_fn = lambda x, o=omega: resolvent.matvec_real(o, x)

        # Warm start
        if warm_start and k > 0:
            x0 = x_all[:, k-1].copy()

        # Solve
        if solver == "block_jacobi":
            xk, info = solve_fn(A_fn, b, x0, P_inv_fn, max_iter=max_iter, alpha=1.0, beta=0.2, check_conv=False)
        elif solver == "gmres":
            xk, info = solve_fn(A_fn, b, x0, P_inv_fn, tol=tol, max_iter=max_iter, restart=restart)
        else:
            xk, info = solve_fn(A_fn, b, x0, P_inv_fn, tol=tol, max_iter=max_iter)

        x_all[:, k] = xk

        r_true = b - A_fn(xk)
        b_norm = np.linalg.norm(b) + 1e-30
        info["true_residual"] = float(np.linalg.norm(r_true) / b_norm)
        info["omega"] = float(omega)

        if response_mode == "norm":
            responses[k] = np.linalg.norm(xk)
        elif response_mode == "proj_im":
            responses[k] = np.imag(np.vdot(b, xk))
        elif response_mode == "proj_abs":
            responses[k] = abs(np.vdot(b, xk))
        elif response_mode == "proj_real":
            responses[k] = np.real(np.vdot(b, xk))
        else:
            raise ValueError(f"Unknown response_mode: {response_mode}")
        info_list.append(info)

    return responses, x_all, info_list


# ---------------------------------------------------------------------------
# Stochastic DOS estimate from multiple probe vectors
# ---------------------------------------------------------------------------

def stochastic_dos_sweep(omegas, resolvent, n_probes=4, solver="minres", eta=0.0,
                         tol=1e-4, max_iter=200, preconditioner=None, warm_start=True,
                         omega_density=True, seed=42):
    """Stochastic DOS estimate using Hutchinson trace estimator.

    We want the *mass-weighted* resolvent trace in lambda-space:
        Tr[(Hmw - z I)^{-1}],   z=(omega+i*eta)^2

    Using r with E[r r^T]=I, we have:
        E[r^T (Hmw - z I)^{-1} r] = Tr[(Hmw - z I)^{-1}]

    In physical coordinates, this corresponds to choosing b = M^{1/2} r and solving
        (K - z M) x = b
    then the quadratic form satisfies:
        r^T y = b^T x,   where y = M^{1/2} x.

    Therefore we estimate DOS via Im(b^H x) and (optionally) multiply by 2*omega
    to convert lambda-density to omega-density.
    """
    N = resolvent.ndim
    m_sqrt = np.sqrt(resolvent.mass)
    dos = np.zeros(len(omegas), dtype=float)
    rng = np.random.default_rng(seed)
    for p in range(int(n_probes)):
        r = rng.integers(0, 2, size=N, dtype=np.int8) * 2 - 1  # +/-1
        b = (m_sqrt * r.astype(float)).astype(complex if eta > 0 else float)
        resp, _, _ = batched_resolvent_sweep(
            omegas, resolvent, b, solver=solver, eta=eta,
            tol=tol, max_iter=max_iter, preconditioner=preconditioner,
            warm_start=warm_start, verbose=False,
            response_mode="proj_im",
        )
        dos += resp
    dos /= max(1, int(n_probes))
    if omega_density:
        dos = dos * (2.0 * np.asarray(omegas, dtype=float))
    return dos
