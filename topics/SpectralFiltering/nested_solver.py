"""
Nested dissection + block diagonalization solver for vibration benchmarks.

- Geometric recursive bisection of atoms into clusters
- Reorder DOFs by cluster -> bordered block-diagonal form
- Extract and diagonalize blocks independently (Python + OpenCL Jacobi)
- Compare approximate spectrum to exact numpy eigh
"""

from __future__ import annotations

import json
import os
from pathlib import Path

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
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if device.type & cl.device_type.GPU and "NVIDIA" in device.vendor.upper():
                return cl.Context([device])
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if device.type & cl.device_type.GPU:
                return cl.Context([device])
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if device.type & cl.device_type.CPU:
                return cl.Context([device])
    raise RuntimeError("No OpenCL device found")


# ---------------------------------------------------------------------------
# Load system and build dense mass-weighted Hamiltonian
# ---------------------------------------------------------------------------

def _reconstruct_raw_K(path):
    """Reconstruct raw sparse stiffness K from blocks/neigh_idx/neigh_count."""
    d = np.load(path, allow_pickle=True)
    neigh_count = d["neigh_count"]
    neigh_idx = d["neigh_idx"]
    blocks = d["blocks"]
    natoms = len(neigh_count)
    rows, cols, data = [], [], []
    for p in range(natoms):
        for j in range(int(neigh_count[p])):
            o = int(neigh_idx[p, j])
            if o < 0:
                continue
            b = blocks[p, j]
            for ii in range(3):
                for jj in range(3):
                    val = float(b[ii, jj])
                    if abs(val) > 1e-12:
                        rows.append(o * 3 + ii)
                        cols.append(p * 3 + jj)
                        data.append(val)
    K = sp.coo_matrix((data, (rows, cols)), shape=(3 * natoms, 3 * natoms)).tocsr()
    K = K + K.T
    K = 0.5 * K
    return K


def load_system_dense(system: str, fixtures_dir: str | None = None, use_raw: bool = True):
    """Load benchmark and return dense H = M^{-1/2} K M^{-1/2}, positions, and metadata.

    If use_raw=True (default), reconstructs K from raw blocks/neighbor lists,
    giving the actual sparse stiffness matrix before projection.
    If use_raw=False, loads the projected K_csr which is dense due to
    rigid-mode projection.
    """
    path = resolve_benchmark_path(system, fixtures_dir)
    bench = load_vibration_benchmark(path)
    mass = bench["mass"]
    pos = bench.get("pos")
    if use_raw:
        K = _reconstruct_raw_K(path)
        Kd = K.toarray()
    else:
        K = bench["K"]
        Kd = K.toarray() if bench.get("K_dense") is None else bench["K_dense"]
    # mass-weighted H = M^{-1/2} K M^{-1/2}
    m_inv_sqrt = 1.0 / np.sqrt(mass)
    H = (m_inv_sqrt[:, None] * Kd) * m_inv_sqrt[None, :]
    H = 0.5 * (H + H.T)
    return {
        "H": H,
        "K": K,
        "mass": mass,
        "pos": pos,
        "ndof": bench["ndof"],
        "natoms": bench["ndof"] // 3,
        "omegas_vib": bench["omegas_vib"],
        "name": bench["name"],
    }


# ---------------------------------------------------------------------------
# Geometric recursive coordinate bisection (RCB)
# ---------------------------------------------------------------------------

def rcb_cluster_atoms(pos: np.ndarray, n_clusters: int):
    """
    Recursive coordinate bisection of atom positions.
    Returns (atom_to_cluster, cluster_centers).
    """
    natoms = pos.shape[0]
    atom_to_cluster = np.zeros(natoms, dtype=int)
    centers = np.zeros((n_clusters, 3), dtype=float)

    def _split(indices, cluster_id, n_splits):
        if n_splits == 0:
            atom_to_cluster[indices] = cluster_id
            centers[cluster_id] = pos[indices].mean(axis=0)
            return
        # find longest axis of bounding box
        bb_min = pos[indices].min(axis=0)
        bb_max = pos[indices].max(axis=0)
        spans = bb_max - bb_min
        axis = int(np.argmax(spans))
        # sort along axis and split
        order = np.argsort(pos[indices, axis])
        mid = len(indices) // 2
        left = indices[order[:mid]]
        right = indices[order[mid:]]
        _split(left, cluster_id, n_splits - 1)
        _split(right, cluster_id + 2 ** (n_splits - 1), n_splits - 1)

    n_splits = int(np.log2(n_clusters))
    assert 2 ** n_splits == n_clusters, "n_clusters must be power of 2"
    all_indices = np.arange(natoms)
    _split(all_indices, 0, n_splits)
    return atom_to_cluster, centers


def reorder_by_clusters(H: np.ndarray, atom_to_cluster: np.ndarray):
    """
    Reorder DOFs so that all DOFs of cluster 0 come first, then cluster 1, etc.
    Returns (H_reordered, perm, block_sizes).
    """
    natoms = len(atom_to_cluster)
    n_clusters = int(atom_to_cluster.max()) + 1
    # gather atom indices per cluster
    cluster_atoms = [np.where(atom_to_cluster == c)[0] for c in range(n_clusters)]
    # each atom contributes 3 DOFs
    dofs = []
    block_sizes = []
    for atoms in cluster_atoms:
        ndof_c = 3 * len(atoms)
        block_sizes.append(ndof_c)
        for a in atoms:
            dofs.extend([3 * a, 3 * a + 1, 3 * a + 2])
    perm = np.array(dofs, dtype=int)
    H_reordered = H[np.ix_(perm, perm)]
    return H_reordered, perm, np.array(block_sizes, dtype=int)


def extract_diagonal_blocks(H_reordered: np.ndarray, block_sizes: np.ndarray):
    """Extract list of dense diagonal blocks."""
    blocks = []
    start = 0
    for bs in block_sizes:
        blocks.append(H_reordered[start:start + bs, start:start + bs].copy())
        start += bs
    return blocks


def build_block_diagonal_approximation(H_reordered: np.ndarray, block_sizes: np.ndarray):
    """Zero out off-diagonal blocks, keeping only diagonal blocks."""
    H_bd = np.zeros_like(H_reordered)
    start = 0
    for bs in block_sizes:
        H_bd[start:start + bs, start:start + bs] = H_reordered[start:start + bs, start:start + bs]
        start += bs
    return H_bd


# ---------------------------------------------------------------------------
# Jacobi rotation (Python reference)
# ---------------------------------------------------------------------------

def jacobi_rotation_block(A: np.ndarray, tol: float = 1e-10, max_sweeps: int = 100):
    """
    Cyclic Jacobi diagonalization of a symmetric matrix A.
    Returns (eigenvalues, eigenvectors, n_sweeps).
    """
    A = np.array(A, dtype=float, copy=True)
    n = A.shape[0]
    V = np.eye(n, dtype=float)
    off_norm = lambda: np.sqrt(np.sum(A ** 2) - np.sum(np.diag(A) ** 2))
    off0 = off_norm()
    if off0 == 0:
        return np.diag(A).copy(), V, 0

    for sweep in range(max_sweeps):
        for p in range(n):
            for q in range(p + 1, n):
                if abs(A[p, q]) < tol:
                    continue
                tau = (A[q, q] - A[p, p]) / (2.0 * A[p, q])
                t = np.sign(tau) / (abs(tau) + np.sqrt(1.0 + tau ** 2))
                c = 1.0 / np.sqrt(1.0 + t ** 2)
                s = t * c
                # rotate
                app, aqq, apq = A[p, p], A[q, q], A[p, q]
                A[p, p] = c * c * app - 2.0 * c * s * apq + s * s * aqq
                A[q, q] = s * s * app + 2.0 * c * s * apq + c * c * aqq
                A[p, q] = A[q, p] = 0.0
                for k in range(n):
                    if k != p and k != q:
                        akp, akq = A[k, p], A[k, q]
                        A[k, p] = A[p, k] = c * akp - s * akq
                        A[k, q] = A[q, k] = s * akp + c * akq
                for k in range(n):
                    vkp, vkq = V[k, p], V[k, q]
                    V[k, p] = c * vkp - s * vkq
                    V[k, q] = s * vkp + c * vkq
        if off_norm() / off0 < tol:
            break
    eigs = np.diag(A)
    order = np.argsort(eigs)
    return eigs[order], V[:, order], sweep + 1


def diagonalize_blocks_python(blocks: list[np.ndarray]):
    """Diagonalize each block with numpy eigh (reference)."""
    all_eigs = []
    all_vecs = []
    for B in blocks:
        w, v = np.linalg.eigh(B)
        all_eigs.append(w)
        all_vecs.append(v)
    return np.concatenate(all_eigs), all_vecs


def diagonalize_blocks_jacobi(blocks: list[np.ndarray], tol=1e-10, max_sweeps=100):
    """Diagonalize each block with cyclic Jacobi."""
    all_eigs = []
    all_vecs = []
    sweeps = []
    for B in blocks:
        w, v, ns = jacobi_rotation_block(B, tol=tol, max_sweeps=max_sweeps)
        all_eigs.append(w)
        all_vecs.append(v)
        sweeps.append(ns)
    return np.concatenate(all_eigs), all_vecs, sweeps


# ---------------------------------------------------------------------------
# OpenCL batched block Jacobi (group-local memory)
# ---------------------------------------------------------------------------

_OPENCL_BLOCK_JACOBI = r"""
// Batched cyclic Jacobi for symmetric blocks of size m x m.
// Each workgroup handles one block.  Block stored in __local memory.
// Input: flat array blocks[n_blocks * m * m], row-major symmetric.
// Output: eigvals[n_blocks * m], eigvecs[n_blocks * m * m] (column-major eigenvectors).
//
// To support variable block sizes efficiently, the caller must pad/truncate
// to the same size per batch or launch one kernel per size class.

__kernel void block_jacobi(
    const int m,
    const int max_sweeps,
    const float tol,
    __global float* blocks,     // input symmetric blocks, size n*m*m
    __global float* eigvals,    // output diagonal
    __global float* eigvecs,   // output orthonormal basis V
    __local  float* A,         // m*m local
    __local  float* V          // m*m local
)
{
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const int n2 = m * m;

    // base pointers for this block
    __global float* gA = blocks + gid * n2;
    __global float* gV = eigvecs + gid * n2;

    // collaborative load into local memory
    for (int i = lid; i < n2; i += lsz) {
        A[i] = gA[i];
        // identity for V
        int r = i / m;
        int c = i % m;
        V[i] = (r == c) ? 1.0f : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // only one thread does the sequential Jacobi sweeps (small m)
    if (lid == 0) {
        float off0 = 0.0f;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                if (i != j) off0 += A[i * m + j] * A[i * m + j];
            }
        }
        off0 = sqrt(off0);
        if (off0 < tol) {
            for (int i = 0; i < m; ++i) eigvals[gid * m + i] = A[i * m + i];
            for (int i = 0; i < n2; ++i) gV[i] = V[i];
            return;
        }

        for (int sweep = 0; sweep < max_sweeps; ++sweep) {
            for (int p = 0; p < m; ++p) {
                for (int q = p + 1; q < m; ++q) {
                    float apq = A[p * m + q];
                    if (fabs(apq) < tol) continue;
                    float app = A[p * m + p];
                    float aqq = A[q * m + q];
                    float tau = (aqq - app) / (2.0f * apq);
                    float t = (tau >= 0.0f) ? 1.0f / (tau + sqrt(1.0f + tau * tau))
                                             : -1.0f / (-tau + sqrt(1.0f + tau * tau));
                    float c = 1.0f / sqrt(1.0f + t * t);
                    float s = t * c;

                    // rotate A
                    float a_pp_new = c * c * app - 2.0f * c * s * apq + s * s * aqq;
                    float a_qq_new = s * s * app + 2.0f * c * s * apq + c * c * aqq;
                    A[p * m + p] = a_pp_new;
                    A[q * m + q] = a_qq_new;
                    A[p * m + q] = A[q * m + p] = 0.0f;

                    for (int k = 0; k < m; ++k) {
                        if (k != p && k != q) {
                            float akp = A[k * m + p];
                            float akq = A[k * m + q];
                            A[k * m + p] = A[p * m + k] = c * akp - s * akq;
                            A[k * m + q] = A[q * m + k] = s * akp + c * akq;
                        }
                    }
                    // rotate V
                    for (int k = 0; k < m; ++k) {
                        float vkp = V[k * m + p];
                        float vkq = V[k * m + q];
                        V[k * m + p] = c * vkp - s * vkq;
                        V[k * m + q] = s * vkp + c * vkq;
                    }
                }
            }
            // check convergence
            float off = 0.0f;
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < m; ++j) {
                    if (i != j) off += A[i * m + j] * A[i * m + j];
                }
            }
            if (sqrt(off) / off0 < tol) break;
        }

        // copy diagonal eigenvalues and eigenvectors back
        for (int i = 0; i < m; ++i) eigvals[gid * m + i] = A[i * m + i];
        for (int i = 0; i < n2; ++i) gV[i] = V[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
"""

# ---------------------------------------------------------------------------
# Padded parallel Jacobi kernel (workgroup-parallel, supports varying m)
# ---------------------------------------------------------------------------
_OPENCL_BLOCK_JACOBI_PADDED = r"""
// Batched cyclic Jacobi with padded local memory.
// Workgroup size = padded dimension (max_m).  Each workgroup = one block.
// Threads collaborate: thread k handles row k (k < m), thread 0 computes angles.
// Blocks on host must be padded to max_m x max_m (zeros in padding).
// Varying actual block sizes are supported via block_sizes[gid].
//
// Memory layout (local): A[row*max_m + col], V[row*max_m + col]

__kernel void block_jacobi_padded(
    const int max_sweeps,
    const float tol,
    __global float* blocks,     // n_blocks * max_m * max_m  (padded on host)
    __global float* eigvals,    // n_blocks * max_m  (padded)
    __global float* eigvecs,    // n_blocks * max_m * max_m (padded)
    __global int* block_sizes,  // n_blocks (actual m per block)
    __local float* A,           // max_m * max_m
    __local float* V            // max_m * max_m
)
{
    int gid = get_group_id(0);
    int lid = get_local_id(0);
    int max_m = get_local_size(0);
    int m = block_sizes[gid];
    int n2 = max_m * max_m;
    int boff = gid * n2;
    int loff = gid * max_m;

    __global float* gA = blocks + boff;
    __global float* gV = eigvecs + boff;

    for (int i = lid; i < n2; i += max_m) {
        A[i] = gA[i];
        int r = i / max_m;
        int c = i % max_m;
        V[i] = (r == c && r < m) ? 1.0f : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int scratch_c = n2 - 2;
    int scratch_s = n2 - 1;

    float off0 = 0.0f;
    if (lid == 0) {
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j)
                if (i != j) off0 += A[i * max_m + j] * A[i * max_m + j];
        off0 = sqrt(off0);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) A[scratch_c] = off0;
    barrier(CLK_LOCAL_MEM_FENCE);
    off0 = A[scratch_c];

    if (off0 < tol) {
        if (lid == 0)
            for (int i = 0; i < m; ++i) eigvals[loff + i] = A[i * max_m + i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = lid; i < n2; i += max_m) gV[i] = V[i];
        return;
    }

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        for (int p = 0; p < m; ++p) {
            for (int q = p + 1; q < m; ++q) {
                if (lid == 0) {
                    float apq = A[p * max_m + q];
                    if (fabs(apq) < tol) {
                        A[scratch_c] = 1.0f; A[scratch_s] = 0.0f;
                    } else {
                        float app = A[p * max_m + p];
                        float aqq = A[q * max_m + q];
                        float tau = (aqq - app) / (2.0f * apq);
                        float t = (tau >= 0.0f)
                            ? 1.0f / (tau + sqrt(1.0f + tau * tau))
                            : -1.0f / (-tau + sqrt(1.0f + tau * tau));
                        float c = 1.0f / sqrt(1.0f + t * t);
                        float s = t * c;
                        A[scratch_c] = c; A[scratch_s] = s;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                float c = A[scratch_c];
                float s = A[scratch_s];
                if (c == 1.0f && s == 0.0f) continue;

                int k = lid;
                if (k < m && k != p && k != q) {
                    float akp = A[k * max_m + p];
                    float akq = A[k * max_m + q];
                    A[k * max_m + p] = c * akp - s * akq;
                    A[k * max_m + q] = s * akp + c * akq;
                    A[p * max_m + k] = A[k * max_m + p];
                    A[q * max_m + k] = A[k * max_m + q];
                }
                if (k < m) {
                    float vkp = V[k * max_m + p];
                    float vkq = V[k * max_m + q];
                    V[k * max_m + p] = c * vkp - s * vkq;
                    V[k * max_m + q] = s * vkp + c * vkq;
                }
                if (lid == 0) {
                    float app = A[p * max_m + p];
                    float aqq = A[q * max_m + q];
                    float apq_val = A[p * max_m + q];
                    A[p * max_m + p] = c*c*app - 2.0f*c*s*apq_val + s*s*aqq;
                    A[q * max_m + q] = s*s*app + 2.0f*c*s*apq_val + c*c*aqq;
                    A[p * max_m + q] = A[q * max_m + p] = 0.0f;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        if (lid == 0) {
            float off = 0.0f;
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < m; ++j)
                    if (i != j) off += A[i * max_m + j] * A[i * max_m + j];
            A[scratch_c] = sqrt(off) / off0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (A[scratch_c] < tol) break;
    }

    if (lid == 0)
        for (int i = 0; i < m; ++i) eigvals[loff + i] = A[i * max_m + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = lid; i < n2; i += max_m) gV[i] = V[i];
}
"""

# ---------------------------------------------------------------------------
# Tall-skinny GEMM kernels (float32, local-memory tiled)
# ---------------------------------------------------------------------------
_OPENCL_GEMM_TALL_SKINNY = r"""
// C = A^T @ B
// A: [K, N] row-major  (A[k, i] at A[k*N + i])
// B: [K, m] row-major  (B[k, j] at B[k*m + j])
// C: [N, m] row-major  (C[i, j] at C[i*m + j])
// 2D tiled output: each workgroup computes TILE_N x TILE_M tile of C.
// local size = (TILE_N, TILE_M) = (32, 8) => 256 threads

#define TILE_K 64
#define TILE_N 32
#define TILE_M 8

__kernel void gemm_tall_skinny(
    const int N,
    const int K,
    const int m,
    __global const float* A,
    __global const float* B,
    __global float* C,
    __local float* A_tile,
    __local float* B_tile
)
{
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int lid = ty * TILE_N + tx;

    int i = get_group_id(0) * TILE_N + tx;
    int j = get_group_id(1) * TILE_M + ty;

    float sum = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // load A tile: [TILE_K, TILE_N]
        for (int idx = lid; idx < TILE_K * TILE_N; idx += TILE_N * TILE_M) {
            int kk = idx / TILE_N;
            int nn = idx - kk * TILE_N;
            int ai = get_group_id(0) * TILE_N + nn;
            int ak = k0 + kk;
            float v = 0.0f;
            if (ai < N && ak < K) v = A[ak * N + ai];
            A_tile[kk * TILE_N + nn] = v;
        }
        // load B tile: [TILE_K, TILE_M]
        for (int idx = lid; idx < TILE_K * TILE_M; idx += TILE_N * TILE_M) {
            int kk = idx / TILE_M;
            int mm = idx - kk * TILE_M;
            int bj = get_group_id(1) * TILE_M + mm;
            int bk = k0 + kk;
            float v = 0.0f;
            if (bj < m && bk < K) v = B[bk * m + bj];
            B_tile[kk * TILE_M + mm] = v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (i < N && j < m) {
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) sum += A_tile[kk * TILE_N + tx] * B_tile[kk * TILE_M + ty];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < N && j < m) C[i * m + j] = sum;
}

"""

_OPENCL_GEMM_SKINNY_TRANSPOSE = r"""
// C = A^T @ B
// A: [N, m] row-major
// B: [N, m] row-major
// C: [m, m] row-major
// 2D tiled output: each workgroup computes TILE_M x TILE_M tile of C.
// local size = (TILE_M, TILE_M) = (16, 16) => 256 threads

#define TILE_N 64
#define TILE_M 16

__kernel void gemm_skinny_transpose(
    const int N,
    const int m,
    __global const float* A,
    __global const float* B,
    __global float* C,
    __local float* A_tile,
    __local float* B_tile
)
{
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int lid = ty * TILE_M + tx;

    int row = get_group_id(0) * TILE_M + tx;
    int col = get_group_id(1) * TILE_M + ty;

    float sum = 0.0f;

    for (int n0 = 0; n0 < N; n0 += TILE_N) {
        // load A tile: [TILE_N, TILE_M] for this row-tile
        for (int idx = lid; idx < TILE_N * TILE_M; idx += TILE_M * TILE_M) {
            int t = idx / TILE_M;
            int mm = idx - t * TILE_M;
            int a_col = get_group_id(0) * TILE_M + mm;
            int n = n0 + t;
            float v = 0.0f;
            if (n < N && a_col < m) v = A[n * m + a_col];
            A_tile[t * TILE_M + mm] = v;
        }
        // load B tile: [TILE_N, TILE_M] for this col-tile
        for (int idx = lid; idx < TILE_N * TILE_M; idx += TILE_M * TILE_M) {
            int t = idx / TILE_M;
            int mm = idx - t * TILE_M;
            int b_col = get_group_id(1) * TILE_M + mm;
            int n = n0 + t;
            float v = 0.0f;
            if (n < N && b_col < m) v = B[n * m + b_col];
            B_tile[t * TILE_M + mm] = v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (row < m && col < m) {
            #pragma unroll
            for (int t = 0; t < TILE_N; ++t) sum += A_tile[t * TILE_M + tx] * B_tile[t * TILE_M + ty];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < m && col < m) C[row * m + col] = sum;
}

"""


def _get_nvidia_context():
    """Auto-select NVIDIA GPU if available, else first GPU, else CPU. No prompts."""
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if device.type & cl.device_type.GPU and "NVIDIA" in device.vendor.upper():
                return cl.Context([device])
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if device.type & cl.device_type.GPU:
                return cl.Context([device])
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if device.type & cl.device_type.CPU:
                return cl.Context([device])
    raise RuntimeError("No OpenCL device found")


class OpenCLBlockJacobi:
    """Batched dense symmetric block diagonalization via cyclic Jacobi on GPU."""

    def __init__(self):
        if not _HAS_OPENCL:
            raise RuntimeError("pyopencl not installed")
        self.ctx = _get_nvidia_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, _OPENCL_BLOCK_JACOBI).build()
        # compile GEMM kernels
        self._gemm_prg = cl.Program(self.ctx, _OPENCL_GEMM_TALL_SKINNY).build()
        self._gemmT_prg = cl.Program(self.ctx, _OPENCL_GEMM_SKINNY_TRANSPOSE).build()
        self._max_local = min(d.local_mem_size for d in self.ctx.devices)
        self._max_m_local = int(np.sqrt(self._max_local / 4 / 2))
        # round down to warp-size multiple (NVIDIA = 32)
        self._max_m_local = (self._max_m_local // 32) * 32
        # probe actual max workgroup size for this kernel's local memory pattern
        prg_probe = cl.Program(self.ctx, _OPENCL_BLOCK_JACOBI_PADDED).build()
        knl_probe = prg_probe.block_jacobi_padded
        dev = self.ctx.devices[0]
        q = cl.CommandQueue(self.ctx)
        for wg in [64, 48, 32, 16]:
            a = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, 3 * wg * wg * 4)
            v = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, 3 * wg * wg * 4)
            l = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, 3 * wg * 4)
            s = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, 3 * 4)
            try:
                knl_probe(q, (3 * wg,), (wg,), np.int32(1), np.float32(1.0),
                        a, l, v, s, cl.LocalMemory(2 * wg * wg * 4), cl.LocalMemory(2 * wg * wg * 4))
                self._max_m_local = wg
                break
            except Exception:
                continue
        print(f"[OpenCL] device local memory = {self._max_local} B -> max_m = {self._max_m_local}")

    def diagonalize(self, blocks: list[np.ndarray], tol: float = 1e-6, max_sweeps: int = 50):
        """
        blocks: list of symmetric float32 matrices (must all be same size for this kernel).
        Returns (eigvals_array[n_blocks, m], eigvecs_array[n_blocks, m, m]).
        """
        n_blocks = len(blocks)
        m = blocks[0].shape[0]
        assert all(B.shape == (m, m) for B in blocks), "All blocks must have same size"
        blocks_flat = np.stack(blocks).astype(np.float32).ravel()  # (n_blocks, m, m)
        eigvals = np.empty((n_blocks, m), dtype=np.float32)
        eigvecs = np.empty((n_blocks, m, m), dtype=np.float32)

        mf = cl.mem_flags
        a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=blocks_flat)
        v_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, eigvecs.nbytes)
        l_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, eigvals.nbytes)

        # local memory size: 2 * m * m floats
        local_bytes = 2 * m * m * np.dtype(np.float32).itemsize
        # one workgroup per block, workgroup size = 1 (sequential per block) or small
        wg_size = 1

        self.prg.block_jacobi(
            self.queue, (n_blocks,), (wg_size,),
            np.int32(m), np.int32(max_sweeps), np.float32(tol),
            a_buf, l_buf, v_buf,
            cl.LocalMemory(local_bytes), cl.LocalMemory(local_bytes),
        )
        cl.enqueue_copy(self.queue, eigvals, l_buf)
        cl.enqueue_copy(self.queue, eigvecs, v_buf)
        return eigvals, eigvecs

    # ------------------------------------------------------------------
    # Tall-skinny GEMM: C = A @ B
    # A: [K, N] column-major, B: [K, m], C: [N, m]
    # ------------------------------------------------------------------
    def gemm_tall_skinny(self, A_colmaj, B, N, K, m):
        """
        C = A @ B on GPU.
        A_colmaj: np.ndarray shape (K, N) in C-order (column-major layout).
        B: np.ndarray shape (K, m).
        Returns C: np.ndarray shape (N, m).
        All arrays are float32.
        """
        assert A_colmaj.shape == (K, N), f"A_colmaj shape {A_colmaj.shape} != ({K}, {N})"
        assert B.shape == (K, m), f"B shape {B.shape} != ({K}, {m})"
        assert A_colmaj.dtype == np.float32
        assert B.dtype == np.float32

        C = np.empty((N, m), dtype=np.float32)
        mf = cl.mem_flags
        a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_colmaj)
        b_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        c_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, C.nbytes)

        # 2D tiled output: local (32, 8), global (ceil(N/32)*32, ceil(m/8)*8)
        TILE_K = 64
        TILE_N = 32
        TILE_M = 8
        global_size = (((N + TILE_N - 1) // TILE_N) * TILE_N,
                       ((m + TILE_M - 1) // TILE_M) * TILE_M)
        local_size = (TILE_N, TILE_M)
        a_tile_bytes = TILE_K * TILE_N * np.dtype(np.float32).itemsize
        b_tile_bytes = TILE_K * TILE_M * np.dtype(np.float32).itemsize

        self._gemm_prg.gemm_tall_skinny(
            self.queue, global_size, local_size,
            np.int32(N), np.int32(K), np.int32(m),
            a_buf, b_buf, c_buf,
            cl.LocalMemory(a_tile_bytes), cl.LocalMemory(b_tile_bytes),
        )
        cl.enqueue_copy(self.queue, C, c_buf)
        return C

    # ------------------------------------------------------------------
    # Skinny-transpose GEMM: C = A.T @ B
    # A, B: [N, m], C: [m, m]
    # ------------------------------------------------------------------
    def gemm_skinny_transpose(self, A, B, N, m):
        """
        C = A.T @ B on GPU.
        A, B: np.ndarray shape (N, m), row-major.
        Returns C: np.ndarray shape (m, m).
        """
        assert A.shape == (N, m)
        assert B.shape == (N, m)
        assert A.dtype == np.float32
        assert B.dtype == np.float32

        C = np.empty((m, m), dtype=np.float32)
        mf = cl.mem_flags
        a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        b_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        c_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, C.nbytes)

        # 2D tiled output: local (16,16), global (ceil(m/16)*16, ceil(m/16)*16)
        TILE_N = 64
        TILE_M = 16
        global_size = (((m + TILE_M - 1) // TILE_M) * TILE_M,
                       ((m + TILE_M - 1) // TILE_M) * TILE_M)
        local_size = (TILE_M, TILE_M)
        a_tile_bytes = TILE_N * TILE_M * np.dtype(np.float32).itemsize
        b_tile_bytes = TILE_N * TILE_M * np.dtype(np.float32).itemsize

        self._gemmT_prg.gemm_skinny_transpose(
            self.queue, global_size, local_size,
            np.int32(N), np.int32(m),
            a_buf, b_buf, c_buf,
            cl.LocalMemory(a_tile_bytes), cl.LocalMemory(b_tile_bytes),
        )
        cl.enqueue_copy(self.queue, C, c_buf)
        return C

    # ------------------------------------------------------------------
    # GPU similarity transform: H_proj = V^T @ H @ V
    # H: [n, n] symmetric float32, V: [n, m] float32
    # ------------------------------------------------------------------
    def similarity_transform(self, H, V, n, m):
        """
        Compute H_proj = V.T @ H @ V on GPU.
        H must be symmetric (so H = H.T).
        Returns H_proj: [m, m] float32, symmetrized.
        """
        assert H.shape == (n, n), f"H shape {H.shape} != ({n}, {n})"
        assert V.shape == (n, m), f"V shape {V.shape} != ({n}, {m})"
        assert H.dtype == np.float32
        assert V.dtype == np.float32
        # temp = H @ V  (H symmetric => H @ V = H.T @ V, kernel computes A.T @ B)
        temp = self.gemm_tall_skinny(H, V, n, n, m)
        # H_proj = V.T @ temp
        H_proj = self.gemm_skinny_transpose(V, temp, n, m)
        # Force symmetry (critical for float32 stability)
        H_proj = 0.5 * (H_proj + H_proj.T)
        return H_proj

    def diagonalize_padded(self, blocks: list[np.ndarray], tol: float = 1e-6,
                           max_sweeps: int = 50, max_m: int | None = None):
        """
        Padded batched Jacobi with varying block sizes.
        Workgroup size = max_m (each thread handles one row).
        Blocks larger than max_m_local are processed on CPU.
        Returns (eigvals_list, eigvecs_list) with actual sizes.
        """
        if not blocks:
            return [], []
        if max_m is None:
            max_m = max(b.shape[0] for b in blocks)
        if max_m > self._max_m_local:
            print(f"[OpenCL] max_m={max_m} exceeds local memory limit {self._max_m_local}; splitting")
        gpu_max_m = min(max_m, self._max_m_local)

        # Split into GPU-fittable and CPU-fallback blocks
        gpu_blocks = []
        gpu_indices = []
        cpu_blocks = []
        cpu_indices = []
        for i, B in enumerate(blocks):
            if B.shape[0] <= gpu_max_m:
                gpu_blocks.append(B)
                gpu_indices.append(i)
            else:
                cpu_blocks.append(B)
                cpu_indices.append(i)

        eigvals_out = [None] * len(blocks)
        eigvecs_out = [None] * len(blocks)

        # CPU fallback for oversized blocks
        for idx, B in zip(cpu_indices, cpu_blocks):
            w, v = np.linalg.eigh(B.astype(np.float32))
            eigvals_out[idx] = w
            eigvecs_out[idx] = v

        if not gpu_blocks:
            return eigvals_out, eigvecs_out

        n_blocks = len(gpu_blocks)
        block_sizes_arr = np.array([B.shape[0] for B in gpu_blocks], dtype=np.int32)

        padded = np.zeros((n_blocks, gpu_max_m, gpu_max_m), dtype=np.float32)
        for i, B in enumerate(gpu_blocks):
            m = B.shape[0]
            padded[i, :m, :m] = B.astype(np.float32)
        blocks_flat = padded.ravel()

        eigvals = np.empty((n_blocks, gpu_max_m), dtype=np.float32)
        eigvecs = np.empty((n_blocks, gpu_max_m, gpu_max_m), dtype=np.float32)

        mf = cl.mem_flags
        a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=blocks_flat)
        v_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, eigvecs.nbytes)
        l_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, eigvals.nbytes)
        s_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=block_sizes_arr)

        local_bytes = 2 * gpu_max_m * gpu_max_m * np.dtype(np.float32).itemsize

        if not hasattr(self, '_prg_padded'):
            self._prg_padded = cl.Program(self.ctx, _OPENCL_BLOCK_JACOBI_PADDED).build()
            dev = self.ctx.devices[0]
            knl = self._prg_padded.block_jacobi_padded
            self._knl_max_wg = knl.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, dev)
            print(f"[OpenCL] padded kernel max workgroup size = {self._knl_max_wg}")

        # further cap by kernel-specific resource limits
        gpu_max_m = min(gpu_max_m, self._knl_max_wg)

        self._prg_padded.block_jacobi_padded(
            self.queue, (n_blocks * gpu_max_m,), (gpu_max_m,),
            np.int32(max_sweeps), np.float32(tol),
            a_buf, l_buf, v_buf, s_buf,
            cl.LocalMemory(local_bytes), cl.LocalMemory(local_bytes),
        )
        cl.enqueue_copy(self.queue, eigvals, l_buf)
        cl.enqueue_copy(self.queue, eigvecs, v_buf)

        for i, idx in enumerate(gpu_indices):
            m = gpu_blocks[i].shape[0]
            eigvals_out[idx] = eigvals[i, :m].copy()
            eigvecs_out[idx] = eigvecs[i, :m, :m].copy()
        return eigvals_out, eigvecs_out

    # ------------------------------------------------------------------
    # Tall-skinny GEMM: C = A @ B
    # A: [K, N] column-major, B: [K, m], C: [N, m]
    # ------------------------------------------------------------------
    def gemm_tall_skinny_legacy(self, A_colmaj, B, N, K, m):
        """
        C = A @ B on GPU.
        A_colmaj: np.ndarray shape (K, N) in C-order (column-major layout).
        B: np.ndarray shape (K, m).
        Returns C: np.ndarray shape (N, m).
        All arrays are float32.
        """
        assert A_colmaj.shape == (K, N), f"A_colmaj shape {A_colmaj.shape} != ({K}, {N})"
        assert B.shape == (K, m), f"B shape {B.shape} != ({K}, {m})"
        assert A_colmaj.dtype == np.float32
        assert B.dtype == np.float32

        C = np.empty((N, m), dtype=np.float32)
        mf = cl.mem_flags
        a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_colmaj)
        b_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        c_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, C.nbytes)

        local_bytes = 64 * m * np.dtype(np.float32).itemsize
        global_size = ((N + 31) // 32) * 32

        self._gemm_prg.gemm_tall_skinny(
            self.queue, (global_size,), (32,),
            np.int32(N), np.int32(K), np.int32(m),
            a_buf, b_buf, c_buf,
            cl.LocalMemory(local_bytes),
        )
        cl.enqueue_copy(self.queue, C, c_buf)
        return C

    # ------------------------------------------------------------------
    # Skinny-transpose GEMM: C = A.T @ B
    # A, B: [N, m], C: [m, m]
    # ------------------------------------------------------------------
    def gemm_skinny_transpose_legacy(self, A, B, N, m):
        """
        C = A.T @ B on GPU.
        A, B: np.ndarray shape (N, m), row-major.
        Returns C: np.ndarray shape (m, m).
        """
        assert A.shape == (N, m)
        assert B.shape == (N, m)
        assert A.dtype == np.float32
        assert B.dtype == np.float32

        C = np.empty((m, m), dtype=np.float32)
        mf = cl.mem_flags
        a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        b_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        c_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, C.nbytes)

        local_bytes = 64 * m * np.dtype(np.float32).itemsize
        wg = 32 if m <= 32 else 64
        global_size = wg

        self._gemmT_prg.gemm_skinny_transpose(
            self.queue, (global_size,), (wg,),
            np.int32(N), np.int32(m),
            a_buf, b_buf, c_buf,
            cl.LocalMemory(local_bytes), cl.LocalMemory(local_bytes),
        )
        cl.enqueue_copy(self.queue, C, c_buf)
        return C


# ---------------------------------------------------------------------------
# Simple nested dissection (geometric) with separator
# ---------------------------------------------------------------------------

def nested_dissection_atoms(pos: np.ndarray, max_leaf_atoms: int = 8):
    """
    Recursive geometric nested dissection.
    Returns a tree structure: list of (type, indices, children) where
    type is 'leaf' or 'separator'.
    """
    natoms = pos.shape[0]
    all_indices = np.arange(natoms)

    def _recurse(indices):
        if len(indices) <= max_leaf_atoms:
            return {"type": "leaf", "atoms": indices, "children": []}
        bb_min = pos[indices].min(axis=0)
        bb_max = pos[indices].max(axis=0)
        spans = bb_max - bb_min
        axis = int(np.argmax(spans))
        order = np.argsort(pos[indices, axis])
        # separator = middle 20% of atoms along longest axis
        n = len(indices)
        sep_start = int(0.4 * n)
        sep_end = int(0.6 * n)
        if sep_start == 0 or sep_end == n:
            sep_start = n // 3
            sep_end = 2 * n // 3
        left = indices[order[:sep_start]]
        sep = indices[order[sep_start:sep_end]]
        right = indices[order[sep_end:]]
        return {
            "type": "separator",
            "atoms": sep,
            "children": [_recurse(left), _recurse(right)],
        }

    return _recurse(all_indices)


def tree_to_dofs(tree, perm_list, block_list):
    """Flatten tree into DOF permutation and block sizes."""
    def _traverse(node):
        if node["type"] == "leaf":
            atoms = node["atoms"]
            ndof = 3 * len(atoms)
            block_list.append(ndof)
            for a in atoms:
                perm_list.extend([3 * a, 3 * a + 1, 3 * a + 2])
        else:
            # recurse children first, then separator (standard nested dissection)
            for child in node["children"]:
                _traverse(child)
            atoms = node["atoms"]
            ndof = 3 * len(atoms)
            if ndof > 0:
                block_list.append(ndof)
                for a in atoms:
                    perm_list.extend([3 * a, 3 * a + 1, 3 * a + 2])
    _traverse(tree)


def nested_dissection_reorder(H: np.ndarray, pos: np.ndarray, max_leaf_atoms: int = 8):
    """
    Reorder H using geometric nested dissection.
    Returns (H_reordered, perm, block_sizes, tree).
    """
    tree = nested_dissection_atoms(pos, max_leaf_atoms)
    perm_list = []
    block_list = []
    tree_to_dofs(tree, perm_list, block_list)
    perm = np.array(perm_list, dtype=int)
    H_reordered = H[np.ix_(perm, perm)]
    return H_reordered, perm, np.array(block_list, dtype=int), tree


# ---------------------------------------------------------------------------
# RCM (Reverse Cuthill-McKee) reordering for narrow bandwidth
# ---------------------------------------------------------------------------

def rcm_reorder(H: np.ndarray):
    """
    Reverse Cuthill-McKee reordering using scipy.sparse.csgraph.
    Returns (H_reordered, perm, bandwidth).
    """
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    # Build adjacency from nonzero pattern of symmetric matrix
    S = sp.csr_matrix(np.abs(H) > 1e-8)
    perm = reverse_cuthill_mckee(S, symmetric_mode=True)
    perm = np.asarray(perm).ravel()
    H_reordered = H[np.ix_(perm, perm)]
    # half-bandwidth (max distance above diagonal for nonzeros)
    n = H_reordered.shape[0]
    rows, cols = np.where(np.abs(H_reordered) > 1e-8)
    upper = cols - rows
    upper = upper[upper >= 0]
    bw = int(upper.max()) if len(upper) > 0 else 0
    return H_reordered, perm, bw


# ---------------------------------------------------------------------------
# Ritz projection correction (two-level method)
# ---------------------------------------------------------------------------

def ritz_correction_from_blocks(H: np.ndarray, blocks: list[np.ndarray],
                                n_modes_per_block: int | None = None,
                                gpu_solver: 'OpenCLBlockJacobi | None' = None):
    """
    Compute block eigenvectors, then project the FULL H onto the block-eigenvector basis.
    This gives a corrected spectrum that accounts for off-diagonal coupling.

    If n_modes_per_block is None, keep ALL modes per block (exact size).
    If set to e.g. 8, truncate to lowest 8 modes per block -> reduced model.

    gpu_solver: optional OpenCLBlockJacobi instance. If provided and the
    reduced dimension m <= 64, the similarity transform V^T @ H @ V is computed
    on GPU for float32 data.
    """
    n = H.shape[0]
    start = 0
    basis_list = []
    for B in blocks:
        bs = B.shape[0]
        w, v = np.linalg.eigh(B)
        if n_modes_per_block is not None:
            keep = min(n_modes_per_block, bs)
            v = v[:, :keep]
        basis_list.append(v)
        start += bs

    V = sp.block_diag(basis_list).toarray()  # n x n (or n x reduced)
    m = V.shape[1]

    # GPU path: similarity transform on GPU for float32
    if gpu_solver is not None and H.dtype == np.float32:
        V = V.astype(np.float32, copy=False)
        H_proj = gpu_solver.similarity_transform(H, V, n, m)
    else:
        # CPU path
        H_proj = V.T @ H @ V
        # Force symmetry (critical for float32 stability)
        H_proj = 0.5 * (H_proj + H_proj.T)

    eigs_proj, vecs_proj = np.linalg.eigh(H_proj)
    # Full eigenvectors / Ritz vectors in original basis
    V_full = V @ vecs_proj
    return eigs_proj, V_full, H_proj


# ---------------------------------------------------------------------------
# Recursive exact AMLS (Automated Multi-Level Substructuring)
# ---------------------------------------------------------------------------

def _annotate_subtree_ranges(tree, start=0):
    """
    Annotate tree nodes with 'subtree_start' / 'subtree_end' (contiguous range
    of ALL DOFs belonging to this subtree) and 'dof_start' / 'dof_end'
    (range of this node's own separator DOFs, empty for leaves).
    In-place.
    """
    if tree["type"] == "leaf":
        ndof = 3 * len(tree["atoms"])
        tree["subtree_start"] = start
        tree["subtree_end"] = start + ndof
        tree["dof_start"] = start
        tree["dof_end"] = start + ndof
        return start + ndof
    else:
        offset = start
        for child in tree["children"]:
            offset = _annotate_subtree_ranges(child, offset)
        ndof = 3 * len(tree["atoms"])
        tree["subtree_start"] = start
        tree["subtree_end"] = offset + ndof
        tree["dof_start"] = offset
        tree["dof_end"] = offset + ndof
        return offset + ndof


def recursive_exact_amls(H_perm: np.ndarray, tree: dict):
    """
    Recursive exact AMLS solver.
    Bottom-up: diagonalize leaves, transform separators into leaf-eigenbasis,
    then diagonalize the reduced separator matrix.
    Returns exact eigenvalues (up to round-off) and full eigenvectors.
    """
    _annotate_subtree_ranges(tree)

    def _solve(node):
        i, j = node["subtree_start"], node["subtree_end"]
        H_node = H_perm[i:j, i:j]
        n = j - i

        if node["type"] == "leaf":
            w, V = np.linalg.eigh(H_node)
            return w, V

        # Separator: recursively solve children, embed their eigenbases
        V_local = np.eye(n)
        for child in node["children"]:
            ci, cj = child["subtree_start"], child["subtree_end"]
            offset = ci - i  # position of child's subtree within H_node
            cw, cV = _solve(child)
            child_ndof = cj - ci
            V_local[offset:offset + child_ndof, offset:offset + child_ndof] = cV

        # Transform H_node into child-eigenbasis + separator-identity
        H_reduced = V_local.T @ H_node @ V_local

        # Diagonalize the reduced matrix
        w, V_red = np.linalg.eigh(H_reduced)

        # Full eigenvectors in this node's DOF ordering
        V_full = V_local @ V_red
        return w, V_full

    w, V = _solve(tree)
    return w, V


# ---------------------------------------------------------------------------
# Static condensation (Guyan reduction at omega=0)
# ---------------------------------------------------------------------------

def static_condensation_spectrum(H_reordered: np.ndarray, block_sizes: np.ndarray):
    """
    2-level static condensation (Guyan reduction).
    Assumes last block is the separator; all preceding blocks are interior.
    Returns approximate eigenvalues.
    """
    n = H_reordered.shape[0]
    n_interior = int(block_sizes[:-1].sum())
    n_sep = int(block_sizes[-1])
    assert n_interior + n_sep == n

    H_ii = H_reordered[:n_interior, :n_interior]
    H_is = H_reordered[:n_interior, n_interior:]
    H_ss = H_reordered[n_interior:, n_interior:]

    # H_ii is block-diagonal; invert each block
    H_ii_inv = np.zeros_like(H_ii)
    start = 0
    for bs in block_sizes[:-1]:
        block = H_ii[start:start + bs, start:start + bs]
        H_ii_inv[start:start + bs, start:start + bs] = np.linalg.inv(block)
        start += bs

    # Reduced (Schur complement at omega=0)
    S = H_ss - H_is.T @ H_ii_inv @ H_is
    w_sep, _ = np.linalg.eigh(S)
    return w_sep
