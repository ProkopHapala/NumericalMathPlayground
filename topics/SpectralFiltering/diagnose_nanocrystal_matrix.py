"""
Diagnose nanocrystal benchmark matrices to understand why they appear dense.

Key finding: The .npz file contains BOTH:
  1. `K_csr_*` — PROJECTED stiffness matrix (dense due to projection operation)
  2. `blocks`, `neigh_idx`, `neigh_count` — raw 3×3 force-constant blocks (sparse!)

The raw Hessian reconstructed from blocks has ~6% density and the correct
vibrational spectrum (ω ≈ 0.26 … 4.3). The projection to remove rigid modes
fills in zeros, making K_csr essentially 100% dense.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from spectral_solvers import resolve_benchmark_path


def load_raw_sparse_K(path):
    """Reconstruct raw sparse K from blocks/neigh_idx/neigh_count."""
    d = np.load(path, allow_pickle=True)
    neigh_count = d["neigh_count"]
    neigh_idx = d["neigh_idx"]
    blocks = d["blocks"]
    natoms = len(neigh_count)
    N = 3 * natoms

    rows, cols, data = [], [], []
    for p in range(natoms):
        for j in range(int(neigh_count[p])):
            o = int(neigh_idx[p, j])
            if o < 0:
                continue
            b = blocks[p, j]  # (3, 3)
            for ii in range(3):
                for jj in range(3):
                    val = b[ii, jj]
                    if abs(val) > 1e-12:
                        rows.append(o * 3 + ii)
                        cols.append(p * 3 + jj)
                        data.append(val)

    K = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    K = K + K.T
    K = 0.5 * K  # symmetrize
    return K


def diagnose(system="nc_C_R5", save_dir="/tmp"):
    path = resolve_benchmark_path(system)
    d = np.load(path, allow_pickle=True)
    print(f"=== {system} ===")
    print(f"  natoms={d['natoms']}, DOF={d['ndof']}")

    # 1) K_csr from .npz (PROJECTED)
    K_proj = sp.csr_matrix(
        (d["K_csr_data"], d["K_csr_indices"], d["K_csr_indptr"]),
        shape=tuple(d["K_csr_shape"]),
    )
    N = K_proj.shape[0]
    print(f"\n  K_csr (PROJECTED): N={N}, nnz={K_proj.nnz}, density={K_proj.nnz/(N*N):.4f}")

    # 2) Raw reconstructed K (from blocks)
    K_raw = load_raw_sparse_K(path)
    print(f"  K_raw (from blocks): N={N}, nnz={K_raw.nnz}, density={K_raw.nnz/(N*N):.4f}")

    # 3) Check spectra
    mass = d["mass_diag"]
    m_inv_sqrt = 1.0 / np.sqrt(mass)

    H_proj = (m_inv_sqrt[:, None] * K_proj.toarray()) * m_inv_sqrt[None, :]
    H_proj = 0.5 * (H_proj + H_proj.T)
    eigs_proj = np.linalg.eigvalsh(H_proj)
    omegas_proj = np.sqrt(np.clip(eigs_proj, 0, None))

    H_raw = (m_inv_sqrt[:, None] * K_raw.toarray()) * m_inv_sqrt[None, :]
    H_raw = 0.5 * (H_raw + H_raw.T)
    eigs_raw = np.linalg.eigvalsh(H_raw)
    omegas_raw = np.sqrt(np.clip(eigs_raw, 0, None))

    print(f"\n  Projected H spectrum: omega={omegas_proj.min():.4f}..{omegas_proj.max():.4f}")
    print(f"  Raw H spectrum:       omega={omegas_raw.min():.4f}..{omegas_raw.max():.4f}")
    print(f"  Reference omegas:     {d['omegas_modes_projected'][:3]}..{d['omegas_modes_projected'][-3:]}")

    # 4) Plot sparsity patterns
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def plot_log(ax, M, title):
        if sp.issparse(M):
            M = M.toarray()
        logM = np.log10(np.abs(M) + 1e-12)
        vmax = np.percentile(logM[logM > -12], 99)
        vmin = max(logM.min(), -12)
        im = ax.imshow(logM, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("DOF")
        ax.set_ylabel("DOF")
        fig.colorbar(im, ax=ax, label="log10(|K_ij|)")

    plot_log(axes[0], K_proj, f"K_csr PROJECTED (density={K_proj.nnz/(N*N):.3f})")
    plot_log(axes[1], K_raw, f"K_raw from blocks (density={K_raw.nnz/(N*N):.3f})")

    # Difference
    diff = K_proj.toarray() - K_raw.toarray()
    plot_log(axes[2], diff, "Difference: K_proj - K_raw")

    plt.tight_layout()
    out = f"{save_dir}/diag_{system}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  Saved plot to {out}")

    # 5) Check eigenvalue match
    vib_proj = omegas_proj[omegas_proj < 100]
    vib_raw = omegas_raw[omegas_raw < 100]
    n = min(len(vib_proj), len(vib_raw))
    print(f"\n  Max |omega_proj - omega_raw| (vib only, first {n}): {np.max(np.abs(vib_proj[:n] - vib_raw[:n])):.4f}")


diagnose("nc_C_R5")
