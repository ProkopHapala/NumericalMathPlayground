# FastDirectSolvers

Approximate and exact eigensolvers for large sparse symmetric matrices
(stiffness K, mass M) arising from vibration problems of nanocrystals and
mechanical systems. The goal is to compute the low-frequency spectrum without
full O(N^3) diagonalization.

## Problem

Given H = M^{-1/2} K M^{-1/2} (mass-weighted Hamiltonian, dimension N = 3*n_atoms),
find the lowest eigenvalues (vibrational frequencies omega = sqrt(lambda)).
For large systems (N > 1000), full `numpy.eigh` becomes expensive. We exploit
spatial locality to split H into blocks, solve blocks independently, then
recover inter-block coupling.

## Files

### Code

| File | Role |
|------|------|
| `nested_solver.py` | Core solver library — all algorithms and GPU kernels |
| `test_nested_solver.py` | Integration test: loads benchmarks, runs all methods, plots |
| `test_gpu_gemm.py` | Unit test: verifies GPU GEMM kernels against NumPy |
| `test_gpu_ritz.py` | Unit test: verifies GPU Ritz correction against CPU |
| `Davidson_Eigensolver.py` | Standalone Davidson eigensolver (external, by Joshua Goings) |

### Documentation

| File | Content |
|------|---------|
| `Direct_Solver_Disection_DivideConquere.chat.md` | Full design discussion (AI chat) |
| `Direct_Solver_Disection_DivideConquere.progress.md` | Step-by-step progress log with benchmarks |
| `BlockCholesky.chat.md` | Discussion of block Cholesky / tiled LDL^T on GPU |

## Methods (all in nested_solver.py)

### 1. Recursive Coordinate Bisection (RCB)

`rcb_cluster_atoms(pos, n_clusters)` — recursively splits atoms along the
longest bounding-box axis. Produces spatially compact clusters. Reorders DOFs
by cluster so H becomes block-diagonal-ish.

### 2. Nested Dissection (ND)

`nested_dissection_reorder(H, pos, max_leaf_atoms)` — like RCB but introduces
a separator strip (middle 20% along the split axis) between two halves at each
level. The separator is processed last, yielding a bordered block-diagonal
form. Fill-in is confined to separator rows/columns.

### 3. Block-diagonal approximation

`build_block_diagonal_approximation(H, block_sizes)` — simply zeros out
off-diagonal blocks. Each diagonal block is diagonalized independently.
**Crude** — ignores all inter-cluster coupling (mean error ~30-60 omega for
nanocrystals where H is nearly dense).

### 4. Ritz correction (two-level)

`ritz_correction_from_blocks(H, blocks, n_modes_per_block, gpu_solver)` —
after block diagonalization, projects the full H onto the block-eigenvector
basis V: H_proj = V^T H V. This recovers inter-block coupling exactly (when
all modes kept) or approximately (when truncated to n_modes_per_block).
The similarity transform can be computed on GPU via tiled GEMM kernels.

### 5. Recursive exact AMLS

`recursive_exact_amls(H_perm, tree)` — Automated Multi-Level Substructuring.
Bottom-up: diagonalize leaf blocks, transform separator matrices into
leaf-eigenbasis, diagonalize reduced separator. Recurses up the ND tree.
**Exact** up to round-off. More numerically stable than Ritz in float32
because it uses hierarchical small transforms instead of one large
similarity transform.

### 6. Static condensation (Guyan reduction)

`static_condensation_spectrum(H, block_sizes)` — eliminates interior DOFs
by Schur complement at omega=0: S = H_ss - H_is^T H_ii^{-1} H_is.
Gives approximate separator eigenvalues. Only works when last block is
the separator.

### 7. RCM reordering

`rcm_reorder(H)` — Reverse Cuthill-McKee bandwidth reduction via scipy.
Included for comparison; not effective for nearly-dense nanocrystal matrices.

## GPU kernels (embedded as OpenCL C strings in nested_solver.py)

### block_jacobi
Batched cyclic Jacobi diagonalization of small symmetric blocks.
One workgroup per block, single-threaded per block, data in __local memory.
Limit: block size <= ~48 (constrained by GPU local memory ~48 KB).

### block_jacobi_padded
Parallel version: one thread per row, supports varying block sizes via
padding. Workgroup size = max_m. Blocks larger than local memory limit
fall back to CPU numpy.eigh.

### gemm_tall_skinny
Tiled C = A^T @ B for [K,N] x [K,m] -> [N,m].
2D tiled: local (32,8) = 256 threads, TILE_K=64.
Used for first step of similarity transform: temp = H @ V.

### gemm_skinny_transpose
Tiled C = A^T @ B for [N,m] x [N,m] -> [m,m].
2D tiled: local (16,16) = 256 threads, TILE_N=64.
Used for second step: H_proj = V^T @ temp.

## Benchmark results (nc_C_R5, 164 atoms, 492 DOF)

| Method | Time | Accuracy | Notes |
|--------|------|----------|-------|
| Exact `numpy.eigh` | 17 ms | Exact | Baseline |
| RCB block-diag | 2 ms | Poor (rmse 103) | Ignores all couplings |
| ND block-diag | 1 ms | Poor (rmse 114) | Same problem |
| Ritz (all modes) | 15 ms | Exact | Change of basis, no truncation |
| Truncated Ritz 8/blk | 3-7 ms | Moderate | Viable reduced model |
| AMLS exact (f64) | 20 ms | ~1e-10 | Numerically exact |
| GPU Jacobi (12 blocks) | 3 ms | ~1e-6 rel | Float32, fast |
| GPU GEMM (m=256) | 1.3 ms | ~1e-6 rel | Float32, verified |

### Larger systems

| System | Atoms | DOF | numpy.eigh | AMLS exact | GPU padded |
|--------|-------|-----|------------|------------|------------|
| R5 | 164 | 492 | 17 ms | 20 ms | 4.5 ms |
| R6 | 270 | 810 | 41 ms | 68 ms | 3.9 ms |
| R7 | 414 | 1242 | 123 ms | 197 ms | 8.4 ms |
| R8 | 558 | 1674 | 225 ms | 452 ms | 9.9 ms |

AMLS is currently slower than numpy.eigh for these sizes because Python
recursion overhead + many small eigh calls don't beat one large LAPACK call.
The divide-and-conquer advantage should appear for DOF > 5000.

## Key insight

For small nanocrystals, H is **dense** because MMFF vdW interactions have a
long cutoff relative to cluster size. Block-diagonal approximation is poor.
Ritz correction or AMLS is required for accuracy. For larger/truly sparse
systems, block-diagonal approximation becomes more effective.

## Usage

```bash
cd topics/LinearAlgebra/FastDirectSolvers

# Default: R5, 8 clusters, no GPU
python test_nested_solver.py --system nc_C_R5 --n_clusters 8

# With GPU block-Jacobi
python test_nested_solver.py --system nc_C_R5 --gpu

# Different cluster counts or leaf sizes
python test_nested_solver.py --system nc_C_R4 --n_clusters 4 --max_leaf_atoms 6

# Test GPU GEMM kernels standalone
python test_gpu_gemm.py

# Test GPU Ritz correction
python test_gpu_ritz.py
```

## Dependencies

- numpy, scipy, matplotlib
- pyopencl (for GPU kernels; falls back to CPU if unavailable)
- Benchmark data from `SpectralFiltering/spectral_solvers.py` (load_vibration_benchmark)
