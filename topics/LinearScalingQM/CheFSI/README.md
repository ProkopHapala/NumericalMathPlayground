# CheFSI

Chebyshev Filtering Subspace Iteration (CheFSI) — GPU-accelerated solver for
finding frontier orbitals (HOMO/LUMO) of large sparse Hamiltonians.

## Files

| File | Role |
|------|------|
| `CheFSI.py` | Python solver class: PyOpenCL implementation with ELLPACK sparse format, Chebyshev polynomial filtering, Gram matrix, and subspace orthogonalization |
| `CheFSI.cl` | OpenCL kernels: SpMM (ELLPACK format), Gram matrix assembly, subspace orthogonalization |
| `test_CheFSI.py` | Test script: generates 1D distorted chain, computes reference via scipy.linalg.eigh, compares GPU frontier orbitals with convergence/orthogonality diagnostics |
| `CheFSI_Frontier_Orbitals.chat.md` | Design discussion: interior eigenvalue methods, Chebyshev filtering theory, GPU implementation strategy |

## Method

CheFSI targets the interior eigenvalue problem (eigenvalues near the band gap)
which cannot be solved by simple power iteration. The approach:

1. Start with a block of random vectors (subspace of size m)
2. Apply Chebyshev polynomial filter T_k(H_scaled) to amplify components
   near the target eigenvalue window
3. Orthogonalize the subspace (Gram matrix + Cholesky/QR)
4. Compute Ritz values/vectors from the filtered subspace
5. Repeat until convergence

The Hamiltonian is scaled to [-1, 1] using spectral bounds (Gershgorin or
Lanczos estimates). The filter degree k controls selectivity vs cost.

## GPU implementation

Sparse matrix-vector multiply uses ELLPACK (ELL) format — fixed number of
nonzeros per row, coalesced memory access, no row-length variability.
This is GPU-friendly compared to CSR for uniform-bandwidth sparse matrices.

## Usage

```bash
cd topics/LinearScalingQM/CheFSI
python test_CheFSI.py
```

## Dependencies

- numpy, scipy, matplotlib
- pyopencl
