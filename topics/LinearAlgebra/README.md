# LinearAlgebra

Iterative and direct solvers for large sparse symmetric eigenvalue problems
and linear systems. Developed for vibration spectra of nanocrystals but
applicable to any sparse symmetric matrix.

## Subfolders

| Folder | Topic |
|--------|-------|
| `FastDirectSolvers/` | Nested dissection, block diagonalization, Ritz correction, AMLS — direct divide-and-conquer eigensolvers with GPU (OpenCL) kernels |
| `SpectralFiltering/` | Chebyshev-filtered subspace iteration, resolvent probing, spectral density estimation — iterative methods for interior spectrum |

## Shared concepts

Both subfolders work with the same benchmark data: mass-weighted Hamiltonians
H = M^{-1/2} K M^{-1/2} from nanocrystal MD simulations (adamantane, carbon
nanocrystals R4-R8). The `SpectralFiltering/spectral_solvers.py` module
provides `load_vibration_benchmark()` which is imported by `FastDirectSolvers/`.
