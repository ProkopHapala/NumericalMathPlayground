# SpectralFiltering

Test scripts for spectral filtering and resolvent-based spectrum estimation on
nanocrystal vibration benchmarks. Depends on solver modules from
`LinearAlgebra/SpectralFiltering/` (`spectral_solvers.py`, `resolvent_solvers.py`).

## Files

| File | Role |
|------|------|
| `test_spectral_filter.py` | KPM Chebyshev spectral filtering and sub-interval eigenvector solvers. Tests on nanocrystal vibration benchmarks (adamantane, nc_C_R4–R8) |
| `test_resolvent_spectrum.py` | Resolvent iterative solvers (MINRES, COCR, block Jacobi) for spectral density estimation via `(K - omega^2 M)^{-1}` sweeps |
| `diagnose_nanocrystal_matrix.py` | Diagnostic tool: compares projected (dense) vs raw block-reconstructed (sparse) stiffness matrices from .npz benchmark files |
| `nested_solver.py.bak_*` | Backup copies of nested dissection solver with alternative GEMM kernel implementations |

## Usage

```bash
# Spectral filtering test
python test_spectral_filter.py --system nc_C_R5
python test_spectral_filter.py --system nc_C_R6 --bands 4

# Resolvent spectrum scan
python test_resolvent_spectrum.py --system nc_C_R5 --solver minres --n_freq 200
python test_resolvent_spectrum.py --system nc_C_R4 --solver cocr --eta 0.5

# Matrix diagnosis
python diagnose_nanocrystal_matrix.py --system nc_C_R5
```

## Relationship to LinearAlgebra/SpectralFiltering

This directory contains **test drivers** that import the core solver
implementations from `../LinearAlgebra/SpectralFiltering/`. Ensure that
directory is on `PYTHONPATH` or run from a context where it can be found.
