# SpectralFiltering

Iterative methods for computing interior eigenvalues and spectral density of
large sparse symmetric matrices. Focuses on Chebyshev polynomial filtering
and resolvent (Green's function) probing for vibration spectra.

## Files

### Code

| File | Role |
|------|------|
| `spectral_solvers.py` | Core solver library: Chebyshev polynomial filtering, Jackson kernel smoothing, sparse matrix loading, OpenCL CSR SpMM, benchmark data loader |
| `resolvent_solvers.py` | Iterative solvers for the resolvent equation A(omega) x = b where A = K - (omega + i*eta)^2 M. CPU (scipy sparse) and GPU (OpenCL) backends. Supports batched [N, k] RHS for interior spectrum scanning |
| `spectral_plotting.py` | Plotting utilities: spectral density histograms, eigenvalue comparisons, convergence plots |
| `spectral_demos.py` | Demo script with built-in toy matrices: showcases Chebyshev-filtered subspace iteration, resolvent probing, and spectral density estimation |
| `spectral_demos-external.py` | Demo script that loads external vibration benchmark matrices and runs the same solvers with visualization |

### Documentation (`doc/`)

| File | Content |
|------|---------|
| `SpectralFiltering_discussion.md` | Design discussion: Chebyshev filtering vs Lanczos vs resolvent methods |
| `SpectralFiltering_doc_tutorial.md` | Tutorial on spectral filtering theory and implementation |
| `RandomizedLinarAlgebra.md` | Notes on randomized SVD / trace estimation |
| `Nanocrystal_vibration_bench.progress.md` | Progress log for nanocrystal vibration benchmark setup |
| `Stabilize_Jacobi_linsolve_vibration_spectra.chat.md` | Chat log: stabilizing Jacobi solver for vibration spectra |
| `Stabilize_Jacobi_linsolve_vibration_spectra.progress.md` | Progress log for Jacobi solver stabilization |

## Methods

### Chebyshev-filtered subspace iteration
Scale H to [-1, 1] via Chebyshev bounds, apply polynomial filter T_k(H) to
amplify eigenvalues near a target window, then orthogonalize the subspace.
Repeated iterations converge to the eigenvalues in the filter window.

### Resolvent probing
Solve (K - (omega + i*eta)^2 M) x = b for multiple frequencies omega and
random probe vectors b. The trace of the resolvent gives the spectral
density (density of states). Stochastic trace estimation with Rademacher
or Gaussian probes avoids full diagonalization.

### Iterative linear solvers
Jacobi, Gauss-Seidel, CG, and Chebyshev-accelerated variants for the
resolvent linear system. GPU versions use OpenCL with CSR sparse format.

## Usage

```bash
cd topics/LinearAlgebra/SpectralFiltering

# Built-in toy matrix demos
python spectral_demos.py

# External benchmark demos
python spectral_demos-external.py --system nc_C_R5
```

## Dependencies

- numpy, scipy, matplotlib
- pyopencl (optional, for GPU kernels)
