# LinarElasticity

Vibration spectra of spring-mass truss systems — a mechanical analogy for
quantum lattice problems. Used as a testbed for iterative solvers and
spectral filtering methods developed in `LinearAlgebra/`.

## Files

| File | Role |
|------|------|
| `Truss.py` | Utilities for building spring-mass truss systems: triangular grid generation, stiffness matrix assembly, edge extraction |
| `VibrationProbing.py` | Mechanical Green-function probing for vibration spectra. Solves A(omega) x = b where A = K - (omega + i*eta)^2 M. Supports iterative solvers (Jacobi, CG, Chebyshev-accelerated) and GPU (OpenCL) backends |
| `elasticity_benchmark.py` | Benchmark script: generates hexagonal grid, builds sparse stiffness matrix, visualizes structure and spectrum |

## Problem

Nodes are mass points connected by axial springs. The stiffness matrix K
is assembled from spring constants, mass matrix M is diagonal. After removing
fixed DOFs (Dirichlet boundary), we study the frequency response to external
forces via the dynamic stiffness operator A(omega) = K - (omega + i*eta)^2 M.

## Usage

```bash
python elasticity_benchmark.py
python VibrationProbing.py
```
