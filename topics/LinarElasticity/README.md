# LinarElasticity

Vibration spectra of spring-mass truss systems — a mechanical analogy for
quantum lattice problems. Used as a testbed for iterative solvers and
spectral filtering methods developed in `LinearAlgebra/`.

## Architecture (three-layer split)

| Layer | File | Role |
|-------|------|------|
| **Core** | `Truss.py` | Geometry & mesh: triangular grid generation, stiffness assembly, edge extraction, boundary nodes, graph utilities |
| **Core** | `TrussSolver.py` | Solvers: global Jacobi, block Jacobi (overlapping + alternating patches), direct local LU, heavy-ball momentum, matrix-free axial spring operators, dynamic stiffness, Cholesky factorization |
| **Plotting** | `TrussPlotting.py` | All visualization: truss meshes, convergence curves, beta sweeps, beam snapshots, deformation + patch overlays, per-node error, 1D partitioning, spectra, mode shapes, hex grids |
| **Script** | `BlockJacobiTruss.py` | CLI benchmark: beam (`--mode beam`) and grid (`--mode grid`) tests comparing global Jacobi, block Jacobi, alternating patches, direct solves |
| **Script** | `VibrationProbing.py` | CLI: mechanical Green-function probing for vibration spectra. Solves A(ω)x = b where A = K − (ω+iη)²M |
| **Script** | `elasticity_benchmark.py` | CLI: hexagonal grid generation, sparse stiffness matrix, structure visualization |

Scripts are thin wrappers — all algorithms live in `TrussSolver.py`, all plotting in `TrussPlotting.py`. See `AGNETs.md` Rule 3.

## Problem

Nodes are mass points connected by axial springs. The stiffness matrix K
is assembled from spring constants, mass matrix M is diagonal. After removing
fixed DOFs (Dirichlet boundary), we study the frequency response to external
forces via the dynamic stiffness operator A(ω) = K − (ω + iη)²M.

The static problem A = M/dt² + K arises in implicit time integration; the
same operators are reused for vibration analysis with A(ω) = K − ω²M.

## Usage

```bash
# Block Jacobi benchmarks
python BlockJacobiTruss.py --mode beam --no-show
python BlockJacobiTruss.py --mode grid --no-show

# Vibration spectra
python VibrationProbing.py --nx 3 --ny 3 --nfreq 50

# Hex grid stiffness visualization
python elasticity_benchmark.py --nx 8 --ny 8 --no-show

# Run Truss.py self-tests
python Truss.py
```

## Notes

- `BlockJacobiGPU.md` — full derivation of block Jacobi with overlapping patches, stiffness-weighted averaging, and heavy-ball momentum.
- `low_rank_perturbation.md` — notes on low-rank perturbation theory for spectral filtering.
- `Ascii2truss.md` — ASCII-art truss specification format.
