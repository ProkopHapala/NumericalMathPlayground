# NumericalMathPlayground

Testing and deriving different things from numerical math mostly for computational chemistry, quantum mechanics and forcefield development.

## Contents

### topics/

| Topic | Description |
|-------|-------------|
| **ChemicalGraphs/** | Python toolkit for molecular graph construction, heterocycle generation, and Kekule pi-bond order optimization. Includes ASCII art heterocycle parser, object-graph AtomicGraph, file I/O (.mol/.mol2/.xyz), and honeycomb lattice geometry |
| **LinarElasticity/** | Vibration spectra of spring-mass truss systems — mechanical analogy for quantum lattice problems. Stiffness matrix assembly, Green-function probing, iterative solvers (Jacobi, CG, Chebyshev), GPU (OpenCL) backends |
| **LinearAlgebra/** | Iterative and direct solvers for large sparse symmetric eigenvalue problems. Subfolders: `FastDirectSolvers/` (nested dissection, AMLS, GPU GEMM), `SpectralFiltering/` (Chebyshev-filtered subspace iteration, resolvent probing, spectral density) |
| **LinearScalingQM/** | O(N) electronic structure methods with localized orbitals. Subfolders: `CheFSI/` (Chebyshev-filtered subspace iteration for frontier orbitals), `DensityMatrix/` (FOE, Green's function contour integration), `OMM/` (orbital minimization, CPU+GPU), `Kekule_BOP/` (bond-order potentials via Chebyshev probe), `KekuleQM/` (OpenCL valence-bond/QEq/SSH pi-electron model), `LLCAO1D/` (educational 1D LCAO solver), `TestSystems/` (1D/2D hydrogen test harnesses) |
| **RadialFunctions/** | C² continuous radial functions optimized for GPU acceleration. Includes polynomial families, Lorentzian alternatives, and force derivative optimizations. (See also `UsefulFunctions/RadialFunctions/`) |
| **SpectralFiltering/** | Test drivers for KPM Chebyshev spectral filtering and resolvent-based spectrum estimation on nanocrystal vibration benchmarks. Depends on solver modules from `LinearAlgebra/SpectralFiltering/` |
| **UsefulFunctions/** | Reusable utility functions for atomistic simulations. Subfolders: `AngularFunctions/` (sp1/sp2/sp3 hybridization potentials via complex numbers and quaternions), `RadialFunctions/` (compact-support radial basis functions with C2-continuous cutoff, interactive `plot_radial.py` slider tool) |

### web/
Interactive WebGL-based visualization tools for exploring angular functions.

- **angular-plotter/** - 2D angular function plotter. Visualizes functions on planes with optional radial envelopes, multiple colormaps, and preset angular functions (sp1, sp2, sp3, octahedral, etc.).
- **angular-plotter-3d/** - Advanced 3D visualizer with multiple render modes:
  - 2D slice through 3D space
  - Volume min/max projection
  - Isosurface rendering
  - Filament visualization (zero-crossing lines)
  - Chiral filament (colors tetrahedral vs anti-tetrahedral)
  - Zero planes (real/imaginary surfaces)
  - Quaternion trackball camera control
