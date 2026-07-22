# NumericalMathPlayground

Testing and deriving different things from numerical math mostly for computational chemistry, quantum mechanics and forcefield development.

## Contents

### topics/

| Topic | Description |
|-------|-------------|
| **ChemicalGraphs/** | Python toolkit for molecular graph construction, heterocycle generation, and Kekule pi-bond order optimization. Includes ASCII art heterocycle parser, object-graph AtomicGraph, file I/O (.mol/.mol2/.xyz), and honeycomb lattice geometry |
| **Clustering/** | Clustering and nearest-neighbor search algorithms in arbitrary dimensions. K-means (Lloyd's), k-d tree NN, iterative refinement — data-adaptive tree structures for high-dimensional and multi-scale distributions |
| **IsingMC/** | GPU-accelerated (OpenCL) exact and Monte Carlo solvers for small Ising clusters on square lattices. Brute-force enumeration up to 16 sites, batch parameter scans (W1/W2 phase diagrams), MQCA logic gate identification (NAND, NOR, etc.), degeneracy analysis, and reusable plotting/cluster-definition modules (`IsingPlotting.py`, `Ising_utils.py`) |
| **IsingQUBO/** | Ising/QUBO solver hierarchy for molecular charge systems (100–1000 sites). Spin mapping of binary charge occupancy, greedy local descent, mean-field annealing, GPU-parallel batch evaluation across tip positions. Includes Coulomb matrix construction and benchmarking on degenerate systems |
| **LinarElasticity/** | Vibration spectra of spring-mass truss systems — mechanical analogy for quantum lattice problems. Stiffness matrix assembly, Green-function probing, iterative solvers (Jacobi, CG, Chebyshev), GPU (OpenCL) backends |
| **LinearAlgebra/** | Iterative and direct solvers for large sparse symmetric eigenvalue problems |
| **LinearAlgebra/FastDirectSolvers/** | Approximate eigensolvers for large sparse matrices (stiffness K, mass M) from nanocrystal vibrations. Nested dissection block splitting, AMLS, GPU GEMM — compute low-frequency spectrum without full O(N³) diagonalization |
| **LinearAlgebra/SpectralFiltering/** | Iterative interior eigenvalue and spectral density methods. Chebyshev polynomial filtering, Jackson kernel smoothing, resolvent (Green's function) probing with CPU (scipy sparse) and GPU (OpenCL CSR) backends |
| **LinearScalingQM/** | O(N) electronic structure methods with localized orbitals |
| **LinearScalingQM/CheFSI/** | Chebyshev-filtered subspace iteration for frontier orbitals (HOMO/LUMO). GPU (PyOpenCL) implementation with ELLPACK sparse format, polynomial filtering, and subspace orthogonalization |
| **LinearScalingQM/DensityMatrix/** | Linear-scaling density matrix computation — Fermi Operator Expansion (FOE), Green's function contour integration, and related methods as alternatives to full diagonalization |
| **LinearScalingQM/Kekule_BOP/** | Bond-order potentials via Chebyshev probe propagation — compute pi-electron bond orders without diagonalization. Applied to Peierls distortion in polyacetylene and graphene flakes |
| **LinearScalingQM/KekuleQM/** | OpenCL valence-bond/QEq/SSH pi-electron model — classical bond-order field solved by GPU gradient descent, linear-scaling (optional O(N²) Coulomb). 3-kernel schedule: local bonding gather, Coulomb direct, DOF update |
| **LinearScalingQM/LLCAO1D/** | Educational 1D quantum solver using LCAO with Gaussian basis functions. Analytical integrals, standard diagonalization, and iterative localized orbital optimization |
| **LinearScalingQM/OMM/** | Orbital Minimization Method — direct optimization of localized molecular orbitals with finite support. Unconstrained energy functional, projective dynamics for orthogonality. CPU and GPU (OpenCL) implementations |
| **LinearScalingQM/TestSystems/** | Benchmark harnesses for Order-N methods vs exact O(N³) diagonalization. 1D hydrogen chain and 2D hydrogen test systems with configurable tight-binding parameters |
| **MultiGridFF/** | Blended rigid-body frame stiffness experiments: eta-coordinate reduction, UFF Hessian fitting, relaxed/static-condensed comparisons, and vibrational validation |
| **NonBondingFFs/** | Non-bonded force field development: directional H-bond and sigma-hole interactions, 2D energy maps (ff_map.py), angular sampling on molecular distance fields (complex phase method), pairwise rigid-body MD with electron pairs (demo_pairff.py + Vispy GUI), compact-support polynomial potentials |
| **QuadratureGrids/** | Numerical integration grids for quantum chemistry — 2D prototypes for CubeEmbededAtoms, triangular/wedge grids, and PySCF DFT grid visualization. Three-layer architecture: core math modules, plotting modules, and thin demo scripts |
| **RadialFunctions/** | C² continuous radial functions optimized for GPU acceleration. Includes polynomial families, Lorentzian alternatives, and force derivative optimizations. (See also `UsefulFunctions/RadialFunctions/`) |
| **SpectralFiltering/** | Test drivers for KPM Chebyshev spectral filtering and resolvent-based spectrum estimation on nanocrystal vibration benchmarks. Depends on solver modules from `LinearAlgebra/SpectralFiltering/` |
| **UsefulFunctions/** | Reusable utility functions for atomistic simulations |
| **UsefulFunctions/AngularFunctions/** | Angular potential functions for sp1, sp2, sp3 geometries using complex numbers (2D) and quaternions (3D) instead of stored port vectors |
| **UsefulFunctions/RadialFunctions/** | Compact-support radial basis functions — smooth, C2-continuous at cutoff, with tunable peak width. Interactive `plot_radial.py` explorer |

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
