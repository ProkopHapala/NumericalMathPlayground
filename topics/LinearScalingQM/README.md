# LinearScalingQM

Linear-scaling (O(N)) methods for electronic structure calculations with
localized orbitals. All approaches avoid full O(N^3) diagonalization of the
Hamiltonian by exploiting spatial locality of atomic basis functions.

## Subfolders

| Folder | Topic |
|--------|-------|
| `CheFSI/` | Chebyshev Filtering Subspace Iteration — GPU-accelerated frontier orbital (HOMO/LUMO) computation for large sparse Hamiltonians |
| `DensityMatrix/` | Density matrix methods: Fermi Operator Expansion (FOE), Green's Function (GF) contour integration, shared Order-N utilities |
| `OMM/` | Orbital Minimization Method — direct optimization of localized molecular orbitals with finite support, CPU and GPU (OpenCL) implementations |
| `Kekule_BOP/` | Bond-Order Potentials — Chebyshev probe propagation for bond densities without diagonalization, applied to Peierls distortion and graphene |
| `LLCAO1D/` | Educational 1D LCAO solver with Gaussian basis — analytical integrals, localized orbital optimization, test scripts |
| `TestSystems/` | Test harnesses: 1D hydrogen chain and 2D hydrogen lattice with tight-binding parameters for benchmarking all Order-N methods |

## Documentation

| File | Content |
|------|---------|
| `LocallyOrthogonalOrbitals.audit.md` | Analysis of file organization and subtopic identification |

## Shared concepts

All methods work with a tight-binding Hamiltonian H and overlap matrix S
in a local atomic-like basis. The goal is to compute the density matrix,
frontier orbitals, or bond orders without diagonalizing the full H.

Key parameters across all subfolders:
- `nMaxNeigh = 16` — maximum neighbors per atom
- `nMaxSupport = 64` — maximum atoms in orbital support
- `nMaxOrbsPerAtom = 8` — maximum orbitals an atom contributes to
- `BLOCK_SIZE = 256` — OpenCL workgroup size
