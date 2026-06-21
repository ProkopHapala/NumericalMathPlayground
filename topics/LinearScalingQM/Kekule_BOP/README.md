# Kekule_BOP

Bond-Order Potentials (BOP) — computing bond densities (pi-electron bond
orders) without full diagonalization, using Chebyshev probe propagation.
Applied to Peierls distortion in polyacetylene and graphene flakes.

## Files

| File | Role |
|------|------|
| `KekuleOrderN_Gemini1.py` | Peierls distortion in polyacetylene (SSH model). Compares exact O(N^3) diagonalization, O(N) Chebyshev FOE, and O(N) Lanczos recursion for pi-electron energy vs dimerization amplitude |
| `KekuleOrderN_Gemini2.py` | Polyacetylene with boundaries and a defect. Compares exact diagonalization, Chebyshev FOE, and Lanczos recursion for bond density estimation in a finite chain with a localized defect |
| `KekuleOrderN_Gemini_BOP.py` | Bond-Order Potentials demo on a 1D carbon chain. Propagates a local Chebyshev probe to compute bond density rho_ij without full diagonalization. Visualizes wave spreading, path contributions, and convergence |
| `KekuleOrderN_Gemini_BOP_2D.py` | 2D Bond-Order Potentials demo on a hexagonal graphene flake. Uses Chebyshev probe propagation to compute bond densities without diagonalization, visualizing wave spreading and bond order convergence in a 2D pi-system |
| `KekuleOrderN_Gemini_BOP_2D_v2.py` | Extended 2D BOP demo on hexagonal graphene flakes. Improved version with enhanced visualization, multiple flake sizes, and detailed convergence analysis |
| `KekuleOrderN.chat.md` | Design discussion: Harris functional, Peierls distortion, bond-order potentials, and whether Kekule structures can be reproduced without diagonalization |
| `KekuleOrderN_1D_comparison.png` | Output plot: 1D comparison of methods |
| `KekuleOrderN_BOP_2D_comparison.png` | Output plot: 2D BOP comparison |

## Methods

### Bond-Order Potentials (BOP)
Instead of computing all eigenstates, BOP propagates a local Chebyshev
probe from a single bond (i,j) to estimate the bond density rho_ij.
The Chebyshev expansion of the Fermi function applied locally gives
the bond order with O(k) matrix-vector products where k is the polynomial
degree — independent of system size N.

### Lanczos recursion
An alternative local method: builds a tridiagonal Krylov subspace from
a local starting vector, then computes the local density of states from
the continued-fraction representation of the Green's function.

### Peierls distortion (SSH model)
The Su-Schrieffer-Heeger model: electron-phonon coupling in a 1D carbon
chain where pi-electrons dimerize the lattice. The bond density rho_ij
determines the equilibrium bond lengths. Used as a test case because
the Kekule structure (alternating bond orders) is visually clear.
