# DensityMatrix

Linear-scaling density matrix computation methods — alternatives to full
diagonalization for obtaining the density matrix P of large sparse systems.

## Files

| File | Role |
|------|------|
| `OrderN.py` | Shared utilities: tight-binding parameter generation, neighbor pair finding, sparse matrix assembly, generalized eigenvalue solver, spectral bounds estimation, linear solvers (Jacobi, Cholesky), plotting helpers |
| `FOE.py` | Fermi Operator Expansion: approximates the Fermi-Dirac function via Chebyshev polynomials, uses stochastic trace estimation (Rademacher probes) to compute the density matrix diagonal without full diagonalization |
| `GF.py` | Green's Function methods: contour integration in the complex plane with deterministic probing (comb-based) and stochastic random-probe variants. Solves linear systems at each contour node |
| `DensityMatrix_Idenpotency_NearestNeighbor.chat.md` | Discussion: enforcing Pauli exclusion (idempotency PSP=P) in density matrix methods with local atomic basis |
| `OrderN_Electronic_Structure_Solver_Gemini.chat.md` | Design discussion for the Order-N electronic structure solver |

## Methods

### Fermi Operator Expansion (FOE)
Approximates the Fermi-Dirac distribution f(H) as a Chebyshev polynomial
expansion. The density matrix P = f(H) is applied to random probe vectors,
and the diagonal is estimated via stochastic trace estimation. This avoids
storing the full N x N density matrix — only matrix-vector products are needed.

### Green's Function (GF)
Computes the density matrix via the resolvent G(z) = (zI - H)^{-1} evaluated
at contour nodes in the complex plane. The Fermi function is recovered by
contour integration. Two variants:
- Deterministic: comb-based probing vectors (structured, reproducible)
- Stochastic: random probe vectors with stochastic trace estimation

Both require solving linear systems (zI - H) x = b at each contour node,
using iterative solvers from `OrderN.py`.

## Usage

These modules are imported by `TestSystems/hydrogen_chain_1d.py` which
serves as the main test harness. See `TestSystems/` for usage examples.

## Dependencies

- numpy, scipy, matplotlib
- `OrderN.py` provides shared utilities imported by `FOE.py` and `GF.py`
