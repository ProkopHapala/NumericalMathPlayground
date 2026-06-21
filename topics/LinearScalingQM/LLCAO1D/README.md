# LLCAO1D

Educational 1D quantum solver using Linear Combination of Atomic Orbitals
(LCAO) with Gaussian basis functions. Provides analytical integrals and
both standard diagonalization and iterative localized orbital optimization.

## Files

| File | Role |
|------|------|
| `quantum_solver_1D.py` | Main solver: analytical integrals (overlap, kinetic, Coulomb) for Gaussian basis, standard diagonalization, and iterative localized orbital optimization |
| `GaussPoly_sympy.py` | SymPy symbolic derivation of overlap integrals between Gaussian and polynomial basis functions for 1D quantum mechanics |
| `run_simulation.py` | Demo script: sets up 1D atom chain with Gaussian basis, runs full diagonalization, plots molecular orbitals |
| `run_localized.py` | Demo script: sets up 1D atom chain with Gaussian basis, runs iterative localized orbital optimization, plots localized molecular orbitals |
| `test_GaussIntegrals1D_plot.py` | Validation of 1D Gaussian basis integrals: compares analytical formulas (overlap, kinetic energy, Coulomb) against numerical grid integration |
| `test_PolyIntegrals1D_plot.py` | Validation of 1D polynomial basis integrals: compares analytical formulas against numerical grid integration |
| `Localized_Solver_1D.md` | Documentation: finding localized molecular orbitals via energy minimization and constraints |
| `OrbitalOrthogonalization.md` | Documentation: efficient computation of orthogonalization gradients for localized orbitals |

## Purpose

This is an educational/testbed framework for understanding how localized
orbital methods work in a simple 1D setting. The Gaussian basis allows
analytical evaluation of all integrals, making it easy to verify numerical
methods. The localized orbital optimizer demonstrates the core ideas later
used in the OMM (Orbital Minimization Method) subfolder.
