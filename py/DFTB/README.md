# DFTB/

DFTB+ integration: ctypes C-API wrapper, basis set parsing, GPU density projection, and basis optimization.

- **DFTBcore.py** — ctypes interface to DFTB+ C-API: `init()`, `run_scf()`, `get_dm_dense()` (density matrix), `get_eigvecs_dense()` (eigenvectors/eigenvalues), `enable_matrix_collection()`.
- **DFTB_utils.py** — DFTB+ I/O utilities: `run_dftb_subprocess()`, `parse_dftb_output()`, `make_dftb_input()`, `write_dftb_input_hessian()`, `read_hessian()`, SK parameter path management. Used by `FFs/Vibrations.py` for the `'dftb'` backend.
- **DFTBplusParser.py** — Parse DFTB+ HSD input files (`wfc.*.hsd`): Slater-type orbital parameters (exponents, contraction coefficients, angular momenta). Also parses `band.out` and `eigenvec.bin`.
- **Grid_dftb.py** — GPU projection of DFTB wavefunctions and density matrices onto 3D real-space grids (OpenCL, sparse and dense modes). `GridProjector` class, `load_basis_sto()`, `project_density()`, `project_orbitals()`.
- **basis_optimizer.py** — Fit single-exponential Slater-tail basis (N, ζ) to reference density via simulated annealing on z-profile points. Default decay constants from GPAW/PySCF fits.

**Note:** DFTB imports are optional in `Vibrations.py` — the `'uff'` backend works without DFTB+ installed.
