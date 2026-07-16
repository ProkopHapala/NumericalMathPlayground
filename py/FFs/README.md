# FFs/

GPU-accelerated force fields and vibrational analysis. All OpenCL modules inherit from `py/OpenCLBase.py`.

## Modules

- **UFF_cl.py** — PyOpenCL UFF runtime: bonds, angles, torsions, inversions, LJ + electrostatic non-bonded. Buffer management, kernel launch, force/energy retrieval. Also provides `make_uff_eval_fn` / `make_ff_eval_fn` — single-point `eval_fn(pos)→(E,F)` for finite-difference Hessians.
- **UFFbuilder.py** — Converts `AtomicSystem` to UFF topology arrays: atom type assignment (trivial, nitro, aromatic ring detection, amide, conjugation, cumulene), bond order assignment, parameter assignment (vdW, bonds, angles, dihedrals, inversions).
- **FFparams.py** — Force field parameter parsing: `SPFFparams` class loads `.dat` files (ElementTypes, AtomTypes, BondTypes, AngleTypes, DihedralTypes). `ElementType` records with RvdW, EvdW, Qbase, mass, color.
- **Vibrations.py** — Normal-mode analysis: Hessian via finite differences (`hessian_fd_forces`), rigid-body mode projection, mass-weighted diagonalization, frequency/mode extraction. `run_vibrations()` is the main entry point. Backends: `'uff'` (GPU finite difference), `'dftb'` (DFTB+ `SecondDerivatives`, optional).
- **VibrationPlot.py** — Top-view normal-mode figures: in-plane arrows + seismic z-circles. `make_mode_figure()`, `plot_mode_topview()`, `plot_softest_modes()`, `save_summary()`.

## Data flow

```
AtomicSystem → UFFbuilder → UFF_cl (GPU) → eval_fn → Vibrations (Hessian FD) → modes
```

1. `AtomicSystem` loads molecular geometry from `.xyz`/`.mol`/`.mol2`
2. `UFF_Builder` assigns UFF atom types, bond orders, and parameters
3. `UFF_cl` uploads topology + positions to GPU, evaluates forces/energies
4. `make_uff_eval_fn` wraps this as `eval_fn(pos) → (E, F)`
5. `hessian_fd_forces` computes Hessian via central finite differences
6. `run_vibrations` projects rigid-body modes, diagonalizes, returns frequencies + eigenvectors

## Kernels

UFF uses three OpenCL files concatenated at build time (see `kernels/README.md`):

| File | Role |
|------|------|
| `common.cl` | Shared types, constants, mixing rules, `clampForce` |
| `Forces.cl` | Inline pairwise potentials: `getLJQH`, `getMorseQH`, `getCoulomb` |
| `UFF.cl` | UFF bonding kernels: bonds, angles, dihedrals, inversions, force assembly |

## Usage

```python
from py.AtomicSystem import AtomicSystem
from py.FFs.UFF_cl import make_uff_eval_fn
from py.FFs.Vibrations import run_vibrations

mol = AtomicSystem(fname='data/xyz/PTCDA.xyz')
result = run_vibrations(mol, backend='uff', delta=1e-4, do_nonbond=False)
print(result.format_table(unit='cm-1'))
```

**Output:** All generated artifacts (`.xyz`, `.svg`, summary) go to `debug/` (see AGENTS.md Rule 7).
