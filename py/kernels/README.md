# kernels/

OpenCL source files for GPU-accelerated force field calculations. Python harnesses concatenate `.cl` snippets in order via `OpenCLBase.load_program_multi` — there is no `#include`.

## File index

| File | Role | Python consumer |
|------|------|-----------------|
| `common.cl` | Shared types, constants, math helpers, mixing rules | (always first) |
| `Forces.cl` | Inline pairwise potentials (LJ, Morse, Coulomb) | all force modules |
| `UFF.cl` | UFF bonding: bonds, angles, dihedrals, inversions, force assembly | `FFs/UFF_cl.py` |
| `rigid.cl` | 6-DOF rigid body dynamics with quaternion integration | (not currently used) |
| `BRBFFF.cl` | Two-frame molecular skinning and atom-force-to-frame-wrench projection | `FFs/BRBFFF.py` |

## Composition rules

Concatenation order matters. Current stack:

| Use case | Files (in order) |
|----------|------------------|
| UFF force eval | `common` + `Forces` + `UFF` |

---

## common.cl

Shared foundation — concatenated **first**. No `__kernel` functions; only types, constants, and inline helpers.

- **Types:** `cl_Mat3` (3×3 row-major matrix)
- **Constants:** `COULOMB_CONST` (14.3996 eV·Å/e²), `R2SAFE` (minimum r²), `EXCL_MAX` (max exclusions per atom = 16)
- **Mixing rules:** `mixREQ_arithmetic` — Lorentz-Berthelot: R_ij = R_i + R_j, E_ij = E_i·E_j, Q_ij = Q_i·Q_j
- **Math:** `modulo` (PBC wrap), `udiv_cmplx` (complex division), `rotMat`/`rotMatT` (rotation matrix), `clampForce` (force capping for stability)

---

## Forces.cl

Inline pairwise potential functions (not standalone kernels). All return `float4 (Fx, Fy, Fz, E)`. Called from UFF non-bonded and angle/dihedral exclusion terms.

| Function | Potential |
|----------|-----------|
| `getLJQH` | Lennard-Jones 12-6 + damped Coulomb + H-bond: V = E₀[(R₀/r)¹²−2(R₀/r)⁶] + Q/√(r²+R²damp) |
| `getMorseQH` | Morse + damped Coulomb: V = E₀[e^{2α(r−r₀)}−2e^{α(r−r₀)}] + Q/√(r²+R²damp) |
| `getMorsePLQH` | Factorized Morse (Pauli/London/Coulomb/H-bond channels) |
| `getCoulomb` | Bare damped Coulomb: V = Q/√(r²+R²damp) |

---

## UFF.cl

Universal Force Field (Rappé et al.): harmonic bonds, cosine angle bending, torsional dihedrals, inversion (improper) terms. Uses eval-then-scatter pattern — compute per-interaction forces into `fint`, then scatter to `fapos` via `assembleForces_UFF`.

| Kernel | Role |
|--------|------|
| `evalBondsAndHNeigh_UFF` | Harmonic bonds (E=k(r−r₀)²) + H-neighbor direction vectors |
| `evalAngles_UFF` | Cosine angle bending with UFF small-angle harmonic variant |
| `evalDihedrals_UFF` | Torsional: E=V_n[1+cos(nφ−φ₀)] with n=1,2,3,6 |
| `evalInversions_UFF` | Inversion (improper): Wilson–Morse form |
| `assembleForces_UFF` | Scatter per-interaction `fint` → per-atom `fapos` (Newton's 3rd law) |
| `clear_fapos_UFF`, `clear_fint_UFF` | Zero force buffers |
| `updateAtomsSPFFf4` | Simplified velocity-Verlet integrator |

**Caveat:** Force scattering uses `atomic_add` — race-free but non-deterministic accumulation order.

---

## rigid.cl

6-DOF rigid-body MD: 3 translational + 3 rotational (quaternion) DOFs per body. Quaternion integration uses exact exponential map with Taylor-series `sinc`/`cosc` for small-angle stability. Not currently wired into any Python module in this repo.

---

## BRBFFF.cl

Three standalone kernels operate on a batch of molecules with one shared atom template and two weighted quaternion frames.  `brbfff_reconstruct_positions` writes the skinned atom geometry.  A compatible external potential writes atom forces into the persistent `atom_force` buffer, then `brbfff_project_atomic_forces` uses one workgroup per system and local-memory reductions to output frame-origin forces and torques.  `brbfff_relax_step` applies a capped, overdamped update directly in those pose coordinates and renormalizes every quaternion.  It uses no atomics.  This is relaxation, not a claim of physically accurate inertial MD: masses are unnecessary for a relaxed structure, while energy backtracking still requires an external energy evaluator.  It also has no internal molecular potential yet; a standalone distorted molecule therefore has no GPU restoring force until that term is ported.
