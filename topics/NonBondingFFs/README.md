# NonBondingFFs — Non-Bonded Force Fields and Sampling

## Overview

This directory contains tools for developing, testing, and visualizing non-bonded
interaction force fields between molecules. The focus is on **directional
interactions** (hydrogen bonds, sigma holes) that go beyond simple isotropic
Morse/Lennard-Jones potentials.

## Components

### Force Field Visualization

- **ff_map.py** — 2D interaction energy maps around molecules. Scans a probe
  atom over a grid in the molecular plane and visualizes Morse + Coulomb +
  Hbond contributions. Supports electron pairs (lone pairs) and multiple probe
  types. Produces publication-quality colormaps with vdW circles and force arrows.

- **fit_radial.py** — Fits compact radial potential basis functions to Morse
  reference data for atom-atom pairs (O-O, O-H, H-H). Supports both numerical
  (Boltzmann-weighted least squares) and analytical coefficient derivation.

  **Basis functions** (selectable via `--basis`):
  - `morse2c`: `fc²·(r-r_node)·(a·fc² + c)` — two-term compact Morse
  - `morse2d`: `fc·(r-r_node)·(a·fc³ + c)` — two-term compact Morse
  - `morse1b`: `fc⁴·a·(r-r_node)` — single-term compact Morse
  - Plus earlier exploratory bases: `lorenz`, `fc22`, `fc22r`–`fc22r5`, `morse1`, `morse1c`, `morse1cmp`, `morse1pw`, `morse2`, `morse2b`

  **Cutoff functions** (selectable via `--morse-cutoff`):
  - `smoothstep`: `fc(r) = 3x² - 2x³`, where `x = 1 - r/rc`
  - `fc22`: `fc(r) = (1 - (r/rc)²)²`

  **Analytical coefficient modes** (selectable via `--analytical-mode`):
  - `minimum`: For fixed `rc`, solves 2 equations (exact depth `V(R0)=-E0` and
    exact equilibrium `V'(R0)=0`) to get analytical `a, c`. May produce `c>0`,
    causing a small repulsive bump near cutoff.
  - `node-slope`: Matches Morse slope at `r_node` and depth at `R0`. Generally
    shifts the minimum away from `R0`.
  - `pure-tail` *(best)*: Sets `c=0` and solves for a **pair-specific cutoff**
    `rc` that simultaneously satisfies exact depth, exact equilibrium, and
    purely attractive tail (no repulsive bump). For `fc22`:
    `rc² = R0² + 16·Δ·R0` where `Δ = ln(2)/β`. Only one energy coefficient `a`
    needed, and `rc²` can be stored directly (no sqrt at runtime).

  **Mixing rules** (analytical modes):
  - `R0 = Ri + Rj` (additive)
  - `E0 = sqrt(Eii·Ejj) = ei·ej` (geometric, where `ei = sqrt(Eii)`)
  - `β` = global constant (`MORSE_BETA = 1.7`)
  - `r_node = R0 - ln(2)/β` (exact Morse zero crossing, `Δ = ln(2)/β` is a
    global constant ≈ 0.405 Å, independent of atom pair)

  Usage:
  ```
  python3 fit_radial.py --basis morse2d --rc 6                          # numerical fit only
  python3 fit_radial.py --basis morse2d --rc 6 --analytical             # + analytical overlay (minimum mode)
  python3 fit_radial.py --basis morse2d --morse-cutoff fc22 --analytical --analytical-mode pure-tail
  python3 fit_radial.py --basis morse2c --morse-cutoff fc22 --analytical --analytical-mode minimum
  ```

### Angular Sampling

- **angular_sampling.py** — Samples points uniformly along isolines of a
  molecular distance field (r_mol) using complex harmonic phase continuation.
  Produces 2×3 comparison plots of logsumexp vs power-mean formulas, plus a
  full sampling grid with gradient arrows. See the file header docstring for
  a full tutorial on the mathematical background.

- **hbond_sampling.py** — H-bond sampling on molecular distance fields with
  isoline visualization and nearest-atom distance backgrounds.

- **AngularPhaseSampling.md** — Extensive mathematical derivation of the
  complex phase sampling method, including topological invariants, winding
  numbers, and the impossibility result for globally orthogonal coordinates.

### Rigid Body Pair FF

- **demo_pairff.py** — Demo script for pairwise rigid-body molecular dynamics.
  Loads uracil (static) and HCOOH (dynamic), sets up Morse + Coulomb + Hbond
  interactions with electron pairs and sigma holes, and launches interactive
  Vispy visualization or headless FIRE relaxation.

  Usage:
  ```
  python3 demo_pairff.py                    # interactive Vispy
  python3 demo_pairff.py --no-vis           # headless relaxation
  python3 demo_pairff.py --no-vis --steps 500
  python3 demo_pairff.py --he -1.0 --hs 1.0 --rc 3.0 --w 1.0  # custom params
  python3 demo_pairff.py --epair-dist 1.4 --sigma-dist 1.0    # dummy atom distances
  ```

### Force Field Model

The pairwise FF (implemented in `kernels/rigid.cl`, `rigid_body_pairff_kernel`)
uses three interaction types:

| Interaction | Formula | Participants |
|-------------|---------|--------------|
| Morse + Coulomb | `E0*(e²-2e) + kQ/√(r²+ε)` | atom ↔ atom |
| Hbond (Lorentzian) | `min(0,Q·He)·fcut(r/rc)·1/(w²+r²)` | atom ↔ epair |
| Sigma-hole | `min(0,Q·Hs)·fcut(r/rc)·1/(w²+r²)` | atom ↔ sigma-hole |

Where:
- **Electron pairs (E)**: lone-pair dummy atoms on O/N at `epair_dist` (default 1.4 Å) from host
- **Sigma holes (Sh)**: electropositive caps on H bonded to O/N at `sigma_dist` (default 1.0 Å)
- **Pseudo-charges**: He (epair, negative) and Hs (sigma, positive) stored in REQ.z,
  enabling branch-free GPU evaluation via `coeff = min(0, Qi*Qj)`
- **Antisymmetry**: He<0 attracts positive probes (H donors), Hs>0 attracts negative
  probes (O/N acceptors). The `min(0,...)` clips to attractive-only.

### GPU Kernel Design (branch-free)

**Data layout**: Both dynamic and static atom arrays are sorted:
`[real_atoms, epairs, sigma_holes]`. Counts `n_static_atoms`/`n_dyn_atoms` refer
only to real atoms (type=0). This lets each thread determine its role by index
comparison (`atom_idx < n_dyn_atoms`) rather than per-pair type branching,
avoiding warp divergence.

**Pseudo-charge in REQ.z**: Storing He/Hs in REQ.z for dummy atoms means the
kernel uses the same `coeff = min(0, Qi*Qj)` formula for all interactions.

**Interaction loops**:
```
if atom_idx < n_dyn_atoms:        // thread is a real atom
    loop j = 0..n_static_atoms:   //   Morse+Coulomb with static real atoms
    loop j = n_static_atoms..n_static:  //   Lorentzian with static epairs+sigma
else:                             // thread is an epair/sigma
    loop j = 0..n_static_atoms:   //   Lorentzian with static real atoms
```

epair-epair, sigma-sigma, and epair-sigma interactions are implicitly skipped.

**Z-harmonic constraint**: Applied per-atom (not per-CoM), producing both force
and torque to keep the molecule planar.

### GUI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `He` | -1.0 | Epair pseudo-charge (negative). Attracts positive probes (H). |
| `Hs` | +1.0 | Sigma-hole pseudo-charge (positive). Attracts negative probes (O). 0=disabled. |
| `rc` | 3.0 | Hbond/sigma cutoff radius [Å]. fcut→0 at r=rc. |
| `w` | 1.0 | Lorentzian width [Å]. Larger = broader attractive well. |
| `k_z` | 5.0 | Z-harmonic constraint strength per atom. |
| `alpha` | 1.8 | Morse alpha [1/Å]. Controls Pauli repulsion steepness. |
| `z_target` | 0.0 | Target z for harmonic constraint. |
| `L_epair` | 1.4 | Epair distance from host atom [Å]. (Requires rebuild) |
| `L_sigma` | 1.0 | Sigma hole distance from H [Å]. 0=disabled. (Requires rebuild) |
| `dt` | 0.02 | Time step [ps]. |
| `steps/frame` | 10 | Kernel iterations per rendered frame. |

**Note**: Changing `L_epair` or `L_sigma` in the GUI stores the values but does
not rebuild molecules at runtime (requires re-running `from_two_molecules`).
All other parameters update live.

### Visualization

- **Potential map**: Morse + epair Hbond + sigma-hole (no Coulomb), matching
  ff_map.py's `morseH_only` mode. Computed on CPU as a 2D grid at z=0.
  Default probe: H with q=+0.4 (shows epair attraction). Switch to O with q=-0.4
  to see sigma-hole attraction.
- **Atom rendering**: CPK colors for real atoms. Epairs: cyan (0,0.8,0.8)
  semi-transparent. Sigma holes: magenta (0.8,0,0.8) semi-transparent. Same size.
- **Bonds**: Real atom bonds as independent line segments (connect='segments').
  Faint dummy-bond lines from epairs (cyan) and sigma holes (magenta) to host atoms.
- **Mouse picking**: LMB drag on dynamic atoms creates anchor springs.
  Ray-nearest-point method ensures force is in XY plane.

### Compact Radial Potentials Design

See **FastPairwisePotentials.chat.md** for the full design discussion. Two
families of compact-support potentials were developed:

#### 1. Polynomial cutoff family (earlier)

Uses `z(r) = (1 - r²/r_c²)^q` as the overlap variable and `V = C_R·z² - C_A·z`.
Converges toward Gaussian-like radial dependence — does not reproduce the
exponential Morse tail well even at high powers.

Analytical `pure-tail` mode: sets `c=0`, solves for pair-specific `rc` that
satisfies exact depth, equilibrium, and purely attractive tail. For `fc22`:
`rc² = R0² + 4n·Δ·R0` where `Δ = ln(2)/β`.

#### 2. Compact exponential family (current, recommended)

Approximates the Morse exponential directly: `y = max(0, 1 - β(ρ-R₀)/n)^n`
converges to `exp(-β(r-R₀))` as `n→∞`. This reproduces the Morse tail much
better than the polynomial family at equal power.

**Unified energy formula** (atoms and epairs, same instructions):
```
V = E₀·y·(α·y - (1+α))    where y = max(0, 1 - β(ρ-R₀)/n)^n
```
- `α=1, w=0`: compact Morse (atom-atom), sharp repulsive core
- `α=0, w>0`: purely attractive smooth blob (atom-epair), no repulsive spike
- `V(R₀)=-E₀`, `V'(R₀)=0`, `V''(R₀)=2E₀β²` exactly for all n

**Soft radius** (blunts epair origin without branch):
```
ρ(r,w) = √(r²+w²) - w = r²/(√(r²+w²)+w)    [numerically stable form]
```
- `w=0`: `ρ=r` (sharp atom core)
- `w>0`: `ρ ≈ r²/(2w)` near origin (smooth parabolic center)

**Branch-free mixing rules** (precomputed in type-pair table):
- `g_ij = g_i·g_j` (core flag: 1=atom, 0=epair)
- `R₀ = g_ij·(Rᵢ+Rⱼ)` (epair-atom: R₀=0)
- `E₀ = eᵢ·eⱼ` (geometric energy mixing)
- `α = g_ij` (atom: α=1, epair: α=0)
- `w = wᵢ+wⱼ` (atom: w=0, epair: w>0)

**Cutoff**: `ρ_c = R₀ + n/β`, physical `rc² = ρ_c·(ρ_c + 2w)`.
Recommended `n=8`: three squarings for `u^8`, truncation error `~2e⁻⁸`.

**GPU evaluation** (branch-free, all lanes same instructions):
```cpp
float r2 = dot(dr, dr);
float rw  = sqrt(r2 + w*w);
float rho = r2 / max(rw + w, eps);
float u   = max(0.0f, 1.0f - (beta/n)*(rho - R0));
float y   = pow_by_squaring(u, n);   // e.g. u²→u⁴→u⁸
float E   = E0 * y * (alpha*y - (1.0f + alpha));
```

**TODO (future kernel refactoring)**: The current `rigid_body_pairff_kernel`
in `kernels/rigid.cl` uses 4 separate interaction loops (atom-atom Morse+Coulomb,
atom-epair Lorentzian, epair-atom Lorentzian, epair-epair skipped). The compact
exponential family eliminates this split: a single loop over all atoms (real +
epair + sigma) with per-pair parameters (R₀, E₀, α, w) from the mixing rules
above. This removes the `if (atom_idx < n_dyn_atoms)` branch and the sorted
array requirement. See `fit_radial.py` (`--compact-exp-demo`) and
`FastPairwisePotentials.chat.md` (from line 1366) for the full derivation.

Usage:
```
python3 fit_radial.py --compact-exp-demo --exp-powers 2,4,8 --soft-radius sqrt --epair-width 0.6 --epair-alpha 0.0
```

## Key Files in Other Directories

- **py/FFs/RigidBodyDynamics.py** — `RigidBodyPairFF` class: GPU rigid-body
  dynamics with pairwise molecule-molecule interactions. Manages electron pair
  construction, REQ extension, kernel launch, and FIRE relaxation.
- **py/GUI/RigidBodyVispy.py** — Interactive Vispy+PyQt5 visualization for
  RigidBodyPairFF with mouse picking, potential map, and parameter controls.
- **kernels/rigid.cl** — OpenCL kernels for 6-DOF rigid body dynamics:
  quaternion integration, gyroscopic torque, FIRE, Newton trust-region,
  folded-basis substrate, and pairwise molecule-molecule forces.

## Testing Report

### Headless FIRE relaxation

Tested with `demo_pairff.py --no-vis --steps 200`:
- Kernel compiles and runs on NVIDIA RTX 3090 without errors
- Dynamic molecule (HCOOH, 5 atoms + 4 epairs + 1 sigma hole = 10 total) correctly
  includes electron pairs and sigma holes
- Static molecule (uracil, 12 atoms + 6 epairs + 2 sigma holes = 20 total) correctly
  populated
- Type arrays: `[0 0 0 0 0 1 1 1 1 2]` (dynamic), `[0×12, 1×6, 2×2]` (static) —
  sorted as required for branch-free kernel
- FIRE relaxation converges toward energy minimum with decreasing force magnitude
- Z-constraint keeps molecule near z≈0 (|z| < 0.003 Å after 200 steps)

### Interactive GUI

Tested with `demo_pairff.py` (timeout 5s):
- Vispy+PyQt5 window opens without errors
- No GL drawing warnings or errors
- Potential map renders with H:+0.4 probe showing epair Hbond attraction wells
- Switching probe to O:-0.4 shows sigma-hole attraction wells (with Hs=1.0)
- Changing FF parameters (He, Hs, rc, w) triggers live map recompute
- Epairs render as cyan dots, sigma holes as magenta dots
- Dummy-bond lines connect epairs/sigma to host atoms
- Real atom bonds render as independent segments (no spurious diagonal lines)

### Parameter verification

- `--hs 0.5` and `--hs 1.0` tested: sigma-hole contribution visible with negative probe
- `--epair-dist 1.4` and `--sigma-dist 1.0` tested: correct dummy atom placement
- Default `Hs=0.0` correctly disables sigma holes (no contribution)
- Default `Hs=1.0` enables sigma holes with same magnitude as |He|

### Open issues

- `L_epair` and `L_sigma` GUI changes store values but don't rebuild molecules
  (requires re-running `from_two_molecules`)
- Potential map does not include Coulomb (by design, matches morseH_only),
  but the GPU kernel does — map is an approximation for visualization
- Bond computation uses fixed 1.8 Å cutoff — may miss long bonds or include
  spurious ones for non-standard geometries
- Map is computed at z=0 (molecular plane) — for 3D molecules with atoms out
  of plane, the map is a projection, not a true cross-section
- FIRE relaxation does not fully converge in 200-300 steps for the uracil+HCOOH
  system (F~0.02, T~0.04) — likely needs more steps or parameter tuning
