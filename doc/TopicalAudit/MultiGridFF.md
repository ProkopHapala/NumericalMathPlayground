---
type: TopicalAudit
title: Blended Rigid-Body Frame Force Fields
description: Inventory and validation status for the MultiGridFF reduced-coordinate stiffness experiment.
tags: [multigrid, forcefield, reduced-order-model, molecular-mechanics]
---

# Blended Rigid-Body Frame Force Fields

## Summary

`topics/MultiGridFF` studies a two-frame, six-coordinate approximation to molecular motion.  It is coupled to UFF finite-difference Hessians and is currently a research/diagnostic topic rather than a production force-field backend.

## Implementations

| Language | Location | Status | Notes |
|---|---|---|---|
| Python | [`topics/MultiGridFF/fit_stiffness.py`](/home/prokop/git/NumericalMathPlayground/topics/MultiGridFF/fit_stiffness.py) | active | Fits eta-coordinate restricted models and validates two-frame skinning, force transfer, reduced damping, and external-force-driven motion. |
| Python | [`py/FFs/Vibrations.py`](/home/prokop/git/NumericalMathPlayground/py/FFs/Vibrations.py) | active | Builds mass-correct modes and provides polynomial stability, SE(3) skinning, `J.T @ F` force transfer, `J.T @ M @ J` mass, and reduced dynamics APIs. |
| Python/OpenCL | [`py/FFs/BRBFFF.py`](/home/prokop/git/NumericalMathPlayground/py/FFs/BRBFFF.py), [`py/kernels/BRBFFF.cl`](/home/prokop/git/NumericalMathPlayground/py/kernels/BRBFFF.cl), [`BRBFFF_gpu_relaxation.md`](/home/prokop/git/NumericalMathPlayground/topics/MultiGridFF/BRBFFF_gpu_relaxation.md) | experimental | Persistent GPU skinning, exact atomic-force-to-frame-origin-wrench projection, and capped overdamped frame relaxation.  It is mass-free generalized descent, not physical inertial MD, and does not yet add the fitted internal restoring potential. |
| Python | [`topics/MultiGridFF/blended_rigid_frames.py`](/home/prokop/git/NumericalMathPlayground/topics/MultiGridFF/blended_rigid_frames.py) | experimental | Smoothstep/radial frame weights and visualization. |
| Python | [`topics/MultiGridFF/run_vib_PTCDA.py`](/home/prokop/git/NumericalMathPlayground/topics/MultiGridFF/run_vib_PTCDA.py) | diagnostic | Baseline PTCDA vibration run. |
| Markdown | [`topics/MultiGridFF/BRBFF_stiffness_fit_plan.md`](/home/prokop/git/NumericalMathPlayground/topics/MultiGridFF/BRBFF_stiffness_fit_plan.md) | active | Design contract, corrected results, caveats, and follow-up work. |
| Markdown | [`topics/MultiGridFF/BlendedRigidBodyFrames.chat.md`](/home/prokop/git/NumericalMathPlayground/topics/MultiGridFF/BlendedRigidBodyFrames.chat.md) | experimental | Design rationale and nonlinear/multiframe roadmap. |
| Python | [`topics/MultiGridFF/fit_stiffness copy.py`](</home/prokop/git/NumericalMathPlayground/topics/MultiGridFF/fit_stiffness copy.py>) | deprecated | Pre-fix copy; it is not an authoritative implementation. |

## Coordinate and physics contract

All reduced stiffnesses use \(\eta=S\xi\), not QR coordinates.  Cartesian force parity is checked through the geometry Jacobian and virtual work.  `Relaxed` means static condensation and must not be compared directly with restricted blended-geometry samples.  Rigid invariance requires absolute SE(3) position transforms; center-relative frame rotations produce spurious energy changes.

`Vibrations.py` projects the mass-weighted dynamical matrix, then maps the projector back to Cartesian coordinates.  Returned modes are mass-normalized and spectroscopic frequencies use \(\omega/(2\pi c)\), not \(\omega/c\).

## Parity Status

- Synthetic SE(3) exponential/Jacobian checks pass at machine precision.
- Edge/direct stiffness parity is `2.5e-7` relative; virtual-work parity is `1.8e-3` relative L2.
- Mass-orthogonal rigid projection and mode normalization pass at approximately `1e-15` in synthetic checks.
- PTCDA rigid-motion energy drift is below `1.4e-6 eV`.
- The represented spectral target has rank `4/6`; this is an expressiveness limit of two frames.
- Representative restricted Galerkin frequencies are `50, 76, 149, 191, 259, 406 cm⁻¹`, versus UFF `46, 65, 123, 125, 139, 164 cm⁻¹`.
- The 182-term cubic/quartic fit is full rank; held-out energy and force-component RMSE are `1.99e-5 eV` and `1.37e-4 eV/Å`, respectively.
- The raw quartic has a sampled negative direction (`-1.18e-5 eV`); the default export adds `1.31e-5 eV * ||eta/scale||^4`, giving sampled range `[1.29e-6, 5.62e-5] eV`.
- Analytic polynomial force agrees with its finite-difference energy gradient to `3.6e-10 eV/Å`; NPZ save/load parity is exact at float64 precision.
- A replay of the serialized stabilized model completes all six half-box load/unload cycles without boundary contact, with maximum interior force-balance residual below `1e-15 eV/Å`.
- Full 12-coordinate frame virtual work agrees with Cartesian atomic work to `5.6e-12` relative.
- A damped PTCDA bend/twist trajectory decreases energy from `4.24e-4` to `3.76e-4 eV` without an energy overshoot; an atom--environment pair-force callback moves the projected molecular COM by `7.94e-3 Å`.

## Open Issues / TODOs

1. Produce constrained-relaxation training samples if an atomistically relaxed potential is the intended target; current load paths relax only the six-coordinate surrogate.
2. Increase frame count or augment with residual internal modes before expecting all six UFF low modes.
3. Optimize weights using mass-weighted principal angles instead of relying only on geometric smoothstep weights.
4. Replace the diagnostic finite-difference SE(3) geometry and relative-log Jacobians with analytic production Jacobians.
5. Quantify float32 force-floor effects and choose a robust relaxation/stationarity criterion.
6. Replace sampled radial stabilization with a formal global-positivity construction if unbounded-domain relaxation or dynamics is required.  Current runtime safety relies on a finite trust box and rejecting boundary hits.
7. Evaluate geodesic SE(3) interpolation before dual-quaternion blending; both change the nonlinear reduced manifold.
8. Revisit graph biharmonic weights only after multi-frame experiments; enforce partition of unity and nonnegative weights if used.
9. Decide whether environment-dependent corrections are needed beyond the isolated restricted UFF manifold.
10. Upgrade damped local velocity-Verlet to a variational/Lie-group integrator including configuration-dependent-mass terms before claiming long-time NVE energy conservation.
11. Run GPU/CPU parity for `BRBFFF` position reconstruction, frame-origin wrenches, and one capped relaxation step once a working OpenCL compiler/device is available; the current host has no functional GPU driver and PoCL fails even a trivial kernel build.
12. Port the stabilized relative-frame potential and its generalized restoring force to `BRBFFF`, then combine it with external forces and total energy before describing the GPU path as a self-contained molecular relaxer.

No duplicate reduced-potential or frame-skinning dynamics implementation was found elsewhere in the Python tree; the reusable API deliberately lives in `Vibrations.py`, while `fit_stiffness.py` owns the PTCDA-specific validation/CSV glue.

Generated CSV/PNG/log/NPZ files under `debug/fit_stiffness` are reproducibility artifacts.  The default NPZ is the stabilized model and keeps the complete scaled-coordinate contract; rerun the script after changing code or force-field inputs.
