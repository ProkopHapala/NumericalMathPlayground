# MultiGridFF

Blended rigid-body frames are a reduced-coordinate experiment for molecular force fields.  Two frames provide a smooth six-dimensional internal motion, while the full UFF Hessian supplies a physically grounded local stiffness.  The topic is deliberately diagnostic: a good fit in six coordinates is not evidence that the molecule has only six soft modes.

## Files

- `BRBFF_stiffness_fit_plan.md` — design, coordinate conventions, corrected results, caveats, and TODOs.
- `BRBFFF_gpu_relaxation.md` — GPU skinning/force-projection/relaxation contract, stability conditions, and implementation roadmap.
- `BlendedRigidBodyFrames.chat.md` — original design discussion, including the nonlinear and multigrid roadmap.
- `fit_stiffness.py` — authoritative PTCDA/UFF diagnostic; fits a local cubic/quartic potential, exports its stabilized form, and verifies force-projected two-frame dynamics.
- `py/FFs/BRBFFF.py` and `py/kernels/BRBFFF.cl` — experimental OpenCL external-force relaxer that keeps skinned positions, atomic-force projection, and capped two-frame pose updates resident on the GPU; it does not yet evaluate the fitted internal restoring potential.
- `blended_rigid_frames.py` — construction and visualization of smooth frame weights.
- `run_vib_PTCDA.py` — baseline UFF vibration driver.
- `fit_stiffness copy.py` — legacy pre-fix copy; do not use for new results.
- `debug/fit_stiffness/*` — generated CSV/PNG/log/NPZ validation artifacts; `anharmonic_model.npz` is the stabilized runtime model, while `anharmonic_model_local.npz` preserves the raw regression result.

## Coordinate contract

The frame displacement is \(\xi=(\Delta t,\Delta\theta)\).  The implementation scales translations and rotations into one internal coordinate, \(\eta=S\xi\), and all stiffness matrices satisfy

\[
  V_2=\tfrac12\eta^T K_\eta\eta, \qquad Q_\eta=-K_\eta\eta .
\]

This distinction matters: a QR basis is useful for measuring the represented subspace, but its coordinates are not physical coordinates.  Stiffness must be transformed by a congruence/Jacobian, not copied from a QR-coordinate matrix.

The script reports three restricted models (Galerkin, Edge-LS, and E/F-fit) against the same unrelaxed blended geometry.  `Relaxed` is static condensation with eliminated Cartesian coordinates allowed to respond, so it is a different target and is not a restricted-force residual competitor.

## Validation snapshot

- Edge-LS/direct parity: relative error `2.5e-7`.
- Virtual-work parity: relative L2 error `1.8e-3` (about `4.1e-4 eV/Å` absolute).
- Two-frame spectral target rank: `4/6`; two low UFF directions are outside the represented tangent space.
- True and blended rigid motions preserve energy to below `1.4e-6 eV`.
- On independent finite-amplitude samples, the raw cubic/quartic model reduces energy RMSE from `1.50e-3` to `1.99e-5 eV` and generalized-force component RMSE from `9.85e-3` to `1.37e-4 eV/Å`.
- The default stabilized export has a positive sampled quartic floor (`1.29e-6 eV` on the dimensionless unit sphere) and completes six half-box load/unload cycles without a boundary hit; replayed force-balance residuals are below `1e-15 eV/Å`.
- Full two-frame virtual work agrees with Cartesian work to `5.6e-12` relative.  A damped bent/twisted trajectory stays bounded and lowers reduced energy from `4.24e-4` to `3.76e-4 eV`.

| Quantity | Current result |
|---|---:|
| Restricted Galerkin eta stiffness eigenvalues | `0.0275, 0.0321, 0.0793, 0.6352, 6.4803, 13.7437 eV/Å²` |
| Restricted Galerkin frequencies | `50, 76, 149, 191, 259, 406 cm⁻¹` |
| UFF target frequencies | `46, 65, 123, 125, 139, 164 cm⁻¹` |
| Spectral overlap rank | `4/6` |

The anharmonic model keeps the Galerkin Hessian fixed and fits 56 cubic plus 126 quartic monomials in dimensionless scaled eta coordinates.  It is trained on paired `±eta` samples and selects ridge regularization on an independent set.  The raw fit has a sampled negative quartic direction, so the exported model adds a small isotropic term \(\lambda\|\eta/\mathrm{scale}\|^4\).  This makes the sampled quartic scan positive but is **not** a mathematical proof of global positivity.

## Relaxation contract

`anharmonic_model.npz` is meant for zero-temperature reduced relaxation, not unconstrained molecular dynamics.  Minimize

\[
V(\eta)-f_{\rm ext}^T\eta
\]

with the provided `relax_reduced_potential` helper and a finite trust radius in units of `coordinate_scale`.  The solver first honors the box with L-BFGS-B and then polishes an interior force-balance root.  A reported `hit_boundary=True` is not a converged material response: it means the external load requested a deformation outside the fitted domain.  The companion `relax_reduced_load_path` performs load continuation, which is preferable to repeatedly starting nonlinear relaxations from zero.

The stabilizer changes the held-out RMSE from `1.99e-5` to `6.18e-5 eV` and force-component RMSE from `1.37e-4` to `4.45e-4 eV/Å`.  That small deliberate loss of local regression accuracy buys a bounded operational energy within the trust box; it does not establish atomistic stability after the eliminated Cartesian coordinates are allowed to relax.

## Force-projected frame dynamics

The runtime state has two full SE(3) frame poses (12 moving coordinates).  The molecule can therefore translate and rotate as a whole while the fitted six-coordinate potential restores only the relative bend/twist.  At each state, positions are reconstructed by weighted frame skinning, then an atomic external evaluator—Coulomb, Morse, a contact model, or any other conservative pairwise implementation—returns `(energy, forces)`.  The generalized load is exactly the chain-rule/virtual-work projection

\[
Q_{\rm ext}=J(\mathrm{frames})^T F_{\rm atom},\qquad M_{\rm frame}=J^T M_{\rm atom}J .
\]

`BlendedFrameDynamics.evaluate()` returns these quantities.  `relax_step()` is the safe default for mechanics: when the external callback supplies energy, it backtracks until the **total** reduced plus external energy decreases.  `step_velocity_verlet()` supports local damped motion in fs and caps each atom's implied displacement; both methods enforce the fitted relative-eta trust box, rejecting an impulse that would leave it.  It is suitable for smoothened/damped animation and exploratory mechanics, but it is not yet a symplectic, energy-conserving long-NVE-MD method because the SE(3) Jacobian and configuration-dependent mass are currently finite-differenced/instantaneous.

For GPU external interactions, `BRBFFF` provides the buffer-resident external-force part of a relaxation loop: it skins atom positions with two quaternion frames, reduces any resident `atom_force` buffer to the two frame-origin wrenches using the virtual-work rule, then performs a capped overdamped pose update.  This is the intended bridge for GPU Coulomb/Morse/grid kernels.  It requires no mass model: with damping disabled it is generalized gradient descent, while damping only smooths the sequence of updates.  Crucially, it does **not** yet add the fitted relative-frame potential; therefore it will not restore a distorted isolated molecule unless the external evaluator itself provides intrinsic molecular forces.  Strict monotonic energy descent needs an external energy evaluator for backtracking.  See `BRBFFF_gpu_relaxation.md` for the complete contract and roadmap.  GPU/CPU numerical parity remains to be run when a functional OpenCL compiler/device is available.

The remaining frequency mismatch is mostly a basis/physics limitation, not a numerical fitting failure.  More frames, residual internal modes, or learned weights are required to represent additional soft modes.  Float32 UFF forces also leave a small stationarity floor near the minimum.

## Future outlook

| Item | Status / rationale |
|---|---|
| Weight optimization | Open; optimize a few smooth parameters against mass-weighted mode overlap before considering free per-atom weights. |
| Constrained relaxation | Open and highest-priority physics extension if the desired target is atomistically relaxed response rather than the current restricted manifold. |
| Multi-frame (K>2) | Open; the main route beyond the current `4/6` representability rank, but requires a frame graph and generalized gauge handling. |
| Nonlinear SE(3) interpolation | Open; geodesic interpolation should be evaluated before dual-quaternion blending because it has a clearer derivative and invariance story. |
| Graph biharmonic weights | Low-priority experiment; positivity and overshoot constraints are unresolved. |
| Cosine/global anharmonic terms | Deferred until a larger rotational training domain and a globally stable potential are defined. |
| Complete GPU relaxation | Highest GPU priority: evaluate the stabilized relative-frame potential and its projected restoring force on device, then expose total energy for acceptance/backtracking. |

## Reproduce

From the repository root:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  python3 topics/MultiGridFF/fit_stiffness.py
```

The run writes diagnostics to `debug/fit_stiffness/`, including `load_relaxation.csv`, `load_relaxation.png`, and `frame_dynamics_validation.csv`.  `anharmonic_model.npz` is the complete stabilized reusable model (harmonic anchor, coordinate scales, monomial powers, and coefficients); the coefficient CSV is intended for inspection.  Do not separate coefficients from their coordinate scales, ignore a boundary hit, or extrapolate beyond the fitted eta box.
