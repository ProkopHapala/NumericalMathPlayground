# BRBFFF GPU Relaxation Contract

`BRBFFF` is the experimental OpenCL bridge between atom-wise force evaluators and a two-frame blended molecular representation.  Its purpose is not to reproduce vibrational masses or long-time MD.  Its purpose is to keep the expensive atom-side geometry and force work on the GPU, while moving the small set of frame poses in a direction that decreases the energy represented by the supplied forces.

The implementation lives in [`py/FFs/BRBFFF.py`](/home/prokop/git/NumericalMathPlayground/py/FFs/BRBFFF.py) and [`py/kernels/BRBFFF.cl`](/home/prokop/git/NumericalMathPlayground/py/kernels/BRBFFF.cl).  The CPU reference for a *complete* reduced potential, energy acceptance, and trust-region handling is [`py/FFs/Vibrations.py`](/home/prokop/git/NumericalMathPlayground/py/FFs/Vibrations.py).

## What it represents

Each atom has one reference position \(r_i\) and two weights \((w_{i0},w_{i1})\).  Each frame has a world position \(p_a\) and a unit quaternion \(q_a\), whose rotation is \(R(q_a)\).  The reconstructed atom position is

\[
x_i = w_{i0}\bigl(R(q_0)r_i+p_0\bigr)
    + w_{i1}\bigl(R(q_1)r_i+p_1\bigr).
\]

The Python initializer requires \(w_{i0}+w_{i1}=1\).  This makes equal frame poses reproduce an exact global rigid transform.  Non-negative weights are strongly recommended, but are not currently enforced; negative weights can amplify motion and make a descent step poorly conditioned.

There are 12 stored pose coordinates: two translations and two rotations.  The fitted reduced potential in the CPU path restores only the six *relative* coordinates, so global molecule translation and rotation remain free.

## Why force projection is the correct coupling

Let an atomic force evaluator return \(f_i=-\partial E/\partial x_i\).  A direct small frame update has translation \(\delta p_a\) and a world-space left rotation \(\delta\theta_a\):

\[
\delta x_i = \sum_a w_{ia}\left(\delta p_a+
    \delta\theta_a\times R(q_a)r_i\right).
\]

Substituting this into virtual work gives the two frame wrenches used by the GPU kernel:

\[
F_a=\sum_i w_{ia} f_i, \qquad
\tau_a=\sum_i w_{ia}\bigl(R(q_a)r_i\bigr)\times f_i .
\]

Therefore

\[
\delta E=-\sum_a\left(F_a\!\cdot\!\delta p_a+
\tau_a\!\cdot\!\delta\theta_a\right).
\]

Moving along \((F_a,\tau_a)\) is consequently generalized gradient descent.  No physical mass is required to find a relaxed structure.  The torque is deliberately about each frame’s own origin, not the world origin: that is what matches the direct update `p += dp`, `q <- exp(dtheta) q`.

## Device-resident iteration

The normal GPU iteration is:

```text
frame_pos, frame_quat
        │
        ▼
brbfff_reconstruct_positions
        │  atom_pos
        ▼
external GPU force evaluator
        │  atom_force
        ▼
brbfff_project_atomic_forces
        │  frame_force, frame_torque
        ▼
brbfff_relax_step
        │
        └── reconstruct atom_pos for the next evaluator call
```

`BRBFFF.device_buffers()` exposes the persistent `atom_pos` and `atom_force` buffers to the external evaluator.  That evaluator must overwrite or clear all atom forces for every iteration; retaining a force from an earlier geometry is physically wrong.  It must use the same `(system, atom)` ordering as `BRBFFF`.

The convenience call `relax_step()` expects that `atom_force` already represents the current geometry, projects it, updates frames, and reconstructs the next geometry.  Calling it with a host `atom_forces` array is useful for diagnostics.  For a fully device-resident loop, launch the external evaluator after reconstruction and then call `relax_step()` without the array.

## The propagator and what “stable” means here

The kernel maintains two step-history vectors.  For each frame it forms

\[
d p = \operatorname{cap}\bigl(\gamma d p_{\rm old}+\mu_t F,
                               d p_{\max}\bigr),
\]
\[
d\theta = \operatorname{cap}\bigl(\gamma d\theta_{\rm old}+\mu_r\tau,
                                   d\theta_{\max}\bigr),
\]

then applies

\[
p\leftarrow p+d p,\qquad q\leftarrow
\operatorname{normalize}\left(\exp(d\theta)q\right).
\]

`damping=0` gives capped gradient descent.  A positive `damping` only smooths successive descent directions; these stored values are not masses, momenta, or a physically calibrated velocity.  Translation and angle caps are the first protection against a singular close-contact force causing a catastrophic update.  Quaternions are renormalized every step.

This makes the map well-defined and practically controllable, but it is not an unconditional proof of monotonic energy decrease.  Explicit gradient steps can still overshoot if the mobility or caps are too large, if the supplied forces are noisy/non-conservative, or if the external potential is discontinuous.  A conservative evaluator that also returns total energy is required for strict line-search/backtracking acceptance.  Until that exists, start with `damping=0`, small caps, and recompute forces after every move.

If a pose is teleported, constrained, or edited externally, call `set_frame_state()` or `reset_relaxation()` to discard the old smoothed step.  Reusing it after a discontinuity can cause an artificial jump.

## Critical current limitation: no internal restoring force on GPU

The current `BRBFFF` kernels do **not** evaluate `anharmonic_model.npz`, the harmonic stiffness, or any other fitted relative-frame potential.  They only propagate the forces present in `atom_force`.

Consequences:

- An isolated distorted molecule with an empty `atom_force` buffer does not return to its fitted relaxed shape.
- A Coulomb/Morse/grid external interaction can move the molecule, but it does not by itself supply the model’s internal bend/twist restoring force.
- The current GPU path is an external-force coupling layer and a generic descent mechanism, not yet a self-contained reduced molecular force field.

This is the highest-priority completion item.  The GPU must evaluate the stabilized relative-frame energy and generalized restoring force, add it to the projected external wrench, and make the combined energy available for backtracking.  Physical masses remain optional for that relaxation goal.

## Relation to CPU reduced dynamics

`BlendedFrameDynamics` in `Vibrations.py` already has a fuller mechanics contract: it evaluates the fitted relative potential, maps its forces to the 12 frame coordinates, can obtain an instantaneous reduced mass \(J^T M J\), enforces the fitted trust region, and accepts energy-decreasing relaxation steps when the external callback supplies energy.

The GPU implementation intentionally does not borrow independent rigid-body inertias from `rigid.cl`.  That would define a different inertial model.  It is not needed for gradient descent, but a configuration-dependent mass and a Lie-group/variational treatment would be needed before claiming accurate long-time inertial molecular dynamics.

## Verification status

- The direct-frame virtual-work formula was checked by finite difference; the current random test residual is `4.68e-6` at a `1e-6` perturbation.
- Python syntax and repository whitespace checks pass.
- GPU/CPU numerical parity for reconstruction, wrenches, and one relaxation step has **not** run in this workspace: the NVIDIA driver is unavailable and PoCL fails even a trivial OpenCL build.  The GPU result must therefore still be treated as experimental.

Required parity tests on a working device are:

1. Compare reconstructed positions against NumPy/CPU skinning for randomized poses and weights.
2. Compare both projected frame forces and frame-origin torques against CPU sums.
3. Compare one zero-damping capped update, including quaternion sign-insensitive orientation comparison.
4. Check finite values and unit-quaternion norms after many capped iterations.
5. For a conservative test potential with energy, verify accepted steps decrease the combined energy.

## Completion roadmap

1. **Port the internal reduced potential and force.** Add the stabilized relative-pose energy and generalized restoring force to the GPU wrenches; this is required for isolated molecular relaxation.
2. **Add total-energy acceptance.** Define an energy buffer/interface for the external evaluator and combine it with the internal energy for backtracking or a device-side line search.
3. **Run CPU/GPU parity.** Do not tune mobility values or call the path stable before the five checks above pass on a real device.
4. **Integrate a real external kernel.** Establish buffer clearing, system batching, exclusions, units, and force/energy semantics for the intended Coulomb/Morse/grid implementation.
5. **Add robust stopping criteria.** Use projected wrench norms, accepted energy change, and cap-hit reporting; raw atomic force norms alone are not a reduced-coordinate convergence criterion.
6. **Improve the representation.** Multi-frame generalization, optimized weights, and nonlinear SE(3) interpolation improve what can be represented, but they do not replace the internal restoring potential.
7. **Only then consider inertial animation.** A configuration-dependent reduced mass and a suitable Lie-group integrator are separate work for physically meaningful dynamics, not prerequisites for relaxation.
