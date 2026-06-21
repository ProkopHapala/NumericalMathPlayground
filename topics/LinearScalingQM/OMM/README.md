# OMM

Orbital Minimization Method (OMM) — direct optimization of localized molecular
orbitals (LMOs) with finite support. Minimizes an unconstrained energy
functional with respect to orbital coefficients, using projective dynamics
for orthogonality constraints. CPU and GPU (OpenCL) implementations.

## Files

| File | Role |
|------|------|
| `OMM.py` | Pure Python reference implementation. Uses explicit loops (C-style) for clarity and as a porting reference for the GPU version. Handles orbital masks, neighbor lists, sparse matrix-vector products, and Jacobi orthogonalization |
| `OMM_ocl.py` | GPU-accelerated implementation using PyOpenCL. Minimizes energy functional with respect to LMOs. Gradient calculation split into 3 kernels to avoid atomic operations |
| `OMM_1D_grid.py` | 1D grid-based OMM solver. Optimizes localized orbitals on a 1D chain with Jacobi-style orthogonalization and optional support truncation. CPU prototype for the GPU version |
| `OMM_1D_grid_FIRE.py` | 1D grid-based OMM with FIRE (Fast Inertial Relaxation Engine) acceleration. Adaptively adjusts time-step and mixing parameters for faster convergence |
| `cl/OMM.cl` | OpenCL kernels for the 3-phase gradient computation: (1) apply_operator, (2) project_orbitals, (3) assemble_gradient |
| `Orbital_Minimization_Methond.chat.md` | Design discussion: localized orbital optimization, projective dynamics for constraints, finite support confinement |
| `OMM_Kernels.chat.md` | Discussion: OpenCL kernel design, data structures (neighbor lists, orbital masks), workload distribution among workgroups |
| `Approximate_Overlap.chat.md` | Discussion: NDDO (Neglect of Differential Overlap) methods and semi-empirical quantum chemistry for handling the overlap matrix |

## Method

### Problem

Find N occupied orbitals {φᵢ} that minimize the total energy

    E = Σᵢ ⟨φᵢ|H|φᵢ⟩

subject to orthonormality constraints

    ⟨φᵢ|φⱼ⟩ = δᵢⱼ    (general overlap: φᵢ^T S φⱼ = δᵢⱼ)

while each φᵢ is confined to a **finite spatial support** (mask) Mᵢ — a
small window of grid points or atom indices around a center. This makes
the method O(N): each orbital interacts only with neighbors whose masks
overlap.

### Localization by hard truncation

Each orbital is zeroed outside its mask Mᵢ after every update. Masks are
periodically **recentered** to the center-of-mass of |φᵢ|², and anything
leaking outside the new window is cut off. There is no soft confinement
potential — localization is purely geometric.

### Iteration loop

Each outer step consists of three phases:

**1. Energy gradient step.** The energy gradient for orbital i is

    ∇Eᵢ = 2 H|φᵢ⟩

(plain gradient descent) or the residual force

    Fᵢ = -(H|φᵢ⟩ - εᵢ|φᵢ⟩,    εᵢ = ⟨φᵢ|H|φᵢ⟩  (Rayleigh quotient)

which is tangent to the unit sphere (FIRE version). The update is applied
only within the mask:

    φᵢ[Mᵢ] -= dt · ∇Eᵢ[Mᵢ]

**2. Orthogonalization (iterative projective sweeps).** Exact
orthogonalization (Löwdin / Gram-Schmidt) is **not possible** with fixed
supports — the orthogonalized combination of two localized functions
generally leaks outside both masks. Instead, damped Jacobi-style
projections are used:

    repeat n_iter times:
      for each orbital i:
        correction = Σ_{j≠i} ⟨φᵢ|φⱼ⟩ · φⱼ    (restricted to Mᵢ)
        φᵢ -= damping · correction
        φᵢ /= ‖φᵢ‖

This iteratively pushes each orbital away from its neighbors proportional
to their overlap, staying within the support. The `OMM.py` version solves
a local 1D optimization per coefficient with inertia and step clamping:

    δcₖ = damping · b / (a + inertia),  clamped to ±max_step

where b is the overlap-error gradient and a is the local curvature. The
FIRE version adds momentum mixing (bmix) to accelerate convergence.

**3. Recenter supports.** Shift each mask to track the orbital
center-of-mass, truncate outside.

### GPU 3-phase gradient

On GPU, the gradient (including orthogonality Lagrange multipliers λᵢⱼ)
is computed in 3 kernels to avoid atomic operations:

1. **apply_operator**: compute H|φᵢ⟩ and S|φᵢ⟩ on expanded supports
   (one shell beyond Mᵢ, to capture overlap with neighbors)
2. **project_orbitals**: compute scalar overlaps λᵢⱼ = ⟨φᵢ|S|φⱼ⟩ for
   each neighbor pair (i,j) via sparse dot product
3. **assemble_gradient**: combine into the final gradient on the
   original support:

       gᵢ = H|φᵢ⟩ - Σⱼ λᵢⱼ · S|φⱼ⟩

   This is the gradient of the Lagrangian L = E - Σ λᵢⱼ (⟨φᵢ|S|φⱼ⟩ - δᵢⱼ).

### FIRE acceleration

The Fast Inertial Relaxation Engine adaptively adjusts the time step dt
and mixing parameter α based on the alignment of force and velocity
(P = F·v). When P > 0 (downhill), dt increases and α decays; when P < 0
(uphill), dt shrinks, velocity is reset, and α returns to its starting
value. This provides faster convergence than plain gradient descent.

## Key parameters

- `nMaxNeigh = 16` — maximum neighbors per atom
- `nMaxSupport = 64` — maximum atoms in orbital support
- `nMaxOrbsPerAtom = 8` — maximum orbitals an atom contributes to
- `PAD_IDX = -1` — padding index for sparse neighbor lists
- `BLOCK_SIZE = 256` — OpenCL workgroup size

## Usage

```bash
cd topics/LinearScalingQM/OMM

# CPU 1D prototype
python OMM_1D_grid.py
python OMM_1D_grid_FIRE.py

# GPU version
python OMM_ocl.py

# Pure Python reference
python OMM.py
```

## Dependencies

- numpy, scipy, matplotlib
- pyopencl (for OMM_ocl.py)
