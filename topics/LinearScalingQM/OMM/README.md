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

### Energy functional
Minimizes E = Tr(C^T H C) where C are orbital coefficients, subject to
orthogonality constraints C^T S C = I. The orbitals are confined to a
finite support region around each atomic site.

### 3-phase gradient (GPU)
To avoid atomic operations on GPU, the gradient is computed in 3 phases:
1. **apply_operator**: compute dual vectors |phi'> = Op |phi> on expanded supports
2. **project_orbitals**: compute scalar overlaps <phi_i | phi'_j>
3. **assemble_gradient**: combine scalars and duals into final gradient

### Projective dynamics
Orthogonality constraints are enforced by solving a linear system
(projective step) after each gradient descent step, similar to constraint
solvers in game physics engines. This separates energy minimization from
constraint satisfaction.

### FIRE acceleration
The Fast Inertial Relaxation Engine adaptively adjusts the time step and
mixing parameter based on the alignment of force and velocity, providing
faster convergence than plain gradient descent.

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
