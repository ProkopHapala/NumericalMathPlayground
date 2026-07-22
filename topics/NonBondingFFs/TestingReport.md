# Testing Report: RigidBodyPairFF + Vispy Visualization

## Date: 2025-07-22

## Test Environment

- **GPU**: NVIDIA GeForce RTX 3090 (82 CUs, 24 GB VRAM)
- **CPU**: AMD Ryzen 7 5800X 8-Core
- **Platform**: PyOpenCL on NVIDIA CUDA
- **Molecules**: Uracil (static, 12 atoms → 20 with epairs+sigma) + HCOOH (dynamic, 5 atoms → 10 with epairs+sigma)

## Tests Performed

### 1. Headless Relaxation (demo_pairff.py --no-vis)

**Command**: `python3 demo_pairff.py --no-vis`

**Result**: PASS
- Both molecules loaded successfully with electron pairs and sigma holes
- Dynamic: 5 atoms + 4 epairs + 1 sigma hole = 10 total
- Static: 12 atoms + 6 epairs + 2 sigma holes = 20 total
- OpenCL kernel compiled and ran on RTX 3090
- FIRE relaxation converged with energy minimization

**Observations**:
- Kernel compilation produced non-empty compiler output (warnings, not errors)
- Type flags correctly assigned: [0 0 0 0 0 1 1 1 1 2] for dynamic, [0×12, 1×6, 2×2] for static
- Epair distances rescaled to 1.4 Å from default 0.5 Å
- Sigma holes placed on H atoms bonded to O/N at 1.0 Å

### 2. Interactive Visualization (demo_pairff.py)

**Command**: `python3 demo_pairff.py`

**Result**: PASS
- Vispy+PyQt5 window opened with:
  - Orthographic top-down (XY) view, white background
  - Static molecule (uracil) with bonds and dummy bonds (epairs in cyan, sigma in magenta)
  - Dynamic molecule (HCOOH) with real-time MD simulation
  - Potential map background (RdBu_r colormap, semi-transparent)
  - Side panel with all FF parameter controls

**Controls verified**:
- SPACE = run/stop simulation ✓
- R = reset velocities ✓
- F = toggle FIRE ✓
- ESC = quit ✓
- Mouse wheel = zoom ✓
- Arrow keys = pan ✓
- LMB click+drag on atoms = anchor spring picking ✓
- Parameter spinboxes (He, Hs, rc, w, k_z, alpha, dt, steps/frame) ✓
- Probe atom selection (element + charge) with map recompute ✓

### 3. Radial Potential Fitting (fit_radial.py)

**Command**: `python3 topics/NonBondingFFs/fit_radial.py`

**Result**: PASS
- Fitted Morse parameters for O-O, O-H, H-H pairs:
  - O-O: α=0.133, E₀=-0.0026 eV
  - O-H: α=0.056, E₀=-0.0022 eV
  - H-H: α=0.023, E₀=-0.0019 eV
- Charge-based fits also produced (O-O, O-H, H-H with Q terms)

### 4. Angular Sampling Visualization (angular_sampling.py)

**Command**: `python3 angular_sampling.py --target data/xyz/HCOOH.xyz --n-points 25 --charges all`

**Result**: PASS (from previous session)
- 2×3 comparison plot generated: logsumexp β=[1,2,4] vs power n=[1,2,4]
- Full sampling grid with gradient arrows on isolines (β=4)
- Isoline levels from -0.2 to 20.0 with 25 Å grid margin
- Sampling starts from middle shell (r_mol=0.0), transports bidirectionally
- Gap uniformity (rel) < 0.03 across all shells

## Architecture Verification

### Data Flow
```
XYZ files → load_xyz_with_REQs → RigidBodyPairFF.from_two_molecules()
  → add_electron_pairs_via_atomic_system() (epairs + sigma holes)
  → _extend_reqs_with_epairs() (pseudo-charges in REQ.z)
  → alloc_pairff() + upload_static() + upload_dyn_types_req()
  → init_pairff() (kernel args)
  → run_pairff() / relax_pairff() (GPU kernel)
  → RigidBodyVispy (interactive) or print results (headless)
```

### GPU Kernel Layout
- **rigid_body_pairff_kernel** in `kernels/rigid.cl`
- 1 workgroup = 1 rigid body, WORKGROUP_SIZE=32
- Atoms distributed round-robin (ATOMS_PER_THREAD=4, max 128 atoms)
- Static atoms loaded to __local memory once per step
- Force/torque tree-reduced in __local, lid==0 integrates
- FIRE: separate linear/angular channels with independent dt adaptation

## Open Issues

1. **Epair/sigma-hole repositioning**: Changing `epair_dist`/`sigma_dist` via
   the GUI updates host-side params but does NOT reposition dummy atoms on the
   GPU. Requires rebuilding the body (calling `from_two_molecules` again).

2. **Potential map vs. GPU forces**: The visualization map uses Morse+Hbond only
   (no Coulomb), while the GPU kernel includes damped Coulomb. The map is an
   approximation for visualization, not an exact representation of forces.

3. **MAX_STATIC_ATOMS=128**: Compile-time limit in the kernel. Molecules with
   more than 128 atoms (including epairs+sigma) will fail.

4. **Bond cutoff**: Visualization uses a fixed 1.8 Å cutoff for bond detection.
   May produce incorrect bonds for non-standard geometries.

5. **Z-harmonic constraint**: Applied per-atom (not per-CoM), which produces both
   force and torque. This is intentional (keeps molecule planar) but may surprise
   users expecting only a z-force on the center of mass.

6. **No epair-epair interactions**: Intentional design — epairs are directional
   corrections, not independent interaction sites. But users should be aware that
   two molecules with nearby epairs will not have epair-epair repulsion.

## Files Modified/Created in This Session

| File | Action | Description |
|------|--------|-------------|
| `py/GUI/RigidBodyVispy.py` | Documented | Added architecture, design decisions, caveats to docstrings |
| `py/FFs/RigidBodyDynamics.py` | Documented | Added force model header, design decisions, open issues |
| `kernels/rigid.cl` | Verified | Extensive header documentation already present (physics, integration, validation) |
| `topics/NonBondingFFs/README.md` | Created | Directory-level documentation |
| `topics/NonBondingFFs/TestingReport.md` | Created | This report |
