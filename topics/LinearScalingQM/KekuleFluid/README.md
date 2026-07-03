# Kekulé Fluid — Bond-Order Dynamics & Dirac Quasiparticles on a Honeycomb Lattice

This project simulates the **Kekulé bond-order pattern** on finite graphene-like
honeycomb flakes (PAH molecules) and its effect on **π-electron quasiparticle
states**.  The physics is split into two coupled models:

- **Model A** — *Kekulé fluid*: a complex Ginzburg–Landau equation for a
  bond-order order parameter `z_i` on each atom, with valence constraint
  projection.  Produces bond orders `x_ij` and the complex Kekulé field `z_i`.

- **Model B** — *Dirac / tight-binding quasiparticle*: a single π electron
  hopping on the same lattice, with Kekulé-modulated hopping `t_ij = t₀ +
  δt·(x_ij − x₀)`.  Supports both real-time wavepacket propagation (OpenCL RK4)
  and exact diagonalization (numpy).

## File Inventory

### Core library (used by all scripts)

| File | Role |
|------|------|
| `Graph.py` | **Honeycomb graph builder** (current).  Defines `Atom`, `Bond`, `HoneycombGraph` with `build_pah()`, `build_rect_patch()`, `build_flake()`.  Provides bond-direction labeling, ring detection, bond-base computation, and array export for OpenCL. |
| `ModelA.py` | **Model A solver** (current).  `KekuleFluidSolver` class: OpenCL RK4 integration of the complex Ginzburg–Landau equation for `z_i`, with pinning, defect core suppression, valence projection (ABBA red-black sweep), and diagnostics. |
| `kekule_fluid.cl` | **OpenCL kernels for Model A**.  Implements `bondsToZ`, `applyPinsToZ`, `rhsZ` (Ginzburg–Landau RHS with Laplacian, Z₃ anisotropy, dissipation, phase rotation, defect core, soft pins), RK4 combine, `zToRawBonds`, `copyRawToX`, `projectBondOrdersSub`. |
| `DiracLattice_ocl.py` | **Model B solver** (current, lattice-based).  `DiracLatticeSolver` class: tight-binding π-electron propagation on `HoneycombGraph` with Kekulé-modulated hopping.  OpenCL RK4 for real-time evolution, numpy for exact diagonalization and LDOS. |
| `DiracLattice.cl` | **OpenCL kernels for Model B** (lattice).  Implements `rhs_psi` (Hamiltonian application: hopping + Kekulé modulation + defect onsite), RK4 intermediate and combine kernels. |
| `visualize.py` | **Visualization layer** (current).  `HexVisualizer` class with `plot_phase_hsv` (smooth Gaussian HSV phase field + zero-contour nodal lines + domain wall + color wheel), `plot_bond_order`, `plot_z_field`, `plot_domain_wall`, `plot_nodal_lines`, `plot_combined`.  `LiveViewer` class for real-time multi-panel animation with smooth interpolation. |

### Entry-point / demo scripts

| File | Role | Status |
|------|------|--------|
| `run_dirac_lattice.py` | **Main production script**.  Two-defect PAH: runs Model A with mirror-symmetric H⁺ defects, diagonalizes Model B, produces 5 figures (HSV phase + domain wall, bond distortion, spectrum, localized eigenstates, LDOS).  Supports `--phase-mode symmetric\|domain-wall`. | **Active, recommended** |
| `demo_proton.py` | **Interactive demo**: single or two H⁺ defects on a PAH, live 6-panel animation (amplitude, phase, bond order, HSV phase field + nodal lines, ring winding, closeup).  Saves final frame. | **Active** |
| `demo.py` | **Interactive demo**: basic Kekulé fluid relaxation on a rectangular patch (vortex/edge/relax experiments), live 4-panel animation. | **Active, simpler** |
| `demo_dirac_lattice.py` | **Interactive demo**: Model A relaxing then Model B wavepacket propagating, live side-by-side animation. | **Active** |
| `main.py` | **Batch experiment script**: vortex-antivortex pair, edge pinning, and relaxation experiments on rectangular patches.  Saves figures, no live animation. | **Active, older** |
| `run_combined.py` | **Comparison script**: side-by-side Model A (lattice) vs Model B (4-component Dirac on Cartesian grid).  Shows causal chain from bond orders to Dirac mass to quasiparticle density. | **Active, bridges both solver families** |

### Legacy / alternative solver family (4-component Dirac on Cartesian grid)

| File | Role | Status |
|------|------|--------|
| `hexgrid.py` | **Older honeycomb builder** using numpy dataclass (not `Graph.py`).  Provides `build_honeycomb_patch()`, `Vortex`, `init_vortex_phase()`, `init_vortex_phase_grid()`.  Used only by `Dirac4_ocl.py` and `run_dirac.py`. | **Legacy, redundant with `Graph.py`** |
| `Dirac4_ocl.py` | **4-component Dirac solver** on a 2D Cartesian grid with periodic boundaries.  Solves `i∂Ψ/∂t = [−ivF(αₓ∂ₓ + αᵧ∂ᵧ) + Δ_R M₁ + Δ_I M₂ + V]Ψ` using RK4.  The Kekulé mass Δ(x,y) is prescribed from vortices or interpolated from a honeycomb graph. | **Legacy, alternative to `DiracLattice_ocl.py`** |
| `Dirac4.cl` | **OpenCL kernels for 4-component Dirac**.  Implements the 4-spinor Hamiltonian on a Cartesian grid. | **Legacy, alternative to `DiracLattice.cl`** |
| `run_dirac.py` | **Demo for 4-component Dirac**: vortex-antivortex pair, Gaussian wavepacket propagation, bilinear response S_K. | **Legacy, alternative to `run_dirac_lattice.py`** |
| `plotting.py` | **Older visualization layer** for the 4-component Dirac family.  Provides `plot_honeycomb`, `plot_dirac_fields`, `plot_combined`, `DiracAnimator`. | **Legacy, redundant with `visualize.py`** |

## Redundancy Analysis

There are **two parallel solver families** in this codebase:

1. **Current family** (recommended): `Graph.py` → `ModelA.py` + `kekule_fluid.cl` → `DiracLattice_ocl.py` + `DiracLattice.cl` → `visualize.py`.  Everything operates on the same `HoneycombGraph` object.  The Dirac solver is a **lattice tight-binding** model (one orbital per atom).

2. **Legacy family**: `hexgrid.py` → `Dirac4_ocl.py` + `Dirac4.cl` → `plotting.py`.  The Dirac solver is a **4-component continuum Dirac** model on a Cartesian grid with periodic boundaries.  The Kekulé mass is interpolated from the honeycomb graph to the Cartesian grid.

**The families are not interchangeable**: the lattice solver (`DiracLattice_ocl.py`) works directly on the honeycomb graph and shares the OpenCL context with Model A, while the 4-component solver (`Dirac4_ocl.py`) works on a separate Cartesian grid and requires interpolation.

**Recommendation**: The current family (`Graph.py` / `ModelA.py` / `DiracLattice_ocl.py` / `visualize.py`) is the actively developed and recommended path.  The legacy family (`hexgrid.py` / `Dirac4_ocl.py` / `Dirac4.cl` / `plotting.py` / `run_dirac.py`) is kept for reference and comparison but is not actively maintained.

## Physics Summary

### Model A: Kekulé Fluid

The complex Kekulé order parameter `z_i` on each atom encodes the local
dimerization pattern.  It evolves according to a modified Ginzburg–Landau
equation:

```
∂z/∂t = κ·∇²z + r·z − u·|z|²·z − λ·z*²  +  (dissipative + conservative terms)
```

- `κ` — spatial stiffness (graph Laplacian)
- `r < 0` — spontaneous Kekulé ordering
- `u` — cubic saturation
- `λ` — Z₃ anisotropy (locks phase to 0, 2π/3, 4π/3)
- `η` — dissipative relaxation
- `Ω` — conservative phase rotation

**Boundary conditions**: H⁺ defects set `targetVal=0` (removing the π electron)
and pin the Kekulé phase to one of the three Kekulé patterns.  The defect core
is suppressed by making the effective `r` positive at defect sites.

**Valence constraint**: after each RK4 step, bond orders `x_ij` are projected
so that `Σ_j x_ij = targetVal_i` using a red-black Gauss-Seidel sweep with
ABBA ordering for mirror symmetry.

### Model B: Tight-Binding Quasiparticle

The π-electron Hamiltonian is:

```
H_{ij} = t₀ + δt·(x_ij − x₀_ij)    (nearest-neighbor hopping)
H_{ii} = V_def · defect_i            (onsite potential)
```

- `t₀` — bare hopping (~1.0)
- `δt` — Kekulé modulation strength (~0.5)
- `x₀_ij` — aromatic baseline (bondBase)
- `V_def` — defect onsite potential (~20, >> bandwidth)

The Kekulé modulation opens a mass gap at the Dirac points.  Vortices in the
Kekulé texture (where `|z|→0`) create topologically protected in-gap states.

## Usage

```bash
# Main production run (two H⁺ defects, symmetric pins)
python run_dirac_lattice.py --phase-mode symmetric

# Domain-wall mode (phase-mismatched pins)
python run_dirac_lattice.py --phase-mode domain-wall

# Interactive single-defect demo
python demo_proton.py --n-shells 2 --steps 600 --save --pin-dir 0

# Interactive basic demo
python demo.py

# Interactive Model A + Model B demo
python demo_dirac_lattice.py

# Batch experiments
python main.py

# Combined Model A + 4-component Dirac comparison
python run_combined.py
```

## Dependencies

- `numpy`
- `pyopencl` (requires OpenCL runtime)
- `matplotlib`
