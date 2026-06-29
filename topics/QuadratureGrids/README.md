# QuadratureGrids

Numerical integration grids for quantum chemistry — 2D prototypes for
CubeEmbededAtoms, triangular/wedge grids, and PySCF DFT grid visualization.

## Architecture

Three-layer separation per [AGNETs.md](../../AGNETs.md):

- **Core modules** — pure math, no I/O, no plotting
- **Plotting modules** — all visualization, no math
- **Demo scripts** — thin CLI wrappers that call core + plotting

```
Core modules                Demo scripts              Plotting modules
────────────────            ──────────────            ─────────────────
CubeEmbededGrid.py    ←──   demo_cube_embeded.py  ──→ GridPlotting.py
BasisFunctions.py     ←──      │
Symmetry.py           ←──      │
WeightOptimizer.py    ←──      │
TriGrids.py           ←──   demo_tri_grids.py     ──→ GridPlotting.py
PySCF (external)      ←──   demo_pyscf_grids.py   ──→ PySCFGridPlotting.py
BasisFunctions.py     ←──   demo_atomic_radial.py ──→ PySCFGridPlotting.py
```

## Core Modules

### `CubeEmbededGrid.py`
2D prototype of the CubeEmbededAtoms integration grid. Embeds a
radial wedge/sphere-blended grid inside a Cartesian cutout region,
seamlessly connecting to the outer Cartesian grid.

- `build_grid(d, n, alpha, n_blend)` — build the blended grid (wedge → sphere interpolation), returns points, cell areas, aspect ratios, and metadata
- `build_cartesian_outer(h, d, margin)` — Cartesian grid points outside the cutout `[-h,h]²`
- `build_combined_grid(grid_dict, d, margin)` — merge inner cell-center points with outer Cartesian into a single integration grid
- `collect_unique_points(grid_dict)` — deduplicate grid points (center point shared by all wedges)
- Helpers: `smoothstep`, `smootherstep`, `radial_map`, `quad_area`

### `BasisFunctions.py`
Gaussian basis set evaluation, analytic 2D integrals, and PySCF radial
wavefunction extraction.

- STO-nG basis data: `STO3G_1S`, `STO6G_1S`, `STO3G_2P_C`
- Orbital evaluation: `gaussian_1s(r2, basis, zeta)`, `gaussian_2p(x, y, basis, zeta)`, `gaussian_2py(...)`
- Analytic integrals: `analytic_1s_norm_sq(basis, zeta)`, `analytic_2p_norm_sq(basis, zeta)`
- Test function generation: `eval_test_func(xy, kind, zeta, x0, y0)`, `build_test_function_set()`
- PySCF radial extraction: `get_radial_functions(mol, r_grid)` — evaluates AOs along z-axis, factors out angular part to get R(r)
- `compute_radial_extent(r_grid, R)` — find where |R(r)| drops below 1% of max

### `TriGrids.py`
Triangular and wedge grid generation using barycentric coordinates.

- `barycentric_grid(n, verts)` — generate barycentric lattice of order n
- `subdivide_triangles(n, idx)` — triangulation of the barycentric grid
- `general_tri_grid(nu, nv, verts)` — generalized triangular grid with independent subdivisions along two edges (nu ≠ nv), handles diagonal nodes
- `wedge_grid(n, m, v0, verts)` — clipped triangle (wedge) grid: clips layers from a barycentric grid and rescales radial coordinates, symmetric about x=y
- `barycentric_lagrange_basis(eval_pts, n, verts)` — evaluate all Lagrange basis functions on the barycentric grid at arbitrary points

### `Symmetry.py`
Point group symmetry operations for reducing quadrature weight variables.

- `c4v_orbit(x, y)` — generate unique (x,y) points under C4v symmetry (4-fold rotation + mirror)
- `group_orbits(points, symmetry_fn)` — group grid points into symmetry orbits, returns orbit IDs, member lists, and representatives

### `WeightOptimizer.py`
Quadrature weight optimization for the CubeEmbededAtoms grid using
Tikhonov-regularized least squares.

- `geometric_weights(grid_dict)` — compute initial weights w0 from surrounding quad areas
- `build_integration_matrix(grid_pts, orbit_members, test_funcs)` — build matrix A[k, orbit] = Σ f_k(x_p)
- `compute_targets(test_funcs, integrals_total, outer_xy, outer_w, h)` — compute b_inner = I_total - I_outer
- `tikhonov_optimize(A, b, w0_orbit, lambdas, q)` — run λ sweep, return per-λ results (weights, errors)
- `expand_orbit_weights(w_orbit, orbit_members, Npts)` — expand per-orbit to per-point weights
- `validate_sto3g(grid_pts, w_inner, outer_xy, outer_w)` — validate against STO-3G orbital densities (not in training set)
- `select_best_lambda(results, w0_orbit)` — pick λ with stable weights and lowest error

### `OctagonGrid.py`
Octagon-shaped radial insert grid using Treutler-Ahlrichs M4 (TA-M4) radial
mapping, snapped to an octagon boundary that fits exactly in a 4×4 Cartesian
grid patch.

- `build_octagon_ta_grid(d, n, n_ta, xi)` — build the octagon grid with TA-M4 radial shells. Inner shells are uniform circles; outermost shell is snapped to octagon boundary points. Returns points, mesh data, TA-M4 weights, and diagnostics.
- `treutler_ahlrichs_radial(N, xi)` — TA-M4 radial quadrature: Chebyshev 2nd kind nodes with Jacobian-weighted integration. Returns radial points and weights `w_r = r * dr`.
- `build_full_mesh(grid_dict, margin)` — extract vertex points, polygons, and region labels (0=inner octagon, 1=interface, 2=outer Cartesian).
- `classify_nodes(nodes, polys, regions, grid_dict)` — classify vertices as `inner`, `near_interface`, `interface`, or `outer`.
- `compute_ta_weights(grid_dict)` — compute per-cell TA-M4 quadrature weights.
- `mesh_diagnostics(nodes, polys, regions, grid_dict)` — angular uniformity, edge ratios, aspect ratios, chirality, strain.

**Grid geometry**: The octagon is a 4×4 Cartesian patch with 4 corner triangles cut (each ½d²), giving area = 16d² - 4×½d² = **14d²**. With d=0.2Å (0.378 Bohr), n=4: h=2d=0.756 Bohr, hs=d=0.378 Bohr.

### `demo_octagon_ta.py`
Octagon grid construction, mesh visualization, and **interface weight optimization**.

```
python demo_octagon_ta.py    # generates mesh plots + optimization results
```

#### Interface Weight Optimization

The grid has three regions with different weight sources:
- **Inner** (TA-M4 radial quadrature, shells 1..n_ta-1): weights = `w_ta[j] * (2π/Nu)`, renormalized so inner + interface geometric = exact octagon area.
- **Interface** (snapped octagon boundary layer, 16 points): **optimized** via Tikhonov-regularized least squares with C4v symmetry (3 orbits: 4+8+4 points).
- **Outer** (Cartesian grid, d² per point): uniform weights with d²/2 on domain edges and d²/4 on corners. Octagon boundary points get partial weights via Monte Carlo cell-fraction estimation.

**Weight normalization** (reality check before optimization):
- TA-M4 weights are renormalized: `2π * Σ_{j=0}^{n_ta-2} w_ta[j] = octagon_area - w_iface_geom`
- Outer grid: `Σ w_outer = total_area - octagon_area` (achieved via partial boundary weights, not global scaling)
- All weight sums match areas to <0.01d²

**Optimization setup**:
- 46 Gaussian test functions (ζ=0.5..12.0, off-center, x²G variants) with analytic integrals
- Area constraint weighted 1000× (const function)
- Tikhonov regularization: λ sweep over [1e-6 .. 1e10], best λ selected by minimum mean error
- C4v symmetry: 3 free orbit weights instead of 16 individual weights

**Results** (d=0.2Å, n=4, n_ta=8, 46 test functions):

| Metric | Baseline (geometric) | Optimized (λ=1.0) |
|--------|---------------------|-------------------|
| Mean error | 7.42% | **5.90%** |
| Max error | 21.68% | 21.38% |
| const (area) | 0.19% | **0.00%** |
| G(ζ=0.5) | 4.37% | **0.07%** |
| G(ζ=1.0) | 9.62% | **0.18%** |
| G(ζ=1.5) | 6.13% | **1.91%** |
| G(ζ=2.0) | 3.54% | 7.34% |
| G(ζ=5.0) | 18.13% | 18.13% |

**Key findings**:
- Optimization dramatically improves broad/medium Gaussians (ζ≤1.5): errors reduced to <2%
- Narrow Gaussians (ζ≥5.0) remain at ~18% — this is the **TA-M4 radial quadrature limit** with only 8 shells, not an interface weight problem
- The area constraint is satisfied exactly after optimization
- Only 3 free parameters (C4v orbits) are sufficient to improve broad Gaussian integration by 5-10×

**Orbit weights** (w0 → w_opt):
- Orbit 0 (edge midpoints, 4 pts): 0.0495 → -0.0062
- Orbit 1 (edge quarters, 8 pts): 0.0584 → 0.0670
- Orbit 2 (corners, 4 pts): 0.0000 → 0.0380

## Plotting Modules

### `GridPlotting.py`
All visualization for triangular, wedge, and CubeEmbededAtoms grids.

- Triangular: `plot_tri_grid`, `compare_tri_levels`, `plot_general_tri_grid`, `compare_general_tri_grids`
- Wedge: `plot_wedge_grid`, `compare_wedge_grids`
- CubeEmbeded: `plot_cube_full_view`, `plot_cube_cell_areas`, `plot_cube_radial_profile`, `plot_cube_three_grids`, `plot_blend_weight`
- Orbital/density: `scatter_plot`, `plot_orbitals_on_grid`, `plot_density_overlay`, `plot_convergence`
- Weight optimization: `plot_weight_comparison`, `plot_weight_errors`, `plot_lambda_sweep`

### `PySCFGridPlotting.py`
Visualization for PySCF DFT integration grids and atomic radial wavefunctions.

- Multi-atom: `plot_3d_scatter`, `plot_2d_projections`, `plot_radial_distribution`, `plot_grid_count_vs_level`
- Single-atom: `plot_single_atom_3d`, `plot_single_atom_2d`, `plot_single_atom_radial`, `plot_lebedev_shells`
- Radial wavefunctions: `plot_radial_wavefunctions`, `plot_radial_wavefunctions_normalized`, `plot_radial_density`

## Demo Scripts

### `demo_tri_grids.py`
Triangular and wedge grid visualization.

```
python demo_tri_grids.py --mode barycentric   # Pascal-triangle subdivision levels
python demo_tri_grids.py --mode general       # nu ≠ nv generalized grids
python demo_tri_grids.py --mode wedge         # clipped triangle (wedge) grids
python demo_tri_grids.py --mode lagrange      # partition-of-unity test
python demo_tri_grids.py --mode all           # all of the above
```

### `demo_cube_embeded.py`
CubeEmbededAtoms integration grid: construction, orbital integration,
weight optimization, and convergence tests.

```
python demo_cube_embeded.py --mode grid         # grid structure, cell areas, radial profiles
python demo_cube_embeded.py --mode integration  # STO-nG orbital integration accuracy
python demo_cube_embeded.py --mode weights      # Tikhonov weight optimization + λ sweep
python demo_cube_embeded.py --mode convergence  # error vs grid resolution
python demo_cube_embeded.py --mode all          # all of the above

# Optional parameters:
#   --d 0.5 --n 10 --alpha 1.8 --n-blend 4
```

### `demo_pyscf_grids.py`
PySCF DFT real-space integration grid visualization.

```
python demo_pyscf_grids.py --mode multi   # H2O molecule: 3D, 2D projections, radial dist
python demo_pyscf_grids.py --mode single  # O atom: 3D, 2D, radial shells, Lebedev shells
python demo_pyscf_grids.py --mode all     # both
```

### `demo_atomic_radial.py`
Plot radial wavefunctions R(r) for atomic orbitals from PySCF basis sets.

```
python demo_atomic_radial.py                           # default: H,C,N,O with cc-pVDZ
python demo_atomic_radial.py --basis aug-cc-pVTZ       # different basis set
python demo_atomic_radial.py --elements H,C,O          # subset of elements
python demo_atomic_radial.py --rmax 10 --nr 2000       # radial range and resolution
```

Produces 3 figures: raw R(r), normalized R(r)/R_max, and radial density r²|R(r)|².
Also prints a table of radial extents per orbital.

### `demo_octagon_ta.py`
Octagon grid with TA-M4 radial insert: mesh construction, interface weight
optimization with C4v symmetry, and error diagnostics. See the
[Interface Weight Optimization](#interface-weight-optimization) section above
for detailed results.

```
python demo_octagon_ta.py    # mesh sweep + interface optimization + diagnostics
```

## Output

All figures are saved to `Quadrature_Images/`.

## Dependencies

- `numpy`
- `scipy` (for `scipy.linalg.lstsq` in weight optimization)
- `matplotlib`
- `pyscf` (for `demo_pyscf_grids.py` and `demo_atomic_radial.py`)
