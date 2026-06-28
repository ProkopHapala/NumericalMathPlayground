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

## Output

All figures are saved to `Quadrature_Images/`.

## Dependencies

- `numpy`
- `scipy` (for `scipy.linalg.lstsq` in weight optimization)
- `matplotlib`
- `pyscf` (for `demo_pyscf_grids.py` and `demo_atomic_radial.py`)
