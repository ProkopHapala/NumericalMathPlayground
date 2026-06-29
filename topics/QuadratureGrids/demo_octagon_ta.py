"""
demo_octagon_ta.py — Visualize octagon grid with TA-M4 radial insert.

Shows:
1. TA-M4 radial point distribution vs power-law
2. Full mesh with TA-M4 inner grid for different n_ta
3. Interface classification (inner=TA-M4, interface=seam, outer=Cartesian)
4. Zoom on the TA-M4 / transition boundary

Usage:
    python demo_octagon_ta.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

from OctagonGrid import (
    build_octagon_ta_grid, build_full_mesh,
    treutler_ahlrichs_radial,
    extract_vertex_points_classified, extract_cell_center_points_classified,
    compute_ta_weights, classify_nodes, classify_polys,
    mesh_diagnostics,
)
from BasisFunctions import build_test_function_set, eval_test_func
from Symmetry import group_orbits
from CubeEmbededGrid import build_cartesian_fixed

IMG_DIR = "Quadrature_Images"


BOHR_TO_ANG = 0.529177  # 1 Bohr = 0.529 Å


def plot_radial_comparison(ax, n_ta=15, xi=1.0, h=0.5):
    """Compare TA-M4 radial distribution vs power-law (in Bohr)."""
    r_ta, w_ta = treutler_ahlrichs_radial(n_ta, xi=xi)
    r_ta_scaled = r_ta / r_ta[-1] * h

    # Power law for comparison
    t = np.linspace(0, 1, n_ta + 1)
    r_power = h * t**1.8

    ax.plot(np.arange(1, n_ta + 1), r_ta_scaled * BOHR_TO_ANG, 'o-',
            label='TA-M4', color='steelblue', markersize=5)
    ax.plot(np.arange(1, n_ta + 2), r_power * BOHR_TO_ANG, 's-',
            label='Power α=1.8', color='coral', markersize=4)
    ax.set_xlabel('Shell index')
    ax.set_ylabel('r (Å)')
    ax.set_title(f'Radial point distribution (N={n_ta}, h={h:.2f} Bohr)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_mesh(nodes, polys, regions, grid_dict, ax=None, title=None,
              show_points=False, zoom=None, show_ta_boundary=True):
    """Plot mesh with regions color-coded."""
    p = grid_dict['params']
    h = p['h']
    R_cut = p.get('R_cut', None)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    colors = {0: '#4ECDC4', 1: '#FFA07A', 2: '#D3D3D3'}
    for k, (poly, reg) in enumerate(zip(polys, regions)):
        pts = nodes[poly]
        closed = np.vstack([pts, pts[0]])
        ax.fill(closed[:, 0], closed[:, 1], color=colors.get(int(reg), 'white'),
                edgecolor='k', linewidth=0.3, alpha=0.6)

    if show_ta_boundary and R_cut is not None:
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(R_cut * np.cos(theta), R_cut * np.sin(theta), 'b--', lw=1.5, alpha=0.5, label=f'R_cut={R_cut:.2f}')

    oct_v = grid_dict['oct_verts']
    oct_closed = np.vstack([oct_v, oct_v[0]])
    ax.plot(oct_closed[:, 0], oct_closed[:, 1], 'k-', lw=1.5)
    sq = np.array([[-h, -h], [h, -h], [h, h], [-h, h], [-h, -h]])
    ax.plot(sq[:, 0], sq[:, 1], 'k--', lw=1.0, alpha=0.4)

    if show_points:
        ax.plot(nodes[:, 0], nodes[:, 1], '.', color='k', markersize=1.5, alpha=0.5)

    ax.set_aspect('equal')
    if zoom:
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[2], zoom[3])
    else:
        margin = 1.3 * h
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
    ax.set_title(title or '')
    ax.legend(fontsize=7, loc='upper right')
    ax.axis('off')


def plot_classified(pts, w, labels, grid_dict, ax=None, title=None):
    """Plot quadrature points colored by classification."""
    p = grid_dict['params']
    h = p['h']
    R_cut = p.get('R_cut', None)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    color_map = {'inner': 'steelblue', 'near_interface': 'mediumpurple',
                 'interface': 'coral', 'outer': 'gray'}
    for lbl in ['inner', 'near_interface', 'interface', 'outer']:
        mask = labels == lbl
        if mask.any():
            ax.scatter(pts[mask, 0], pts[mask, 1], c=color_map[lbl], s=20,
                       edgecolors='none', alpha=0.8, label=f"{lbl} ({mask.sum()})")

    if R_cut is not None:
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(R_cut * np.cos(theta), R_cut * np.sin(theta), 'b--', lw=1.0, alpha=0.4)

    oct_v = grid_dict['oct_verts']
    oct_closed = np.vstack([oct_v, oct_v[0]])
    ax.plot(oct_closed[:, 0], oct_closed[:, 1], 'k-', lw=1.0, alpha=0.5)
    sq = np.array([[-h, -h], [h, -h], [h, h], [-h, h], [-h, -h]])
    ax.plot(sq[:, 0], sq[:, 1], 'k--', lw=0.8, alpha=0.3)

    ax.set_aspect('equal')
    margin = 1.2 * h
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_title(title or f"N={len(pts)}  Σw={w.sum():.3f}")
    ax.legend(loc='upper right', fontsize=8)
    ax.axis('off')
    return ax


def optimize_interface_weights(grid_dict, nodes, polys, regions, d, h,
                                n_ta=8, lambdas=None, margin=8):
    """Optimize interface weights using Gaussian test functions + C4v symmetry.

    Setup:
      - Inner radial grid (TA-M4): FIXED weights from Chebyshev quadrature.
          Each point (i,j) for j=1..n_ta-1 gets w_ta[j-1] * (2π/Nu).
          Center point (j=0) gets weight 0 (not a TA-M4 quadrature point).
      - Outer Cartesian grid: FIXED d² weights, covering [-R_cut, R_cut]².
          R_cut is computed from test functions (large enough for Gaussian tails).
          Points inside octagon are removed (no double-counting).
      - Interface points (snapped octagon layer + corner shared nodes):
          OPTIMIZED via Tikhonov regularized least squares with C4v symmetry.

    Total integral = I_inner(fixed) + I_interface(optimized) + I_outer(fixed) ≈ I_total

    Returns dict with optimized weights, errors, and diagnostics.
    """
    from OctagonGrid import _poly_area

    if lambdas is None:
        lambdas = [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1e4, 1e6, 1e8, 1e10]

    p = grid_dict['params']
    Nu = p['Nu']
    Nv = p['Nv']
    hs = p['hs']
    r_vals = grid_dict['r_vals']
    w_ta = grid_dict['w_ta']  # (n_ta,) — already rescaled by scale_r²

    R_oct = p['R_oct']

    # Compute mesh-based geometric weights for interface w0 (region 0 only)
    w_geom = np.zeros(len(nodes))
    for k_poly, poly in enumerate(polys):
        if int(regions[k_poly]) != 0:
            continue
        area = _poly_area(nodes, poly)
        n_vert = len(poly)
        for vi in poly:
            w_geom[vi] += area / n_vert

    # Classify nodes
    lbl_v = classify_nodes(nodes, polys, regions, grid_dict)
    iface_mask_early = (lbl_v == 'interface')

    # Octagon area from mesh
    inner_area = sum(_poly_area(nodes, poly) for k, poly in enumerate(polys)
                     if int(regions[k]) == 0)

    # === Renormalize TA-M4 weights ===
    # TA-M4 is designed for open interval, so raw weights don't match the
    # finite octagon area. We only use shells j=1..n_ta-1 (j=n_ta is interface).
    # Renormalize so: 2π * Σ_{j=0}^{n_ta-2} w_ta[j] = inner_area - w_iface_geom
    w_iface_geom_total = w_geom[iface_mask_early].sum()
    target_inner = inner_area - w_iface_geom_total
    current_inner = 2 * np.pi * w_ta[:n_ta - 1].sum()
    ta_scale = target_inner / current_inner
    w_ta_scaled = w_ta * ta_scale

    # === Test functions ===
    # Constant integrates to total area (2*R_cut)²
    # (set after R_cut is known)
    test_funcs = []
    test_integrals = []
    test_funcs.append(('const', ('const', 0, 0, 0)))
    test_integrals.append(None)  # will be set to total_area below
    for zeta in [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]:
        test_funcs.append((f'G(ζ={zeta})', ('gauss', zeta, 0.0, 0.0)))
        test_integrals.append(np.pi / zeta**2)
    for zeta in [1.0, 2.0, 3.0]:
        for (x0, y0) in [(0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.5, 0.0),
                         (0.1, 0.1), (0.2, 0.2), (0.3, 0.1), (0.5, 0.5)]:
            test_funcs.append((f'G(ζ={zeta},@({x0},{y0}))', ('gauss', zeta, x0, y0)))
            test_integrals.append(np.pi / zeta**2)
    for zeta in [1.0, 2.0, 3.0, 5.0]:
        for (x0, y0) in [(0.0, 0.0), (0.2, 0.0), (0.5, 0.0)]:
            test_funcs.append((f'x²G(ζ={zeta},@({x0},{y0}))', ('x2g', zeta, x0, y0)))
            test_integrals.append(np.pi / (2 * zeta**4))

    Ntest = len(test_funcs)

    # === Classify nodes (already done above) ===
    # lbl_v computed during renormalization

    # === Assign TA-M4 weights to inner nodes ===
    # Per-point weight = w_ta_scaled[j-1] * (2π/Nu)
    angular_w = 2 * np.pi / Nu
    w_ta_per_point = w_ta_scaled * angular_w

    # Map each node to its TA-M4 weight by matching radius
    w_inner = np.zeros(len(nodes))
    for idx in range(len(nodes)):
        r_i = np.linalg.norm(nodes[idx])
        if r_i < 1e-12:
            w_inner[idx] = 0.0
        else:
            for j in range(1, n_ta):  # j=1..n_ta-1 are inner (not snapped)
                if abs(r_i - r_vals[j]) < 1e-8 * max(r_vals[j], 1):
                    w_inner[idx] = w_ta_per_point[j - 1]
                    break
            # j=n_ta (snapped layer) — not assigned here (interface)

    # === Build outer Cartesian grid ===
    # R_cut from test functions: R where exp(-zeta_min²*R²) < 1e-10
    zeta_min = min(params[1] for _, params in test_funcs
                   if params is not None and params[1] > 0)
    R_cut = np.sqrt(-np.log(1e-10) / zeta_min**2)

    n_side = int(round(R_cut / d))
    x_cart = np.arange(-n_side, n_side + 1) * d
    y_cart = np.arange(-n_side, n_side + 1) * d
    xx, yy = np.meshgrid(x_cart, y_cart)
    outer_xy = np.column_stack([xx.ravel(), yy.ravel()])
    outer_w = np.full(len(outer_xy), d * d)

    # Boundary weights: edge points get d²/2, corner points get d²/4
    # (only half/quarter of their cell is inside the integration domain)
    on_x_edge = (np.abs(np.abs(outer_xy[:, 0]) - n_side * d) < 1e-10)
    on_y_edge = (np.abs(np.abs(outer_xy[:, 1]) - n_side * d) < 1e-10)
    outer_w[on_x_edge & on_y_edge] = d * d / 4.0   # corners
    outer_w[on_x_edge ^ on_y_edge] = d * d / 2.0   # edges

    # Remove points inside octagon with proper partial weights at boundary
    # Octagon: |x| <= h, |y| <= h, |x|+|y| <= h+hs
    # Points strictly inside → removed (weight 0)
    # Points on octagon boundary → partial weight (fraction of cell outside octagon)
    oct_h = h
    oct_hs = hs
    d_step = d
    np.random.seed(42)
    n_mc = 10000
    keep = np.ones(len(outer_xy), dtype=bool)
    for idx in range(len(outer_xy)):
        cx, cy = outer_xy[idx]
        # Quick check: if clearly outside octagon, keep with full weight
        if np.abs(cx) > oct_h + 1e-10 or np.abs(cy) > oct_h + 1e-10 or \
           np.abs(cx) + np.abs(cy) > oct_h + oct_hs + 1e-10:
            continue
        # Point is inside octagon — compute fraction of cell inside
        sx = cx + (np.random.rand(n_mc) - 0.5) * d_step
        sy = cy + (np.random.rand(n_mc) - 0.5) * d_step
        frac_in = np.mean((np.abs(sx) <= oct_h) & (np.abs(sy) <= oct_h) & \
                          (np.abs(sx) + np.abs(sy) <= oct_h + oct_hs))
        if frac_in > 0.999:
            keep[idx] = False  # strictly inside, remove
        else:
            outer_w[idx] *= (1.0 - frac_in)  # partial weight outside octagon
    outer_xy = outer_xy[keep]
    outer_w = outer_w[keep]

    total_area = (2 * n_side * d) ** 2
    test_integrals[0] = total_area

    # === Evaluate test functions ===
    F_nodes = np.zeros((Ntest, len(nodes)))
    F_outer = np.zeros((Ntest, len(outer_xy)))
    for k, (label, params) in enumerate(test_funcs):
        if params is None:
            F_nodes[k] = 1.0
            F_outer[k] = 1.0
        else:
            kind, zeta, x0, y0 = params
            F_nodes[k] = eval_test_func(nodes, kind, zeta, x0, y0)
            F_outer[k] = eval_test_func(outer_xy, kind, zeta, x0, y0)

    # === Compute targets: b = I_total - I_outer - I_inner_fixed ===
    I_outer = F_outer @ outer_w
    I_total = np.array([iv if iv is not None else 0 for iv in test_integrals])

    # Interface mask: zero out interface weights for fixed contribution
    iface_mask = (lbl_v == 'interface')
    w_inner_fixed = w_inner.copy()
    w_inner_fixed[iface_mask] = 0.0
    I_inner_fixed = F_nodes @ w_inner_fixed

    # Area constraint (inner_area already computed above)

    b = I_total - I_outer - I_inner_fixed
    # Area constraint: total_area - I_outer - I_inner_fixed (for const function)
    b[0] = total_area - I_outer[0] - I_inner_fixed[0]

    # Odd functions: target = 0
    odd_mask = np.zeros(Ntest, dtype=bool)
    for k, (label, params) in enumerate(test_funcs):
        if params is not None and params[0] in ('px', 'xyg'):
            b[k] = 0.0
            odd_mask[k] = True

    # === Group interface points into C4v orbits ===
    iface_indices = np.where(iface_mask)[0]
    iface_pts = nodes[iface_indices]
    orbit_id, orbit_members_local, orbit_rep = group_orbits(iface_pts)
    N_orbits = len(orbit_rep)

    # Map local orbit members to global indices
    orbit_members_global = [[iface_indices[m] for m in members]
                            for members in orbit_members_local]

    # Geometric weights for interface (w0) — w_geom already computed above
    w0_iface = np.array([np.mean(w_geom[iface_indices[members]])
                         for members in orbit_members_local])
    orbit_sizes = np.array([len(m) for m in orbit_members_global])

    # === Reality check: weight sums vs areas ===
    oct_area = inner_area
    outer_area = total_area - oct_area

    w_inner_sum = w_inner_fixed.sum()
    w_outer_sum = outer_w.sum()
    w_iface_geom_sum = w0_iface @ orbit_sizes

    print(f"\n  === REALITY CHECK (weight sums vs areas) ===")
    print(f"  Total square area [-R_cut,R_cut]²: {total_area:.6f}")
    print(f"  Octagon area (mesh):               {oct_area:.6f}")
    print(f"  Expected outer area:               {outer_area:.6f}")
    print(f"  ---")
    print(f"  Σ w_inner (TA-M4, fixed):          {w_inner_sum:.6f}  (expect ≈ {oct_area:.6f})")
    print(f"  Σ w_outer (Cartesian d², fixed):   {w_outer_sum:.6f}  (expect ≈ {outer_area:.6f}")
    print(f"  Σ w_iface (geometric w0):          {w_iface_geom_sum:.6f}")
    print(f"  Σ w_inner + w_iface_geom:          {w_inner_sum + w_iface_geom_sum:.6f}  (expect ≈ {oct_area:.6f})")
    print(f"  Σ ALL (inner+iface+outer):         {w_inner_sum + w_iface_geom_sum + w_outer_sum:.6f}  (expect ≈ {total_area:.6f})")
    print(f"  ---")
    print(f"  Outer grid points: {len(outer_xy)}, d²={d*d:.6f}")

    # Per-function reality check (baseline = no optimization, interface=0)
    print(f"\n  Per-function baseline (interface weight=0):")
    print(f"    {'function':30s} {'I_inner':>10s} {'I_outer':>10s} {'I_num':>10s} {'I_exact':>10s} {'err%':>8s}")
    for k, (label, _) in enumerate(test_funcs):
        if k > 5 and k < Ntest - 5:
            continue
        I_in = I_inner_fixed[k]
        I_out = I_outer[k]
        I_num = I_in + I_out
        I_ex = I_total[k]
        if odd_mask[k] or abs(I_ex) < 1e-10:
            err_str = "  (odd/zero)"
        else:
            err_str = f"{abs(I_num - I_ex)/abs(I_ex)*100:8.2f}%"
        print(f"    {label:30s} {I_in:10.6f} {I_out:10.6f} {I_num:10.6f} {I_ex:10.6f} {err_str}")

    # === Build A matrix: A[k, orbit] = Σ_{p in orbit} F_nodes[k, p] ===
    A = np.zeros((Ntest, N_orbits))
    for oid, members in enumerate(orbit_members_global):
        A[:, oid] = F_nodes[:, members].sum(axis=1)

    # Area constraint row
    A[0] = orbit_sizes

    # Equation weights: area constraint gets very high weight
    q = np.ones(Ntest)
    q[0] = 1000.0

    # === Tikhonov optimization ===
    A_w = A * q[:, None]
    b_w = b * q

    valid = (np.abs(I_total) > 1e-10) & (~odd_mask)

    results = {}
    for lam in lambdas:
        A_aug = np.vstack([A_w, np.sqrt(lam) * np.eye(N_orbits)])
        b_aug = np.concatenate([b_w, np.sqrt(lam) * w0_iface])
        w_orbit, _, _, _ = lstsq(A_aug, b_aug)

        # Total integral: I_inner_full + I_outer
        w_full = w_inner_fixed + expand_weights(w_orbit, orbit_members_global, len(nodes))
        I_inner_total = F_nodes @ w_full
        I_total_num = I_inner_total + I_outer
        total_errs = np.zeros(Ntest)
        total_errs[valid] = np.abs(I_total_num[valid] - I_total[valid]) / np.abs(I_total[valid]) * 100

        results[lam] = {
            'w_orbit': w_orbit.copy(),
            'total_errs': total_errs,
            'total_mean_err': np.mean(total_errs[valid]),
            'total_max_err': np.max(total_errs[valid]),
        }

    best_lam = min(results.keys(), key=lambda l: results[l]['total_mean_err'])

    # Baseline: TA-M4 inner + geometric interface + outer (no optimization)
    w_baseline = w_inner.copy()
    w_baseline[iface_mask] = w_geom[iface_mask]
    I_inner_baseline = F_nodes @ w_baseline
    I_total_baseline = I_inner_baseline + I_outer
    baseline_errs = np.zeros(Ntest)
    baseline_errs[valid] = np.abs(I_total_baseline[valid] - I_total[valid]) / np.abs(I_total[valid]) * 100

    return dict(
        results=results,
        best_lam=best_lam,
        best_result=results[best_lam],
        w_opt=results[best_lam]['w_orbit'],
        w0_iface=w0_iface,
        orbit_members=orbit_members_global,
        orbit_rep=orbit_rep,
        N_orbits=N_orbits,
        N_iface=int(iface_mask.sum()),
        nodes=nodes,
        lbl_v=lbl_v,
        w_inner=w_inner,
        w_inner_fixed=w_inner_fixed,
        iface_mask=iface_mask,
        outer_xy=outer_xy,
        outer_w=outer_w,
        test_funcs=test_funcs,
        inner_area=inner_area,
        A=A,
        b=b,
        baseline_errs=baseline_errs,
        R_cut=R_cut,
    )


def expand_weights(w_orbit, orbit_members, Npts):
    """Expand per-orbit weights to per-point weights."""
    w_full = np.zeros(Npts)
    for oid, members in enumerate(orbit_members):
        for p_idx in members:
            w_full[p_idx] = w_orbit[oid]
    return w_full


def main():
    import os
    os.makedirs(IMG_DIR, exist_ok=True)

    # 4x4 embedding: octagon fits in 4x4 patch of outer Cartesian grid
    # n=4 → h=2d, octagon diameter = 4d
    # With d=0.2 Å: h=0.4 Å, octagon diameter=0.8 Å (< 1 Å, no overlap)
    d_ang = 0.20  # Ångström
    d = d_ang / BOHR_TO_ANG  # convert to Bohr
    n = 4         # 4x4 embedding
    h = n * d / 2  # = 2d

    print(f"Grid: d={d_ang:.2f} Å = {d:.4f} Bohr, n={n} (4x4 embedding)")
    print(f"  h={h:.4f} Bohr = {h*BOHR_TO_ANG:.3f} Å")
    print(f"  Octagon diameter = {2*h*BOHR_TO_ANG:.3f} Å = {2*h:.4f} Bohr")

    # === Fig 1: TA-M4 radial distribution vs power-law ===
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    plot_radial_comparison(ax1, n_ta=15, xi=1.0, h=h)
    fig1.tight_layout()
    fig1.savefig(f"{IMG_DIR}/octagon_ta_radial.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {IMG_DIR}/octagon_ta_radial.png")

    # === Fig 2: Full mesh for different n_ta values ===
    n_ta_values = [3, 5, 8, 12, 15]
    fig2, axes2 = plt.subplots(1, len(n_ta_values), figsize=(5 * len(n_ta_values), 5))
    if len(n_ta_values) == 1:
        axes2 = [axes2]

    print("\n=== TA-M4 mesh for different n_ta ===")
    for ax, n_ta in zip(axes2, n_ta_values):
        grid_dict = build_octagon_ta_grid(d=d, n=n, n_ta=n_ta, xi=1.0)
        nodes, polys, regions = build_full_mesh(grid_dict, margin=3)
        p = grid_dict['params']
        n_q = sum(1 for pp in polys if len(pp) == 4)
        n_t = sum(1 for pp in polys if len(pp) == 3)
        diag = mesh_diagnostics(nodes, polys, regions, grid_dict)
        print(f"  n_ta={n_ta:2d}: Nv={p['Nv']}, R_cut={p['R_cut']:.4f} Bohr "
              f"({p['R_cut']*BOHR_TO_ANG:.3f} Å), "
              f"nodes={len(nodes):4d}, polys={len(polys):4d} (quads={n_q}, tris={n_t})")
        print(f"    diag: ang_dev={diag['angular_uniformity_deg']:.2f}°, "
              f"edge_ratio={diag['edge_ratio_inner']:.2f}, "
              f"aspect={diag['aspect_max']:.2f} (mean={diag['aspect_mean']:.2f}), "
              f"chirality={diag['chirality']:.4f}, "
              f"strain={diag['strain']:.4f}")
        plot_mesh(nodes, polys, regions, grid_dict, ax=ax,
                  title=f"n_ta={n_ta}  Nv={p['Nv']}", show_points=False)
    fig2.suptitle(f"TA-M4 octagon mesh (h={h:.2f} Bohr = {h*BOHR_TO_ANG:.3f} Å, snapped)",
                  fontsize=14)
    fig2.tight_layout()
    fig2.savefig(f"{IMG_DIR}/octagon_ta_mesh_sweep.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {IMG_DIR}/octagon_ta_mesh_sweep.png")

    # === Fig 3: Interface classification (n_ta=15) ===
    grid_dict = build_octagon_ta_grid(d=d, n=n, n_ta=8, xi=1.0)
    nodes, polys, regions = build_full_mesh(grid_dict, margin=5)
    p = grid_dict['params']
    h = p['h']
    R_cut = p['R_cut']

    pts_v, w_v, lbl_v = extract_vertex_points_classified(nodes, polys, regions, grid_dict)
    pts_c, w_c, lbl_c = extract_cell_center_points_classified(nodes, polys, regions, grid_dict)

    n_inner_v = (lbl_v == 'inner').sum()
    n_near_v = (lbl_v == 'near_interface').sum()
    n_iface_v = (lbl_v == 'interface').sum()
    n_outer_v = (lbl_v == 'outer').sum()
    n_inner_c = (lbl_c == 'inner').sum()
    n_near_c = (lbl_c == 'near_interface').sum()
    n_iface_c = (lbl_c == 'interface').sum()
    n_outer_c = (lbl_c == 'outer').sum()

    print(f"\n=== n_ta=8 interface classification ===")
    print(f"  R_cut = {R_cut:.4f} Bohr ({R_cut*BOHR_TO_ANG:.4f} Å), "
          f"h = {h:.4f} Bohr ({h*BOHR_TO_ANG:.4f} Å)")
    print(f"  Vertex mode:     {len(pts_v)} points (inner={n_inner_v}, near={n_near_v}, interface={n_iface_v}, outer={n_outer_v})")
    print(f"  Cell-center mode: {len(pts_c)} points (inner={n_inner_c}, near={n_near_c}, interface={n_iface_c}, outer={n_outer_c})")
    print(f"  Free parameters (interface): vertex={n_iface_v}, cell-center={n_iface_c}")
    print(f"  Near-interface (adjacent to snap): vertex={n_near_v}, cell-center={n_near_c}")
    print(f"  Σw: vertex={w_v.sum():.4f}, cell-center={w_c.sum():.4f}")

    # Mesh diagnostics
    diag = mesh_diagnostics(nodes, polys, regions, grid_dict)
    print(f"\n  Mesh diagnostics:")
    print(f"    angular deviation from uniform: {diag['angular_uniformity_deg']:.2f}°")
    print(f"    edge length ratio (max/min):    {diag['edge_ratio_inner']:.2f}")
    print(f"    cell aspect ratio:              max={diag['aspect_max']:.2f}, mean={diag['aspect_mean']:.2f}")
    print(f"    area ratio (max/min):           {diag['area_ratio_inner']:.2f}")
    print(f"    chirality (C4v asymmetry):      {diag['chirality']:.6f}")
    print(f"    strain (circle vs uniform):     {diag['strain']:.6f}")

    # TA-M4 weights
    w_ta_cells = compute_ta_weights(grid_dict)
    valid = ~np.isnan(w_ta_cells[0])
    n_ta_shells = valid.sum()
    print(f"  TA-M4 shells: {n_ta_shells} (weights from Chebyshev quadrature)")
    if n_ta_shells > 0:
        print(f"    TA-M4 cell weights: {[f'{w_ta_cells[0,j]:.6f}' for j in range(min(5, n_ta_shells))]}")

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 7))
    plot_classified(pts_v, w_v, lbl_v, grid_dict, ax=ax3a,
                    title=f"Vertex mode  N={len(pts_v)}  (interface={n_iface_v})")
    plot_classified(pts_c, w_c, lbl_c, grid_dict, ax=ax3b,
                    title=f"Cell-center mode  N={len(pts_c)}  (interface={n_iface_c})")
    fig3.suptitle(f"TA-M4 interface classification (n_ta=8, R_cut={R_cut:.3f} Bohr, Nu={p['Nu']})",
                  fontsize=14)
    fig3.savefig(f"{IMG_DIR}/octagon_ta_interface.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {IMG_DIR}/octagon_ta_interface.png")

    # === Fig 4: Zoom on TA-M4 / transition boundary (NE quadrant) ===
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))
    zoom = [-0.5 * h, 1.3 * h, -0.5 * h, 1.3 * h]

    color_map = {'inner': 'steelblue', 'near_interface': 'mediumpurple',
                 'interface': 'coral', 'outer': 'gray'}
    for ax, pts, w, lbl, title in [(ax4a, pts_v, w_v, lbl_v, 'Vertex mode'),
                                   (ax4b, pts_c, w_c, lbl_c, 'Cell-center mode')]:
        for l in ['inner', 'near_interface', 'interface', 'outer']:
            mask = lbl == l
            if mask.any():
                ax.scatter(pts[mask, 0], pts[mask, 1], c=color_map[l], s=40,
                           edgecolors='k', linewidths=0.3, alpha=0.9, label=l)
        # Draw R_cut circle
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(R_cut * np.cos(theta), R_cut * np.sin(theta), 'b--', lw=1.5, alpha=0.5, label=f'R_cut={R_cut:.2f}')
        # Octagon
        oct_v = grid_dict['oct_verts']
        oct_closed = np.vstack([oct_v, oct_v[0]])
        ax.plot(oct_closed[:, 0], oct_closed[:, 1], 'k-', lw=1.5)
        sq = np.array([[-h, -h], [h, -h], [h, h], [-h, h], [-h, -h]])
        ax.plot(sq[:, 0], sq[:, 1], 'k--', lw=1.0, alpha=0.4)
        ax.set_aspect('equal')
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[2], zoom[3])
        ax.set_title(f"{title} (zoom)")
        ax.legend(fontsize=8)
        ax.axis('off')

    fig4.suptitle(f"TA-M4 / transition zoom (n_ta=8, h={h*BOHR_TO_ANG:.3f} Å)",
                  fontsize=14)
    fig4.savefig(f"{IMG_DIR}/octagon_ta_zoom.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {IMG_DIR}/octagon_ta_zoom.png")

    # === Fig 5: TA-M4 radial points + weights ===
    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 5))
    r_ta = grid_dict['r_ta']
    w_ta = grid_dict['w_ta']
    r_vals = grid_dict['r_vals']

    ax5a.plot(np.arange(1, len(r_vals) + 1), r_vals, 'o-', color='steelblue', markersize=5, label='All r_vals')
    ax5a.axvline(8 + 0.5, color='coral', linestyle='--', label='n_ta=8')
    ax5a.set_xlabel('Shell index j')
    ax5a.set_ylabel('r (Bohr)')
    ax5a.set_title('Radial point positions')
    ax5a.legend(fontsize=9)
    ax5a.grid(True, alpha=0.3)

    ax5b.semilogy(r_ta, w_ta, 'o-', color='steelblue', markersize=5, label='TA-M4 weights')
    ax5b.set_xlabel('r')
    ax5b.set_ylabel('w_r (radial weight)')
    ax5b.set_title('TA-M4 radial weights')
    ax5b.legend(fontsize=9)
    ax5b.grid(True, alpha=0.3)

    fig5.suptitle(f"TA-M4 radial grid (n_ta=8, ξ=1.0, R_cut={R_cut:.4f} Bohr, Nu={p['Nu']})",
                  fontsize=14)
    fig5.tight_layout()
    fig5.savefig(f"{IMG_DIR}/octagon_ta_radial_detail.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {IMG_DIR}/octagon_ta_radial_detail.png")

    print(f"\nSaved 5 figures to {IMG_DIR}/")

    # === Fig 6: Interface weight optimization ===
    print("\n=== Interface weight optimization ===")
    opt = optimize_interface_weights(grid_dict, nodes, polys, regions, d, h, n_ta=8)

    print(f"  Interface points: {opt['N_iface']}")
    print(f"  C4v orbits: {opt['N_orbits']}")
    print(f"  Test functions: {len(opt['test_funcs'])}")
    print(f"  R_cut: {opt['R_cut']:.2f} Bohr ({opt['R_cut']*BOHR_TO_ANG:.2f} Å)")
    print(f"  Best λ: {opt['best_lam']}")
    best = opt['best_result']
    baseline_mean = np.mean(opt['baseline_errs'][opt['baseline_errs'] > 0])
    baseline_max = np.max(opt['baseline_errs'])
    print(f"  Baseline (geometric): mean={baseline_mean:.4f}%, max={baseline_max:.4f}%")
    print(f"  Optimized:            mean={best['total_mean_err']:.4f}%, max={best['total_max_err']:.4f}%")
    print(f"  Orbit weights (w0 → w_opt):")
    for i, (w0, wopt, rep) in enumerate(zip(opt['w0_iface'], opt['w_opt'], opt['orbit_rep'])):
        print(f"    orbit {i}: rep=({rep[0]:.4f},{rep[1]:.4f}), "
              f"w0={w0:.6f}, w_opt={wopt:.6f}, members={len(opt['orbit_members'][i])}")

    # Per-function errors
    print(f"\n  Per-function total errors (best λ={opt['best_lam']}):")
    for k, (label, _) in enumerate(opt['test_funcs']):
        err = best['total_errs'][k]
        if k == 0 or err > 0.01:  # show area constraint + any with noticeable error
            print(f"    {label:30s}: {best['total_errs'][k]:.4f}%")

    # Plot: weights before/after + error vs lambda
    fig6, (ax6a, ax6b, ax6c) = plt.subplots(1, 3, figsize=(18, 5))

    # Left: optimized weights on grid
    color_map = {'inner': 'steelblue', 'near_interface': 'mediumpurple',
                 'interface': 'coral', 'outer': 'gray'}
    for lbl_name in ['inner', 'near_interface', 'interface', 'outer']:
        mask = opt['lbl_v'] == lbl_name
        if mask.any():
            ax6a.scatter(opt['nodes'][mask, 0], opt['nodes'][mask, 1],
                         c=color_map[lbl_name], s=30, edgecolors='k', linewidths=0.3,
                         alpha=0.8, label=f"{lbl_name} ({mask.sum()})")
    # Annotate interface orbit weights
    for oid, members in enumerate(opt['orbit_members']):
        rep_pt = opt['nodes'][members[0]]
        ax6a.annotate(f"w={opt['w_opt'][oid]:.4f}", rep_pt, fontsize=7,
                      textcoords="offset points", xytext=(5, 5))
    oct_v = grid_dict['oct_verts']
    oct_closed = np.vstack([oct_v, oct_v[0]])
    ax6a.plot(oct_closed[:, 0], oct_closed[:, 1], 'k-', lw=1.0, alpha=0.5)
    ax6a.set_aspect('equal')
    ax6a.set_title(f"Optimized interface weights (λ={opt['best_lam']})")
    ax6a.legend(fontsize=7)
    ax6a.axis('off')

    # Middle: w0 vs w_opt
    ax6b.plot(range(len(opt['w0_iface'])), opt['w0_iface'], 'o-', label='w0 (geometric)',
              color='gray', alpha=0.5)
    ax6b.plot(range(len(opt['w_opt'])), opt['w_opt'], 's-', label='w_opt (optimized)',
              color='coral', markersize=8)
    ax6b.set_xlabel('Orbit index')
    ax6b.set_ylabel('Weight')
    ax6b.set_title(f'Interface orbit weights ({opt["N_orbits"]} orbits)')
    ax6b.legend()
    ax6b.grid(True, alpha=0.3)

    # Right: error vs lambda
    lams = sorted(opt['results'].keys())
    mean_errs = [opt['results'][l]['total_mean_err'] for l in lams]
    max_errs = [opt['results'][l]['total_max_err'] for l in lams]
    ax6c.semilogx(lams, mean_errs, 'o-', label='mean error', color='steelblue')
    ax6c.semilogx(lams, max_errs, 's-', label='max error', color='coral')
    ax6c.axvline(opt['best_lam'], color='green', linestyle='--', alpha=0.5, label=f"best λ={opt['best_lam']}")
    ax6c.set_xlabel('Regularization λ')
    ax6c.set_ylabel('Total integral error (%)')
    ax6c.set_title('Error vs λ')
    ax6c.legend()
    ax6c.grid(True, alpha=0.3)

    fig6.suptitle(f"Interface weight optimization (n_ta=8, Nu={p['Nu']}, "
                  f"{opt['N_orbits']} C4v orbits, {opt['N_iface']} interface points)",
                  fontsize=14)
    fig6.tight_layout()
    fig6.savefig(f"{IMG_DIR}/octagon_ta_optimization.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {IMG_DIR}/octagon_ta_optimization.png")


if __name__ == "__main__":
    main()
