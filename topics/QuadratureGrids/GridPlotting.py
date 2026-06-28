"""
GridPlotting.py — Visualization functions for all quadrature grids.

Plotting module: all matplotlib visualization for triangular grids,
wedge grids, and CubeEmbededAtoms grids.
No core math, no grid generation.
"""
import numpy as np
import matplotlib.pyplot as plt

from TriGrids import REF_VERTS


# ── Triangular grid plotting ──────────────────────────────────────────────────

def plot_tri_grid(n, ax=None, show_indices=False, show_bary=False,
                  node_color='red', edge_color='steelblue', title=None):
    """Visualize the barycentric subdivided triangle of order n."""
    from TriGrids import barycentric_grid, subdivide_triangles
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    bary, xy, idx = barycentric_grid(n)
    tris = subdivide_triangles(n, idx)
    for (a, b, c) in tris:
        pts = xy[[a, b, c, a]]
        ax.plot(pts[:, 0], pts[:, 1], '-', color=edge_color, lw=0.8, alpha=0.7)
    outer = np.vstack([REF_VERTS, REF_VERTS[0]])
    ax.plot(outer[:, 0], outer[:, 1], 'k-', lw=2.0)
    ax.plot(xy[:, 0], xy[:, 1], 'o', color=node_color, ms=5, zorder=5)
    if show_indices:
        for (ijk, r) in idx.items():
            label = f"{ijk[0]},{ijk[1]},{ijk[2]}" if show_bary else str(r)
            ax.annotate(label, xy[r], fontsize=6, ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points')
    N = len(xy)
    ax.set_aspect('equal')
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.set_title(title or f"n={n}  nodes={N}  sub-tris={len(tris)}")
    ax.axis('off')
    return ax


def compare_tri_levels(levels=(1, 2, 3, 4, 5)):
    """Side-by-side plot of multiple subdivision levels."""
    n_plots = len(levels)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4.5))
    if n_plots == 1:
        axes = [axes]
    for ax, n in zip(axes, levels):
        plot_tri_grid(n, ax=ax)
    fig.suptitle("Barycentric triangular subdivision (Pascal-triangle grid)", y=0.98)
    fig.tight_layout()
    return fig


def plot_general_tri_grid(nu, nv, ax=None, show_indices=False,
                          node_color='red', edge_color='steelblue',
                          diag_color='orange', title=None, verts=REF_VERTS):
    """Visualize the generalized triangular grid with (nu, nv) subdivisions."""
    from TriGrids import general_tri_grid
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    nodes, tris, idx = general_tri_grid(nu, nv, verts)
    n_grid = max(idx.values()) + 1 if idx else 0
    for tri in tris:
        pts = nodes[[*tri, tri[0]]]
        ax.plot(pts[:, 0], pts[:, 1], '-', color=edge_color, lw=0.8, alpha=0.7)
    outer = np.vstack([verts, verts[0]])
    ax.plot(outer[:, 0], outer[:, 1], 'k-', lw=2.0)
    if n_grid > 0:
        ax.plot(nodes[:n_grid, 0], nodes[:n_grid, 1], 'o', color=node_color, ms=5, zorder=5)
    if len(nodes) > n_grid:
        ax.plot(nodes[n_grid:, 0], nodes[n_grid:, 1], 's', color=diag_color, ms=5, zorder=5)
    if show_indices:
        for (ij, r) in idx.items():
            ax.annotate(f"{ij[0]},{ij[1]}", nodes[r], fontsize=6, ha='center',
                        va='bottom', xytext=(0, 4), textcoords='offset points')
    N = len(nodes)
    ax.set_aspect('equal')
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.set_title(title or f"nu={nu} nv={nv}  nodes={N}  tris={len(tris)}")
    ax.axis('off')
    return ax


def compare_general_tri_grids(specs, verts=REF_VERTS):
    """Side-by-side plot of multiple (nu, nv) grids."""
    n_plots = len(specs)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4.5))
    if n_plots == 1:
        axes = [axes]
    for ax, (nu, nv) in zip(axes, specs):
        plot_general_tri_grid(nu, nv, ax=ax, verts=verts)
    fig.suptitle("Generalized triangular grids (nu, nv)", y=0.98)
    fig.tight_layout()
    return fig


# ── Wedge grid plotting ───────────────────────────────────────────────────────

def plot_wedge_grid(n, m, v0=0.0, ax=None, show_indices=False,
                    show_layers=False, node_color='red',
                    edge_color='steelblue', title=None, verts=REF_VERTS):
    """Visualize the wedge grid."""
    from TriGrids import wedge_grid
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    nodes, tris, bary, info = wedge_grid(n, m, v0, verts)
    for tri in tris:
        pts = nodes[[*tri, tri[0]]]
        ax.plot(pts[:, 0], pts[:, 1], '-', color=edge_color, lw=0.8, alpha=0.7)
    outer = np.vstack([verts, verts[0]])
    ax.plot(outer[:, 0], outer[:, 1], 'k-', lw=2.0)
    if v0 > 0:
        inner = np.array([[v0, 0], [0, v0]])
        ax.plot(inner[:, 0], inner[:, 1], 'k--', lw=1.5)
    if show_layers:
        colors = plt.cm.viridis(np.linspace(0, 0.9, info['n_layers']))
        for layer, node_ids in info['layers'].items():
            c = colors[layer]
            pts = nodes[node_ids]
            ax.plot(pts[:, 0], pts[:, 1], 'o', color=c, ms=6, zorder=6)
    else:
        ax.plot(nodes[:, 0], nodes[:, 1], 'o', color=node_color, ms=4, zorder=5)
    if show_indices:
        for r in range(len(nodes)):
            ax.annotate(str(r), nodes[r], fontsize=5, ha='center',
                        va='bottom', xytext=(0, 3), textcoords='offset points')
    N = len(nodes)
    ax.set_aspect('equal')
    margin = 0.08
    ax.set_xlim(-margin, 1 + margin)
    ax.set_ylim(-margin, 1 + margin)
    ax.set_title(title or f"n={n} m={m} v0={v0}  nodes={N}  tris={len(tris)}")
    ax.axis('off')
    return ax


def compare_wedge_grids(specs, v0=0.1):
    """Side-by-side plot of multiple wedge grids."""
    n_plots = len(specs)
    fig, axes = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    for ax, (n, m) in zip(axes, specs):
        plot_wedge_grid(n, m, v0, ax=ax)
    fig.suptitle(f"Wedge grids (v0={v0})", y=0.98)
    fig.tight_layout()
    return fig


# ── CubeEmbededAtoms grid plotting ────────────────────────────────────────────

def draw_cube_grid(ax, pts, grid_dict, title=''):
    """Draw grid lines and points for any of the three cube-embedded grids."""
    p = grid_dict['params']
    Nu, Nv = p['Nu'], p['Nv']
    n_per_wedge, Nw = p['n_per_wedge'], p['Nw']
    h = p['h']
    for j in range(1, Nv):
        for i_w in range(Nw):
            i0 = i_w * n_per_wedge
            i1 = i0 + n_per_wedge
            seg = pts[i0:i1+1, j]
            if i_w == Nw - 1:
                seg = np.vstack([seg, pts[0:1, j]])
            ax.plot(seg[:, 0], seg[:, 1], 'b-', linewidth=0.6, alpha=0.6)
    for i in range(Nu):
        ax.plot(pts[i, :, 0], pts[i, :, 1], 'r-', linewidth=0.3, alpha=0.4)
    ax.scatter(pts[:, 1:, 0].ravel(), pts[:, 1:, 1].ravel(), s=3, c='black', zorder=5)
    ax.scatter(0, 0, c='red', s=60, marker='*', zorder=6)
    ax.plot([-h, h, h, -h, -h], [-h, -h, h, h, -h], 'k--', linewidth=1, alpha=0.3)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlim(-h * 1.05, h * 1.05)
    ax.set_ylim(-h * 1.05, h * 1.05)


def plot_cube_full_view(grid_dict, margin=3):
    """Full view: Cartesian outer + blended inner + grid lines."""
    p = grid_dict['params']
    points = grid_dict['points']
    h, d = p['h'], p['d']
    Nu, Nv = p['Nu'], p['Nv']
    n_per_wedge, Nw = p['n_per_wedge'], p['Nw']
    corners = grid_dict['corners']
    edge_pts = grid_dict['edge_pts']

    fig, ax = plt.subplots(figsize=(14, 14))
    x_outer = np.arange(-h - margin * d, h + margin * d + d / 2, d)
    y_outer = np.arange(-h - margin * d, h + margin * d + d / 2, d)
    for x in x_outer:
        ax.plot([x, x], [y_outer[0], y_outer[-1]], 'k-', linewidth=0.3, alpha=0.2)
    for y in y_outer:
        ax.plot([x_outer[0], x_outer[-1]], [y, y], 'k-', linewidth=0.3, alpha=0.2)
    ax.plot([-h, h, h, -h, -h], [-h, -h, h, h, -h], 'k--', linewidth=1.5, alpha=0.4)

    for j in range(1, Nv):
        for i_w in range(Nw):
            i0 = i_w * n_per_wedge
            i1 = i0 + n_per_wedge
            pts = points[i0:i1+1, j]
            if i_w == Nw - 1:
                pts = np.vstack([pts, points[0:1, j]])
            ax.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=0.7, alpha=0.6)
    for i in range(Nu):
        ax.plot(points[i, :, 0], points[i, :, 1], 'r-', linewidth=0.4, alpha=0.4)
    for i_w in range(Nw):
        i = i_w * n_per_wedge
        ax.plot(points[i, :, 0], points[i, :, 1], 'r-', linewidth=1.2, alpha=0.6)

    ax.scatter(points[:, 1:, 0].ravel(), points[:, 1:, 1].ravel(), s=5, c='black', zorder=5)
    ax.scatter(0, 0, c='red', s=80, marker='*', zorder=6)
    boundary = points[:, -1, :]
    ax.scatter(boundary[:, 0], boundary[:, 1], s=30, c='lime',
               edgecolors='black', linewidths=0.5, zorder=7, label='Cartesian snap points')
    ax.scatter(corners[:, 0], corners[:, 1], s=50, c='orange', marker='s',
               zorder=7, label='Wedge corners')

    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'CubeEmbededAtoms grid (2D) — wedge-based\n'
                 f'Inner: {Nw} wedges × {n_per_wedge} pts × {Nv} shells, α={p["alpha"]}\n'
                 f'Outer: Cartesian d={d}, cutout {p["n"]}×{p["n"]} voxels (L={p["L"]})\n'
                 f'Blue=u-lines (shells), Red=v-lines (rays), Thick=corner rays', fontsize=11)
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig


def plot_cube_cell_areas(grid_dict):
    """Zoom + cell area coloring."""
    p = grid_dict['params']
    points = grid_dict['points']
    cell_areas = grid_dict['cell_areas']
    Nu, Nv = p['Nu'], p['Nv']
    n_per_wedge, Nw = p['n_per_wedge'], p['Nw']
    h = p['h']

    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(Nu):
        i2 = (i + 1) % Nu
        for j in range(Nv - 1):
            p0, p1, p2, p3 = points[i, j], points[i2, j], points[i2, j + 1], points[i, j + 1]
            quad = np.array([p0, p1, p2, p3, p0])
            ax.fill(quad[:, 0], quad[:, 1], color=plt.cm.viridis(
                cell_areas[i, j] / cell_areas.max()), alpha=0.5)
    for j in range(1, Nv):
        for i_w in range(Nw):
            i0 = i_w * n_per_wedge
            i1 = i0 + n_per_wedge
            pts = points[i0:i1+1, j]
            if i_w == Nw - 1:
                pts = np.vstack([pts, points[0:1, j]])
            ax.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=0.6, alpha=0.7)
    for i in range(Nu):
        ax.plot(points[i, :, 0], points[i, :, 1], 'r-', linewidth=0.4, alpha=0.5)
    ax.scatter(points[:, 1:, 0].ravel(), points[:, 1:, 1].ravel(), s=4, c='black', zorder=5)
    ax.scatter(0, 0, c='red', s=80, marker='*', zorder=6)
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=cell_areas.min(), vmax=cell_areas.max()))
    plt.colorbar(sm, ax=ax, label='Cell area', shrink=0.7)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Cell area distribution\n'
                 f'min={cell_areas.min():.4f}, max={cell_areas.max():.4f}, '
                 f'ratio={cell_areas.max()/cell_areas.min():.1f}')
    plt.tight_layout()
    return fig


def plot_cube_radial_profile(grid_dict):
    """Radial profiles + cell quality."""
    p = grid_dict['params']
    points = grid_dict['points']
    cell_aspect = grid_dict['cell_aspect']
    Nu, Nv = p['Nu'], p['Nv']
    alpha = p['alpha']
    angular_spacing = p['angular_spacing']

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    shell_idx = np.arange(Nv)
    shell_radii = np.array([np.mean(np.linalg.norm(points[:, j], axis=1)) for j in range(Nv)])
    ax1.plot(shell_idx, shell_radii, 'o-', markersize=4, color='steelblue')
    ax1.set_ylabel('Effective radius (Bohr)')
    ax1.set_title(f'Radial profile (α={alpha}, Nv={Nv}, derived for isotropy)')
    dr = np.diff(shell_radii)
    ax2.plot(shell_idx[1:], dr, 's-', markersize=4, color='darkorange')
    ax2.axhline(y=angular_spacing, color='red', linestyle='--', alpha=0.5,
                label=f'Angular spacing = {angular_spacing:.3f}')
    ax2.set_ylabel('Radial spacing dr')
    ax2.set_title('Radial spacing between shells (target: match angular spacing)')
    ax2.legend(fontsize=9)
    mean_aspect = np.array([cell_aspect[:, j].mean() for j in range(Nv - 1)])
    ax3.plot(shell_idx[1:], mean_aspect, 'D-', markersize=4, color='purple')
    ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Isotropic (1.0)')
    ax3.set_ylabel('Aspect ratio')
    ax3.set_title('Cell aspect ratio (angular/radial edge, 1.0=isotropic)')
    ax3.legend(fontsize=9)
    cumulative = np.cumsum([Nu] * Nv)
    ax4.plot(shell_idx, cumulative, 'o-', markersize=4, color='darkgreen')
    ax4.set_xlabel('Shell index')
    ax4.set_ylabel('Cumulative points')
    ax4.set_title(f'Cumulative grid points (total = {Nu * Nv})')
    plt.tight_layout()
    return fig


def plot_cube_three_grids(grid_dict):
    """Compare all three grids (wedge, blended, sphere)."""
    p = grid_dict['params']
    n_blend = p['n_blend']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    draw_cube_grid(axes[0], grid_dict['points_wedge'], grid_dict,
                   f'1) Pure linear wedges\n(straight triangles, no curvature)')
    draw_cube_grid(axes[1], grid_dict['points'], grid_dict,
                   f'2) Blended (1-t^{n_blend})\n(sphere→wedge, w=1-t^{n_blend})')
    draw_cube_grid(axes[2], grid_dict['points_sphere'], grid_dict,
                   f'3) Pure spherical\n(uniform angular, perfect circles)')
    fig.suptitle('CubeEmbededAtoms: 3-step construction (wedge → blend → sphere)', fontsize=14)
    plt.tight_layout()
    return fig


def plot_blend_weight(grid_dict):
    """Interpolation function on [0,1]."""
    p = grid_dict['params']
    t_vals = grid_dict['t_vals']
    w_blend = grid_dict['w_blend']
    n_blend = p['n_blend']
    from CubeEmbededGrid import smoothstep

    t_fine = np.linspace(0, 1, 500)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    ax1.plot(t_fine, 1 - t_fine, linewidth=1.5, label='linear: 1-t')
    ax1.plot(t_fine, 1 - smoothstep(t_fine), linewidth=1.5, label='smoothstep: 1-t²(3-2t)')
    ax1.plot(t_fine, (1 - t_fine)**2, linewidth=1.5, label='(1-t)²')
    ax1.plot(t_fine, 1 - t_fine**4, linewidth=2, label='1-t⁴ [used]')
    ax1.plot(t_fine, 1 - t_fine**6, linewidth=1.5, label='1-t⁶')
    ax1.axhline(y=1, color='blue', linestyle='--', alpha=0.2)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.2)
    ax1.set_ylabel('Blend weight w')
    ax1.set_title('Interpolation weight w(t): sphere (w=1) → wedge (w=0)')
    ax1.legend(fontsize=9)
    ax1.set_ylim(-0.1, 1.1)
    ax1.text(0.02, 0.95, 'CENTER\n(sphere)', fontsize=9, va='top', color='blue')
    ax1.text(0.85, 0.15, 'BOUNDARY\n(wedge)', fontsize=9, va='bottom', color='red')

    ax2.plot(t_fine, 1 - t_fine**n_blend, linewidth=2, color='darkgreen', label=f'w = 1-t^{n_blend}')
    ax2.scatter(t_vals, w_blend, s=30, c='black', zorder=5, label=f'Grid shells (Nv={len(t_vals)})')
    for j in range(len(t_vals)):
        ax2.annotate(f'{j}', (t_vals[j], w_blend[j]), fontsize=6, ha='center', va='bottom')
    ax2.axhline(y=1, color='blue', linestyle='--', alpha=0.2, label='w=1 (pure sphere)')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.2, label='w=0 (pure wedge)')
    ax2.set_xlabel('Radial parameter t (0=center, 1=boundary)')
    ax2.set_ylabel('Blend weight w')
    ax2.set_title(f'Active interpolation 1-t^{n_blend} with shell positions')
    ax2.legend(fontsize=9)
    ax2.set_ylim(-0.1, 1.1)
    plt.tight_layout()
    return fig


# ── Orbital/density plotting ──────────────────────────────────────────────────

def scatter_plot(ax, xy, vals, title, cmap='RdBu_r', vmin=None, vmax=None,
                 h=None):
    """Scatter colored by value, with optional cutout boundary."""
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=vals, s=1, cmap=cmap, vmin=vmin, vmax=vmax)
    if h is not None:
        ax.plot([-h, h, h, -h, -h], [-h, -h, h, h, -h], 'k--', linewidth=1, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, shrink=0.7)


def plot_orbitals_on_grid(combined_xy, phi_1s, rho_1s, phi_2px, rho_2px, rho_sp,
                          num_1s, exact_1s, num_2p, exact_2p, num_rho_sp, exact_rho_sp,
                          zeta_1s, zeta_2p, h):
    """Orbitals and density on combined grid."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    scatter_plot(axes[0, 0], combined_xy, phi_1s,
                 f'1s orbital (STO-3G, ζ={zeta_1s})\n∫|φ|²={num_1s:.4f} (exact={exact_1s:.4f})', h=h)
    scatter_plot(axes[0, 1], combined_xy, rho_1s, f'1s density |φ|²\n∫ρ={num_1s:.4f}', cmap='hot_r', h=h)
    scatter_plot(axes[0, 2], combined_xy, np.log10(rho_1s + 1e-10),
                 f'1s density (log10)\nerror={abs(num_1s-exact_1s)/exact_1s*100:.3f}%', cmap='hot_r', h=h)
    scatter_plot(axes[1, 0], combined_xy, phi_2px,
                 f'2pₓ orbital (STO-3G, ζ={zeta_2p})\n∫|φ|²={num_2p:.4f} (exact={exact_2p:.4f})', h=h)
    scatter_plot(axes[1, 1], combined_xy, rho_2px, f'2pₓ density |φ|²\n∫ρ={num_2p:.4f}', cmap='hot_r', h=h)
    scatter_plot(axes[1, 2], combined_xy, rho_sp,
                 f'ρ = 1s² + 0.25·2pₓ²\n∫ρ={num_rho_sp:.4f} (exact={exact_rho_sp:.4f})', cmap='hot_r', h=h)
    fig.suptitle('CubeEmbededAtoms integration grid: atomic orbital test functions', fontsize=14)
    plt.tight_layout()
    return fig


def plot_density_overlay(grid_dict, inner_xy, sto3g_1s, zeta_1s, sto3g_2p_c, zeta_2p):
    """Grid structure + density contour + radial profile."""
    from BasisFunctions import gaussian_1s
    p = grid_dict['params']
    points = grid_dict['points']
    h = p['h']
    Nu, Nv = p['Nu'], p['Nv']
    n_per_wedge, Nw = p['n_per_wedge'], p['Nw']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    for j in range(1, Nv):
        for i_w in range(Nw):
            i0 = i_w * n_per_wedge
            i1 = i0 + n_per_wedge
            pts = points[i0:i1+1, j]
            if i_w == Nw - 1:
                pts = np.vstack([pts, points[0:1, j]])
            ax.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=0.4, alpha=0.4)
    for i in range(Nu):
        ax.plot(points[i, :, 0], points[i, :, 1], 'r-', linewidth=0.2, alpha=0.3)
    r_grid = np.linspace(-h * 1.3, h * 1.3, 200)
    Xg, Yg = np.meshgrid(r_grid, r_grid)
    R2g = Xg**2 + Yg**2
    rho_grid = gaussian_1s(R2g, sto3g_1s, zeta_1s)**2
    ax.contourf(Xg, Yg, np.log10(rho_grid + 1e-10), levels=20, cmap='hot_r', alpha=0.3)
    ax.set_aspect('equal')
    ax.set_title('Grid structure + 1s density contour')
    ax.set_xlim(-h * 1.3, h * 1.3)
    ax.set_ylim(-h * 1.3, h * 1.3)

    ax = axes[1]
    r_inner = np.linalg.norm(inner_xy, axis=1)
    sort_idx = np.argsort(r_inner)
    r_sorted = r_inner[sort_idx]
    phi_sorted = gaussian_1s(r_sorted**2, sto3g_1s, zeta_1s)
    rho_sorted = phi_sorted**2
    r_analytic = np.linspace(0, h * 1.5, 500)
    phi_analytic = gaussian_1s(r_analytic**2, sto3g_1s, zeta_1s)
    rho_analytic = phi_analytic**2
    ax.plot(r_analytic, rho_analytic, 'b-', linewidth=2, label='Analytic 1s density')
    ax.scatter(r_sorted, rho_sorted, s=5, c='red', alpha=0.5, label='Grid points (sorted by r)')
    slater_rho = np.exp(-2 * zeta_1s * r_analytic)
    ax.plot(r_analytic, slater_rho, 'g--', linewidth=1.5, alpha=0.5,
            label=f'Slater target: exp(-2ζr), ζ={zeta_1s}')
    ax.set_xlabel('r (Bohr)')
    ax.set_ylabel('ρ(r)')
    ax.set_title('1s radial profile: STO-3G vs Slater target')
    ax.legend(fontsize=9)
    ax.set_xlim(0, h * 1.5)
    ax.set_ylim(0, max(rho_analytic.max(), slater_rho.max()) * 1.1)

    fig.suptitle('CubeEmbededAtoms: orbital density visualization', fontsize=14)
    plt.tight_layout()
    return fig


def plot_convergence(ns_test, errors_1s, errors_2p, npts_list):
    """Convergence test plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.loglog(npts_list, errors_1s, 'o-', label='1s density', color='steelblue')
    ax1.loglog(npts_list, errors_2p, 's-', label='2p density', color='darkorange')
    ax1.set_xlabel('Total grid points')
    ax1.set_ylabel('Integration error (%)')
    ax1.set_title('Convergence: integration error vs grid size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.semilogy(ns_test, errors_1s, 'o-', label='1s density', color='steelblue')
    ax2.semilogy(ns_test, errors_2p, 's-', label='2p density', color='darkorange')
    ax2.set_xlabel('Cutout size n (voxels per side)')
    ax2.set_ylabel('Integration error (%)')
    ax2.set_title('Convergence: error vs cutout resolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.suptitle('CubeEmbededAtoms: integration convergence test', fontsize=14)
    plt.tight_layout()
    return fig


# ── Weight optimization plotting ──────────────────────────────────────────────

def plot_weight_comparison(grid_pts, w_geom, w_opt, h, lam_opt):
    """Weight maps: geometric vs optimized vs relative deviation."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sc0 = axes[0].scatter(grid_pts[:, 0], grid_pts[:, 1], c=w_geom, s=20, cmap='viridis')
    axes[0].plot([-h, h, h, -h, -h], [-h, -h, h, h, -h], 'k--', linewidth=1, alpha=0.3)
    axes[0].set_aspect('equal')
    axes[0].set_title('Geometric weights w₀')
    axes[0].set_xlim(-h * 1.1, h * 1.1)
    axes[0].set_ylim(-h * 1.1, h * 1.1)
    plt.colorbar(sc0, ax=axes[0], shrink=0.7)

    sc1 = axes[1].scatter(grid_pts[:, 0], grid_pts[:, 1], c=w_opt, s=20, cmap='viridis')
    axes[1].plot([-h, h, h, -h, -h], [-h, -h, h, h, -h], 'k--', linewidth=1, alpha=0.3)
    axes[1].set_aspect('equal')
    axes[1].set_title(f'Optimized weights (λ={lam_opt})')
    axes[1].set_xlim(-h * 1.1, h * 1.1)
    axes[1].set_ylim(-h * 1.1, h * 1.1)
    plt.colorbar(sc1, ax=axes[1], shrink=0.7)

    w_ratio = w_opt / (w_geom + 1e-10) - 1.0
    vmax = max(abs(w_ratio.min()), abs(w_ratio.max()))
    sc2 = axes[2].scatter(grid_pts[:, 0], grid_pts[:, 1], c=w_ratio, s=20, cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax)
    axes[2].plot([-h, h, h, -h, -h], [-h, -h, h, h, -h], 'k--', linewidth=1, alpha=0.3)
    axes[2].set_aspect('equal')
    axes[2].set_title('w/w₀ − 1 (relative deviation)')
    axes[2].set_xlim(-h * 1.1, h * 1.1)
    axes[2].set_ylim(-h * 1.1, h * 1.1)
    plt.colorbar(sc2, ax=axes[2], shrink=0.7)

    fig.suptitle('CubeEmbededAtoms: quadrature weight optimization', fontsize=14)
    plt.tight_layout()
    return fig


def plot_weight_errors(test_labels, errs_geom, errs_opt, lam_opt):
    """Per-function integration error bar chart."""
    Ntest = len(test_labels)
    fig, ax = plt.subplots(figsize=(14, 6))
    x_pos = np.arange(Ntest)
    width = 0.35
    ax.bar(x_pos - width / 2, errs_geom, width, label='Geometric', color='steelblue', alpha=0.7)
    ax.bar(x_pos + width / 2, errs_opt, width, label='Optimized', color='darkorange', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([t[:15] for t in test_labels], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Integration error (%)')
    ax.set_title(f'Test function integration errors (λ={lam_opt})')
    ax.legend()
    ax.set_ylim(0, max(errs_geom.max(), errs_opt.max()) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_lambda_sweep(results, lambdas, w0_orbit):
    """Error and weight deviation vs λ."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    lams_plot = sorted(results.keys())
    max_errs = [results[l]['max_err'] for l in lams_plot]
    mean_errs = [results[l]['mean_err'] for l in lams_plot]
    w_stds = [np.std(results[l]['w_orbit'] / (w0_orbit + 1e-10)) for l in lams_plot]
    ax1.semilogy(lams_plot, max_errs, 'o-', label='Max error', color='red')
    ax1.semilogy(lams_plot, mean_errs, 's-', label='Mean error', color='blue')
    ax1.set_xlabel('Regularization λ')
    ax1.set_ylabel('Integration error (%)')
    ax1.set_title('Error vs regularization strength')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.semilogy(lams_plot, w_stds, 'D-', color='green')
    ax2.set_xlabel('Regularization λ')
    ax2.set_ylabel('std(w / w0)')
    ax2.set_title('Weight deviation from geometric (vs λ)')
    ax2.grid(True, alpha=0.3)
    fig.suptitle('CubeEmbededAtoms: regularization sweep', fontsize=14)
    plt.tight_layout()
    return fig
