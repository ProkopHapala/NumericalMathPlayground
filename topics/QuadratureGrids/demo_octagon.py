"""
demo_octagon.py — Visualize the octagon-intermediate grid.

Usage:
    python demo_octagon.py
"""
import numpy as np
import matplotlib.pyplot as plt

from OctagonGrid import (
    build_octagon_grid, build_combined_octagon_grid,
    build_cartesian_corners, octagon_vertices,
    build_full_mesh, extract_vertex_points, extract_cell_center_points,
)

IMG_DIR = "Quadrature_Images"


def plot_grid_structure(grid_dict, ax=None, title=None):
    """Plot the octagon grid: mesh lines + boundary + octagon outline."""
    p = grid_dict['params']
    points = grid_dict['points']
    Nu, Nv = points.shape[:2]
    h = p['h']

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    # Draw mesh lines (radial + angular)
    for i in range(Nu):
        i2 = (i + 1) % Nu
        ax.plot(points[i, :, 0], points[i, :, 1], '-', color='steelblue', lw=0.4, alpha=0.6)
        for j in range(Nv - 1):
            ax.plot([points[i, j, 0], points[i2, j, 0]],
                    [points[i, j, 1], points[i2, j, 1]],
                    '-', color='steelblue', lw=0.4, alpha=0.6)

    # Draw octagon boundary
    oct_v = grid_dict['oct_verts']
    oct_closed = np.vstack([oct_v, oct_v[0]])
    ax.plot(oct_closed[:, 0], oct_closed[:, 1], 'k-', lw=2.0, label='Octagon')

    # Draw square
    sq = np.array([[-h, -h], [h, -h], [h, h], [-h, h], [-h, -h]])
    ax.plot(sq[:, 0], sq[:, 1], 'k--', lw=1.5, alpha=0.5, label='Square')

    # Draw circle (inscribed, radius h)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(h * np.cos(theta), h * np.sin(theta), 'g--', lw=1.0, alpha=0.5, label='Circle (r=h)')

    # Plot nodes
    ax.plot(points[:, :, 0].ravel(), points[:, :, 1].ravel(),
            'o', color='red', ms=2, zorder=5)

    ax.set_aspect('equal')
    margin = 0.3 * h
    ax.set_xlim(-h - margin, h + margin)
    ax.set_ylim(-h - margin, h + margin)
    ax.set_title(title or f"Octagon grid  n={p['Nu']}×{p['Nv']}  (α={p['alpha']})")
    ax.legend(loc='upper right', fontsize=8)
    ax.axis('off')
    return ax


def plot_three_grids(grid_dict, ax=None):
    """Show octagon, circle, and blended side by side."""
    points_oct = grid_dict['points_octagon']
    points_cir = grid_dict['points_circle']
    points_blend = grid_dict['points']
    Nu, Nv = points_blend.shape[:2]
    p = grid_dict['params']
    h = p['h']

    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    else:
        axes = ax

    titles = ['Pure octagon (scaled)', 'Pure circle', 'Blended (circle→octagon)']
    datasets = [points_oct, points_cir, points_blend]

    for k, (ax_k, data, title) in enumerate(zip(axes, datasets, titles)):
        for i in range(Nu):
            i2 = (i + 1) % Nu
            ax_k.plot(data[i, :, 0], data[i, :, 1], '-', color='steelblue', lw=0.4, alpha=0.5)
            for j in range(Nv - 1):
                ax_k.plot([data[i, j, 0], data[i2, j, 0]],
                          [data[i, j, 1], data[i2, j, 1]],
                          '-', color='steelblue', lw=0.4, alpha=0.5)
        ax_k.plot(data[:, :, 0].ravel(), data[:, :, 1].ravel(),
                  'o', color='red', ms=2, zorder=5)
        oct_v = grid_dict['oct_verts']
        oct_closed = np.vstack([oct_v, oct_v[0]])
        ax_k.plot(oct_closed[:, 0], oct_closed[:, 1], 'k-', lw=1.5)
        theta = np.linspace(0, 2 * np.pi, 200)
        ax_k.plot(h * np.cos(theta), h * np.sin(theta), 'g--', lw=0.8, alpha=0.5)
        ax_k.set_aspect('equal')
        ax_k.set_title(title)
        ax_k.axis('off')

    return axes


def plot_combined(grid_dict, combined, ax=None, title=None):
    """Plot inner octagon grid + outer Cartesian points."""
    p = grid_dict['params']
    h = p['h']

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Inner grid mesh
    points = grid_dict['points']
    Nu, Nv = points.shape[:2]
    for i in range(Nu):
        i2 = (i + 1) % Nu
        ax.plot(points[i, :, 0], points[i, :, 1], '-', color='steelblue', lw=0.3, alpha=0.5)
        for j in range(Nv - 1):
            ax.plot([points[i, j, 0], points[i2, j, 0]],
                    [points[i, j, 1], points[i2, j, 1]],
                    '-', color='steelblue', lw=0.3, alpha=0.5)

    # Outer Cartesian
    ax.plot(combined['outer_xy'][:, 0], combined['outer_xy'][:, 1],
            's', color='gray', ms=3, alpha=0.6)

    # Inner cell centers
    ax.plot(combined['inner_xy'][:, 0], combined['inner_xy'][:, 1],
            'o', color='red', ms=3, zorder=5)

    # Octagon + square outline
    oct_v = grid_dict['oct_verts']
    oct_closed = np.vstack([oct_v, oct_v[0]])
    ax.plot(oct_closed[:, 0], oct_closed[:, 1], 'k-', lw=1.5)
    sq = np.array([[-h, -h], [h, -h], [h, h], [-h, h], [-h, -h]])
    ax.plot(sq[:, 0], sq[:, 1], 'k--', lw=1.0, alpha=0.4)

    ax.set_aspect('equal')
    margin = 1.5 * h
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_title(title or f"Combined grid  inner={len(combined['inner_xy'])}  outer={len(combined['outer_xy'])}")
    ax.axis('off')
    return ax


def plot_cell_quality(grid_dict, ax=None):
    """Plot cell area + aspect ratio as function of shell index."""
    p = grid_dict['params']
    areas = grid_dict['cell_areas']
    aspect = grid_dict['cell_aspect']
    Nu, Nv1 = areas.shape

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        ax1, ax2 = ax[0], ax[1]

    for j in range(Nv1):
        ax1.plot(j, areas[:, j].mean(), 'o', color='steelblue', ms=4)
        ax1.fill_between([j], areas[:, j].min(), areas[:, j].max(), alpha=0.2, color='steelblue')
    ax1.set_xlabel('Shell j')
    ax1.set_ylabel('Cell area')
    ax1.set_title('Cell area distribution per shell')

    for j in range(Nv1):
        ax2.plot(j, aspect[:, j].mean(), 'o', color='coral', ms=4)
        ax2.fill_between([j], aspect[:, j].min(), aspect[:, j].max(), alpha=0.2, color='coral')
    ax2.set_xlabel('Shell j')
    ax2.set_ylabel('Aspect ratio (1=isotropic)')
    ax2.set_title('Cell aspect ratio per shell')
    ax2.axhline(1.0, color='k', ls='--', lw=0.5)

    return [ax1, ax2]


def plot_full_mesh(nodes, tris, regions, grid_dict, ax=None, title=None):
    """Plot the full mesh with regions color-coded."""
    p = grid_dict['params']
    h = p['h']

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))

    region_colors = {0: 'steelblue', 1: 'coral', 2: 'gray'}
    region_labels = {0: 'Inner (octagon)', 1: 'Transition (corners)', 2: 'Outer (Cartesian)'}

    for k, tri in enumerate(tris):
        pts = nodes[[*tri, tri[0]]]
        c = region_colors.get(regions[k], 'purple')
        ax.plot(pts[:, 0], pts[:, 1], '-', color=c, lw=0.3, alpha=0.5)

    for r in [0, 1, 2]:
        mask = regions == r
        if mask.any():
            tri_masked = tris[mask]
            centroids = nodes[tri_masked].mean(axis=1)
            ax.plot(centroids[:, 0], centroids[:, 1], 'o',
                    color=region_colors[r], ms=2, alpha=0.8, label=region_labels[r])

    oct_v = grid_dict['oct_verts']
    oct_closed = np.vstack([oct_v, oct_v[0]])
    ax.plot(oct_closed[:, 0], oct_closed[:, 1], 'k-', lw=1.5)
    sq = np.array([[-h, -h], [h, -h], [h, h], [-h, h], [-h, -h]])
    ax.plot(sq[:, 0], sq[:, 1], 'k--', lw=1.0, alpha=0.4)

    ax.set_aspect('equal')
    margin = 1.5 * h
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_title(title or f"Full mesh  nodes={len(nodes)}  tris={len(tris)}")
    ax.legend(loc='upper right', fontsize=7)
    ax.axis('off')
    return ax


def plot_quadrature_points(pts, w, grid_dict, ax=None, title=None):
    """Plot quadrature points colored by weight."""
    p = grid_dict['params']
    h = p['h']

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    sc = ax.scatter(pts[:, 0], pts[:, 1], c=w, s=10, cmap='viridis',
                    edgecolors='none', alpha=0.8)
    plt.colorbar(sc, ax=ax, label='weight', shrink=0.7)

    oct_v = grid_dict['oct_verts']
    oct_closed = np.vstack([oct_v, oct_v[0]])
    ax.plot(oct_closed[:, 0], oct_closed[:, 1], 'k-', lw=1.0, alpha=0.5)
    sq = np.array([[-h, -h], [h, -h], [h, h], [-h, h], [-h, -h]])
    ax.plot(sq[:, 0], sq[:, 1], 'k--', lw=0.8, alpha=0.3)

    ax.set_aspect('equal')
    margin = 1.2 * h
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_title(title or f"Quadrature points  N={len(pts)}  Σw={w.sum():.3f}")
    ax.axis('off')
    return ax


def main():
    import os
    os.makedirs(IMG_DIR, exist_ok=True)

    grid_dict = build_octagon_grid(d=0.5, n=10, alpha=1.8, n_blend=4)
    p = grid_dict['params']
    print(f"Octagon grid: Nu={p['Nu']}, Nv={p['Nv']}, h={p['h']:.3f}")
    print(f"  Corner cut s={p['s']:.3f}, axis edge half-length hs={p['hs']:.3f}")
    print(f"  Angular spacing: {p['angular_spacing']:.3f}")
    print(f"  Cell areas: min={grid_dict['cell_areas'].min():.4f}, max={grid_dict['cell_areas'].max():.4f}")
    print(f"  Aspect ratios: mean={grid_dict['cell_aspect'].mean():.3f}, max={grid_dict['cell_aspect'].max():.3f}")

    # Fig 1: Grid structure
    fig1, ax = plt.subplots(figsize=(7, 7))
    plot_grid_structure(grid_dict, ax=ax)
    fig1.savefig(f"{IMG_DIR}/octagon_grid_structure.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Fig 2: Three grids comparison
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    plot_three_grids(grid_dict, ax=axes2)
    fig2.savefig(f"{IMG_DIR}/octagon_three_grids.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Fig 3: Full mesh with transition topology
    print("\nBuilding full mesh...")
    nodes, tris, regions = build_full_mesh(grid_dict, margin=5)
    n_inner = (regions == 0).sum()
    n_trans = (regions == 1).sum()
    n_outer = (regions == 2).sum()
    print(f"  Nodes: {len(nodes)}, Triangles: {len(tris)}")
    print(f"  Inner: {n_inner}, Transition: {n_trans}, Outer: {n_outer}")

    fig3, ax = plt.subplots(figsize=(9, 9))
    plot_full_mesh(nodes, tris, regions, grid_dict, ax=ax)
    fig3.savefig(f"{IMG_DIR}/octagon_full_mesh.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Fig 4: Both quadrature modes side by side
    pts_v, w_v = extract_vertex_points(nodes, tris, regions)
    pts_c, w_c = extract_cell_center_points(nodes, tris, regions)
    print(f"\n  Vertex mode:     {len(pts_v)} points, Σw={w_v.sum():.4f}")
    print(f"  Cell-center mode: {len(pts_c)} points, Σw={w_c.sum():.4f}")

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 7))
    plot_quadrature_points(pts_v, w_v, grid_dict, ax=ax4a,
                           title=f"Vertex mode  N={len(pts_v)}  Σw={w_v.sum():.3f}")
    plot_quadrature_points(pts_c, w_c, grid_dict, ax=ax4b,
                           title=f"Cell-center mode  N={len(pts_c)}  Σw={w_c.sum():.3f}")
    fig4.savefig(f"{IMG_DIR}/octagon_quadrature_modes.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Fig 5: Zoom on one corner transition
    fig5, ax = plt.subplots(figsize=(7, 7))
    plot_full_mesh(nodes, tris, regions, grid_dict, ax=ax,
                   title="Full mesh (zoom on NE corner)")
    h = p['h']
    ax.set_xlim(-0.2 * h, 1.8 * h)
    ax.set_ylim(-0.2 * h, 1.8 * h)
    fig5.savefig(f"{IMG_DIR}/octagon_corner_zoom.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Fig 6: Cell quality
    fig6, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plot_cell_quality(grid_dict, ax=[ax1, ax2])
    fig6.savefig(f"{IMG_DIR}/octagon_cell_quality.png", dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nSaved 6 figures to {IMG_DIR}/")


if __name__ == "__main__":
    main()
