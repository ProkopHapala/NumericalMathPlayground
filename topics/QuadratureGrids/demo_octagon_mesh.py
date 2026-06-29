"""
demo_octagon_mesh.py — Visualize quad-based octagon mesh for different n values.

Shows:
  1) Full mesh (quads, not split to triangles) for several n values
  2) Vertex mode vs cell-center mode quadrature points
  3) Zoom on corner transition topology

Usage:
    python demo_octagon_mesh.py
"""
import numpy as np
import matplotlib.pyplot as plt

from OctagonGrid import (
    build_octagon_grid, build_full_mesh,
    extract_vertex_points, extract_cell_center_points,
    extract_vertex_points_classified, extract_cell_center_points_classified,
    classify_polys, classify_nodes, compute_spherical_weights,
)

IMG_DIR = "Quadrature_Images"


def plot_mesh(nodes, polys, regions, grid_dict, ax=None, title=None,
              show_quads=True, show_points=True, zoom=None):
    """Plot quad-based mesh with regions color-coded."""
    p = grid_dict['params']
    h = p['h']

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    region_colors = {0: 'steelblue', 1: 'coral', 2: 'gray'}
    region_labels = {0: 'Inner (octagon)', 1: 'Transition (corners)', 2: 'Outer (Cartesian)'}

    if show_quads:
        for k, poly in enumerate(polys):
            pts = nodes[[*poly, poly[0]]]
            c = region_colors.get(regions[k], 'purple')
            ax.plot(pts[:, 0], pts[:, 1], '-', color=c, lw=0.3, alpha=0.5)

    if show_points:
        for r in [0, 1, 2]:
            mask = regions == r
            if not mask.any():
                continue
            # Collect unique vertices for this region
            verts = set()
            for poly in np.array(polys, dtype=object)[mask]:
                for v in poly:
                    verts.add(v)
            verts = sorted(verts)
            pts = nodes[verts]
            ax.plot(pts[:, 0], pts[:, 1], 'o',
                    color=region_colors[r], ms=2, alpha=0.8, label=region_labels[r])

    # Octagon + square outline
    oct_v = grid_dict['oct_verts']
    oct_closed = np.vstack([oct_v, oct_v[0]])
    ax.plot(oct_closed[:, 0], oct_closed[:, 1], 'k-', lw=1.5)
    sq = np.array([[-h, -h], [h, -h], [h, h], [-h, h], [-h, -h]])
    ax.plot(sq[:, 0], sq[:, 1], 'k--', lw=1.0, alpha=0.4)

    ax.set_aspect('equal')
    if zoom is not None:
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[2], zoom[3])
    else:
        margin = 1.5 * h
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
    ax.set_title(title or f"Mesh  nodes={len(nodes)}  polys={len(polys)}")
    ax.legend(loc='upper right', fontsize=7)
    ax.axis('off')
    return ax


def plot_quadrature(pts, w, grid_dict, ax=None, title=None):
    """Plot quadrature points colored by weight."""
    p = grid_dict['params']
    h = p['h']

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    sc = ax.scatter(pts[:, 0], pts[:, 1], c=w, s=12, cmap='viridis',
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
    ax.set_title(title or f"N={len(pts)}  Σw={w.sum():.3f}")
    ax.axis('off')
    return ax


def plot_classified(pts, w, labels, grid_dict, ax=None, title=None):
    """Plot quadrature points colored by classification (inner/outer/interface)."""
    p = grid_dict['params']
    h = p['h']

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    color_map = {'inner': 'steelblue', 'outer': 'gray', 'interface': 'coral'}
    for lbl in ['inner', 'outer', 'interface']:
        mask = labels == lbl
        if mask.any():
            ax.scatter(pts[mask, 0], pts[mask, 1], c=color_map[lbl], s=20,
                       edgecolors='none', alpha=0.8, label=f"{lbl} ({mask.sum()})")

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


def main():
    import os
    os.makedirs(IMG_DIR, exist_ok=True)

    d = 0.5
    ns = [4, 6, 8, 10, 14]

    # === Fig 1: Full mesh for different n values (immediate blend) ===
    fig1, axes1 = plt.subplots(1, len(ns), figsize=(5 * len(ns), 5))
    if len(ns) == 1:
        axes1 = [axes1]

    print("=== Full mesh (immediate blend) for different n ===")
    for ax, n in zip(axes1, ns):
        grid_dict = build_octagon_grid(d=d, n=n, alpha=1.8, blend_mode='immediate')
        nodes, polys, regions = build_full_mesh(grid_dict, margin=3)
        n_q = sum(1 for p in polys if len(p) == 4)
        n_t = sum(1 for p in polys if len(p) == 3)
        print(f"  n={n:2d}: nodes={len(nodes):4d}, polys={len(polys):4d} "
              f"(quads={n_q}, tris={n_t})")
        plot_mesh(nodes, polys, regions, grid_dict, ax=ax,
                  title=f"n={n}  quads={n_q}  tris={n_t}",
                  show_points=False)
    fig1.suptitle("Quad-based octagon mesh (immediate blend) for different n", fontsize=14)
    fig1.tight_layout()
    fig1.savefig(f"{IMG_DIR}/octagon_mesh_n_sweep.png", dpi=150, bbox_inches='tight')
    plt.show()

    # === Fig 2: Zoom on NE corner for different n values ===
    fig2, axes2 = plt.subplots(1, len(ns), figsize=(5 * len(ns), 5))
    if len(ns) == 1:
        axes2 = [axes2]

    for ax, n in zip(axes2, ns):
        grid_dict = build_octagon_grid(d=d, n=n, alpha=1.8, blend_mode='immediate')
        nodes, polys, regions = build_full_mesh(grid_dict, margin=3)
        h = grid_dict['params']['h']
        zoom = [-0.3 * h, 1.5 * h, -0.3 * h, 1.5 * h]
        plot_mesh(nodes, polys, regions, grid_dict, ax=ax,
                  title=f"n={n}  (NE corner zoom)",
                  show_points=True, zoom=zoom)
    fig2.suptitle("Corner transition topology for different n", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(f"{IMG_DIR}/octagon_corner_n_sweep.png", dpi=150, bbox_inches='tight')
    plt.show()

    # === Fig 3: Interface classification (cell-center mode, n=10) ===
    grid_dict = build_octagon_grid(d=d, n=10, alpha=1.8, blend_mode='immediate')
    nodes, polys, regions = build_full_mesh(grid_dict, margin=5)
    h = grid_dict['params']['h']

    pts_v, w_v, lbl_v = extract_vertex_points_classified(nodes, polys, regions, grid_dict)
    pts_c, w_c, lbl_c = extract_cell_center_points_classified(nodes, polys, regions, grid_dict)

    n_inner_v = (lbl_v == 'inner').sum()
    n_iface_v = (lbl_v == 'interface').sum()
    n_outer_v = (lbl_v == 'outer').sum()
    n_inner_c = (lbl_c == 'inner').sum()
    n_iface_c = (lbl_c == 'interface').sum()
    n_outer_c = (lbl_c == 'outer').sum()

    print(f"\n=== n=10 interface classification ===")
    print(f"  Vertex mode:     {len(pts_v)} points (inner={n_inner_v}, interface={n_iface_v}, outer={n_outer_v})")
    print(f"  Cell-center mode: {len(pts_c)} points (inner={n_inner_c}, interface={n_iface_c}, outer={n_outer_c})")
    print(f"  Free parameters (interface): vertex={n_iface_v}, cell-center={n_iface_c}")
    print(f"  Σw: vertex={w_v.sum():.4f}, cell-center={w_c.sum():.4f}")

    # Spherical weights for inner cells
    w_sph = compute_spherical_weights(grid_dict)
    print(f"  Spherical weights: {w_sph.shape} (Nu × Nv-1), depends only on j")
    print(f"    shells: {[f'{w_sph[0,j]:.4f}' for j in range(w_sph.shape[1])]}")

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 7))
    plot_classified(pts_v, w_v, lbl_v, grid_dict, ax=ax3a,
                    title=f"Vertex mode  N={len(pts_v)}  (interface={n_iface_v})")
    plot_classified(pts_c, w_c, lbl_c, grid_dict, ax=ax3b,
                    title=f"Cell-center mode  N={len(pts_c)}  (interface={n_iface_c})")
    fig3.suptitle("Interface classification (n=10, immediate blend)", fontsize=14)
    fig3.savefig(f"{IMG_DIR}/octagon_interface_class.png", dpi=150, bbox_inches='tight')
    plt.show()

    # === Fig 4: Zoom on NE corner, classified ===
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))
    zoom = [-0.3 * h, 1.5 * h, -0.3 * h, 1.5 * h]

    color_map = {'inner': 'steelblue', 'outer': 'gray', 'interface': 'coral'}
    for ax, pts, w, lbl, title in [(ax4a, pts_v, w_v, lbl_v, 'Vertex mode'),
                                   (ax4b, pts_c, w_c, lbl_c, 'Cell-center mode')]:
        for l in ['inner', 'outer', 'interface']:
            mask = lbl == l
            if mask.any():
                ax.scatter(pts[mask, 0], pts[mask, 1], c=color_map[l], s=40,
                           edgecolors='k', linewidths=0.3, alpha=0.9, label=l)
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

    fig4.suptitle("Interface classification — NE corner zoom (n=10)", fontsize=14)
    fig4.savefig(f"{IMG_DIR}/octagon_interface_zoom.png", dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nSaved 4 figures to {IMG_DIR}/")


if __name__ == "__main__":
    main()
