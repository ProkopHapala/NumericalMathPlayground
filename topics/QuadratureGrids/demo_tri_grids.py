"""
demo_tri_grids.py — Demo for triangular and wedge grids.

Usage:
    python demo_tri_grids.py --mode barycentric
    python demo_tri_grids.py --mode general
    python demo_tri_grids.py --mode wedge
    python demo_tri_grids.py --mode lagrange
    python demo_tri_grids.py --mode all
"""
import argparse
import matplotlib.pyplot as plt

from TriGrids import (
    barycentric_grid, subdivide_triangles, general_tri_grid,
    wedge_grid, barycentric_lagrange_basis,
)
from GridPlotting import (
    compare_tri_levels, plot_tri_grid,
    compare_general_tri_grids, plot_general_tri_grid,
    compare_wedge_grids, plot_wedge_grid,
)

IMG_DIR = "Quadrature_Images"


def mode_barycentric():
    fig = compare_tri_levels(levels=(1, 2, 3, 4, 5))
    fig.savefig(f"{IMG_DIR}/tri_subdivision_levels.png", dpi=150, bbox_inches='tight')
    plt.show()

    fig2, ax = plt.subplots(figsize=(7, 7))
    plot_tri_grid(4, ax=ax, show_indices=True, show_bary=True,
                  title="Barycentric grid n=4 (indices shown as i,j,k)")
    fig2.savefig(f"{IMG_DIR}/tri_subdivision_n4_labeled.png", dpi=150, bbox_inches='tight')
    plt.show()


def mode_general():
    fig = compare_general_tri_grids([(3, 3), (3, 5), (5, 3), (4, 6), (6, 4)])
    fig.savefig(f"{IMG_DIR}/tri_general_grids.png", dpi=150, bbox_inches='tight')
    plt.show()

    fig2, ax = plt.subplots(figsize=(7, 7))
    plot_general_tri_grid(3, 5, ax=ax, show_indices=True,
                          title="Generalized grid (nu=3, nv=5) — orange = diagonal nodes")
    fig2.savefig(f"{IMG_DIR}/tri_general_n3v5_labeled.png", dpi=150, bbox_inches='tight')
    plt.show()

    print("\nGeneralized grid statistics:")
    print(f"{'nu':>4} {'nv':>4} {'nodes':>6} {'tris':>6} {'diag_extra':>10}")
    for nu, nv in [(3, 3), (3, 5), (5, 3), (4, 6), (6, 4), (4, 7), (7, 4)]:
        nodes, tris, idx = general_tri_grid(nu, nv)
        n_grid = max(idx.values()) + 1
        n_diag = len(nodes) - n_grid
        print(f"{nu:4d} {nv:4d} {len(nodes):6d} {len(tris):6d} {n_diag:10d}")


def mode_wedge():
    fig = compare_wedge_grids([
        (6, 6), (6, 4), (6, 3), (8, 4), (10, 5),
    ], v0=0.1)
    fig.savefig(f"{IMG_DIR}/wedge_grids_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()

    fig2, ax = plt.subplots(figsize=(7, 7))
    plot_wedge_grid(8, 5, 0.1, ax=ax, show_layers=True,
                    title="Wedge grid n=8 m=5 v0=0.1\n"
                          "colors = radial layers (inner->outer)")
    fig2.savefig(f"{IMG_DIR}/wedge_grid_layers.png", dpi=150, bbox_inches='tight')
    plt.show()

    fig3, ax = plt.subplots(figsize=(7, 7))
    plot_wedge_grid(6, 4, 0.1, ax=ax, show_indices=True,
                    title="Wedge grid n=6 m=4 v0=0.1 (symmetric about x=y)")
    fig3.savefig(f"{IMG_DIR}/wedge_grid_indices.png", dpi=150, bbox_inches='tight')
    plt.show()

    fig4, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, v0 in zip(axes, [0.0, 0.1, 0.3, 0.5]):
        plot_wedge_grid(8, 4, v0, ax=ax)
    fig4.suptitle("Effect of v0 (inner edge position)", y=0.98)
    fig4.tight_layout()
    fig4.savefig(f"{IMG_DIR}/wedge_grids_v0.png", dpi=150, bbox_inches='tight')
    plt.show()

    print("\nWedge grid statistics:")
    print(f"{'n':>3} {'m':>3} {'v0':>5} {'clip':>4} {'nodes':>6} {'tris':>5} "
          f"{'inner_pts':>9} {'outer_pts':>9}")
    for n, m, v0 in [
        (6, 6, 0.0), (6, 4, 0.1), (6, 3, 0.1),
        (8, 4, 0.1), (8, 5, 0.1), (10, 5, 0.1),
        (10, 7, 0.2), (12, 6, 0.1),
    ]:
        nodes, tris, bary, info = wedge_grid(n, m, v0)
        inner = info['counts'][0]
        outer = info['counts'][m]
        print(f"{n:3d} {m:3d} {v0:5.2f} {n-m:4d} {len(nodes):6d} {len(tris):5d} "
              f"{inner:9d} {outer:9d}")


def mode_lagrange():
    import numpy as np
    n = 3
    test_pts = np.array([[0.3, 0.2], [0.5, 0.5], [0.1, 0.8]])
    vals, nodes = barycentric_lagrange_basis(test_pts, n)
    sums = vals.sum(axis=1)
    print(f"n={n}: Lagrange basis sum at test points (should be 1): {sums}")
    print(f"  Node count: {len(nodes)}")

    for n in [1, 2, 3, 4, 5]:
        test_pts = np.array([[0.3, 0.2], [0.5, 0.5], [0.1, 0.8]])
        vals, nodes = barycentric_lagrange_basis(test_pts, n)
        sums = vals.sum(axis=1)
        max_err = max(abs(s - 1.0) for s in sums)
        print(f"  n={n}: nodes={len(nodes)}, partition of unity max error={max_err:.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triangular grid demos")
    parser.add_argument('--mode', default='all',
                        choices=['barycentric', 'general', 'wedge', 'lagrange', 'all'])
    args = parser.parse_args()

    if args.mode in ('barycentric', 'all'):
        mode_barycentric()
    if args.mode in ('general', 'all'):
        mode_general()
    if args.mode in ('wedge', 'all'):
        mode_wedge()
    if args.mode in ('lagrange', 'all'):
        mode_lagrange()
