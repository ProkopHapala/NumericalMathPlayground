#!/usr/bin/env python3
"""Elasticity solver benchmark: generate hexagonal grid, build sparse neighbor matrix, visualize.

Thin wrapper that imports solver utilities from TrussSolver and plotting from TrussPlotting.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from TrussSolver import build_stiffness_hex
from TrussPlotting import plot_hex_mesh, plot_matrix


def generate_hex_grid(nx: int, ny: int, a: float = 1.0) -> np.ndarray:
    """Generate hexagonal/triangular lattice points.
    Returns array of shape (nnode, 2) with (x, y) coordinates.
    """
    dx = a
    dy = a * np.sqrt(3.0) / 2.0
    nnode = nx * ny
    pos = np.zeros((nnode, 2), dtype=np.float64)
    for iy in range(ny):
        for ix in range(nx):
            idx = iy * nx + ix
            x = ix * dx
            if iy % 2 == 1:
                x += dx * 0.5
            y = iy * dy
            pos[idx] = (x, y)
    return pos


def perturb_positions(pos: np.ndarray, sigma: float) -> np.ndarray:
    """Add Gaussian noise to positions."""
    if sigma <= 0.0:
        return pos.copy()
    return pos + np.random.normal(scale=sigma, size=pos.shape)


def build_neighbors_hex(pos: np.ndarray, nx: int, ny: int, n_neigh_max: int = 8) -> np.ndarray:
    """Build neighbor index matrix for a proper triangular grid.
    Returns index array of shape (nnode, n_neigh_max) padded with -1.
    Interior nodes have 6 neighbors forming equilateral triangles.
    """
    nnode = pos.shape[0]
    neighs = np.full((nnode, n_neigh_max), -1, dtype=np.int32)
    for iy in range(ny):
        for ix in range(nx):
            idx = iy * nx + ix
            nbrs = []
            # horizontal neighbors
            if ix + 1 < nx:
                nbrs.append(iy * nx + (ix + 1))
            if ix - 1 >= 0:
                nbrs.append(iy * nx + (ix - 1))
            if iy % 2 == 0:
                # even row: point sits above the gap of the row below
                if iy + 1 < ny:
                    nbrs.append((iy + 1) * nx + ix)       # up-right
                    if ix - 1 >= 0:
                        nbrs.append((iy + 1) * nx + (ix - 1))  # up-left
                if iy - 1 >= 0:
                    nbrs.append((iy - 1) * nx + ix)       # down-right
                    if ix - 1 >= 0:
                        nbrs.append((iy - 1) * nx + (ix - 1))  # down-left
            else:
                # odd row: point sits below the gap of the row above
                if iy + 1 < ny:
                    nbrs.append((iy + 1) * nx + ix)       # up-left
                    if ix + 1 < nx:
                        nbrs.append((iy + 1) * nx + (ix + 1))  # up-right
                if iy - 1 >= 0:
                    nbrs.append((iy - 1) * nx + ix)       # down-left
                    if ix + 1 < nx:
                        nbrs.append((iy - 1) * nx + (ix + 1))  # down-right
            for j, nb in enumerate(nbrs):
                if j < n_neigh_max:
                    neighs[idx, j] = nb
    return neighs


def main():
    parser = argparse.ArgumentParser(description="Elasticity benchmark: hex grid sparse matrix generation and visualization.")
    parser.add_argument('--nx', type=int, default=16, help='Nodes in x direction')
    parser.add_argument('--ny', type=int, default=16, help='Nodes in y direction')
    parser.add_argument('--a', type=float, default=1.0, help='Lattice spacing')
    parser.add_argument('--pos-sigma', type=float, default=0.1, help='Position perturbation stddev')
    parser.add_argument('--k0', type=float, default=1.0, help='Base stiffness')
    parser.add_argument('--k-sigma', type=float, default=0.2, help='Stiffness relative perturbation stddev')
    parser.add_argument('--n-neigh-max', type=int, default=8, help='Max neighbors per node')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save', type=str, default=None, help='Save figures to prefix (prefix_mesh.png, prefix_neigh.png, prefix_stiff.png)')
    parser.add_argument('--no-show', action='store_true', help='Do not call plt.show()')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Generate
    pos0 = generate_hex_grid(args.nx, args.ny, a=args.a)
    pos = perturb_positions(pos0, args.pos_sigma)
    neighs = build_neighbors_hex(pos, args.nx, args.ny, n_neigh_max=args.n_neigh_max)
    stiff = build_stiffness_hex(pos.shape[0], neighs, args.k0, args.k_sigma)

    # Summary
    nnode = pos.shape[0]
    n_edges = np.sum(neighs >= 0)
    print(f"Nodes: {nnode}")
    print(f"Edges (directed): {n_edges}")
    print(f"Avg neighbors per node: {n_edges / nnode:.2f}")
    print(f"Neighbor index matrix shape: {neighs.shape}")
    print(f"Stiffness matrix shape: {stiff.shape}")
    print(f"Stiffness range: [{stiff[neighs >= 0].min():.4f}, {stiff[neighs >= 0].max():.4f}]")

    # Plot
    fig_mesh, ax_mesh = plt.subplots(figsize=(6, 6))
    plot_hex_mesh(pos, neighs, title=f"Hex Grid ({args.nx}x{args.ny}), pos_σ={args.pos_sigma}", ax=ax_mesh)
    plt.tight_layout()

    fig_neigh, ax_neigh = plt.subplots(figsize=(8, 6))
    ax_neigh, im_neigh = plot_matrix(neighs, title="Neighbor Index Matrix", cmap='tab20', ax=ax_neigh)
    plt.colorbar(im_neigh, ax=ax_neigh)
    plt.tight_layout()

    fig_stiff, ax_stiff = plt.subplots(figsize=(8, 6))
    ax_stiff, im_stiff = plot_matrix(stiff, title="Stiffness Matrix", cmap='plasma', ax=ax_stiff)
    plt.colorbar(im_stiff, ax=ax_stiff)
    plt.tight_layout()

    if args.save:
        fig_mesh.savefig(f"{args.save}_mesh.png", dpi=150)
        fig_neigh.savefig(f"{args.save}_neigh.png", dpi=150)
        fig_stiff.savefig(f"{args.save}_stiff.png", dpi=150)
        print(f"Saved figures with prefix: {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
