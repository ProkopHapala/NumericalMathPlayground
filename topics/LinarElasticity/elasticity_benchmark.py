#!/usr/bin/env python3
"""Elasticity solver benchmark: generate hexagonal grid, build sparse neighbor matrix, visualize."""

import argparse
import numpy as np
import matplotlib.pyplot as plt


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


def build_stiffness(nnode: int, neighs: np.ndarray, k0: float, k_sigma: float) -> np.ndarray:
    """Build stiffness value matrix same shape as neighs.
    Base stiffness k0 with relative Gaussian perturbation k_sigma.
    Invalid slots (neigh == -1) remain 0.0.
    """
    k = np.zeros_like(neighs, dtype=np.float64)
    mask = neighs >= 0
    noise = np.random.normal(loc=0.0, scale=k_sigma, size=neighs.shape)
    vals = k0 * (1.0 + noise)
    k[mask] = vals[mask]
    # ensure positive stiffness
    k = np.clip(k, a_min=1e-6, a_max=None)
    k[~mask] = 0.0
    return k


def plot_mesh(pos: np.ndarray, neighs: np.ndarray, title: str = "Mesh") -> None:
    """Plot nodes and edges."""
    fig, ax = plt.subplots(figsize=(6, 6))
    nnode = pos.shape[0]
    for i in range(nnode):
        for j in range(neighs.shape[1]):
            nb = neighs[i, j]
            if nb >= 0:
                ax.plot([pos[i, 0], pos[nb, 0]], [pos[i, 1], pos[nb, 1]], 'k-', lw=0.5, alpha=0.4)
    ax.scatter(pos[:, 0], pos[:, 1], c='royalblue', s=20, zorder=5)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()
    return fig


def plot_matrix(mat: np.ndarray, title: str = "Matrix", cmap: str = 'viridis', invalid_color: str = 'white') -> None:
    """Plot matrix with nearest-neighbor interpolation.
    For integer matrices (neighbor indices), -1 is treated as invalid (NaN)
    and rendered in invalid_color.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_mat = mat.astype(np.float64).copy()
    plot_mat[plot_mat == -1] = np.nan
    im = ax.imshow(plot_mat, aspect='auto', interpolation='nearest', cmap=cmap)
    # Set invalid (-1) color
    im.cmap.set_bad(color=invalid_color)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


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
    stiff = build_stiffness(pos.shape[0], neighs, args.k0, args.k_sigma)

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
    fig_mesh = plot_mesh(pos, neighs, title=f"Hex Grid ({args.nx}x{args.ny}), pos_σ={args.pos_sigma}")
    fig_neigh = plot_matrix(neighs, title="Neighbor Index Matrix", cmap='tab20')
    fig_stiff = plot_matrix(stiff, title="Stiffness Matrix", cmap='plasma')

    if args.save:
        fig_mesh.savefig(f"{args.save}_mesh.png", dpi=150)
        fig_neigh.savefig(f"{args.save}_neigh.png", dpi=150)
        fig_stiff.savefig(f"{args.save}_stiff.png", dpi=150)
        print(f"Saved figures with prefix: {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
