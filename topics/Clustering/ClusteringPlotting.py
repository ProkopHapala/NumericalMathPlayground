"""
ClusteringPlotting.py — Visualization functions for clustering and NN search.

This is the **plotting layer** of the Clustering module system.
All matplotlib visualization functions live here.  No algorithms.

Functions
---------
- plot_pointcloud:     scatter a 2D point cloud, optionally with ground-truth centers
- plot_clusters:       scatter colored by cluster label, with centroids and Voronoi cells
- plot_voronoi:        draw Voronoi cell boundaries (via scipy.spatial.Voronoi)
- plot_convergence:    K-means inertia history vs iteration
- plot_kdtree_splits:  draw KD-tree splitting lines on top of point cloud
- plot_nn_graph:       draw k-NN graph (edges between each point and its k neighbors)
- plot_silhouette:     silhouette bar chart sorted by cluster + score
- plot_tree_comparison: side-by-side NN graph from different methods
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import Voronoi, voronoi_plot_2d


def plot_pointcloud(P, centers=None, ax=None, title="Point Cloud", color='steelblue', s=10, alpha=0.6):
    """Scatter a 2D point cloud. Optionally overlay ground-truth centers as red stars."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(P[:, 0], P[:, 1], c=color, s=s, alpha=alpha, edgecolors='none')
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='*', s=200, zorder=5, edgecolors='black', linewidths=0.5)
    ax.set_aspect('equal')
    ax.set_title(title)
    return ax


def plot_voronoi(centroids, P=None, ax=None, color='black', linewidth=0.8, alpha=0.5):
    """
    Draw Voronoi cell boundaries for the given centroids (2D only).
    Uses scipy.spatial.Voronoi.  If P is given, clips to the data bounding box.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    vor = Voronoi(centroids)
    voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False,
                    line_colors=color, line_width=linewidth, line_alpha=alpha)
    return ax


def plot_clusters(P, labels, centroids, ax=None, title="K-means Clusters",
                  show_voronoi=True, cmap='tab10', s=10, alpha=0.6):
    """
    Scatter points colored by cluster label, with centroids as black stars.
    Optionally draw Voronoi cell boundaries (2D only).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    k = centroids.shape[0]
    colors = plt.get_cmap(cmap, max(k, 10))
    for c in range(k):
        mask = labels == c
        if np.any(mask):
            ax.scatter(P[mask, 0], P[mask, 1], c=[colors(c)], s=s, alpha=alpha, edgecolors='none', label=f'C{c}')
    ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='*', s=250, zorder=5, edgecolors='white', linewidths=0.8)
    if show_voronoi and P.shape[1] == 2 and k >= 2:
        plot_voronoi(centroids, P=P, ax=ax, color='gray', linewidth=0.5, alpha=0.4)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend(fontsize=8, loc='upper right')
    return ax


def plot_convergence(history, ax=None, title="K-means Convergence", ylabel="Inertia"):
    """Plot inertia (or other metric) vs iteration."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(history)), history, 'o-', markersize=4, color='steelblue')
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_kdtree_splits(root, P, ax=None, depth=0, max_depth=20, title="KD-tree Splits"):
    """
    Draw KD-tree splitting lines on top of the 2D point cloud.
    Each internal node's split is drawn as a line spanning the parent's bounding box.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.get_cmap('cool', max_depth + 1)

    def _draw(node, lo, hi, depth):
        if depth > max_depth or node.axis == -1:
            return
        axis = node.axis
        split = node.split
        if axis == 0:
            ax.plot([split, split], [lo[1], hi[1]], color=colors(depth), alpha=0.6, linewidth=0.8)
        else:
            ax.plot([lo[0], hi[0]], [split, split], color=colors(depth), alpha=0.6, linewidth=0.8)
        _draw(node.left, lo, np.where(np.arange(len(lo)) == axis, split, hi), depth + 1)
        _draw(node.right, np.where(np.arange(len(lo)) == axis, split, lo), hi, depth + 1)

    bbox_lo = P.min(axis=0)
    bbox_hi = P.max(axis=0)
    _draw(root, bbox_lo, bbox_hi, 0)
    ax.scatter(P[:, 0], P[:, 1], c='steelblue', s=8, alpha=0.5, edgecolors='none')
    ax.set_aspect('equal')
    ax.set_title(title)
    return ax


def plot_nn_graph(P, nn_idx, ax=None, title="k-NN Graph", color='gray', alpha=0.3, point_size=10):
    """
    Draw the k-NN graph: for each point, draw edges to its k nearest neighbors.
    nn_idx: (n, k) array of neighbor indices.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    n, k = nn_idx.shape
    # build line segments
    segs = []
    for i in range(n):
        for j in range(k):
            segs.append([P[i], P[nn_idx[i, j]]])
    lc = LineCollection(segs, colors=color, alpha=alpha, linewidths=0.5)
    ax.add_collection(lc)
    ax.scatter(P[:, 0], P[:, 1], c='steelblue', s=point_size, alpha=0.6, edgecolors='none', zorder=2)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.autoscale()
    return ax


def plot_silhouette(labels, sil_per, k, ax=None, title="Silhouette Scores"):
    """
    Horizontal bar chart of silhouette scores, grouped by cluster, sorted within cluster.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.get_cmap('tab10', max(k, 10))
    # sort by cluster then by silhouette score within cluster
    order = np.lexsort((sil_per, labels))
    sorted_labels = labels[order]
    sorted_sil = sil_per[order]
    y = np.arange(len(order))
    bar_colors = [colors(c) for c in sorted_labels]
    ax.barh(y, sorted_sil, color=bar_colors, edgecolor='none', height=1.0)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.axvline(x=sorted_sil.mean(), color='red', linewidth=1.0, linestyle='--', label=f'mean={sorted_sil.mean():.3f}')
    ax.set_xlabel("Silhouette score s_i")
    ax.set_ylabel("Points (grouped by cluster)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    return ax


def plot_tree_comparison(P, kd_idx, kd_dist, ball_idx, ball_dist, brute_idx, brute_dist,
                         k=1, figsize=(18, 6)):
    """
    Side-by-side comparison of k-NN graphs from KD-tree, ball tree, and brute force.
    Verifies that all three produce identical neighbors.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    plot_nn_graph(P, kd_idx, ax=axes[0], title=f"KD-tree (k={k})")
    plot_nn_graph(P, ball_idx, ax=axes[1], title=f"Ball tree (k={k})")
    plot_nn_graph(P, brute_idx, ax=axes[2], title=f"Brute force (k={k})")
    # check correctness
    kd_match = np.all(kd_idx == brute_idx)
    ball_match = np.all(ball_idx == brute_idx)
    dist_match = np.allclose(kd_dist, brute_dist, atol=1e-10) and np.allclose(ball_dist, brute_dist, atol=1e-10)
    fig.suptitle(f"KD-tree match: {kd_match}  |  Ball tree match: {ball_match}  |  Distances match: {dist_match}",
                 fontsize=11, y=0.98)
    plt.tight_layout()
    return fig, axes


def plot_metrics_bar(metrics_dict, ax=None, title="Clustering Quality Metrics"):
    """
    Bar chart of multiple quality metrics (balance, separation, silhouette).
    metrics_dict: {name: value}
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    ax.bar(names, values, color='steelblue', alpha=0.7)
    ax.set_ylim(0, max(values) * 1.2 if values else 1)
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    return ax
