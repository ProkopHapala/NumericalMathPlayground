"""
demo_clustering.py — CLI demo for clustering and nearest-neighbor search.

Thin wrapper that calls core (Pointcloud, Clustering) and plotting
(ClusteringPlotting) modules.  No complex functions defined here.

Usage
-----
    python demo_clustering.py --mode kmeans   --dist blobs --k 5 --n 500
    python demo_clustering.py --mode kmeans   --dist random_walk --k 5 --n 1000
    python demo_clustering.py --mode kdtree   --dist blobs --n 300 --nn_k 3
    python demo_clustering.py --mode balltree --dist blobs --n 300 --nn_k 3
    python demo_clustering.py --mode compare  --dist blobs --n 300 --nn_k 3
    python demo_clustering.py --mode distributions --n 500
"""
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

from Pointcloud import gen_pointcloud
from Clustering import (
    kmeans, kmeans_multiple_runs, build_kdtree, kdtree_all_nn,
    build_balltree, balltree_all_nn, brute_force_nn, evaluate_clustering,
)
from ClusteringPlotting import (
    plot_pointcloud, plot_clusters, plot_convergence, plot_kdtree_splits,
    plot_nn_graph, plot_silhouette, plot_tree_comparison, plot_metrics_bar,
)


def mode_kmeans(args):
    P, gt_centers = gen_pointcloud(args.dist, args.n, d=2, seed=args.seed, k=args.k)
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    # Top-left: point cloud with ground truth
    plot_pointcloud(P, centers=gt_centers, ax=axes[0, 0], title=f"Input: {args.dist} (n={args.n})")
    # Run K-means with kmeans++ init
    res = kmeans(P, args.k, init='kmeans++', max_iter=args.max_iter, seed=args.seed)
    # Top-right: clustering result
    plot_clusters(P, res['labels'], res['centroids'], ax=axes[0, 1],
                  title=f"K-means++ (k={args.k}, inertia={res['inertia']:.2f})")
    # Bottom-left: convergence
    plot_convergence(res['history'], ax=axes[1, 0], title="Inertia vs Iteration")
    # Bottom-right: silhouette
    metrics = evaluate_clustering(P, res['labels'], res['centroids'])
    plot_silhouette(res['labels'], metrics['silhouette_per'], args.k, ax=axes[1, 1],
                    title=f"Silhouette (mean={metrics['silhouette']:.3f})")
    print(f"\n=== K-means Results ===")
    print(f"  inertia:     {metrics['inertia']:.4f}")
    print(f"  balance:     {metrics['balance']:.4f}")
    print(f"  separation:  {metrics['separation']:.4f}")
    print(f"  silhouette:  {metrics['silhouette']:.4f}")
    print(f"  n_iter:      {res['n_iter']}")
    print(f"  converged:   {res['converged']}")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")
    if not args.no_show:
        plt.show()


def mode_kmeans_compare_init(args):
    """Compare random vs kmeans++ initialization."""
    P, gt_centers = gen_pointcloud(args.dist, args.n, d=2, seed=args.seed, k=args.k)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # Row 1: random init
    res_rand = kmeans(P, args.k, init='random', max_iter=args.max_iter, seed=args.seed)
    plot_clusters(P, res_rand['labels'], res_rand['centroids'], ax=axes[0, 0],
                  title=f"Random init (inertia={res_rand['inertia']:.2f})")
    plot_convergence(res_rand['history'], ax=axes[0, 1], title="Random: Convergence")
    plot_silhouette(res_rand['labels'], evaluate_clustering(P, res_rand['labels'], res_rand['centroids'])['silhouette_per'],
                    args.k, ax=axes[0, 2], title="Random: Silhouette")
    # Row 2: kmeans++ init
    res_pp = kmeans(P, args.k, init='kmeans++', max_iter=args.max_iter, seed=args.seed)
    plot_clusters(P, res_pp['labels'], res_pp['centroids'], ax=axes[1, 0],
                  title=f"K-means++ init (inertia={res_pp['inertia']:.2f})")
    plot_convergence(res_pp['history'], ax=axes[1, 1], title="K-means++: Convergence")
    plot_silhouette(res_pp['labels'], evaluate_clustering(P, res_pp['labels'], res_pp['centroids'])['silhouette_per'],
                    args.k, ax=axes[1, 2], title="K-means++: Silhouette")
    print(f"\n=== Init Comparison ===")
    print(f"  random:    inertia={res_rand['inertia']:.4f}  n_iter={res_rand['n_iter']}")
    print(f"  kmeans++:  inertia={res_pp['inertia']:.4f}   n_iter={res_pp['n_iter']}")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")
    if not args.no_show:
        plt.show()


def mode_kdtree(args):
    P, _ = gen_pointcloud(args.dist, args.n, d=2, seed=args.seed, k=args.k)
    root = build_kdtree(P, leaf_size=args.leaf_size)
    nn_idx, nn_dist = kdtree_all_nn(P, k=args.nn_k, leaf_size=args.leaf_size)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plot_kdtree_splits(root, P, ax=axes[0], title=f"KD-tree Splits (leaf_size={args.leaf_size})")
    plot_nn_graph(P, nn_idx, ax=axes[1], title=f"KD-tree {args.nn_k}-NN Graph")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")
    if not args.no_show:
        plt.show()


def mode_balltree(args):
    P, _ = gen_pointcloud(args.dist, args.n, d=2, seed=args.seed, k=args.k)
    nn_idx, nn_dist = balltree_all_nn(P, k=args.nn_k, leaf_size=args.leaf_size)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_nn_graph(P, nn_idx, ax=ax, title=f"Ball Tree {args.nn_k}-NN Graph (leaf_size={args.leaf_size})")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")
    if not args.no_show:
        plt.show()


def mode_compare(args):
    """Compare KD-tree, ball tree, and brute force NN search."""
    P, _ = gen_pointcloud(args.dist, args.n, d=2, seed=args.seed, k=args.k)
    import time
    t0 = time.time()
    kd_idx, kd_dist = kdtree_all_nn(P, k=args.nn_k, leaf_size=args.leaf_size)
    t_kd = time.time() - t0
    t0 = time.time()
    ball_idx, ball_dist = balltree_all_nn(P, k=args.nn_k, leaf_size=args.leaf_size)
    t_ball = time.time() - t0
    t0 = time.time()
    brute_idx, brute_dist = brute_force_nn(P, k=args.nn_k)
    t_brute = time.time() - t0
    fig, axes = plot_tree_comparison(P, kd_idx, kd_dist, ball_idx, ball_dist, brute_idx, brute_dist, k=args.nn_k)
    print(f"\n=== NN Search Comparison (n={args.n}, k={args.nn_k}) ===")
    print(f"  KD-tree:     {t_kd:.4f}s")
    print(f"  Ball tree:   {t_ball:.4f}s")
    print(f"  Brute force: {t_brute:.4f}s")
    kd_match = np.all(kd_idx == brute_idx)
    ball_match = np.all(ball_idx == brute_idx)
    print(f"  KD-tree correct:   {kd_match}")
    print(f"  Ball tree correct: {ball_match}")
    if not kd_match:
        mismatches = np.where(kd_idx != brute_idx)[0]
        print(f"  KD-tree mismatches: {len(mismatches)} points")
        print(f"  First few: {mismatches[:10]}")
    if not ball_match:
        mismatches = np.where(ball_idx != brute_idx)[0]
        print(f"  Ball tree mismatches: {len(mismatches)} points")
        print(f"  First few: {mismatches[:10]}")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")
    if not args.no_show:
        plt.show()


def mode_distributions(args):
    """Show all available point cloud distributions side by side."""
    dists = ['uniform', 'gaussian', 'blobs', 'anisotropic', 'random_walk', 'rings']
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for i, dist in enumerate(dists):
        ax = axes[i // 3, i % 3]
        P, centers = gen_pointcloud(dist, args.n, d=2, seed=args.seed, k=5)
        plot_pointcloud(P, centers=centers, ax=ax, title=dist)
    fig.suptitle(f"Point Cloud Distributions (n={args.n})", fontsize=14)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")
    if not args.no_show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Clustering & NN search demos")
    parser.add_argument('--mode', default='kmeans',
                        choices=['kmeans', 'compare_init', 'kdtree', 'balltree', 'compare', 'distributions'],
                        help='Demo mode')
    parser.add_argument('--dist', default='blobs',
                        choices=['uniform', 'gaussian', 'blobs', 'anisotropic', 'random_walk', 'rings'],
                        help='Point cloud distribution')
    parser.add_argument('--n', type=int, default=500, help='Number of points')
    parser.add_argument('--k', type=int, default=5, help='Number of clusters / blob centers')
    parser.add_argument('--nn_k', type=int, default=3, help='Number of nearest neighbors')
    parser.add_argument('--max_iter', type=int, default=100, help='Max K-means iterations')
    parser.add_argument('--leaf_size', type=int, default=10, help='Tree leaf size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', default='clustering_demo.png', help='Output image file')
    parser.add_argument('--no-show', action='store_true', help='Do not show plot window')
    args = parser.parse_args()

    mode_funcs = {
        'kmeans': mode_kmeans,
        'compare_init': mode_kmeans_compare_init,
        'kdtree': mode_kdtree,
        'balltree': mode_balltree,
        'compare': mode_compare,
        'distributions': mode_distributions,
    }
    mode_funcs[args.mode](args)


if __name__ == '__main__':
    main()
