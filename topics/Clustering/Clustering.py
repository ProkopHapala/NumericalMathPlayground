"""
Clustering.py — Clustering algorithms and nearest-neighbor search structures.

This is the **algorithm layer** of the Clustering module system.
It provides:

1. **Distance computations** — pairwise distances, nearest-centroid assignment.
2. **K-means** — Lloyd's algorithm with kmeans++ initialization, mini-batch
   variant, convergence tracking.
3. **KD-tree** — space-partitioning tree for exact NN search in any dimension.
4. **Ball tree** — hypersphere-based tree for exact NN search.
5. **Quality metrics** — inertia, silhouette score, cluster balance, separation.

All algorithms work in arbitrary dimension d.  No matplotlib dependency.

Role in the system
------------------
- **Pointcloud.py**: data generation.
- **Clustering.py** (this file): algorithms and data structures.
- **ClusteringPlotting.py**: visualization.
- **demo_clustering.py**: CLI wrapper.
"""
import numpy as np
from collections import deque


# ---------------------------------------------------------------------------
# Distance computations
# ---------------------------------------------------------------------------

def pairwise_sq_dist(A, B):
    """
    Squared Euclidean distances between all pairs of rows of A (n,d) and B (m,d).
    Returns (n, m) array.  Uses the (a-b)^2 = a^2 - 2ab + b^2 identity for
    vectorized computation.
    """
    A_sq = np.sum(A * A, axis=1)[:, None]  # (n, 1)
    B_sq = np.sum(B * B, axis=1)[None, :]  # (1, m)
    D = A_sq - 2 * A @ B.T + B_sq
    return np.maximum(D, 0)  # clip negative from floating-point error


def assign_nearest(P, centroids):
    """
    Assign each point in P (n,d) to its nearest centroid (k,d).
    Returns labels (n,) and squared distances (n,).
    """
    D = pairwise_sq_dist(P, centroids)  # (n, k)
    labels = np.argmin(D, axis=1)
    sq_dists = D[np.arange(len(P)), labels]
    return labels, sq_dists


# ---------------------------------------------------------------------------
# K-means
# ---------------------------------------------------------------------------

def kmeans_plusplus_init(P, k, seed=None):
    """
    K-means++ initialization.  Picks k centroids from P (n,d) with probability
    proportional to squared distance to nearest already-chosen centroid.
    This spreads initial centroids across the data and gives O(log k)
    approximation guarantee in expectation.
    """
    rng = np.random.default_rng(seed)
    n = P.shape[0]
    centroids = np.empty((k, P.shape[1]))
    # first centroid: uniform random
    idx0 = rng.integers(n)
    centroids[0] = P[idx0]
    # squared distances to nearest centroid so far
    D_sq = pairwise_sq_dist(P, centroids[:1]).ravel()
    for i in range(1, k):
        # pick next centroid with prob ~ D_sq
        probs = D_sq / (D_sq.sum() + 1e-30)
        idx = rng.choice(n, p=probs)
        centroids[i] = P[idx]
        new_D = pairwise_sq_dist(P, centroids[i:i+1]).ravel()
        D_sq = np.minimum(D_sq, new_D)
    print(f"#DEBUG kmeans_plusplus_init k={k} seed={seed} init_inertia={D_sq.sum():.4f}")
    return centroids


def kmeans(P, k, init='kmeans++', max_iter=100, tol=1e-6, seed=None, centroids0=None):
    """
    Lloyd's K-means algorithm.
    
    Parameters
    ----------
    P : (n, d) data points
    k : number of clusters
    init : 'kmeans++' | 'random' | 'fixed'
    max_iter : maximum iterations
    tol : relative inertia decrease tolerance for convergence
    seed : random seed for init
    centroids0 : (k, d) initial centroids (used when init='fixed')
    
    Returns
    -------
    result : dict with keys:
        centroids (k,d), labels (n,), inertia (float),
        history (list of inertia per iteration), n_iter (int), converged (bool)
    """
    n, d = P.shape
    if centroids0 is not None:
        centroids = centroids0.copy()
    elif init == 'kmeans++':
        centroids = kmeans_plusplus_init(P, k, seed=seed)
    elif init == 'random':
        rng = np.random.default_rng(seed)
        centroids = P[rng.choice(n, k, replace=False)].copy()
    else:
        raise ValueError(f"Unknown init '{init}'")

    history = []
    prev_inertia = None
    converged = False

    for it in range(max_iter):
        # Assignment step: each point → nearest centroid
        labels, sq_dists = assign_nearest(P, centroids)
        inertia = sq_dists.sum()
        history.append(inertia)

        # Check convergence
        if prev_inertia is not None:
            rel_decrease = (prev_inertia - inertia) / (prev_inertia + 1e-30)
            if rel_decrease < tol:
                converged = True
                print(f"#DEBUG kmeans converged at iter={it} inertia={inertia:.6f} rel_decr={rel_decrease:.2e}")
                break
        prev_inertia = inertia

        # Update step: centroids → mean of assigned points
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=int)
        np.add.at(new_centroids, labels, P)
        np.add.at(counts, labels, 1)
        # handle empty clusters: keep old centroid
        empty = counts == 0
        if np.any(empty):
            print(f"#DEBUG kmeans empty clusters at it={it}: {np.where(empty)[0]}")
            new_centroids[empty] = centroids[empty]
        else:
            new_centroids /= counts[:, None]
        centroids = new_centroids

    # final assignment
    labels, sq_dists = assign_nearest(P, centroids)
    inertia = sq_dists.sum()
    if not history or history[-1] != inertia:
        history.append(inertia)

    print(f"#DEBUG kmeans k={k} n={n} d={d} init={init} n_iter={len(history)} converged={converged} inertia={inertia:.4f}")
    return {
        'centroids': centroids,
        'labels': labels,
        'inertia': inertia,
        'history': history,
        'n_iter': len(history),
        'converged': converged,
    }


def kmeans_multiple_runs(P, k, n_runs=10, init='kmeans++', max_iter=100, tol=1e-6, seed=None):
    """
    Run K-means multiple times with different seeds, return the best (lowest inertia).
    Useful because K-means converges to local minima.
    """
    best = None
    for r in range(n_runs):
        s = seed + r if seed is not None else None
        res = kmeans(P, k, init=init, max_iter=max_iter, tol=tol, seed=s)
        if best is None or res['inertia'] < best['inertia']:
            best = res
            best['run'] = r
    print(f"#DEBUG kmeans_multiple_runs n_runs={n_runs} best_inertia={best['inertia']:.4f} best_run={best['run']}")
    return best


# ---------------------------------------------------------------------------
# KD-tree
# ---------------------------------------------------------------------------

class KDNode:
    __slots__ = ['axis', 'split', 'point', 'idx', 'left', 'right', 'lo', 'hi']
    def __init__(self):
        self.axis = 0
        self.split = 0.0
        self.point = None   # (d,) point at this leaf
        self.idx = -1       # original index
        self.left = None
        self.right = None
        self.lo = None      # bounding box min (d,)
        self.hi = None      # bounding box max (d,)


def build_kdtree(P, leaf_size=10):
    """
    Build a KD-tree from points P (n, d).
    
    Each internal node splits on the axis of largest spread at the median.
    Leaves contain at most leaf_size points (stored as a list of indices).
    
    Returns root KDNode.  Each node has:
      - axis: split axis (internal) or -1 (leaf)
      - lo, hi: bounding box (d,) min and max
      - left, right: children (internal) or None (leaf)
      - point, idx: for leaves with a single point
    """
    n, d = P.shape
    indices = np.arange(n)

    def _build(idx_arr):
        node = KDNode()
        pts = P[idx_arr]
        node.lo = pts.min(axis=0)
        node.hi = pts.max(axis=0)
        if len(idx_arr) <= leaf_size:
            node.axis = -1
            node.point = pts
            node.idx = idx_arr
            node.left = None
            node.right = None
            return node
        # split on axis of largest spread
        spreads = node.hi - node.lo
        axis = int(np.argmax(spreads))
        if spreads[axis] < 1e-12:
            # all points identical on this axis — make leaf
            node.axis = -1
            node.point = pts
            node.idx = idx_arr
            return node
        node.axis = axis
        # median split
        med = np.median(pts[:, axis])
        left_mask = pts[:, axis] <= med
        right_mask = ~left_mask
        # ensure non-empty
        if not np.any(left_mask) or not np.any(right_mask):
            node.axis = -1
            node.point = pts
            node.idx = idx_arr
            return node
        node.split = med
        node.left = _build(idx_arr[left_mask])
        node.right = _build(idx_arr[right_mask])
        return node

    root = _build(indices)
    print(f"#DEBUG build_kdtree n={n} d={d} leaf_size={leaf_size}")
    return root


def kdtree_nn_search(root, P, query, k=1):
    """
    Find k nearest neighbors of query (d,) in the KD-tree built from P (n,d).
    
    Uses best-bin-first search: traverse the tree, pruning subtrees whose
    bounding box is farther than the current k-th best distance.
    
    Returns (indices (k,), distances (k,)) sorted by increasing distance.
    """
    d = P.shape[1]
    # max-heap via sorted list of (dist_sq, idx); keep at most k
    best = []  # list of (dist_sq, global_idx)

    def _dist_sq_to_bbox(q, lo, hi):
        """Squared distance from q to axis-aligned bbox [lo, hi]."""
        delta = np.maximum(0, np.maximum(lo - q, q - hi))
        return np.sum(delta * delta)

    def _search(node):
        if node.axis == -1:
            # leaf: check all points
            pts = node.point  # (m, d)
            idxs = node.idx   # (m,)
            D = np.sum((pts - query[None, :]) ** 2, axis=1)
            for i in range(len(idxs)):
                ds = D[i]
                if len(best) < k:
                    best.append((ds, int(idxs[i])))
                    best.sort()
                elif ds < best[-1][0]:
                    best[-1] = (ds, int(idxs[i]))
                    best.sort()
            return
        # internal: visit near child first, then far child if needed
        diff = query[node.axis] - node.split
        near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)
        _search(near)
        # check if far child could contain a closer point
        if len(best) < k or diff * diff < best[-1][0]:
            _search(far)

    _search(root)
    indices = np.array([b[1] for b in best])
    dists = np.sqrt(np.array([b[0] for b in best]))
    return indices, dists


def kdtree_all_nn(P, k=1, leaf_size=10):
    """
    For each point in P, find its k nearest neighbors (excluding itself).
    Returns (indices (n,k), distances (n,k)).
    """
    root = build_kdtree(P, leaf_size=leaf_size)
    n = P.shape[0]
    all_idx = np.zeros((n, k), dtype=int)
    all_dist = np.zeros((n, k))
    for i in range(n):
        idx, dist = kdtree_nn_search(root, P, P[i], k=k + 1)  # k+1 to exclude self
        # remove self (should be first, distance ~0)
        mask = idx != i
        idx = idx[mask][:k]
        dist = dist[mask][:k]
        all_idx[i] = idx
        all_dist[i] = dist
    print(f"#DEBUG kdtree_all_nn n={n} k={k} leaf_size={leaf_size}")
    return all_idx, all_dist


# ---------------------------------------------------------------------------
# Ball tree
# ---------------------------------------------------------------------------

class BallNode:
    __slots__ = ['center', 'radius', 'point', 'idx', 'left', 'right']
    def __init__(self):
        self.center = None
        self.radius = 0.0
        self.point = None
        self.idx = None
        self.left = None
        self.right = None


def build_balltree(P, leaf_size=10):
    """
    Build a ball tree from points P (n, d).
    
    Each node stores a center and radius covering all its points.
    Splitting: find the dimension of largest spread, pick the two points
    at the extremes as seeds, partition by nearest seed.
    """
    n, d = P.shape
    indices = np.arange(n)

    def _build(idx_arr):
        node = BallNode()
        pts = P[idx_arr]
        node.center = pts.mean(axis=0)
        node.radius = np.max(np.sqrt(np.sum((pts - node.center[None, :]) ** 2, axis=1)))
        if len(idx_arr) <= leaf_size:
            node.point = pts
            node.idx = idx_arr
            node.left = None
            node.right = None
            return node
        # find dimension of largest spread
        spreads = pts.max(axis=0) - pts.min(axis=0)
        axis = int(np.argmax(spreads))
        if spreads[axis] < 1e-12:
            node.point = pts
            node.idx = idx_arr
            return node
        # pick two seed points: min and max along axis
        i_min = int(np.argmin(pts[:, axis]))
        i_max = int(np.argmax(pts[:, axis]))
        p1 = pts[i_min]
        p2 = pts[i_max]
        # partition by nearest seed
        d1 = np.sum((pts - p1[None, :]) ** 2, axis=1)
        d2 = np.sum((pts - p2[None, :]) ** 2, axis=1)
        left_mask = d1 <= d2
        if not np.any(left_mask) or np.all(left_mask):
            node.point = pts
            node.idx = idx_arr
            return node
        node.left = _build(idx_arr[left_mask])
        node.right = _build(idx_arr[~left_mask])
        return node

    root = _build(indices)
    print(f"#DEBUG build_balltree n={n} d={d} leaf_size={leaf_size}")
    return root


def balltree_nn_search(root, P, query, k=1):
    """
    Find k nearest neighbors of query (d,) in the ball tree built from P (n,d).
    
    Pruning: skip a subtree if ||query - center|| - radius > current k-th best distance.
    
    Returns (indices (k,), distances (k,)) sorted by increasing distance.
    """
    best = []  # sorted list of (dist_sq, global_idx), kept at length <= k

    def _search(node):
        if node.left is None:  # leaf
            pts = node.point
            idxs = node.idx
            D = np.sum((pts - query[None, :]) ** 2, axis=1)
            for i in range(len(idxs)):
                ds = D[i]
                if len(best) < k:
                    best.append((ds, int(idxs[i])))
                    best.sort()
                elif ds < best[-1][0]:
                    best[-1] = (ds, int(idxs[i]))
                    best.sort()
            return
        # visit nearer child first, then prune far child
        dl = np.sqrt(np.sum((query - node.left.center) ** 2))
        dr = np.sqrt(np.sum((query - node.right.center) ** 2))
        if dl <= dr:
            near, far, d_far = node.left, node.right, dr
        else:
            near, far, d_far = node.right, node.left, dl
        _search(near)
        # prune: closest possible point in far ball is at distance (d_far - far.radius)
        if len(best) < k or (d_far - far.radius) < np.sqrt(best[-1][0]):
            _search(far)

    _search(root)
    indices = np.array([b[1] for b in best])
    dists = np.sqrt(np.array([b[0] for b in best]))
    return indices, dists


def balltree_all_nn(P, k=1, leaf_size=10):
    """
    For each point in P, find its k nearest neighbors (excluding itself).
    Returns (indices (n,k), distances (n,k)).
    """
    root = build_balltree(P, leaf_size=leaf_size)
    n = P.shape[0]
    all_idx = np.zeros((n, k), dtype=int)
    all_dist = np.zeros((n, k))
    for i in range(n):
        idx, dist = balltree_nn_search(root, P, P[i], k=k + 1)
        mask = idx != i
        idx = idx[mask][:k]
        dist = dist[mask][:k]
        all_idx[i] = idx
        all_dist[i] = dist
    print(f"#DEBUG balltree_all_nn n={n} k={k} leaf_size={leaf_size}")
    return all_idx, all_dist


# ---------------------------------------------------------------------------
# Brute-force NN (reference for validation)
# ---------------------------------------------------------------------------

def brute_force_nn(P, k=1):
    """
    For each point in P, find its k nearest neighbors (excluding itself).
    O(n^2) reference implementation.
    Returns (indices (n,k), distances (n,k)).
    """
    n = P.shape[0]
    D = pairwise_sq_dist(P, P)  # (n, n)
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :k]
    dists = np.sqrt(D[np.arange(n)[:, None], idx])
    print(f"#DEBUG brute_force_nn n={n} k={k}")
    return idx, dists


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def cluster_inertia(P, labels, centroids):
    """Within-cluster sum of squared distances."""
    _, sq_dists = assign_nearest(P, centroids)
    return sq_dists.sum()


def cluster_balance(labels, k):
    """
    Balance metric: 1 - (std of cluster sizes / mean cluster size).
    1.0 = perfectly balanced, 0.0 = maximally imbalanced.
    """
    counts = np.bincount(labels, minlength=k)
    mean_size = counts.mean()
    if mean_size < 1e-30:
        return 0.0
    return 1.0 - np.std(counts) / mean_size


def cluster_separation(centroids):
    """
    Mean pairwise distance between centroids.  Higher = more separated.
    """
    k = centroids.shape[0]
    if k < 2:
        return 0.0
    D = pairwise_sq_dist(centroids, centroids)
    # exclude diagonal (self-distance)
    mask = ~np.eye(k, dtype=bool)
    return np.mean(np.sqrt(D[mask]))


def silhouette_score(P, labels, k):
    """
    Silhouette score: mean over all points of s_i = (b_i - a_i) / max(a_i, b_i).
    a_i = mean intra-cluster distance, b_i = mean nearest-cluster distance.
    Returns (score, per_point_scores).
    
    O(n^2) — use subsampled for large n.
    """
    n = P.shape[0]
    D = pairwise_sq_dist(P, P)  # (n, n)
    s = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if np.sum(same) == 0:
            s[i] = 0.0  # singleton cluster
            continue
        a_i = np.sqrt(D[i, same]).mean()
        b_i = np.inf
        for c in range(k):
            if c == labels[i]:
                continue
            mask_c = labels == c
            if np.sum(mask_c) == 0:
                continue
            d_c = np.sqrt(D[i, mask_c]).mean()
            b_i = min(b_i, d_c)
        s[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 1e-30 else 0.0
    print(f"#DEBUG silhouette_score n={n} k={k} mean_s={s.mean():.4f}")
    return s.mean(), s


def evaluate_clustering(P, labels, centroids):
    """
    Compute all quality metrics for a clustering result.
    Returns dict with inertia, balance, separation, silhouette.
    """
    k = centroids.shape[0]
    inertia = cluster_inertia(P, labels, centroids)
    balance = cluster_balance(labels, k)
    separation = cluster_separation(centroids)
    sil, sil_per = silhouette_score(P, labels, k)
    return {
        'inertia': inertia,
        'balance': balance,
        'separation': separation,
        'silhouette': sil,
        'silhouette_per': sil_per,
    }
