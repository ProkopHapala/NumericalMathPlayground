"""
Pointcloud.py — Point cloud generation for clustering and NN benchmarks.

This is the **data layer** of the Clustering module system.
It generates synthetic point clouds in arbitrary dimensions with various
distributions, suitable for testing clustering algorithms and nearest-neighbor
search structures.

All generators return (N, d) float64 arrays.  2D clouds (d=2) are used for
debugging/visualization; higher dimensions stress-test scalability.

Distributions
-------------
- uniform:        uniform random in [0,1]^d
- gaussian:       single isotropic Gaussian blob
- blobs:          K well-separated Gaussian blobs (classic K-means test)
- anisotropic:    K elongated Gaussian blobs with random covariance orientation
- random_walk:    hierarchical random-walk clusters — each cluster is a random
                  walk starting from a parent seed; produces multi-scale,
                  fractal-like distributions that break single-scale methods
- rings:          concentric rings/annuli (non-convex clusters, hard for K-means)
"""
import numpy as np


def gen_uniform(n, d=2, seed=None):
    """Uniform random points in [0,1]^d."""
    rng = np.random.default_rng(seed)
    P = rng.random((n, d))
    print(f"#DEBUG gen_uniform n={n} d={d} seed={seed}")
    return P


def gen_gaussian(n, d=2, center=None, sigma=1.0, seed=None):
    """Single isotropic Gaussian blob. center: (d,) or None (origin)."""
    rng = np.random.default_rng(seed)
    if center is None:
        center = np.zeros(d)
    P = rng.normal(center, sigma, (n, d))
    print(f"#DEBUG gen_gaussian n={n} d={d} sigma={sigma} seed={seed}")
    return P


def gen_blobs(n, k=5, d=2, spread=3.0, sigma=0.5, seed=None):
    """
    K well-separated isotropic Gaussian blobs.
    Blob centers are placed on a grid or randomly in [0, spread]^d.
    Each blob gets ~n/k points.
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, spread, (k, d))
    per = n // k
    rem = n - per * k
    parts = []
    for i in range(k):
        ni = per + (1 if i < rem else 0)
        parts.append(rng.normal(centers[i], sigma, (ni, d)))
    P = np.vstack(parts)
    rng.shuffle(P)
    print(f"#DEBUG gen_blobs n={n} k={k} d={d} spread={spread} sigma={sigma} seed={seed}")
    return P, centers


def gen_anisotropic(n, k=5, d=2, spread=5.0, sigma_long=1.5, sigma_short=0.2, seed=None):
    """
    K elongated Gaussian blobs with random orientation.
    Each blob has covariance with eigenvalues (sigma_long, sigma_short, ...) 
    and a random rotation in the first 2 axes.
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, spread, (k, d))
    per = n // k
    rem = n - per * k
    parts = []
    for i in range(k):
        ni = per + (1 if i < rem else 0)
        # random rotation in the (0,1) plane
        theta = rng.uniform(0, 2 * np.pi)
        R = np.eye(d)
        R[0, 0] = np.cos(theta); R[0, 1] = -np.sin(theta)
        R[1, 0] = np.sin(theta); R[1, 1] = np.cos(theta)
        sigmas = np.ones(d) * sigma_short
        sigmas[0] = sigma_long
        pts = rng.normal(0, 1, (ni, d)) * sigmas[None, :]
        pts = pts @ R.T
        parts.append(pts + centers[i])
    P = np.vstack(parts)
    rng.shuffle(P)
    print(f"#DEBUG gen_anisotropic n={n} k={k} d={d} spread={spread} sigma_long={sigma_long} sigma_short={sigma_short} seed={seed}")
    return P, centers


def gen_random_walk(n, k=5, d=2, spread=8.0, step_size=0.3, walk_len=None, seed=None):
    """
    Hierarchical random-walk clusters.
    
    Each of K cluster centers is placed randomly in [0, spread]^d.
    From each center, a random walk of length walk_len (= n/k by default)
    generates the cluster points.  This produces multi-scale, fractal-like
    distributions where local density varies and clusters have irregular
    shapes — harder for K-means than Gaussian blobs.
    
    The random walk step is drawn from N(0, step_size) per dimension.
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, spread, (k, d))
    if walk_len is None:
        walk_len = n // k
    per = n // k
    rem = n - per * k
    parts = []
    for i in range(k):
        ni = per + (1 if i < rem else 0)
        steps = rng.normal(0, step_size, (ni, d))
        pts = np.cumsum(steps, axis=0) + centers[i]
        parts.append(pts)
    P = np.vstack(parts)
    rng.shuffle(P)
    print(f"#DEBUG gen_random_walk n={n} k={k} d={d} spread={spread} step_size={step_size} walk_len={walk_len} seed={seed}")
    return P, centers


def gen_rings(n, k=3, d=2, r_min=1.0, r_step=2.0, thickness=0.3, seed=None):
    """
    Concentric rings (annuli) in 2D.  Non-convex clusters that K-means
    handles poorly (it assumes convex Voronoi cells).
    
    Each ring has radius r_min + i*r_step and Gaussian radial thickness.
    Points are uniformly distributed in angle.
    """
    assert d == 2, "gen_rings only supports 2D"
    rng = np.random.default_rng(seed)
    per = n // k
    rem = n - per * k
    parts = []
    centers = np.zeros((k, 2))
    for i in range(k):
        ni = per + (1 if i < rem else 0)
        r = r_min + i * r_step
        angles = rng.uniform(0, 2 * np.pi, ni)
        radii = rng.normal(r, thickness, ni)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        parts.append(pts)
    P = np.vstack(parts)
    rng.shuffle(P)
    print(f"#DEBUG gen_rings n={n} k={k} d={d} r_min={r_min} r_step={r_step} thickness={thickness} seed={seed}")
    return P, centers


# Dispatcher for CLI usage
_GENERATORS = {
    'uniform': lambda n, d, seed, **kw: (gen_uniform(n, d, seed=seed), None),
    'gaussian': lambda n, d, seed, **kw: (gen_gaussian(n, d, seed=seed, **{k: v for k, v in kw.items() if k in ('center', 'sigma')}), None),
    'blobs': lambda n, d, seed, **kw: gen_blobs(n, d=d, seed=seed, **{k: v for k, v in kw.items() if k in ('k', 'spread', 'sigma')}),
    'anisotropic': lambda n, d, seed, **kw: gen_anisotropic(n, d=d, seed=seed, **{k: v for k, v in kw.items() if k in ('k', 'spread', 'sigma_long', 'sigma_short')}),
    'random_walk': lambda n, d, seed, **kw: gen_random_walk(n, d=d, seed=seed, **{k: v for k, v in kw.items() if k in ('k', 'spread', 'step_size', 'walk_len')}),
    'rings': lambda n, d, seed, **kw: gen_rings(n, d=d, seed=seed, **{k: v for k, v in kw.items() if k in ('k', 'r_min', 'r_step', 'thickness')}),
}


def gen_pointcloud(distribution, n, d=2, seed=None, **kwargs):
    """
    Dispatch to the named generator.
    Returns (P, centers) where centers may be None (unknown ground truth).
    """
    if distribution not in _GENERATORS:
        raise ValueError(f"Unknown distribution '{distribution}'. Available: {list(_GENERATORS.keys())}")
    return _GENERATORS[distribution](n, d, seed, **kwargs)
