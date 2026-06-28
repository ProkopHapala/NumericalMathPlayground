# Clustering & Nearest-Neighbor Search

Algorithms for **clustering** and **nearest-neighbor (NN) search** in arbitrary
dimensions, debugged and visualized in 2D for simplicity.  The focus is on
**fast heuristic** methods that scale to high dimensions and parallelize
naturally — not on exact but slow solutions.

## Why not grid-based (PIC) methods?

Particle-in-cell / uniform grid hashing assigns each point to a cell by
floor-division of coordinates.  NN search then checks only the cell and its
neighbors.  This is O(1) per query in low dimensions but suffers from the
**curse of dimensionality**: the number of neighbor cells to check grows as
3^d, and for non-uniform distributions most cells are empty (wasted memory).
More critically, grid methods require **tuning the cell size** to the data
scale — too small and you miss neighbors, too large and you degenerate to
brute force.  Anisotropic or multi-scale distributions (hierarchical
random-walk clusters) break the assumption of a single characteristic length
scale.  We therefore focus on **data-adaptive** tree structures and
iterative refinement.

## Algorithms

### K-means (Lloyd's algorithm)

Given N points {x_i} in R^d and K clusters, K-means minimizes the
**inertia** (within-cluster sum of squares):

    I = Σ_k Σ_{i∈C_k} ||x_i - μ_k||²

Lloyd's algorithm alternates:
1. **Assignment:** each point → nearest centroid (by Euclidean distance).
2. **Update:** each centroid → mean of assigned points.

This is coordinate descent on I: each step is guaranteed to not increase I,
so it converges to a local minimum.  The quality depends heavily on
**initialization**:

- **Random init:** pick K random points as initial centroids.  Simple but
  often yields poor local minima (e.g., two centroids collapse onto the same
  cluster).
- **K-means++ init:** pick the first centroid uniformly at random, then each
  subsequent centroid is chosen with probability proportional to D(x)²,
  where D(x) is the distance to the nearest already-chosen centroid.  This
  spreads initial centroids across the data and gives an O(log K)
  approximation guarantee in expectation.

**Parallelization:** the assignment step is embarrassingly parallel — each
point's nearest centroid is computed independently.  The update step is a
parallel reduction (sum per cluster).  Both map well to GPU/SIMD.

**Convergence criterion:** stop when the assignment stops changing (labels
stable) or when the relative decrease in inertia falls below a tolerance.

### Voronoi diagrams & Delaunay triangulation

The K-means assignment step partitions space into **Voronoi cells**: the
cell of centroid μ_k is the set of all points closer to μ_k than to any
other centroid.  In 2D these are convex polygons bounded by perpendicular
bisectors.  The **Delaunay triangulation** is the dual graph: two centroids
are connected if their Voronoi cells share an edge.  We use Voronoi
visualization to inspect cluster boundaries but do not compute the full
Delaunay triangulation (which is O(N log N) in 2D but complex in higher
dimensions).

### KD-tree (k-dimensional tree)

A **space-partitioning tree** for fast NN search.  Each node splits the
data along one axis at the median:

- **Build:** O(N log N) — recursively split on the axis of largest spread.
- **Query:** O(log N) average, O(N) worst case (when many points are
  equidistant from the query along the split axes).

**Parallelization:** the tree is built top-down; each subtree is independent
after the split, so subtrees can be built in parallel.  For NN queries,
multiple queries are independent and parallelize trivially.

**Curse of dimensionality:** in high dimensions (d ≳ 20), KD-trees
degenerate toward brute force because the hyper-sphere of the search radius
intersects most splitting planes.  For very high dimensions, approximate
methods (LSH, randomized trees) are preferred.

### Ball tree (sphere tree)

An alternative to KD-tree that splits by **hyperspheres** instead of
axis-aligned planes.  Each node stores a center c and radius r covering all
its points.  Splitting chooses the direction of largest spread and
partitions points into two children, each with its own bounding sphere.

- **Pruning:** for a query point q with current best distance D_best, skip
  a subtree if ||q - c|| - r > D_best (the sphere cannot contain a closer
  point).
- **Advantage over KD-tree:** the bounding sphere adapts to the data shape
  (not axis-aligned), so pruning can be tighter for non-axis-aligned
  distributions.  In high dimensions, ball trees degrade more gracefully
  than KD-trees because the pruning condition is rotation-invariant.

### AABB (Axis-Aligned Bounding Box) tree

Each node stores the min/max corner of the bounding box of its points.
Splitting is similar to KD-tree (axis-aligned split at median).  Pruning for
NN search: skip a subtree if the distance from the query to the box is
greater than D_best.  The distance to an AABB is computed component-wise
(max(0, |q_j - c_j| - half_extent_j) per axis), which is cheaper than the
ball-tree distance but the bounding region is tighter only when the data is
axis-aligned.

## Quality metrics

How do we measure whether a clustering is "good"?  There is no single
number — a good clustering balances several (often conflicting) objectives.
We track the following metrics.

### Inertia (within-cluster sum of squares)

The quantity that K-means explicitly minimizes.  For each cluster k with
centroid μ_k and points {x_i ∈ C_k}:

    I_k = Σ_{i∈C_k} ||x_i - μ_k||²        (per-cluster inertia)
    I   = Σ_k I_k                          (total inertia)

**Interpretation:** inertia is the sum of squared distances from each point
to its assigned centroid.  It measures **compactness** — how tightly points
huddle around their cluster center.  Lower is better.

**Caveat:** inertia always decreases as you add more clusters (K→∞ gives
I→0, since each point becomes its own cluster).  So inertia alone cannot
tell you the "right" K.  It is useful for comparing clusterings with the
same K (e.g., different initializations) or for monitoring convergence of
a single run.

**Units:** squared distance (same units as the data squared).  Not
normalized — depends on the scale of the data and the number of points.

### Silhouette score

A **per-point** measure that combines compactness and separation without
requiring the same K.  For each point x_i in cluster C_k:

    a_i = (1/|C_k|-1) · Σ_{j∈C_k, j≠i} ||x_i - x_j||     (mean intra-cluster distance)
    b_i = min_{l≠k} (1/|C_l|) · Σ_{j∈C_l} ||x_i - x_j||   (mean nearest-cluster distance)
    s_i = (b_i - a_i) / max(a_i, b_i)

**Interpretation:**

- **s_i ≈ +1:** the point is much closer to its own cluster than to any
  other — well-classified.
- **s_i ≈ 0:** the point sits on the boundary between two clusters —
  ambiguous assignment.
- **s_i ≈ -1:** the point is closer to a different cluster than to its own
  — misclassified.

The overall silhouette score is the mean of s_i over all points.  It ranges
from -1 (worst) to +1 (best).  Unlike inertia, it is **dimensionless**
(normalized per point) and can be compared across different K values.

**Cost:** O(n²) — requires all pairwise distances.  For large n, use a
subsample.

### Balance

Measures how evenly points are distributed across clusters:

    balance = 1 - std(n_k) / mean(n_k)

where n_k = |C_k| is the number of points in cluster k.

**Interpretation:**

- **balance ≈ 1:** all clusters have similar size — even partition.
- **balance ≈ 0:** highly imbalanced — some clusters are huge, others
  nearly empty.

**Why it matters:** K-means tends to produce balanced clusters when the
data is uniform, but on skewed distributions it can create tiny or empty
clusters.  Balance is important in applications like load balancing
(distributing work across processors) or quantization (evenly representing
the data).

### Separation

Mean pairwise distance between cluster centroids:

    S = (2 / K(K-1)) · Σ_{j<k} ||μ_j - μ_k||

**Interpretation:** how far apart the cluster centers are from each other.
Higher is better — well-separated clusters are easier to distinguish.

**Caveat:** separation alone is misleading — you can have centroids far
apart but with huge overlap (high inertia).  It must be considered
alongside compactness.  The silhouette score captures this tradeoff
implicitly.

### Summary: what to look at

| Metric        | What it measures  | Range     | Depends on K? | Cost   |
|---------------|-------------------|-----------|---------------|--------|
| Inertia       | Compactness       | [0, ∞)    | Yes (↓ with K)| O(nK)  |
| Silhouette    | Compact + Separate| [-1, +1]  | No            | O(n²)  |
| Balance       | Evenness of sizes | [0, 1]    | Indirectly    | O(n)   |
| Separation    | Centroid spread   | [0, ∞)    | Yes           | O(K²)  |

In practice: **inertia** monitors K-means convergence, **silhouette**
evaluates overall quality and compares different K, **balance** checks for
degenerate clusters, and **separation** gives a quick sanity check on
centroid placement.

## File structure (three-layer architecture)

| Layer | File | Responsibility |
|-------|------|----------------|
| Core | `Pointcloud.py` | Point cloud generation (uniform, Gaussian blobs, random-walk hierarchical, anisotropic) |
| Core | `Clustering.py` | K-means (Lloyd + kmeans++ init), KD-tree, ball tree, NN search, quality metrics |
| Plotting | `ClusteringPlotting.py` | Scatter plots, Voronoi cells, cluster coloring, convergence curves, tree visualization |
| Script | `demo_clustering.py` | CLI wrapper: `--mode kmeans\|kdtree\|balltree\|compare` |
