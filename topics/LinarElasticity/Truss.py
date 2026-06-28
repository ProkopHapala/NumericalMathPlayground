"""
Truss.py — Truss geometry generation, refinement, deformation, IO, and bookkeeping.

This is the **geometry layer** of the LinearElasticity module system.
It is responsible for everything related to building, describing, and
manipulating truss meshes — but knows nothing about how to *solve* them.

Responsibilities
----------------
1. **Grid generation** — structured triangular grids (`build_triangular_grid`,
   `grid_edges`), node indexing (`node_index`), edge lengths (`edge_lengths`).
2. **Coarse-to-fine refinement** — turn a coarse graph into a lattice truss
   with junction circles and regular beams (`refine_truss`, `refine_truss_fan`).
   Three spring types are distinguished: `k_long` (axial), `k_perp`
   (perpendicular), `k_diag` (diagonal bracing).
3. **ASCII art parsing** — parse ASCII truss diagrams into coarse node/edge
   graphs (`parse_ascii_truss`) with trace-through bar detection and
   convergence auto-connection.
4. **Stiffness & mass assembly** (dense) — `assemble_stiffness_dense`,
   `assemble_stiffness_typed`, `mass_matrix`.
5. **Boundary conditions** — `boundary_nodes`, `apply_dirichlet` (DOF removal).
6. **Deformation / distortion** — `distort_noise`, `distort_affine`,
   `distort_bend`, `distort_planewaves`, and the dispatcher `distort`.
7. **High-level interface** — the `Truss` class wraps the full workflow:
   `Truss.from_ascii(...) → refine() → distort() → assemble() → plot()`.
8. **Basic visualization** — `plot_truss` for quick 2D edge-colored rendering.

Role in the system
------------------
- **Truss.py** (this file): geometry, mesh, assembly, bookkeeping.
- **TrussSolver.py**: iterative and direct linear algebra solvers
  (Jacobi, Block Jacobi, alternating patches, heavy-ball momentum).
- **TrussPlotting.py**: reusable plotting functions for convergence curves,
  beam snapshots, partitioning diagrams, spectra, etc.
- Scripts (`BlockJacobiTruss.py`, `VibrationProbing.py`,
  `elasticity_benchmark.py`): thin wrappers that combine the three modules.


Coarse-to-fine refinement
-------------------------
A coarse graph (vertices + edges) is refined into a lattice truss.  Each edge
becomes a beam — a strip of quad cells (boxes) with diagonal cross-bracing.
At each coarse vertex, a circle (radius = thickness/2 by default) is placed;
beam sides are clamped to the circle at angle-bisector points so that adjacent
beams join seamlessly with a rounded corner.

Three spring stiffnesses are distinguished:
  k_long  — along the beam axis (longitudinal struts)
  k_perp  — across the beam width (perpendicular struts)
  k_diag  — diagonal bracing inside each quad cell

3D generalization (not yet implemented):
  In 3D each beam becomes a tube with a circular or rectangular cross-section.
  At each coarse vertex a sphere is placed.  The angle-bisector idea generalises
  to computing the intersection curve of adjacent beam side-planes with the
  sphere, but the sorting of beams around a node is no longer a simple 1-D
  angular sort — one needs a spherical ordering or an explicit face-graph of
  the junction polyhedron.  Left/right side assignment becomes a choice of
  outward normal on each beam's cross-section.
"""
import numpy as np

# DEBUG: utilities for building and assembling truss systems


def node_index(ix, iy, nx):
    return iy * nx + ix


def build_triangular_grid(nx, ny, a=1.0, jitter=0.0, seed=None):
    """
    Build 2D grid of points (z=0) arranged on square lattice with optional jitter.
    Geometry is 3D-compatible (z=0) to keep solver general.
    """
    if seed is not None:
        np.random.seed(seed)
    pos = np.zeros((nx * ny, 3))
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            x = a * ix
            y = a * iy
            if jitter > 0:
                x += jitter * (np.random.rand() - 0.5)
                y += jitter * (np.random.rand() - 0.5)
            pos[idx, 0] = x
            pos[idx, 1] = y
            idx += 1
    print(f"#DEBUG build_triangular_grid nx={nx} ny={ny} a={a} jitter={jitter}")
    return pos


def grid_edges(nx, ny, include_diag=True):
    """
    Create edge list (i,j) connecting nearest neighbors on rectangular grid.
    include_diag adds both diagonals to form triangles.
    """
    edges = []
    for iy in range(ny):
        for ix in range(nx):
            i = node_index(ix, iy, nx)
            if ix + 1 < nx:
                edges.append((i, node_index(ix + 1, iy, nx)))
            if iy + 1 < ny:
                edges.append((i, node_index(ix, iy + 1, nx)))
            if include_diag and ix + 1 < nx and iy + 1 < ny:
                edges.append((i, node_index(ix + 1, iy + 1, nx)))
            if include_diag and ix - 1 >= 0 and iy + 1 < ny:
                edges.append((i, node_index(ix - 1, iy + 1, nx)))
    print(f"#DEBUG grid_edges nx={nx} ny={ny} n_edges={len(edges)} include_diag={include_diag}")
    return edges


def edge_lengths(pos, edges):
    lens = np.zeros(len(edges))
    for idx, (i, j) in enumerate(edges):
        lens[idx] = np.linalg.norm(pos[j] - pos[i])
    print(f"#DEBUG edge_lengths n_edges={len(edges)} l_min={lens.min():.4f} l_max={lens.max():.4f}")
    return lens


def assemble_stiffness_dense(pos, edges, k_spring=1.0, dim=3):
    """
    Assemble dense stiffness matrix for linear springs between nodes.
    K size is (dim*N, dim*N). Contributions are k * (u u^T).
    """
    n_nodes = pos.shape[0]
    ndof = dim * n_nodes
    K = np.zeros((ndof, ndof), dtype=np.float64)
    for (i, j) in edges:
        d = pos[j] - pos[i]
        L = np.linalg.norm(d)
        if L <= 1e-12:
            print(f"#DEBUG assemble_stiffness_dense zero length edge i={i} j={j}")
            continue
        u = d / L
        k_fac = k_spring / (L * L)
        # block indices
        ia = i * dim
        ja = j * dim
        outer = k_fac * np.outer(u, u)
        K[ia:ia+dim, ia:ia+dim] += outer
        K[ja:ja+dim, ja:ja+dim] += outer
        K[ia:ia+dim, ja:ja+dim] -= outer
        K[ja:ja+dim, ia:ia+dim] -= outer
    print(f"#DEBUG assemble_stiffness_dense ndof={ndof} n_edges={len(edges)} k={k_spring}")
    return K


def mass_matrix(masses, dim=3):
    masses = np.asarray(masses)
    n_nodes = masses.size
    diag = np.repeat(masses, dim)
    M = np.diag(diag)
    print(f"#DEBUG mass_matrix n_nodes={n_nodes} dim={dim} m_min={masses.min():.4f} m_max={masses.max():.4f}")
    return M


def boundary_nodes(nx, ny, which="bottom"):
    nodes = []
    if which == "bottom":
        nodes = [node_index(ix, 0, nx) for ix in range(nx)]
    elif which == "top":
        nodes = [node_index(ix, ny - 1, nx) for ix in range(nx)]
    elif which == "left":
        nodes = [node_index(0, iy, nx) for iy in range(ny)]
    elif which == "right":
        nodes = [node_index(nx - 1, iy, nx) for iy in range(ny)]
    elif which == "none":
        nodes = []
    else:
        raise ValueError(f"Unknown boundary selector {which}")
    print(f"#DEBUG boundary_nodes which={which} n={len(nodes)}")
    return nodes


def apply_dirichlet(K, M, fixed_nodes, dim=3):
    """
    Remove DOFs corresponding to fixed nodes. Returns reduced matrices and mask.
    """
    n_nodes = K.shape[0] // dim
    mask = np.ones(n_nodes * dim, dtype=bool)
    for n in fixed_nodes:
        start = n * dim
        mask[start:start+dim] = False
    K_red = K[np.ix_(mask, mask)]
    M_red = M[np.ix_(mask, mask)]
    print(f"#DEBUG apply_dirichlet fixed={len(fixed_nodes)} ndof_full={n_nodes*dim} ndof_red={K_red.shape[0]}")
    return K_red, M_red, mask


def build_test_truss(
    nx=6,
    ny=6,
    a=1.0,
    jitter=0.0,
    k_spring=1.0,
    mass_value=1.0,
    fixed_boundary="none",
    dim=3,
):
    pos = build_triangular_grid(nx, ny, a=a, jitter=jitter)
    edges = grid_edges(nx, ny, include_diag=True)
    K = assemble_stiffness_dense(pos, edges, k_spring=k_spring, dim=dim)
    masses = np.full(nx * ny, mass_value)
    M = mass_matrix(masses, dim=dim)
    fixed = boundary_nodes(nx, ny, which=fixed_boundary)
    if len(fixed) == 0:
        mask = np.ones(K.shape[0], dtype=bool)
        K_red, M_red = K, M
    else:
        K_red, M_red, mask = apply_dirichlet(K, M, fixed, dim=dim)
    print(f"#DEBUG build_test_truss ndof_full={K.shape[0]} ndof_red={K_red.shape[0]} fixed={len(fixed)}")
    return pos, edges, K_red, M_red, mask


# ---------------------------------------------------------------------------
# Coarse-to-fine refinement: turn a coarse graph into a lattice truss
# ---------------------------------------------------------------------------

def _perp2d(u):
    """Counterclockwise perpendicular of a 2D vector embedded in 3D (z=0)."""
    return np.array([-u[1], u[0], 0.0])


def refine_truss_fan(pos, edges, thickness=0.3, radius=None, min_angle_deg=45.0):
    """
    Refine a coarse graph into a fine lattice truss with triangle-fan junctions.

    In this variant, beam sides run from bisector point to bisector point,
    so the end cross-sections may be tilted relative to the beam axis.
    The triangle fan at each junction fills the interior with radial edges
    and chords.

    See :func:`refine_truss` for the variant with undistorted beams.

    Parameters
    ----------
    pos : (N, 3) coarse node positions (z=0 for 2D)
    edges : list of (i, j) coarse edges
    thickness : target cell size (both segment length and beam width)
    radius : junction circle radius override (default: auto from min_angle_deg)
    min_angle_deg : sharpest expected angle between beams (default 45°);

    Returns
    -------
    fine_pos : (M, 3) fine node positions
    fine_edges : list of (i, j, type)  type in {'long','perp','diag'}
    coarse_to_fine : dict {coarse_node -> list of fine node indices at that junction}
    """
    pos = np.asarray(pos, dtype=np.float64)
    edges = list(edges)
    n_coarse = pos.shape[0]
    w = thickness  # beam width
    min_angle_rad = np.radians(min_angle_deg)

    # --- adjacency: node -> [(edge_idx, outward_unit_dir), ...] ---
    adj = [[] for _ in range(n_coarse)]
    for ei, (i, j) in enumerate(edges):
        d = pos[j] - pos[i]
        L = np.linalg.norm(d)
        if L < 1e-12:
            continue
        u = d / L
        adj[i].append((ei, u))
        adj[j].append((ei, -u))

    fe = []  # fine edges
    jmap = {}        # (node, edge_idx, side) -> fine_node_index
    coarse_to_fine = {}
    fp = []          # fine positions list

    # --- Phase 1: junction nodes + fan edges ---
    for v in range(n_coarse):
        inc = adj[v]
        if not inc:
            continue

        if len(inc) == 1:
            # flat cap — two nodes at beam width apart, no fan
            ei, u = inc[0]
            n = _perp2d(u)
            li = len(fp)
            ri = li + 1
            fp.append(pos[v] + (w * 0.5) * n)
            fp.append(pos[v] - (w * 0.5) * n)
            jmap[(v, ei, 'L')] = li
            jmap[(v, ei, 'R')] = ri
            coarse_to_fine[v] = [li, ri]
            continue

        # 2+ edges: sort CCW by angle
        inc.sort(key=lambda x: np.arctan2(x[1][1], x[1][0]))
        k = len(inc)

        # compute per-node radius from actual minimum angle
        if radius is not None:
            r_node = radius
        else:
            min_theta = float('inf')
            for m in range(k):
                u_m = inc[m][1]
                u_n = inc[(m + 1) % k][1]
                dot = np.clip(np.dot(u_m, u_n), -1, 1)
                cross = u_m[0] * u_n[1] - u_m[1] * u_n[0]
                theta = np.arccos(dot)
                if cross < 0:
                    theta = 2 * np.pi - theta
                min_theta = min(min_theta, theta)
            min_theta = min(min_theta, min_angle_rad)
            r_node = (w * 0.5) / np.sin(min_theta * 0.5)

        # center node
        c_idx = len(fp)
        fp.append(pos[v].copy())

        # bisector points + fan edges
        sector_angles = []
        bis_indices = []
        for m in range(k):
            ei_m, u_m = inc[m]
            ei_n, u_n = inc[(m + 1) % k]
            bis = u_m + u_n
            bn = np.linalg.norm(bis)
            cross = u_m[0] * u_n[1] - u_m[1] * u_n[0]
            dot = np.clip(np.dot(u_m, u_n), -1, 1)
            theta = np.arccos(dot)
            if cross < 0:
                theta = 2 * np.pi - theta
            sector_angles.append(theta)

            if bn < 1e-8:
                bis = _perp2d(u_m)
            else:
                bis = bis / bn
                if cross < 0:
                    bis = -bis
            pt = pos[v] + r_node * bis
            idx = len(fp)
            fp.append(pt)
            bis_indices.append(idx)
            jmap[(v, ei_m, 'L')] = idx
            jmap[(v, ei_n, 'R')] = idx

        # radial edges: center -> bisector point, only for sectors with angle < π
        # chord edges: bisector_m -> bisector_m+1, one per sector (deduplicated)
        radial_added = set()
        chord_added = set()
        for m in range(k):
            if sector_angles[m] < np.pi - 1e-10:
                bm = bis_indices[m]
                bn_ = bis_indices[(m + 1) % k]
                rkey = (c_idx, bm) if c_idx < bm else (bm, c_idx)
                if rkey not in radial_added:
                    radial_added.add(rkey)
                    fe.append((c_idx, bm, 'perp'))
                rkey = (c_idx, bn_) if c_idx < bn_ else (bn_, c_idx)
                if rkey not in radial_added:
                    radial_added.add(rkey)
                    fe.append((c_idx, bn_, 'perp'))

            # chord for this sector (beam cross-section at junction)
            bm = bis_indices[m]
            bn_ = bis_indices[(m + 1) % k]
            ckey = (bm, bn_) if bm < bn_ else (bn_, bm)
            if ckey not in chord_added:
                chord_added.add(ckey)
                fe.append((bm, bn_, 'perp'))

        coarse_to_fine[v] = [c_idx] + bis_indices

    # --- Phase 2: beam interior nodes + edges ---
    for ei, (i, j) in enumerate(edges):
        d = pos[j] - pos[i]
        L = np.linalg.norm(d)
        if L < 1e-12:
            continue
        u = d / L
        n = _perp2d(u)

        # endpoint fine-node indices
        iL = jmap[(i, ei, 'L')]
        iR = jmap[(i, ei, 'R')]
        jL = jmap[(j, ei, 'R')]
        jR = jmap[(j, ei, 'L')]

        # beam side endpoints (actual fine positions)
        p_iL, p_iR = fp[iL], fp[iR]
        p_jL, p_jR = fp[jL], fp[jR]

        # beam length along each side (shortened by junction radii)
        L_left = np.linalg.norm(p_jL - p_iL)
        L_right = np.linalg.norm(p_jR - p_iR)
        L_beam = max(L_left, L_right)
        n_seg = max(1, int(round(L_beam / thickness)))

        left = [iL]
        right = [iR]
        for s in range(1, n_seg):
            t = s / n_seg
            li = len(fp); ri = li + 1
            fp.append(p_iL + t * (p_jL - p_iL))
            fp.append(p_iR + t * (p_jR - p_iR))
            left.append(li)
            right.append(ri)
        left.append(jL)
        right.append(jR)

        # edges for n_seg quad cells
        # Start/end perp chords are already added at multi-beam junctions
        # (Phase 1).  For dead-end caps (1 beam), add them here.
        i_multi = len(adj[i]) > 1
        j_multi = len(adj[j]) > 1
        for s in range(n_seg):
            fe.append((left[s],  left[s+1],   'long'))
            fe.append((right[s], right[s+1],  'long'))
            if s > 0 or not i_multi:          # skip start chord if junction added it
                fe.append((left[s],  right[s],    'perp'))
            fe.append((left[s],  right[s+1],  'diag'))
            fe.append((right[s], left[s+1],   'diag'))
        if not j_multi:                       # skip end chord if junction added it
            fe.append((left[n_seg], right[n_seg], 'perp'))

    fine_pos = np.array(fp)
    return fine_pos, fe, coarse_to_fine


def refine_truss(pos, edges, thickness=0.3, radius=None, min_angle_deg=45.0):
    """
    Refine a coarse graph into a fine lattice truss with undistorted beams.

    Each coarse edge becomes a **regular** beam: parallel sides, perpendicular
    cross-sections, subdivided evenly.  Beams are shortened by the junction
    radius at each end so they fit against the junction circle.

    At each coarse vertex with 2+ beams:

      1. The smallest angle between adjacent beams is found and used to
         compute the circle radius: r = (w/2) / sin(θ/2).
      2. A **center node** is placed at the coarse vertex position.
      3. Each beam's end cross-section is perpendicular to the beam axis,
         at distance r from the center.  The two corners are at ±w/2.
      4. For sectors with angle **< 180°** (inner corners), the adjacent
         beam corners are **snapped/merged** to the bisector point on the
         circle.  This distorts only the tip triangle, not the beam.
      5. For sectors with angle **> 180°** (outer corners), the corners
         are **not merged** — they stay at their regular perpendicular
         positions.
      6. **Radial edges** connect every end corner to the center node.

    Interior beam nodes are placed along the beam **axis** at regular
    perpendicular cross-sections — they are NOT interpolated from the
    (possibly snapped) corner positions.  Only the first and last quad
    cells may be slightly irregular; all interior cells are perfect.

    Parameters
    ----------
    pos : (N, 3) coarse node positions (z=0 for 2D)
    edges : list of (i, j) coarse edges
    thickness : target cell size (both segment length and beam width)
    radius : junction circle radius override (default: auto from min_angle_deg)
    min_angle_deg : sharpest expected angle between beams (default 45°);

    Returns
    -------
    fine_pos : (M, 3) fine node positions
    fine_edges : list of (i, j, type)  type in {'long','perp','diag'}
    coarse_to_fine : dict {coarse_node -> list of fine node indices at that junction}
    """
    pos = np.asarray(pos, dtype=np.float64)
    edges = list(edges)
    n_coarse = pos.shape[0]
    w = thickness
    min_angle_rad = np.radians(min_angle_deg)

    # --- adjacency: node -> [(edge_idx, outward_unit_dir), ...] ---
    adj = [[] for _ in range(n_coarse)]
    for ei, (i, j) in enumerate(edges):
        d = pos[j] - pos[i]
        L = np.linalg.norm(d)
        if L < 1e-12:
            continue
        u = d / L
        adj[i].append((ei, u))
        adj[j].append((ei, -u))

    fe = []
    jmap = {}
    coarse_to_fine = {}
    fp = []
    node_radius = {}  # coarse node -> junction radius

    # --- Phase 1: junction nodes + fan edges ---
    for v in range(n_coarse):
        inc = adj[v]
        if not inc:
            continue

        if len(inc) == 1:
            # flat cap — two nodes at beam width apart, no fan
            ei, u = inc[0]
            n = _perp2d(u)
            li = len(fp)
            ri = li + 1
            fp.append(pos[v] + (w * 0.5) * n)
            fp.append(pos[v] - (w * 0.5) * n)
            jmap[(v, ei, 'L')] = li
            jmap[(v, ei, 'R')] = ri
            coarse_to_fine[v] = [li, ri]
            node_radius[v] = 0.0
            continue

        # 2+ edges: sort CCW by angle
        inc.sort(key=lambda x: np.arctan2(x[1][1], x[1][0]))
        k = len(inc)

        # 1) compute per-node radius from minimum angle
        if radius is not None:
            r_node = radius
        else:
            min_theta = float('inf')
            for m in range(k):
                u_m = inc[m][1]
                u_n = inc[(m + 1) % k][1]
                dot = np.clip(np.dot(u_m, u_n), -1, 1)
                cross = u_m[0] * u_n[1] - u_m[1] * u_n[0]
                theta = np.arccos(dot)
                if cross < 0:
                    theta = 2 * np.pi - theta
                min_theta = min(min_theta, theta)
            min_theta = max(min_theta, min_angle_rad)  # floor only, don't inflate wider angles
            r_node = (w * 0.5) / np.sin(min_theta * 0.5)

        node_radius[v] = r_node

        # 2) center node
        c_idx = len(fp)
        fp.append(pos[v].copy())

        # 3) compute sector angles and bisector points
        sector_angles = []
        for m in range(k):
            u_m = inc[m][1]
            u_n = inc[(m + 1) % k][1]
            dot = np.clip(np.dot(u_m, u_n), -1, 1)
            cross = u_m[0] * u_n[1] - u_m[1] * u_n[0]
            theta = np.arccos(dot)
            if cross < 0:
                theta = 2 * np.pi - theta
            sector_angles.append(theta)

        # bisector points for sectors < 180° (shared corners)
        bisector_indices = [None] * k
        for m in range(k):
            if sector_angles[m] < np.pi - 1e-10:
                u_m = inc[m][1]
                u_n = inc[(m + 1) % k][1]
                bis = u_m + u_n
                bn = np.linalg.norm(bis)
                cross = u_m[0] * u_n[1] - u_m[1] * u_n[0]
                if bn < 1e-8:
                    bis = _perp2d(u_m)
                else:
                    bis = bis / bn
                    if cross < 0:
                        bis = -bis
                idx = len(fp)
                fp.append(pos[v] + r_node * bis)
                bisector_indices[m] = idx

        # 4) assign corner nodes for each beam
        #    left corner (+n side): bisector of sector (m, m+1) if < 180°
        #    right corner (-n side): bisector of sector (m-1, m) if < 180°
        radial_added = set()
        all_fine = [c_idx]
        for m in range(k):
            ei_m = inc[m][0]
            u_m = inc[m][1]
            n_m = _perp2d(u_m)
            axis_pt = pos[v] + r_node * u_m

            # left corner: bisector of this sector (m, m+1)
            if bisector_indices[m] is not None:
                left_idx = bisector_indices[m]
            else:
                left_idx = len(fp)
                fp.append(axis_pt + (w * 0.5) * n_m)
            jmap[(v, ei_m, 'L')] = left_idx

            # right corner: bisector of prev sector (m-1, m)
            prev = (m - 1) % k
            if bisector_indices[prev] is not None:
                right_idx = bisector_indices[prev]
            else:
                right_idx = len(fp)
                fp.append(axis_pt - (w * 0.5) * n_m)
            jmap[(v, ei_m, 'R')] = right_idx

            # chord (beam cross-section at junction)
            fe.append((left_idx, right_idx, 'perp'))

            # 5) radial edges to center (deduplicated)
            for corner_idx in (left_idx, right_idx):
                rkey = (c_idx, corner_idx) if c_idx < corner_idx else (corner_idx, c_idx)
                if rkey not in radial_added:
                    radial_added.add(rkey)
                    fe.append((c_idx, corner_idx, 'perp'))

            all_fine.extend([left_idx, right_idx])

        coarse_to_fine[v] = sorted(set(all_fine))

    # --- Phase 2: beam interior nodes + edges ---
    # Interior nodes are placed along the beam AXIS at regular perpendicular
    # cross-sections.  They are NOT interpolated from corner positions.
    # Only the first and last cells may be irregular (snapped corners).
    for ei, (i, j) in enumerate(edges):
        d = pos[j] - pos[i]
        L = np.linalg.norm(d)
        if L < 1e-12:
            continue
        u = d / L          # unit direction along beam
        n = _perp2d(u)     # perpendicular

        R_i = node_radius.get(i, 0.0)
        R_j = node_radius.get(j, 0.0)

        # axis endpoints (beam shortened by junction radii)
        axis_start = pos[i] + R_i * u
        axis_end   = pos[j] - R_j * u
        L_beam = np.linalg.norm(axis_end - axis_start)
        n_seg = max(1, int(round(L_beam / thickness)))

        # tip corner indices (from Phase 1, may be snapped to bisectors)
        iL = jmap[(i, ei, 'L')]
        iR = jmap[(i, ei, 'R')]
        jL = jmap[(j, ei, 'R')]
        jR = jmap[(j, ei, 'L')]

        left = [iL]
        right = [iR]
        for s in range(1, n_seg):
            t = s / n_seg
            axis_pt = axis_start + t * (axis_end - axis_start)
            li = len(fp); ri = li + 1
            fp.append(axis_pt + (w * 0.5) * n)   # regular perpendicular
            fp.append(axis_pt - (w * 0.5) * n)
            left.append(li)
            right.append(ri)
        left.append(jL)
        right.append(jR)

        # quad cell edges (skip start/end perp chords for multi-beam junctions)
        i_multi = len(adj[i]) > 1
        j_multi = len(adj[j]) > 1
        for s in range(n_seg):
            fe.append((left[s],  left[s+1],   'long'))
            fe.append((right[s], right[s+1],  'long'))
            if s > 0 or not i_multi:
                fe.append((left[s],  right[s],    'perp'))
            fe.append((left[s],  right[s+1],  'diag'))
            fe.append((right[s], left[s+1],   'diag'))
        if not j_multi:
            fe.append((left[n_seg], right[n_seg], 'perp'))

    fine_pos = np.array(fp)
    return fine_pos, fe, coarse_to_fine


def assemble_stiffness_typed(pos, edges, k_long=1.0, k_perp=1.0, k_diag=1.0, dim=3):
    """
    Assemble dense stiffness matrix with per-type spring constants.

    edges: list of (i, j, type)  type in {'long','perp','diag'}.
    Untyped edges fall back to k_long.
    """
    n_nodes = pos.shape[0]
    ndof = dim * n_nodes
    K = np.zeros((ndof, ndof), dtype=np.float64)
    kmap = {'long': k_long, 'perp': k_perp, 'diag': k_diag}

    for e in edges:
        i, j = e[0], e[1]
        et = e[2] if len(e) > 2 else 'long'
        k = kmap.get(et, k_long)
        d = pos[j] - pos[i]
        L = np.linalg.norm(d)
        if L <= 1e-12:
            continue
        u = d / L
        kf = k / (L * L)
        ia, ja = i * dim, j * dim
        outer = kf * np.outer(u, u)
        K[ia:ia+dim, ia:ia+dim] += outer
        K[ja:ja+dim, ja:ja+dim] += outer
        K[ia:ia+dim, ja:ja+dim] -= outer
        K[ja:ja+dim, ia:ia+dim] -= outer

    return K


def build_refined_truss(pos, edges, thickness=0.3, radius=None, min_angle_deg=45.0,
                        k_long=1.0, k_perp=1.0, k_diag=1.0,
                        mass_value=1.0, fixed_coarse_nodes=None, dim=3):
    """
    Convenience wrapper: refine coarse mesh, assemble typed stiffness + mass,
    apply Dirichlet BCs on coarse nodes.

    fixed_coarse_nodes: list of coarse node indices whose fine nodes are fixed.
    """
    fine_pos, fine_edges, c2f = refine_truss(pos, edges, thickness, radius, min_angle_deg)
    K = assemble_stiffness_typed(fine_pos, fine_edges, k_long, k_perp, k_diag, dim)
    M = mass_matrix(np.full(fine_pos.shape[0], mass_value), dim=dim)

    fixed = []
    if fixed_coarse_nodes:
        for v in fixed_coarse_nodes:
            fixed.extend(c2f.get(v, []))

    if not fixed:
        mask = np.ones(K.shape[0], dtype=bool)
        return fine_pos, fine_edges, K, M, mask

    K_red, M_red, mask = apply_dirichlet(K, M, fixed, dim=dim)
    return fine_pos, fine_edges, K_red, M_red, mask


# ---------------------------------------------------------------------------
# ASCII art -> coarse graph
# ---------------------------------------------------------------------------

_ASCII_EDGE_DIRS = {
    '-': [((0, -1), (0, +1))],
    '_': [((0, -1), (0, +1))],
    '|': [((-1, 0), (+1, 0))],
    '/': [((+1, -1), (-1, +1))],
    '\\': [((-1, -1), (+1, +1))],
    'x': [((+1, -1), (-1, +1)), ((-1, -1), (+1, +1))],
    '*': [((+1, -1), (-1, +1)), ((-1, -1), (+1, +1))],
}

_ASCII_NODE_CHARS = {'.', '+'}
_ASCII_BAR_CHARS = set(_ASCII_EDGE_DIRS.keys())
_ASCII_PASSABLE = _ASCII_BAR_CHARS | _ASCII_NODE_CHARS | {' '}


def _trace_endpoint(grid, r, c, dr, dc, all_nodes):
    """
    Trace from bar at (r,c) in direction (dr,dc) to find the endpoint node.

    - If we hit a joint ('.' or '+') or a known auto-node, return it.
    - If we hit another bar, jump to that bar's endpoint in our direction
      (trace-through), and continue.
    - If we go out of bounds, return the out-of-bounds position as an
      implicit node.
    - If we hit a non-passable char, return None.
    """
    r += dr
    c += dc
    nrows = len(grid)
    while True:
        if not (0 <= r < nrows and 0 <= c < len(grid[r])):
            return (r, c)
        ch = grid[r][c]
        if ch in _ASCII_NODE_CHARS:
            return (r, c)
        if (r, c) in all_nodes:
            return (r, c)
        if ch in _ASCII_BAR_CHARS:
            best_d = None
            best_dot = -999
            for d1, d2 in _ASCII_EDGE_DIRS[ch]:
                for d in (d1, d2):
                    dot = d[0] * dr + d[1] * dc
                    if dot > best_dot:
                        best_dot = dot
                        best_d = d
            r += best_d[0]
            c += best_d[1]
            continue
        if ch not in _ASCII_PASSABLE:
            return None
        r += dr
        c += dc


def _find_convergence_nodes(grid):
    """
    Detect pairs of bars that converge (would intersect) and compute
    intersection points.  Returns set of (r, c) grid positions.
    """
    nrows = len(grid)
    nodes = set()
    bars = []
    for r in range(nrows):
        for c in range(len(grid[r])):
            ch = grid[r][c]
            if ch in _ASCII_BAR_CHARS:
                for d1, d2 in _ASCII_EDGE_DIRS[ch]:
                    for d in (d1, d2):
                        bars.append((r, c, d))

    for i, (r1, c1, d1) in enumerate(bars):
        for r2, c2, d2 in bars[i + 1:]:
            if (r1, c1) == (r2, c2):
                continue
            if d1[0] * d2[0] + d1[1] * d2[1] >= 0:
                continue
            det = d1[0] * (-d2[1]) - d1[1] * (-d2[0])
            if abs(det) < 1e-12:
                continue
            t = ((r2 - r1) * (-d2[1]) - (c2 - c1) * (-d2[0])) / det
            s = (d1[0] * (c2 - c1) - d1[1] * (r2 - r1)) / det
            if t < 0.5 or s < 0.5:
                continue
            ir = r1 + t * d1[0]
            ic = c1 + t * d1[1]
            ir_r, ic_r = round(ir), round(ic)
            if abs(ir - ir_r) <= 0.5 and abs(ic - ic_r) <= 0.5:
                dist = abs(r1 - r2) + abs(c1 - c2)
                if dist <= 4:
                    nodes.add((ir_r, ic_r))
    return nodes


def parse_ascii_truss(ascii_art, scale=1.0):
    """
    Parse ASCII art into coarse node positions and edges.

    Uses a **trace-through** model: each bar character traces along its
    direction until it finds a joint (``.`` or ``+``).  If it hits another
    bar along the way, it passes through to that bar's far endpoint.

    ============  ============================================
    Character     Meaning
    ------------  --------------------------------------------
    ``.`` or ``+``  explicit joint / node
    ``|``         vertical bar (traces up & down)
    ``-`` ``_``   horizontal bar (traces left & right)
    ``/``         rising diagonal (traces down-left & up-right)
    ``\\``         falling diagonal (traces up-left & down-right)
    ``x``         two crossing diagonals (no central joint)
    ``*``         two crossing diagonals WITH central joint
    ============  ============================================

    **Convergence auto-connection**: when two bars point toward each
    other (e.g. ``/\\``), a node is auto-created at their intersection.
    Parallel bars (``| |``) do NOT connect.

    This handles both compact forms (``/_\\``) and explicit forms
    (``.-.\\n| |\\n.-.``).

    Parameters
    ----------
    ascii_art : str  (multi-line ASCII art)
    scale : float    spatial units per character cell

    Returns
    -------
    pos : (N, 3) coarse node positions (z=0)
    edges : list of (i, j) coarse edges
    """
    lines = ascii_art.strip('\n').split('\n')
    if not lines:
        return np.zeros((0, 3)), []

    max_len = max(len(line) for line in lines)
    grid = [line.ljust(max_len) for line in lines]
    nrows = len(grid)

    node_map = {}
    node_list = []

    def get_node(r, c):
        key = (r, c)
        if key not in node_map:
            node_map[key] = len(node_list)
            node_list.append(key)
        return node_map[key]

    # 1) collect explicit nodes
    for r in range(nrows):
        for c in range(len(grid[r])):
            if grid[r][c] in _ASCII_NODE_CHARS:
                get_node(r, c)

    # 2) handle '*' : create central node
    for r in range(nrows):
        for c in range(len(grid[r])):
            if grid[r][c] == '*':
                get_node(r, c)

    # 3) convergence nodes
    conv_nodes = _find_convergence_nodes(grid)
    all_nodes = set(node_map.keys()) | conv_nodes
    for pt in conv_nodes:
        get_node(*pt)

    # 4) trace endpoints for each bar and create edges
    edge_set = set()
    edges = []
    for r in range(nrows):
        for c in range(len(grid[r])):
            ch = grid[r][c]
            if ch not in _ASCII_EDGE_DIRS:
                continue
            for d1, d2 in _ASCII_EDGE_DIRS[ch]:
                a = _trace_endpoint(grid, r, c, d1[0], d1[1], all_nodes)
                b = _trace_endpoint(grid, r, c, d2[0], d2[1], all_nodes)
                if a is not None and b is not None:
                    ia, ib = get_node(*a), get_node(*b)
                    all_nodes.add(a)
                    all_nodes.add(b)
                    if ia != ib:
                        ekey = (min(ia, ib), max(ia, ib))
                        if ekey not in edge_set:
                            edge_set.add(ekey)
                            edges.append((ia, ib))

    pos = np.zeros((len(node_list), 3))
    for idx, (r, c) in enumerate(node_list):
        pos[idx] = [c * scale, -r * scale, 0.0]

    return pos, edges


# ---------------------------------------------------------------------------
# Distortion functions
# ---------------------------------------------------------------------------

def distort_noise(pos, amplitude=0.05, rng=None):
    """
    High-frequency noise: independent random displacement per node.

    Parameters
    ----------
    pos : (N, D) node positions
    amplitude : float  standard deviation of Gaussian displacement
    rng : np.random.Generator or None
    """
    if rng is None:
        rng = np.random.default_rng()
    return pos + rng.normal(0.0, amplitude, size=pos.shape)


def distort_affine(pos, scale=(1.0, 1.0), shear=0.0, rotation=0.0, translation=(0.0, 0.0)):
    """
    Linear (affine) distortion: anisotropic scaling + shear + rotation.

    All are global, low-frequency (wavelength = domain size).

    Parameters
    ----------
    pos : (N, D) node positions (uses x, y columns)
    scale : (sx, sy)  anisotropic scaling factors
    shear : float  shear coefficient (x' = x + shear*y)
    rotation : float  rotation angle in radians
    translation : (tx, ty)  rigid shift
    """
    pos = np.asarray(pos, dtype=np.float64).copy()
    x, y = pos[:, 0], pos[:, 1]
    # shear
    x = x + shear * y
    # scale
    x = x * scale[0]
    y = y * scale[1]
    # rotation
    cr, sr = np.cos(rotation), np.sin(rotation)
    xr = x * cr - y * sr
    yr = x * sr + y * cr
    pos[:, 0] = xr + translation[0]
    pos[:, 1] = yr + translation[1]
    return pos


def distort_bend(pos, axis='x', curvature=0.02, direction='y'):
    """
    Quadratic bending: deflect nodes along a parabola.

    The deflection grows quadratically along ``axis`` and is applied
    in ``direction``.  This is a single low-frequency mode.

    Parameters
    ----------
    pos : (N, D) node positions
    axis : 'x' or 'y'  the axis along which bending varies
    curvature : float  quadratic coefficient (deflection = c * t^2 * L)
    direction : 'x' or 'y'  the direction of deflection
    """
    pos = np.asarray(pos, dtype=np.float64).copy()
    if axis == 'x':
        t = pos[:, 0]
    else:
        t = pos[:, 1]
    t0 = t.min()
    L = t.max() - t0
    if L < 1e-12:
        return pos
    tn = (t - t0) / L  # normalized [0, 1]
    defl = curvature * L * tn * tn  # quadratic
    if direction == 'x':
        pos[:, 0] += defl
    else:
        pos[:, 1] += defl
    return pos


def distort_planewaves(pos, amplitude=0.1, n_waves=5, wavelength=None, rng=None):
    """
    Low-frequency omnidirectional distortion via random plane waves.

    Sum of ``n_waves`` terms:  a_i * cos(k_i . r + phi_i)

    Each wave has a random direction (isotropic), random phase, and
    wavelength drawn from a low-frequency range.  This produces smooth,
    omnidirectional distortion without preferred direction.

    Parameters
    ----------
    pos : (N, D) node positions
    amplitude : float  per-wave amplitude (total RMS ~ amplitude * sqrt(n/2))
    n_waves : int  number of plane wave terms
    wavelength : float or None  if None, uses 2x the domain diameter
    rng : np.random.Generator or None
    """
    pos = np.asarray(pos, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng()
    center = pos[:, :2].mean(axis=0)
    r = pos[:, :2] - center
    if wavelength is None:
        diameter = np.max(np.linalg.norm(pos[:, :2] - center, axis=1)) * 2
        wavelength = max(diameter * 2.0, 1.0)
    k_mag = 2 * np.pi / wavelength
    disp = np.zeros_like(pos[:, :2])
    for _ in range(n_waves):
        theta = rng.uniform(0, 2 * np.pi)
        k = k_mag * np.array([np.cos(theta), np.sin(theta)])
        phi = rng.uniform(0, 2 * np.pi)
        phase = r @ k + phi
        disp += amplitude * np.cos(phase)[:, None] * np.array([np.cos(theta), np.sin(theta)])[None, :]
    result = pos.copy()
    result[:, :2] += disp
    return result


def distort(pos, mode='noise', **kwargs):
    """
    Apply distortion to node positions.

    mode : 'noise', 'affine', 'bend', 'planewaves', or 'combined'
    **kwargs : passed to the corresponding distort_* function

    For 'combined', pass a list of (mode, kwargs) dicts via ``steps=``.
    """
    if mode == 'combined':
        for step in kwargs.get('steps', []):
            pos = distort(pos, **step)
        return pos
    fn = {
        'noise': distort_noise,
        'affine': distort_affine,
        'bend': distort_bend,
        'planewaves': distort_planewaves,
    }[mode]
    return fn(pos, **kwargs)


class Truss:
    """
    High-level interface for building, refining, and solving truss systems.

    Workflow::

        t = Truss(coarse_pos, coarse_edges, thickness=0.4)
        t.refine()                         # generate fine lattice mesh
        t.assemble(k_long=10, k_diag=5)    # build K, M, apply BCs
        t.plot()                           # visualize

    Parameters
    ----------
    coarse_pos : (N, 3) array of coarse node positions (z=0 for 2D)
    coarse_edges : list of (i, j) coarse edges
    thickness : target cell size (beam width and segment length)
    radius : junction circle radius override (default: auto per-node)
    min_angle_deg : floor for angle-based radius computation (default 45°)
    """
    def __init__(self, coarse_pos, coarse_edges, thickness=0.3, radius=None, min_angle_deg=45.0):
        self.coarse_pos = np.asarray(coarse_pos, dtype=np.float64)
        self.coarse_edges = list(coarse_edges)
        self.thickness = thickness
        self.radius = radius
        self.min_angle_deg = min_angle_deg

        # populated by refine()
        self.fine_pos = None
        self.fine_edges = None
        self.coarse_to_fine = None

        # populated by assemble()
        self.K = None
        self.M = None
        self.mask = None
        self.dim = 3

    @classmethod
    def from_ascii(cls, ascii_art, scale=1.0, **kwargs):
        """
        Create a Truss from ASCII art.

        See :func:`parse_ascii_truss` for the character set.
        """
        pos, edges = parse_ascii_truss(ascii_art, scale)
        return cls(pos, edges, **kwargs)

    def refine(self, method='regular'):
        """
        Refine the coarse graph into a fine lattice truss.

        method : 'regular' (undistorted beams, snapped inner tips) or
                 'fan' (bisector corners, may distort beams)
        """
        if method == 'fan':
            fn = refine_truss_fan
        else:
            fn = refine_truss
        self.fine_pos, self.fine_edges, self.coarse_to_fine = fn(
            self.coarse_pos, self.coarse_edges,
            thickness=self.thickness,
            radius=self.radius,
            min_angle_deg=self.min_angle_deg,
        )
        return self

    def distort(self, mode='noise', target='fine', **kwargs):
        """
        Apply distortion to node positions.

        mode : 'noise', 'affine', 'bend', 'planewaves', or 'combined'
        target : 'fine' (distort refined mesh) or 'coarse' (distort before refine)
        **kwargs : passed to the distort_* function

        For 'combined', pass ``steps=[{'mode': ..., **kw}, ...]``.
        """
        if target == 'fine':
            if self.fine_pos is None:
                raise RuntimeError("Call refine() before distort(target='fine')")
            self.fine_pos = distort(self.fine_pos, mode=mode, **kwargs)
        else:
            self.coarse_pos = distort(self.coarse_pos, mode=mode, **kwargs)
        return self

    def assemble(self, k_long=1.0, k_perp=1.0, k_diag=1.0,
                 mass_value=1.0, fixed_coarse_nodes=None, dim=3):
        """
        Assemble stiffness and mass matrices from the refined mesh.

        Must be called after :meth:`refine`.

        fixed_coarse_nodes : list of coarse node indices to pin (Dirichlet BC).
        """
        if self.fine_pos is None:
            raise RuntimeError("Call refine() before assemble()")
        self.dim = dim
        self.K = assemble_stiffness_typed(
            self.fine_pos, self.fine_edges, k_long, k_perp, k_diag, dim)
        self.M = mass_matrix(np.full(self.fine_pos.shape[0], mass_value), dim=dim)

        fixed = []
        if fixed_coarse_nodes:
            for v in fixed_coarse_nodes:
                fixed.extend(self.coarse_to_fine.get(v, []))

        if not fixed:
            self.mask = np.ones(self.K.shape[0], dtype=bool)
        else:
            self.K, self.M, self.mask = apply_dirichlet(self.K, self.M, fixed, dim=dim)
        return self

    def plot(self, ax=None, **kwargs):
        """Visualize the refined truss (requires matplotlib)."""
        if self.fine_pos is None:
            raise RuntimeError("Call refine() before plot()")
        return plot_truss(self.fine_pos, self.fine_edges, ax=ax, **kwargs)

    @property
    def n_coarse_nodes(self):
        return len(self.coarse_pos)

    @property
    def n_fine_nodes(self):
        return len(self.fine_pos) if self.fine_pos is not None else 0

    @property
    def n_fine_edges(self):
        return len(self.fine_edges) if self.fine_edges is not None else 0

    def __repr__(self):
        s = f"Truss({self.n_coarse_nodes} coarse nodes, {len(self.coarse_edges)} coarse edges)"
        if self.fine_pos is not None:
            s += f" -> {self.n_fine_nodes} fine nodes, {self.n_fine_edges} fine edges"
        return s


def plot_truss(pos, edges, ax=None, color_by_type=True, node_size=3, linewidth=0.6):
    """
    Quick 2D visualisation of a truss (requires matplotlib).

    Edge colours: blue=longitudinal, green=perpendicular, red=diagonal.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    cmap = {'long': '#2196F3', 'perp': '#4CAF50', 'diag': '#F44336'}
    for e in edges:
        i, j = e[0], e[1]
        et = e[2] if len(e) > 2 else None
        c = cmap.get(et, '#999')
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                color=c, linewidth=linewidth)

    ax.scatter(pos[:, 0], pos[:, 1], s=node_size, c='black', zorder=5)
    ax.set_aspect('equal')
    return ax


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    examples = [
        ("triangle (compact)", "/_\\"),
        ("triangle (explicit)", "  .\n / \\\n.---."),
        ("right triangle", ".\n|\\\n| \\\n.--."),
        ("square", ".-.\n| |\n.-."),
        ("square 1-diag", ".-.\n|/|\n.-."),
        ("square 2-diag", ".-.\n|x|\n.-."),
        ("girder 3-tri", "  .---.\n / \\ / \\\n.---.---."),
        ("girder compact", "   _\n/_\\/_\\"),
        ("hex fan", "  .---.\n / \\ / \\\n.---.---.\n \\ / \\ /\n  .---."),
    ]

    n = len(examples)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten()

    for ax, (title, art) in zip(axes, examples):
        pos, edges = parse_ascii_truss(art, scale=1.0)
        # plot coarse graph
        for i, j in edges:
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 'b-', linewidth=1.5)
        ax.scatter(pos[:, 0], pos[:, 1], s=60, c='red', zorder=5)
        for idx in range(len(pos)):
            ax.annotate(str(idx), pos[idx, :2], textcoords="offset points",
                        xytext=(5, 5), fontsize=8)
        ax.set_title(f"{title}\n{len(pos)} nodes, {len(edges)} edges", fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlim(pos[:, 0].min() - 0.5, pos[:, 0].max() + 0.5)
        ax.set_ylim(pos[:, 1].min() - 0.5, pos[:, 1].max() + 0.5)
        # ASCII art inset
        art_display = art.replace('\\', '\\\\')
        ax.text(0.02, 0.98, art_display, transform=ax.transAxes,
                fontsize=9, family='DejaVu Sans Mono', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    for ax in axes[n:]:
        ax.set_visible(False)

    print("ASCII art examples (coarse graphs):")
    for title, art in examples:
        pos, edges = parse_ascii_truss(art, scale=1.0)
        print(f"  {title}: {len(pos)} nodes, {len(edges)} edges")
        print(f"    pos = {pos.tolist()}")
        print(f"    edges = {edges}")

    plt.tight_layout()
    plt.savefig("ascii_truss_demo.png", dpi=150)

    # --- Distortion demo ---
    rng = np.random.default_rng(42)
    art = ".-.\n| |\n.-."
    t = Truss.from_ascii(art, scale=1.0, thickness=0.3)
    t.refine()

    distort_modes = [
        ("original", {}),
        ("noise", {'mode': 'noise', 'amplitude': 0.05}),
        ("affine scale", {'mode': 'affine', 'scale': (1.3, 0.8)}),
        ("affine shear", {'mode': 'affine', 'shear': 0.2}),
        ("bend x->y", {'mode': 'bend', 'axis': 'x', 'curvature': 0.1, 'direction': 'y'}),
        ("planewaves", {'mode': 'planewaves', 'amplitude': 0.08, 'n_waves': 5, 'rng': np.random.default_rng(42)}),
        ("combined", {'mode': 'combined', 'steps': [
            {'mode': 'affine', 'shear': 0.1},
            {'mode': 'bend', 'axis': 'x', 'curvature': 0.05, 'direction': 'y'},
            {'mode': 'noise', 'amplitude': 0.02, 'rng': np.random.default_rng(7)},
        ]}),
    ]

    n_d = len(distort_modes)
    ncols_d = 4
    nrows_d = (n_d + ncols_d - 1) // ncols_d
    fig2, axes2 = plt.subplots(nrows_d, ncols_d, figsize=(5 * ncols_d, 5 * nrows_d))
    axes2 = axes2.flatten()

    for ax, (title, kw) in zip(axes2, distort_modes):
        t2 = Truss.from_ascii(art, scale=1.0, thickness=0.3)
        t2.refine()
        if kw:
            t2.distort(**kw)
        t2.plot(ax=ax)
        ax.set_title(title, fontsize=11)

    for ax in axes2[n_d:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig("distortion_demo.png", dpi=150)
    plt.show()
