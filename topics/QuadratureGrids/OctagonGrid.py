"""
OctagonGrid.py — Square-to-radial grid using octagon as intermediate.

The octagon provides a much smoother transition from square to circle
than the previous wedge approach:
  - Square:  radial variation 41% (h to h*sqrt(2))
  - Octagon: radial variation  8% (h to h/cos(pi/8))

Construction:
1. Square [-h, h]^2 with grid step d
2. Cut corners to form regular octagon (apothem = h)
3. Octagon boundary: 8 edges (4 on square edges + 4 diagonal cuts)
4. Inside octagon: blend from circle (center) to octagon (boundary)
5. Outside octagon: Cartesian grid (corner triangles + outer region)

The 4 axis-aligned octagon edges lie ON the square edges, so they
connect seamlessly to the Cartesian grid.  The 4 diagonal edges cut
off the corners; the corner triangles are filled by Cartesian points.
"""
import numpy as np


# ── TA-M4 radial mapping (Treutler-Ahlrichs) ──────────────────────────────────

def treutler_ahlrichs_radial(N, xi=1.0):
    """Treutler-Ahlrichs M4 radial grid with Chebyshev 2nd kind quadrature.

    r(x) = -(ξ/ln2) * (1+x)^0.6 * ln((1-x)/2)
    x ∈ [-1,1] at Gauss-Chebyshev 2nd kind nodes.

    Returns
    -------
    r   : (N,) — radial points (sorted ascending)
    w_r : (N,) — radial weights including Jacobian r*dr
    """
    step = np.pi / (N + 1)
    k = np.arange(1, N + 1)
    x = np.cos(k * step)
    ln2 = xi / np.log(2)
    r = -ln2 * (1 + x)**0.6 * np.log((1 - x) / 2)
    rprime = ln2 * (1 + x)**0.6 * (
        -0.6 / (1 + x) * np.log((1 - x) / 2) + 1.0 / (1 - x)
    )
    dr = step * np.sin(k * step) * rprime
    w_r = r * dr
    idx = np.argsort(r)
    return r[idx], w_r[idx]


# ── Helpers ───────────────────────────────────────────────────────────────────

def octagon_vertices(h):
    """8 vertices of regular octagon with apothem h, inscribed in [-h,h]^2.

    Returns (verts, s) where s is the corner cut size.
    """
    s = h * (2 - np.sqrt(2))     # cut size
    hs = h - s                   # h*(sqrt(2)-1), half-length of axis-aligned edges
    verts = np.array([
        [ h,  hs], [ hs,  h], [-hs,  h], [-h,  hs],
        [-h, -hs], [-hs, -h], [ hs, -h], [ h, -hs],
    ])
    return verts, s


def octagon_radius(phi, h):
    """Distance from origin to regular octagon boundary (apothem h) in direction phi."""
    delta = phi - np.round(phi / (np.pi / 4)) * (np.pi / 4)
    return h / np.cos(delta)


def octagon_boundary(n_axis, n_diag, h, hs):
    """Boundary points of grid-aligned octagon.

    Axis-aligned edges have n_axis segments, diagonal edges have n_diag segments.
    All boundary points coincide with Cartesian grid points.

    Returns
    -------
    pts  : (Nu, 2) — boundary points (counterclockwise)
    phis : (Nu,)   — polar angles
    """
    verts = np.array([
        [ h,  hs], [ hs,  h], [-hs,  h], [-h,  hs],
        [-h, -hs], [-hs, -h], [ hs, -h], [ h, -hs],
    ])
    n_per = [n_diag, n_axis, n_diag, n_axis, n_diag, n_axis, n_diag, n_axis]
    Nu = sum(n_per)
    pts = np.zeros((Nu, 2))
    phis = np.zeros(Nu)
    idx = 0
    for k in range(8):
        v0 = verts[k]
        v1 = verts[(k + 1) % 8]
        npk = n_per[k]
        for j in range(npk):
            t = j / npk
            pts[idx] = v0 + t * (v1 - v0)
            phis[idx] = np.arctan2(pts[idx, 1], pts[idx, 0])
            idx += 1
    return pts, phis


def radial_map(t, alpha):
    """t in [0,1] -> r in [0,h], dense near center for alpha>1."""
    return t ** alpha


def quad_area(p0, p1, p2, p3):
    """Area of quadrilateral given 4 vertex arrays (each (...,2))."""
    return 0.5 * np.abs(
        p0[..., 0] * (p1[..., 1] - p3[..., 1]) +
        p1[..., 0] * (p2[..., 1] - p0[..., 1]) +
        p2[..., 0] * (p3[..., 1] - p1[..., 1]) +
        p3[..., 0] * (p0[..., 1] - p2[..., 1])
    )


# ── Grid construction ─────────────────────────────────────────────────────────

def build_octagon_grid(d=0.5, n=10, alpha=1.8, n_blend=4, blend_mode='immediate'):
    """
    Build octagon-intermediate grid.

    Parameters
    ----------
    d : float       — Cartesian grid step
    n : int         — cutout size in voxels per side (h = n*d/2)
    alpha : float   — radial mapping exponent (>1 = denser near center)
    n_blend : int   — blend exponent (w=1 -> circle, w=0 -> octagon)
    blend_mode : str — 'smooth' for gradual blend, 'immediate' for octagon-only
                       at boundary layer (j=Nv-1), circle everywhere else

    Returns
    -------
    dict with:
        points         : (Nu, Nv, 2) — blended grid points
        points_octagon : (Nu, Nv, 2) — pure scaled octagon
        points_circle  : (Nu, Nv, 2) — pure circle
        boundary_pts   : (Nu, 2)     — octagon boundary at full scale
        phis           : (Nu,)       — polar angles of boundary points
        corners        : (4, 2)      — square corners
        oct_verts      : (8, 2)      — octagon vertices
        t_vals, r_vals, w_blend      — radial parameter arrays
        cell_areas     : (Nu, Nv-1)  — cell areas
        cell_aspect    : (Nu, Nv-1)  — cell aspect ratios
        params         : dict        — all parameters
    """
    h = n * d / 2

    # Grid-aligned octagon: snap hs to nearest multiple of d
    # so axis-aligned edge points coincide with Cartesian grid points
    hs_ideal = h * (np.sqrt(2) - 1)
    hs = round(hs_ideal / d) * d
    s = h - hs                     # corner cut size

    oct_verts = np.array([
        [ h,  hs], [ hs,  h], [-hs,  h], [-h,  hs],
        [-h, -hs], [-hs, -h], [ hs, -h], [ h, -hs],
    ])

    # subdivisions per edge type — all boundary points are Cartesian grid points
    n_axis = max(1, int(round(2 * hs / d)))   # axis-aligned edge segments
    n_diag = max(1, int(round(s / d)))         # diagonal edge segments
    Nu = 4 * n_axis + 4 * n_diag

    angular_spacing = d  # all boundary segments span d in each coordinate

    # Radial shell count — octagon max radius is h/cos(pi/8) ~ 1.082h
    target_ratio = 1 - angular_spacing / h
    base = target_ratio ** (1.0 / alpha)
    Nv_base = int(np.ceil(1 / (1 - base))) + 1
    Nv = int(np.ceil(Nv_base / np.cos(np.pi / 8)))

    # Boundary
    boundary_pts, phis = octagon_boundary(n_axis, n_diag, h, hs)

    # Radial mapping
    t_vals = np.linspace(0, 1, Nv)
    r_vals = h * radial_map(t_vals, alpha)

    corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])

    # Pure octagon (scaled by r/h)
    points_octagon = np.zeros((Nu, Nv, 2))
    for j in range(Nv):
        scale = r_vals[j] / h
        points_octagon[:, j, 0] = scale * boundary_pts[:, 0]
        points_octagon[:, j, 1] = scale * boundary_pts[:, 1]

    # Pure circle (radius = r_vals, same angles)
    points_circle = np.zeros((Nu, Nv, 2))
    for j in range(Nv):
        points_circle[:, j, 0] = r_vals[j] * np.cos(phis)
        points_circle[:, j, 1] = r_vals[j] * np.sin(phis)

    # Blend: w=1 near center (circle), w=0 near boundary (octagon)
    if blend_mode == 'immediate':
        # Only outermost layer (j=Nv-1) is octagon, all others are circle
        w_blend = np.ones(Nv)
        w_blend[-1] = 0.0
    else:
        w_blend = 1 - t_vals ** n_blend
    points = np.zeros((Nu, Nv, 2))
    for j in range(Nv):
        w = w_blend[j]
        points[:, j, 0] = w * points_circle[:, j, 0] + (1 - w) * points_octagon[:, j, 0]
        points[:, j, 1] = w * points_circle[:, j, 1] + (1 - w) * points_octagon[:, j, 1]

    # Cell areas + aspect ratios
    cell_areas = np.zeros((Nu, Nv - 1))
    cell_aspect = np.zeros((Nu, Nv - 1))
    for i in range(Nu):
        i2 = (i + 1) % Nu
        for j in range(Nv - 1):
            p0 = points[i, j]
            p1 = points[i2, j]
            p2 = points[i2, j + 1]
            p3 = points[i, j + 1]
            cell_areas[i, j] = quad_area(p0, p1, p2, p3)
            e0 = np.linalg.norm(p1 - p0)
            e1 = np.linalg.norm(p2 - p1)
            e2 = np.linalg.norm(p3 - p2)
            e3 = np.linalg.norm(p0 - p3)
            ang_len = 0.5 * (e0 + e2)
            rad_len = 0.5 * (e1 + e3)
            if ang_len > 1e-12 and rad_len > 1e-12:
                cell_aspect[i, j] = max(ang_len, rad_len) / min(ang_len, rad_len)
            else:
                cell_aspect[i, j] = 1.0

    params = dict(d=d, n=n, alpha=alpha, n_blend=n_blend, blend_mode=blend_mode,
                  h=h, s=s, hs=hs,
                  n_axis=n_axis, n_diag=n_diag, Nu=Nu, Nv=Nv,
                  angular_spacing=angular_spacing)

    return dict(
        points=points,
        points_octagon=points_octagon,
        points_circle=points_circle,
        boundary_pts=boundary_pts,
        phis=phis,
        corners=corners,
        oct_verts=oct_verts,
        t_vals=t_vals,
        r_vals=r_vals,
        w_blend=w_blend,
        cell_areas=cell_areas,
        cell_aspect=cell_aspect,
        params=params,
    )


def build_octagon_ta_grid(d=0.1, n=10, n_ta=15, xi=1.0, n_transition=0,
                          alpha_cut=1.0):
    """Build octagon grid with TA-M4 radial insert — snapped to octagon.

    The inner radial shells use perfectly uniform angular distribution
    (phi_i = 2*pi*i/Nu) — isotropic, no preferred directions.

    The outermost radial shell is SNAPPED to the octagon boundary points.
    This absorbs all angular non-uniformity in a single layer (no wasted
    transition layers).

    Construction:
      1. Compute octagon boundary points and their average radius R_oct
      2. Generate TA-M4 radial points, rescale outer radius to R_oct
      3. Inner layers: uniform circle at r_j * (cos(2pi i/Nu), sin(2pi i/Nu))
      4. Outer layer: snapped to octagon boundary points (with optimal shift)

    Parameters
    ----------
    d : float       — Cartesian grid step (Bohr)
    n : int         — cutout size (h = n*d/2)
    n_ta : int      — number of TA-M4 radial shells (including snapped outer)
    xi : float      — TA-M4 scaling parameter
    n_transition : int — unused (kept for API compatibility), always 0
    alpha_cut : float — unused (kept for API compatibility), R_oct used instead

    Returns
    -------
    dict with grid points, mesh data, and diagnostics.
    """
    h = n * d / 2

    hs_ideal = h * (np.sqrt(2) - 1)
    hs = round(hs_ideal / d) * d
    s = h - hs

    oct_verts = np.array([
        [ h,  hs], [ hs,  h], [-hs,  h], [-h,  hs],
        [-h, -hs], [-hs, -h], [ hs, -h], [ h, -hs],
    ])

    n_axis = max(1, int(round(2 * hs / d)))
    n_diag = max(1, int(round(s / d)))
    Nu = 4 * n_axis + 4 * n_diag

    # Octagon boundary points (counterclockwise)
    boundary_pts, phis_oct = octagon_boundary(n_axis, n_diag, h, hs)

    # === Step 1: R_oct = average radius of octagon boundary points ===
    R_oct = np.mean(np.linalg.norm(boundary_pts, axis=1))

    # === Step 2: TA-M4 radial grid, rescaled so outer radius = R_oct ===
    r_ta_raw, w_ta_raw = treutler_ahlrichs_radial(n_ta, xi=xi)
    r_max_raw = r_ta_raw[-1]
    scale_r = R_oct / r_max_raw
    r_ta = r_ta_raw * scale_r
    w_ta = w_ta_raw * scale_r**2

    R_ins = r_ta[-1]  # = R_oct by construction

    # === Step 3: Uniform angular distribution for inner layers ===
    phis_uniform = np.array([2 * np.pi * i / Nu for i in range(Nu)])

    # === Step 4: Find optimal circular shift for snapping ===
    # Try all Nu shifts, pick the one that minimizes total displacement
    # between uniform circle at R_oct and octagon boundary points
    uniform_outer = np.column_stack([R_oct * np.cos(phis_uniform),
                                      R_oct * np.sin(phis_uniform)])
    best_shift = 0
    best_disp = np.inf
    for shift in range(Nu):
        oct_shifted = np.roll(boundary_pts, shift, axis=0)
        disp = np.sum(np.linalg.norm(uniform_outer - oct_shifted, axis=1))
        if disp < best_disp:
            best_disp = disp
            best_shift = shift

    # Snap: reorder octagon boundary points to best match uniform angles
    boundary_snapped = np.roll(boundary_pts, best_shift, axis=0)
    phis_snapped = np.arctan2(boundary_snapped[:, 1], boundary_snapped[:, 0])

    # === Build grid ===
    # j=0: center, j=1..n_ta-1: uniform circle, j=n_ta: snapped to octagon
    Nv = n_ta + 1  # center + n_ta shells

    r_vals = np.zeros(Nv)
    r_vals[0] = 0.0
    r_vals[1:n_ta + 1] = r_ta

    t_vals = np.linspace(0, 1, Nv)

    corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])

    points = np.zeros((Nu, Nv, 2))
    points_octagon = np.zeros((Nu, Nv, 2))
    points_circle = np.zeros((Nu, Nv, 2))

    for j in range(Nv):
        r = r_vals[j]
        # Uniform circle (isotropic)
        points_circle[:, j, 0] = r * np.cos(phis_uniform)
        points_circle[:, j, 1] = r * np.sin(phis_uniform)
        # Octagon (scaled boundary)
        scale = r / h if h > 0 else 0
        points_octagon[:, j, 0] = scale * boundary_snapped[:, 0]
        points_octagon[:, j, 1] = scale * boundary_snapped[:, 1]
        # Actual points:
        #   j < n_ta: uniform circle (perfectly isotropic)
        #   j = n_ta: snapped to octagon boundary
        if j < n_ta:
            points[:, j] = points_circle[:, j]
        else:  # j == n_ta: outermost layer = octagon boundary
            points[:, j] = boundary_snapped

    # w_blend: 1=circle, 0=octagon (only outermost is octagon)
    w_blend = np.ones(Nv)
    w_blend[Nv - 1] = 0.0

    # Cell areas + aspect ratios
    cell_areas = np.zeros((Nu, Nv - 1))
    cell_aspect = np.zeros((Nu, Nv - 1))
    for i in range(Nu):
        i2 = (i + 1) % Nu
        for j in range(Nv - 1):
            p0 = points[i, j]
            p1 = points[i2, j]
            p2 = points[i2, j + 1]
            p3 = points[i, j + 1]
            cell_areas[i, j] = quad_area(p0, p1, p2, p3)
            e0 = np.linalg.norm(p1 - p0)
            e1 = np.linalg.norm(p2 - p1)
            e2 = np.linalg.norm(p3 - p2)
            e3 = np.linalg.norm(p0 - p3)
            ang_len = 0.5 * (e0 + e2)
            rad_len = 0.5 * (e1 + e3)
            if ang_len > 1e-12 and rad_len > 1e-12:
                cell_aspect[i, j] = max(ang_len, rad_len) / min(ang_len, rad_len)
            else:
                cell_aspect[i, j] = 1.0

    params = dict(d=d, n=n, n_ta=n_ta, xi=xi, n_transition=0,
                  alpha_cut=1.0,
                  h=h, s=s, hs=hs, R_cut=R_oct, R_oct=R_oct,
                  n_axis=n_axis, n_diag=n_diag, Nu=Nu, Nv=Nv,
                  angular_spacing=d, blend_mode='ta_m4_snapped',
                  snap_shift=best_shift)

    return dict(
        points=points,
        points_octagon=points_octagon,
        points_circle=points_circle,
        boundary_pts=boundary_snapped,
        boundary_pts_original=boundary_pts,
        phis=phis_uniform,
        phis_oct=phis_snapped,
        corners=corners,
        oct_verts=oct_verts,
        t_vals=t_vals,
        r_vals=r_vals,
        w_blend=w_blend,
        cell_areas=cell_areas,
        cell_aspect=cell_aspect,
        params=params,
        r_ta=r_ta,
        w_ta=w_ta,
    )


def mesh_diagnostics(nodes, polys, regions, grid_dict):
    """Compute mesh quality diagnostics.

    Returns dict with:
        angular_uniformity : max deviation from uniform angular spacing (degrees)
        edge_ratio_inner : max/min edge length among inner+transition cells only
        aspect_max : max cell aspect ratio (inner+transition only)
        aspect_mean : mean cell aspect ratio (inner+transition only)
        area_ratio_inner : max/min cell area among inner+transition cells
        chirality : measure of C4v rotational asymmetry (0=perfect)
        strain : max relative displacement of circle points from uniform-angle positions
    """
    p = grid_dict['params']
    phis = grid_dict['phis']
    Nu = p['Nu']
    r_vals = grid_dict['r_vals']

    # Angular uniformity — unwrap phis to continuous range
    phis_unwrapped = np.unwrap(phis)
    dphi = np.diff(phis_unwrapped)
    dphi = np.append(dphi, 2 * np.pi - (phis_unwrapped[-1] - phis_unwrapped[0]))
    dphi_uniform = 2 * np.pi / Nu
    angular_dev = np.max(np.abs(dphi - dphi_uniform)) * 180 / np.pi

    # Edge lengths / areas — inner+transition cells, excluding center fan (j=0)
    all_edges = []
    all_areas = []
    all_aspects = []
    for k, poly in enumerate(polys):
        if int(regions[k]) == 2:  # skip outer Cartesian
            continue
        # Skip center fan triangles (they have tiny edges by design)
        poly_pts = nodes[poly]
        r_max = np.max(np.linalg.norm(poly_pts, axis=1))
        if r_max < 0.01 * p['h']:  # center fan
            continue
        n = len(poly)
        edges = [np.linalg.norm(nodes[poly[(k2+1)%n]] - nodes[poly[k2]]) for k2 in range(n)]
        all_edges.extend(edges)
        area = _poly_area(nodes, poly)
        all_areas.append(area)
        if n == 4:
            e0, e1, e2, e3 = edges
            ang_len = 0.5 * (e0 + e2)
            rad_len = 0.5 * (e1 + e3)
            if ang_len > 1e-12 and rad_len > 1e-12:
                all_aspects.append(max(ang_len, rad_len) / min(ang_len, rad_len))

    edge_ratio = max(all_edges) / min(all_edges) if all_edges else 0
    area_ratio = max(all_areas) / min(all_areas) if all_areas else 0
    aspect_max = max(all_aspects) if all_aspects else 0
    aspect_mean = np.mean(all_aspects) if all_aspects else 0

    # Chirality: compare edge lengths under 90° rotation (C4v symmetry)
    points = grid_dict['points']
    Nu_pts, Nv_pts = points.shape[:2]
    chirality = 0.0
    for j in range(1, Nv_pts):
        for i in range(Nu_pts // 4):
            i_rot = (i + Nu_pts // 4) % Nu_pts
            e1 = np.linalg.norm(points[(i+1)%Nu_pts, j] - points[i, j])
            e2 = np.linalg.norm(points[(i_rot+1)%Nu_pts, j] - points[i_rot, j])
            if e1 > 1e-12:
                chirality = max(chirality, abs(e1 - e2) / e1)

    # Strain: non-uniformity of angular spacing (after optimal rotation alignment)
    # Measures how much the octagon angles deviate from uniform beyond a simple rotation
    phis_unwrapped = np.unwrap(phis)
    # Optimal rotation: align mean of octagon phis with mean of uniform phis
    uniform_phis = 2 * np.pi * np.arange(Nu) / Nu
    rotation = phis_unwrapped[0]  # first octagon angle
    # Shift uniform to match first octagon angle, then compare spacing
    uniform_shifted = uniform_phis + rotation
    # Compare angular positions modulo 2π
    angular_diff = np.abs(np.angle(np.exp(1j * (phis_unwrapped - uniform_shifted))))
    strain_max = np.max(angular_diff)

    return dict(
        angular_uniformity_deg=angular_dev,
        edge_ratio_inner=edge_ratio,
        area_ratio_inner=area_ratio,
        aspect_max=aspect_max,
        aspect_mean=aspect_mean,
        chirality=chirality,
        strain=strain_max,
    )


def build_cartesian_corners(h, d, margin=5):
    """Build Cartesian grid outside the octagon.

    Removes points inside the octagon (|x|+|y| <= h*sqrt(2)).
    Keeps corner triangles + outer region.
    """
    n_side = int(np.ceil(h / d)) + margin
    x_cart = np.arange(-n_side, n_side + 1) * d
    y_cart = np.arange(-n_side, n_side + 1) * d
    xx, yy = np.meshgrid(x_cart, y_cart)
    cart_xy = np.column_stack([xx.ravel(), yy.ravel()])
    cart_w = np.full(len(cart_xy), d * d)

    inside_oct = (np.abs(cart_xy[:, 0]) + np.abs(cart_xy[:, 1])
                  <= h * np.sqrt(2) + 1e-10)
    return cart_xy[~inside_oct], cart_w[~inside_oct]


def build_combined_octagon_grid(grid_dict, d=None, margin=5):
    """Combine inner octagon grid (cell centers) with outer Cartesian.

    Returns
    -------
    dict with inner_xy, inner_w, outer_xy, outer_w, combined_xy, combined_w
    """
    p = grid_dict['params']
    points = grid_dict['points']
    Nu, Nv = p['Nu'], p['Nv']
    h = p['h']
    d = d or p['d']

    inner_centers = np.zeros((Nu, Nv - 1, 2))
    inner_weights = np.zeros((Nu, Nv - 1))
    for i in range(Nu):
        i2 = (i + 1) % Nu
        for j in range(Nv - 1):
            inner_centers[i, j] = 0.25 * (
                points[i, j] + points[i2, j] +
                points[i2, j + 1] + points[i, j + 1])
            inner_weights[i, j] = grid_dict['cell_areas'][i, j]

    inner_xy = inner_centers.reshape(-1, 2)
    inner_w = inner_weights.ravel()

    outer_xy, outer_w = build_cartesian_corners(h, d, margin)

    combined_xy = np.vstack([inner_xy, outer_xy])
    combined_w = np.concatenate([inner_w, outer_w])

    return dict(
        inner_xy=inner_xy,
        inner_w=inner_w,
        outer_xy=outer_xy,
        outer_w=outer_w,
        combined_xy=combined_xy,
        combined_w=combined_w,
    )


def collect_unique_points(grid_dict):
    """Collect unique grid points from (Nu, Nv, 2) array.

    j=0: all Nu points at origin -> single center point.
    j>0: Nu distinct points per shell.
    """
    points = grid_dict['points']
    Nu, Nv = points.shape[:2]
    grid_pts = [np.array([0.0, 0.0])]
    grid_idx = {}
    for i in range(Nu):
        grid_idx[(i, 0)] = 0
    for j in range(1, Nv):
        for i in range(Nu):
            grid_idx[(i, j)] = len(grid_pts)
            grid_pts.append(points[i, j].copy())
    return np.array(grid_pts), grid_idx


# ── Full mesh with transition topology (quad-based) ────────────────────────────

def _poly_area(nodes, poly):
    """Area of a polygon (list of vertex indices), using shoelace formula."""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for k in range(n):
        k2 = (k + 1) % n
        s += nodes[poly[k], 0] * nodes[poly[k2], 1]
        s -= nodes[poly[k2], 0] * nodes[poly[k], 1]
    return abs(s) * 0.5


def _poly_centroid(nodes, poly):
    """Centroid of a polygon (list of vertex indices)."""
    pts = nodes[poly]
    return pts.mean(axis=0)


def build_full_mesh(grid_dict, d=None, margin=5):
    """Build complete quad-based mesh: inner octagon + transition corners + outer Cartesian.

    Corner triangles are filled with actual Cartesian grid cells:
      - Full quads for cells entirely inside the corner triangle
      - Natural triangles for cells cut by the diagonal (half-cell, no chirality)
    All boundary nodes are shared between regions via coordinate lookup.

    Returns
    -------
    nodes   : (N, 2)     — all unique mesh vertices
    polys   : list of lists — polygon vertex indices (3 for triangles, 4 for quads)
    regions : (T,)       — region label per polygon (0=inner, 1=transition, 2=outer)
    """
    p = grid_dict['params']
    h = p['h']
    d = d or p['d']
    Nu, Nv = p['Nu'], p['Nv']
    hs = p['hs']
    points = grid_dict['points']

    nodes_list = []
    node_lookup = {}

    def add_node(xy):
        key = (round(float(xy[0]), 8), round(float(xy[1]), 8))
        if key in node_lookup:
            return node_lookup[key]
        idx = len(nodes_list)
        nodes_list.append(np.array(xy, dtype=float))
        node_lookup[key] = idx
        return idx

    all_polys = []
    all_regions = []

    # === Inner octagon mesh (quads + center triangle fan) ===
    center_idx = add_node(np.array([0.0, 0.0]))
    shell_idx = {}
    for i in range(Nu):
        shell_idx[(i, 0)] = center_idx
    for j in range(1, Nv):
        for i in range(Nu):
            shell_idx[(i, j)] = add_node(points[i, j])

    for i in range(Nu):
        i2 = (i + 1) % Nu
        for j in range(Nv - 1):
            if j == 0:
                a, b, c = center_idx, shell_idx[(i, 1)], shell_idx[(i2, 1)]
                if a != b and b != c and a != c:
                    all_polys.append([a, b, c])
                    all_regions.append(0)
            else:
                a = shell_idx[(i, j)]
                b = shell_idx[(i2, j)]
                c = shell_idx[(i2, j + 1)]
                dd = shell_idx[(i, j + 1)]
                all_polys.append([a, b, c, dd])
                all_regions.append(0)

    # === Transition mesh: Cartesian grid cells in 4 corner triangles ===
    # Diagonal octagon edges pass through Cartesian grid vertices.
    # Corner triangles filled with real Cartesian cells: quads + natural half-cell triangles.
    n_h = int(round(h / d))
    n_hs = int(round(hs / d))
    threshold = n_h + n_hs

    # 4 corners: (a, b) = diagonal normal coefficients
    # Diagonal: a*ix + b*iy = threshold (grid units). Inside corner: >= threshold.
    # Cell corners: 0=(ix,iy), 1=(ix+1,iy), 2=(ix,iy+1), 3=(ix+1,iy+1)
    corner_configs = [
        (1, 1),    # NE: x+y >= h+hs
        (-1, 1),   # NW: -x+y >= h+hs
        (-1, -1),  # SW: -x-y >= h+hs
        (1, -1),   # SE: x-y >= h+hs
    ]

    for a, b in corner_configs:
        ix_lo = n_hs if a > 0 else -n_h
        ix_hi = n_h - 1 if a > 0 else -n_hs - 1
        iy_lo = n_hs if b > 0 else -n_h
        iy_hi = n_h - 1 if b > 0 else -n_hs - 1

        offset_min = min(0, a, a + b, b)
        v_min_corner = (1 if a < 0 else 0) + (2 if b < 0 else 0)

        for ix in range(ix_lo, ix_hi + 1):
            for iy in range(iy_lo, iy_hi + 1):
                v00 = a * ix + b * iy
                v_min = v00 + offset_min

                if v_min >= threshold:
                    # Full quad: corners 0,1,3,2 = (ix,iy),(ix+1,iy),(ix+1,iy+1),(ix,iy+1)
                    i00 = add_node([ix * d, iy * d])
                    i10 = add_node([(ix + 1) * d, iy * d])
                    i11 = add_node([(ix + 1) * d, (iy + 1) * d])
                    i01 = add_node([ix * d, (iy + 1) * d])
                    all_polys.append([i00, i10, i11, i01])
                    all_regions.append(1)
                elif v_min == threshold - 1:
                    # Triangle: 3 corners excluding v_min_corner
                    cell_pts = [
                        (ix, iy),       # 0
                        (ix + 1, iy),   # 1
                        (ix, iy + 1),   # 2
                        (ix + 1, iy + 1),  # 3
                    ]
                    tri = [cell_pts[k] for k in range(4) if k != v_min_corner]
                    idx_list = [add_node([cx * d, cy * d]) for cx, cy in tri]
                    all_polys.append(idx_list)
                    all_regions.append(1)

    # === Outer Cartesian mesh (quads outside square) ===
    n_side = n_h + margin
    cart_grid = {}
    for ix in range(-n_side, n_side + 1):
        for iy in range(-n_side, n_side + 1):
            xy = np.array([ix * d, iy * d])
            if not (abs(xy[0]) < h - 1e-10 and abs(xy[1]) < h - 1e-10):
                cart_grid[(ix, iy)] = add_node(xy)

    for ix in range(-n_side, n_side):
        for iy in range(-n_side, n_side):
            keys = [(ix, iy), (ix + 1, iy), (ix + 1, iy + 1), (ix, iy + 1)]
            indices = [cart_grid.get(k) for k in keys]
            if all(idx is not None for idx in indices):
                all_polys.append(indices)
                all_regions.append(2)

    nodes = np.array(nodes_list)
    regions = np.array(all_regions)
    return nodes, all_polys, regions


def extract_vertex_points(nodes, polys, regions):
    """Extract vertex-based quadrature points.

    Each vertex gets weight = 1/n * sum of adjacent polygon areas
    (n = number of vertices of each polygon: 3 for triangles, 4 for quads).

    Returns
    -------
    pts : (N, 2) — same as nodes
    w   : (N,)   — weight per vertex
    """
    w = np.zeros(len(nodes))
    for poly in polys:
        area = _poly_area(nodes, poly)
        n = len(poly)
        for k in range(n):
            w[poly[k]] += area / n
    return nodes.copy(), w


def extract_cell_center_points(nodes, polys, regions):
    """Extract cell-center (centroid) quadrature points.

    Each polygon contributes one point at its centroid with weight = area.

    Returns
    -------
    pts : (T, 2) — polygon centroids
    w   : (T,)   — polygon areas
    """
    pts = np.zeros((len(polys), 2))
    w = np.zeros(len(polys))
    for k, poly in enumerate(polys):
        pts[k] = _poly_centroid(nodes, poly)
        w[k] = _poly_area(nodes, poly)
    return pts, w


# ── Interface classification & spherical weights ──────────────────────────────

def classify_nodes(nodes, polys, regions, grid_dict):
    """Classify each node as 'inner', 'near_interface', 'interface', or 'outer'.

    For the snapped TA-M4 grid:
      - 'interface'     : nodes on the snapped octagon layer (j=n_ta)
      - 'near_interface': nodes on the last uniform circle layer (j=n_ta-1)
      - 'inner'         : nodes inside the near_interface layer
      - 'outer'         : nodes in the Cartesian grid outside the octagon

    Nodes shared between inner and outer regions (corner triangles) are
    also classified as 'interface'.

    Returns
    -------
    labels : (N,) array of strings
    """
    p = grid_dict['params']
    n_ta = p.get('n_ta', 0)
    Nv = p.get('Nv', 0)
    r_vals = grid_dict.get('r_vals', None)

    # Identify interface and near_interface by radius
    r_interface = r_vals[n_ta] if n_ta < len(r_vals) else None
    r_near = r_vals[n_ta - 1] if n_ta >= 2 and n_ta - 1 < len(r_vals) else None

    # Also use region-based classification for corner/outer boundary nodes
    node_regions = [set() for _ in range(len(nodes))]
    for k, poly in enumerate(polys):
        for vi in poly:
            node_regions[vi].add(int(regions[k]))

    labels = np.empty(len(nodes), dtype=object)
    for i in range(len(nodes)):
        r_i = np.linalg.norm(nodes[i])
        regs = node_regions[i]

        # Check if on snapped layer (interface)
        if r_interface is not None and abs(r_i - r_interface) < 1e-8 * max(r_interface, 1):
            labels[i] = 'interface'
        # Check if on near-interface layer
        elif r_near is not None and abs(r_i - r_near) < 1e-8 * max(r_near, 1):
            labels[i] = 'near_interface'
        # Region-based: shared between inner and outer/transition
        elif len(regs) > 1:
            labels[i] = 'interface'
        elif regs == {0}:
            labels[i] = 'inner'
        elif regs == {2}:
            labels[i] = 'outer'
        elif regs == {1}:
            labels[i] = 'interface'
        else:
            labels[i] = 'inner'

    return labels


def classify_polys(nodes, polys, regions, grid_dict):
    """Classify each polygon as 'inner', 'near_interface', 'interface', or 'outer'.

    A polygon is:
      - 'interface'      if it contains at least one interface node
      - 'near_interface' if it contains at least one near_interface node (but no interface)
      - 'inner'          if all nodes are inner
      - 'outer'          if in the outer Cartesian region

    Returns
    -------
    labels : (T,) array of strings
    """
    node_labels = classify_nodes(nodes, polys, regions, grid_dict)

    labels = np.empty(len(polys), dtype=object)
    for k, poly in enumerate(polys):
        has_interface = any(node_labels[vi] == 'interface' for vi in poly)
        has_near = any(node_labels[vi] == 'near_interface' for vi in poly)
        if has_interface:
            labels[k] = 'interface'
        elif has_near:
            labels[k] = 'near_interface'
        elif regions[k] == 0:
            labels[k] = 'inner'
        elif regions[k] == 2:
            labels[k] = 'outer'
        else:
            labels[k] = 'interface'
    return labels


def compute_spherical_weights(grid_dict):
    """Compute weights for inner spherical cells (depend only on radial index j).

    For a circular shell grid, all cells in the same shell j have equal weight.
    Weight = area of the annular shell / Nu.

    Returns
    -------
    w_spherical : (Nu, Nv-1) — weight per inner cell (same for all i at given j)
    """
    p = grid_dict['params']
    Nu, Nv = p['Nu'], p['Nv']
    r_vals = grid_dict['r_vals']

    w_spherical = np.zeros((Nu, Nv - 1))
    for j in range(Nv - 1):
        r_inner = r_vals[j]
        r_outer = r_vals[j + 1]
        shell_area = np.pi * (r_outer**2 - r_inner**2)
        w_spherical[:, j] = shell_area / Nu

    return w_spherical


def compute_ta_weights(grid_dict):
    """Compute TA-M4 weights for inner radial cells.

    For TA-M4 grid, the inner shells (j=1..n_ta) have weights from the
    Chebyshev quadrature. Weight per cell = w_ta[j-1] * (2π/Nu).

    The transition shells (j=n_ta+1..Nv-1) do NOT have TA-M4 weights;
    they use geometric weights and need optimization.

    Returns
    -------
    w_ta_cells : (Nu, Nv-1) — weight per cell, NaN for non-TA-M4 cells
    """
    p = grid_dict['params']
    Nu, Nv = p['Nu'], p['Nv']
    n_ta = p.get('n_ta', 0)
    w_ta = grid_dict.get('w_ta', None)

    w_ta_cells = np.full((Nu, Nv - 1), np.nan)
    if w_ta is None or n_ta == 0:
        return w_ta_cells

    angular_w = 2 * np.pi / Nu
    # TA-M4 shells: cells between j and j+1 for j=0..n_ta-1
    # j=0 is center → triangle fan, weight = w_ta[0] * angular_w / Nu
    # (center point gets full shell weight, each triangle gets 1/Nu of it)
    for j in range(n_ta):
        if j == 0:
            # Center fan: each triangle gets w_ta[0] * angular_w / Nu
            w_ta_cells[:, 0] = w_ta[0] * angular_w / Nu
        else:
            # Shell j: between r_ta[j-1] and r_ta[j]
            w_ta_cells[:, j] = w_ta[j] * angular_w / Nu

    return w_ta_cells


def extract_vertex_points_classified(nodes, polys, regions, grid_dict):
    """Extract vertex-based quadrature points with classification.

    Returns
    -------
    pts     : (N, 2) — node positions
    w       : (N,)   — weight per vertex (1/n * adjacent poly areas)
    labels  : (N,)   — 'inner', 'outer', or 'interface'
    """
    w = np.zeros(len(nodes))
    for poly in polys:
        area = _poly_area(nodes, poly)
        n = len(poly)
        for k in range(n):
            w[poly[k]] += area / n
    labels = classify_nodes(nodes, polys, regions, grid_dict)
    return nodes.copy(), w, labels


def extract_cell_center_points_classified(nodes, polys, regions, grid_dict):
    """Extract cell-center quadrature points with classification.

    Returns
    -------
    pts     : (T, 2) — polygon centroids
    w       : (T,)   — polygon areas
    labels  : (T,)   — 'inner', 'outer', or 'interface'
    """
    pts = np.zeros((len(polys), 2))
    w = np.zeros(len(polys))
    for k, poly in enumerate(polys):
        pts[k] = _poly_centroid(nodes, poly)
        w[k] = _poly_area(nodes, poly)
    labels = classify_polys(nodes, polys, regions, grid_dict)
    return pts, w, labels
