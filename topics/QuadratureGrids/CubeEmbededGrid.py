"""
CubeEmbededGrid.py — CubeEmbededAtoms grid construction (2D).

Core module: grid generation, cell areas, combined inner+outer grid.
No I/O, no plotting.

The grid embeds a radial wedge/sphere-blended grid inside a Cartesian
cutout region, seamlessly connecting to the outer Cartesian grid.
"""
import numpy as np


# ── Helper functions ──────────────────────────────────────────────────────────

def smoothstep(t):
    return t * t * (3 - 2 * t)


def smootherstep(t):
    """Ken Perlin's smootherstep: 6t⁵-15t⁴+10t³. C2 continuous."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def radial_map(t, alpha):
    """Treutler-like: t∈[0,1] → r∈[0,h], dense near center for alpha>1."""
    return t ** alpha


def quad_area(p0, p1, p2, p3):
    """Area of a quadrilateral given 4 vertex arrays (each (...,2))."""
    return 0.5 * np.abs(
        p0[..., 0] * (p1[..., 1] - p3[..., 1]) +
        p1[..., 0] * (p2[..., 1] - p0[..., 1]) +
        p2[..., 0] * (p3[..., 1] - p1[..., 1]) +
        p3[..., 0] * (p0[..., 1] - p2[..., 1])
    )


# ── Grid construction ─────────────────────────────────────────────────────────

def build_grid(d=0.5, n=10, alpha=1.8, n_blend=4):
    """
    Build CubeEmbededAtoms grid (2D prototype).

    Parameters
    ----------
    d : float         — Cartesian grid step
    n : int           — cutout size in voxels per side
    alpha : float     — radial mapping exponent (>1 = denser near center)
    n_blend : int     — blend exponent (w = 1 - t^n_blend)

    Returns
    -------
    dict with keys:
        points       : (Nu, Nv, 2) — blended grid points
        points_wedge : (Nu, Nv, 2) — pure linear wedge grid
        points_sphere: (Nu, Nv, 2) — pure spherical grid
        edge_pts     : (Nu, 2)     — square edge points (boundary)
        circle_phis  : (Nu,)       — uniform angles for spherical grid
        corners      : (4, 2)      — square corners
        t_vals       : (Nv,)       — radial parameter
        r_vals       : (Nv,)       — actual radii
        w_blend      : (Nv,)       — blend weight per shell
        cell_areas   : (Nu, Nv-1)  — area of each cell
        cell_aspect  : (Nu, Nv-1)  — aspect ratio of each cell
        params       : dict        — all input params + derived (h, L, Nu, Nv, ...)
    """
    L = n * d
    h = L / 2
    n_per_wedge = n
    Nw = 4
    Nu = Nw * n_per_wedge

    angular_spacing = 2 * h / n_per_wedge
    target_ratio = 1 - angular_spacing / h
    base = target_ratio ** (1.0 / alpha)
    Nv_base = int(np.ceil(1 / (1 - base))) + 1
    Nv = int(np.ceil(Nv_base * np.sqrt(2)))

    corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    corner_angles = np.arctan2(corners[:, 1], corners[:, 0])
    corner_angles = np.where(corner_angles < 0, corner_angles + 2 * np.pi, corner_angles)
    wedge_start = corner_angles.copy()
    wedge_end = np.roll(corner_angles, -1)
    wedge_end[wedge_end < wedge_start] += 2 * np.pi

    t_vals = np.linspace(0, 1, Nv)
    r_vals = h * radial_map(t_vals, alpha)

    edge_pts = np.zeros((Nu, 2))
    circle_phis = np.zeros(Nu)
    for i_w in range(Nw):
        for i_loc in range(n_per_wedge):
            i = i_w * n_per_wedge + i_loc
            u_local = i_loc / n_per_wedge
            c0 = corners[i_w]
            c1 = corners[(i_w + 1) % Nw]
            edge_pts[i] = c0 + (c1 - c0) * u_local
            circle_phis[i] = wedge_start[i_w] + (wedge_end[i_w] - wedge_start[i_w]) * u_local

    # Pure wedge
    points_wedge = np.zeros((Nu, Nv, 2))
    for j in range(Nv):
        scale = r_vals[j] / h
        points_wedge[:, j, 0] = scale * edge_pts[:, 0]
        points_wedge[:, j, 1] = scale * edge_pts[:, 1]

    # Pure sphere
    points_sphere = np.zeros((Nu, Nv, 2))
    for j in range(Nv):
        points_sphere[:, j, 0] = r_vals[j] * np.cos(circle_phis)
        points_sphere[:, j, 1] = r_vals[j] * np.sin(circle_phis)

    # Blended
    w_blend = 1 - t_vals ** n_blend
    points = np.zeros((Nu, Nv, 2))
    for j in range(Nv):
        w = w_blend[j]
        points[:, j, 0] = w * points_sphere[:, j, 0] + (1 - w) * points_wedge[:, j, 0]
        points[:, j, 1] = w * points_sphere[:, j, 1] + (1 - w) * points_wedge[:, j, 1]

    # Cell areas + aspect ratios
    cell_areas = np.zeros((Nu, Nv - 1))
    cell_aspect = np.zeros((Nu, Nv - 1))
    for i in range(Nu):
        i2 = (i + 1) % Nu
        for j in range(Nv - 1):
            p0, p1, p2, p3 = points[i, j], points[i2, j], points[i2, j + 1], points[i, j + 1]
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

    params = dict(d=d, n=n, alpha=alpha, n_blend=n_blend, L=L, h=h,
                  n_per_wedge=n_per_wedge, Nw=Nw, Nu=Nu, Nv=Nv,
                  angular_spacing=angular_spacing)

    return dict(
        points=points, points_wedge=points_wedge, points_sphere=points_sphere,
        edge_pts=edge_pts, circle_phis=circle_phis, corners=corners,
        t_vals=t_vals, r_vals=r_vals, w_blend=w_blend,
        cell_areas=cell_areas, cell_aspect=cell_aspect, params=params,
    )


def build_cartesian_outer(h, d, margin=5):
    """Build Cartesian grid outside the cutout region [-h,h]²."""
    x_cart = np.arange(-h - margin * d, h + margin * d + d / 2, d)
    y_cart = np.arange(-h - margin * d, h + margin * d + d / 2, d)
    xx, yy = np.meshgrid(x_cart, y_cart)
    cart_xy = np.column_stack([xx.ravel(), yy.ravel()])
    cart_w = np.full(len(cart_xy), d * d)
    inside = (np.abs(cart_xy[:, 0]) < h) & (np.abs(cart_xy[:, 1]) < h)
    return cart_xy[~inside], cart_w[~inside]


def build_combined_grid(grid_dict, d=None, margin=5):
    """
    Build combined grid: inner cell-center points + outer Cartesian.

    Returns
    -------
    inner_xy    : (M_inner, 2) — cell center coordinates
    inner_w     : (M_inner,)   — cell areas as weights
    outer_xy    : (M_outer, 2) — Cartesian outer grid points
    outer_w     : (M_outer,)   — Cartesian weights (d²)
    combined_xy : (M_total, 2)
    combined_w  : (M_total,)
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
                points[i, j] + points[i2, j] + points[i2, j + 1] + points[i, j + 1])
            inner_weights[i, j] = grid_dict['cell_areas'][i, j]

    inner_xy = inner_centers.reshape(-1, 2)
    inner_w = inner_weights.ravel()

    outer_xy, outer_w = build_cartesian_outer(h, d, margin)

    combined_xy = np.vstack([inner_xy, outer_xy])
    combined_w = np.concatenate([inner_w, outer_w])

    return dict(
        inner_xy=inner_xy, inner_w=inner_w,
        outer_xy=outer_xy, outer_w=outer_w,
        combined_xy=combined_xy, combined_w=combined_w,
    )


def collect_unique_points(grid_dict):
    """
    Collect unique grid points from the (Nu, Nv, 2) array.
    j=0: all Nu points at origin → single center point.
    j>0: Nu distinct points per shell.

    Returns
    -------
    grid_pts : (Npts, 2)
    grid_idx : dict (i, j) -> point index
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
