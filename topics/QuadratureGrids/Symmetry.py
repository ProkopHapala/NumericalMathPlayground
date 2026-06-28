"""
Symmetry.py — Point group symmetry operations for grid points.

Core module: C4v orbit computation and grouping.
No I/O, no plotting.
"""
import numpy as np


def c4v_orbit(x, y, tol=1e-10):
    """Return list of unique (x,y) points under C4v symmetry."""
    transforms = [
        (x, y), (-y, x), (-x, -y), (y, -x),
        (-x, y), (y, x), (-y, -x), (x, -y)
    ]
    unique = []
    for tx, ty in transforms:
        found = False
        for ux, uy in unique:
            if abs(tx - ux) < tol and abs(ty - uy) < tol:
                found = True
                break
        if not found:
            unique.append((tx, ty))
    return unique


def group_orbits(points, symmetry_fn=c4v_orbit, tol=1e-9):
    """
    Group points into symmetry orbits.

    Parameters
    ----------
    points : (N, 2) array
    symmetry_fn : callable (x, y) -> list of (x, y) transforms
    tol : float — tolerance for matching transformed points

    Returns
    -------
    orbit_id : (N,) int array — orbit index for each point
    orbit_members : list of lists — point indices per orbit
    orbit_rep : list of (x, y) — representative point per orbit
    """
    Npts = len(points)
    orbit_id = np.full(Npts, -1, dtype=int)
    orbit_members = []
    orbit_rep = []

    for p in range(Npts):
        if orbit_id[p] >= 0:
            continue
        x, y = points[p]
        orbit_pts = symmetry_fn(x, y)
        members = []
        for p2 in range(Npts):
            if orbit_id[p2] >= 0:
                continue
            x2, y2 = points[p2]
            for tx, ty in orbit_pts:
                if abs(x2 - tx) < tol and abs(y2 - ty) < tol:
                    members.append(p2)
                    break
        oid = len(orbit_rep)
        for m in members:
            orbit_id[m] = oid
        orbit_rep.append(points[members[0]].copy())
        orbit_members.append(members)

    return orbit_id, orbit_members, orbit_rep
