#!/usr/bin/python
r"""
angular_sampling.py — Angular Sampling on Molecular Distance Fields
====================================================================

Tutorial / Reference for students, developers, and future users
---------------------------------------------------------------

**Goal.**
When computing DFT reference data for non-bonded interactions, we need to place
probe molecules at various distances and orientations around a central molecule.
The *radial* placement is straightforward — use a soft-minimum molecular distance
field r_mol and pick isoline levels.  The hard part is *angular* sampling:
how to distribute N points *uniformly* along a closed isoline of r_mol, which
is generally not circular and may have concavities, multiple lobes, etc.

This module implements and compares several strategies for that angular sampling,
selects the best one, and produces visualization figures.


Mathematical Background
-----------------------

**Molecular distance field.**
Two formulas for r_mol are implemented and compared:

1. **Logsumexp soft-min** (used for final sampling):

   .. math::
       r_\text{mol}(\mathbf{x}) = -\frac{1}{\beta}\log\sum_j \exp\!\big[-\beta\,(|\mathbf{x}-\mathbf{r}_j| - R_j)\big]

   where :math:`R_j` are vdW radii and :math:`\beta` controls sharpness.
   At :math:`\beta\to\infty` this becomes the exact signed distance to the
   nearest vdW sphere surface.  The gradient is

   .. math::
       \nabla r_\text{mol} = \sum_j p_j\,\hat{\mathbf{n}}_j, \qquad
       p_j = \text{softmax}(-\beta\,d_j), \qquad
       \hat{\mathbf{n}}_j = \frac{\mathbf{x}-\mathbf{r}_j}{|\mathbf{x}-\mathbf{r}_j|}

2. **Reciprocal power-mean** (compared but not used for sampling):

   .. math::
       r_\text{mol}(\mathbf{x}) = \frac{1}{\big(\sum_j (R_j/|\mathbf{x}-\mathbf{r}_j|)^n\big)^{1/n}}

   This is always positive (→0 near atoms, →∞ far away), so the vdW surface
   sits at r_mol ≈ 1.0 rather than 0.0.  We shift by +1 for comparison plots.

   **Key insight:** the logsumexp formula can produce *negative* r_mol (inside
   vdW spheres), which is physically meaningful — it means the point is inside
   the molecular volume.  The power-mean formula cannot.  This is why we use
   logsumexp for the final sampling.

**Complex harmonic phase — the topological angular coordinate.**
For N atoms at positions :math:`z_j = x_j + i\,y_j` with integer charges
:math:`q_j` (usually all 1), define:

.. math::
   F(z) = \prod_j (z - z_j)^{q_j}

The **phase** :math:`\Phi(z) = \arg F(z) = \sum_j q_j \arg(z - z_j)` is a
multi-valued function whose total winding around all atoms is
:math:`\Delta\Phi = 2\pi Q` where :math:`Q = \sum_j q_j`.

This is a **topological invariant** — the winding number of the map
:math:`S^1 \to U(1)`.  Each atom acts as a phase vortex of charge :math:`q_j`.
The phase gradient is analytic:

.. math::
   \nabla\Phi = (\text{Im}\,S,\;\text{Re}\,S), \qquad
   S(z) = \frac{F'(z)}{F(z)} = \sum_j \frac{q_j}{z - z_j}

**Important:** :math:`\Phi` is *not* the harmonic conjugate of r_mol (r_mol is
not harmonic — :math:`\Delta r_\text{mol} \neq 0`).  We use r_mol as the radial
coordinate and :math:`\Phi` only as the angular coordinate.  Equal phase steps
give **harmonic-measure sampling**, not equal-arc-length sampling.  A local
relaxation step corrects this.

**Why not just use uniform polar angles?**
Gradient trajectories from the molecular center are not radial rays — they
converge, diverge, and can be blocked by critical points (:math:`\nabla u = 0`).
Uniform angular density at the center is NOT preserved by gradient flow.

**Impossibility result.**
A globally orthogonal coordinate :math:`(r_\text{mol}, \phi)` that is
simultaneously equal-arc-length on every isoline generally does NOT exist.
The integrability condition requires :math:`\kappa/|\nabla u|` to be constant
along each isoline, which fails for general molecular shapes.  We can have
either consistent phase transport OR independent equal-arc-length, but not
both exactly.  Our hybrid approach gets close to both.


Methods Compared
----------------

1. **Naive angular rays** (rejected):
   Start from uniform polar angles at center, walk along :math:`\nabla r_\text{mol}`.
   Fails because gradient flow distorts angular density badly near concavities.

2. **Tangential relaxation only** (works, but no phase correspondence):
   Place particles on level set, redistribute tangentially via Laplacian
   smoothing.  Gives equal arc length on each shell independently, but points
   on neighboring shells have no angular correspondence — you cannot track
   which point on shell m corresponds to which point on shell m+1.

3. **Complex phase continuation + local gap relaxation** (★ chosen method):
   - Trace the r_mol = c contour using uniform *unwrapped* phase steps
     :math:`\Delta\Psi = 2\pi Q / M`
   - At each step: predict along tangent by :math:`ds = \Delta\Psi / (\nabla\Phi \cdot \mathbf{t})`,
     then Newton-correct to exact :math:`(r_\text{mol}=c,\;\Phi=\Psi_k)`
   - After full contour: apply a few iterations of local gap relaxation
     (tangential spring forces) to equalize arc length
   - Transport between shells: preserve phase identity, predict via gradient
     flow :math:`\Delta\mathbf{x} = \Delta c \cdot \nabla u / |\nabla u|^2`,
     Newton-correct to new level with same phase

   **Why this is best:**
   - Topological guarantee: exactly M points, one per phase branch
   - Natural angular correspondence between shells (same phase = same "angle")
   - Local relaxation fixes the harmonic-measure → equal-arc-length discrepancy
   - Works with concavities, multiple atoms, arbitrary molecular shapes
   - All evaluation is analytic (no grid interpolation for sampling)


Sampling Strategy
-----------------

The final sampling grid is :math:`[n_\text{radial} \times n_\phi]` where:

- **Radial levels** are chosen from a fixed set of r_mol values
  (ISOLINE_LEVELS), e.g. [-0.2, -0.1, 0.0, 0.1, ..., 20.0]
- **Angular points** per shell = n_points (default 25)
- Sampling starts from a **middle shell** (r_mol=0.0) where uniformity is best,
  then transports **outward** and **inward** to all other shells
- At each sample point, the gradient :math:`\nabla r_\text{mol}` gives the
  **radial direction** (perpendicular to the isoline) — this is the orientation
  for placing a probe molecule

The output is a 2D array of (position, gradient) pairs that can be used to
place probe molecules at controlled distances and orientations.


Usage
-----

Basic comparison plot (2×3 grid: power n=[1,2,4] vs logsumexp β=[1,2,4])::

    python angular_sampling.py
    python angular_sampling.py --target data/xyz/HCOOH.xyz --n-points 25 --charges all

Full sampling grid with gradient arrows (β=4 only)::

    python angular_sampling.py --target data/xyz/HCOOH.xyz --n-points 25 --charges all

CLI arguments:
  --target    : Path to molecule XYZ file (default: data/xyz/HCOOH.xyz)
  --n-points  : Number of angular samples per shell (default: 25)
  --margin    : Grid margin in Å for plotting (default: 3.5)
  --step      : Grid step in Å for plotting (default: 0.05)
  --charges   : Phase charges: "all" (q_j=1 for all atoms) or "heavy" (skip H)
  --save      : Output directory (default: fig_hsamp/)

Outputs:
  - ``rmol_comparison_<molecule>.png`` — 2×3 comparison of r_mol formulas
  - ``sampling_grid_<molecule>_b4.png`` — full [n_radial × n_phi] grid with
    gradient arrows on isolines (β=4 logsumexp)


Open Issues and Challenges
--------------------------

1. **Topology changes at critical levels:**
   Isolines can split or merge at critical points where :math:`\nabla r_\text{mol} = 0`.
   The current code assumes a single closed contour.  For molecules with deep
   concavities at high β, inner shells may develop multiple components.

2. **Phase charge selection:**
   The choice of :math:`q_j` (which atoms get phase weight) affects sampling
   density.  Using all atoms (Q=N) gives more angular resolution but may
   over-sample near H atoms.  Using only heavy atoms (Q=N_heavy) gives fewer
   points per hue cycle.  The optimal choice is molecule-dependent.

3. **Equal-arc-length vs. equal-configuration-space:**
   Current relaxation equalizes arc length.  A more physically motivated metric
   would account for normal-direction rotation (curvature), placing more points
   where the surface normal rotates rapidly.  See AngularPhaseSampling.md §7.

4. **3D generalization:**
   The 2D complex phase has no direct 3D analogue.  Extension to 3D would
   require either: (a) sampling on 2D iso-surfaces with tangent-plane particle
   repulsion, or (b) using multiple slicing planes with 2D phase sampling on each.

5. **Large r_mol shells:**
   Far from the molecule (r_mol > 10), isolines become nearly circular and
   N points may be too few.  The gap uniformity (rel) stays good due to phase
   transport, but absolute spacing grows large.

6. **Seed point sensitivity:**
   The initial seed point for phase tracing affects which physical point gets
   phase :math:`\Phi_0`.  Different seeds rotate the entire sampling pattern.
   This is not a bug but a convention to be aware of.

See ``AngularPhaseSampling.md`` for the full mathematical derivation and
discussion, including the ChatGPT conversation that motivated this work.
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from matplotlib.colors import hsv_to_rgb

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from py.AtomicSystem import AtomicSystem
from py.elements import ELEMENTS

ELEM_RVDW = {row[1]: row[7] for row in ELEMENTS}
ELEM_COLORS = {row[1]: row[8] for row in ELEMENTS}

ISOLINE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 15.0, 20.0]


# ========== Distance field (corrected) ==========

def get_vdw_radii(mol):
    return np.array([ELEM_RVDW.get(e, 1.7) for e in mol.enames])


def eval_rmol(xy, apos, rvdw, beta, z_height=0.0):
    """Evaluate corrected soft molecular distance and xy gradient at arbitrary points.

    d_j = |r - r_j| - R_j
    u = -1/β · log( Σ_j exp(-β·d_j) )
    ∇u = Σ_j p_j · n_j,  p_j = softmax(-β·d_j),  n_j = (r-r_j)/|r-r_j|

    Args:
        xy: (M, 2) query positions
        apos: (N, 3) atom positions
        rvdw: (N,) vdW radii
        beta: sharpness parameter
    Returns: u (M,), grad (M, 2)  (xy components only)
    """
    xy = np.asarray(xy, dtype=float)
    dxy = xy[:, None, :] - apos[None, :, :2]       # (M, N, 2)
    dz = z_height - apos[None, :, 2]                 # (1, N)
    rr = np.sqrt(np.sum(dxy*dxy, axis=2) + dz*dz)    # (M, N)
    rr_safe = np.maximum(rr, 1e-12)

    exponents = -beta * (rr - rvdw[None, :])         # corrected: d_j = r_j - R_j
    amax = np.max(exponents, axis=1, keepdims=True)
    weights = np.exp(exponents - amax)
    wsum = np.sum(weights, axis=1, keepdims=True)
    probs = weights / wsum                           # (M, N)

    u = -(amax[:, 0] + np.log(wsum[:, 0])) / beta

    atom_grad = dxy / rr_safe[:, :, None]            # (M, N, 2)
    grad = np.sum(probs[:, :, None] * atom_grad, axis=1)  # (M, 2)
    return u, grad


def eval_rmol_power(xy, apos, rvdw, n_exp, z_height=0.0):
    """Reciprocal power-mean molecular distance: r_mol = 1 / (Σ_j (R_j/d_j)^n)^(1/n).

    Near atoms r_mol → 0 (inside), far from atoms r_mol → ∞ (outside).
    At the vdW surface of a single atom (d_j=R_j), r_mol = 1.

    S = Σ_j (R_j / d_j)^n
    r_mol = S^(-1/n)
    ∇r_mol = S^(-1/n-1) · Σ_j R_j^n · d_j^(-n-2) · (x - x_j)

    Args:
        xy: (M, 2) query positions
        apos: (N, 3) atom positions
        rvdw: (N,) vdW radii
        n_exp: exponent n (1, 2, 4, ...)
    Returns: u (M,), grad (M, 2)
    """
    xy = np.asarray(xy, dtype=float)
    dxy = xy[:, None, :] - apos[None, :, :2]       # (M, N, 2)
    dz = z_height - apos[None, :, 2]                 # (1, N)
    rr = np.sqrt(np.sum(dxy*dxy, axis=2) + dz*dz)    # (M, N)
    rr_safe = np.maximum(rr, 1e-12)

    # S = Σ_j (R_j / d_j)^n
    ratios = rvdw[None, :] / rr_safe                  # (M, N)
    S = np.sum(ratios**n_exp, axis=1)                 # (M,)
    S_safe = np.maximum(S, 1e-30)

    u = 1.0 / S_safe**(1.0/n_exp)                     # (M,)

    # ∇r_mol = S^(-1/n-1) · Σ_j R_j^n · d_j^(-n-2) · (x - x_j)
    factor = S_safe**(-1.0/n_exp - 1.0)               # (M,)
    weight_j = rvdw[None, :]**n_exp / rr_safe**(n_exp + 2)  # (M, N)

    grad = factor[:, None] * np.sum(weight_j[:, :, None] * dxy, axis=1)  # (M, 2)
    return u, grad


def compute_r_mol_grid(mol, beta, margin=8.0, step=0.05, z_height=0.0, eval_func=None):
    """Compute r_mol on a 2D grid (for visualization).

    Args:
        eval_func: callable(xy) -> (u, grad). If None, uses eval_rmol with beta.
    """
    apos = mol.apos
    rvdw = get_vdw_radii(mol)
    if eval_func is None:
        eval_func = lambda xy: eval_rmol(xy, apos, rvdw, beta, z_height)
    xmin, ymin = apos[:, 0].min() - margin, apos[:, 1].min() - margin
    xmax, ymax = apos[:, 0].max() + margin, apos[:, 1].max() + margin
    xs = np.arange(xmin, xmax + step, step)
    ys = np.arange(ymin, ymax + step, step)
    X, Y = np.meshgrid(xs, ys)
    xy = np.stack([X.ravel(), Y.ravel()], axis=1)
    u, grad = eval_func(xy)
    r_mol = u.reshape(len(ys), len(xs))
    grad_x = grad[:, 0].reshape(len(ys), len(xs))
    grad_y = grad[:, 1].reshape(len(ys), len(xs))
    extent = [xmin, xmax, ymin, ymax]
    return r_mol, grad_x, grad_y, xs, ys, extent


# ========== Complex harmonic field ==========

def compute_complex_field_grid(mol, margin=8.0, step=0.05, charges=None):
    """Compute F(z) = Π_j (z - z_j)^{q_j} on a 2D grid.

    Args:
        mol: AtomicSystem
        charges: (N,) integer charges q_j. Default: all 1.
    Returns:
        F (ny, nx) complex, U=log|F| (ny,nx), V=arg(F) (ny,nx), xs, ys, extent
    """
    apos = mol.apos
    if charges is None:
        charges = np.ones(len(apos), dtype=int)
    else:
        charges = np.asarray(charges, dtype=int)

    xmin, ymin = apos[:, 0].min() - margin, apos[:, 1].min() - margin
    xmax, ymax = apos[:, 0].max() + margin, apos[:, 1].max() + margin
    xs = np.arange(xmin, xmax + step, step)
    ys = np.arange(ymin, ymax + step, step)
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y  # (ny, nx)

    F = np.ones_like(Z)
    for j in range(len(apos)):
        zj = apos[j, 0] + 1j * apos[j, 1]
        F *= (Z - zj) ** charges[j]

    U = np.log(np.abs(F) + 1e-30)  # log|F|
    V = np.angle(F)                  # arg(F) ∈ (-π, π]
    extent = [xmin, xmax, ymin, ymax]
    return F, U, V, xs, ys, extent


def eval_complex_field(xy, apos, charges=None):
    """Evaluate F(z) at arbitrary points. Returns F (M,) complex."""
    xy = np.asarray(xy, dtype=float)
    z = xy[:, 0] + 1j * xy[:, 1]
    if charges is None:
        charges = np.ones(len(apos), dtype=int)
    F = np.ones(len(z), dtype=complex)
    for j in range(len(apos)):
        zj = apos[j, 0] + 1j * apos[j, 1]
        F *= (z - zj) ** charges[j]
    return F


# ========== Complex phase evaluation ==========

def eval_complex_phase(xy, apos, charges=None):
    """Unwrapped local phase value and analytic phase gradient.

    Phi(z) = sum_j q_j * arg(z - z_j)
    grad Phi = (Im(S), Re(S))  where S = sum_j q_j / (z - z_j)

    Args:
        xy: (M, 2) query positions
        apos: (N, 3) or (N, 2) atom positions
        charges: (N,) integer q_j; default all 1
    Returns: phi (M,), grad_phi (M, 2)
    """
    xy = np.asarray(xy, dtype=float)
    apos2 = np.asarray(apos, dtype=float)[:, :2]
    if charges is None:
        charges = np.ones(len(apos2), dtype=float)
    else:
        charges = np.asarray(charges, dtype=float)

    dxy = xy[:, None, :] - apos2[None, :, :]  # (M, N, 2)
    dx = dxy[:, :, 0]
    dy = dxy[:, :, 1]

    phi_raw = np.sum(charges[None, :] * np.arctan2(dy, dx), axis=1)

    dz = dx + 1j * dy
    dz = np.where(np.abs(dz) > 1e-14, dz, 1e-14 + 0j)
    S = np.sum(charges[None, :] / dz, axis=1)  # F'/F

    grad_phi = np.column_stack((S.imag, S.real))  # ∇Φ = (Im S, Re S)
    return phi_raw, grad_phi


def wrap_phase(phi):
    """Wrap phase to (-pi, pi]."""
    return np.arctan2(np.sin(phi), np.cos(phi))


def compute_phase_grid(mol, margin=3.5, step=0.05, charges=None):
    """Compute complex phase Phi on a 2D grid for visualization."""
    apos = mol.apos
    if charges is None:
        charges = np.ones(len(apos), dtype=int)
    xmin, ymin = apos[:, 0].min() - margin, apos[:, 1].min() - margin
    xmax, ymax = apos[:, 0].max() + margin, apos[:, 1].max() + margin
    xs = np.arange(xmin, xmax + step, step)
    ys = np.arange(ymin, ymax + step, step)
    X, Y = np.meshgrid(xs, ys)
    xy = np.stack([X.ravel(), Y.ravel()], axis=1)
    phi, _ = eval_complex_phase(xy, apos, charges)
    Phi_grid = phi.reshape(len(ys), len(xs))
    extent = [xmin, xmax, ymin, ymax]
    return Phi_grid, xs, ys, extent


# ========== Newton projection + tangential relaxation ==========

def project_to_level(points, level, eval_field, n_iter=10, max_step=2.0):
    """Newton projection onto u(x) = level along normal direction.
    Uses Newton step with clamping to avoid huge jumps near critical points."""
    x = np.asarray(points, dtype=float).copy()
    for it in range(n_iter):
        u, grad = eval_field(x)
        gg = np.sum(grad * grad, axis=1)
        dx = -((u - level) / np.maximum(gg, 1e-14))[:, None] * grad
        dxnorm = np.linalg.norm(dx, axis=1)
        scale = np.minimum(1.0, max_step / np.maximum(dxnorm, 1e-14))
        x += dx * scale[:, None]
    return x


def equalize_level_points(points, level, eval_field, n_iter=500, eta=0.5, tol=1e-3):
    """Equal-arc-length redistribution on a level set via tangential spring forces.

    Instead of Laplacian smoothing (which shrinks the polygon), this computes
    the cumulative arc-length along the current polygon, then moves each point
    tangentially toward its target position at equal arc-length fractions.
    The movement is projected onto the tangent direction and the point is
    re-projected onto the level set after each step.
    """
    x = project_to_level(points, level, eval_field, n_iter=10)
    u_check, _ = eval_field(x)
    print(f"  After projection: u range=[{u_check.min():.4f}, {u_check.max():.4f}], target={level}")
    N = len(x)
    for it in range(n_iter):
        # Current cumulative arc-length
        diffs = np.roll(x, -1, axis=0) - x
        seg_lens = np.linalg.norm(diffs, axis=1)
        total_len = seg_lens.sum()
        cumlen = np.cumsum(seg_lens)  # cumlen[i] = length from point 0 to point i+1

        # Target positions at equal arc-length: s_k = k * total_len / N
        target_s = np.arange(N) * total_len / N

        # For each point, find where it should be on the current polygon
        # by interpolating along the polygon edges
        target_pos = np.zeros_like(x)
        for k in range(N):
            s_target = target_s[k]
            # Find which segment s_target falls on
            seg_idx = np.searchsorted(cumlen, s_target) % N
            # Fraction along that segment
            s_start = cumlen[seg_idx - 1] if seg_idx > 0 else 0.0
            frac = (s_target - s_start) / max(seg_lens[seg_idx], 1e-14)
            frac = np.clip(frac, 0, 1)
            p0 = x[seg_idx]
            p1 = x[(seg_idx + 1) % N]
            target_pos[k] = p0 + frac * (p1 - p0)

        # Move toward target position (tangential component only)
        _, grad = eval_field(x)
        gnorm = np.linalg.norm(grad, axis=1)
        normal = grad / np.maximum(gnorm[:, None], 1e-14)
        disp = target_pos - x
        disp_t = disp - normal * np.sum(normal * disp, axis=1, keepdims=True)
        x += eta * disp_t
        x = project_to_level(x, level, eval_field, n_iter=3)

        gaps = np.linalg.norm(np.roll(x, -1, axis=0) - x, axis=1)
        rel_spread = np.std(gaps) / max(np.mean(gaps), 1e-14)
        if it % 50 == 0 or rel_spread < tol:
            print(f"  iter {it}: gaps mean={gaps.mean():.4f}, std={gaps.std():.4f}, rel={rel_spread:.4f}, total_len={total_len:.3f}")
        if rel_spread < tol:
            break
    return x


def transport_between_shells(points, level_from, level_to, eval_field, n_relax=50, eta=0.3):
    """Transport points from one isoline to another via gradient flow, then relax."""
    u, grad = eval_field(points)
    gnorm2 = np.sum(grad * grad, axis=1)
    dc = level_to - level_from
    # x(new) ≈ x(old) + dc * ∇u / |∇u|²
    x_new = points + (dc / np.maximum(gnorm2, 1e-14))[:, None] * grad
    x_new = project_to_level(x_new, level_to, eval_field, n_iter=3)
    x_new = equalize_level_points(x_new, level_to, eval_field, n_iter=n_relax, eta=eta)
    return x_new


# ========== Phase-continuation sampling ==========

def correct_level_and_phase(x, level, target_phase, eval_field, apos, charges=None,
                            n_iter=6, max_step=0.5):
    """Newton: solve u(x)=level and Phi(x)=target_phase (mod 2π)."""
    x = np.asarray(x, dtype=float).copy()
    for _ in range(n_iter):
        u, grad_u = eval_field(x[None])
        phi, grad_phi = eval_complex_phase(x[None], apos, charges)
        residual = np.array([u[0] - level, wrap_phase(phi[0] - target_phase)])
        jac = np.vstack([grad_u[0], grad_phi[0]])
        det = np.linalg.det(jac)
        if abs(det) < 1e-10:
            break
        dx = np.linalg.solve(jac, -residual)
        dxnorm = np.linalg.norm(dx)
        if dxnorm > max_step:
            dx *= max_step / dxnorm
        x += dx
    return x


def sample_level_by_complex_phase(seed, level, n_points, eval_field, apos, charges=None):
    """Trace u=level contour using uniform unwrapped phase steps.

    Uses predictor-corrector: predict along tangent by ds = dPhi / (grad_Phi · t),
    then Newton-correct to exact (u=level, Phi=target).
    """
    apos = np.asarray(apos, dtype=float)
    if charges is None:
        charges = np.ones(len(apos), dtype=int)
    else:
        charges = np.asarray(charges, dtype=int)
    winding = int(np.sum(charges))
    if winding <= 0:
        raise ValueError("Total phase winding must be positive")

    x0 = project_to_level(np.asarray(seed)[None, :], level, eval_field, n_iter=8)[0]
    phi0, _ = eval_complex_phase(x0[None, :], apos, charges)
    phi0 = phi0[0]

    dphi = 2.0 * np.pi * winding / n_points
    points = np.empty((n_points, 2), dtype=float)
    points[0] = x0

    for k in range(1, n_points):
        x = points[k - 1]
        _, grad_u = eval_field(x[None])
        _, grad_phi = eval_complex_phase(x[None], apos, charges)

        normal = grad_u[0]
        normal /= max(np.linalg.norm(normal), 1e-14)
        tangent = np.array([-normal[1], normal[0]])
        phase_rate = np.dot(grad_phi[0], tangent)

        if phase_rate < 0.0:
            tangent = -tangent
            phase_rate = -phase_rate

        if phase_rate < 1e-8:
            predictor = x + 0.1 * tangent
        else:
            ds = dphi / phase_rate
            ds = np.clip(ds, -1.5, 1.5)
            predictor = x + ds * tangent

        target_phase = phi0 + k * dphi
        points[k] = correct_level_and_phase(predictor, level, target_phase,
                                            eval_field, apos, charges, n_iter=6)

    return points


def local_gap_relaxation(points, level, eval_field, n_iter=3, eta=0.4):
    """Purely local equal-gap correction, followed by level projection.

    Moves each point tangentially by ds = eta/2 * (gap_next - gap_prev).
    """
    x = np.asarray(points, dtype=float).copy()
    for _ in range(n_iter):
        _, grad = eval_field(x)
        normal = grad / np.maximum(np.linalg.norm(grad, axis=1)[:, None], 1e-14)
        tangent = np.column_stack((-normal[:, 1], normal[:, 0]))

        x_next = np.roll(x, -1, axis=0)
        x_prev = np.roll(x, +1, axis=0)
        gap_next = np.linalg.norm(x_next - x, axis=1)
        gap_prev = np.linalg.norm(x - x_prev, axis=1)

        ds = 0.5 * eta * (gap_next - gap_prev)
        max_ds = 0.25 * np.minimum(gap_next, gap_prev)
        ds = np.clip(ds, -max_ds, max_ds)

        x += ds[:, None] * tangent
        x = project_to_level(x, level, eval_field, n_iter=3)
    return x


def transport_by_phase(points, level_from, level_to, eval_field, apos, charges,
                       n_relax=2, eta=0.4):
    """Transport points between shells preserving phase identity.

    For each point, use gradient flow to estimate new position, then
    Newton-correct to (u=level_to, Phi=same as before).
    """
    phi_vals, _ = eval_complex_phase(points, apos, charges)
    u, grad = eval_field(points)
    gnorm2 = np.sum(grad * grad, axis=1)
    dc = level_to - level_from
    x_pred = points + (dc / np.maximum(gnorm2, 1e-14))[:, None] * grad

    x_new = np.zeros_like(points)
    for k in range(len(points)):
        x_new[k] = correct_level_and_phase(x_pred[k], level_to, phi_vals[k],
                                           eval_field, apos, charges, n_iter=6)
    if n_relax > 0:
        x_new = local_gap_relaxation(x_new, level_to, eval_field, n_iter=n_relax, eta=eta)
    return x_new


# ========== Initial point generation ==========

def initial_points_by_rays(mol, eval_field, level, n_points=16, r_start=0.3):
    """Generate initial points by walking along gradient from center at uniform angles."""
    center = mol.apos[:, :2].mean(axis=0)
    pts = []
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        x0 = center + r_start * np.array([np.cos(angle), np.sin(angle)])
        # Walk along gradient until u >= level
        x = x0.copy()
        for _ in range(1000):
            u, grad = eval_field(x[None])
            gm = np.linalg.norm(grad)
            if gm < 1e-10:
                break
            x += 0.05 * grad[0] / gm
            if u[0] >= level:
                break
        pts.append(x)
    return np.array(pts)


def initial_points_on_lemniscate(mol, level_rho, charges=None, n_points=16):
    """Generate points at equal phase on |F(z)| = rho lemniscate.

    Solves F(z) = rho * exp(i*phi) for uniform phi.
    This is NOT trivial for degree > 2; we use a root-finding approach:
    start from a ray walk to get approximate positions, then refine.
    """
    apos = mol.apos
    center = apos[:, :2].mean(axis=0)
    pts = []
    for i in range(n_points):
        phi = 2 * np.pi * i / n_points
        # Search along ray from center at angle phi
        angle = 2 * np.pi * i / n_points
        for r_test in np.arange(0.5, 30.0, 0.05):
            x = center + r_test * np.array([np.cos(angle), np.sin(angle)])
            F = eval_complex_field(x[None], apos, charges)[0]
            if np.abs(F) >= level_rho:
                pts.append(x)
                break
    return np.array(pts) if pts else np.zeros((0, 2))


# ========== Plotting ==========

def plot_molecule_overlay(ax, mol, Rvdw=None):
    for i, e in enumerate(mol.enames):
        x, y = mol.apos[i, 0], mol.apos[i, 1]
        color = ELEM_COLORS.get(e, '#FFA500')
        size = ELEM_RVDW.get(e, 1.7) * 6
        ax.plot(x, y, 'o', color=color, markersize=size, markeredgecolor='gray')
        ax.text(x, y, e, fontsize=7, ha='center', va='center',
                color='white' if e in ('C', 'O', 'N') else 'black')
        if Rvdw is not None:
            ax.add_patch(Circle((x, y), Rvdw[i], fill=False, edgecolor=color, linewidth=0.5, alpha=0.3))
    if mol.bonds is not None:
        segs = [[[mol.apos[i, 0], mol.apos[i, 1]], [mol.apos[j, 0], mol.apos[j, 1]]] for i, j in mol.bonds]
        ax.add_collection(LineCollection(segs, colors='k', linewidths=1.0, alpha=0.5))


def compute_nearest_atom_dist_grid(mol, margin=3.5, step=0.05):
    """Compute min_j |r - r_j| on a 2D grid (distance to nearest atom nucleus)."""
    apos = mol.apos
    xmin, ymin = apos[:, 0].min() - margin, apos[:, 1].min() - margin
    xmax, ymax = apos[:, 0].max() + margin, apos[:, 1].max() + margin
    xs = np.arange(xmin, xmax + step, step)
    ys = np.arange(ymin, ymax + step, step)
    X, Y = np.meshgrid(xs, ys)
    dmin = np.full_like(X, 1e30)
    for j in range(len(apos)):
        dx = X - apos[j, 0]
        dy = Y - apos[j, 1]
        d = np.sqrt(dx*dx + dy*dy)
        dmin = np.minimum(dmin, d)
    extent = [xmin, xmax, ymin, ymax]
    return dmin, xs, ys, extent


NEAREST_ATOM_LEVELS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]


def plot_r_mol_field(ax, r_mol, grad_x, grad_y, xs, ys, extent, mol, beta,
                     levels=ISOLINE_LEVELS, quiver_step=15, sample_points=None, sample_color='red',
                     vmin=None, vmax=None, nearest_dist=None, nearest_levels=None,
                     multi_shell_points=None):
    """Plot r_mol heatmap with isolines, gradient quiver, and optional sample points.

    Args:
        nearest_dist: (ny,nx) grid of min|r-rj| for background isolines
        nearest_levels: levels for nearest-atom isolines (e.g. [1.0,1.2,1.4,...])
        multi_shell_points: dict {level: points} to plot multiple shells in one axes
    """
    if vmin is None: vmin = r_mol.min()
    if vmax is None: vmax = min(r_mol.max(), 25.0)
    im = ax.imshow(r_mol, origin='lower', extent=extent, aspect='equal', cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='r_mol [Å]')
    levels_arr = np.array(levels)
    levels_plot = levels_arr[(levels_arr > r_mol.min()) & (levels_arr < r_mol.max())]
    if len(levels_plot) > 0:
        cs = ax.contour(xs, ys, r_mol, levels=levels_plot, colors='white', linewidths=0.5, alpha=0.4)
    # Nearest-atom distance isolines (dashed orange)
    if nearest_dist is not None and nearest_levels is not None:
        nl = np.array(nearest_levels)
        nl_plot = nl[(nl > nearest_dist.min()) & (nl < nearest_dist.max())]
        if len(nl_plot) > 0:
            ax.contour(xs, ys, nearest_dist, levels=nl_plot, colors='orange', linewidths=0.6,
                       linestyles='dashed', alpha=0.7)
            ax.plot([], [], '--', color='orange', alpha=0.7, label='min|r-rⱼ| isolines')
    s = quiver_step
    gx, gy = grad_x[::s, ::s], grad_y[::s, ::s]
    mag = np.sqrt(gx**2 + gy**2); mag[mag < 1e-10] = 1.0
    ax.quiver(xs[::s], ys[::s], gx/mag, gy/mag, color='red', alpha=0.3, scale=30, width=0.003)
    if sample_points is not None and len(sample_points) > 0:
        ax.plot(sample_points[:, 0], sample_points[:, 1], '.', color=sample_color, markersize=3, zorder=10)
        ax.plot(np.append(sample_points[:, 0], sample_points[0, 0]),
                np.append(sample_points[:, 1], sample_points[0, 1]), '-', color=sample_color, alpha=0.4, linewidth=0.4)
    # Multi-shell: plot all shells in same axes with different colors
    if multi_shell_points is not None:
        shell_colors = ['#ff4444', '#44ff44', '#44aaff', '#ffff44', '#ff44ff', '#44ffff']
        for i, (lvl, pts) in enumerate(sorted(multi_shell_points.items())):
            c = shell_colors[i % len(shell_colors)]
            ax.plot(pts[:, 0], pts[:, 1], '.', color=c, markersize=3, zorder=10)
            ax.plot(np.append(pts[:, 0], pts[0, 0]),
                    np.append(pts[:, 1], pts[0, 1]), '-', color=c, alpha=0.5, linewidth=0.4,
                    label=f'r_mol={lvl:.1f}')
        ax.legend(fontsize=7, loc='upper right')
    plot_molecule_overlay(ax, mol, Rvdw=get_vdw_radii(mol))
    ax.set_title(f'r_mol field (β={beta:.1f})')
    ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')


def plot_hsv_phase(ax, Phi_grid, r_mol, xs, ys, extent, mol, r_mol_levels=None, sample_points=None):
    """HSV checkerboard: Hue=wrapped Phi, Value=checkerboard in (r_mol, Phi) space."""
    H = np.mod(Phi_grid, 2*np.pi) / (2*np.pi)  # [0, 1]
    S = np.ones_like(H)

    n_phase_bands = 24
    phase_idx = np.floor(H * n_phase_bands).astype(int)

    if r_mol_levels is not None:
        radial_idx = np.digitize(r_mol, np.array(r_mol_levels))
    else:
        U_norm = (r_mol - r_mol.min()) / (r_mol.max() - r_mol.min() + 1e-30)
        radial_idx = np.floor(U_norm * 10).astype(int)

    checker = (phase_idx + radial_idx) % 2
    Val = 0.35 + 0.65 * checker

    hsv = np.stack([H, S, Val], axis=-1)
    rgb = hsv_to_rgb(hsv)
    ax.imshow(rgb, origin='lower', extent=extent, aspect='equal')

    # Overlay r_mol isolines
    if r_mol_levels is not None:
        levels_arr = np.array(r_mol_levels)
        levels_plot = levels_arr[(levels_arr > r_mol.min()) & (levels_arr < r_mol.max())]
        if len(levels_plot) > 0:
            ax.contour(xs, ys, r_mol, levels=levels_plot, colors='white', linewidths=0.4, alpha=0.5)

    if sample_points is not None and len(sample_points) > 0:
        ax.plot(sample_points[:, 0], sample_points[:, 1], '.', color='white', markersize=3, zorder=10)
        ax.plot(np.append(sample_points[:, 0], sample_points[0, 0]),
                np.append(sample_points[:, 1], sample_points[0, 1]), '-', color='white', alpha=0.5, linewidth=0.4)

    plot_molecule_overlay(ax, mol)
    ax.set_title('Checkerboard (r_mol, Φ)\nHue=arg(F), bands=phase×r_mol shells')
    ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')


def plot_hsv_smooth(ax, Phi_grid, r_mol, xs, ys, extent, mol, r_mol_levels=None, sample_points=None):
    """Smooth HSV: Hue=wrapped Phi, Value=normalized r_mol."""
    H = np.mod(Phi_grid, 2*np.pi) / (2*np.pi)
    S = np.ones_like(H)
    U_norm = (r_mol - r_mol.min()) / (r_mol.max() - r_mol.min() + 1e-30)
    Val = 0.2 + 0.8 * U_norm
    hsv = np.stack([H, S, Val], axis=-1)
    rgb = hsv_to_rgb(hsv)
    ax.imshow(rgb, origin='lower', extent=extent, aspect='equal')
    if r_mol_levels is not None:
        levels_arr = np.array(r_mol_levels)
        levels_plot = levels_arr[(levels_arr > r_mol.min()) & (levels_arr < r_mol.max())]
        if len(levels_plot) > 0:
            ax.contour(xs, ys, r_mol, levels=levels_plot, colors='white', linewidths=0.3, alpha=0.4)
    if sample_points is not None and len(sample_points) > 0:
        ax.plot(sample_points[:, 0], sample_points[:, 1], '.', color='white', markersize=3, zorder=10)
    plot_molecule_overlay(ax, mol)
    ax.set_title('Smooth HSV (r_mol, Φ)\nHue=arg(F), Value=r_mol')
    ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')


def plot_gap_histogram(ax, points, title):
    """Plot histogram of consecutive point gaps to show uniformity."""
    gaps = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    ax.bar(range(len(gaps)), gaps, color='steelblue', alpha=0.7)
    ax.axhline(gaps.mean(), color='red', linestyle='--', label=f'mean={gaps.mean():.3f}')
    ax.set_title(title)
    ax.set_xlabel('Point index'); ax.set_ylabel('Gap [Å]')
    ax.legend(fontsize=8)


# ========== Main ==========

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Angular sampling experiments')
    parser.add_argument('--target', default='data/xyz/HCOOH.xyz', help='Target molecule XYZ')
    parser.add_argument('--n-points', type=int, default=25, help='Number of angular samples')
    parser.add_argument('--margin', type=float, default=3.5, help='Grid margin [Å]')
    parser.add_argument('--step', type=float, default=0.05, help='Grid step [Å]')
    parser.add_argument('--charges', default='all', help='Phase charges: "all" (q_j=1 for all atoms) or "heavy" (skip H)')
    parser.add_argument('--save', default=None, help='Save dir (default: fig_hsamp/)')
    args = parser.parse_args()

    target_path = args.target if os.path.isabs(args.target) else os.path.join(REPO_ROOT, args.target)
    mol = AtomicSystem(fname=target_path)
    mol.neighs()
    print(f"Target: {os.path.basename(target_path)}, {len(mol.enames)} atoms")
    for i, e in enumerate(mol.enames):
        print(f"  {i}: {e:>2s}  pos={mol.apos[i]}")

    apos = mol.apos
    rvdw = get_vdw_radii(mol)
    n_pts = args.n_points
    seed = apos[:, :2].mean(axis=0) + np.array([2.0, 0.0])

    # Phase charges
    if args.charges == 'heavy':
        charges = np.array([0 if e == 'H' else 1 for e in mol.enames], dtype=int)
    else:
        charges = np.ones(len(mol.enames), dtype=int)
    winding = int(np.sum(charges))
    print(f"Phase charges: {charges}, winding Q={winding}")

    # Nearest-atom distance grid (shared background)
    nearest_dist, nxs, nys, nextent = compute_nearest_atom_dist_grid(mol, margin=args.margin, step=args.step)

    # Isoline levels (absolute r_mol values) and linestyles
    # For power formula: shift by +1 so vdW surface ≈ 1.0 (raw gives 1.0 at single-atom vdW)
    iso_levels_logsumexp = [-0.2, 0.0, 0.5]
    iso_levels_power = [0.8, 1.0, 1.5]  # = 1 + [-0.2, 0.0, 0.5]
    iso_styles = [':', '-', '--']
    iso_colors = ['#ff4444', '#ff4444', '#ff4444']

    # Configs: top row = power n=[1,2,4], bottom row = logsumexp β=[1,2,4]
    configs = [
        ('power', 1.0, 'n=1'),
        ('power', 2.0, 'n=2'),
        ('power', 4.0, 'n=4'),
        ('logsumexp', 1.0, 'β=1'),
        ('logsumexp', 2.0, 'β=2'),
        ('logsumexp', 4.0, 'β=4'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(21, 14), squeeze=False)
    fig.suptitle(f'Angular sampling comparison — {os.path.basename(target_path)} (Q={winding}, {n_pts} pts)\n'
                 f'Top: reciprocal power 1/(Σ(R_j/d_j)^n)^(1/n), Bottom: logsumexp -1/β·log(Σexp(-β·d_j))',
                 fontsize=13)

    for idx, (rmol_type, param, label) in enumerate(configs):
        row, col = idx // 3, idx % 3
        ax = axes[row][col]

        if rmol_type == 'power':
            eval_field = lambda xy: eval_rmol_power(xy, apos, rvdw, param)
            cur_iso_levels = iso_levels_power
            sample_level = 1.0
        else:
            eval_field = lambda xy: eval_rmol(xy, apos, rvdw, param)
            cur_iso_levels = iso_levels_logsumexp
            sample_level = 0.0

        r_mol, gx, gy, xs, ys, extent = compute_r_mol_grid(mol, param, margin=args.margin, step=args.step, eval_func=eval_field)
        print(f"\n[{label}] r_mol range: [{r_mol.min():.3f}, {r_mol.max():.3f}]")

        # Find vdW surface level for this formula
        center = apos[:, :2].mean(axis=0)
        rvdw_vals = []
        for j in range(len(apos)):
            direction = apos[j, :2] - center
            d = np.linalg.norm(direction)
            if d > 1e-10:
                pt = apos[j, :2] + rvdw[j] * direction / d
            else:
                pt = apos[j, :2] + np.array([rvdw[j], 0])
            u_pt, _ = eval_field(pt[None, :])
            rvdw_vals.append(u_pt[0])
        rvdw_level = np.median(rvdw_vals)
        print(f"  vdW surface level: {rvdw_level:.4f}")

        # Background: nearest-atom distance, saturated 1.0–3.0
        im = ax.imshow(nearest_dist, origin='lower', extent=nextent, aspect='equal',
                       cmap='inferno_r', vmin=1.0, vmax=3.0)

        # Isolines of r_mol
        for ilvl, (lvl, ls, lc) in enumerate(zip(cur_iso_levels, iso_styles, iso_colors)):
            if r_mol.min() < lvl < r_mol.max():
                ax.contour(xs, ys, r_mol, levels=[lvl], colors=lc, linewidths=1.2,
                           linestyles=ls, alpha=0.9)

        # Sample points on the middle isoline (r_mol=0.0 for logsumexp, r_mol=1.0 for power)
        if r_mol.min() < sample_level < r_mol.max():
            pts_phase = sample_level_by_complex_phase(seed, sample_level, n_pts, eval_field, apos, charges)
            pts_phase = local_gap_relaxation(pts_phase, sample_level, eval_field, n_iter=5, eta=0.4)
            gaps = np.linalg.norm(np.roll(pts_phase, -1, axis=0) - pts_phase, axis=1)
            rel = gaps.std() / gaps.mean()
            print(f"  Phase+relax at r_mol={sample_level:.4f}: gaps mean={gaps.mean():.3f}, rel={rel:.4f}")
            ax.plot(pts_phase[:, 0], pts_phase[:, 1], '.', color='lime', markersize=3, zorder=10)
            ax.plot(np.append(pts_phase[:, 0], pts_phase[0, 0]),
                    np.append(pts_phase[:, 1], pts_phase[0, 1]), '-', color='lime', alpha=0.5, linewidth=0.4)
            title_rel = f'rel={rel:.3f}'
        else:
            print(f"  Sample level {sample_level:.4f} out of range, skipping sampling")
            title_rel = 'N/A'

        plot_molecule_overlay(ax, mol, Rvdw=rvdw)

        # Legend for isolines
        from matplotlib.lines import Line2D
        legend_lines = []
        for lvl, ls, lc in zip(cur_iso_levels, iso_styles, iso_colors):
            legend_lines.append(Line2D([0], [0], color=lc, linestyle=ls, linewidth=1.2,
                                       label=f'r_mol={lvl}'))
        legend_lines.append(Line2D([0], [0], color='lime', marker='.', markersize=5,
                                   linestyle='-', linewidth=0.4, label=f'samples ({title_rel})'))
        ax.legend(handles=legend_lines, fontsize=7, loc='upper right')

        ax.set_title(f'{label}  (vdW level={rvdw_level:.3f})')
        ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')

    fig.tight_layout(rect=[0, 0, 0.92, 0.95])

    # Shared colorbar for nearest-atom distance
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, aspect=30, pad=0.02)
    cbar.set_label('min$_j$ |x - r$_j$|  [Å]', fontsize=11)

    # --- Save comparison figure ---
    save_dir = args.save if args.save else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_hsamp')
    os.makedirs(save_dir, exist_ok=True)
    basename = os.path.basename(target_path).replace('.xyz', '')
    fname = os.path.join(save_dir, f'rmol_comparison_{basename}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fname}")
    plt.close(fig)

    # --- Figure 2: Full [n_radial x n_phi] sampling grid with gradient arrows ---
    # Use logsumexp beta=4
    beta_grid = 4.0
    eval_grid = lambda xy: eval_rmol(xy, apos, rvdw, beta_grid)
    # Use larger margin so we can reach high r_mol levels (up to 20.0)
    grid_margin = max(args.margin, 25.0)
    r_mol_g, gx_g, gy_g, xs_g, ys_g, extent_g = compute_r_mol_grid(
        mol, beta_grid, margin=grid_margin, step=args.step, eval_func=eval_grid)

    ISOLINE_LEVELS_GRID = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 15.0, 20.0]
    n_phi = n_pts
    # Use isoline levels that are in range as radial shells — arrows sit exactly on isolines
    radial_levels = [l for l in ISOLINE_LEVELS_GRID if r_mol_g.min() < l < r_mol_g.max()]
    n_radial = len(radial_levels)

    print(f"\n--- Full sampling grid (β={beta_grid}, {n_radial} shells × {n_phi} angles) ---")
    print(f"  Radial levels: {radial_levels}")

    # Start from a middle shell where sampling uniformity is best
    mid_idx = radial_levels.index(0.0) if 0.0 in radial_levels else n_radial // 2
    mid_level = radial_levels[mid_idx]
    pts_mid = sample_level_by_complex_phase(seed, mid_level, n_phi, eval_grid, apos, charges)
    pts_mid = local_gap_relaxation(pts_mid, mid_level, eval_grid, n_iter=5, eta=0.4)

    all_pts = np.zeros((n_radial, n_phi, 2))
    all_grad = np.zeros((n_radial, n_phi, 2))
    all_pts[mid_idx] = pts_mid

    # Transport outward from middle
    for ir in range(mid_idx + 1, n_radial):
        all_pts[ir] = transport_by_phase(all_pts[ir - 1], radial_levels[ir - 1], radial_levels[ir],
                                         eval_grid, apos, charges, n_relax=5, eta=0.4)

    # Transport inward from middle
    for ir in range(mid_idx - 1, -1, -1):
        all_pts[ir] = transport_by_phase(all_pts[ir + 1], radial_levels[ir + 1], radial_levels[ir],
                                         eval_grid, apos, charges, n_relax=5, eta=0.4)

    # Evaluate gradient at all sample points
    for ir in range(n_radial):
        u, g = eval_grid(all_pts[ir])
        all_grad[ir] = g
        gaps = np.linalg.norm(np.roll(all_pts[ir], -1, axis=0) - all_pts[ir], axis=1)
        rel = gaps.std() / gaps.mean()
        print(f"  Shell r_mol={radial_levels[ir]:.2f}: gaps mean={gaps.mean():.3f}, rel={rel:.4f}")

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 9))
    fig2.suptitle(f'Sampling grid + radial directions (β={beta_grid}, {n_radial}×{n_phi}) — {os.path.basename(target_path)}',
                  fontsize=13)

    # Background: nearest-atom distance (recompute with larger margin to match grid)
    nearest_dist_g, _, _, nextent_g = compute_nearest_atom_dist_grid(mol, margin=grid_margin, step=args.step)
    ax2.imshow(nearest_dist_g, origin='lower', extent=nextent_g, aspect='equal',
               cmap='inferno_r', vmin=1.0, vmax=3.0)

    # Isolines — same levels as radial shells
    levels_in_range = [l for l in ISOLINE_LEVELS_GRID if r_mol_g.min() < l < r_mol_g.max()]
    if levels_in_range:
        ax2.contour(xs_g, ys_g, r_mol_g, levels=levels_in_range, colors='#ff4444',
                    linewidths=0.6, alpha=0.5)

    # Plot sample points and gradient arrows
    radial_colors = plt.cm.coolwarm(np.linspace(0, 1, n_radial))
    for ir in range(n_radial):
        pts = all_pts[ir]
        grad = all_grad[ir]
        gnorm = np.linalg.norm(grad, axis=1)
        gnorm_safe = np.maximum(gnorm, 1e-14)
        # Normalize gradient to unit length for visualization
        grad_norm = grad / gnorm_safe[:, None]
        # Arrow length proportional to local spacing
        gaps = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        arrow_len = 0.3 * np.median(gaps)
        u_arr = grad_norm[:, 0] * arrow_len
        v_arr = grad_norm[:, 1] * arrow_len

        c = radial_colors[ir]
        ax2.plot(np.append(pts[:, 0], pts[0, 0]),
                 np.append(pts[:, 1], pts[0, 1]), '.-', color=c, markersize=4, linewidth=0.8, zorder=10)
        ax2.quiver(pts[:, 0], pts[:, 1], u_arr, v_arr,
                   color=c, scale=1.0, scale_units='xy', angles='xy',
                   width=0.003, headwidth=3, headlength=4, zorder=11)

    plot_molecule_overlay(ax2, mol, Rvdw=rvdw)

    # Legend
    from matplotlib.lines import Line2D
    legend_lines = [Line2D([0], [0], color='#ff4444', linestyle='-', linewidth=0.6, alpha=0.5, label='r_mol isolines')]
    for ir in range(n_radial):
        legend_lines.append(Line2D([0], [0], color=radial_colors[ir], marker='.', markersize=5,
                                   linestyle='-', linewidth=0.5,
                                   label=f'r_mol={radial_levels[ir]:.2f}'))
    ax2.legend(handles=legend_lines, fontsize=7, loc='upper right')
    ax2.set_xlabel('x [Å]'); ax2.set_ylabel('y [Å]')

    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fname2 = os.path.join(save_dir, f'sampling_grid_{basename}_b4.png')
    fig2.savefig(fname2, dpi=150, bbox_inches='tight')
    print(f"Saved: {fname2}")
    plt.close(fig2)


if __name__ == '__main__':
    main()
