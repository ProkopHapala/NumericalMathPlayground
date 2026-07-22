#!/usr/bin/python
r"""
hbond_sampling.py — Distance-field sampling for hydrogen-bond DFT calculations
=================================================================================

Goal
----
When computing hydrogen-bond interaction energies by DFT, we need to place a
small "probe" molecule (e.g. HF, HCN) at many positions and orientations
around a larger "target" molecule.  This script defines a smooth scalar
*distance field* r_mol around the target that serves as a natural coordinate
for sampling: isolines of r_mol give shells at constant "distance from the
molecular surface", and the gradient ∇r_mol gives the outward normal direction
along which we orient the probe.

The distance field: soft-min / log-sum-exp aggregate
-----------------------------------------------------
For a target molecule with atoms at positions r_j and vdW radii R_j, we define:

    r_mol(r) = -1/β · log( Σ_j exp( -β·|r - r_j| - R_j ) )

This is a *soft-minimum* (via the log-sum-exp trick) of the per-atom distances
|r - r_j| shifted by the vdW radius R_j.  The parameter β controls the
sharpness of the minimum:

  - **β → ∞**: r_mol → min_j ( |r - r_j| + R_j/β ) → min_j |r - r_j|,
    i.e. the exact Euclidean distance to the nearest atom (hard min).
  - **β → 0**:  r_mol → average of all (|r - r_j| + R_j/β), a very smooth,
    nearly spherical field that loses molecular shape information.
  - **β ~ 1–4**: practical range.  β=1 gives a smooth, rounded field;
    β=4 preserves more molecular detail but can develop kinks near concave
    regions.

The -1/β prefactor rescales the log-sum-exp so that r_mol has units of Å
and grows linearly far from the molecule (asymptotically r_mol ≈ |r - r_center|
- R_eff).  Near the vdW surface, r_mol ≈ 0.  The vdW radii R_j act as per-atom
offsets: larger atoms "push out" the field, so the zero-level isoline
approximates the molecular vdW surface.

**Numerical stability:** The log-sum-exp is computed by subtracting the max
exponent before exponentiating (standard trick), avoiding overflow for large β.

Gradient and probe orientation
-------------------------------
The gradient ∇r_mol is computed analytically from the same weights:

    ∇r_mol = Σ_j w_j · (r - r_j)/|r - r_j|  /  Σ_j w_j,    w_j = exp(-β|r-r_j| - R_j)

This gives a unit-like vector pointing away from the molecule.  We place the
probe molecule with a chosen *pivot atom* at the sampling point, oriented so
that the probe's molecular axis aligns with ∇r_mol (pointing outward).

Two orientations are supported:
  - **H-donor**: H atom is the pivot (closest to target), the electronegative
    end (F, N, ...) points away.  This models the probe *donating* an H-bond.
  - **H-acceptor**: electronegative end is the pivot (closest to target),
    H points away.  This models the probe *accepting* an H-bond.

Isoline levels and sampling density
-------------------------------------
Sampling is done at discrete isolines r_mol = const.  The default levels are:

    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 15.0, 20.0]

These are dense near the surface (Δ=0.1) where H-bond interactions are strong
and rapidly varying, and sparse far away (Δ=5.0) where the energy landscape is
flat.  The isoline-spacing plot visualizes this on a linear scale so you can
judge whether the sampling resolution is adequate.

Open issues and caveats
-----------------------
1. **Angular sampling along isolines (unsolved).**  The current implementation
   places probes at uniform angular intervals around the molecular center and
   walks each along ∇r_mol to the target isoline.  This is crude: it clusters
   samples near protruding atoms and under-samples concave regions.  A better
   approach would distribute samples at *equal arc-length* along each isoline
   contour.  This requires computing the arc-length parameterization of the
   level set {r : r_mol(r) = c}.

   - *Numerical approach:* Extract contour vertices from matplotlib's
     `ContourSet`, compute cumulative arc-length, and place samples at
     s_k = k·L/N.  This is straightforward but purely numerical.
   - *Analytical approach (open question):* For a sum of exponentials of
     Euclidean distances, the level sets are not conics and have no simple
     closed form.  However, if the field could be expressed via complex
     analysis — e.g. as a potential of point charges in 2D (logarithmic
     potential) rather than exponential — then level sets become |Σ z_j| = const
     and arc-length integrals might admit closed forms via residues or the
     gamma function.  This remains speculative; the exponential kernel does not
     obviously map to a holomorphic function.  Mentioned as a research
     direction, not attempted here.
   - *Phase parameter:* Ideally we want a single scalar "phase" φ ∈ [0, 2π)
     (or [0, L) in arc-length) that uniquely labels positions along an isoline,
     analogous to the angle θ in polar coordinates.  This would give a full
     2D coordinate system (r_mol, φ) → (x, y) for sampling.  Defining φ
     robustly for non-convex isolines (which can self-intersect in principle)
     is an open problem.

2. **β selection.**  There is no principled way to choose β a priori.  It
   should be large enough to preserve molecular shape but small enough to keep
   the field smooth (no kinks).  Inspect the multi-β plot and pick by eye.

3. **2D only.**  The current implementation samples in the molecular plane
   (z=0).  For out-of-plane H-bonds (e.g. above/below aromatic rings), a 3D
   version would be needed.  The math extends straightforwardly but
   visualization and angular sampling become harder.

4. **Probe rigidity.**  The probe is treated as a rigid body.  Relaxing the
   probe geometry (e.g. allowing the H–F bond to stretch) would require an
   additional internal coordinate in the sampling.

5. **Single pivot.**  Only one pivot atom is used per orientation.  For
   polyatomic probes (e.g. H₂O), multiple pivot atoms and orientations would
   be needed.

Usage
-----
::

    # Default: CH2O target, HF probe, β=[0.5,1,2,4], saves to fig_hsamp/
    python hbond_sampling.py

    # Custom target and probe
    python hbond_sampling.py --target data/xyz/azaindol.xyz --probe HCN --betas 0.5 1.0 2.0 4.0

    # Adjust sampling isoline and angular resolution
    python hbond_sampling.py --r-sample 1.5 --n-angles 16

    # Save to custom directory
    python hbond_sampling.py --save /tmp/my_plots

Output
------
Three PNG files are saved to ``fig_hsamp/`` (or ``--save`` dir):
  - ``rmol_field_{target}_{probe}.png``   — r_mol heatmap for each β with isolines + gradient quiver
  - ``probe_placement_{target}_{probe}.png`` — H-donor and H-acceptor probes placed at r_mol = r_sample
  - ``isoline_spacing_{target}_{probe}.png``  — isoline levels on linear scale
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from py.AtomicSystem import AtomicSystem
from py.elements import ELEMENTS, ELEMENT_DICT

# vdW radii lookup
ELEM_RVDW = {row[1]: row[7] for row in ELEMENTS}
ELEM_COLORS = {row[1]: row[8] for row in ELEMENTS}

ISOLINE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 15.0, 20.0]


def load_molecule(xyz_path):
    """Load molecule from XYZ file, return AtomicSystem."""
    mol = AtomicSystem(fname=xyz_path)
    mol.neighs()
    return mol


def get_vdw_radii(mol):
    """Get vdW radii array for molecule atoms."""
    return np.array([ELEM_RVDW.get(e, 1.7) for e in mol.enames])


def compute_r_mol_grid(mol, beta, margin=8.0, step=0.05, z_height=0.0):
    """Compute r_mol distance field on a 2D grid.

    r_mol(r) = -1/beta * log( sum_j exp(-beta*|r-r_j| - R_j) )

    Uses log-sum-exp trick for numerical stability.

    Returns:
        r_mol (ny, nx), grad_x (ny, nx), grad_y (ny, nx), xs, ys, extent
    """
    apos = mol.apos
    Rvdw = get_vdw_radii(mol)

    xmin, ymin = apos[:, 0].min() - margin, apos[:, 1].min() - margin
    xmax, ymax = apos[:, 0].max() + margin, apos[:, 1].max() + margin
    xs = np.arange(xmin, xmax + step, step)
    ys = np.arange(ymin, ymax + step, step)
    X, Y = np.meshgrid(xs, ys)  # (ny, nx)

    # Distances: (ny, nx, N)
    dx = X[:, :, None] - apos[None, None, :, 0]
    dy = Y[:, :, None] - apos[None, None, :, 1]
    dz = z_height - apos[None, None, :, 2]
    r = np.sqrt(dx*dx + dy*dy + dz*dz)  # (ny, nx, N)

    # Exponents: -beta*(|r-r_j| - R_j)  — corrected: d_j = r_j - R_j
    exponents = -beta * (r - Rvdw[None, None, :])  # (ny, nx, N)

    # Log-sum-exp trick: subtract max for stability
    max_exp = np.max(exponents, axis=2, keepdims=True)
    log_sum = max_exp[:, :, 0] + np.log(np.sum(np.exp(exponents - max_exp), axis=2))

    r_mol = -log_sum / beta  # (ny, nx)

    # Gradient of r_mol: dr_mol/dx = -1/beta * (sum_j w_j * (-beta * dx_j/r_j)) / sum_j exp(...)
    # where w_j = exp(exponent_j)
    # = sum_j (w_j * dx_j / r_j) / sum_j w_j
    weights = np.exp(exponents - max_exp)  # (ny, nx, N), stable
    w_sum = weights.sum(axis=2)  # (ny, nx)
    w_sum_safe = np.where(w_sum > 1e-30, w_sum, 1e-30)
    ir = 1.0 / np.where(r > 1e-10, r, 1e-10)
    grad_x = (weights * dx * ir).sum(axis=2) / w_sum_safe
    grad_y = (weights * dy * ir).sum(axis=2) / w_sum_safe

    extent = [xmin, xmax, ymin, ymax]
    return r_mol, grad_x, grad_y, xs, ys, extent


def load_probe_molecule(probe_name):
    """Load a linear probe molecule from XYZ. Returns mol, pivot indices."""
    xyz_path = os.path.join(REPO_ROOT, 'data', 'xyz', f'{probe_name}.xyz')
    mol = load_molecule(xyz_path)
    # H-donor pivot: first H atom (index 0 for HF, HCN)
    # H-acceptor pivot: last atom (F for HF, N for HCN)
    h_donor_pivot = 0  # H is always first in our XYZ files
    h_acceptor_pivot = len(mol.enames) - 1  # last atom (F, N, etc.)
    return mol, h_donor_pivot, h_acceptor_pivot


def get_probe_direction(mol, pivot_idx):
    """Get unit vector from pivot atom to the far end of the molecule."""
    apos = mol.apos
    pivot = apos[pivot_idx]
    far = apos[-1] if pivot_idx == 0 else apos[0]
    direction = far - pivot
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0])
    return direction / norm


def place_probe_along_gradient(pivot_pos, grad_dir, probe_mol, pivot_idx):
    """Place probe molecule with pivot at pivot_pos, oriented along grad_dir.

    grad_dir: 2D direction (away from target molecule)
    The probe extends from pivot in the +grad_dir direction (away from target).
    Returns positions (N, 3) for the placed probe.
    """
    apos = probe_mol.apos.copy()
    pivot = apos[pivot_idx].copy()

    # Direction from pivot to far end in the probe's local frame
    far_idx = len(apos) - 1 if pivot_idx == 0 else 0
    local_dir = apos[far_idx] - pivot
    local_norm = np.linalg.norm(local_dir)
    if local_norm < 1e-10:
        return apos - pivot + pivot_pos

    local_dir_unit = local_dir / local_norm

    # Target direction in 2D (embed in xy plane)
    target_dir = np.array([grad_dir[0], grad_dir[1], 0.0])
    target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-10)

    # Rotation: align local_dir_unit -> target_dir
    # Compute rotation axis and angle
    rot_axis = np.cross(local_dir_unit, target_dir)
    rot_axis_norm = np.linalg.norm(rot_axis)
    cos_angle = np.dot(local_dir_unit, target_dir)
    sin_angle = rot_axis_norm  # default for non-degenerate case

    if rot_axis_norm < 1e-10:
        # Parallel or anti-parallel
        if cos_angle < 0:
            # 180 degree rotation around any perpendicular axis
            perp = np.array([0, 0, 1.0])
            if abs(np.dot(local_dir_unit, perp)) > 0.99:
                perp = np.array([1.0, 0, 0])
            rot_axis = perp
            rot_axis_norm = 1.0
            cos_angle = -1.0
            sin_angle = 0.0  # 180° rotation: sin(π)=0
        else:
            # No rotation needed
            return apos - pivot + pivot_pos

    rot_axis = rot_axis / rot_axis_norm

    # Rodrigues rotation
    k = rot_axis
    rotated = np.zeros_like(apos)
    for i in range(len(apos)):
        v = apos[i] - pivot
        rotated[i] = pivot + v * cos_angle + np.cross(k, v) * sin_angle + k * np.dot(k, v) * (1 - cos_angle)

    # Translate to target position
    placed = rotated - pivot + pivot_pos
    return placed


def plot_molecule_overlay(ax, mol, Rvdw=None):
    """Draw atoms and bonds on ax."""
    for i, e in enumerate(mol.enames):
        x, y = mol.apos[i, 0], mol.apos[i, 1]
        color = ELEM_COLORS.get(e, '#FFA500')
        size = ELEM_RVDW.get(e, 1.7) * 6
        ax.plot(x, y, 'o', color=color, markersize=size, markeredgecolor='gray')
        ax.text(x, y, e, fontsize=7, ha='center', va='center', color='white' if e in ('C', 'O', 'N') else 'black')
        if Rvdw is not None:
            ax.add_patch(Circle((x, y), Rvdw[i], fill=False, edgecolor=color, linewidth=0.5, alpha=0.3))
    if mol.bonds is not None:
        segs = [[[mol.apos[i, 0], mol.apos[i, 1]], [mol.apos[j, 0], mol.apos[j, 1]]]
                for i, j in mol.bonds]
        ax.add_collection(LineCollection(segs, colors='k', linewidths=1.0, alpha=0.5))


def plot_r_mol_field(ax, r_mol, grad_x, grad_y, xs, ys, extent, mol, beta,
                     levels=ISOLINE_LEVELS, quiver_step=12, show_quiver=True):
    """Plot r_mol heatmap with isolines and optional gradient quiver."""
    vmax = max(r_mol.max(), max(levels) if levels else 10.0)
    im = ax.imshow(r_mol, origin='lower', extent=extent, aspect='equal', cmap='viridis',
                   vmin=0, vmax=min(vmax, 25.0))
    plt.colorbar(im, ax=ax, label='r_mol [Å]')

    # Isolines
    levels_arr = np.array(levels)
    levels_plot = levels_arr[levels_arr < r_mol.max()]
    if len(levels_plot) > 0:
        cs = ax.contour(xs, ys, r_mol, levels=levels_plot, colors='white', linewidths=0.7, alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=5, fmt='%.2f')

    # Quiver (normalized gradient)
    if show_quiver:
        s = quiver_step
        gx = grad_x[::s, ::s]
        gy = grad_y[::s, ::s]
        mag = np.sqrt(gx**2 + gy**2)
        mag[mag < 1e-10] = 1.0
        ax.quiver(xs[::s], ys[::s], gx/mag, gy/mag, color='red', alpha=0.4, scale=30, width=0.003)

    plot_molecule_overlay(ax, mol, Rvdw=get_vdw_radii(mol))
    ax.set_title(f'r_mol field (β={beta:.1f})')
    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')


def plot_probes_on_field(ax, r_mol, grad_x, grad_y, xs, ys, extent, mol, beta,
                          probe_mol, pivot_idx, orientation, n_angles=8, r_sample=2.0):
    """Plot the r_mol field with placed probe molecules at sampling points."""
    vmax = min(r_mol.max(), 25.0)
    im = ax.imshow(r_mol, origin='lower', extent=extent, aspect='equal', cmap='viridis',
                   vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label='r_mol [Å]')

    # Isolines
    levels_arr = np.array(ISOLINE_LEVELS)
    levels_plot = levels_arr[levels_arr < r_mol.max()]
    if len(levels_plot) > 0:
        cs = ax.contour(xs, ys, r_mol, levels=levels_plot, colors='white', linewidths=0.5, alpha=0.4)

    # Find sampling points: walk along gradient from molecule center to r_mol = r_sample
    # Use multiple starting angles from molecule center
    center = mol.apos[:, :2].mean(axis=0)

    # Grid interpolation helper
    def interp_grad(px, py):
        ix = np.argmin(np.abs(xs - px))
        iy = np.argmin(np.abs(ys - py))
        return grad_x[iy, ix], grad_y[iy, ix], r_mol[iy, ix]

    # Sample at isoline r_mol = r_sample by walking from center outward
    probe_positions = []
    for i_angle in range(n_angles):
        angle = 2 * np.pi * i_angle / n_angles
        # Start from center + small offset in this angle direction
        px = center[0] + 0.5 * np.cos(angle)
        py = center[1] + 0.5 * np.sin(angle)

        # Walk along gradient until r_mol >= r_sample
        for _ in range(500):
            gx, gy, rm = interp_grad(px, py)
            gm = np.sqrt(gx**2 + gy**2)
            if gm < 1e-10:
                break
            step = 0.05
            px += step * gx / gm
            py += step * gy / gm
            if rm >= r_sample:
                break

        gx, gy, rm = interp_grad(px, py)
        if rm >= r_sample * 0.9:  # close enough
            probe_positions.append((px, py, np.array([gx, gy]) / (np.sqrt(gx**2 + gy**2) + 1e-10)))

    # Place and draw probes
    pivot_name = probe_mol.enames[pivot_idx]
    for px, py, gdir in probe_positions:
        placed = place_probe_along_gradient(np.array([px, py, 0.0]), gdir, probe_mol, pivot_idx)
        # Draw probe atoms
        for j, e in enumerate(probe_mol.enames):
            color = ELEM_COLORS.get(e, '#FFA500')
            size = ELEM_RVDW.get(e, 1.7) * 5
            ax.plot(placed[j, 0], placed[j, 1], 'o', color=color, markersize=size,
                    markeredgecolor='black', alpha=0.8)
        # Draw probe bonds
        if probe_mol.bonds is not None:
            segs = [[[placed[i, 0], placed[i, 1]], [placed[j, 0], placed[j, 1]]]
                    for i, j in probe_mol.bonds]
            ax.add_collection(LineCollection(segs, colors='gray', linewidths=1.5, alpha=0.6))

    plot_molecule_overlay(ax, mol, Rvdw=get_vdw_radii(mol))
    probe_name = f"{probe_mol.enames[0]}{probe_mol.enames[-1]}" if len(probe_mol.enames) == 2 else \
                 ''.join(probe_mol.enames)
    ax.set_title(f'{probe_name} {orientation} (pivot={pivot_name}, r_mol={r_sample:.1f}, β={beta:.1f})')
    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')


def plot_isoline_spacing(ax, levels=ISOLINE_LEVELS):
    """Plot isoline values on a linear scale to visualize sampling density."""
    levels_arr = np.array(levels)
    # Vertical lines at each level
    for lv in levels_arr:
        ax.axvline(lv, color='steelblue', linewidth=2, alpha=0.7)
    # Mark spacing
    for i in range(len(levels_arr)):
        ax.plot(levels_arr[i], 0.5, '|', color='red', markersize=15, markeredgewidth=2)
    # Connect with arrows showing spacing
    for i in range(len(levels_arr) - 1):
        mid = (levels_arr[i] + levels_arr[i+1]) / 2
        spacing = levels_arr[i+1] - levels_arr[i]
        ax.annotate('', xy=(levels_arr[i+1], 0.3), xytext=(levels_arr[i], 0.3),
                    arrowprops=dict(arrowstyle='<->', color='gray', lw=0.8))
        ax.text(mid, 0.35, f'Δ={spacing:.2f}', ha='center', va='bottom', fontsize=6, color='gray')

    ax.set_xlim(-0.5, levels_arr[-1] + 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('r_mol [Å]')
    ax.set_yticks([])
    ax.set_title('Isoline spacing on linear scale')
    ax.set_xticks(levels_arr)
    ax.tick_params(axis='x', rotation=45, labelsize=7)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='H-bond sampling via distance field')
    parser.add_argument('--target', default='data/xyz/CH2O.xyz', help='Target molecule XYZ path')
    parser.add_argument('--probe', default='HF', choices=['HF', 'HCN', 'HCl', 'CO', 'HCCH'],help='Linear probe molecule')
    parser.add_argument('--betas', nargs='+', type=float, default=[0.5, 1.0, 2.0, 4.0], help='Beta values to plot')
    parser.add_argument('--beta-main', type=float, default=2.0, help='Beta for probe placement plots')
    parser.add_argument('--r-sample', type=float, default=2.0, help='r_mol isoline for probe placement')
    parser.add_argument('--n-angles', type=int, default=8, help='Number of angular samples')
    parser.add_argument('--margin', type=float, default=8.0, help='Grid margin [Å]')
    parser.add_argument('--step', type=float, default=0.05, help='Grid step [Å]')
    parser.add_argument('--save', default=None, help='Save figure to PNG path (default: fig_hsamp/ in script dir)')
    args = parser.parse_args()

    target_path = args.target if os.path.isabs(args.target) else os.path.join(REPO_ROOT, args.target)
    mol = load_molecule(target_path)
    print(f"Target: {os.path.basename(target_path)}, {len(mol.enames)} atoms")
    for i, e in enumerate(mol.enames):
        print(f"  {i}: {e:>2s}  pos={mol.apos[i]}")

    probe_mol, h_donor_pivot, h_acceptor_pivot = load_probe_molecule(args.probe)
    print(f"\nProbe: {args.probe}, {len(probe_mol.enames)} atoms")
    print(f"  H-donor pivot: atom {h_donor_pivot} ({probe_mol.enames[h_donor_pivot]})")
    print(f"  H-acceptor pivot: atom {h_acceptor_pivot} ({probe_mol.enames[h_acceptor_pivot]})")

    # --- Figure 1: r_mol field for different beta values ---
    n_betas = len(args.betas)
    fig1, axes1 = plt.subplots(1, n_betas, figsize=(6*n_betas, 5), squeeze=False)
    fig1.suptitle(f'r_mol distance field — target: {os.path.basename(target_path)}', fontsize=12)

    fields = {}
    for ib, beta in enumerate(args.betas):
        r_mol, gx, gy, xs, ys, extent = compute_r_mol_grid(mol, beta, margin=args.margin, step=args.step)
        fields[beta] = (r_mol, gx, gy, xs, ys, extent)
        print(f"  β={beta:.1f}: r_mol range=[{r_mol.min():.3f}, {r_mol.max():.3f}]")
        plot_r_mol_field(axes1[0, ib], r_mol, gx, gy, xs, ys, extent, mol, beta,
                         levels=ISOLINE_LEVELS, quiver_step=15)
    fig1.tight_layout()

    # --- Figure 2: Probe placement ---
    beta_main = args.beta_main
    r_mol, gx, gy, xs, ys, extent = fields.get(beta_main,
        compute_r_mol_grid(mol, beta_main, margin=args.margin, step=args.step))

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6), squeeze=False)
    fig2.suptitle(f'Probe placement — {args.probe} around {os.path.basename(target_path)} (β={beta_main}, r_mol={args.r_sample})', fontsize=12)

    # H-donor orientation
    plot_probes_on_field(axes2[0, 0], r_mol, gx, gy, xs, ys, extent, mol, beta_main,
                         probe_mol, h_donor_pivot, 'H-donor',
                         n_angles=args.n_angles, r_sample=args.r_sample)

    # H-acceptor orientation
    plot_probes_on_field(axes2[0, 1], r_mol, gx, gy, xs, ys, extent, mol, beta_main,
                         probe_mol, h_acceptor_pivot, 'H-acceptor',
                         n_angles=args.n_angles, r_sample=args.r_sample)
    fig2.tight_layout()

    # --- Figure 3: Isoline spacing ---
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 3))
    plot_isoline_spacing(ax3, levels=ISOLINE_LEVELS)
    fig3.tight_layout()

    save_dir = args.save if args.save else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_hsamp')
    if os.path.isfile(save_dir):
        save_dir = os.path.dirname(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    for fig, name in [(fig1, 'rmol_field'), (fig2, 'probe_placement'), (fig3, 'isoline_spacing')]:
        basename = os.path.basename(target_path).replace('.xyz', '')
        fname = os.path.join(save_dir, f'{name}_{basename}_{args.probe}.png')
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
        plt.close(fig)


if __name__ == '__main__':
    main()
