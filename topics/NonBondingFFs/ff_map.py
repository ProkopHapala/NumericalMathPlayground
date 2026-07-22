#!/usr/bin/python
"""
ff_map.py — Forcefield interaction maps around flat molecules
=============================================================

What this script does
---------------------
Visualizes non-bonded interaction energy between a molecule (including electron
pairs as dummy atoms) and a probe atom scanned over a 2D grid in the molecular
plane (z=0).  The result is a color map (imshow) showing where the probe atom
feels attraction (blue) or repulsion (red).

Key concepts
------------
- **Electron pairs (E atoms):** Lone pairs on O/N are added as dummy atoms by
  ``AtomicSystem.add_electron_pairs()``.  Each epair carries a charge (e.g. -0.2e)
  that is subtracted from its host atom to conserve total charge.

- **Morse potential:** ``E(r) = E0 * [exp(-2β(r-re)) - 2*exp(-β(r-re))]``
  with β = 1.7 Å⁻¹.  The repulsive part (Pauli) is ``E0*exp(-2β(r-re))``,
  the attractive part is ``-2*E0*exp(-β(r-re))``.

- **REQ parameters:** Each atom has [RvdW, sqrt(EvdW), Q, Hb].  Pair parameters
  use arithmetic mixing: R = Ri+Rj, E = sqrtEi*sqrtEj, Q = Qi*Qj.

- **Charge sources:** 4th column of XYZ file (default) or QEq charge
  equilibration (``--charge-mode qeq``).

- **Force arrows (quiver):** Overlay normalized direction arrows for either
  the Coulomb electric field (``--quiver efield``) or the Pauli repulsion
  force from the Morse repulsive part (``--quiver rep``).

- **vdW circles:** Each atom is drawn with two circles — its own vdW radius
  (solid) and the contact distance Ri+Rj with the probe (dashed).

Grid: bounding box + margin around the molecule, sampled at fixed step.
Color scale is symmetric: vmin = -vmax = |Emin|.

Usage examples
--------------
::

    # Default: Morse, Morse+Coulomb, Morse+CoulDipole; O(-0.4e) and H(+0.4e) probes
    python ff_map.py

    # All three components side by side
    python ff_map.py --ff-types morse coulomb morseQ

    # With Pauli repulsion force arrows
    python ff_map.py --quiver rep

    # With electric field arrows, QEq charges, different molecule
    python ff_map.py --quiver efield --charge-mode qeq --xyz data/xyz/NTCDA.xyz

    # Custom probes and finer grid
    python ff_map.py --probes O:-0.4 N:-0.3 H:+0.4 --step 0.05 --margin 6

CLI options
-----------
  --ff-types      Column components: morse, morseQ, coulomb (default: morse morseQ)
  --probes        Row probes as NAME:CHARGE (default: O:-0.4 H:+0.4)
  --xyz           XYZ file path (default: data/xyz/CH2O.xyz)
  --charge-mode   xyz (4th column) or qeq (charge equilibration)
  --epair-charge  Charge per electron pair [e], subtracted from host (default: -0.2)
  --margin        Grid margin around molecule [Å] (default: 4.0)
  --step          Grid spacing [Å] (default: 0.1)
  --z-height      Probe plane z [Å] (default: 0 = molecular plane)
  --quiver        efield | rep | none (default: none)
  --quiver-step   Arrow subsampling step in grid points (default: 8)
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from py.AtomicSystem import AtomicSystem
from py.FFs.FFparams import read_element_types, read_atom_types, make_REQs_from_enames
from py.elements import ELEMENTS, ELEMENT_DICT
from py.FFs.QEq import solve_from_elements as qeq_solve

COULOMB_CONST = 14.3996448915  # eV·Å/e²
MORSE_BETA = 1.7  # [1/Å] Morse decay parameter

# Build element color/size lookup from elements.py
ELEM_COLORS = {}
ELEM_SIZES = {}
for row in ELEMENTS:
    name, color, rvdw = row[1], row[8], row[7]
    ELEM_COLORS[name] = color
    ELEM_SIZES[name] = rvdw * 6  # scale vdW radius for plotting
ELEM_COLORS['E'] = '#00FF00'  # electron pairs: lime green
ELEM_SIZES['E'] = 4

# ---- Forcefield backends ----

def get_REQs(mol, atom_types, type_map=None):
    """Build (N,4) REQ array: [RvdW, sqrt(EvdW), Q, Hb] for each atom."""
    qs = mol.qs if mol.qs is not None else np.zeros(len(mol.apos))
    return make_REQs_from_enames(mol.enames, qs, atom_types, type_map=type_map)

# ---- Grid sampling ----

def compute_ff_map(mol, REQs, probe_ename, probe_q, z_height=0.0,
                   margin=4.0, step=0.1, ff_type='morseQ', R2damp=1.0,
                   atom_types=None):
    """Compute 2D energy map over a grid in the molecular plane (z=0).

    Args:
        mol: AtomicSystem (with electron pairs already added)
        REQs: (N,4) array of molecule atom parameters
        probe_ename: element name for probe (e.g. 'O', 'H')
        probe_q: charge of probe atom [e]
        z_height: z-coordinate of probe plane (0 = molecular plane)
        margin: margin around molecule bounding box [Å]
        step: grid spacing [Å]
        ff_type: 'morseQ' (Morse+Coulomb), 'coulomb' (Coulomb only), 'morse' (Morse only)
        atom_types: dict from read_atom_types (for probe REQ lookup)
    Returns:
        Emap (ny, nx), xs, ys, extent
    """
    apos = mol.apos
    # Bounding box
    xmin, ymin = apos[:, 0].min() - margin, apos[:, 1].min() - margin
    xmax, ymax = apos[:, 0].max() + margin, apos[:, 1].max() + margin
    xs = np.arange(xmin, xmax + step, step)
    ys = np.arange(ymin, ymax + step, step)
    nx, ny = len(xs), len(ys)

    # Probe REQ
    probe_REQ = np.zeros(4)
    if atom_types and probe_ename in atom_types:
        at = atom_types[probe_ename]
        probe_REQ[0] = at.RvdW
        probe_REQ[1] = np.sqrt(at.EvdW)
        probe_REQ[3] = at.Hb
    probe_REQ[2] = probe_q  # charge set by user

    # Precompute mixed REQs for all atoms (broadcast mixing)
    REQs_mixed = np.empty_like(REQs)
    REQs_mixed[:, 0] = probe_REQ[0] + REQs[:, 0]
    REQs_mixed[:, 1] = probe_REQ[1] * REQs[:, 1]
    REQs_mixed[:, 2] = probe_REQ[2] * REQs[:, 2]
    REQs_mixed[:, 3] = probe_REQ[3] * REQs[:, 3]
    REQs_mixed[REQs_mixed[:, 3] > 0, 3] = 0.0  # H <= 0

    # Vectorized grid computation: (ny, nx, N) -> sum over N
    X, Y = np.meshgrid(xs, ys)  # (ny, nx)
    # dp shape: (ny, nx, N, 3)
    dx = X[:, :, None] - apos[None, None, :, 0]
    dy = Y[:, :, None] - apos[None, None, :, 1]
    dz = z_height - apos[None, None, :, 2]
    r2 = dx*dx + dy*dy + dz*dz
    r = np.sqrt(r2)

    if ff_type == 'morseQ':
        e_exp = np.exp(-MORSE_BETA * (r - REQs_mixed[None, None, :, 0]))
        E_morse = REQs_mixed[None, None, :, 1] * (e_exp*e_exp - 2.0*e_exp)
        E_coul = COULOMB_CONST * REQs_mixed[None, None, :, 2] / np.sqrt(r2 + R2damp)
        Emap = (E_morse + E_coul).sum(axis=2)
    elif ff_type == 'coulomb':
        Q_prod = probe_q * REQs[:, 2]
        Emap = (COULOMB_CONST * Q_prod[None, None, :] / np.sqrt(r2 + R2damp)).sum(axis=2)
    elif ff_type == 'morse':
        e_exp = np.exp(-MORSE_BETA * (r - REQs_mixed[None, None, :, 0]))
        Emap = (REQs_mixed[None, None, :, 1] * (e_exp*e_exp - 2.0*e_exp)).sum(axis=2)
    elif ff_type == 'morseD':
        # Morse + CoulombDipole: primary particle (Morse+Coulomb) + compensating charge (Coulomb only)
        # at 1.0A along normalized repulsion direction. Net charge = 0.
        e_exp = np.exp(-MORSE_BETA * (r - REQs_mixed[None, None, :, 0]))
        E_morse = REQs_mixed[None, None, :, 1] * (e_exp*e_exp - 2.0*e_exp)
        E_coul = COULOMB_CONST * REQs_mixed[None, None, :, 2] / np.sqrt(r2 + R2damp)
        E_primary = (E_morse + E_coul).sum(axis=2)
        # Repulsion direction (3D, normalized)
        R0_mix = probe_REQ[0] + REQs[:, 0]
        E0_mix = probe_REQ[1] * REQs[:, 1]
        e2 = np.exp(-2*MORSE_BETA * (r - R0_mix[None, None, :]))
        coeff = 2*MORSE_BETA * E0_mix[None, None, :] * e2
        ir = 1.0/np.where(r > 1e-10, r, 1e-10)
        Fx = (coeff * dx * ir).sum(axis=2)
        Fy = (coeff * dy * ir).sum(axis=2)
        Fz = (coeff * dz * ir).sum(axis=2)
        Fmag = np.sqrt(Fx**2 + Fy**2 + Fz**2)
        Fmag_safe = np.where(Fmag > 1e-10, Fmag, 1.0)
        # Compensating charge position: grid_point + 1.0A * rep_dir
        cx = X + 1.0 * Fx / Fmag_safe
        cy = Y + 1.0 * Fy / Fmag_safe
        cz = z_height + 1.0 * Fz / Fmag_safe
        q_comp = -probe_q
        Q_prod_comp = q_comp * REQs[:, 2]
        dcx = cx[:, :, None] - apos[None, None, :, 0]
        dcy = cy[:, :, None] - apos[None, None, :, 1]
        dcz = cz[:, :, None] - apos[None, None, :, 2]
        rc2 = dcx*dcx + dcy*dcy + dcz*dcz
        E_comp = (COULOMB_CONST * Q_prod_comp[None, None, :] / np.sqrt(rc2 + R2damp)).sum(axis=2)
        Emap = E_primary + E_comp
    else:
        raise ValueError(f"Unknown ff_type: {ff_type}")

    extent = [xmin, xmax, ymin, ymax]
    return Emap, xs, ys, extent

def compute_efield_map(mol, z_height=0.0, margin=4.0, step=0.1, R2damp=1.0):
    """Compute 2D electric field (Ex, Ey) from molecule charges on a grid.
    Returns Ex, Ey (ny,nx), xs, ys, extent."""
    apos = mol.apos
    qs = mol.qs if mol.qs is not None else np.zeros(len(apos))
    xmin, ymin = apos[:, 0].min() - margin, apos[:, 1].min() - margin
    xmax, ymax = apos[:, 0].max() + margin, apos[:, 1].max() + margin
    xs = np.arange(xmin, xmax + step, step)
    ys = np.arange(ymin, ymax + step, step)
    X, Y = np.meshgrid(xs, ys)
    dx = X[:, :, None] - apos[None, None, :, 0]
    dy = Y[:, :, None] - apos[None, None, :, 1]
    dz = z_height - apos[None, None, :, 2]
    r2 = dx*dx + dy*dy + dz*dz + R2damp
    r3 = r2 * np.sqrt(r2)
    kq = COULOMB_CONST * qs[None, None, :]
    Ex = (kq * dx / r3).sum(axis=2)
    Ey = (kq * dy / r3).sum(axis=2)
    extent = [xmin, xmax, ymin, ymax]
    return Ex, Ey, xs, ys, extent

def compute_rep_force_map(mol, REQs, probe_ename, probe_q, z_height=0.0,
                           margin=4.0, step=0.1, atom_types=None):
    """Compute 2D Pauli repulsion force (Fx, Fy) from Morse repulsive part.
    E_rep = E0 * exp(-2*beta*(r-re)),  F = -dE/dr * r_hat = 2*beta*E0*exp(-2*beta*(r-re)) * r_hat
    Returns Fx, Fy (ny,nx), xs, ys, extent."""
    apos = mol.apos
    xmin, ymin = apos[:, 0].min() - margin, apos[:, 1].min() - margin
    xmax, ymax = apos[:, 0].max() + margin, apos[:, 1].max() + margin
    xs = np.arange(xmin, xmax + step, step)
    ys = np.arange(ymin, ymax + step, step)
    X, Y = np.meshgrid(xs, ys)
    dx = X[:, :, None] - apos[None, None, :, 0]
    dy = Y[:, :, None] - apos[None, None, :, 1]
    dz = z_height - apos[None, None, :, 2]
    r2 = dx*dx + dy*dy + dz*dz
    r = np.sqrt(r2)

    # Probe REQ and mixing
    probe_REQ = np.zeros(4)
    if atom_types and probe_ename in atom_types:
        at = atom_types[probe_ename]
        probe_REQ[0] = at.RvdW
        probe_REQ[1] = np.sqrt(at.EvdW)
    R0_mix = probe_REQ[0] + REQs[:, 0]
    E0_mix = probe_REQ[1] * REQs[:, 1]

    # F_rep = 2*beta*E0*exp(-2*beta*(r-re)) * (dp/r)  [points away from atom]
    e2 = np.exp(-2*MORSE_BETA * (r - R0_mix[None, None, :]))
    coeff = 2*MORSE_BETA * E0_mix[None, None, :] * e2  # (ny,nx,N)
    ir = 1.0/np.where(r > 1e-10, r, 1e-10)
    Fx = (coeff * dx * ir).sum(axis=2)
    Fy = (coeff * dy * ir).sum(axis=2)
    extent = [xmin, xmax, ymin, ymax]
    return Fx, Fy, xs, ys, extent

# ---- Plotting ----

FF_LABELS = {'morseQ': 'Morse+Coulomb', 'coulomb': 'Coulomb', 'morse': 'Morse', 'morseD': 'Morse+CoulDipole'}

def plot_atom_overlay(ax, mol, atom_Rs=None, probe_R=None):
    """Draw atoms (colored by element), charge labels, vdW circles, and bonds on ax."""
    from matplotlib.patches import Circle
    from matplotlib.collections import LineCollection
    for i, e in enumerate(mol.enames):
        x, y = mol.apos[i, 0], mol.apos[i, 1]
        q = mol.qs[i] if i < len(mol.qs) else 0.0
        color = ELEM_COLORS.get(e, '#FFA500')
        size = ELEM_SIZES.get(e, 6)
        ax.plot(x, y, 'o', color=color, markersize=size, markeredgecolor='gray')
        text_color = 'white' if e in ('C', 'O', 'N') else 'black'
        ax.text(x, y, f'{q:+.1f}', fontsize=6, ha='center', va='center', color=text_color)
        if atom_Rs is not None:
            Ri = atom_Rs[i]
            ax.add_patch(Circle((x, y), Ri, fill=False, edgecolor=color, linewidth=0.5, alpha=0.4))
            if probe_R is not None:
                ax.add_patch(Circle((x, y), Ri + probe_R, fill=False, edgecolor='gray', linewidth=0.5, linestyle='--', alpha=0.3))
    if mol.bonds is not None:
        segs = [[[mol.apos[i, 0], mol.apos[i, 1]], [mol.apos[j, 0], mol.apos[j, 1]]]
                for i, j in mol.bonds]
        ax.add_collection(LineCollection(segs, colors='k', linewidths=1.0, alpha=0.5))

def plot_ff_panel(ax, Emap, extent, mol, probe_ename, probe_q, ff_type,
                  z_height=0.0, atom_Rs=None, probe_R=None, title_suffix=""):
    """Plot a single ff map panel onto given ax. Symmetric colorscale: vmin=-vmax=|Emin|."""
    vmax = max(abs(Emap.min()), 0.01)
    im = ax.imshow(Emap, origin='lower', extent=extent, aspect='equal', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Energy [eV]')
    plot_atom_overlay(ax, mol, atom_Rs=atom_Rs, probe_R=probe_R)
    ff_label = FF_LABELS.get(ff_type, ff_type)
    ax.set_title(f'{ff_label} | {probe_ename} (q={probe_q:+.1f}e){title_suffix}')
    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')

# ---- Main ----

def set_epair_charges(mol, epair_charge=-0.2):
    """Set epair charges and subtract from host atom (charge conservation).
    Uses mol.bonds to find host atoms: each epair bond is [host, epair]."""
    ep_mask = np.array([e == 'E' for e in mol.enames])
    ep_indices = np.where(ep_mask)[0]
    if len(ep_indices) == 0:
        print("No electron pairs found")
        return mol
    mol.qs[ep_indices] = epair_charge
    # Each epair has exactly one neighbor (its host) in mol.ngs
    for i in ep_indices:
        host = next(j for j in mol.ngs[i] if mol.enames[j] != 'E')
        mol.qs[host] -= epair_charge
        print(f"  E{i}: q={epair_charge:+.2f}  host={mol.enames[host]}{host} q_host={mol.qs[host]:+.2f}")
    print(f"Set {len(ep_indices)} electron pair charges to q={epair_charge:+.2f}e (subtracted from hosts)")
    return mol

def load_molecule_with_epairs(xyz_path, atom_types=None, epair_charge=-0.2,
                               charge_mode='xyz', etypes=None, Q_target=0.0):
    """Load XYZ molecule, assign charges, add electron pairs.

    Args:
        xyz_path: path to .xyz file
        atom_types: dict from read_atom_types (for REQ lookup)
        epair_charge: charge per electron pair [e], set to 0 to disable
        charge_mode: 'xyz' = use 4th column from file, 'qeq' = charge equilibration
        etypes: dict from read_element_types (needed for QEq)
        Q_target: total charge constraint for QEq
    """
    mol = AtomicSystem(fname=xyz_path)
    mol.neighs()

    # --- Charge assignment (before epairs) ---
    n_real = len(mol.enames)
    if charge_mode == 'qeq':
        if etypes is None:
            raise ValueError("etypes required for QEq charge mode")
        q = qeq_solve(mol.apos.copy(), list(mol.enames), etypes, Q_target=Q_target)
        q = -q  # QEq returns electron occupancy; invert to get physical charges
        mol.qs = q.astype(np.float64)
        print(f"QEq charges (Q_target={Q_target}): sum={q.sum():.4f}")
        for i in range(n_real):
            print(f"  {i}: {mol.enames[i]:>2s}  q={q[i]:+.4f}")
    else:
        # 'xyz' mode: charges already loaded from 4th column by AtomicSystem
        if mol.qs is None:
            mol.qs = np.zeros(n_real, dtype=np.float64)
        print(f"XYZ charges: sum={mol.qs[:n_real].sum():.4f}")

    # --- Add electron pairs (place_electron_pair extends qs automatically) ---
    mol.add_electron_pairs()

    # --- Set epair charges (subtract from host) ---
    if epair_charge != 0.0:
        set_epair_charges(mol, epair_charge=epair_charge)
    else:
        ep_mask = np.array([e == 'E' for e in mol.enames])
        mol.qs[ep_mask] = 0.0
        print(f"Electron pair charges set to 0")

    return mol

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Forcefield interaction maps around flat molecules')
    parser.add_argument('--ff-types', nargs='+', default=['morse', 'morseQ', 'morseD'],
                        help='FF components to plot as columns (e.g. morse morseQ morseD coulomb)')
    parser.add_argument('--probes', nargs='+', default=['O:-0.4', 'H:+0.4'],
                        help='Probe atoms as NAME:CHARGE (e.g. O:-0.4 H:+0.4)')
    parser.add_argument('--xyz', default=None, help='XYZ file path')
    parser.add_argument('--charge-mode', default='xyz', choices=['xyz', 'qeq'], help='Charge source')
    parser.add_argument('--epair-charge', type=float, default=-0.2, help='Electron pair charge [e]')
    parser.add_argument('--margin', type=float, default=4.0, help='Grid margin [Å]')
    parser.add_argument('--step', type=float, default=0.1, help='Grid step [Å]')
    parser.add_argument('--z-height', type=float, default=0.0, help='Probe plane z [Å]')
    parser.add_argument('--quiver', choices=['efield', 'rep', 'auto', 'none'], default='auto',
                        help='Overlay arrows: efield (Coulomb), rep (Pauli repulsion), auto (efield on morseQ, rep on morse), none')
    parser.add_argument('--quiver-step', type=int, default=8, help='Arrow subsampling step (grid points)')
    parser.add_argument('--save', default=None, help='Save figure to PNG path (instead of showing)')
    parser.add_argument('--batch', nargs='+', default=None,
                        help='Batch mode: list of XYZ basenames (e.g. uracil CH2O pyridine)')
    args = parser.parse_args()

    data_path = os.path.join(REPO_ROOT, 'data')
    etypes = read_element_types(os.path.join(data_path, 'ElementTypes.dat'))
    atom_types = read_atom_types(os.path.join(data_path, 'AtomTypes.dat'), etypes)

    # Determine molecule list
    if args.batch:
        mol_names = args.batch
    else:
        xyz_path = args.xyz or os.path.join(data_path, 'xyz', 'CH2O.xyz')
        mol_names = [os.path.splitext(os.path.basename(xyz_path))[0]]

    for mol_name in mol_names:
        xyz_path = os.path.join(data_path, 'xyz', f'{mol_name}.xyz')
        if not os.path.exists(xyz_path):
            print(f"SKIP {mol_name}: {xyz_path} not found")
            continue
        print(f"\n{'='*60}\n  {mol_name}\n{'='*60}")
        fig = plot_molecule(xyz_path, mol_name, etypes, atom_types, args)
        if fig is None:
            continue
        if args.save:
            out_dir = args.save if os.path.isdir(args.save) else os.path.dirname(args.save)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f'ff_map_{mol_name}.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {out_path}")
            plt.close(fig)
        else:
            plt.show()


def plot_molecule(xyz_path, mol_name, etypes, atom_types, args):
    """Build and return the figure for one molecule. Returns None on error."""
    mol = load_molecule_with_epairs(xyz_path, atom_types=atom_types,
                                     epair_charge=args.epair_charge, charge_mode=args.charge_mode,
                                     etypes=etypes, Q_target=0.0)

    print(f"\nMolecule: {len(mol.enames)} atoms (incl. electron pairs)")
    for i, e in enumerate(mol.enames):
        print(f"  {i}: {e:>2s}  pos={mol.apos[i]}  q={mol.qs[i]:+.3f}")

    type_map = {'C': 'C_2', 'O': 'O_2', 'H': 'H'}
    REQs = get_REQs(mol, atom_types, type_map=type_map)

    probes = []
    for p in args.probes:
        name, q = p.split(':')
        probes.append((name, float(q)))

    ff_types = args.ff_types
    n_rows = len(probes)
    n_cols = len(ff_types)

    # Precompute E-field once (independent of probe)
    Ex_ef = Ey_ef = None
    if args.quiver in ('efield', 'auto'):
        Ex_ef, Ey_ef, _, _, _ = compute_efield_map(mol, z_height=args.z_height,
                                                     margin=args.margin, step=args.step)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), squeeze=False)
    fig.suptitle(f'{mol_name} | epair q={args.epair_charge} | z={args.z_height}', fontsize=10)

    for irow, (probe_ename, probe_q) in enumerate(probes):
        # Compute repulsion force for this probe (depends on probe REQ mixing)
        Fx_rep = Fy_rep = None
        if args.quiver in ('rep', 'auto'):
            Fx_rep, Fy_rep, _, _, _ = compute_rep_force_map(mol, REQs, probe_ename, probe_q,
                                                             z_height=args.z_height,
                                                             margin=args.margin, step=args.step,
                                                             atom_types=atom_types)
        for icol, ff_type in enumerate(ff_types):
            Emap, xs, ys, extent = compute_ff_map(
                mol, REQs, probe_ename, probe_q, z_height=args.z_height,
                margin=args.margin, step=args.step, ff_type=ff_type, atom_types=atom_types)
            print(f"{ff_type} | {probe_ename} q={probe_q:+.1f}e: E_range=[{Emap.min():.3f}, {Emap.max():.3f}] eV")
            probe_R = atom_types[probe_ename].RvdW if probe_ename in atom_types else None
            ax = axes[irow, icol]
            plot_ff_panel(ax, Emap, extent, mol, probe_ename, probe_q, ff_type,
                          z_height=args.z_height, atom_Rs=REQs[:, 0], probe_R=probe_R)
            # Quiver overlay: auto selects per ff_type
            s = args.quiver_step
            if args.quiver == 'auto':
                if ff_type in ('morseQ', 'morseD') and Ex_ef is not None:
                    Ux, Uy = Ex_ef[::s, ::s], Ey_ef[::s, ::s]
                elif ff_type == 'morse' and Fx_rep is not None:
                    Ux, Uy = Fx_rep[::s, ::s], Fy_rep[::s, ::s]
                else:
                    Ux = Uy = None
            elif args.quiver == 'efield' and Ex_ef is not None:
                Ux, Uy = Ex_ef[::s, ::s], Ey_ef[::s, ::s]
            elif args.quiver == 'rep' and Fx_rep is not None:
                Ux, Uy = Fx_rep[::s, ::s], Fy_rep[::s, ::s]
            else:
                Ux = Uy = None
            if Ux is not None:
                mag = np.sqrt(Ux**2 + Uy**2)
                mag[mag == 0] = 1.0
                ax.quiver(xs[::s], ys[::s], Ux/mag, Uy/mag,
                          color='black', alpha=0.5, scale=30, width=0.003)

    fig.tight_layout()
    return fig

if __name__ == '__main__':
    main()
