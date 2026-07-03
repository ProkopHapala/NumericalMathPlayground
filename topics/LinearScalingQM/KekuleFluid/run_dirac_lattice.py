#!/usr/bin/env python3
"""run_dirac_lattice.py — Two-defect PAH: nodal line, HSV phase, eigenstates.

Physical background
-------------------
When two H+ defects are placed on a symmetric PAH flake and pinned to different
Kekulé patterns, the Kekulé order parameter z must rotate in phase between the
two pinning sites.  This phase rotation forces a nodal line (|z|→0 contour)
to emerge, connecting the two defects — a topological domain wall.  The bond
orders along this line are suppressed, weakening the Kekulé mass gap locally.
In the π-electron spectrum, this creates in-gap states localized at the defects
and along the nodal line.

Role in the system
------------------
This is the main production script.  It runs the full causal chain:
  1. Model A: two H+ defects → Kekulé order parameter z_i → bond orders x_ij
  2. Model B: bond orders → Hamiltonian → eigenstates → LDOS

Produces 5 figures:
  - HSV phase field + domain wall + nodal lines (3-panel)
  - Bond-length distortion (2-panel: with/without defects)
  - Energy spectrum and DOS (2-panel, physical band only)
  - Localized eigenstate density plots
  - Local density of states (LDOS) map

Key functions
-------------
- `run_two_defects(n_shells, phase_mode, outdir, ...)` — main entry point.
- `find_mirror_atom_x(graph, atom_idx)` — find mirror image across y-axis.
- `find_pin_neighbor_toward_center(graph, defect_idx)` — find pin atom.
- `mirror_residual_x(graph, z)` — quantify mirror symmetry of z field.
- CLI: `--phase-mode symmetric|domain-wall`, `--n-shells`, `--outdir`.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Graph import HoneycombGraph
from ModelA import KekuleFluidSolver
from DiracLattice_ocl import DiracLatticeSolver
from visualize import HexVisualizer
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def find_edge_atom_at_angle(graph, target_angle):
    """Find the edge atom closest to the given polar angle."""
    pos = np.array([[a.pos[0], a.pos[1]] for a in graph.atoms])
    edge = graph.find_edge_atoms()
    edge_arr = np.array(edge)
    angles = np.arctan2(pos[edge_arr, 1], pos[edge_arr, 0])
    dist = np.abs(np.angle(np.exp(1j * (angles - target_angle))))
    best = np.argmin(dist)
    return edge_arr[best]


def find_mirror_atom_x(graph, atom_idx):
    pos = np.array([[a.pos[0], a.pos[1]] for a in graph.atoms])
    target = np.array([-pos[atom_idx, 0], pos[atom_idx, 1]])
    dist = np.linalg.norm(pos - target, axis=1)
    return int(np.argmin(dist))


def find_pin_neighbor_toward_center(graph, atom_idx):
    neighbors = graph.get_neighbor_list()
    best_idx = None
    best_d2 = 1e18
    for d in range(3):
        nb = neighbors[atom_idx, d]
        if nb < 0:
            continue
        px, py = graph.atoms[nb].pos
        d2 = px * px + py * py
        if d2 < best_d2:
            best_d2 = d2
            best_idx = int(nb)
    return best_idx


def mirror_residual_x(graph, z):
    pos = np.array([[a.pos[0], a.pos[1]] for a in graph.atoms])
    mirror = np.array([
        int(np.argmin(np.linalg.norm(pos - np.array([-p[0], p[1]]), axis=1)))
        for p in pos
    ])
    amp = np.abs(z)
    return dict(
        amp=float(np.max(np.abs(amp - amp[mirror]))),
        z_conj=float(np.max(np.abs(z[mirror] - np.conj(z)))),
    )


def plot_eigenstate_on_lattice(ax, graph, evec, eval, defect_indices,
                               bond_x=None, title="", cmap='RdBu_r'):
    """Plot |psi_i|^2 for one eigenstate on the honeycomb lattice."""
    pos = np.array([[a.pos[0], a.pos[1]] for a in graph.atoms])
    density = np.abs(evec)**2

    bond_segs = [[pos[b.iA], pos[b.iB]] for b in graph.bonds]
    if bond_x is not None:
        bx = np.asarray(bond_x)
        norm_b = Normalize(vmin=bx.min(), vmax=bx.max())
        colors = plt.cm.bwr(norm_b(bx))
        lw = 0.5 + 3.0 * np.abs(bx - bx.mean()) / (bx.std() + 1e-8)
        lc = LineCollection(bond_segs, colors=colors, linewidths=lw, zorder=0)
    else:
        lc = LineCollection(bond_segs, colors='gray', linewidths=0.5, alpha=0.3, zorder=0)
    ax.add_collection(lc)

    vmax = max(density.max(), 1e-8)
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=density, cmap=cmap, s=60,
                    edgecolors='black', linewidths=0.3, vmin=0, vmax=vmax, zorder=5)

    for idx in defect_indices:
        dp = pos[idx]
        ax.plot(dp[0], dp[1], '*', color='lime', markersize=15,
                markeredgecolor='black', markeredgewidth=0.5, zorder=10)

    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.set_title(title)
    return sc


def find_gap(evals):
    """Find the Kekule gap (around E=0 for bipartite lattice)."""
    n = len(evals)
    mid = n // 2
    if mid > 0 and mid < n:
        return evals[mid] - evals[mid - 1]
    return 0.0


def find_ingap_states(evals, reference_gap, tolerance=0.5):
    """Find states inside the gap region (|E| < reference_gap * tolerance)."""
    threshold = reference_gap * tolerance
    return [k for k in range(len(evals)) if abs(evals[k]) < threshold]


def find_near_gap_states(evals, n):
    """Find n states closest to E=0 (band edges)."""
    sorted_idx = np.argsort(np.abs(evals))
    return list(sorted_idx[:n])


def run_two_defects(n_shells=3, nsteps_A=600, outdir='output', phase_mode='symmetric'):
    """Two H+ defects on opposite edges with mirror-controlled phase pins."""

    # --- Build PAH ---
    g = HoneycombGraph(aCC=1.0)
    g.build_pah(n_shells=n_shells)
    print(f"PAH flake (n_shells={n_shells}): {g.natom} atoms, {g.nbond} bonds, {len(g.rings)} rings")
    assert all(sum(1 for b in a.bonds if b >= 0) >= 2 for a in g.atoms), "Dangling bonds!"

    pos = np.array([[a.pos[0], a.pos[1]] for a in g.atoms])

    defect1_idx = find_edge_atom_at_angle(g, 0.0)
    defect2_idx = find_mirror_atom_x(g, defect1_idx)
    pin_dir1 = 0
    pin_dir2 = 1 if phase_mode == 'domain-wall' else 0
    d1p = pos[defect1_idx]
    d2p = pos[defect2_idx]
    print(f"Defect 1: atom {defect1_idx} at ({d1p[0]:.2f}, {d1p[1]:.2f}) — pin_dir={pin_dir1}")
    print(f"Defect 2: atom {defect2_idx} at ({d2p[0]:.2f}, {d2p[1]:.2f}) — pin_dir={pin_dir2}")

    # --- Run Model A WITH two defects ---
    print("Running Model A with two H+ defects...")
    g_def = HoneycombGraph(aCC=1.0)
    g_def.build_pah(n_shells=n_shells)
    defect1_idx_def = g_def.find_nearest_atom(d1p[0], d1p[1])
    defect2_idx_def = find_mirror_atom_x(g_def, defect1_idx_def)
    pin1_idx_def = find_pin_neighbor_toward_center(g_def, defect1_idx_def)
    pin2_idx_def = find_mirror_atom_x(g_def, pin1_idx_def)
    p1p = np.array(g_def.atoms[pin1_idx_def].pos)
    p2p = np.array(g_def.atoms[pin2_idx_def].pos)
    print(f"Pin 1: atom {pin1_idx_def} at ({p1p[0]:.2f}, {p1p[1]:.2f})")
    print(f"Pin 2: atom {pin2_idx_def} at ({p2p[0]:.2f}, {p2p[1]:.2f})")

    solverA_def = KekuleFluidSolver(g_def)
    ctx = solverA_def.ctx
    queue = solverA_def.queue
    solverA_def.init_uniform_kekule(phi0=0.0)
    solverA_def.add_proton_defect(defect1_idx_def, pin_dir=pin_dir1, pin_strength=1.0,
                                  pin_atom_idx=pin1_idx_def)
    solverA_def.add_proton_defect(defect2_idx_def, pin_dir=pin_dir2, pin_strength=1.0,
                                  pin_atom_idx=pin2_idx_def)
    solverA_def.run(nsteps_A)
    solverA_def.pull_to_graph()

    zr_def, zi_def = solverA_def.get_z()
    z_def = zr_def + 1j * zi_def
    bx_def = solverA_def.get_bond_x()
    diag = solverA_def.compute_diagnostics()
    print(f"  |z| range=[{diag['z_abs_min']:.3f},{diag['z_abs_max']:.3f}]  mean={diag['z_abs_mean']:.3f}")
    print(f"  Valence residual: max={diag['valence_max_err']:.4f}  mean={diag['valence_mean_err']:.4f}")
    print(f"  Low-|z| atoms: {diag['n_low_amp']}")
    sym = mirror_residual_x(g_def, z_def)
    print(f"  Mirror residual: |z|={sym['amp']:.4e}  z↔conj(z)={sym['z_conj']:.4e}")

    # Count nodal crossings
    re_nodal = sum(1 for b in g_def.bonds
                   if zr_def[b.iA] * zr_def[b.iB] < 0)
    im_nodal = sum(1 for b in g_def.bonds
                   if zi_def[b.iA] * zi_def[b.iB] < 0)
    print(f"  Nodal crossings: Re(z)=0 on {re_nodal} bonds, Im(z)=0 on {im_nodal} bonds")

    # --- Run Model A WITHOUT defects (reference) ---
    print("Running Model A without defects (reference)...")
    g_clean = HoneycombGraph(aCC=1.0)
    g_clean.build_pah(n_shells=n_shells)
    solverA_clean = KekuleFluidSolver(g_clean, ctx=ctx, queue=queue)
    solverA_clean.init_uniform_kekule(phi0=0.0)
    solverA_clean.run(nsteps_A)
    solverA_clean.pull_to_graph()

    zr_clean, zi_clean = solverA_clean.get_z()
    bx_clean = solverA_clean.get_bond_x()

    # --- Diagonalize both Hamiltonians ---
    defect_indices_def = [defect1_idx_def, defect2_idx_def]

    print("Diagonalizing Hamiltonian (with defects)...")
    solverB_def = DiracLatticeSolver(g_def, ctx=ctx, queue=queue, t0=1.0, dt_kek=0.5, dt=0.05, V_def=20.0)
    solverB_def.update_bond_x(bx_def)
    defect_arr_def = np.array([a.defect for a in g_def.atoms], dtype=np.float64)
    evals_def, evecs_def = solverB_def.diagonalize(bx_def, defect_arr_def)

    print("Diagonalizing Hamiltonian (without defects)...")
    solverB_clean = DiracLatticeSolver(g_clean, ctx=ctx, queue=queue, t0=1.0, dt_kek=0.5, dt=0.05, V_def=20.0)
    solverB_clean.update_bond_x(bx_clean)
    defect_arr_clean = np.zeros(g_clean.natom, dtype=np.float64)
    evals_clean, evecs_clean = solverB_clean.diagonalize(bx_clean, defect_arr_clean)

    print(f"  Energy spectrum (defects):    [{evals_def.min():.3f}, {evals_def.max():.3f}]")
    print(f"  Energy spectrum (no defect):  [{evals_clean.min():.3f}, {evals_clean.max():.3f}]")
    print(f"  Gap (no defect): {find_gap(evals_clean):.4f}")
    print(f"  Gap (defects):    {find_gap(evals_def):.4f}")

    gap_clean = find_gap(evals_clean)
    ingap_states = find_ingap_states(evals_def, gap_clean, tolerance=0.5)
    print(f"  In-gap states: {len(ingap_states)}")
    for k in ingap_states:
        locs = [np.abs(evecs_def[idx, k])**2 / np.sum(np.abs(evecs_def[:, k])**2)
                for idx in defect_indices_def]
        print(f"    E[{k}] = {evals_def[k]:.4f}  weights at defects = {locs}")

    # --- Visualization ---
    viz = HexVisualizer(g_def)
    pos_def = np.array([[a.pos[0], a.pos[1]] for a in g_def.atoms])

    # ===== Figure 1: HSV phase field + domain wall =====
    fig1, axes1 = plt.subplots(1, 3, figsize=(22, 7))

    # Left: HSV phase with graph overlay + nodal lines + low-|z| contour
    viz.plot_phase_hsv(zr_def, zi_def, bond_x=bx_def, ax=axes1[0],
                       title="Kekulé phase (HSV) + nodal lines",
                       defect_indices=defect_indices_def, resolution=300,
                       mode='smooth')

    # Middle: |z| amplitude showing suppression along nodal line
    viz.plot_z_field(zr_def, zi_def, mode='amplitude', ax=axes1[1],
                     bond_x=bx_def, title="$|z|$ (amplitude — dark = nodal line)",
                     add_colorbar=True)
    for idx in defect_indices_def:
        axes1[1].plot(pos_def[idx, 0], pos_def[idx, 1], '*',
                      color='lime', markersize=15, markeredgecolor='black',
                      markeredgewidth=0.5, zorder=10)

    # Right: Physical domain wall (low-|z| contour, phase-convention independent)
    viz.plot_domain_wall(zr_def, zi_def, ax=axes1[2],
                         title="Domain wall (low $|z|$)",
                         defect_indices=defect_indices_def)

    # Draw dashed line between defects for didactic clarity
    for ax in axes1:
        ax.plot([d1p[0], d2p[0]], [d1p[1], d2p[1]], '--',
                color='white', alpha=0.4, linewidth=1, zorder=1)

    phase_title = "mirror-symmetric pins" if phase_mode == 'symmetric' else "phase-mismatched pins"
    fig1.suptitle(f"Two H+ defects: {phase_title} (PAH {g.natom} atoms)", fontsize=14)
    fig1.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(outdir, exist_ok=True)
    fig1.savefig(os.path.join(outdir, "pah_hsv_phase.png"), dpi=150)
    print(f"Saved {outdir}/pah_hsv_phase.png")

    # ===== Figure 2: Bond-length distortion + clean comparison =====
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    bondBase_def = g_def.bondBase
    delta_x = bx_def - bondBase_def
    viz.plot_bond_order(delta_x, ax=axes2[0],
                        title="Bond distortion $\\delta x_{ij}$ (two defects)",
                        cmap='RdBu_r',
                        vmin=-np.abs(delta_x).max(), vmax=np.abs(delta_x).max(),
                        add_colorbar=True)
    for idx in defect_indices_def:
        axes2[0].plot(pos_def[idx, 0], pos_def[idx, 1], '*',
                      color='lime', markersize=15, markeredgecolor='black',
                      markeredgewidth=0.5, zorder=10)

    delta_x_clean = bx_clean - g_clean.bondBase
    viz_clean = HexVisualizer(g_clean)
    viz_clean.plot_bond_order(delta_x_clean, ax=axes2[1],
                              title="Bond distortion $\\delta x_{ij}$ (no defect)",
                              cmap='RdBu_r',
                              vmin=-np.abs(delta_x).max(), vmax=np.abs(delta_x).max(),
                              add_colorbar=True)

    fig2.suptitle("Bond-length distortion field", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, "pah_bond_distortion.png"), dpi=150)
    print(f"Saved {outdir}/pah_bond_distortion.png")

    # ===== Figure 3: Energy spectra comparison =====
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

    sigma_dos = 0.03
    # Focus on physical band; defect states at ~V_def are excluded from plot range
    E_phys_max = max(abs(evals_clean.min()), abs(evals_clean.max())) + 0.5
    E_range = (-E_phys_max, E_phys_max)
    energies = np.linspace(E_range[0], E_range[1], 500)

    dos_clean = np.zeros_like(energies)
    dos_def = np.zeros_like(energies)
    for e in evals_clean:
        dos_clean += np.exp(-(energies - e)**2 / (2*sigma_dos**2))
    for e in evals_def:
        if E_range[0] <= e <= E_range[1]:
            dos_def += np.exp(-(energies - e)**2 / (2*sigma_dos**2))

    n_defect_states = int(np.sum(np.abs(evals_def) > E_phys_max))

    axes3[0].plot(energies, dos_clean, 'b-', label='no defect', alpha=0.7)
    axes3[0].plot(energies, dos_def, 'r-', label='two H+ defects', alpha=0.7)
    for k in ingap_states:
        axes3[0].axvline(evals_def[k], color='green', linestyle='--', alpha=0.5)
    axes3[0].set_xlabel('Energy')
    axes3[0].set_ylabel('DOS')
    axes3[0].set_title(f'Density of states ({n_defect_states} defect states at ~V_def excluded)')
    axes3[0].legend()

    # Plot only physical-band states; mark excluded defect states count
    evals_def_phys = evals_def[np.abs(evals_def) <= E_phys_max]
    axes3[1].plot(evals_clean, 'b.', label='no defect', markersize=4)
    axes3[1].plot(evals_def_phys, 'r.', label='two H+ defects', markersize=4)
    for k in ingap_states:
        axes3[1].plot(k, evals_def[k], 'g*', markersize=12)
    axes3[1].set_xlabel('state index')
    axes3[1].set_ylabel('Energy')
    axes3[1].set_title(f'Energy spectrum ({n_defect_states} states at ~±V_def excluded)')
    axes3[1].legend()

    fig3.suptitle("Energy spectrum: two defects open in-gap states", fontsize=14)
    fig3.tight_layout()
    fig3.savefig(os.path.join(outdir, "pah_spectrum.png"), dpi=150)
    print(f"Saved {outdir}/pah_spectrum.png")

    # ===== Figure 4: Localized eigenstates =====
    n_show = min(8, len(ingap_states) + 2)
    show_states = ingap_states[:n_show] if len(ingap_states) >= n_show else (
        ingap_states + find_near_gap_states(evals_def, n_show - len(ingap_states)))

    n_cols = min(4, len(show_states))
    n_rows = (len(show_states) + n_cols - 1) // n_cols
    fig4, axes4 = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes4 = np.array([[axes4]])
    elif n_rows == 1:
        axes4 = axes4.reshape(1, -1)
    elif n_cols == 1:
        axes4 = axes4.reshape(-1, 1)

    for idx, k in enumerate(show_states):
        r, c = idx // n_cols, idx % n_cols
        locs = [np.abs(evecs_def[idx2, k])**2 / np.sum(np.abs(evecs_def[:, k])**2)
                for idx2 in defect_indices_def]
        title = f"E = {evals_def[k]:.4f}  (defect weights = {locs[0]:.2f}, {locs[1]:.2f})"
        sc = plot_eigenstate_on_lattice(axes4[r, c], g_def, evecs_def[:, k],
                                        evals_def[k], defect_indices_def,
                                        bond_x=bx_def, title=title)
        fig4.colorbar(sc, ax=axes4[r, c], fraction=0.046, label='$|\\psi_i|^2$')

    for idx in range(len(show_states), n_rows * n_cols):
        r, c = idx // n_cols, idx % n_cols
        axes4[r, c].set_visible(False)

    fig4.suptitle("Localized eigenstates: two H+ defects + nodal line", fontsize=14)
    fig4.tight_layout(rect=[0, 0, 1, 0.94])
    fig4.savefig(os.path.join(outdir, "pah_localized_states.png"), dpi=150)
    print(f"Saved {outdir}/pah_localized_states.png")

    # ===== Figure 5: LDOS at defect vs nodal line vs far =====
    fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6))

    energies_ldos, ldos_def, _, _ = solverB_def.compute_ldos(bx_def, defect_arr_def, sigma=0.03)

    ldos_at_defect1 = ldos_def[defect1_idx_def, :]
    ldos_at_defect2 = ldos_def[defect2_idx_def, :]

    # Find a point near the middle of the nodal line (between the two defects)
    mid_pos = 0.5 * (pos_def[defect1_idx_def] + pos_def[defect2_idx_def])
    dist_to_mid = np.sqrt((pos_def[:, 0] - mid_pos[0])**2 +
                          (pos_def[:, 1] - mid_pos[1])**2)
    nodal_idx = np.argmin(dist_to_mid)

    # Far from both defects and nodal line
    dist_to_defects = np.minimum(
        np.sqrt((pos_def[:, 0] - d1p[0])**2 + (pos_def[:, 1] - d1p[1])**2),
        np.sqrt((pos_def[:, 0] - d2p[0])**2 + (pos_def[:, 1] - d2p[1])**2)
    )
    far_mask = dist_to_defects > 5.0
    ldos_far = ldos_def[far_mask, :].mean(axis=0)

    axes5[0].plot(energies_ldos, ldos_far, 'b-', label='far from defects', alpha=0.7)
    axes5[0].plot(energies_ldos, ldos_at_defect1, 'r-', label='at defect 1', alpha=0.7)
    axes5[0].plot(energies_ldos, ldos_at_defect2, 'm-', label='at defect 2', alpha=0.7)
    axes5[0].plot(energies_ldos, ldos_def[nodal_idx, :], 'g-', label='nodal line midpoint', alpha=0.7)
    axes5[0].set_xlabel('Energy')
    axes5[0].set_ylabel('LDOS')
    axes5[0].set_title('Local density of states')
    axes5[0].legend()

    # LDOS map at the in-gap energy
    if len(ingap_states) > 0:
        target_E = evals_def[ingap_states[0]]
    else:
        target_E = 0.0
    idx_E = np.argmin(np.abs(energies_ldos - target_E))
    ldos_map = ldos_def[:, idx_E]

    sc = axes5[1].scatter(pos_def[:, 0], pos_def[:, 1], c=ldos_map, cmap='hot',
                          s=60, edgecolors='black', linewidths=0.3)
    bond_segs = [[pos_def[b.iA], pos_def[b.iB]] for b in g_def.bonds]
    lc = LineCollection(bond_segs, colors='gray', linewidths=0.5, alpha=0.3, zorder=0)
    axes5[1].add_collection(lc)
    for idx in defect_indices_def:
        axes5[1].plot(pos_def[idx, 0], pos_def[idx, 1], '*',
                      color='lime', markersize=15, markeredgecolor='black',
                      markeredgewidth=0.5, zorder=10)
    axes5[1].set_aspect('equal')
    axes5[1].set_title(f'LDOS map at E = {energies_ldos[idx_E]:.3f}')
    fig5.colorbar(sc, ax=axes5[1], fraction=0.046, label='LDOS')

    fig5.suptitle("LDOS: localization near defects and low-|z| texture", fontsize=14)
    fig5.tight_layout()
    fig5.savefig(os.path.join(outdir, "pah_ldos.png"), dpi=150)
    print(f"Saved {outdir}/pah_ldos.png")

    print("\nDone.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='PAH lattice Dirac solver: two H+ defects with nodal line')
    parser.add_argument('--n-shells', type=int, default=3,
                        help='Ring shells (0=benzene, 1=coronene, 2=circumcoronene, 3=...)')
    parser.add_argument('--steps', type=int, default=600, help='Model A timesteps')
    parser.add_argument('--outdir', default='output', help='Output directory for images')
    parser.add_argument('--phase-mode', choices=['symmetric', 'domain-wall'], default='symmetric',
                        help='symmetric uses mirror-compatible pins; domain-wall uses different Z3 phases')
    args = parser.parse_args()

    print("=" * 60)
    print("PAH lattice Dirac solver: two H+ defects")
    print("=" * 60)
    run_two_defects(n_shells=args.n_shells, nsteps_A=args.steps, outdir=args.outdir,
                    phase_mode=args.phase_mode)
