"""
How Phase Evolution of Molecular Orbitals Creates Moving Density Packets
========================================================================

THE CORE IDEA (what this script visualizes):
---------------------------------------------
Each eigenstate of the 1D chain is a standing wave (molecular orbital):
    ψ_n(x) ~ sin(k_n x)   or   cos(k_n x)

Under time evolution, each MO picks up a phase:
    ψ_n(x, t) = ψ_n(x) · exp(-i E_n t)

Individually, each MO just oscillates in place — the density |ψ_n|² is
time-independent. No motion.

But when you superpose several MOs with different energies:
    Ψ(x, t) = Σ_n c_n ψ_n(x) exp(-i E_n t)

the RELATIVE phases between the MOs drift at different rates (since each
E_n is different). This phase drift is what creates a MOVING wavepacket:
    |Ψ(x, t)|² = |Σ_n c_n ψ_n(x) e^{-iE_n t}|²

The packet moves at the GROUP VELOCITY:
    v_g = dE/dk

For the NORMAL chain (Δ=0):  E_k = ξ_k = -2t cos(k) - μ
    → dE/dk = 2t sin(k)  →  fast near k_F = π/2

For the SUPERCONDUCTING chain (Δ>0):  E_k = √(ξ_k² + Δ²)
    → dE/dk = (ξ_k/E_k) · 2t sin(k)  →  SLOW near gap edge (ξ_k→0)
    The factor ξ_k/E_k < 1 always reduces the velocity.
    At the exact gap edge: v_g = 0 (packet is nearly stationary).

Additionally, in the SC case the electron component oscillates into a
hole component (Andreev oscillation), which is a second visual signature.

WHAT YOU SEE (3 rows × 2 columns):
----------------------------------
Row 1: Individual MOs near k_F (Re part), each oscillating in place
       at its own frequency E_n. Colors distinguish different MOs.
       — Left: normal chain,  Right: SC chain

Row 2: Superposition Re(Ψ(x,t)) = Re(Σ c_n ψ_n e^{-iE_n t})
       The moving wave — you can see the phase fronts traveling.
       — Left: normal (fast),  Right: SC (slow + amplitude oscillation)

Row 3: Density |Ψ(x,t)|² — the actual electron density packet
       — Left: normal (moves fast across the chain)
       — Right: SC (moves slowly, breathes due to Andreev oscillation)

Run:  python sc_wavepacket_motion.py
      python sc_wavepacket_motion.py --save motion.gif
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# ============================================================================
# 1D Periodic Tight-Binding + BdG Hamiltonian
# ============================================================================
#
# H_0:  N×N tight-binding with periodic boundary conditions
#       H_0[i,i] = -μ,  H_0[i,i±1] = -t  (with wrapping)
#
# H_BdG: 2N×2N in Nambu basis (c_0↑,...,c_{N-1}↑, c†_0↓,...,c†_{N-1}↓)
#        [ H_0      Δ·I  ]
#        [ Δ·I    -H_0*  ]
#
# Eigenstates of H_0 are plane waves: ψ_k(x) = e^{ikx}/√N,  E_k = -2t cos(k) - μ
# Eigenstates of H_BdG are spinors: (u_k, v_k),  E_k = ±√(ξ_k² + Δ²)

def build_h0(N, t=1.0, mu=0.0):
    H = np.zeros((N, N), dtype=complex)
    for i in range(N):
        H[i, i] = -mu
        H[i, (i + 1) % N] = -t
        H[i, (i - 1) % N] = -t
    return H

def build_bdg(N, t=1.0, mu=0.0, Delta=0.15):
    H0 = build_h0(N, t, mu)
    H = np.zeros((2 * N, 2 * N), dtype=complex)
    H[:N, :N] = H0
    H[N:, N:] = -H0.conjugate()
    H[:N, N:] = Delta * np.eye(N)
    H[N:, :N] = Delta * np.eye(N)
    return H

# ============================================================================
# Select eigenstates near k_F and build wavepacket
# ============================================================================
#
# For a periodic chain, the normal eigenstates are plane waves with
# k_n = 2πn/N. We identify k for each eigenstate via FFT of the eigenvector.
#
# We select states near k_0 ≈ k_F + δk (slightly above Fermi point) with
# Gaussian weights c_n = exp(-(k_n - k_0)² / (2σ_k²)).
#
# For the BdG case, each eigenstate is a spinor (u, v). We use the electron
# component u_i as the "MO" and also track v_i for the hole density.
# The BdG eigenstate energies are E_n = +√(ξ_n² + Δ²) (positive branch).
# A pure electron wavepacket is expanded in BOTH +E and -E branches.

def get_k_values(eigvecs, N):
    """Identify the dominant k-value for each eigenvector via FFT."""
    k_grid = np.linspace(-np.pi, np.pi, N, endpoint=False)
    k_of_state = np.zeros(eigvecs.shape[1])
    for n in range(eigvecs.shape[1]):
        fft_vals = np.fft.fftshift(np.abs(np.fft.fft(eigvecs[:, n])))
        k_of_state[n] = k_grid[np.argmax(fft_vals)]
    return k_of_state

def prepare_wavepacket(N=200, t=1.0, mu=0.0, Delta=0.15,
                       k0=None, sigma_k=0.06, n_mos_show=5,
                       nsteps=500, dt=0.25):
    """
    Prepare all data for the animation.

    Returns a dict with:
    - Individual MOs (time-evolved): mo_normal[t, n, x], mo_bdg_u[t, n, x]
    - Superpositions: psi_normal[t, x], psi_bdg_u[t, x], psi_bdg_v[t, x]
    - Densities: rho_normal[t, x], rho_bdg_e[t, x], rho_bdg_h[t, x]
    - Selected k-values and energies for annotation
    """
    if k0 is None:
        k0 = np.pi / 2 + 0.2  # slightly above k_F for nonzero group velocity

    x = np.arange(N)

    # --- Normal chain ---
    H_n = build_h0(N, t, mu)
    E_n, V_n = np.linalg.eigh(H_n)
    k_n = get_k_values(V_n, N)

    # Select states near k0 with Gaussian weights
    weights_n = np.exp(-(k_n - k0)**2 / (2 * sigma_k**2))
    # Pick the top n_mos_show states by weight for individual display
    top_idx_n = np.argsort(weights_n)[-n_mos_show:]
    # All states with significant weight for the superposition
    significant_n = weights_n > 0.01 * weights_n.max()

    # --- BdG chain ---
    H_b = build_bdg(N, t, mu, Delta)
    E_b, V_b = np.linalg.eigh(H_b)

    # For BdG, we need to inject a PURE ELECTRON and expand in full basis.
    # Build initial pure-electron Gaussian wavepacket
    x0 = N // 4
    sigma_x = 1.0 / sigma_k  # Δx ~ 1/Δk
    u0 = np.exp(-(x - x0)**2 / (2 * sigma_x**2)) * np.exp(1j * k0 * x)
    u0 /= np.sqrt(np.sum(np.abs(u0)**2))
    psi0_bdg = np.concatenate([u0, np.zeros(N, dtype=complex)])
    coeffs_bdg = V_b.conjugate().T @ psi0_bdg

    # Also expand in normal basis for comparison
    coeffs_normal = V_n.conjugate().T @ u0

    # For displaying individual BdG MOs: pick positive-energy states near gap edge
    pos_b = E_b > 0
    E_b_pos = E_b[pos_b]
    V_b_pos = V_b[:, pos_b]
    k_b_pos = get_k_values(V_b_pos[:N], N)  # k from electron component

    weights_b = np.exp(-(k_b_pos - k0)**2 / (2 * sigma_k**2))
    top_idx_b = np.argsort(weights_b)[-n_mos_show:]

    # --- Precompute frames ---
    # Normal: individual MOs (top n_mos_show) and full superposition
    mo_normal = np.zeros((nsteps, n_mos_show, N), dtype=complex)
    psi_normal = np.zeros((nsteps, N), dtype=complex)

    # BdG: individual electron components of top MOs, and full superposition
    mo_bdg_u = np.zeros((nsteps, n_mos_show, N), dtype=complex)
    psi_bdg_u = np.zeros((nsteps, N), dtype=complex)
    psi_bdg_v = np.zeros((nsteps, N), dtype=complex)

    for step in range(nsteps):
        time = step * dt

        # --- Normal chain ---
        # Full superposition: Ψ(t) = Σ_n c_n ψ_n e^{-iE_n t}
        phase_n = np.exp(-1j * E_n * time)
        psi_normal[step] = V_n @ (phase_n * coeffs_normal)

        # Individual MOs (the top-weighted ones, with their coefficients)
        for j, idx in enumerate(top_idx_n):
            mo_normal[step, j] = V_n[:, idx] * np.exp(-1j * E_n[idx] * time) * weights_n[idx]

        # --- BdG chain ---
        # Full superposition (pure electron injected into full BdG basis)
        phase_b = np.exp(-1j * E_b * time)
        psi_bdg = V_b @ (phase_b * coeffs_bdg)
        psi_bdg_u[step] = psi_bdg[:N]
        psi_bdg_v[step] = psi_bdg[N:]

        # Individual BdG MOs (electron component of positive-energy states)
        for j, idx in enumerate(top_idx_b):
            u = V_b_pos[:N, idx]
            mo_bdg_u[step, j] = u * np.exp(-1j * E_b_pos[idx] * time) * weights_b[idx]

    # Densities
    rho_normal = np.abs(psi_normal)**2
    rho_bdg_e = np.abs(psi_bdg_u)**2
    rho_bdg_h = np.abs(psi_bdg_v)**2
    rho_bdg_charge = rho_bdg_e - rho_bdg_h

    # Energies of selected MOs for annotation
    E_selected_n = E_n[top_idx_n]
    k_selected_n = k_n[top_idx_n]
    E_selected_b = E_b_pos[top_idx_b]
    k_selected_b = k_b_pos[top_idx_b]

    return {
        'N': N, 'x': x, 'dt': dt, 'nsteps': nsteps,
        'Delta': Delta, 'k0': k0, 'sigma_k': sigma_k,
        'n_mos_show': n_mos_show,
        'mo_normal': mo_normal, 'psi_normal': psi_normal,
        'mo_bdg_u': mo_bdg_u, 'psi_bdg_u': psi_bdg_u, 'psi_bdg_v': psi_bdg_v,
        'rho_normal': rho_normal, 'rho_bdg_e': rho_bdg_e,
        'rho_bdg_h': rho_bdg_h, 'rho_bdg_charge': rho_bdg_charge,
        'E_selected_n': E_selected_n, 'k_selected_n': k_selected_n,
        'E_selected_b': E_selected_b, 'k_selected_b': k_selected_b,
        'weights_n': weights_n[top_idx_n], 'weights_b': weights_b[top_idx_b],
    }

# ============================================================================
# Animation
# ============================================================================

def animate_motion(data, save_path=None):
    """
    3 rows × 2 columns animation:
      Row 1: Individual MOs (Re part, overlaid) — each oscillates in place
      Row 2: Superposition Re(Ψ(x,t)) — the moving wave
      Row 3: Density |Ψ(x,t)|² — the moving packet

    Left column:  Normal (Δ=0) — fast packet
    Right column: Superconducting (Δ>0) — slow packet + Andreev breathing
    """
    N = data['N']
    x = data['x']
    dt = data['dt']
    nsteps = data['nsteps']
    Delta = data['Delta']
    k0 = data['k0']
    n_mos = data['n_mos_show']

    # Colors for individual MOs
    mo_colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_mos))

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex='col')

    # --- Row 1: Individual MOs ---
    # We'll show Re(ψ_n e^{-iE_n t}) for each selected MO
    lines_mo_n = []
    lines_mo_b = []
    for j in range(n_mos):
        ln_n, = axes[0, 0].plot([], [], color=mo_colors[j], lw=0.8, alpha=0.7)
        ln_b, = axes[0, 1].plot([], [], color=mo_colors[j], lw=0.8, alpha=0.7)
        lines_mo_n.append(ln_n)
        lines_mo_b.append(ln_b)

    # --- Row 2: Superposition Re(Ψ) ---
    line_psi_n, = axes[1, 0].plot([], [], 'b-', lw=1.5)
    line_psi_b, = axes[1, 1].plot([], [], 'b-', lw=1.5)

    # --- Row 3: Density |Ψ|² ---
    line_rho_n, = axes[2, 0].plot([], [], 'b-', lw=2)
    line_rho_b_e, = axes[2, 1].plot([], [], 'b-', lw=2, label=r"$|u|^2$ (electron)")
    line_rho_b_h, = axes[2, 1].plot([], [], 'r-', lw=1.5, alpha=0.7, label=r"$|v|^2$ (hole)")

    # Fill for density
    fill_n = axes[2, 0].fill_between(x, 0, 0, color='blue', alpha=0.15)
    fill_b = axes[2, 1].fill_between(x, 0, 0, color='blue', alpha=0.15)

    # --- Labels ---
    axes[0, 0].set_title(f"Normal ($\\Delta$=0): {n_mos} MOs near $k_F$\n"
                         r"Each oscillates in place at its own frequency $E_n$",
                         fontsize=9, loc='left')
    axes[0, 1].set_title(f"Superconducting ($\\Delta$={Delta}): {n_mos} BdG MOs near gap edge\n"
                         r"Each oscillates at $E_n=\sqrt{\xi_n^2+\Delta^2}$ (nearly equal → slow beating)",
                         fontsize=9, loc='left')

    axes[1, 0].set_title(r"Superposition $\mathrm{Re}\,\Psi(x,t) = \sum_n c_n\,\psi_n\,e^{-iE_n t}$"
                         "\nPhase fronts travel → moving wave", fontsize=9, loc='left')
    axes[1, 1].set_title(r"SC superposition $\mathrm{Re}\,u(x,t)$"
                         "\nSlow phase motion + amplitude breathing (Andreev)", fontsize=9, loc='left')

    axes[2, 0].set_title(r"Density $|\Psi(x,t)|^2$ — moving electron packet", fontsize=9, loc='left')
    axes[2, 1].set_title(r"SC density $|u|^2$, $|v|^2$ — slow packet + hole grows", fontsize=9, loc='left')

    for ax in axes[2]:
        ax.set_xlabel("site $i$")

    axes[0, 0].set_ylabel("Individual MOs\n" + r"$\mathrm{Re}\,\psi_n$", fontsize=9)
    axes[1, 0].set_ylabel("Superposition\n" + r"$\mathrm{Re}\,\Psi$", fontsize=9)
    axes[2, 0].set_ylabel("Density\n" + r"$|\Psi|^2$", fontsize=9)

    # --- Limits ---
    # Individual MOs: amplitude ~ weight * |ψ| ~ weight/√N
    mo_ymax = max(np.abs(data['mo_normal'].real).max(),
                  np.abs(data['mo_bdg_u'].real).max()) * 1.3
    for ax in axes[0]:
        ax.set_ylim(-mo_ymax, mo_ymax)
        ax.set_xlim(0, N)
        ax.axhline(0, color='gray', lw=0.3)

    # Superposition Re(Ψ)
    psi_ymax = max(np.abs(data['psi_normal'].real).max(),
                   np.abs(data['psi_bdg_u'].real).max()) * 1.2
    for ax in axes[1]:
        ax.set_ylim(-psi_ymax, psi_ymax)
        ax.set_xlim(0, N)
        ax.axhline(0, color='gray', lw=0.3)

    # Density
    rho_ymax = max(data['rho_normal'].max(), data['rho_bdg_e'].max(),
                   data['rho_bdg_h'].max()) * 1.2
    for ax in axes[2]:
        ax.set_ylim(-0.01, rho_ymax)
        ax.set_xlim(0, N)
        ax.axhline(0, color='gray', lw=0.3)

    axes[2, 1].legend(fontsize=8, loc='upper right')

    T_andreev = np.pi / Delta
    fig.suptitle(
        "Phase Evolution of Molecular Orbitals → Moving Density Packets\n"
        r"Normal: $E_k=\xi_k$, fast group velocity $v_g=2t\sin k$   |   "
        r"SC: $E_k=\sqrt{\xi_k^2+\Delta^2}$, slow $v_g=(\xi_k/E_k)\cdot 2t\sin k$   |   "
        f"Andreev period $T=\\pi/\\Delta$={T_andreev:.1f}",
        fontsize=11)

    # --- Animation functions ---
    nonlocal_fill = {'n': fill_n, 'b': fill_b}

    def init():
        all_lines = lines_mo_n + lines_mo_b + [line_psi_n, line_psi_b,
                                               line_rho_n, line_rho_b_e, line_rho_b_h]
        for ln in all_lines:
            ln.set_data([], [])
        return all_lines

    def update(frame):
        time = frame * dt

        # Row 1: Individual MOs (Re part)
        for j in range(n_mos):
            lines_mo_n[j].set_data(x, data['mo_normal'][frame, j].real)
            lines_mo_b[j].set_data(x, data['mo_bdg_u'][frame, j].real)

        # Row 2: Superposition Re(Ψ)
        line_psi_n.set_data(x, data['psi_normal'][frame].real)
        line_psi_b.set_data(x, data['psi_bdg_u'][frame].real)

        # Row 3: Density
        line_rho_n.set_data(x, data['rho_normal'][frame])
        line_rho_b_e.set_data(x, data['rho_bdg_e'][frame])
        line_rho_b_h.set_data(x, data['rho_bdg_h'][frame])

        # Update fills
        nonlocal_fill['n'].remove()
        nonlocal_fill['b'].remove()
        nonlocal_fill['n'] = axes[2, 0].fill_between(x, 0, data['rho_normal'][frame],
                                                      color='blue', alpha=0.15)
        nonlocal_fill['b'] = axes[2, 1].fill_between(x, 0, data['rho_bdg_e'][frame],
                                                      color='blue', alpha=0.15)

        # Time annotation
        axes[0, 0].set_title(f"Normal ($\\Delta$=0): {n_mos} MOs near $k_F$   t={time:.1f}",
                             fontsize=9, loc='left')
        axes[0, 1].set_title(f"Superconducting ($\\Delta$={Delta}): {n_mos} BdG MOs   t={time:.1f}",
                             fontsize=9, loc='left')

        all_lines = lines_mo_n + lines_mo_b + [line_psi_n, line_psi_b,
                                               line_rho_n, line_rho_b_e, line_rho_b_h]
        return all_lines

    anim = FuncAnimation(fig, update, frames=nsteps, init_func=init,
                         blit=True, interval=30)

    if save_path:
        anim.save(save_path, writer='pillow', fps=30, dpi=100)
        print(f"Saved to {save_path}")

    return fig, anim

# ============================================================================
# Decomposition Animation: Wavepacket → Individual MO Contributions
# ============================================================================
#
# THE KEY DECOMPOSITION:
#   Ψ(x,t) = Σ_n c_n ψ_n(x) e^{-iE_n t}
#
# The density is:
#   |Ψ(x,t)|² = Σ_n |c_n|² |ψ_n(x)|²                          (static part)
#             + Σ_{n≠m} Re[c_n c_m* ψ_n(x) ψ_m*(x) e^{-i(E_n-E_m)t}]  (interference)
#
# The STATIC part is just the sum of individual MO densities — it doesn't move.
# The INTERFERENCE terms are what create the time-dependent motion.
# Each interference term oscillates at the BEAT FREQUENCY (E_n - E_m).
#
# For the SC case, the beat frequencies are smaller (because E_n values are
# closer together near the gap), so the motion is slower.

def animate_decomposition(data, save_path=None):
    """
    4-row animation showing the wavepacket decomposition:

    Row 1: Bar chart of |c_n|² weights (which MOs contribute)
           — static, but shows the "recipe" of the wavepacket

    Row 2: Waterfall — each MO contribution Re(c_n ψ_n e^{-iE_n t}) stacked
           vertically. You see each wave oscillating at its own frequency,
           and how their sum (bottom trace) creates the moving packet.

    Row 3: Static density Σ|c_n|²|ψ_n|² vs full |Ψ|²
           — shows that the static sum doesn't move; only interference moves

    Row 4: Interference density = |Ψ|² - Σ|c_n|²|ψ_n|²
           — the purely time-dependent part that creates the motion
           (oscillates positive/negative, integrates to zero)

    Left column: Normal (Δ=0),  Right column: Superconducting (Δ>0)
    """
    N = data['N']
    x = data['x']
    dt = data['dt']
    nsteps = data['nsteps']
    Delta = data['Delta']
    k0 = data['k0']
    n_mos = data['n_mos_show']

    mo_colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_mos))

    # --- Precompute static density (sum of individual |c_n|²|ψ_n|²) ---
    # For normal chain: use the top n_mos_show states
    # We need the actual eigenvectors and coefficients
    # Rebuild from data
    H_n = build_h0(N)
    E_n_all, V_n_all = np.linalg.eigh(H_n)
    k_n_all = get_k_values(V_n_all, N)
    weights_n_all = np.exp(-(k_n_all - k0)**2 / (2 * data['sigma_k']**2))

    # Initial wavepacket
    x0 = N // 4
    sigma_x = 1.0 / data['sigma_k']
    u0 = np.exp(-(x - x0)**2 / (2 * sigma_x**2)) * np.exp(1j * k0 * x)
    u0 /= np.sqrt(np.sum(np.abs(u0)**2))
    coeffs_n_all = V_n_all.conjugate().T @ u0

    # Top states
    top_idx_n = np.argsort(weights_n_all)[-n_mos:]

    # Static density for normal: Σ |c_n|² |ψ_n|² (over ALL states)
    rho_static_n = np.zeros(N)
    for n in range(N):
        rho_static_n += np.abs(coeffs_n_all[n])**2 * np.abs(V_n_all[:, n])**2

    # For BdG: the decomposition is in BdG eigenstates, not normal MOs.
    # The "static" part in BdG basis is Σ|c_n|²|Φ_n|² where Φ_n = (u_n, v_n).
    # For the electron density: Σ|c_n|²|u_n|²
    H_b = build_bdg(N, Delta=Delta)
    E_b_all, V_b_all = np.linalg.eigh(H_b)
    psi0_bdg = np.concatenate([u0, np.zeros(N, dtype=complex)])
    coeffs_b_all = V_b_all.conjugate().T @ psi0_bdg

    rho_static_b_e = np.zeros(N)
    rho_static_b_h = np.zeros(N)
    for n in range(2 * N):
        rho_static_b_e += np.abs(coeffs_b_all[n])**2 * np.abs(V_b_all[:N, n])**2
        rho_static_b_h += np.abs(coeffs_b_all[n])**2 * np.abs(V_b_all[N:, n])**2

    # --- Precompute individual MO time-evolved contributions ---
    # For waterfall: Re(c_n ψ_n e^{-iE_n t}) for each of the top n_mos
    mo_contribs_n = np.zeros((nsteps, n_mos, N))  # real part
    mo_contribs_b = np.zeros((nsteps, n_mos, N))  # real part (electron component)

    # BdG positive-energy states near gap
    pos_b = E_b_all > 0
    E_b_pos = E_b_all[pos_b]
    V_b_pos = V_b_all[:, pos_b]
    k_b_pos = get_k_values(V_b_pos[:N], N)
    weights_b_all = np.exp(-(k_b_pos - k0)**2 / (2 * data['sigma_k']**2))
    top_idx_b = np.argsort(weights_b_all)[-n_mos:]

    for step in range(nsteps):
        time = step * dt
        for j, idx in enumerate(top_idx_n):
            phase = np.exp(-1j * E_n_all[idx] * time)
            mo_contribs_n[step, j] = (V_n_all[:, idx] * phase * coeffs_n_all[idx]).real
        for j, idx in enumerate(top_idx_b):
            phase = np.exp(-1j * E_b_pos[idx] * time)
            mo_contribs_b[step, j] = (V_b_pos[:N, idx] * phase * coeffs_b_all[np.where(pos_b)[0][idx]]).real

    # --- Set up figure ---
    fig, axes = plt.subplots(4, 2, figsize=(14, 11), sharex='col')

    # Row 1: Bar chart of |c_n|²
    bar_n = axes[0, 0].bar(range(n_mos), np.abs(coeffs_n_all[top_idx_n])**2,
                            color=mo_colors, edgecolor='black', lw=0.5)
    axes[0, 0].set_title(f"Normal: $|c_n|^2$ weights (which MOs make up the packet)", fontsize=9, loc='left')
    axes[0, 0].set_ylabel("$|c_n|^2$")
    axes[0, 0].set_xticks(range(n_mos))
    E_n_sel = E_n_all[top_idx_n]
    k_n_sel = k_n_all[top_idx_n]
    axes[0, 0].set_xticklabels([f"$k$={k:.2f}\n$E$={e:.2f}" for k, e in zip(k_n_sel, E_n_sel)],
                                fontsize=6)

    bar_b = axes[0, 1].bar(range(n_mos), np.abs(coeffs_b_all[np.where(pos_b)[0][top_idx_b]])**2,
                            color=mo_colors, edgecolor='black', lw=0.5)
    axes[0, 1].set_title(f"SC: $|c_n|^2$ weights (BdG eigenstates near gap edge)", fontsize=9, loc='left')
    axes[0, 1].set_ylabel("$|c_n|^2$")
    axes[0, 1].set_xticks(range(n_mos))
    E_b_sel = E_b_pos[top_idx_b]
    k_b_sel = k_b_pos[top_idx_b]
    axes[0, 1].set_xticklabels([f"$k$={k:.2f}\n$E$={e:.2f}" for k, e in zip(k_b_sel, E_b_sel)],
                                fontsize=6)

    # Row 2: Waterfall — stacked MO contributions with offset
    # Each MO trace is offset vertically for visibility
    waterfall_offset = max(np.abs(mo_contribs_n).max(), np.abs(mo_contribs_b).max()) * 2.5
    lines_wf_n = []
    lines_wf_b = []
    for j in range(n_mos):
        ln_n, = axes[1, 0].plot([], [], color=mo_colors[j], lw=0.8)
        ln_b, = axes[1, 1].plot([], [], color=mo_colors[j], lw=0.8)
        lines_wf_n.append(ln_n)
        lines_wf_b.append(ln_b)
    # Sum trace (thick black)
    line_sum_n, = axes[1, 0].plot([], [], 'k-', lw=2)
    line_sum_b, = axes[1, 1].plot([], [], 'k-', lw=2)

    wf_ymax = waterfall_offset * (n_mos + 1)
    for ax in axes[1]:
        ax.set_ylim(-waterfall_offset, wf_ymax)
        ax.set_xlim(0, N)
        ax.set_ylabel("MO contributions\n(stacked, offset)")
    axes[1, 0].set_title("Normal: each MO Re($c_n\\psi_n e^{-iE_nt}$)\n"
                         "Sum (black) = moving wavepacket", fontsize=9, loc='left')
    axes[1, 1].set_title("SC: each BdG MO Re($c_n u_n e^{-iE_nt}$)\n"
                         "Sum (black) = slow + breathing packet", fontsize=9, loc='left')

    # Row 3: Static density vs full density
    line_static_n, = axes[2, 0].plot(x, rho_static_n, 'g--', lw=1.5,
                                      label=r"$\sum|c_n|^2|\psi_n|^2$ (static, no motion)")
    line_full_n, = axes[2, 0].plot([], [], 'b-', lw=2, label=r"$|\Psi(x,t)|^2$ (full, moves)")
    line_static_b_e, = axes[2, 1].plot(x, rho_static_b_e, 'g--', lw=1.5,
                                        label=r"$\sum|c_n|^2|u_n|^2$ (static)")
    line_full_b_e, = axes[2, 1].plot([], [], 'b-', lw=2, label=r"$|u(x,t)|^2$ (full)")
    line_full_b_h, = axes[2, 1].plot([], [], 'r-', lw=1.5, alpha=0.7, label=r"$|v(x,t)|^2$ (hole)")

    rho_ymax = max(rho_static_n.max(), data['rho_normal'].max(),
                   rho_static_b_e.max(), data['rho_bdg_e'].max(),
                   data['rho_bdg_h'].max()) * 1.2
    for ax in axes[2]:
        ax.set_ylim(-0.01, rho_ymax)
        ax.set_xlim(0, N)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc='upper right')
    axes[2, 0].set_title("Normal: static sum (green) doesn't move!\n"
                         "Only the full |Ψ|² (blue) moves — that's interference", fontsize=9, loc='left')
    axes[2, 1].set_title("SC: static electron part (green) is flat\n"
                         "Full |u|² moves + |v|² grows from zero", fontsize=9, loc='left')

    # Row 4: Interference = full - static
    line_interf_n, = axes[3, 0].plot([], [], 'purple', lw=1.5)
    line_interf_b, = axes[3, 1].plot([], [], 'purple', lw=1.5)

    # Precompute interference
    interf_n = data['rho_normal'] - rho_static_n[np.newaxis, :]
    interf_b = data['rho_bdg_e'] - rho_static_b_e[np.newaxis, :]

    interf_ymax = max(abs(interf_n).max(), abs(interf_b).max()) * 1.3
    for ax in axes[3]:
        ax.set_ylim(-interf_ymax, interf_ymax)
        ax.set_xlim(0, N)
        ax.axhline(0, color='gray', lw=0.3)
        ax.set_ylabel("Interference\n$|\\Psi|^2 - \\sum|c_n|^2|\\psi_n|^2$")
        ax.set_xlabel("site $i$")
    axes[3, 0].set_title("Normal: interference pattern (moves → drives the packet)", fontsize=9, loc='left')
    axes[3, 1].set_title("SC: interference (slower oscillation → slower packet)", fontsize=9, loc='left')

    fig.suptitle(
        "Wavepacket Decomposition into Molecular Orbitals\n"
        r"$\Psi(x,t)=\sum_n c_n\,\psi_n(x)\,e^{-iE_n t}$   "
        r"$\Rightarrow$   $|\Psi|^2 = \sum|c_n|^2|\psi_n|^2$ (static) "
        r"$+ \sum_{n\neq m}\mathrm{Re}[c_nc_m^*\psi_n\psi_m^*e^{-i(E_n-E_m)t}]$ (interference = motion)",
        fontsize=10)

    def init():
        all_lines = (lines_wf_n + lines_wf_b + [line_sum_n, line_sum_b,
                     line_full_n, line_full_b_e, line_full_b_h,
                     line_interf_n, line_interf_b])
        for ln in all_lines:
            ln.set_data([], [])
        return all_lines

    def update(frame):
        time = frame * dt

        # Row 2: Waterfall
        for j in range(n_mos):
            offset = (n_mos - 1 - j) * waterfall_offset
            lines_wf_n[j].set_data(x, mo_contribs_n[frame, j] + offset)
            lines_wf_b[j].set_data(x, mo_contribs_b[frame, j] + offset)
        line_sum_n.set_data(x, data['psi_normal'][frame].real)
        line_sum_b.set_data(x, data['psi_bdg_u'][frame].real)

        # Row 3: Static vs full
        line_full_n.set_data(x, data['rho_normal'][frame])
        line_full_b_e.set_data(x, data['rho_bdg_e'][frame])
        line_full_b_h.set_data(x, data['rho_bdg_h'][frame])

        # Row 4: Interference
        line_interf_n.set_data(x, interf_n[frame])
        line_interf_b.set_data(x, interf_b[frame])

        # Time annotation
        axes[1, 0].set_title(f"Normal: MO contributions   t={time:.1f}", fontsize=9, loc='left')
        axes[1, 1].set_title(f"SC: BdG MO contributions   t={time:.1f}", fontsize=9, loc='left')

        all_lines = (lines_wf_n + lines_wf_b + [line_sum_n, line_sum_b,
                     line_full_n, line_full_b_e, line_full_b_h,
                     line_interf_n, line_interf_b])
        return all_lines

    anim = FuncAnimation(fig, update, frames=nsteps, init_func=init,
                         blit=True, interval=30)

    if save_path:
        anim.save(save_path, writer='pillow', fps=30, dpi=100)
        print(f"Saved decomposition to {save_path}")

    return fig, anim

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="How MO phase evolution creates moving density packets")
    parser.add_argument('--save', type=str, default=None, help="Save as GIF (e.g. motion.gif)")
    parser.add_argument('--Delta', type=float, default=0.15, help="SC gap parameter")
    parser.add_argument('--N', type=int, default=200, help="Number of lattice sites")
    parser.add_argument('--nmos', type=int, default=5, help="Number of individual MOs to show")
    parser.add_argument('--nsteps', type=int, default=500, help="Animation frames")
    parser.add_argument('--dt', type=float, default=0.25, help="Time step")
    parser.add_argument('--decomp', action='store_true',
                        help="Show decomposition animation instead of motion animation")
    args = parser.parse_args()

    print(f"Preparing wavepacket data (N={args.N}, Δ={args.Delta})...")
    data = prepare_wavepacket(
        N=args.N, Delta=args.Delta, n_mos_show=args.nmos,
        nsteps=args.nsteps, dt=args.dt)

    # Physics check
    print(f"  t=0: Σ|u|²={data['rho_bdg_e'][0].sum():.4f}, "
          f"Σ|v|²={data['rho_bdg_h'][0].sum():.6f} (should be ~0)")
    print(f"  Normal packet peak: t=0 at x={np.argmax(data['rho_normal'][0])}, "
          f"t={args.nsteps*args.dt/2:.0f} at x={np.argmax(data['rho_normal'][args.nsteps//2])}")
    print(f"  Selected normal E_n: {np.sort(data['E_selected_n'])}")
    print(f"  Selected BdG E_n:    {np.sort(data['E_selected_b'])}")
    print(f"  Energy spread normal: {data['E_selected_n'].max()-data['E_selected_n'].min():.4f}")
    print(f"  Energy spread BdG:    {data['E_selected_b'].max()-data['E_selected_b'].min():.4f}")
    print(f"  (Smaller spread → slower packet, as dE/dk is flatter)")

    if args.decomp:
        print("Building decomposition animation...")
        fig, anim = animate_decomposition(data, save_path=args.save)
    else:
        print("Building motion animation...")
        fig, anim = animate_motion(data, save_path=args.save)

    plt.show()

if __name__ == '__main__':
    main()
