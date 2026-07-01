"""
Superconducting Gap & Wavepacket Visualization on a 1D Tight-Binding Chain
==========================================================================

This script is a didactic visualization of how superconductivity modifies
the electronic structure of a 1D crystal and how the modified eigenstates
near the gap edge combine into moving "superconducting wavepackets".

PHYSICAL SETUP
--------------
We consider a 1D tight-binding chain (the simplest model of a crystal):

    H_0 = -t Σ_i (c†_{i+1} c_i + h.c.) - μ Σ_i c†_i c_i

In k-space (periodic chain), the normal-state dispersion is:

    ξ_k = -2t cos(k) - μ        (measured from Fermi level)

At half-filling (μ=0), the Fermi points are at k_F = ±π/2, where ξ_k = 0.

SUPERCONDUCTING PAIRING (BCS/BdG)
---------------------------------
We add an on-site s-wave pairing gap Δ (from electron-phonon coupling
mediated attraction, treated at mean-field level). The Bogoliubov–de Gennes
(BdG) Hamiltonian in Nambu space (c_k, c†_{-k}) is:

    H_BdG(k) = [ ξ_k    Δ  ]    →    eigenvalues: E_k = ±√(ξ_k² + Δ²)
               [ Δ*   -ξ_k ]

KEY PHYSICS:
1. GAP OPENING: The normal dispersion ξ_k is "folded" — positive and negative
   branches repel each other due to the off-diagonal Δ coupling. The result
   is E_k = √(ξ_k² + Δ²), which has a minimum gap of 2Δ at the Fermi points.

2. ELECTRON-HOLE MIXING: Each BdG eigenstate is a spinor (u_k, v_k):
       u_k² = ½(1 + ξ_k/E_k)    (electron character)
       v_k² = ½(1 - ξ_k/E_k)    (hole character)
   At the gap edge (ξ_k=0): u_k² = v_k² = ½  →  50/50 electron-hole mix.
   Far from the gap (|ξ_k| >> Δ): either u≈1 (pure electron) or v≈1 (pure hole).

3. GROUP VELOCITY: v_g = dE_k/dk = (ξ_k/E_k) · dξ_k/dk
   At the gap edge: v_g → 0. The quasiparticle is "heavy" — it barely moves.
   This is why superconducting excitations near the gap are slow.

4. ANDREEV OSCILLATION: If you inject a pure electron at the Fermi point,
   it oscillates between electron and hole character at frequency Δ/ℏ:
       u(t) = cos(Δt),   v(t) = -i sin(Δt)
   Period: T_Andreev = π/Δ. This is the real-space signature of the gap.

WAVEPACKET CONSTRUCTION
-----------------------
Following the recipe from the document:
  1. Diagonalize the BdG Hamiltonian → get E_n, (u_n, v_n) for each eigenstate.
  2. Select states near the positive gap edge (E ≈ +Δ).
  3. Weight them with a Gaussian in k-space: c_k = exp(-(k-k_0)²/2σ_k²).
  4. The real-space wavepacket at time t is:
       Ψ_e(x,t) = Σ_k c_k u_k(x) e^{-iE_k t}
       Ψ_h(x,t) = Σ_k c_k v_k(x) e^{-iE_k t}
  5. Plot |Ψ_e|² (electron density), |Ψ_h|² (hole density), and their
     difference (net charge density).

WHAT YOU SEE IN THE ANIMATION
-----------------------------
- |u(x,t)|²:  electron component of the Bogoliubov quasiparticle
- |v(x,t)|²:  hole component (starts at zero, grows via Andreev oscillation)
- |u|²-|v|²:  net charge — much smaller than either component alone
              (this is the "neutral" nature of the superconducting excitation)

Run:  python sc_gap_wavepacket.py
      python sc_gap_wavepacket.py --save movie.gif
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec

# ============================================================================
# Model: 1D Periodic Tight-Binding + BdG
# ============================================================================
#
# We use a PERIODIC chain (not open) so that eigenstates are Bloch waves
# with well-defined k. This makes the connection to the BdG spectrum E(k)
# exact and clean.
#
# H_0(k) = ξ_k = -2t cos(k) - μ
# H_BdG(k) = [ ξ_k   Δ  ]
#             [ Δ    -ξ_k ]
#
# For each k, the 2×2 BdG Hamiltonian is diagonalized analytically:
#   E_k = ±√(ξ_k² + Δ²)
#   u_k = cos(θ_k/2),  v_k = sin(θ_k/2),  where tan(θ_k) = Δ/ξ_k
#
# But for the real-space wavepacket we need the BdG eigenstates in the
# site basis, so we build the full 2N×2N matrix and diagonalize numerically.

def build_bdg_periodic(N, t=1.0, mu=0.0, Delta=0.15):
    """
    Build the BdG Hamiltonian for a PERIODIC 1D chain.

    The normal Hamiltonian H_0 has:
      - diagonal: -μ (onsite energy = chemical potential)
      - off-diagonal: -t (nearest-neighbor hopping)
      - periodic boundary: H[0, N-1] = H[N-1, 0] = -t

    The BdG matrix in Nambu basis (c_0↑, ..., c_{N-1}↑, c†_0↓, ..., c†_{N-1}↓):

        H_BdG = [ H_0       Δ·I  ]
                [ Δ·I    -H_0*   ]

    Returns the 2N×2N complex matrix.
    """
    H0 = np.zeros((N, N), dtype=complex)
    for i in range(N):
        H0[i, i] = -mu
        H0[i, (i + 1) % N] = -t  # forward hop with periodic BC
        H0[i, (i - 1) % N] = -t  # backward hop
    H = np.zeros((2 * N, 2 * N), dtype=complex)
    H[:N, :N] = H0
    H[N:, N:] = -H0.conjugate()  # hole block = -H_0* (time-reversed)
    H[:N, N:] = Delta * np.eye(N)  # pairing: Δ c†_i↑ c†_i↓ + h.c.
    H[N:, :N] = Delta * np.eye(N)
    return H

# ============================================================================
# Static Figure 1: Gap Opening — Normal vs BdG Spectrum
# ============================================================================

def plot_gap_opening(t=1.0, mu=0.0, Delta=0.15):
    """
    Show how the superconducting gap opens in the BdG spectrum.

    Left panel:  Normal dispersion ξ_k = -2t cos(k) - μ (dashed)
                 BdG dispersion    E_k = √(ξ_k² + Δ²) (solid)
                 The gap 2Δ opens at the Fermi points k_F = ±π/2.

    Right panel: Electron/hole weights u_k² and v_k² vs k.
                 At gap edge: u² = v² = ½ (maximum mixing).
                 Far from gap: pure electron (u²→1) or pure hole (v²→1).
    """
    k = np.linspace(-np.pi, np.pi, 1000)
    xi = -2 * t * np.cos(k) - mu
    E = np.sqrt(xi**2 + Delta**2)

    # BdG coherence factors (electron/hole weights)
    # u_k² = ½(1 + ξ_k/E_k),  v_k² = ½(1 - ξ_k/E_k)
    u2 = 0.5 * (1 + xi / E)
    v2 = 0.5 * (1 - xi / E)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: Spectrum ---
    ax1.plot(k, xi, 'g--', lw=1.5, alpha=0.7, label=r"Normal $\xi_k = -2t\cos k - \mu$")
    ax1.plot(k, -xi, 'g--', lw=1.5, alpha=0.7)
    ax1.plot(k, E, 'b-', lw=2.5, label=r"BdG $E_k = \sqrt{\xi_k^2+\Delta^2}$")
    ax1.plot(k, -E, 'b-', lw=2.5)
    ax1.fill_between(k, -Delta, Delta, alpha=0.15, color='red', label=f"Gap $2\\Delta$={2*Delta}")
    ax1.axhline(Delta, color='r', ls=':', lw=1)
    ax1.axhline(-Delta, color='r', ls=':', lw=1)
    ax1.axhline(0, color='gray', lw=0.5)
    ax1.axvline(np.pi / 2, color='orange', ls=':', lw=1, alpha=0.7, label="$k_F$")
    ax1.axvline(-np.pi / 2, color='orange', ls=':', lw=1, alpha=0.7)
    ax1.set_xlabel("$k$ (lattice units)")
    ax1.set_ylabel("Energy")
    ax1.set_title("Superconducting Gap Opening", fontsize=12)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.set_xlim(-np.pi, np.pi)
    ax1.set_ylim(-3, 3)

    # --- Right: Coherence factors ---
    ax2.plot(k, u2, 'b-', lw=2, label=r"$|u_k|^2 = \frac{1}{2}(1+\xi_k/E_k)$  (electron)")
    ax2.plot(k, v2, 'r-', lw=2, label=r"$|v_k|^2 = \frac{1}{2}(1-\xi_k/E_k)$  (hole)")
    ax2.axvline(np.pi / 2, color='orange', ls=':', lw=1, alpha=0.7, label="$k_F$")
    ax2.axvline(-np.pi / 2, color='orange', ls=':', lw=1, alpha=0.7)
    ax2.axhline(0.5, color='gray', ls=':', lw=0.5)
    ax2.annotate("50/50 mix\n(gap edge)", xy=(np.pi/2, 0.5), xytext=(np.pi/2+0.5, 0.75),
                 fontsize=8, arrowprops=dict(arrowstyle='->', color='black'),
                 ha='left')
    ax2.annotate("50/50 mix\n(gap edge)", xy=(-np.pi/2, 0.5), xytext=(-np.pi/2-0.5, 0.75),
                 fontsize=8, arrowprops=dict(arrowstyle='->', color='black'),
                 ha='right')
    ax2.set_xlabel("$k$")
    ax2.set_ylabel("Weight")
    ax2.set_title("BdG Coherence Factors: Electron–Hole Mixing", fontsize=12)
    ax2.legend(fontsize=8, loc='center')
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    return fig

# ============================================================================
# Static Figure 2: BdG Eigenstates Near Gap Edge in Real Space
# ============================================================================

def plot_gap_edge_states(N=128, t=1.0, mu=0.0, Delta=0.15, n_states=4):
    """
    Show the real-space structure of BdG eigenstates near the positive gap edge.

    Each BdG eigenstate is a spinor (u_i, v_i) on each site.
    Near the gap edge, |u_i|² ≈ |v_i|² — the state is an equal
    superposition of electron and hole amplitudes.

    The eigenstates are Bloch-like (plane waves modulated by the gap),
    with |u_i|² and |v_i|² being spatially uniform for a translationally
    invariant chain. The real-space structure becomes visible when we
    build a wavepacket (next figure).
    """
    H = build_bdg_periodic(N, t, mu, Delta)
    E, V = np.linalg.eigh(H)
    x = np.arange(N)

    # Positive-energy states sorted by energy, pick those near +Delta
    pos_mask = E > 0
    E_pos = E[pos_mask]
    V_pos = V[:, pos_mask]

    # Sort by proximity to gap edge E=Delta
    order = np.argsort(np.abs(E_pos - Delta))
    gap_edge_idx = order[:n_states]

    fig, axes = plt.subplots(n_states, 2, figsize=(12, 2.5 * n_states), squeeze=False)
    for row, idx in enumerate(gap_edge_idx):
        u = V_pos[:N, idx]
        v = V_pos[N:, idx]
        E_val = E_pos[idx]

        # Real part of u and v (showing the wave-like character)
        axes[row, 0].plot(x, u.real, 'b-', lw=1, label=r"$\mathrm{Re}\,u_i$")
        axes[row, 0].plot(x, v.real, 'r-', lw=1, label=r"$\mathrm{Re}\,v_i$")
        axes[row, 0].set_ylabel(f"E={E_val:.4f}")
        axes[row, 0].legend(fontsize=7, loc='upper right')
        if row == 0:
            axes[row, 0].set_title("Real-space amplitude (Re part)", fontsize=10)

        # Densities |u|² and |v|²
        axes[row, 1].plot(x, np.abs(u)**2, 'b-', lw=1.5, label=r"$|u_i|^2$")
        axes[row, 1].plot(x, np.abs(v)**2, 'r-', lw=1.5, label=r"$|v_i|^2$")
        axes[row, 1].legend(fontsize=7, loc='upper right')
        if row == 0:
            axes[row, 1].set_title(r"Density $|u|^2$, $|v|^2$ (≈uniform for Bloch states)", fontsize=10)

        # Annotate electron/hole fraction
        frac_e = np.sum(np.abs(u)**2) / (np.sum(np.abs(u)**2) + np.sum(np.abs(v)**2))
        axes[row, 1].text(0.02, 0.95, f"electron frac: {frac_e:.2f}",
                          transform=axes[row, 1].transAxes, fontsize=8, va='top')

    axes[-1, 0].set_xlabel("site $i$")
    axes[-1, 1].set_xlabel("site $i$")
    fig.suptitle(f"BdG Eigenstates Near Gap Edge ($E \\approx \\Delta$={Delta})\n"
                 f"Note: $|u|^2 \\approx |v|^2$ → equal electron-hole mixing", fontsize=11, y=1.01)
    plt.tight_layout()
    return fig

# ============================================================================
# Interactive Animation: Superconducting Wavepacket with Time Slider
# ============================================================================

def animate_sc_wavepacket(N=256, t=1.0, mu=0.0, Delta=0.15,
                          k0=None, sigma_k=0.08, x0=None,
                          nsteps=400, dt=0.3):
    """
    Build and animate a superconducting (Bogoliubov) wavepacket.

    CONSTRUCTION (following the document's recipe):
    1. Diagonalize BdG → E_n, V_n = (u_n, v_n)
    2. Select states near positive gap edge (E ≈ +Δ)
    3. Weight with Gaussian in k-space around k_0 (slightly off k_F
       so the packet has nonzero group velocity)
    4. Time evolve: Ψ(t) = Σ_n c_n V_n exp(-i E_n t)

    The wavepacket shows:
    - |u(x,t)|²:  electron density of the Bogoliubov quasiparticle
    - |v(x,t)|²:  hole density (grows from zero via Andreev oscillation)
    - |u|²-|v|²:  net charge density (small — the excitation is nearly neutral)

    PHYSICAL INTERPRETATION:
    - The packet moves with BdG group velocity v_g = (ξ_k/E_k)·dξ_k/dk,
      which is REDUCED compared to the normal state by the factor ξ_k/E_k.
    - Simultaneously, electron character oscillates into hole character
      and back at the Andreev frequency ω_A = Δ (period T = π/Δ).
    - Near the gap edge, v_g → 0, so the packet is nearly stationary
      but still oscillates between electron and hole.
    - The net charge |u|²-|v|² can be much smaller than either |u|² or |v|²
      separately — this is the hallmark of a superconducting excitation:
      it is partly "added electron" and partly "removed electron" (hole).
    """
    if k0 is None:
        k0 = np.pi / 2 + 0.25  # slightly above k_F for nonzero v_g
    if x0 is None:
        x0 = N // 4

    # --- Build and diagonalize BdG ---
    H = build_bdg_periodic(N, t, mu, Delta)
    E, V = np.linalg.eigh(H)

    # --- Identify k-values for each eigenstate ---
    # For a periodic chain, eigenstates of H_0 are plane waves with
    # k_n = 2πn/N. The BdG eigenstates inherit this structure.
    k_vals = 2 * np.pi * np.arange(N) / N - np.pi  # first Brillouin zone

    # --- Select positive-energy states near gap edge ---
    pos_mask = E > 0
    E_pos = E[pos_mask]
    V_pos = V[:, pos_mask]

    # For each positive-energy BdG state, find its k-value.
    # The BdG eigenstates come in pairs; we need to identify k for each.
    # We do this by looking at the momentum content of u component.
    k_of_state = np.zeros(len(E_pos))
    for n in range(len(E_pos)):
        u = V_pos[:N, n]
        fft_u = np.fft.fftshift(np.abs(np.fft.fft(u)))
        k_idx = np.argmax(fft_u)
        k_of_state[n] = np.linspace(-np.pi, np.pi, N, endpoint=False)[k_idx]

    # --- Build Gaussian weights in k-space ---
    # c_k = exp(-(k - k_0)² / (2 σ_k²))
    # This selects a narrow band of k-states around k_0, near the gap edge.
    # The width σ_k controls how localized the packet is in real space:
    #   Δx ~ 1/σ_k  (uncertainty principle)
    weights = np.exp(-(k_of_state - k0)**2 / (2 * sigma_k**2))
    weights /= np.sqrt(np.sum(weights**2))

    # --- Build initial BdG spinor ---
    # Ψ(0) = Σ_n c_n V_n
    # This is a superposition of gap-edge BdG eigenstates.
    # At t=0, it is mostly electron-like (if k_0 is near k_F from above)
    # because u_k > v_k for ξ_k > 0.
    psi0 = V_pos @ weights  # shape (2N,)
    psi0 = np.concatenate([psi0[:N], psi0[N:]])  # ensure correct shape

    # Actually, let's be more careful: V_pos is (2N, N_pos), weights is (N_pos,)
    psi0 = V_pos @ weights  # (2N,)

    # Normalize
    psi0 /= np.sqrt(np.sum(np.abs(psi0)**2))

    # --- Precompute expansion coefficients for time evolution ---
    # ψ(t) = V_pos @ (exp(-i E_pos t) * weights)
    # We precompute weights once; at each time step just apply phases.
    coeffs = weights.copy()
    coeffs /= np.sqrt(np.sum(np.abs(coeffs)**2))

    # --- Also build a NORMAL-state wavepacket for comparison ---
    # Same k_0, same σ_k, but Δ=0 (pure electron, no hole component)
    H_normal = build_bdg_periodic(N, t, mu, Delta=0.0)
    E_n, V_n = np.linalg.eigh(H_normal)
    pos_n = E_n > 0
    E_n_pos = E_n[pos_n]
    V_n_pos = V_n[:, pos_n]

    # Find k for normal states
    k_of_normal = np.zeros(len(E_n_pos))
    for n in range(len(E_n_pos)):
        u = V_n_pos[:N, n]
        fft_u = np.fft.fftshift(np.abs(np.fft.fft(u)))
        k_idx = np.argmax(fft_u)
        k_of_normal[n] = np.linspace(-np.pi, np.pi, N, endpoint=False)[k_idx]

    weights_n = np.exp(-(k_of_normal - k0)**2 / (2 * sigma_k**2))
    weights_n /= np.sqrt(np.sum(weights_n**2))
    coeffs_n = weights_n.copy()

    # --- Set up figure ---
    x = np.arange(N)
    fig = plt.figure(figsize=(13, 9))
    gs = GridSpec(4, 1, height_ratios=[1, 1, 1, 0.15], hspace=0.35, top=0.92, bottom=0.15)

    ax_normal = fig.add_subplot(gs[0])
    ax_u = fig.add_subplot(gs[1], sharex=ax_normal)
    ax_v = fig.add_subplot(gs[2], sharex=ax_normal)
    ax_charge = fig.add_subplot(gs[3], sharex=ax_normal)

    # Plot lines
    line_normal, = ax_normal.plot([], [], 'g-', lw=1.5)
    line_u, = ax_u.plot([], [], 'b-', lw=1.8)
    line_v, = ax_v.plot([], [], 'r-', lw=1.8)
    line_charge, = ax_charge.plot([], [], 'k-', lw=1.5)

    # Fill areas for visual impact
    fill_u = ax_u.fill_between(x, 0, 0, color='blue', alpha=0.2)
    fill_v = ax_v.fill_between(x, 0, 0, color='red', alpha=0.2)

    # Labels and limits
    ax_normal.set_ylabel(r"$|\psi|^2$")
    ax_normal.set_title(r"Normal chain ($\Delta$=0): pure electron wavepacket — moves at Fermi velocity", fontsize=10, loc='left')
    ax_normal.axhline(0, color='gray', lw=0.3)

    ax_u.set_ylabel(r"$|u_i|^2$")
    ax_u.set_title(r"Superconducting ($\Delta$=%.2f): electron component $|u_i(t)|^2$" % Delta, fontsize=10, loc='left')
    ax_u.axhline(0, color='gray', lw=0.3)

    ax_v.set_ylabel(r"$|v_i|^2$")
    ax_v.set_title(r"Hole component $|v_i(t)|^2$ (grows from zero via Andreev oscillation)", fontsize=10, loc='left')
    ax_v.axhline(0, color='gray', lw=0.3)

    ax_charge.set_ylabel(r"$|u|^2\!-\!|v|^2$")
    ax_charge.set_title(r"Net charge density (much smaller than either component → nearly neutral excitation)", fontsize=10, loc='left')
    ax_charge.set_xlabel("site $i$")
    ax_charge.axhline(0, color='gray', lw=0.3)

    for ax in [ax_normal, ax_u, ax_v, ax_charge]:
        ax.set_xlim(0, N)
        ax.set_ylim(-0.01, 0.06)

    ax_charge.set_ylim(-0.05, 0.05)

    fig.suptitle("Superconducting Wavepacket: Gap-Edge BdG Quasiparticle in Real Space\n"
                 r"$E_k=\sqrt{\xi_k^2+\Delta^2}$,  $v_g=(\xi_k/E_k)\,d\xi_k/dk \to 0$ at gap edge,  Andreev period $T=\pi/\Delta$"
                 f" = {np.pi/Delta:.1f}",
                 fontsize=12)

    # --- Time slider ---
    ax_slider = fig.add_axes([0.15, 0.02, 0.7, 0.03])
    t_max = nsteps * dt
    slider = Slider(ax_slider, 'time', 0, t_max, valinit=0, valfmt='%.1f')

    # --- Precompute all frames for smooth slider ---
    print("Precomputing frames...")
    all_ne = np.zeros((nsteps, N))
    all_nh = np.zeros((nsteps, N))
    all_charge = np.zeros((nsteps, N))
    all_ne_normal = np.zeros((nsteps, N))

    for step in range(nsteps):
        time = step * dt

        # BdG wavepacket
        phase = np.exp(-1j * E_pos * time)
        psi = V_pos @ (phase * coeffs)
        u = psi[:N]
        v = psi[N:]
        all_ne[step] = np.abs(u)**2
        all_nh[step] = np.abs(v)**2
        all_charge[step] = all_ne[step] - all_nh[step]

        # Normal wavepacket
        phase_n = np.exp(-1j * E_n_pos * time)
        psi_n = V_n_pos @ (phase_n * coeffs_n)
        all_ne_normal[step] = np.abs(psi_n[:N])**2

    print("Done precomputing.")

    # --- Auto-scale y limits ---
    ymax = max(all_ne.max(), all_nh.max(), all_ne_normal.max()) * 1.15
    ax_normal.set_ylim(-0.01, ymax)
    ax_u.set_ylim(-0.01, ymax)
    ax_v.set_ylim(-0.01, ymax)
    cmax = max(abs(all_charge.min()), abs(all_charge.max())) * 1.3
    ax_charge.set_ylim(-cmax, cmax)

    # --- Update function ---
    def update(val):
        t = slider.val
        step = int(t / dt)
        step = min(step, nsteps - 1)

        line_normal.set_data(x, all_ne_normal[step])
        line_u.set_data(x, all_ne[step])
        line_v.set_data(x, all_nh[step])
        line_charge.set_data(x, all_charge[step])

        # Update fills
        nonlocal fill_u, fill_v
        fill_u.remove()
        fill_v.remove()
        fill_u = ax_u.fill_between(x, 0, all_ne[step], color='blue', alpha=0.2)
        fill_v = ax_v.fill_between(x, 0, all_nh[step], color='red', alpha=0.2)

        fig.canvas.draw_idle()

    slider.on_changed(update)

    # --- Play button ---
    ax_play = fig.add_axes([0.87, 0.02, 0.08, 0.03])
    play_button = Button(ax_play, 'Play')

    playing = [False]
    play_idx = [0]

    def play(event):
        if playing[0]:
            playing[0] = False
            play_button.label.set_text('Play')
        else:
            playing[0] = True
            play_button.label.set_text('Stop')
            play_idx[0] = int(slider.val / dt)
            def play_loop():
                if not playing[0]:
                    return
                play_idx[0] = (play_idx[0] + 1) % nsteps
                slider.set_val(play_idx[0] * dt)
                fig.canvas.draw_idle()
                if playing[0]:
                    fig.canvas.after(30, play_loop)  # ~30ms per frame
            # matplotlib doesn't have .after in all backends; use timer instead
            timer = fig.canvas.new_timer(interval=30)
            def on_timer():
                if not playing[0]:
                    timer.stop()
                    return
                play_idx[0] = (play_idx[0] + 1) % nsteps
                slider.set_val(play_idx[0] * dt)
            timer.add_callback(on_timer)
            timer.start()

    play_button.on_clicked(play)

    # Initial draw
    update(0)

    return fig, slider, all_ne, all_nh, all_charge, all_ne_normal, dt, nsteps

# ============================================================================
# Animation version (FuncAnimation, for saving to GIF)
# ============================================================================

def make_animation(N=256, t=1.0, mu=0.0, Delta=0.15, k0=None,
                   sigma_k=0.08, nsteps=400, dt=0.3, save_path=None):
    """
    FuncAnimation version of the superconducting wavepacket.
    Saves to GIF if save_path is given.
    """
    if k0 is None:
        k0 = np.pi / 2 + 0.25
    x0 = N // 4

    H = build_bdg_periodic(N, t, mu, Delta)
    E, V = np.linalg.eigh(H)
    pos_mask = E > 0
    E_pos = E[pos_mask]
    V_pos = V[:, pos_mask]

    k_of_state = np.zeros(len(E_pos))
    for n in range(len(E_pos)):
        u = V_pos[:N, n]
        fft_u = np.fft.fftshift(np.abs(np.fft.fft(u)))
        k_idx = np.argmax(fft_u)
        k_of_state[n] = np.linspace(-np.pi, np.pi, N, endpoint=False)[k_idx]

    weights = np.exp(-(k_of_state - k0)**2 / (2 * sigma_k**2))
    weights /= np.sqrt(np.sum(weights**2))

    # Normal state
    H_n = build_bdg_periodic(N, t, mu, 0.0)
    E_n, V_n = np.linalg.eigh(H_n)
    pos_n = E_n > 0
    E_n_pos = E_n[pos_n]
    V_n_pos = V_n[:, pos_n]

    k_of_normal = np.zeros(len(E_n_pos))
    for n in range(len(E_n_pos)):
        u = V_n_pos[:N, n]
        fft_u = np.fft.fftshift(np.abs(np.fft.fft(u)))
        k_idx = np.argmax(fft_u)
        k_of_normal[n] = np.linspace(-np.pi, np.pi, N, endpoint=False)[k_idx]

    weights_n = np.exp(-(k_of_normal - k0)**2 / (2 * sigma_k**2))
    weights_n /= np.sqrt(np.sum(weights_n**2))

    x = np.arange(N)
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    line_n, = axes[0].plot([], [], 'g-', lw=1.5)
    line_u, = axes[1].plot([], [], 'b-', lw=1.8)
    line_v, = axes[2].plot([], [], 'r-', lw=1.8)
    line_c, = axes[3].plot([], [], 'k-', lw=1.5)

    for ax in axes:
        ax.axhline(0, color='gray', lw=0.3)
        ax.set_xlim(0, N)
    axes[0].set_ylim(-0.01, 0.06)
    axes[1].set_ylim(-0.01, 0.06)
    axes[2].set_ylim(-0.01, 0.06)
    axes[3].set_ylim(-0.05, 0.05)
    axes[0].set_ylabel(r"$|\psi|^2$")
    axes[1].set_ylabel(r"$|u_i|^2$")
    axes[2].set_ylabel(r"$|v_i|^2$")
    axes[3].set_ylabel(r"$|u|^2\!-\!|v|^2$")
    axes[3].set_xlabel("site $i$")

    axes[0].set_title(r"Normal ($\Delta$=0): electron packet — full Fermi velocity", fontsize=9, loc='left')
    axes[1].set_title(r"SC: electron $|u_i(t)|^2$", fontsize=9, loc='left')
    axes[2].set_title(r"SC: hole $|v_i(t)|^2$ (Andreev oscillation, $T=\pi/\Delta$"+f"={np.pi/Delta:.1f})", fontsize=9, loc='left')
    axes[3].set_title(r"Net charge (nearly neutral)", fontsize=9, loc='left')

    fig.suptitle(f"Superconducting Wavepacket ($\\Delta$={Delta}, $k_0$={k0:.2f})\n"
                 r"Gap-edge BdG states superposed → moving electron-hole wavepacket",
                 fontsize=12)

    def init():
        for ln in [line_n, line_u, line_v, line_c]:
            ln.set_data([], [])
        return line_n, line_u, line_v, line_c

    def update(frame):
        time = frame * dt
        # BdG
        phase = np.exp(-1j * E_pos * time)
        psi = V_pos @ (phase * weights)
        u, v = psi[:N], psi[N:]
        ne, nh = np.abs(u)**2, np.abs(v)**2
        # Normal
        phase_n = np.exp(-1j * E_n_pos * time)
        psi_n = V_n_pos @ (phase_n * weights_n)
        ne_n = np.abs(psi_n[:N])**2

        line_n.set_data(x, ne_n)
        line_u.set_data(x, ne)
        line_v.set_data(x, nh)
        line_c.set_data(x, ne - nh)
        axes[0].set_title(f"Normal ($\\Delta$=0): electron packet   t={time:.1f}", fontsize=9, loc='left')
        return line_n, line_u, line_v, line_c

    anim = FuncAnimation(fig, update, frames=nsteps, init_func=init, blit=True, interval=30)

    if save_path:
        anim.save(save_path, writer='pillow', fps=30, dpi=100)
        print(f"Saved animation to {save_path}")

    return fig, anim

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Superconducting gap & wavepacket visualization")
    parser.add_argument('--save', type=str, default=None, help="Save animation to file (e.g. movie.gif)")
    parser.add_argument('--static-only', action='store_true', help="Only show static figures")
    parser.add_argument('--Delta', type=float, default=0.15, help="Superconducting gap parameter")
    parser.add_argument('--N', type=int, default=256, help="Number of lattice sites")
    args = parser.parse_args()

    Delta = args.Delta
    N = args.N

    # --- Static Figure 1: Gap opening ---
    print("[1] Plotting gap opening (normal vs BdG spectrum)...")
    fig1 = plot_gap_opening(Delta=Delta)
    fig1.savefig("sc_gap_opening.png", dpi=150, bbox_inches='tight')
    print("    Saved sc_gap_opening.png")

    # --- Static Figure 2: Gap-edge eigenstates ---
    print("[2] Plotting BdG eigenstates near gap edge...")
    fig2 = plot_gap_edge_states(N=N, Delta=Delta)
    fig2.savefig("sc_gap_edge_states.png", dpi=150, bbox_inches='tight')
    print("    Saved sc_gap_edge_states.png")

    if not args.static_only:
        # --- Interactive animation with slider ---
        print("[3] Building interactive wavepacket animation...")
        fig3, slider, ne, nh, charge, ne_n, dt, nsteps = animate_sc_wavepacket(
            N=N, Delta=Delta, nsteps=400, dt=0.3)
        print("    Interactive window open — use the time slider or Play button.")

        if args.save:
            print(f"[4] Saving animation to {args.save}...")
            # Use FuncAnimation version for saving
            fig4, anim = make_animation(N=N, Delta=Delta, nsteps=400, dt=0.3, save_path=args.save)

    plt.show()

if __name__ == '__main__':
    main()
