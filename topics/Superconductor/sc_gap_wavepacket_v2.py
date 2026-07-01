"""
Superconducting Gap & Wavepacket Visualization on a 1D Tight-Binding Chain
==========================================================================

This script visualizes how superconductivity modifies the electronic structure
of a 1D crystal and how a pure electron wavepacket evolves into a mixed
electron-hole (Bogoliubov) wavepacket when the gap is open.

PHYSICAL SETUP
--------------
We consider a 1D tight-binding chain (the simplest model of a crystal):

    H_0 = -t Σ_i (c†_{i+1} c_i + h.c.) - μ Σ_i c†_i c_i

In k-space (periodic chain), the normal-state dispersion is:

    ξ_k = -2t cos(k) - μ        (energy measured from Fermi level)

At half-filling (μ=0), the Fermi points are at k_F = ±π/2, where ξ_k = 0.

SUPERCONDUCTING PAIRING (BCS/BdG)
---------------------------------
We add an on-site s-wave pairing gap Δ (from electron-phonon coupling
mediated attraction, treated at mean-field level). The Bogoliubov–de Gennes
(BdG) Hamiltonian in Nambu space Ψ = (c_{k↑}, c†_{-k↓}) is:

    H_BdG(k) = [ ξ_k    Δ  ]    →    eigenvalues: E_k = ±√(ξ_k² + Δ²)
               [ Δ*   -ξ_k ]

KEY PHYSICS:
1. GAP OPENING: The normal dispersion ξ_k is "folded" — positive and negative
   branches repel each other due to the off-diagonal Δ coupling. The result
   is E_k = √(ξ_k² + Δ²), which has a minimum gap of 2Δ at the Fermi points.
   The normal states |k↑⟩ and |-k↓⟩ (which were degenerate at the Fermi level)
   are now mixed into two new eigenstates separated by 2Δ.

2. ELECTRON-HOLE MIXING: Each BdG eigenstate is a spinor (u_k, v_k):
       u_k² = ½(1 + ξ_k/E_k)    (electron character)
       v_k² = ½(1 - ξ_k/E_k)    (hole character)
   At the gap edge (ξ_k=0): u_k² = v_k² = ½  →  50/50 electron-hole mix.
   Far from the gap (|ξ_k| >> Δ): either u≈1 (pure electron) or v≈1 (pure hole).

3. GROUP VELOCITY: v_g = dE_k/dk = (ξ_k/E_k) · dξ_k/dk
   At the gap edge: v_g → 0. The quasiparticle is "heavy" — it barely moves.
   This is why superconducting excitations near the gap are slow.
   Away from the gap edge, v_g approaches the normal-state velocity but is
   always reduced by the factor ξ_k/E_k < 1.

4. ANDREEV OSCILLATION: If you inject a PURE ELECTRON at the Fermi point
   (where ξ_k=0), the BdG Hamiltonian acts as a 2-level system:
       H_k = Δ τ_x   (Pauli x in Nambu space)
   The time evolution of an initial pure electron state (1, 0) is:
       u_k(t) = cos(Δt),   v_k(t) = -i sin(Δt)
   So the electron periodically converts into a hole and back!
   Period: T_Andreev = π/Δ. This is the real-space signature of the gap.

   IMPORTANT: To see this oscillation, we must inject a PURE ELECTRON
   (v=0 initially) and expand it in the FULL BdG basis (both positive
   and negative energy eigenstates). A pure electron at the Fermi point
   is an EQUAL superposition of the +E_k and -E_k BdG eigenstates.
   The relative phase between these two components oscillates at frequency
   2E_k/ℏ = 2Δ/ℏ, producing the electron↔hole oscillation.

WAVEPACKET CONSTRUCTION
-----------------------
Following the recipe from the document:
  1. Diagonalize the FULL BdG Hamiltonian (2N × 2N) → E_n, V_n
  2. Construct a PURE ELECTRON wavepacket in real space:
       u_i(0) = Gaussian × exp(i k_0 x),   v_i(0) = 0
  3. Expand this initial state in the full BdG eigenbasis:
       c_n = V_n† Ψ(0)
  4. Time evolve each component:
       Ψ(t) = Σ_n c_n V_n exp(-i E_n t)
  5. Extract u_i(t) = Ψ_i(t), v_i(t) = Ψ_{i+N}(t)
  6. Plot |u_i(t)|² (electron density), |v_i(t)|² (hole density),
     and |u|²-|v|² (net charge density).

WHAT YOU SEE IN THE ANIMATION
-----------------------------
- |u(x,t)|²:  electron component of the Bogoliubov quasiparticle
              (starts as a Gaussian packet, oscillates in amplitude)
- |v(x,t)|²:  hole component (starts at ZERO, grows via Andreev oscillation,
              reaches maximum at t=π/(2Δ), back to zero at t=π/Δ)
- |u|²-|v|²:  net charge density — oscillates but is always smaller than
              the individual components. At the gap edge, the time-averaged
              charge is zero: the excitation is NEUTRAL on average.

- For comparison, the top panel shows the SAME initial wavepacket evolving
  on a NORMAL chain (Δ=0): it simply propagates as a Gaussian, no hole
  component appears, no oscillation.

Run:  python sc_gap_wavepacket_v2.py
      python sc_gap_wavepacket_v2.py --static-only
      python sc_gap_wavepacket_v2.py --save movie.gif
      python sc_gap_wavepacket_v2.py --Delta 0.3 --N 256
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

        H_BdG = [ H_0       Δ·I  ]    (electron block: H_0,  pairing: Δ)
                [ Δ·I    -H_0*   ]    (hole block: -H_0*,  pairing: Δ*)

    The electron block H_0 describes the normal tight-binding Hamiltonian.
    The hole block -H_0* is the time-reversed partner (holes have opposite
    charge, so their energy is negated).
    The off-diagonal Δ·I couples electrons and holes via Cooper pairing.

    Returns the 2N×2N complex matrix.
    """
    H0 = np.zeros((N, N), dtype=complex)
    for i in range(N):
        H0[i, i] = -mu
        H0[i, (i + 1) % N] = -t  # forward hop with periodic BC
        H0[i, (i - 1) % N] = -t  # backward hop
    H = np.zeros((2 * N, 2 * N), dtype=complex)
    H[:N, :N] = H0                    # electron block
    H[N:, N:] = -H0.conjugate()       # hole block = -H_0* (time-reversed)
    H[:N, N:] = Delta * np.eye(N)     # pairing: Δ c†_i↑ c†_i↓ + h.c.
    H[N:, :N] = Delta * np.eye(N)
    return H

# ============================================================================
# Static Figure 1: Gap Opening — Normal vs BdG Spectrum
# ============================================================================

def plot_gap_opening(t=1.0, mu=0.0, Delta=0.15):
    """
    Show how the superconducting gap opens in the BdG spectrum.

    Left panel:  Normal dispersion ξ_k = -2t cos(k) - μ (dashed green)
                 BdG dispersion    E_k = √(ξ_k² + Δ²) (solid blue)
                 The gap 2Δ opens at the Fermi points k_F = ±π/2.
                 The normal states cross zero at k_F; the BdG states
                 "repel" each other, opening a gap of exactly 2Δ.

                 This is analogous to how a perturbation opens a gap
                 at an avoided crossing in molecular orbital theory:
                 the "perturbation" here is the Cooper pairing Δ.

    Right panel: Electron/hole weights u_k² and v_k² vs k.
                 These are the BdG coherence factors:
                   u_k² = ½(1 + ξ_k/E_k)  → electron fraction
                   v_k² = ½(1 - ξ_k/E_k)  → hole fraction
                 At gap edge (ξ_k=0): u² = v² = ½ (maximum mixing).
                 Far from gap: pure electron (u²→1) or pure hole (v²→1).

                 u_k² and v_k² tell you the PROBABILITY of finding the
                 BdG quasiparticle as an electron vs a hole if you measure
                 it in the normal-state basis.
    """
    k = np.linspace(-np.pi, np.pi, 1000)
    xi = -2 * t * np.cos(k) - mu
    E = np.sqrt(xi**2 + Delta**2)

    # BdG coherence factors (electron/hole weights)
    # u_k² = ½(1 + ξ_k/E_k),  v_k² = ½(1 - ξ_k/E_k)
    # These come from diagonalizing the 2×2 BdG Hamiltonian:
    #   H_BdG(k) = ξ_k τ_z + Δ τ_x
    # Eigenstates: (cos(θ/2), sin(θ/2)) with tan(θ) = Δ/ξ_k
    # → u² = cos²(θ/2) = ½(1+cos θ) = ½(1+ξ/E)
    # → v² = sin²(θ/2) = ½(1-cos θ) = ½(1-ξ/E)
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
    ax1.set_title("Superconducting Gap Opening\n"
                  "(analogous to avoided crossing in MO theory)", fontsize=11)
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
    ax2.set_title("BdG Coherence Factors: Electron–Hole Mixing\n"
                  "(probability of finding electron vs hole character)", fontsize=11)
    ax2.legend(fontsize=8, loc='center')
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    return fig

# ============================================================================
# Static Figure 2: How Normal States Mix into BdG Quasiparticles
# ============================================================================

def plot_state_mixing(N=128, t=1.0, mu=0.0, Delta=0.15, n_show=5):
    """
    Show how normal-state eigenstates near the Fermi point get mixed
    by the pairing gap Δ into BdG quasiparticles.

    For each k near k_F, the normal electron state |k⟩ and hole state |-k⟩
    are mixed by Δ into two BdG eigenstates:
      |BdG,+k⟩ = u_k |k,e⟩ + v_k |-k,h⟩    (energy +E_k)
      |BdG,-k⟩ = -v_k |k,e⟩ + u_k |-k,h⟩   (energy -E_k)

    We show this by:
    1. Diagonalizing both the normal (Δ=0) and BdG (Δ>0) Hamiltonians
    2. For each BdG eigenstate near the gap edge, computing its overlap
       with the corresponding normal eigenstates
    3. Plotting the electron fraction |u|² and hole fraction |v|²

    This directly shows the "mixing" that opens the gap: the normal states
    that used to cross at E=0 are now split by 2Δ and each carries both
    electron and hole character.
    """
    H_normal = build_bdg_periodic(N, t, mu, Delta=0.0)
    E_n, V_n = np.linalg.eigh(H_normal)

    H_bdg = build_bdg_periodic(N, t, mu, Delta)
    E_b, V_b = np.linalg.eigh(H_bdg)

    x = np.arange(N)

    # Find normal eigenstates near E=0 (Fermi point)
    pos_n = E_n > 0
    E_n_pos = E_n[pos_n]
    V_n_pos = V_n[:, pos_n]
    # Sort by energy (closest to 0 first)
    order_n = np.argsort(E_n_pos)
    normal_idx = order_n[:n_show]

    # Find BdG eigenstates near E=+Delta (gap edge)
    pos_b = E_b > 0
    E_b_pos = E_b[pos_b]
    V_b_pos = V_b[:, pos_b]
    order_b = np.argsort(np.abs(E_b_pos - Delta))
    bdg_idx = order_b[:n_show]

    fig, axes = plt.subplots(n_show, 3, figsize=(14, 2.5 * n_show), squeeze=False)

    for row in range(n_show):
        ni = normal_idx[row]
        bi = bdg_idx[row]

        # Normal state: pure electron (v=0)
        u_n = V_n_pos[:N, ni]
        v_n = V_n_pos[N:, ni]
        E_n_val = E_n_pos[ni]

        # BdG state: mixed electron-hole
        u_b = V_b_pos[:N, bi]
        v_b = V_b_pos[N:, bi]
        E_b_val = E_b_pos[bi]

        # Col 0: Normal state (electron only)
        axes[row, 0].plot(x, np.abs(u_n)**2, 'b-', lw=1.2, label=r"$|u_i|^2$ (electron)")
        axes[row, 0].plot(x, np.abs(v_n)**2, 'r-', lw=1.2, label=r"$|v_i|^2$ (hole)")
        axes[row, 0].set_ylabel(f"$E_n$={E_n_val:.3f}")
        if row == 0:
            axes[row, 0].set_title("Normal ($\\Delta$=0)\nPure electron state", fontsize=10)
        axes[row, 0].legend(fontsize=7)

        # Col 1: BdG state (mixed)
        axes[row, 1].plot(x, np.abs(u_b)**2, 'b-', lw=1.2, label=r"$|u_i|^2$ (electron)")
        axes[row, 1].plot(x, np.abs(v_b)**2, 'r-', lw=1.2, label=r"$|v_i|^2$ (hole)")
        frac_e = np.sum(np.abs(u_b)**2) / (np.sum(np.abs(u_b)**2) + np.sum(np.abs(v_b)**2))
        axes[row, 1].set_ylabel(f"$E_{{BdG}}$={E_b_val:.3f}")
        axes[row, 1].text(0.02, 0.95, f"e-frac: {frac_e:.2f}",
                          transform=axes[row, 1].transAxes, fontsize=8, va='top')
        if row == 0:
            axes[row, 1].set_title(f"BdG ($\\Delta$={Delta})\nMixed electron-hole state", fontsize=10)
        axes[row, 1].legend(fontsize=7)

        # Col 2: Real-space wavefunction (Re part) — shows the wave character
        axes[row, 2].plot(x, u_b.real, 'b-', lw=0.8, alpha=0.7, label=r"Re $u_i$")
        axes[row, 2].plot(x, v_b.real, 'r-', lw=0.8, alpha=0.7, label=r"Re $v_i$")
        if row == 0:
            axes[row, 2].set_title("Real-space wavefunction\n(Re part — shows Bloch character)", fontsize=10)
        axes[row, 2].legend(fontsize=7)

    axes[-1, 0].set_xlabel("site $i$")
    axes[-1, 1].set_xlabel("site $i$")
    axes[-1, 2].set_xlabel("site $i$")

    fig.suptitle("How Normal States Mix into BdG Quasiparticles\n"
                 "Left: pure electron (Δ=0) → Middle: mixed electron-hole (Δ>0)\n"
                 r"At gap edge: $|u|^2 \approx |v|^2 = \frac{1}{2}$ (maximum mixing)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    return fig

# ============================================================================
# Static Figure 3: BdG Eigenstates Near Gap Edge in Real Space
# ============================================================================

def plot_gap_edge_states(N=128, t=1.0, mu=0.0, Delta=0.15, n_states=4):
    """
    Show the real-space structure of BdG eigenstates near the positive gap edge.

    Each BdG eigenstate is a spinor (u_i, v_i) on each site i.
    Near the gap edge (E ≈ +Δ), |u_i|² ≈ |v_i|² — the state is an equal
    superposition of electron and hole amplitudes.

    The eigenstates are Bloch-like (plane waves with definite k),
    so |u_i|² and |v_i|² are spatially uniform for a translationally
    invariant chain. The real-space structure becomes visible when we
    build a wavepacket (next figure/animation).

    The Re(u_i) and Re(v_i) show the underlying Bloch wave character:
    they oscillate as cos(k·i) with a definite wavelength λ = 2π/k.
    Near the gap edge, k ≈ k_F = π/2, so λ ≈ 4 lattice spacings.
    """
    H = build_bdg_periodic(N, t, mu, Delta)
    E, V = np.linalg.eigh(H)
    x = np.arange(N)

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

        # Real part of u and v (showing the Bloch wave character)
        axes[row, 0].plot(x, u.real, 'b-', lw=1, label=r"$\mathrm{Re}\,u_i$ (electron)")
        axes[row, 0].plot(x, v.real, 'r-', lw=1, label=r"$\mathrm{Re}\,v_i$ (hole)")
        axes[row, 0].set_ylabel(f"E={E_val:.4f}")
        axes[row, 0].legend(fontsize=7, loc='upper right')
        if row == 0:
            axes[row, 0].set_title("Real-space amplitude (Re part)\nShows Bloch wave character", fontsize=10)

        # Densities |u|² and |v|²
        axes[row, 1].plot(x, np.abs(u)**2, 'b-', lw=1.5, label=r"$|u_i|^2$ (electron)")
        axes[row, 1].plot(x, np.abs(v)**2, 'r-', lw=1.5, label=r"$|v_i|^2$ (hole)")
        axes[row, 1].legend(fontsize=7, loc='upper right')
        if row == 0:
            axes[row, 1].set_title(r"Density $|u|^2$, $|v|^2$ (uniform for Bloch states)", fontsize=10)

        # Annotate electron/hole fraction
        frac_e = np.sum(np.abs(u)**2) / (np.sum(np.abs(u)**2) + np.sum(np.abs(v)**2))
        axes[row, 1].text(0.02, 0.95, f"electron frac: {frac_e:.2f}",
                          transform=axes[row, 1].transAxes, fontsize=8, va='top')

    axes[-1, 0].set_xlabel("site $i$")
    axes[-1, 1].set_xlabel("site $i$")
    fig.suptitle(f"BdG Eigenstates Near Gap Edge ($E \\approx \\Delta$={Delta})\n"
                 f"Note: $|u|^2 \\approx |v|^2$ → equal electron-hole mixing at gap edge",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    return fig

# ============================================================================
# Wavepacket: Pure Electron Injected into BdG System
# ============================================================================
#
# THE KEY PHYSICS:
# A pure electron state |ψ_e⟩ = (u_i, 0) is NOT an eigenstate of the BdG
# Hamiltonian when Δ > 0. It is a superposition of +E_k and -E_k BdG
# eigenstates:
#
#   |k, electron⟩ = u_k |BdG,+k⟩ - v_k |BdG,-k⟩
#
# where |BdG,±k⟩ are the BdG eigenstates with energies ±E_k.
# The time evolution then produces:
#
#   |ψ(t)⟩ = u_k e^{-iE_k t} |BdG,+k⟩ - v_k e^{+iE_k t} |BdG,-k⟩
#
# The relative phase between the +E_k and -E_k components oscillates
# at frequency 2E_k, causing the electron character to oscillate into
# hole character and back. At the gap edge (E_k=Δ), this gives the
# clean Andreev oscillation: u(t)=cos(Δt), v(t)=-i sin(Δt).
#
# For a wavepacket (superposition of k-values), each k-component oscillates
# at its own frequency 2E_k, and the packet also propagates with the
# BdG group velocity v_g(k) = (ξ_k/E_k) dξ_k/dk.

def build_wavepacket_data(N=256, t=1.0, mu=0.0, Delta=0.15,
                          k0=None, sigma_x=15.0, x0=None,
                          nsteps=400, dt=0.3):
    """
    Precompute all animation frames for the superconducting wavepacket.

    CONSTRUCTION:
    1. Build and diagonalize the FULL BdG Hamiltonian (2N × 2N) → E_n, V_n
    2. Construct a PURE ELECTRON wavepacket in real space:
         u_i(0) = exp(-(i-x0)²/(2σ_x²)) × exp(i k0 i),   v_i(0) = 0
       This is a Gaussian-enveloped plane wave — a localized electron
       with central momentum k0.
    3. Expand in the full BdG eigenbasis: c_n = V_n† Ψ(0)
       (This projects the pure electron onto BOTH +E and -E BdG states)
    4. Time evolve: Ψ(t) = V @ (exp(-i E t) ⊙ c)
    5. Extract u_i(t) = Ψ_i(t), v_i(t) = Ψ_{i+N}(t)

    Also computes the NORMAL (Δ=0) wavepacket for comparison:
    same initial state, but no hole component ever appears.

    Returns: dict with all frames + metadata
    """
    if k0 is None:
        k0 = np.pi / 2 + 0.25  # slightly above k_F for nonzero group velocity
    if x0 is None:
        x0 = N // 4

    x = np.arange(N)

    # --- Build initial PURE ELECTRON wavepacket ---
    # u_i = Gaussian envelope × plane wave with momentum k0
    # v_i = 0 (no hole component initially)
    u0 = np.exp(-(x - x0)**2 / (2 * sigma_x**2)) * np.exp(1j * k0 * x)
    u0 /= np.sqrt(np.sum(np.abs(u0)**2))  # normalize
    v0 = np.zeros(N, dtype=complex)
    psi0 = np.concatenate([u0, v0])  # shape (2N,)

    # --- Diagonalize BdG ---
    H_bdg = build_bdg_periodic(N, t, mu, Delta)
    E_bdg, V_bdg = np.linalg.eigh(H_bdg)

    # Expand initial state in BdG eigenbasis
    coeffs_bdg = V_bdg.conjugate().T @ psi0  # c_n = ⟨V_n|Ψ(0)⟩

    # --- Diagonalize normal (Δ=0) for comparison ---
    H_normal = build_bdg_periodic(N, t, mu, Delta=0.0)
    E_normal, V_normal = np.linalg.eigh(H_normal)
    coeffs_normal = V_normal.conjugate().T @ psi0

    # --- Precompute frames ---
    all_ne = np.zeros((nsteps, N))
    all_nh = np.zeros((nsteps, N))
    all_charge = np.zeros((nsteps, N))
    all_ne_normal = np.zeros((nsteps, N))

    for step in range(nsteps):
        time = step * dt

        # BdG time evolution: Ψ(t) = V @ (exp(-iEt) ⊙ c)
        phase = np.exp(-1j * E_bdg * time)
        psi = V_bdg @ (phase * coeffs_bdg)
        u = psi[:N]
        v = psi[N:]
        all_ne[step] = np.abs(u)**2
        all_nh[step] = np.abs(v)**2
        all_charge[step] = all_ne[step] - all_nh[step]

        # Normal time evolution (for comparison)
        phase_n = np.exp(-1j * E_normal * time)
        psi_n = V_normal @ (phase_n * coeffs_normal)
        all_ne_normal[step] = np.abs(psi_n[:N])**2

    return {
        'N': N, 'x': x, 'dt': dt, 'nsteps': nsteps,
        'Delta': Delta, 'k0': k0, 'sigma_x': sigma_x, 'x0': x0,
        'all_ne': all_ne, 'all_nh': all_nh,
        'all_charge': all_charge, 'all_ne_normal': all_ne_normal,
        'E_bdg': E_bdg, 'V_bdg': V_bdg,
        'E_normal': E_normal, 'V_normal': V_normal,
    }

# ============================================================================
# Interactive Animation with Time Slider
# ============================================================================

def animate_sc_wavepacket(data):
    """
    Build interactive figure with time slider for the SC wavepacket.

    4 panels (top to bottom):
    1. Normal (Δ=0): |ψ_e(x,t)|² — pure electron, propagates freely
    2. Superconducting: |u_i(t)|² — electron component (oscillates)
    3. Superconducting: |v_i(t)|² — hole component (grows from zero!)
    4. Net charge: |u|²-|v|² — much smaller than either component

    The time slider lets you scrub through the evolution.
    The Play button auto-advances time.
    """
    N = data['N']
    x = data['x']
    dt = data['dt']
    nsteps = data['nsteps']
    Delta = data['Delta']
    k0 = data['k0']
    all_ne = data['all_ne']
    all_nh = data['all_nh']
    all_charge = data['all_charge']
    all_ne_normal = data['all_ne_normal']

    fig = plt.figure(figsize=(13, 9))
    gs = GridSpec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.4, top=0.90, bottom=0.18)

    ax_normal = fig.add_subplot(gs[0])
    ax_u = fig.add_subplot(gs[1], sharex=ax_normal)
    ax_v = fig.add_subplot(gs[2], sharex=ax_normal)
    ax_charge = fig.add_subplot(gs[3], sharex=ax_normal)

    line_normal, = ax_normal.plot([], [], 'g-', lw=1.5)
    line_u, = ax_u.plot([], [], 'b-', lw=1.8)
    line_v, = ax_v.plot([], [], 'r-', lw=1.8)
    line_charge, = ax_charge.plot([], [], 'k-', lw=1.5)

    # Fill areas for visual impact
    fill_u = ax_u.fill_between(x, 0, 0, color='blue', alpha=0.2)
    fill_v = ax_v.fill_between(x, 0, 0, color='red', alpha=0.2)

    # Auto-scale y limits
    ymax = max(all_ne.max(), all_nh.max(), all_ne_normal.max()) * 1.15
    cmax = max(abs(all_charge.min()), abs(all_charge.max())) * 1.3

    for ax in [ax_normal, ax_u, ax_v]:
        ax.set_ylim(-0.01, ymax)
        ax.set_xlim(0, N)
        ax.axhline(0, color='gray', lw=0.3)
    ax_charge.set_ylim(-cmax, cmax)
    ax_charge.set_xlim(0, N)
    ax_charge.axhline(0, color='gray', lw=0.3)

    ax_normal.set_ylabel(r"$|\psi_e|^2$")
    ax_normal.set_title(
        r"Normal chain ($\Delta$=0): pure electron wavepacket — propagates at Fermi velocity, no hole component",
        fontsize=9, loc='left')

    ax_u.set_ylabel(r"$|u_i|^2$")
    ax_u.set_title(
        r"Superconducting ($\Delta$=%.2f): electron component $|u_i(t)|^2$ — oscillates due to Andreev physics" % Delta,
        fontsize=9, loc='left')

    ax_v.set_ylabel(r"$|v_i|^2$")
    ax_v.set_title(
        r"Hole component $|v_i(t)|^2$ — starts at ZERO, grows as electron converts to hole",
        fontsize=9, loc='left')

    ax_charge.set_ylabel(r"$|u|^2\!-\!|v|^2$")
    ax_charge.set_title(
        r"Net charge density — much smaller than either $|u|^2$ or $|v|^2$ → nearly neutral excitation",
        fontsize=9, loc='left')
    ax_charge.set_xlabel("site $i$")

    T_andreev = np.pi / Delta
    fig.suptitle(
        "Superconducting Wavepacket: Pure Electron Injected into BdG System\n"
        r"$E_k=\sqrt{\xi_k^2+\Delta^2}$,  Andreev period $T=\pi/\Delta$"
        f" = {T_andreev:.1f},  $k_0$={k0:.2f}",
        fontsize=12)

    # --- Time slider ---
    ax_slider = fig.add_axes([0.15, 0.04, 0.7, 0.025])
    t_max = nsteps * dt
    slider = Slider(ax_slider, 'time', 0, t_max - dt, valinit=0, valfmt='%.1f')

    # --- Play button ---
    ax_play = fig.add_axes([0.87, 0.04, 0.08, 0.025])
    play_button = Button(ax_play, 'Play')

    playing = [False]

    def update(val):
        t = slider.val
        step = int(round(t / dt))
        step = min(max(step, 0), nsteps - 1)

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

    def play(event):
        if playing[0]:
            playing[0] = False
            play_button.label.set_text('Play')
        else:
            playing[0] = True
            play_button.label.set_text('Stop')
            timer = fig.canvas.new_timer(interval=30)

            def on_timer():
                if not playing[0]:
                    timer.stop()
                    return
                t_cur = slider.val
                t_new = t_cur + dt
                if t_new >= t_max - dt:
                    t_new = 0
                slider.set_val(t_new)

            timer.add_callback(on_timer)
            timer.start()

    play_button.on_clicked(play)

    update(0)
    return fig, slider

# ============================================================================
# FuncAnimation version (for saving to GIF)
# ============================================================================

def make_animation(data, save_path=None):
    """
    FuncAnimation version of the superconducting wavepacket.
    Saves to GIF if save_path is given.
    """
    N = data['N']
    x = data['x']
    dt = data['dt']
    nsteps = data['nsteps']
    Delta = data['Delta']
    k0 = data['k0']
    all_ne = data['all_ne']
    all_nh = data['all_nh']
    all_charge = data['all_charge']
    all_ne_normal = data['all_ne_normal']

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    line_n, = axes[0].plot([], [], 'g-', lw=1.5)
    line_u, = axes[1].plot([], [], 'b-', lw=1.8)
    line_v, = axes[2].plot([], [], 'r-', lw=1.8)
    line_c, = axes[3].plot([], [], 'k-', lw=1.5)

    ymax = max(all_ne.max(), all_nh.max(), all_ne_normal.max()) * 1.15
    cmax = max(abs(all_charge.min()), abs(all_charge.max())) * 1.3

    for ax in axes[:3]:
        ax.axhline(0, color='gray', lw=0.3)
        ax.set_xlim(0, N)
        ax.set_ylim(-0.01, ymax)
    axes[3].axhline(0, color='gray', lw=0.3)
    axes[3].set_xlim(0, N)
    axes[3].set_ylim(-cmax, cmax)

    axes[0].set_ylabel(r"$|\psi_e|^2$")
    axes[1].set_ylabel(r"$|u_i|^2$")
    axes[2].set_ylabel(r"$|v_i|^2$")
    axes[3].set_ylabel(r"$|u|^2\!-\!|v|^2$")
    axes[3].set_xlabel("site $i$")

    T_andreev = np.pi / Delta
    axes[0].set_title(r"Normal ($\Delta$=0): electron packet — full Fermi velocity", fontsize=9, loc='left')
    axes[1].set_title(r"SC: electron $|u_i(t)|^2$ (oscillates)", fontsize=9, loc='left')
    axes[2].set_title(r"SC: hole $|v_i(t)|^2$ (Andreev oscillation, $T=\pi/\Delta$"+f"={T_andreev:.1f})", fontsize=9, loc='left')
    axes[3].set_title(r"Net charge (nearly neutral)", fontsize=9, loc='left')

    fig.suptitle(f"Superconducting Wavepacket ($\\Delta$={Delta}, $k_0$={k0:.2f})\n"
                 r"Pure electron injected → electron-hole oscillation + propagation",
                 fontsize=12)

    def init():
        for ln in [line_n, line_u, line_v, line_c]:
            ln.set_data([], [])
        return line_n, line_u, line_v, line_c

    def update(frame):
        time = frame * dt
        line_n.set_data(x, all_ne_normal[frame])
        line_u.set_data(x, all_ne[frame])
        line_v.set_data(x, all_nh[frame])
        line_c.set_data(x, all_charge[frame])
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
    parser.add_argument('--k0', type=float, default=None, help="Central wavevector (default: k_F + 0.25)")
    parser.add_argument('--sigma', type=float, default=15.0, help="Wavepacket spatial width")
    args = parser.parse_args()

    Delta = args.Delta
    N = args.N
    k0 = args.k0
    sigma = args.sigma

    if k0 is None:
        k0 = np.pi / 2 + 0.25

    # --- Static Figure 1: Gap opening ---
    print("[1] Plotting gap opening (normal vs BdG spectrum)...")
    fig1 = plot_gap_opening(Delta=Delta)
    fig1.savefig("sc_gap_opening.png", dpi=150, bbox_inches='tight')
    print("    Saved sc_gap_opening.png")

    # --- Static Figure 2: State mixing ---
    print("[2] Plotting how normal states mix into BdG quasiparticles...")
    fig2 = plot_state_mixing(N=N, Delta=Delta)
    fig2.savefig("sc_state_mixing.png", dpi=150, bbox_inches='tight')
    print("    Saved sc_state_mixing.png")

    # --- Static Figure 3: Gap-edge eigenstates ---
    print("[3] Plotting BdG eigenstates near gap edge...")
    fig3 = plot_gap_edge_states(N=N, Delta=Delta)
    fig3.savefig("sc_gap_edge_states.png", dpi=150, bbox_inches='tight')
    print("    Saved sc_gap_edge_states.png")

    if not args.static_only:
        # --- Build wavepacket data ---
        print(f"[4] Building wavepacket data (N={N}, Δ={Delta}, k₀={k0:.2f})...")
        data = build_wavepacket_data(N=N, Delta=Delta, k0=k0, sigma_x=sigma,
                                     nsteps=400, dt=0.3)

        # Verify physics: at t=0, hole component should be ~zero
        print(f"    t=0 check: Σ|u|²={data['all_ne'][0].sum():.4f}, "
              f"Σ|v|²={data['all_nh'][0].sum():.6f} (should be ~0)")

        # --- Interactive animation ---
        print("[5] Building interactive wavepacket animation...")
        fig5, slider = animate_sc_wavepacket(data)
        print("    Interactive window open — use the time slider or Play button.")

        if args.save:
            print(f"[6] Saving animation to {args.save}...")
            fig6, anim = make_animation(data, save_path=args.save)

    plt.show()

if __name__ == '__main__':
    main()
