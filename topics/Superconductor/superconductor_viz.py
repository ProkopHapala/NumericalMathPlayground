"""
Didactic visualization of superconducting wavepackets on a 1D tight-binding chain.

Shows 4 panels:
  1. Individual molecular orbitals (standing waves) of the 1D chain
  2. How superposing N adjacent MOs merges them into a moving wavepacket
  3. BdG quasiparticle wavepacket: electron-hole oscillation + propagation
  4. Cooper-pair correlation function F(r) in real space

Run:  python superconductor_viz.py
      python superconductor_viz.py --save   # save frames as PNG
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# ============================================================================
# 1D Tight-Binding Chain
# ============================================================================

def build_tb_hamiltonian(N, t=1.0, mu=0.0):
    """Build N×N tight-binding Hamiltonian (open boundary)."""
    H = np.zeros((N, N), dtype=complex)
    for i in range(N):
        H[i, i] = -mu
        if i + 1 < N:
            H[i, i + 1] = -t
            H[i + 1, i] = -t
    return H

def build_bdg_hamiltonian(N, t=1.0, mu=0.0, Delta=0.15):
    """Build 2N×2N BdG Hamiltonian in Nambu basis."""
    H0 = build_tb_hamiltonian(N, t, mu)
    H = np.zeros((2 * N, 2 * N), dtype=complex)
    H[:N, :N] = H0
    H[N:, N:] = -H0.conjugate()
    H[:N, N:] = Delta * np.eye(N)
    H[N:, :N] = Delta * np.eye(N)
    return H

# ============================================================================
# Panel 1: Individual MOs (standing waves)
# ============================================================================

def plot_individual_mos(N=64, t=1.0, mu=0.0, n_show=6):
    """Show first n_show molecular orbitals of the 1D chain."""
    H = build_tb_hamiltonian(N, t, mu)
    E, psi = np.linalg.eigh(H)
    x = np.arange(N)

    fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.2 * n_show), sharex=True)
    for i in range(n_show):
        ax = axes[i]
        wf = psi[:, i].real
        ax.plot(x, wf, 'b-', lw=1.5)
        ax.fill_between(x, wf, 0, alpha=0.2, color='blue')
        ax.axhline(0, color='gray', lw=0.5)
        n_nodes = np.sum(np.diff(np.sign(wf)) != 0)
        ax.set_ylabel(f"$\\phi_{{{i}}}$")
        ax.set_title(f"MO {i}: E={E[i]:.3f}, {n_nodes} nodes", fontsize=10, loc='left')
        ax.set_ylim(-0.35, 0.35)
    axes[-1].set_xlabel("site $i$")
    fig.suptitle("1D Tight-Binding Chain: Molecular Orbitals (Standing Waves)", fontsize=13, y=0.98)
    plt.tight_layout()
    return fig

# ============================================================================
# Panel 2: Superposition of MOs → Moving Wavepacket
# ============================================================================

def make_wavepacket_superposition(N=128, t=1.0, mu=0.0, x0=None, sigma=15.0, k0=None):
    """
    Build a wavepacket by superposing a band of MOs with Gaussian weights.
    Returns eigenvalues, eigenvectors, and expansion coefficients.
    """
    H = build_tb_hamiltonian(N, t, mu)
    E, psi = np.linalg.eigh(H)

    # Map eigenstate index to approximate k-value for open chain
    # k_n ≈ n*pi/(N+1)
    k_vals = np.arange(N) * np.pi / (N + 1)

    if k0 is None:
        k0 = np.pi / 2 + 0.3  # slightly above Fermi point
    if x0 is None:
        x0 = N // 3

    # Gaussian weight in k-space
    coeffs_k = np.exp(-(k_vals - k0)**2 / (2 * sigma**2))
    coeffs_k /= np.sqrt(np.sum(coeffs_k**2))

    # Also include spatial Gaussian to localize
    x = np.arange(N)
    spatial_env = np.exp(-(x - x0)**2 / (2 * (N/8)**2))
    spatial_env /= np.sqrt(np.sum(spatial_env**2))

    # Project spatial Gaussian onto MOs to get coefficients
    coeffs = psi.conjugate().T @ spatial_env
    # Combine with k-space filter
    coeffs = coeffs * coeffs_k
    coeffs /= np.sqrt(np.sum(np.abs(coeffs)**2))

    return E, psi, coeffs

def animate_wavepacket_formation(N=128, t=1.0, mu=0.0, n_bands=[1, 2, 3, 5, 10, 20], nsteps=200, dt=0.4):
    """
    Animation showing how superposing an increasing number of MOs
    creates an increasingly localized, moving wavepacket.
    """
    H = build_tb_hamiltonian(N, t, mu)
    E, psi = np.linalg.eigh(H)
    x = np.arange(N)

    # Build wavepacket coefficients using a spatial Gaussian
    x0 = N // 3
    sigma_x = 12.0
    k0 = np.pi / 2 + 0.3
    spatial_gauss = np.exp(-(x - x0)**2 / (2 * sigma_x**2)) * np.exp(1j * k0 * x)
    spatial_gauss /= np.sqrt(np.sum(np.abs(spatial_gauss)**2))
    all_coeffs = psi.conjugate().T @ spatial_gauss

    # Sort by |coefficient| descending to pick most important MOs
    order = np.argsort(np.abs(all_coeffs))[::-1]

    fig, axes = plt.subplots(len(n_bands), 1, figsize=(10, 2.0 * len(n_bands)), sharex=True)
    lines = []
    for ax, nb in zip(axes, n_bands):
        # Pick the nb most important MOs
        idx = order[:nb]
        c = np.zeros(N, dtype=complex)
        c[idx] = all_coeffs[idx]
        line, = ax.plot([], [], 'b-', lw=1.2)
        ax.set_ylim(-0.25, 0.25)
        ax.set_ylabel(f"{nb} MOs")
        ax.axhline(0, color='gray', lw=0.3)
        # Store coefficients for this subplot
        line._coeffs = c
        lines.append(line)

    axes[-1].set_xlabel("site $i$")
    fig.suptitle("Merging MOs into a Moving Wavepacket\n(more MOs → more localized → faster packet)", fontsize=12)

    def init():
        for ln in lines:
            ln.set_data([], [])
        return lines

    def update(frame):
        time = frame * dt
        for ln in lines:
            c = ln._coeffs
            phase = np.exp(-1j * E * time)
            wf = psi @ (phase * c)
            ln.set_data(x, wf.real)
        axes[0].set_title(f"t = {time:.1f}", fontsize=10, loc='left')
        return lines

    anim = FuncAnimation(fig, update, frames=nsteps, init_func=init, blit=True, interval=30)
    return fig, anim

# ============================================================================
# Panel 3: BdG Quasiparticle Wavepacket (electron-hole oscillation)
# ============================================================================

def animate_bdg_wavepacket(N=200, t=1.0, mu=0.0, Delta=0.15, x0=None, sigma=15.0,
                           k0=None, nsteps=300, dt=0.5):
    """
    BdG wavepacket: inject pure electron, watch it convert to hole and propagate.
    """
    if x0 is None:
        x0 = N // 3
    if k0 is None:
        k0 = np.pi / 2 + 0.35

    Hbdg = build_bdg_hamiltonian(N, t, mu, Delta)
    E, V = np.linalg.eigh(Hbdg)

    x = np.arange(N)
    u0 = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
    u0 /= np.sqrt(np.sum(np.abs(u0)**2))
    v0 = np.zeros(N, dtype=complex)
    psi0 = np.concatenate([u0, v0])
    coeffs = V.conjugate().T @ psi0

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    line_u, = axes[0].plot([], [], 'b-', lw=1.5, label="$|u_i|^2$ (electron)")
    line_v, = axes[1].plot([], [], 'r-', lw=1.5, label="$|v_i|^2$ (hole)")
    line_c, = axes[2].plot([], [], 'k--', lw=1.5, label="$|u|^2-|v|^2$ (charge)")

    for ax in axes:
        ax.axhline(0, color='gray', lw=0.3)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(0, N)
    axes[0].set_ylim(-0.01, 0.08)
    axes[1].set_ylim(-0.01, 0.08)
    axes[2].set_ylim(-0.08, 0.08)
    axes[2].set_xlabel("site $i$")
    fig.suptitle(f"BdG Quasiparticle Wavepacket ($\\Delta$={Delta})\n"
                 f"Electron→Hole oscillation + propagation", fontsize=12)

    def init():
        for ln in [line_u, line_v, line_c]:
            ln.set_data([], [])
        return line_u, line_v, line_c

    def update(frame):
        time = frame * dt
        phase = np.exp(-1j * E * time)
        psi = V @ (phase * coeffs)
        u = psi[:N]
        v = psi[N:]
        ne = np.abs(u)**2
        nh = np.abs(v)**2
        charge = ne - nh
        line_u.set_data(x, ne)
        line_v.set_data(x, nh)
        line_c.set_data(x, charge)
        axes[0].set_title(f"t = {time:.1f}   (Andreev period ~$\\pi/\\Delta$ = {np.pi/Delta:.1f})",
                          fontsize=9, loc='left')
        return line_u, line_v, line_c

    anim = FuncAnimation(fig, update, frames=nsteps, init_func=init, blit=True, interval=30)
    return fig, anim

# ============================================================================
# Panel 4: Cooper Pair Correlation F(r)
# ============================================================================

def plot_cooper_pair_correlation(N=4096, t=1.0, mu=0.0, Delta=0.05):
    """Plot real-space Cooper-pair correlation function F(r)."""
    k = 2 * np.pi * np.arange(N) / N - np.pi
    xi = -2 * t * np.cos(k) - mu
    E = np.sqrt(xi**2 + Delta**2)
    Fk = Delta / (2 * E)
    Fr = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Fk)))
    r = np.arange(-N // 2, N // 2)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(r, np.real(Fr), 'b-', lw=1.5)
    ax.axhline(0, color='gray', lw=0.3)
    ax.set_xlim(-200, 200)
    ax.set_xlabel("relative separation $r$ (sites)")
    ax.set_ylabel("$F(r)$")
    ax.set_title(f"Cooper-Pair Correlation $F(r)$ ($\\Delta$={Delta})\n"
                 f"Pair size $\\xi_{{coh}} \\sim v_F/\\Delta \\sim 1/\\Delta$ = {1/Delta:.0f} sites")
    plt.tight_layout()
    return fig

# ============================================================================
# Panel 5: BdG Spectrum + Group Velocity
# ============================================================================

def plot_bdg_spectrum(t=1.0, mu=0.0, Delta=0.15):
    """Show BdG dispersion E(k) and group velocity v_g(k)."""
    k = np.linspace(-np.pi, np.pi, 500)
    xi = -2 * t * np.cos(k) - mu
    E = np.sqrt(xi**2 + Delta**2)
    vg = xi / E * 2 * t * np.sin(k)  # dξ/dk = 2t sin(k)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1, ax2 = axes

    ax1.plot(k, E, 'b-', lw=2, label="$E_k = \\sqrt{\\xi_k^2 + \\Delta^2}$")
    ax1.plot(k, np.abs(xi), 'g--', lw=1, label="$|\\xi_k|$ (normal)")
    ax1.axhline(Delta, color='r', ls=':', lw=1, label=f"gap $\\Delta$={Delta}")
    ax1.set_xlabel("$k$")
    ax1.set_ylabel("Energy")
    ax1.set_title("BdG Spectrum")
    ax1.legend(fontsize=9)
    ax1.set_xlim(-np.pi, np.pi)

    ax2.plot(k, vg, 'b-', lw=2)
    ax2.axhline(0, color='gray', lw=0.5)
    ax2.axvline(np.pi / 2, color='r', ls=':', lw=1, label="$k_F$")
    ax2.set_xlabel("$k$")
    ax2.set_ylabel("$v_g(k)$")
    ax2.set_title("Group Velocity (vanishes at gap edge)")
    ax2.legend(fontsize=9)
    ax2.set_xlim(-np.pi, np.pi)
    plt.tight_layout()
    return fig

# ============================================================================
# Combined didactic animation: MOs → wavepacket → BdG
# ============================================================================

def animate_full_story(N=150, t=1.0, mu=0.0, Delta=0.15, nsteps=300, dt=0.4):
    """
    Single figure with 4 subplots showing the full story side by side:
      - Top:    Normal wavepacket (Delta=0) — pure electron, moves freely
      - Mid:    BdG wavepacket (Delta>0) — electron component |u|^2
      - Mid:    BdG wavepacket — hole component |v|^2
      - Bottom: Net charge |u|^2 - |v|^2
    """
    # Normal chain
    H0 = build_tb_hamiltonian(N, t, mu)
    E0, psi0 = np.linalg.eigh(H0)

    # BdG chain
    Hbdg = build_bdg_hamiltonian(N, t, mu, Delta)
    Ebdg, Vbdg = np.linalg.eigh(Hbdg)

    x = np.arange(N)
    x0 = N // 4
    sigma = 12.0
    k0 = np.pi / 2 + 0.35

    # Initial wavepacket (same for both)
    u0 = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
    u0 /= np.sqrt(np.sum(np.abs(u0)**2))

    # Normal: expand in MOs
    c_normal = psi0.conjugate().T @ u0
    # BdG: expand in BdG eigenstates
    psi0_bdg = np.concatenate([u0, np.zeros(N, dtype=complex)])
    c_bdg = Vbdg.conjugate().T @ psi0_bdg

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    titles = [
        f"Normal chain ($\\Delta$=0): electron wavepacket — moves freely",
        f"Superconducting chain ($\\Delta$={Delta}): $|u_i|^2$ (electron part)",
        f"Superconducting chain: $|v_i|^2$ (hole part — grows from zero!)",
        f"Net charge $|u|^2 - |v|^2$ (much smaller than either component)",
    ]
    colors = ['blue', 'blue', 'red', 'black']
    styles = ['-', '-', '-', '--']
    lines = []
    for ax, title, color, style in zip(axes, titles, colors, styles):
        ln, = ax.plot([], [], color=color, linestyle=style, lw=1.5)
        ax.axhline(0, color='gray', lw=0.3)
        ax.set_xlim(0, N)
        ax.set_ylim(-0.02, 0.08)
        ax.set_title(title, fontsize=9, loc='left')
        lines.append(ln)
    axes[-1].set_ylim(-0.08, 0.08)
    axes[-1].set_xlabel("site $i$")
    fig.suptitle("Normal vs Superconducting Wavepacket Propagation", fontsize=13)

    def init():
        for ln in lines:
            ln.set_data([], [])
        return lines

    def update(frame):
        time = frame * dt
        # Normal
        phase0 = np.exp(-1j * E0 * time)
        wf_normal = psi0 @ (phase0 * c_normal)
        ne_normal = np.abs(wf_normal)**2
        lines[0].set_data(x, ne_normal)

        # BdG
        phase_bdg = np.exp(-1j * Ebdg * time)
        psi_bdg = Vbdg @ (phase_bdg * c_bdg)
        u = psi_bdg[:N]
        v = psi_bdg[N:]
        ne = np.abs(u)**2
        nh = np.abs(v)**2
        charge = ne - nh
        lines[1].set_data(x, ne)
        lines[2].set_data(x, nh)
        lines[3].set_data(x, charge)

        axes[0].set_title(f"t = {time:.1f}   Normal ($\\Delta$=0): packet moves freely",
                          fontsize=9, loc='left')
        return lines

    anim = FuncAnimation(fig, update, frames=nsteps, init_func=init, blit=True, interval=30)
    return fig, anim

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Superconductor wavepacket visualization")
    parser.add_argument('--save', action='store_true', help="Save static figures as PNG")
    parser.add_argument('--no-anim', action='store_true', help="Skip animations (static only)")
    args = parser.parse_args()

    # --- Static figures ---
    print("[1/4] Individual molecular orbitals...")
    fig1 = plot_individual_mos()
    if args.save:
        fig1.savefig("sc_mos.png", dpi=150, bbox_inches='tight')

    print("[2/4] BdG spectrum and group velocity...")
    fig_spec = plot_bdg_spectrum()
    if args.save:
        fig_spec.savefig("sc_spectrum.png", dpi=150, bbox_inches='tight')

    print("[3/4] Cooper pair correlation F(r)...")
    fig3 = plot_cooper_pair_correlation()
    if args.save:
        fig3.savefig("sc_cooper_pair.png", dpi=150, bbox_inches='tight')

    if not args.no_anim:
        print("[4/4] Animations (close window to continue)...")

        print("  - Wavepacket formation from MOs...")
        fig2, anim2 = animate_wavepacket_formation()
        if args.save:
            anim2.save("sc_wavepacket_formation.gif", writer='pillow', fps=30, dpi=100)

        print("  - BdG quasiparticle wavepacket...")
        fig4, anim4 = animate_bdg_wavepacket()
        if args.save:
            anim4.save("sc_bdg_wavepacket.gif", writer='pillow', fps=30, dpi=100)

        print("  - Full story: normal vs superconducting...")
        fig5, anim5 = animate_full_story()
        if args.save:
            anim5.save("sc_full_story.gif", writer='pillow', fps=30, dpi=100)

    plt.show()

if __name__ == '__main__':
    main()
