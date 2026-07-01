#!/usr/bin/env python3
"""
Real-space and k-space visualization of a 1D superconducting BdG wavepacket.

This is a didactic mean-field BCS/BdG model, not a self-consistent
Eliashberg/phonon simulation and not a supercurrent calculation.  It shows an
injected electron-like excitation evolving under a fixed superconducting gap.

Important physical point
------------------------
For a time-independent Hamiltonian, the coefficients in the ENERGY EIGENBASIS
are constant in magnitude.  Only their phases rotate.  Therefore an imshow of
|c_n(t)|^2 in BdG eigenstates should be time-independent.  The interesting
oscillating quantity is the coefficient in the BARE electron/hole k basis:

    (u_k(t), v_k(t))^T

because the pairing term mixes electron and hole amplitudes.  This script shows
both:

  1. real-space propagation: |u(x,t)|^2, |v(x,t)|^2, charge = |u|^2-|v|^2
  2. bare k-basis occupations: |u_k(t)|^2 and |v_k(t)|^2
  3. BdG eigenbranch coefficients: |c_+(k)|^2 and |c_-(k)|^2, which are constant
  4. phase-resolved BdG coefficients: Re[c_+(k) exp(-i E_k t)],
     Re[c_-(k) exp(+i E_k t)]

Hamiltonian
-----------
Normal 1D periodic tight-binding band:

    xi_k = -2 t cos(k) - mu

Uniform real s-wave BdG Hamiltonian for each k:

    H_k = [[ xi_k,  Delta ],
           [ Delta, -xi_k ]]

with spectrum E_k = sqrt(xi_k^2 + Delta^2).  Units: hbar = 1.

Run examples
------------
    python sc_wavepacket_motion_improved.py --show
    python sc_wavepacket_motion_improved.py --save-prefix sc_bdg
    python sc_wavepacket_motion_improved.py --Delta 0.08 --dk0 0.10 --sigma-k 0.045 --show
"""

import argparse
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


def wrap_pi(a):
    """Map angle differences to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def make_k_grid(N):
    """Sorted periodic k-grid in [-pi, pi)."""
    return 2 * np.pi * (np.arange(N) - N // 2) / N


def fermi_k(mu, t):
    """Positive Fermi point for xi_k = -2t cos(k)-mu, if the band is crossed."""
    val = -mu / (2 * t)
    if abs(val) <= 1:
        return float(np.arccos(val))
    # Outside the band there is no Fermi surface.  Return the band minimum side.
    return 0.0 if val > 1 else np.pi


@dataclass
class BdGData:
    N: int
    t_hop: float
    mu: float
    Delta: float
    x: np.ndarray
    k: np.ndarray
    xi: np.ndarray
    E: np.ndarray
    times: np.ndarray
    kF: float
    k0: float
    a0: np.ndarray
    psi_normal_x: np.ndarray
    u_x: np.ndarray
    v_x: np.ndarray
    u_k: np.ndarray
    v_k: np.ndarray
    c_plus: np.ndarray
    c_minus: np.ndarray
    Ucoh: np.ndarray
    Vcoh: np.ndarray
    k_mask: np.ndarray


def prepare_data(
    N=256,
    t_hop=1.0,
    mu=0.0,
    Delta=0.15,
    dk0=0.10,
    sigma_k=0.045,
    x0=None,
    nsteps=500,
    dt=0.25,
    k_window=0.75,
):
    if Delta <= 0:
        raise ValueError("Delta must be positive for the BdG comparison.")
    if x0 is None:
        x0 = N // 4

    x = np.arange(N)
    k = make_k_grid(N)
    kF = fermi_k(mu, t_hop)
    k0 = kF + dk0

    xi = -2 * t_hop * np.cos(k) - mu
    E = np.sqrt(xi * xi + Delta * Delta)

    # Initial pure electron Gaussian wavepacket in k-space, centered near +kF.
    # exp(-i k x0) places the real-space packet around x0 with the convention
    # psi(x) = sum_k a_k exp(+i k x) / sqrt(N).
    dk = wrap_pi(k - k0)
    a0 = np.exp(-0.5 * (dk / sigma_k) ** 2) * np.exp(-1j * k * x0)
    a0 = a0 / np.sqrt(np.sum(np.abs(a0) ** 2))

    times = dt * np.arange(nsteps)

    # Normal evolution in k basis.
    psi_normal_k = a0[None, :] * np.exp(-1j * times[:, None] * xi[None, :])

    # BdG evolution of a pure electron initial state (u_k(0)=a0, v_k(0)=0).
    cEt = np.cos(times[:, None] * E[None, :])
    sEt = np.sin(times[:, None] * E[None, :])
    u_k = a0[None, :] * (cEt - 1j * (xi[None, :] / E[None, :]) * sEt)
    v_k = a0[None, :] * (-1j * (Delta / E[None, :]) * sEt)

    # Real-space transform.  This is just the molecular-orbital expansion
    # psi(x,t)=sum_k a_k(t) exp(i k x)/sqrt(N), written explicitly so it is
    # readable and independent of FFT ordering conventions.
    expikx = np.exp(1j * np.outer(k, x)) / np.sqrt(N)  # [k,x]
    psi_normal_x = psi_normal_k @ expikx
    u_x = u_k @ expikx
    v_x = v_k @ expikx

    # BdG eigenvectors for each k.  Positive branch |+>=(U,V), negative branch
    # |->=(-V,U).  Decomposition of initial (a0,0): c+=U*a0, c-=-V*a0.
    Ucoh = np.sqrt(0.5 * (1.0 + xi / E))
    Vcoh = np.sign(Delta) * np.sqrt(0.5 * (1.0 - xi / E))
    c_plus = Ucoh * a0
    c_minus = -Vcoh * a0

    # Window for coefficient heatmaps around the avoided crossing at +kF.
    k_mask = np.abs(wrap_pi(k - kF)) < k_window

    return BdGData(
        N=N,
        t_hop=t_hop,
        mu=mu,
        Delta=Delta,
        x=x,
        k=k,
        xi=xi,
        E=E,
        times=times,
        kF=kF,
        k0=k0,
        a0=a0,
        psi_normal_x=psi_normal_x,
        u_x=u_x,
        v_x=v_x,
        u_k=u_k,
        v_k=v_k,
        c_plus=c_plus,
        c_minus=c_minus,
        Ucoh=Ucoh,
        Vcoh=Vcoh,
        k_mask=k_mask,
    )


def ring_center(rho):
    """Center of a packet on a periodic ring, returned in site units."""
    N = rho.shape[1]
    theta = 2 * np.pi * np.arange(N) / N
    z = rho @ np.exp(1j * theta)
    center = (np.angle(z) % (2 * np.pi)) * N / (2 * np.pi)
    return center


def heatmap(ax, arr, extent, title, xlabel, ylabel, cmap="magma", aspect="auto", vcenter=None):
    if vcenter is None:
        im = ax.imshow(arr, origin="lower", aspect=aspect, extent=extent, cmap=cmap, interpolation="nearest")
    else:
        vmax = np.max(np.abs(arr))
        im = ax.imshow(arr, origin="lower", aspect=aspect, extent=extent, cmap=cmap,
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return im


def plot_summary(data: BdGData, save_path=None, show=False):
    x = data.x
    t = data.times
    k_rel = data.k[data.k_mask] - data.kF
    # Because k is sorted and the mask is local for default kF, this is fine.
    # For display, use relative k around +kF.

    rho_n = np.abs(data.psi_normal_x) ** 2
    rho_e = np.abs(data.u_x) ** 2
    rho_h = np.abs(data.v_x) ** 2
    rho_q = rho_e - rho_h

    # k-space occupations in the bare electron/hole basis.
    occ_e_k = np.abs(data.u_k[:, data.k_mask]) ** 2
    occ_h_k = np.abs(data.v_k[:, data.k_mask]) ** 2

    # BdG eigenbranch occupation magnitudes; deliberately time-independent.
    occ_plus = np.tile(np.abs(data.c_plus[data.k_mask]) ** 2, (len(t), 1))
    occ_minus = np.tile(np.abs(data.c_minus[data.k_mask]) ** 2, (len(t), 1))
    occ_pm = np.concatenate([occ_minus, occ_plus], axis=1)

    # Integrated weights are useful sanity checks.
    Pe = rho_e.sum(axis=1)
    Ph = rho_h.sum(axis=1)
    Q = Pe - Ph
    Pn = rho_n.sum(axis=1)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)

    # 1. Band structure near the superconducting gap.
    ax = axes[0, 0]
    mask_band = np.abs(data.k - data.kF) < 1.0
    kr = data.k[mask_band] - data.kF
    ax.plot(kr, data.xi[mask_band], "--", lw=1.0, label=r"bare electron $\xi_k$")
    ax.plot(kr, -data.xi[mask_band], "--", lw=1.0, label=r"bare hole $-\xi_k$")
    ax.plot(kr, data.E[mask_band], lw=2.0, label=r"BdG $+E_k$")
    ax.plot(kr, -data.E[mask_band], lw=2.0, label=r"BdG $-E_k$")
    # Scaled initial wavepacket envelope, drawn on the positive branch.
    env = np.abs(data.a0[mask_band]) ** 2
    if env.max() > 0:
        env = env / env.max()
        ax.fill_between(kr, data.Delta, data.Delta + 0.4 * data.Delta * env, alpha=0.25,
                        label="initial k envelope")
    ax.axhline(0, lw=0.5, color="k")
    ax.axvline(data.k0 - data.kF, lw=0.8, color="k", alpha=0.5)
    ax.set_title("Avoided crossing: bare electron/hole bands become BdG bands", fontsize=9, loc="left")
    ax.set_xlabel(r"$k-k_F$")
    ax.set_ylabel("energy")
    ax.legend(fontsize=7, loc="best")

    extent_x_t = [x[0], x[-1], t[0], t[-1]]
    extent_k_t = [k_rel[0], k_rel[-1], t[0], t[-1]]

    heatmap(axes[0, 1], rho_n, extent_x_t,
            r"normal chain: $|\psi(x,t)|^2$", "site x", "time", cmap="magma")
    heatmap(axes[0, 2], rho_q, extent_x_t,
            r"SC quasiparticle charge: $|u|^2-|v|^2$", "site x", "time", cmap="coolwarm", vcenter=0.0)

    heatmap(axes[1, 0], rho_e, extent_x_t,
            r"SC electron component: $|u(x,t)|^2$", "site x", "time", cmap="Blues")
    heatmap(axes[1, 1], rho_h, extent_x_t,
            r"SC hole component: $|v(x,t)|^2$", "site x", "time", cmap="Reds")

    ax = axes[1, 2]
    ax.plot(t, Pn, label=r"normal norm $\sum|\psi|^2$")
    ax.plot(t, Pe, label=r"SC electron weight $\sum|u|^2$")
    ax.plot(t, Ph, label=r"SC hole weight $\sum|v|^2$")
    ax.plot(t, Q, label=r"quasiparticle charge $\sum(|u|^2-|v|^2)$")
    ax.set_title("Integrated weights: total Nambu norm is conserved", fontsize=9, loc="left")
    ax.set_xlabel("time")
    ax.set_ylabel("integrated weight")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.25)

    heatmap(axes[2, 0], occ_e_k, extent_k_t,
            r"bare electron k coefficient: $|u_k(t)|^2$", r"$k-k_F$", "time", cmap="Blues")
    heatmap(axes[2, 1], occ_h_k, extent_k_t,
            r"bare hole k coefficient: $|v_k(t)|^2$", r"$k-k_F$", "time", cmap="Reds")

    # Combined +/- branch occupancy map.  The y axis is branch-index rather than k;
    # label it explicitly to avoid pretending this is a continuous physical axis.
    ax = axes[2, 2]
    im = ax.imshow(occ_pm.T, origin="lower", aspect="auto",
                   extent=[t[0], t[-1], 0, occ_pm.shape[1]], cmap="viridis", interpolation="nearest")
    ax.axhline(occ_minus.shape[1], color="w", lw=0.8)
    ax.set_title(r"BdG eigenbasis occupation $|c_\pm(k)|^2$ is static", fontsize=9, loc="left")
    ax.set_xlabel("time")
    ax.set_ylabel(r"state index: $-E_k$ below, $+E_k$ above")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(
        rf"1D tight-binding BdG wavepacket: $t={data.t_hop:g}$, $\mu={data.mu:g}$, "
        rf"$\Delta={data.Delta:g}$, $k_0-k_F={data.k0-data.kF:.3f}$\n"
        "Real-space motion comes from phase interference; electron-hole conversion is visible in the bare k basis.",
        fontsize=12,
    )

    if save_path:
        fig.savefig(save_path, dpi=160)
        print(f"saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_bdg_coefficients(data: BdGData, save_path=None, show=False):
    t = data.times
    k_rel = data.k[data.k_mask] - data.kF
    E = data.E[data.k_mask]
    cp = data.c_plus[data.k_mask]
    cm = data.c_minus[data.k_mask]

    # Magnitudes are static, phases rotate with +/-E.
    Cp_abs = np.tile(np.abs(cp) ** 2, (len(t), 1))
    Cm_abs = np.tile(np.abs(cm) ** 2, (len(t), 1))
    Cp_re = np.real(cp[None, :] * np.exp(-1j * t[:, None] * E[None, :]))
    Cm_re = np.real(cm[None, :] * np.exp(+1j * t[:, None] * E[None, :]))

    extent = [k_rel[0], k_rel[-1], t[0], t[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    heatmap(axes[0, 0], Cp_abs, extent,
            r"positive BdG branch: $|c_+(k,t)|^2$", r"$k-k_F$", "time", cmap="viridis")
    heatmap(axes[0, 1], Cm_abs, extent,
            r"negative BdG branch: $|c_-(k,t)|^2$", r"$k-k_F$", "time", cmap="viridis")
    heatmap(axes[1, 0], Cp_re, extent,
            r"phase-resolved: Re$[c_+(k)e^{-iE_kt}]$", r"$k-k_F$", "time", cmap="coolwarm", vcenter=0.0)
    heatmap(axes[1, 1], Cm_re, extent,
            r"phase-resolved: Re$[c_-(k)e^{+iE_kt}]$", r"$k-k_F$", "time", cmap="coolwarm", vcenter=0.0)
    fig.suptitle(
        "BdG eigenstate coefficients: magnitudes are conserved; phase gradients create real-space motion",
        fontsize=12,
    )
    if save_path:
        fig.savefig(save_path, dpi=160)
        print(f"saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def print_diagnostics(data: BdGData):
    rho_n = np.abs(data.psi_normal_x) ** 2
    rho_e = np.abs(data.u_x) ** 2
    rho_h = np.abs(data.v_x) ** 2
    norm_bdg = rho_e.sum(axis=1) + rho_h.sum(axis=1)
    charge_bdg = rho_e.sum(axis=1) - rho_h.sum(axis=1)

    vg_normal = 2 * data.t_hop * np.sin(data.k0)
    xi0 = -2 * data.t_hop * np.cos(data.k0) - data.mu
    E0 = np.sqrt(xi0 * xi0 + data.Delta * data.Delta)
    vg_bdg = (xi0 / E0) * vg_normal

    print("Physical diagnostics")
    print("--------------------")
    print(f"kF = {data.kF:.6f}, k0 = {data.k0:.6f}, k0-kF = {data.k0-data.kF:.6f}")
    print(f"xi(k0) = {xi0:.6f}, E(k0) = {E0:.6f}")
    print(f"normal group velocity dxi/dk = {vg_normal:.6f} sites/time")
    print(f"BdG group velocity dE/dk = (xi/E) dxi/dk = {vg_bdg:.6f} sites/time")
    print(f"min/max normal norm: {rho_n.sum(axis=1).min():.10f}, {rho_n.sum(axis=1).max():.10f}")
    print(f"min/max BdG Nambu norm sum(|u|^2+|v|^2): {norm_bdg.min():.10f}, {norm_bdg.max():.10f}")
    print(f"initial electron/hole weights: {rho_e[0].sum():.10f}, {rho_h[0].sum():.10f}")
    print(f"quasiparticle charge range sum(|u|^2-|v|^2): {charge_bdg.min():.6f} .. {charge_bdg.max():.6f}")
    print()
    print("Interpretation note:")
    print("  |c_+(k)|^2 and |c_-(k)|^2 are eigenstate occupations and therefore do not change.")
    print("  |u_k(t)|^2 and |v_k(t)|^2 are bare electron/hole weights and do oscillate due to pairing.")


def main():
    parser = argparse.ArgumentParser(description="1D BdG wavepacket visualizer with real-space and k-space imshow plots")
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--t", dest="t_hop", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--Delta", type=float, default=0.15)
    parser.add_argument("--dk0", type=float, default=0.10, help="packet center measured from +kF")
    parser.add_argument("--sigma-k", type=float, default=0.045)
    parser.add_argument("--x0", type=int, default=None)
    parser.add_argument("--nsteps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--k-window", type=float, default=0.75)
    parser.add_argument("--save-prefix", type=str, default=None,
                        help="If set, save PREFIX_summary.png and PREFIX_coefficients.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    data = prepare_data(
        N=args.N,
        t_hop=args.t_hop,
        mu=args.mu,
        Delta=args.Delta,
        dk0=args.dk0,
        sigma_k=args.sigma_k,
        x0=args.x0,
        nsteps=args.nsteps,
        dt=args.dt,
        k_window=args.k_window,
    )
    print_diagnostics(data)

    if args.save_prefix:
        plot_summary(data, save_path=f"{args.save_prefix}_summary.png", show=False)
        plot_bdg_coefficients(data, save_path=f"{args.save_prefix}_coefficients.png", show=False)
    else:
        plot_summary(data, show=args.show)
        plot_bdg_coefficients(data, show=args.show)


if __name__ == "__main__":
    main()
