#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RC = 1.0
N_GRID = 401
OUT_DIR = Path(__file__).resolve().parent


def radial_envelope(r, rc=RC):
    q = (r / rc) ** 2
    return np.where(r <= rc, (1.0 - q) ** 2, 0.0)


def radial_envelope_derivative(r, rc=RC):
    q = (r / rc) ** 2
    return np.where(r <= rc, -4.0 * r * (1.0 - q) / rc**2, 0.0)


def c_for_radial_minimum(rc, r0):
    return (rc**2 + r0**2) / (2.0 * r0**4)


def radial_minimum_potential(r, rc, r0):
    c = c_for_radial_minimum(rc, r0)
    return (1.0 / r**2 - c) * radial_envelope(r, rc)


def radial_minimum_force(r, rc, r0):
    c = c_for_radial_minimum(rc, r0)
    R = radial_envelope(r, rc)
    dR = radial_envelope_derivative(r, rc)
    dV = -2.0 * R / r**3 + (1.0 / r**2 - c) * dR
    return -dV


def c_for_angular_minimum(rc, r0, fang_sign=-1.0):
    q = (r0 / rc) ** 2
    return 2.0 * rc**3 * (1.0 + q) / (fang_sign * r0**5 * (3.0 - 7.0 * q))


def angular_radial_cut(r, rc, r0, fang_sign=-1.0):
    c = c_for_angular_minimum(rc, r0, fang_sign)
    fang = fang_sign * (r / rc) ** 3
    return (1.0 / r**2 + c * fang) * radial_envelope(r, rc)


def angular_modulated_potential(x, y, rc, r0, rmin=1.0e-3):
    r = np.sqrt(x**2 + y**2)
    c = c_for_angular_minimum(rc, r0)
    fang = angular_cubic(x / rc, y / rc)
    return (1.0 / np.maximum(r, rmin) ** 2 + c * fang) * radial_envelope(r, rc)


def angular_cubic(x, y):
    return x * (x**2 - 3.0 * y**2)


def make_grid(rc=RC, n=N_GRID):
    lim = 1.15 * rc
    x = np.linspace(-lim, lim, n)
    y = np.linspace(-lim, lim, n)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    return x, y, xx, yy, rr


def plot_basis():
    r = np.linspace(0.0, 1.25 * RC, 1000)
    x, y, xx, yy, rr = make_grid()
    mask = rr <= RC
    R = radial_envelope(rr)
    A = angular_cubic(xx / RC, yy / RC)
    B = R * A
    B_masked = np.ma.array(B, mask=~mask)
    A_masked = np.ma.array(A, mask=~mask)

    step = x[1] - x[0]
    dBdy, dBdx = np.gradient(B, step, step, edge_order=2)
    Fx = np.ma.array(-dBdx, mask=~mask)
    Fy = np.ma.array(-dBdy, mask=~mask)
    Fmag = np.ma.sqrt(Fx**2 + Fy**2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(r, radial_envelope(r), lw=2.5)
    ax.axvline(RC, color="k", ls="--", lw=1.0, alpha=0.6)
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    ax.set_title(r"Radial envelope $R(r)=(1-(r/r_c)^2)^2$")
    ax.set_xlabel("r")
    ax.set_ylabel("R(r)")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    im = ax.imshow(
        A_masked,
        origin="lower",
        extent=(x[0], x[-1], y[0], y[-1]),
        cmap="seismic",
        vmin=-1.0,
        vmax=1.0,
    )
    ax.contour(xx, yy, A, levels=[0.0], colors="k", linewidths=1.0)
    ax.set_title(r"Angular cubic $A(x,y)=x^3-3xy^2$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[1, 0]
    im = ax.imshow(
        B_masked,
        origin="lower",
        extent=(x[0], x[-1], y[0], y[-1]),
        cmap="seismic",
        vmin=-0.25,
        vmax=0.25,
    )
    ax.contour(xx, yy, B, levels=[0.0], colors="k", linewidths=0.8)
    ax.set_title(r"Basis $B(x,y)=R(r)A(x,y)$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[1, 1]
    im = ax.imshow(
        Fmag,
        origin="lower",
        extent=(x[0], x[-1], y[0], y[-1]),
        cmap="viridis",
    )
    stride = 18
    ax.quiver(
        xx[::stride, ::stride],
        yy[::stride, ::stride],
        Fx[::stride, ::stride],
        Fy[::stride, ::stride],
        color="white",
        pivot="mid",
        scale=45,
        width=0.003,
    )
    ax.set_title(r"Force-like field $F=-\nabla B$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"$|F|$")

    fig.suptitle("First radial-angular basis for ReactiveFF", fontsize=16)
    fig.tight_layout()
    outfile = OUT_DIR / "radial_angular_basis.png"
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(outfile)


def plot_angular_modulated_potential(rc=2.0, r0=1.5):
    x, y, xx, yy, rr = make_grid(rc=rc)
    mask = rr <= rc
    V = angular_modulated_potential(xx, yy, rc, r0)
    V_masked = np.ma.array(V, mask=~mask)
    Vmin = float(V_masked.min())
    Vmax = -Vmin
    step = x[1] - x[0]
    dVdy, dVdx = np.gradient(V, step, step, edge_order=2)
    Fx = np.ma.array(-dVdx, mask=~mask)
    Fy = np.ma.array(-dVdy, mask=~mask)
    Fmag = np.ma.sqrt(Fx**2 + Fy**2)
    c = c_for_angular_minimum(rc, r0)
    r = np.linspace(0.04 * rc, 1.2 * rc, 1600)
    V_attr = angular_radial_cut(r, rc, r0, -1.0)
    V_rep = angular_radial_cut(r, rc, r0, 1.0)
    V0 = angular_radial_cut(np.array([r0]), rc, r0, -1.0)[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    im = ax.imshow(
        V_masked,
        origin="lower",
        extent=(x[0], x[-1], y[0], y[-1]),
        cmap="seismic",
        vmin=Vmin,
        vmax=Vmax,
    )
    ax.contour(xx, yy, V, levels=[0.0], colors="k", linewidths=0.8)
    ax.contour(xx, yy, rr, levels=[r0, rc], colors=["tab:orange", "tab:red"], linestyles=[":", "--"], linewidths=[1.4, 1.2])
    ax.set_title(r"$V=(1/r^2+c f_{ang})f_{cut}$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.85, label="V")

    ax = axes[0, 1]
    im = ax.imshow(
        np.clip(Fmag, 0.0, 10.0),
        origin="lower",
        extent=(x[0], x[-1], y[0], y[-1]),
        cmap="viridis",
        vmin=0.0,
        vmax=10.0,
    )
    stride = 18
    ax.quiver(
        xx[::stride, ::stride],
        yy[::stride, ::stride],
        Fx[::stride, ::stride],
        Fy[::stride, ::stride],
        color="white",
        pivot="mid",
        scale=180,
        width=0.003,
    )
    ax.set_title(r"Force field $F=-\nabla V$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.85, label="clipped |F|")

    ax = axes[1, 0]
    ax.plot(r, V_attr, lw=2.2, label=r"attractive lobe: $f_{ang}=-(r/r_c)^3$")
    ax.plot(r, V_rep, lw=2.2, label=r"repulsive lobe: $f_{ang}=+(r/r_c)^3$")
    ax.scatter([r0], [V0], color="tab:orange", zorder=5, label=rf"target $r_0={r0:.2f}$")
    ax.axvline(r0, color="tab:orange", ls=":", lw=1.5)
    ax.axvline(rc, color="tab:red", ls="--", lw=1.2)
    ax.axhline(0.0, color="k", lw=0.7, alpha=0.45)
    ax.set_title(rf"radial cuts, $c={c:.4f}$")
    ax.set_xlabel("r")
    ax.set_ylabel("V(r)")
    ax.set_xlim(0.0, 1.2 * rc)
    ax.set_ylim(1.35 * V0, 3.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    rs = np.linspace(0.67 * rc, 0.95 * rc, 800)
    ax.plot(rs, c_for_angular_minimum(rc, rs), lw=2.2)
    ax.scatter([r0], [c], color="tab:orange", zorder=5)
    ax.axvline(r0, color="tab:orange", ls=":", lw=1.5)
    ax.set_title(r"$c(r_0)$ for attractive 3-fold lobes")
    ax.set_xlabel(r"target minimum $r_0$")
    ax.set_ylabel("c")
    ax.grid(alpha=0.3)

    fig.suptitle("Angular-modulated radial potential", fontsize=16)
    fig.tight_layout()
    outfile = OUT_DIR / "angular_modulated_potential.png"
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(outfile)


def plot_radial_minimum_potential(rc=2.0, r0=1.2):
    r = np.linspace(0.04 * rc, 1.3 * rc, 1600)
    V = radial_minimum_potential(r, rc, r0)
    F = radial_minimum_force(r, rc, r0)
    c = c_for_radial_minimum(rc, r0)
    V0 = radial_minimum_potential(np.array([r0]), rc, r0)[0]
    r0s = np.array([0.4 * rc, 0.6 * rc, 0.8 * rc])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(r, V, lw=2.2, label=rf"$c={c:.4f}$")
    ax.scatter([r0], [V0], color="tab:orange", zorder=5, label=rf"minimum at $r_0={r0:.2f}$")
    ax.axvline(r0, color="tab:orange", ls=":", lw=1.5)
    ax.axvline(rc, color="tab:red", ls="--", lw=1.2, label=rf"$r_c={rc:.2f}$")
    ax.axhline(0.0, color="k", lw=0.7, alpha=0.45)
    ax.set_title(r"$V(r)=(1/r^2-c)(1-(r/r_c)^2)^2$")
    ax.set_xlabel("r")
    ax.set_ylabel("V(r)")
    ax.set_xlim(0.0, 1.3 * rc)
    ax.set_ylim(1.35 * V0, 2.0)
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(r, F, lw=2.2)
    ax.axvline(r0, color="tab:orange", ls=":", lw=1.5)
    ax.axvline(rc, color="tab:red", ls="--", lw=1.2)
    ax.axhline(0.0, color="k", lw=0.7, alpha=0.45)
    ax.set_title(r"Force $F(r)=-dV/dr$")
    ax.set_xlabel("r")
    ax.set_ylabel("F(r)")
    ax.set_xlim(0.0, 1.3 * rc)
    ax.set_ylim(-2.0, 6.0)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for ri in r0s:
        ci = c_for_radial_minimum(rc, ri)
        ax.plot(r, radial_minimum_potential(r, rc, ri), lw=1.8, label=rf"$r_0={ri:.2f}, c={ci:.2f}$")
        ax.axvline(ri, color="k", ls=":", lw=0.8, alpha=0.35)
    ax.axvline(rc, color="tab:red", ls="--", lw=1.2)
    ax.axhline(0.0, color="k", lw=0.7, alpha=0.45)
    ax.set_title(r"Moving the minimum by changing $c$")
    ax.set_xlabel("r")
    ax.set_ylabel("V(r)")
    ax.set_xlim(0.0, 1.3 * rc)
    ax.set_ylim(-1.2, 3.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    rs = np.linspace(0.25 * rc, 0.95 * rc, 800)
    ax.plot(rs, c_for_radial_minimum(rc, rs), lw=2.2)
    ax.scatter([r0], [c], color="tab:orange", zorder=5)
    ax.axvline(r0, color="tab:orange", ls=":", lw=1.5)
    ax.set_title(r"$c(r_0)=(r_c^2+r_0^2)/(2r_0^4)$")
    ax.set_xlabel(r"target minimum $r_0$")
    ax.set_ylabel("c")
    ax.grid(alpha=0.3)

    fig.suptitle("Radial potential with prescribed minimum", fontsize=16)
    fig.tight_layout()
    outfile = OUT_DIR / "radial_minimum_potential.png"
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(outfile)


if __name__ == "__main__":
    plot_basis()
    plot_radial_minimum_potential()
    plot_angular_modulated_potential()
