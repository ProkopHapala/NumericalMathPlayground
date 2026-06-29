"""
test_radial_grids.py — Compare established radial grid mappings for 2D integration.

Tests: Power law, Treutler-Ahlrichs M4, Becke, Log3, MultiExp, Euler-Maclaurin.
Uses atomic orbital product trial functions (same as weight optimization).

The 2D integral in polar coordinates is:
  ∫∫ f(x,y) dA = ∫₀^∞ ∫₀^{2π} f(r,θ) r dr dθ

Each radial mapping provides (r_i, w_r_i) where w_r_i includes the Jacobian r*dr.
The angular grid is uniform with N_ang points (trapezoidal rule, exact for high l).
"""

import numpy as np
import matplotlib.pyplot as plt
from BasisFunctions import eval_trial_functions, trial_analytic_integrals, build_atomic_trial_set


# ── Radial grid mappings ──────────────────────────────────────────────────────

def power_law_grid(N, R, alpha):
    """Power law: r = R * t^alpha, midpoint rule on uniform t in [0,1].

    Maps [0,1] → [0,R]. Simple but suboptimal — constant spacing at large r
    instead of logarithmic.
    """
    t = (np.arange(N) + 0.5) / N
    dt = 1.0 / N
    r = R * t**alpha
    drdt = R * alpha * t**(alpha - 1)
    w_r = r * drdt * dt  # Jacobian r * dr
    return r, w_r


def treutler_ahlrichs_grid(N, xi=1.0):
    """Treutler-Ahlrichs M4 mapping with Chebyshev 2nd kind quadrature.

    r(x) = -(ξ/ln2) * (1+x)^0.6 * ln((1-x)/2)
    x ∈ [-1,1] at Gauss-Chebyshev 2nd kind nodes.

    The standard in quantum chemistry (TURBOMOLE, ORCA, PySCF).
    Combines logarithmic spacing near origin with power-law spreading.
    """
    step = np.pi / (N + 1)
    k = np.arange(1, N + 1)
    x = np.cos(k * step)
    ln2 = xi / np.log(2)
    r = -ln2 * (1 + x)**0.6 * np.log((1 - x) / 2)
    rprime = ln2 * (1 + x)**0.6 * (
        -0.6 / (1 + x) * np.log((1 - x) / 2) + 1.0 / (1 - x)
    )
    dr = step * np.sin(k * step) * rprime
    w_r = r * dr
    idx = np.argsort(r)
    return r[idx], w_r[idx]


def becke_grid(N, R=1.0):
    """Becke mapping with Chebyshev 2nd kind quadrature.

    r(x) = R * (1+x)/(1-x)
    Simple rational mapping, concentrates near r=0.
    """
    step = np.pi / (N + 1)
    k = np.arange(1, N + 1)
    x = np.cos(k * step)
    r = R * (1 + x) / (1 - x)
    rprime = R * 2.0 / (1 - x)**2
    dr = step * np.sin(k * step) * rprime
    w_r = r * dr
    idx = np.argsort(r)
    return r[idx], w_r[idx]


def log3_grid(N, R=1.0):
    """Mura-Knowles Log3 mapping with Gauss-Legendre quadrature on [0,1].

    r(t) = -R * ln(1 - t³)
    Logarithmic mapping, good for Gaussian decays.
    """
    t, w_gl = np.polynomial.legendre.leggauss(N)
    t = (t + 1) / 2
    w_gl = w_gl / 2
    r = -R * np.log(1 - t**3)
    rprime = R * 3 * t**2 / (1 - t**3)
    w_r = r * rprime * w_gl
    idx = np.argsort(r)
    return r[idx], w_r[idx]


def multiexp_grid(N, R=1.0):
    """MultiExp mapping with Gauss-Legendre quadrature on [0,1].

    r(t) = -R * ln(1 - t)
    Simplest logarithmic mapping.
    """
    t, w_gl = np.polynomial.legendre.leggauss(N)
    t = (t + 1) / 2
    w_gl = w_gl / 2
    r = -R * np.log(1 - t)
    rprime = R / (1 - t)
    w_r = r * rprime * w_gl
    idx = np.argsort(r)
    return r[idx], w_r[idx]


def euler_maclaurin_grid(N, R=1.0, m=2):
    """Euler-Maclaurin (Murray-Handy-Laming) mapping.

    r(k) = R * (k/(n-k))^m, k=1..N, n=N+1
    Equispaced k with trapezoidal rule (EM corrections vanish at endpoints).
    """
    n = N + 1
    k = np.arange(1, N + 1)
    r = R * (k / (n - k))**m
    w_r = R**2 * m * n * k**(2*m - 1) / (n - k)**(2*m + 1)
    return r, w_r


# ── 2D polar grid ─────────────────────────────────────────────────────────────

def build_polar_grid(r, w_r, N_ang=128):
    """Build 2D polar grid from radial points/weights and uniform angular grid.

    Returns (xy, w) where xy are (N_r*N_ang, 2) point coordinates and
    w are the corresponding quadrature weights.
    """
    theta = np.linspace(0, 2 * np.pi, N_ang, endpoint=False)
    w_theta = 2 * np.pi / N_ang
    R_grid, T_grid = np.meshgrid(r, theta, indexing='ij')
    xy = np.column_stack([
        (R_grid * np.cos(T_grid)).ravel(),
        (R_grid * np.sin(T_grid)).ravel(),
    ])
    w = np.outer(w_r, np.full(N_ang, w_theta)).ravel()
    return xy, w


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    trial_set = build_atomic_trial_set(
        elements=['H', 'C', 'N', 'O'], basis_name='cc-pVDZ',
        n_angular_dirs=8, n_combos=500, seed=42)
    I_exact = trial_analytic_integrals(trial_set)

    N_ang = 128  # high enough to make angular error negligible
    N_r_values = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100]

    # All mappings scaled to cover the same physical region (R_cut = 9.71 Bohr = 5.14 Å)
    # At r=10 Bohr, basis functions are ~1e-6, trial functions ~1e-13 → negligible
    R_cut = 9.71

    mappings = {
        'Power α=1.0':         lambda N: power_law_grid(N, R=R_cut, alpha=1.0),
        'Power α=1.8':         lambda N: power_law_grid(N, R=R_cut, alpha=1.8),
        'Power α=3.0':         lambda N: power_law_grid(N, R=R_cut, alpha=3.0),
        'TA-M4 ξ=1.0':         lambda N: treutler_ahlrichs_grid(N, xi=1.0),
        'TA-M4 ξ=2.0':         lambda N: treutler_ahlrichs_grid(N, xi=2.0),
        'Becke R=1.0':         lambda N: becke_grid(N, R=1.0),
        'Log3 R=1.0':          lambda N: log3_grid(N, R=1.0),
        'MultiExp R=1.0':      lambda N: multiexp_grid(N, R=1.0),
        'EM R=1.0 m=2':        lambda N: euler_maclaurin_grid(N, R=1.0, m=2),
    }

    # Compute errors
    results = {}
    print(f"\n{'Mapping':25s}  " + "  ".join(f"N={n:3d}" for n in N_r_values))
    print("-" * 120)

    for name, grid_func in mappings.items():
        errors = []
        for N_r in N_r_values:
            r, w_r = grid_func(N_r)
            xy, w = build_polar_grid(r, w_r, N_ang)
            F = eval_trial_functions(xy, trial_set)
            I_num = F @ w
            rel_err = np.mean(np.abs(I_num - I_exact) / (np.abs(I_exact) + 1e-30) * 100)
            errors.append(rel_err)
        results[name] = errors
        print(f"  {name:23s}  " + "  ".join(f"{e:7.4f}%" for e in errors))

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Error vs N_r (semilogy)
    ax = axes[0, 0]
    for name, errors in results.items():
        ax.semilogy(N_r_values, errors, 'o-', label=name, markersize=4)
    ax.set_xlabel('N_r (radial shells)')
    ax.set_ylabel('Mean relative error (%)')
    ax.set_title('Radial grid convergence (2D polar, N_ang=128)')
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(True, alpha=0.3)

    # 2. Radial point distribution (semilogx)
    ax = axes[0, 1]
    N_show = 20
    for name, grid_func in mappings.items():
        r, w_r = grid_func(N_show)
        ax.plot(np.arange(1, len(r) + 1), r, 'o-', label=name, markersize=3)
    ax.set_xlabel('Shell index')
    ax.set_ylabel('r (Bohr)')
    ax.set_title(f'Radial point positions (N_r={N_show})')
    ax.set_yscale('log')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 3. Weight distribution (loglog)
    ax = axes[1, 0]
    for name, grid_func in mappings.items():
        r, w_r = grid_func(N_show)
        ax.loglog(r, w_r, 'o-', label=name, markersize=3)
    ax.set_xlabel('r (Bohr)')
    ax.set_ylabel('w_r (radial weight incl. Jacobian)')
    ax.set_title(f'Weight distribution (N_r={N_show})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 4. Spacing dr/di
    ax = axes[1, 1]
    for name, grid_func in mappings.items():
        r, w_r = grid_func(N_show)
        dr = np.diff(r)
        ax.semilogy(np.arange(len(dr)), dr, 'o-', label=name, markersize=3)
    ax.set_xlabel('Shell index')
    ax.set_ylabel('dr (radial spacing)')
    ax.set_title(f'Radial spacing (N_r={N_show})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Quadrature_Images/radial_grid_comparison.png', dpi=150)
    print(f"\nSaved: Quadrature_Images/radial_grid_comparison.png")

    # ── 2D grid visualization (same region for all) ──
    N_show_2d = 20
    N_ang_show = 32  # fewer angular points for visual clarity
    n_maps = len(mappings)
    ncols = 3
    nrows = int(np.ceil(n_maps / ncols))
    plot_lim = R_cut * 1.05  # same axes limits for all subplots

    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.5))
    axes2 = axes2.ravel()

    # Global weight range for consistent color scale
    all_w = []
    for name, grid_func in mappings.items():
        r, w_r = grid_func(N_show_2d)
        xy, w = build_polar_grid(r, w_r, N_ang_show)
        all_w.append(w)
    w_global_max = max(np.max(w) for w in all_w)

    for idx, (name, grid_func) in enumerate(mappings.items()):
        ax = axes2[idx]
        r, w_r = grid_func(N_show_2d)
        xy, w = build_polar_grid(r, w_r, N_ang_show)

        # Filter to plot region
        mask = np.sqrt(xy[:, 0]**2 + xy[:, 1]**2) <= plot_lim
        xy_plot, w_plot = xy[mask], w[mask]

        # Scale point size by weight (sqrt for area)
        sizes = np.sqrt(w_plot) * 300 / (np.sqrt(w_global_max) + 1e-30)
        ax.scatter(xy_plot[:, 0], xy_plot[:, 1], s=sizes, c=w_plot,
                   cmap='viridis', vmin=0, vmax=w_global_max,
                   edgecolors='none', alpha=0.8)

        # Draw radial shells
        for ri in r:
            if ri <= plot_lim:
                circle = plt.Circle((0, 0), ri, fill=False, color='gray',
                                    linewidth=0.3, alpha=0.3)
                ax.add_patch(circle)

        ax.set_aspect('equal')
        ax.set_xlim(-plot_lim, plot_lim)
        ax.set_ylim(-plot_lim, plot_lim)
        ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('x (Bohr)', fontsize=8)
        ax.set_ylabel('y (Bohr)', fontsize=8)

    for idx in range(n_maps, len(axes2)):
        axes2[idx].set_visible(False)

    fig2.suptitle(f'2D polar grids — same region [-{R_cut}, {R_cut}] Bohr '
                  f'(N_r={N_show_2d}, N_ang={N_ang_show})\n'
                  f'point size ∝ √weight, color = weight', fontsize=12)
    plt.tight_layout()
    plt.savefig('Quadrature_Images/radial_grid_2d.png', dpi=150)
    print(f"Saved: Quadrature_Images/radial_grid_2d.png")

    # ── Zoomed 2D grid (inner region only) ──
    fig3, axes3 = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.5))
    axes3 = axes3.ravel()

    zoom_r = 3.0  # show only r < 3 Bohr

    for idx, (name, grid_func) in enumerate(mappings.items()):
        ax = axes3[idx]
        r, w_r = grid_func(N_show_2d)
        xy, w = build_polar_grid(r, w_r, N_ang_show)

        mask = np.sqrt(xy[:, 0]**2 + xy[:, 1]**2) <= zoom_r
        sizes = np.sqrt(w[mask]) * 300 / (np.sqrt(w_global_max) + 1e-30)
        ax.scatter(xy[mask, 0], xy[mask, 1], s=sizes, c=w[mask],
                   cmap='viridis', vmin=0, vmax=w_global_max,
                   edgecolors='none', alpha=0.8)

        for ri in r:
            if ri <= zoom_r:
                circle = plt.Circle((0, 0), ri, fill=False, color='gray',
                                    linewidth=0.3, alpha=0.3)
                ax.add_patch(circle)

        ax.set_aspect('equal')
        ax.set_xlim(-zoom_r * 1.05, zoom_r * 1.05)
        ax.set_ylim(-zoom_r * 1.05, zoom_r * 1.05)
        ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('x (Bohr)', fontsize=8)
        ax.set_ylabel('y (Bohr)', fontsize=8)

    for idx in range(n_maps, len(axes3)):
        axes3[idx].set_visible(False)

    fig3.suptitle(f'2D polar grids zoomed to r ≤ {zoom_r} Bohr '
                  f'(N_r={N_show_2d}, N_ang={N_ang_show})', fontsize=12)
    plt.tight_layout()
    plt.savefig('Quadrature_Images/radial_grid_2d_zoom.png', dpi=150)
    print(f"Saved: Quadrature_Images/radial_grid_2d_zoom.png")


if __name__ == '__main__':
    main()
