"""
Plotting utilities for spectral filtering visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from spectral_solvers import cheb_rect_coeffs, eval_cheb_series, jackson_kernel


def save_or_show(fig, save_path):
    """Save figure to file or display interactively."""
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Saved to {save_path}')
    else:
        plt.show()


def plot_debug_rectangle(band_lo, band_hi, exact_eigs, degs, rect_deg_max, rect_degs, rect_plot, save_path):
    """
    Plot Chebyshev band-pass polynomial p(x) converging to a rectangle for each band.
    Shows how the filter sharpens with increasing degree.
    """
    try:
        degs = [int(s) for s in rect_degs.split(',') if len(s.strip()) > 0]
    except Exception:
        degs = [4, 8, 16, 32, 64, 128]
    degs = [d for d in degs if d > 0 and d <= rect_deg_max]
    if len(degs) == 0:
        degs = [4, 8, 16, 32, 64, 128]

    nbands = len(band_lo)
    x = np.linspace(-1.0, 1.0, 2000)
    fig, axes = plt.subplots(nbands, 1, figsize=(12, 3 * nbands), sharex=True, sharey=True)
    if nbands == 1:
        axes = [axes]

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(degs)))

    for bi, ax in enumerate(axes):
        f_lo, f_hi = band_lo[bi], band_hi[bi]
        for color, d in zip(colors, degs):
            c = cheb_rect_coeffs(f_lo, f_hi, d, use_jackson=True)
            y = eval_cheb_series(x, c)
            if rect_plot == 'abs':
                y = np.abs(y)
            elif rect_plot == 'square':
                y = y * y
            y = np.clip(y, -0.2, 1.2)
            ax.plot(x, y, color=color, linewidth=1.4, label=f'deg={d}')
        ax.axvspan(f_lo, f_hi, alpha=0.15, color='green')
        for eig in exact_eigs:
            ax.axvline(eig, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
        ax.set_ylabel(f'Band {bi}\n[{f_lo:.2f},{f_hi:.2f}]', fontsize=9)
        ax.set_ylim(-0.2, 1.2)
        ax.grid(True, alpha=0.3)
        if bi == 0:
            ax.legend(loc='upper right', fontsize='small')
            ax.set_title('Chebyshev band-pass polynomial p(x) vs degree\n(green=target band, should converge to rectangle)')

    axes[-1].set_xlabel('x')
    save_or_show(fig, save_path)


def plot_convergence_3panel(band_lo, band_hi, exact_w, band_raw_w, band_raw_r, band_keep, final_traj,
                             coarse_iters, square_filter, filter_reps, conv_iters, method, prune_mode,
                             nvec, save_path, x_label="Frequency / Eigenvalue", title_suffix="",
                             band_lo_filt=None, band_hi_filt=None):
    """
    3-panel plot: filter responses (top), convergence trajectories (middle), residuals (bottom).
    All panels share frequency x-axis.
    """
    blo_f = band_lo if band_lo_filt is None else band_lo_filt
    bhi_f = band_hi if band_hi_filt is None else band_hi_filt
    nbands = len(blo_f)
    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.2, 2.2, 1.2])
    ax_filt = fig.add_subplot(gs[0])
    ax_conv = fig.add_subplot(gs[1])
    ax_res = fig.add_subplot(gs[2], sharex=ax_conv)

    # --- Top panel: filter responses (scaled Chebyshev coordinate) ---
    x_lo_f, x_hi_f = float(np.min(blo_f)), float(np.max(blo_f))
    pad_f = 0.02 * max(x_hi_f - x_lo_f, 1e-6)
    x_fine = np.linspace(x_lo_f - pad_f, x_hi_f + pad_f, 2000)
    band_colors = plt.cm.tab10(np.linspace(0, 0.9, nbands))

    for bi in range(nbands):
        f_lo, f_hi = blo_f[bi], bhi_f[bi]
        c_band = cheb_rect_coeffs(f_lo, f_hi, coarse_iters, use_jackson=True)
        y = eval_cheb_series(x_fine, c_band)
        if square_filter:
            y = y * y
        y = np.clip(y, -0.2, 1.2)
        ax_filt.plot(x_fine, y, '-', color=band_colors[bi], lw=1.4,
                    label=f'Band {bi} [{f_lo:.2f},{f_hi:.2f}]')
        ax_filt.axvspan(f_lo, f_hi, alpha=0.08, color=band_colors[bi])

    ax_filt.set_ylabel('Filter response p(x)')
    ax_filt.set_xlabel('scaled eigenvalue (filter panel)')
    ax_filt.set_ylim(-0.2, 1.2)
    ax_filt.grid(True, alpha=0.2)
    ax_filt.legend(loc='upper right', fontsize='x-small', ncol=min(nbands, 3))
    filt_label = f'p(H)^{{2}}' if square_filter else 'p(H)'
    ax_filt.set_title(f'Chebyshev band-pass filters  ({filt_label}, deg={coarse_iters}, '
                      f'filter_reps={filter_reps})', fontsize=11)

    # --- Middle panel: convergence trajectories ---
    for w0 in exact_w:
        ax_conv.axvline(w0, color='red', linestyle='--', alpha=0.35, linewidth=0.8)
    ax_conv.plot([], [], color='red', linestyle='--', alpha=0.35, label='Exact eigenvalues')

    for bi in range(nbands):
        y_off = bi * 0.15
        color = band_colors[bi]
        # scatter all raw points (faint)
        for it in range(len(band_raw_w[bi])):
            ws = band_raw_w[bi][it]
            ax_conv.scatter(ws, np.full_like(ws, it + y_off),
                           c=[color], s=12, alpha=0.25, edgecolors='none')

        # plot each trajectory line until its last kept iteration
        for tid, pts in final_traj[bi].items():
            if len(pts) < 2:
                continue
            xs = [p[1] for p in pts]
            ys = [p[0] + y_off for p in pts]
            last_kept = -1
            for idx, p in enumerate(pts):
                if p[3]:
                    last_kept = idx
            if last_kept < 0:
                last_kept = len(pts) - 1
            ax_conv.plot(xs[:last_kept+1], ys[:last_kept+1],
                        '-', color=color, lw=1.2, alpha=0.7)
            if last_kept >= 0:
                ax_conv.plot(xs[last_kept], ys[last_kept], 'o',
                            color=color, ms=4, alpha=0.9)
            if last_kept < len(pts) - 1 and last_kept >= 0:
                ax_conv.plot(xs[last_kept+1], ys[last_kept+1], 'x',
                            color=color, ms=7, mew=2, alpha=0.9)

    ax_conv.set_ylabel('Iteration (+ band offset)')
    x_lo, x_hi = float(np.min(band_lo)), float(np.max(band_hi))
    pad = 0.02 * max(x_hi - x_lo, 1e-6)
    ax_conv.set_xlim(x_lo - pad, x_hi + pad)
    ax_conv.grid(True, alpha=0.2)
    ax_conv.legend(loc='upper right', fontsize='small')

    # --- Bottom panel: residual trajectories ---
    for bi in range(nbands):
        color = band_colors[bi]
        for tid, pts in final_traj[bi].items():
            if len(pts) < 1:
                continue
            xs = [p[1] for p in pts]
            rs = [p[2] for p in pts]
            last_kept = -1
            for idx, p in enumerate(pts):
                if p[3]:
                    last_kept = idx
            if last_kept < 0:
                last_kept = len(pts) - 1
            plot_to = min(last_kept + 2, len(xs))
            ax_res.semilogy(xs[:plot_to], rs[:plot_to],
                           '-', color=color, lw=1.2, alpha=0.7)
            if last_kept >= 0:
                ax_res.semilogy(xs[last_kept], rs[last_kept], 'o',
                               color=color, ms=4, alpha=0.9)
            if last_kept < len(pts) - 1 and last_kept >= 0:
                ax_res.semilogy(xs[last_kept+1], rs[last_kept+1], 'x',
                               color=color, ms=7, mew=2, alpha=0.9)

    ax_res.set_xlabel(x_label)
    ax_res.set_ylabel('Residual ‖H u - λ u‖')
    ax_res.grid(True, alpha=0.25)

    suptitle = (f'Band-pass Chebyshev + {method.upper()} convergence  '
                f'(nvec={nvec}, conv_iters={conv_iters}, prune={prune_mode})')
    if title_suffix:
        suptitle = f'{title_suffix}\n{suptitle}'
    fig.suptitle(suptitle, fontsize=12, fontweight='bold', y=1.01)

    save_or_show(fig, save_path)


def plot_spectral_evolution(freqs, total_amps, vec_amps, iters, exact_eigs=None,
                            method='chebyshev', normalize=False, save_path=None,
                            x_label="Frequency", title_suffix=""):
    """
    Visualizes the spectral filter evolution (KPM Chebyshev or power iteration).
    Top panel: total amplitude vs frequency for each iteration count.
    Bottom panels: per-vector amplitude strips.
    """
    transform_inverse = (method == 'power')
    num_iters = len(iters)

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, num_iters, height_ratios=[2.5, 1])

    # --- Top Plot: 1D Total Amplitude ---
    ax_top = fig.add_subplot(gs[0, :])

    for n in iters:
        y = total_amps[n].copy()
        if transform_inverse:
            y = 1.0 / (y + 1e-10)
        if normalize:
            ymax = y.max()
            if ymax > 0:
                y /= ymax
        ax_top.plot(freqs, y, label=f'{n} Iterations', linewidth=2)

    if exact_eigs is not None:
        for eig in exact_eigs:
            ax_top.axvline(eig, color='r', linestyle='--', alpha=0.4)

    title = "Chebyshev Filter Evolution" if not transform_inverse else "Power Iteration (1/A Transformation)"
    if normalize:
        title += " (normalized)"
    if title_suffix:
        title = f"{title_suffix}: {title}"
    ax_top.set_title(title, fontsize=14, fontweight='bold')
    ax_top.set_xlabel(x_label)
    ax_top.set_ylabel("Normalized Amplitude" if normalize else ("Transformed Amplitude (1/A)" if transform_inverse else "Total Amplitude"))
    ax_top.set_xlim(freqs[0], freqs[-1])
    ax_top.legend(loc='upper right')
    ax_top.grid(True, alpha=0.3)

    # --- Bottom Plots: imshow strips ---
    for idx, n in enumerate(iters):
        ax_strip = fig.add_subplot(gs[1, idx], sharex=ax_top)

        img_data = vec_amps[n]
        if transform_inverse:
            img_data = 1.0 / (img_data + 1e-10)
        if normalize:
            img_max = img_data.max()
            if img_max > 0:
                img_data = img_data / img_max

        vmin, vmax = np.min(img_data), np.max(img_data)
        ax_strip.imshow(img_data.T, aspect='auto', origin='lower',
                        extent=[freqs[0], freqs[-1], 0, img_data.shape[1]],
                        vmin=vmin, vmax=vmax, cmap='magma', interpolation='nearest')
        ax_strip.set_title(f'Iter {n}')
        ax_strip.set_yticks([])
        ax_strip.set_xlabel(x_label)
        if idx == 0:
            ax_strip.set_ylabel("Random Vectors")

    save_or_show(fig, save_path)
