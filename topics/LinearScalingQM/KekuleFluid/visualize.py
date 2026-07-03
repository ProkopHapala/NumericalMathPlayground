"""Visualization layer for hexagonal lattice solvers.

Physical background
-------------------
The Kekulé order parameter z_i is complex: its phase arg(z_i) selects one of
three dimerization patterns, and its amplitude |z_i| measures the alternation
strength.  The HSV color mapping (Hue=phase, Value=amplitude) provides an
intuitive visualization where color encodes the Kekulé pattern and brightness
encodes the strength.  Nodal lines (Re(z)=0, Im(z)=0) are drawn as smooth
zero-contours on the interpolated field; their crossings reveal vortex cores
where the phase winds by ±2π.

Role in the system
------------------
This module provides all plotting and visualization for Model A and Model B
outputs.  It works with any `HoneycombGraph` and accepts z_real, z_imag,
bond_x arrays.  It is used by `run_dirac_lattice.py`, `demo_proton.py`,
`demo.py`, `demo_dirac_lattice.py`, and `main.py`.

Key classes
-----------
- `HexVisualizer` — static plotting on a honeycomb graph:
    - `plot_phase_hsv(...)` — HSV phase field with smooth Gaussian interpolation,
      zero-contour nodal lines, low-|z| domain wall contour, color wheel inset.
    - `plot_bond_order(...)` — bonds colored/thickened by bond order.
    - `plot_z_field(...)` — amplitude or phase scatter plot on atoms.
    - `plot_domain_wall(...)` — low-|z| contour (phase-convention independent).
    - `plot_nodal_lines(...)` — discrete bond-based nodal line plot.
    - `plot_combined(...)` — 4-panel summary figure.
    - `_compute_phase_grid(...)` — Gaussian-weighted or Voronoi interpolation of
      complex z onto a 2D grid; returns RGB image, amplitude grid, Re/Im grids.
    - `_add_color_wheel(...)` — inset color wheel showing hue→phase mapping.
- `LiveViewer` — real-time multi-panel animation with in-place artist updates:
    - 4-panel (amplitude, phase, bond order, HSV+nodal) or 6-panel (+winding, closeup).
    - Smooth Gaussian interpolation for HSV background and contour-based nodal lines.
    - `update(z_real, z_imag, bond_x, ...)` — update all panels without recreating artists.
    - `mark_defect(positions)` — mark defect positions with stars on all panels.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from typing import Optional, Callable, List
import colorsys


class HexVisualizer:
    """Visualize fields on a honeycomb graph."""

    def __init__(self, graph, figsize=(10, 8)):
        self.graph = graph
        self.figsize = figsize
        self._pos = np.array([[a.pos[0], a.pos[1]] for a in graph.atoms])

        # Precompute bond segments
        self._bond_segs = np.array([
            [self._pos[b.iA], self._pos[b.iB]] for b in graph.bonds
        ]) if graph.nbond > 0 else np.zeros((0, 2, 2))

        # Ring centers
        self._ring_centers = []
        for ring in graph.rings:
            pts = self._pos[ring]
            self._ring_centers.append(pts.mean(axis=0))
        self._ring_centers = np.array(self._ring_centers) if self._ring_centers else np.zeros((0, 2))

    def plot_bond_order(self, bond_x, ax=None, title="Bond Order",
                        cmap='RdYlBu_r', vmin=None, vmax=None, show_atoms=True,
                        add_colorbar=None):
        """Plot bonds colored by bond order. Atoms shown as dots."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            if add_colorbar is None:
                add_colorbar = True
        else:
            fig = ax.figure
            if add_colorbar is None:
                add_colorbar = False

        if len(self._bond_segs) == 0:
            ax.set_title("No bonds")
            return fig, ax

        bx = np.asarray(bond_x)
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = max(1.0, bx.max() + 0.01)

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.cm.get_cmap(cmap)

        colors = cmap_obj(norm(bx))
        linewidths = 1.0 + 4.0 * bx

        lc = LineCollection(self._bond_segs, colors=colors, linewidths=linewidths)
        ax.add_collection(lc)

        if show_atoms:
            ax.scatter(self._pos[:, 0], self._pos[:, 1], c='black', s=8, zorder=5)

        ax.set_aspect('equal')
        ax.autoscale_view()
        ax.set_title(title)
        if add_colorbar:
            fig.colorbar(lc, ax=ax, label='bond order')
        return fig, ax

    def plot_z_field(self, z_real, z_imag, mode='amplitude', ax=None,
                     title=None, cmap='viridis', show_bonds=True, bond_x=None,
                     add_colorbar=None):
        """Plot the complex Kekulé order parameter z on atoms.

        mode: 'amplitude' (|z|), 'phase' (arg(z)), 'real' (Re z), 'imag' (Im z)
        add_colorbar: True/False to force, None = auto (only when creating new figure)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            if add_colorbar is None:
                add_colorbar = True
        else:
            fig = ax.figure
            if add_colorbar is None:
                add_colorbar = False

        z = z_real + 1j * z_imag

        if mode == 'amplitude':
            vals = np.abs(z)
            if title is None: title = '|z| (Kekulé amplitude)'
            norm = mcolors.Normalize(vmin=0, vmax=max(vals.max(), 0.01))
        elif mode == 'phase':
            vals = np.angle(z)
            if title is None: title = 'arg(z) (Kekulé phase)'
            norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
            cmap = 'twilight'
        elif mode == 'real':
            vals = z_real
            if title is None: title = 'Re(z)'
            vmax = max(np.abs(vals).max(), 0.01)
            norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
            cmap = 'seismic'
        elif mode == 'imag':
            vals = z_imag
            if title is None: title = 'Im(z)'
            vmax = max(np.abs(vals).max(), 0.01)
            norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
            cmap = 'seismic'
        else:
            raise ValueError(f"Unknown mode: {mode}")

        cmap_obj = plt.cm.get_cmap(cmap)
        colors = cmap_obj(norm(vals))

        if show_bonds and bond_x is not None:
            bx = np.asarray(bond_x)
            bond_colors = np.ones((len(self._bond_segs), 4))
            bond_colors[:, :3] = 0.7
            bond_colors[:, 3] = 0.3 + 0.5 * bx
            lw = 0.5 + 2.0 * bx
            lc = LineCollection(self._bond_segs, colors=bond_colors, linewidths=lw, zorder=1)
            ax.add_collection(lc)

        sc = ax.scatter(self._pos[:, 0], self._pos[:, 1], c=vals, cmap=cmap,
                        s=60, norm=norm, zorder=5, edgecolors='black', linewidths=0.3)
        if add_colorbar:
            fig.colorbar(sc, ax=ax, label=mode)

        ax.set_aspect('equal')
        ax.autoscale_view()
        ax.set_title(title)
        return fig, ax

    def plot_nodal_lines(self, z_real, z_imag, ax=None, title="Nodal Lines"):
        """Plot nodal lines where Re(z) or Im(z) changes sign across bonds."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Draw all bonds faintly
        faint = np.ones((len(self._bond_segs), 4))
        faint[:, :3] = 0.8
        faint[:, 3] = 0.2
        lc_faint = LineCollection(self._bond_segs, colors=faint, linewidths=0.5)
        ax.add_collection(lc_faint)

        # Find Re nodal crossings
        re_segs = []
        im_segs = []
        for b, bond in enumerate(self.graph.bonds):
            i, j = bond.iA, bond.iB
            if z_real[i] * z_real[j] < 0:
                re_segs.append(self._bond_segs[b])
            if z_imag[i] * z_imag[j] < 0:
                im_segs.append(self._bond_segs[b])

        if re_segs:
            lc_re = LineCollection(re_segs, colors='red', linewidths=2.5, label='Re(z)=0')
            ax.add_collection(lc_re)
        if im_segs:
            lc_im = LineCollection(im_segs, colors='blue', linewidths=2.5, label='Im(z)=0')
            ax.add_collection(lc_im)

        ax.scatter(self._pos[:, 0], self._pos[:, 1], c='black', s=5, zorder=5)

        ax.set_aspect('equal')
        ax.autoscale_view()
        ax.set_title(title)
        if re_segs or im_segs:
            ax.legend()
        return fig, ax

    def plot_ring_winding(self, z_real, z_imag, ax=None, title="Ring Winding",
                          add_colorbar=None):
        """Plot rings colored by winding number."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            if add_colorbar is None:
                add_colorbar = True
        else:
            fig = ax.figure
            if add_colorbar is None:
                add_colorbar = False

        z = z_real + 1j * z_imag

        # Draw bonds faintly
        faint = np.ones((len(self._bond_segs), 4))
        faint[:, :3] = 0.8
        faint[:, 3] = 0.2
        lc = LineCollection(self._bond_segs, colors=faint, linewidths=0.5)
        ax.add_collection(lc)

        windings = []
        for ring in self.graph.rings:
            w = self._ring_winding(ring, z)
            windings.append(w)

        windings = np.array(windings)
        if len(windings) > 0:
            vmax = max(np.abs(windings).max(), 1.0)
            norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
            cmap = plt.cm.get_cmap('seismic')
            colors = cmap(norm(windings))

            for idx, ring in enumerate(self.graph.rings):
                pts = self._pos[ring]
                poly = plt.Polygon(pts, facecolor=colors[idx], alpha=0.5,
                                   edgecolor='gray', linewidth=0.5)
                ax.add_patch(poly)

            if add_colorbar:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array(windings)
                fig.colorbar(sm, ax=ax, label='winding / 2π')

        ax.scatter(self._pos[:, 0], self._pos[:, 1], c='black', s=5, zorder=5)
        ax.set_aspect('equal')
        ax.autoscale_view()
        ax.set_title(title)
        return fig, ax

    def _ring_winding(self, ring, z):
        """Compute winding number for a ring."""
        total = 0.0
        n = len(ring)
        for k in range(n):
            i = ring[k]
            j = ring[(k + 1) % n]
            dphi = np.angle(z[j]) - np.angle(z[i])
            while dphi > np.pi:
                dphi -= 2 * np.pi
            while dphi < -np.pi:
                dphi += 2 * np.pi
            total += dphi
        return total / (2 * np.pi)

    def _compute_phase_grid(self, z_real, z_imag, resolution=200, mode='voronoi'):
        """Interpolate complex z onto a 2D grid and convert to HSV.

        Returns (rgb_image, extent, grid_amp, z_max) where:
        - rgb_image: (resolution, resolution, 3) uint8 array
        - extent: (xmin, xmax, ymin, ymax) for imshow
        - grid_amp: (resolution, resolution) amplitude field (for contour overlays)
        - z_max: max |z| (for normalization)

        HSV mapping:
        - H = (arg(z) + pi) / (2*pi)   — phase as hue (rainbow)
        - S = 1                         — full saturation
        - V = min(|z| / z_max, 1)      — amplitude as brightness

        mode: 'voronoi' for nearest-atom (blocky but honest),
              'smooth' for Gaussian-weighted interpolation
        """
        pos = self._pos
        z = z_real + 1j * z_imag
        amp = np.abs(z)
        phase = np.angle(z)
        z_max = max(amp.max(), 1e-8)

        xmin, xmax = pos[:, 0].min() - 0.5, pos[:, 0].max() + 0.5
        ymin, ymax = pos[:, 1].min() - 0.5, pos[:, 1].max() + 0.5

        gx = np.linspace(xmin, xmax, resolution)
        gy = np.linspace(ymin, ymax, resolution)
        GX, GY = np.meshgrid(gx, gy)

        if mode == 'smooth':
            sigma = 1.5 * self.graph.aCC
            re_grid = np.zeros((resolution, resolution))
            im_grid = np.zeros((resolution, resolution))
            weight_sum = np.zeros((resolution, resolution))
            for i in range(len(pos)):
                dx = GX - pos[i, 0]
                dy = GY - pos[i, 1]
                w = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                re_grid += w * z_real[i]
                im_grid += w * z_imag[i]
                weight_sum += w
            weight_sum = np.maximum(weight_sum, 1e-30)
            re_grid /= weight_sum
            im_grid /= weight_sum
            grid_amp = np.sqrt(re_grid**2 + im_grid**2)
            grid_phase = np.angle(re_grid + 1j * im_grid)
        else:
            re_grid = np.zeros((resolution, resolution))
            im_grid = np.zeros((resolution, resolution))
            grid_amp = np.zeros((resolution, resolution))
            grid_phase = np.zeros((resolution, resolution))
            min_dist = np.full((resolution, resolution), 1e18)
            for i in range(len(pos)):
                dx = GX - pos[i, 0]
                dy = GY - pos[i, 1]
                d2 = dx * dx + dy * dy
                mask = d2 < min_dist
                min_dist[mask] = d2[mask]
                grid_amp[mask] = amp[i]
                grid_phase[mask] = phase[i]
                re_grid[mask] = z_real[i]
                im_grid[mask] = z_imag[i]

        H = np.clip((grid_phase + np.pi) / (2 * np.pi), 0.0, 1.0)
        S = np.ones_like(H)
        V = np.clip(grid_amp / z_max, 0.0, 1.0)

        hsv = np.stack([H, S, V], axis=-1)
        rgb = mcolors.hsv_to_rgb(hsv)
        rgb = (rgb * 255).astype(np.uint8)

        extent = (xmin, xmax, ymin, ymax)
        return rgb, extent, grid_amp, z_max, re_grid, im_grid

    def _add_color_wheel(self, ax, size=0.15):
        """Add a color wheel inset showing hue→phase angle mapping.

        The three Kekulé minima (0, 2π/3, 4π/3) are marked.
        """
        n = 100
        theta = np.linspace(0, 2*np.pi, n)
        r = np.linspace(0, 1, n)
        T, R = np.meshgrid(theta, r)
        H = np.clip((T + np.pi) / (2 * np.pi), 0.0, 1.0)
        S = np.ones_like(H)
        V = np.clip(R, 0.0, 1.0)
        hsv = np.stack([H, S, V], axis=-1)
        rgb = mcolors.hsv_to_rgb(hsv)
        rgb = (rgb * 255).astype(np.uint8)

        ax_inset = ax.inset_axes([1.02, 0.7, size, size], transform=ax.transAxes)
        ax_inset.imshow(rgb, extent=[-1, 1, -1, 1], origin='lower', aspect='equal')
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_title('phase', fontsize=8, pad=2)

        for phi, label in [(0, '0'), (2*np.pi/3, '2π/3'), (4*np.pi/3, '4π/3')]:
            x = 0.85 * np.cos(phi - np.pi)
            y = 0.85 * np.sin(phi - np.pi)
            ax_inset.plot(x, y, 'o', color='white', markersize=4, markeredgecolor='black')

        return ax_inset

    def plot_domain_wall(self, z_real, z_imag, ax=None, threshold=0.3,
                         title='Domain wall (low |z|)', defect_indices=None):
        """Plot the physical domain wall as a low-|z| contour overlay.

        Unlike Re(z)=0/Im(z)=0 nodal lines, the low-|z| region is
        phase-convention independent and represents the true
        amplitude suppression zone.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        z = z_real + 1j * z_imag
        amp = np.abs(z)
        z_max = max(amp.max(), 1e-8)
        rel_amp = amp / z_max

        sc = ax.scatter(self._pos[:, 0], self._pos[:, 1], c=rel_amp,
                        cmap='hot_r', s=60, vmin=0, vmax=1,
                        edgecolors='black', linewidths=0.3, zorder=5)

        low_mask = rel_amp < threshold
        if low_mask.any():
            ax.scatter(self._pos[low_mask, 0], self._pos[low_mask, 1],
                       facecolors='none', edgecolors='cyan', s=120,
                       linewidths=2, zorder=6, label=f'|z|<{threshold:.1f}·|z|_max')
            ax.legend(fontsize=8, loc='upper right')

        lc = LineCollection(self._bond_segs, colors='gray',
                            linewidths=0.5, alpha=0.3, zorder=1)
        ax.add_collection(lc)

        if defect_indices is not None:
            for idx in defect_indices:
                ax.plot(self._pos[idx, 0], self._pos[idx, 1], '*',
                        color='lime', markersize=15, markeredgecolor='black',
                        markeredgewidth=0.5, zorder=10)

        ax.set_aspect('equal')
        ax.autoscale_view()
        ax.set_title(title)
        fig.colorbar(sc, ax=ax, label='|z|/|z|_max')
        return fig, ax

    def plot_phase_hsv(self, z_real, z_imag, bond_x=None, ax=None,
                       title="Kekulé phase (HSV)", show_nodal=True,
                       defect_indices=None, resolution=200,
                       mode='voronoi', color_wheel=True):
        """Plot complex z as HSV rainbow imshow with graph overlay.

        HSV mapping:
        - Hue  = arg(z)       (phase → rainbow color)
        - Sat  = 1            (full)
        - Val  = |z|/|z|_max  (amplitude → brightness)

        The graph (bonds + atoms) is overlaid on top.  Nodal lines
        (where Re(z) or Im(z) changes sign across bonds) are highlighted.

        defect_indices: list of atom indices to mark with stars.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # --- HSV background ---
        rgb, extent, grid_amp, z_max, re_grid, im_grid = self._compute_phase_grid(
            z_real, z_imag, resolution=resolution, mode=mode)
        ax.imshow(rgb, extent=extent, origin='lower', aspect='equal', zorder=0)

        # --- Grid coordinates for contours ---
        gx = np.linspace(extent[0], extent[1], grid_amp.shape[1])
        gy = np.linspace(extent[2], extent[3], grid_amp.shape[0])
        GX, GY = np.meshgrid(gx, gy)

        # --- Low-|z| contour (physical domain wall, phase-convention independent) ---
        if show_nodal:
            ax.contour(GX, GY, grid_amp / z_max, levels=[0.3],
                       colors='cyan', linewidths=1.5, alpha=0.6, zorder=3)

        # --- Bonds overlay ---
        if bond_x is not None:
            bx = np.asarray(bond_x)
            bond_colors = np.ones((len(self._bond_segs), 4))
            bond_colors[:, :3] = 0.3
            bond_colors[:, 3] = 0.4 + 0.4 * bx
            lw = 0.5 + 2.0 * bx
            lc = LineCollection(self._bond_segs, colors=bond_colors,
                                linewidths=lw, zorder=2)
            ax.add_collection(lc)

        # --- Smooth zero-contour nodal lines ---
        if show_nodal:
            ax.contour(GX, GY, re_grid, levels=[0],
                       colors='white', linewidths=2.0, alpha=0.8, zorder=3)
            ax.contour(GX, GY, im_grid, levels=[0],
                       colors='black', linewidths=2.0, alpha=0.8, zorder=3)
            # Proxy artists for legend
            ax.plot([], [], color='white', linewidth=2, label='Re(z)=0')
            ax.plot([], [], color='black', linewidth=2, label='Im(z)=0')

        # --- Atoms ---
        z = z_real + 1j * z_imag
        amp = np.abs(z)
        ax.scatter(self._pos[:, 0], self._pos[:, 1], c='black', s=10,
                   zorder=4, edgecolors='white', linewidths=0.2)

        # --- Defect markers ---
        if defect_indices is not None:
            for idx in defect_indices:
                ax.plot(self._pos[idx, 0], self._pos[idx, 1], '*',
                        color='lime', markersize=15, markeredgecolor='black',
                        markeredgewidth=0.5, zorder=10)

        ax.set_aspect('equal')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(title)
        if show_nodal:
            ax.legend(loc='upper right', fontsize=8)
        if color_wheel:
            self._add_color_wheel(ax)
        return fig, ax

    def plot_combined(self, z_real, z_imag, bond_x, figsize=None):
        """4-panel figure: amplitude, phase, bond order, nodal lines."""
        figsize = figsize or (16, 14)
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        self.plot_z_field(z_real, z_imag, mode='amplitude', ax=axes[0, 0],
                          bond_x=bond_x, title='|z| (Kekulé amplitude)',
                          add_colorbar=True)
        self.plot_z_field(z_real, z_imag, mode='phase', ax=axes[0, 1],
                          bond_x=bond_x, title='arg(z) (Kekulé phase)',
                          add_colorbar=True)
        self.plot_bond_order(bond_x, ax=axes[1, 0], title='Bond order',
                             add_colorbar=True)
        self.plot_nodal_lines(z_real, z_imag, ax=axes[1, 1])

        plt.tight_layout()
        return fig, axes

    def animate(self, solver, nsteps, interval=50, mode='combined',
                callback_interval=5, save_path=None):
        """Animate the solver evolution.

        mode: 'amplitude', 'phase', 'bond_order', 'combined'
        """
        frames_data = []

        def capture(slv, step):
            state = slv.get_state()
            frames_data.append((step, state.copy()))

        solver.run(nsteps, callback=capture, callback_interval=callback_interval)

        if mode == 'combined':
            return self._animate_combined(frames_data, interval, save_path)
        else:
            return self._animate_single(frames_data, mode, interval, save_path)

    def _animate_single(self, frames_data, mode, interval, save_path):
        fig, ax = plt.subplots(figsize=self.figsize)

        def update(frame_idx):
            ax.clear()
            step, state = frames_data[frame_idx]
            if mode == 'amplitude':
                self.plot_z_field(state['z_real'], state['z_imag'], mode='amplitude',
                                  ax=ax, bond_x=state['bond_x'], title=f'|z| step={step}')
            elif mode == 'phase':
                self.plot_z_field(state['z_real'], state['z_imag'], mode='phase',
                                  ax=ax, bond_x=state['bond_x'], title=f'arg(z) step={step}')
            elif mode == 'bond_order':
                self.plot_bond_order(state['bond_x'], ax=ax, title=f'Bond order step={step}')

        anim = FuncAnimation(fig, update, frames=len(frames_data),
                             interval=interval, repeat=True)
        if save_path:
            anim.save(save_path, writer='pillow')
        plt.show()
        return anim

    def _animate_combined(self, frames_data, interval, save_path):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        def update(frame_idx):
            for ax in axes.flat:
                ax.clear()
            step, state = frames_data[frame_idx]
            self.plot_z_field(state['z_real'], state['z_imag'], mode='amplitude',
                              ax=axes[0, 0], bond_x=state['bond_x'],
                              title=f'|z| step={step}')
            self.plot_z_field(state['z_real'], state['z_imag'], mode='phase',
                              ax=axes[0, 1], bond_x=state['bond_x'],
                              title=f'arg(z) step={step}')
            self.plot_bond_order(state['bond_x'], ax=axes[1, 0],
                                 title=f'Bond order step={step}')
            self.plot_nodal_lines(state['z_real'], state['z_imag'],
                                  ax=axes[1, 1], title=f'Nodal lines step={step}')

        anim = FuncAnimation(fig, update, frames=len(frames_data),
                             interval=interval, repeat=True)
        if save_path:
            anim.save(save_path, writer='pillow')
        plt.show()
        return anim


class LiveViewer:
    """Live updating viewer with persistent matplotlib artists.

    Instead of clearing and re-plotting each frame (which causes colorbar
    accumulation and performance issues), this creates all artists once
    and updates their data in-place via set_array / set_segments / set_colors.
    """

    def __init__(self, viz, n_panels=4, figsize=None, titles=None):
        self.viz = viz
        self.graph = viz.graph
        self.n_panels = n_panels

        if n_panels == 4:
            figsize = figsize or (14, 12)
            self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
            self.ax_amp, self.ax_phase = self.axes[0, 0], self.axes[0, 1]
            self.ax_bond, self.ax_nodal = self.axes[1, 0], self.axes[1, 1]
            self.ax_winding = None
            self.ax_closeup = None
        elif n_panels == 6:
            figsize = figsize or (18, 12)
            self.fig, self.axes = plt.subplots(2, 3, figsize=figsize)
            self.ax_amp, self.ax_phase, self.ax_bond = self.axes[0]
            self.ax_nodal, self.ax_winding, self.ax_closeup = self.axes[1]
        else:
            raise ValueError("n_panels must be 4 or 6")

        self._init_artists()

        if titles:
            for ax, t in zip(self.axes.flat, titles):
                ax.set_title(t)

        plt.tight_layout()

    def _init_artists(self):
        pos = self.viz._pos
        segs = self.viz._bond_segs
        nb = len(segs)
        na = len(pos)

        # --- Amplitude panel ---
        ax = self.ax_amp
        ax.set_aspect('equal')
        self.bond_bg_amp = LineCollection(segs, colors=np.ones((nb, 4)),
                                          linewidths=np.ones(nb), zorder=1)
        ax.add_collection(self.bond_bg_amp)
        self.sc_amp = ax.scatter(pos[:, 0], pos[:, 1], c=np.zeros(na),
                                 cmap='viridis', s=50, vmin=0, vmax=1,
                                 zorder=5, edgecolors='black', linewidths=0.3)
        ax.autoscale_view()
        ax.set_title('|z|')

        # --- Phase panel ---
        ax = self.ax_phase
        ax.set_aspect('equal')
        self.bond_bg_phase = LineCollection(segs, colors=np.ones((nb, 4)),
                                            linewidths=np.ones(nb), zorder=1)
        ax.add_collection(self.bond_bg_phase)
        self.sc_phase = ax.scatter(pos[:, 0], pos[:, 1], c=np.zeros(na),
                                   cmap='twilight', s=50, vmin=-np.pi, vmax=np.pi,
                                   zorder=5, edgecolors='black', linewidths=0.3)
        ax.autoscale_view()
        ax.set_title('arg(z)')

        # --- Bond order panel ---
        ax = self.ax_bond
        ax.set_aspect('equal')
        self.lc_bond = LineCollection(segs, colors=np.ones((nb, 4)),
                                      linewidths=np.ones(nb))
        ax.add_collection(self.lc_bond)
        self.sc_bond_atoms = ax.scatter(pos[:, 0], pos[:, 1], c='black',
                                        s=8, zorder=5)
        ax.autoscale_view()
        ax.set_title('Bond order')

        # --- Phase field + nodal lines panel ---
        # HSV phase background with nodal lines overlaid
        ax = self.ax_nodal
        ax.set_aspect('equal')

        # Precompute grid for smooth Gaussian interpolation
        res = 200
        xmin, xmax = pos[:, 0].min() - 0.5, pos[:, 0].max() + 0.5
        ymin, ymax = pos[:, 1].min() - 0.5, pos[:, 1].max() + 0.5
        gx = np.linspace(xmin, xmax, res)
        gy = np.linspace(ymin, ymax, res)
        GX, GY = np.meshgrid(gx, gy)
        self._hsv_extent = (xmin, xmax, ymin, ymax)
        self._hsv_res = res
        self._hsv_GX = GX
        self._hsv_GY = GY
        self._hsv_sigma = 1.5 * self.graph.aCC

        # Initial HSV image (will be updated in update())
        self.img_hsv = ax.imshow(np.zeros((res, res, 3)),
                                  extent=self._hsv_extent, origin='lower',
                                  aspect='auto', zorder=0)

        self.lc_faint_nodal = LineCollection(segs, colors=np.ones((nb, 4)),
                                             linewidths=np.ones(nb) * 0.5,
                                             zorder=2)
        ax.add_collection(self.lc_faint_nodal)
        self.lc_nodal_re = LineCollection(np.zeros((nb, 2, 2)), colors='red',
                                          linewidths=2.5, label='Re(z)=0', zorder=4)
        self.lc_nodal_im = LineCollection(np.zeros((nb, 2, 2)), colors='cyan',
                                          linewidths=2.5, label='Im(z)=0', zorder=4)
        ax.add_collection(self.lc_nodal_re)
        ax.add_collection(self.lc_nodal_im)
        self.sc_nodal_atoms = ax.scatter(pos[:, 0], pos[:, 1], c='black',
                                         s=5, zorder=5)
        ax.autoscale_view()
        ax.set_title('Phase field (HSV) + nodal lines')
        ax.legend(loc='upper right', fontsize=8)

        # --- Winding panel (if 6-panel) ---
        if self.ax_winding is not None:
            ax = self.ax_winding
            ax.set_aspect('equal')
            self.lc_faint_wind = LineCollection(segs, colors=np.ones((nb, 4)),
                                                linewidths=np.ones(nb) * 0.5)
            ax.add_collection(self.lc_faint_wind)
            self.ring_polys = []
            for ring in self.graph.rings:
                pts = pos[ring]
                poly = plt.Polygon(pts, facecolor='white', alpha=0.5,
                                   edgecolor='gray', linewidth=0.5)
                ax.add_patch(poly)
                self.ring_polys.append(poly)
            self.sc_wind_atoms = ax.scatter(pos[:, 0], pos[:, 1], c='black',
                                            s=5, zorder=5)
            ax.autoscale_view()
            ax.set_title('Ring winding')

        # --- Closeup panel: HSV phase + bonds (if 6-panel) ---
        if self.ax_closeup is not None:
            ax = self.ax_closeup
            ax.set_aspect('equal')
            self.img_hsv_closeup = ax.imshow(np.zeros((res, res, 3)),
                                              extent=self._hsv_extent, origin='lower',
                                              aspect='auto', zorder=0)
            self.bond_bg_closeup = LineCollection(segs, colors=np.ones((nb, 4)),
                                                  linewidths=np.ones(nb), zorder=2)
            ax.add_collection(self.bond_bg_closeup)
            self.sc_closeup = ax.scatter(pos[:, 0], pos[:, 1], c='black',
                                         s=20, zorder=5)
            ax.autoscale_view()
            ax.set_title('Phase field close-up')

        # Colorbars (created once, never recreated)
        self.fig.colorbar(self.sc_amp, ax=self.ax_amp, label='|z|')
        self.fig.colorbar(self.sc_phase, ax=self.ax_phase, label='arg(z)')

    def _update_bond_bg(self, lc, bx):
        n = len(self.viz._bond_segs)
        colors = np.ones((n, 4))
        colors[:, :3] = 0.7
        colors[:, 3] = 0.3 + 0.5 * bx
        lc.set_colors(colors)
        lc.set_linewidths(0.5 + 2.0 * bx)

    def update(self, z_real, z_imag, bond_x, step=None, defect_pos=None,
               suptitle=None):
        """Update all artists with new data. No clearing/recreation."""
        z = z_real + 1j * z_imag
        amp = np.abs(z)
        phase = np.angle(z)
        bx = np.asarray(bond_x)

        # Amplitude
        self.sc_amp.set_array(amp)
        vmax = max(amp.max(), 0.01)
        self.sc_amp.set_clim(0, vmax)
        self._update_bond_bg(self.bond_bg_amp, bx)

        # Phase
        self.sc_phase.set_array(phase)
        self._update_bond_bg(self.bond_bg_phase, bx)

        # Bond order
        cmap_bond = plt.cm.get_cmap('RdYlBu_r')
        norm_bond = mcolors.Normalize(vmin=0, vmax=max(bx.max(), 0.01))
        self.lc_bond.set_colors(cmap_bond(norm_bond(bx)))
        self.lc_bond.set_linewidths(1.0 + 4.0 * bx)

        # HSV phase field background (smooth Gaussian interpolation)
        sigma = self._hsv_sigma
        re_grid = np.zeros((self._hsv_res, self._hsv_res))
        im_grid = np.zeros((self._hsv_res, self._hsv_res))
        weight_sum = np.zeros((self._hsv_res, self._hsv_res))
        for i in range(len(self.viz._pos)):
            dx = self._hsv_GX - self.viz._pos[i, 0]
            dy = self._hsv_GY - self.viz._pos[i, 1]
            w = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
            re_grid += w * z_real[i]
            im_grid += w * z_imag[i]
            weight_sum += w
        weight_sum = np.maximum(weight_sum, 1e-30)
        re_grid /= weight_sum
        im_grid /= weight_sum
        grid_amp = np.sqrt(re_grid**2 + im_grid**2)
        grid_phase = np.angle(re_grid + 1j * im_grid)
        z_max = max(amp.max(), 1e-8)
        H = np.clip((grid_phase + np.pi) / (2 * np.pi), 0.0, 1.0)
        S = np.ones_like(H)
        V = np.clip(grid_amp / z_max, 0.0, 1.0)
        hsv = np.stack([H, S, V], axis=-1)
        rgb = mcolors.hsv_to_rgb(hsv)
        self.img_hsv.set_data(rgb)

        # Store grids for contour-based nodal lines
        self._hsv_re_grid = re_grid
        self._hsv_im_grid = im_grid

        # Nodal lines: smooth zero-contours on interpolated grid
        # Remove old contour collections, then redraw
        for coll in list(self.ax_nodal.collections):
            if hasattr(coll, '_is_nodal_contour'):
                coll.remove()
        re_grid = self._hsv_re_grid
        im_grid = self._hsv_im_grid
        cr = self.ax_nodal.contour(self._hsv_GX, self._hsv_GY, re_grid,
                                   levels=[0], colors='red', linewidths=2.0,
                                   alpha=0.8, zorder=4)
        cr._is_nodal_contour = True
        ci = self.ax_nodal.contour(self._hsv_GX, self._hsv_GY, im_grid,
                                   levels=[0], colors='cyan', linewidths=2.0,
                                   alpha=0.8, zorder=4)
        ci._is_nodal_contour = True

        # Hide old discrete nodal line collections
        self.lc_nodal_re.set_visible(False)
        self.lc_nodal_im.set_visible(False)

        # Winding
        if self.ax_winding is not None:
            windings = []
            for ring in self.graph.rings:
                w = self.viz._ring_winding(ring, z)
                windings.append(w)
            windings = np.array(windings)
            if len(windings) > 0:
                vmax_w = max(np.abs(windings).max(), 1.0)
                norm_w = mcolors.Normalize(vmin=-vmax_w, vmax=vmax_w)
                cmap_w = plt.cm.get_cmap('seismic')
                colors_w = cmap_w(norm_w(windings))
                for idx, poly in enumerate(self.ring_polys):
                    poly.set_facecolor(colors_w[idx])

        # Closeup: HSV phase + bonds
        if self.ax_closeup is not None:
            self.img_hsv_closeup.set_data(rgb)
            self._update_bond_bg(self.bond_bg_closeup, bx)
            if defect_pos is not None:
                self.ax_closeup.set_xlim(defect_pos[0] - 3, defect_pos[0] + 3)
                self.ax_closeup.set_ylim(defect_pos[1] - 3, defect_pos[1] + 3)

        # Title
        if suptitle:
            self.fig.suptitle(suptitle, fontsize=12)
        elif step is not None:
            self.fig.suptitle(f'step={step}  |z|_mean={amp.mean():.3f}',
                              fontsize=12)

        self.fig.canvas.draw_idle()

    def mark_defect(self, positions):
        """Mark defect position(s) with red stars on all panels.

        Args:
            positions: single (x, y) tuple or list of (x, y) tuples for multiple defects.
        """
        if isinstance(positions[0], (int, float)):
            positions = [positions]  # single defect

        if hasattr(self, '_defect_stars'):
            for s in self._defect_stars:
                s.remove()
        self._defect_stars = []
        for ax in self.axes.flat:
            for px, py in positions:
                star = ax.plot(px, py, 'r*', markersize=12, zorder=10)[0]
                self._defect_stars.append(star)

    def show(self):
        plt.show()
