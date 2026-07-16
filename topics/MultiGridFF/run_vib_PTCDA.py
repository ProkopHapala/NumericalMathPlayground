#!/usr/bin/env python3
"""
Compute UFF Hessian and vibration modes for PTCDA.
Saves modes as multi-frame .xyz files (each frame = one vibrational mode displacement).
"""
import sys
import os
import numpy as np

# Make py/ importable as a package (repo root is 3 levels up from topics/MultiGridFF/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from py.AtomicSystem import AtomicSystem
from py.FFs.Vibrations import run_vibrations, atomic_masses
from py.FFs.VibrationPlot import save_summary, plot_softest_modes, plot_mode_topview

PTCDA_FILE = os.path.join(REPO_ROOT, 'data', 'xyz', 'PTCDA.xyz')
OUT_DIR = os.path.join(REPO_ROOT, 'debug', 'vib_PTCDA')


def save_mode_xyz(result, mode_idx, filepath, scale=0.5):
    """Save a single vibration mode as a 3-frame xyz: equilibrium, +mode, -mode."""
    pos = result.pos
    enames = result.enames
    mode = result.modes[:, mode_idx].reshape(-1, 3)
    freq = result.frequencies_cm1[mode_idx]

    with open(filepath, 'w') as f:
        for frame, label in [(0, 'equilibrium'), (1, '+mode'), (-1, '-mode')]:
            f.write(f"{len(enames)}\n")
            f.write(f"freq={freq:.2f} cm-1 mode={mode_idx} {label} scale={scale}\n")
            for i, e in enumerate(enames):
                p = pos[i] + frame * scale * mode[i]
                f.write(f"{e} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def save_all_modes_xyz(result, outdir, scale=0.5, n_modes=None):
    """Save all vibration modes as individual .xyz files."""
    os.makedirs(outdir, exist_ok=True)
    n = n_modes or len(result.frequencies_cm1)
    for i in range(n):
        freq = result.frequencies_cm1[i]
        fname = os.path.join(outdir, f'mode_{i:03d}_{freq:.0f}cm.xyz')
        save_mode_xyz(result, i, fname, scale=scale)
    print(f"Saved {n} mode .xyz files to {outdir}")


def main():
    print(f"Loading PTCDA from {PTCDA_FILE}")
    mol = AtomicSystem(fname=PTCDA_FILE)
    print(f"  {len(mol.apos)} atoms, {len(mol.bonds)} bonds")

    print("Computing UFF Hessian via finite differences...")
    result = run_vibrations(mol, backend='uff', delta=1e-4, do_nonbond=False)

    print(f"\nVibration results: {len(result.frequencies_cm1)} modes")
    print(result.format_table(unit='cm-1'))

    os.makedirs(OUT_DIR, exist_ok=True)

    summary_path = os.path.join(OUT_DIR, 'vib_summary.txt')
    save_summary(result, summary_path)
    print(f"\nSummary saved to {summary_path}")

    save_all_modes_xyz(result, OUT_DIR, scale=0.5)
    print(f"\nAll mode .xyz files saved to {OUT_DIR}")

    # Plot 6 softest modes as SVG
    svg_dir = os.path.join(OUT_DIR, 'svg')
    os.makedirs(svg_dir, exist_ok=True)
    for i in range(min(6, len(result.frequencies_cm1))):
        freq = result.frequencies_cm1[i]
        svg_path = os.path.join(svg_dir, f'mode_{i:03d}_{freq:.0f}cm.svg')
        plot_mode_topview(result, i, svg_path)
        print(f"  SVG: {svg_path}")
    print(f"\n6 softest mode SVGs saved to {svg_dir}")

    # Also save 6 softest modes as .xyz
    soft_dir = os.path.join(OUT_DIR, 'softest6')
    os.makedirs(soft_dir, exist_ok=True)
    for i in range(min(6, len(result.frequencies_cm1))):
        freq = result.frequencies_cm1[i]
        save_mode_xyz(result, i, os.path.join(soft_dir, f'mode_{i:03d}_{freq:.0f}cm.xyz'), scale=0.5)
    print(f"6 softest mode .xyz files saved to {soft_dir}")


if __name__ == '__main__':
    main()
