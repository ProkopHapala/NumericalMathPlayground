"""
demo_pyscf_grids.py — Demo for PySCF DFT integration grid visualization.

Usage:
    python demo_pyscf_grids.py --mode multi
    python demo_pyscf_grids.py --mode single
    python demo_pyscf_grids.py --mode all
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from PySCFGridPlotting import (
    plot_3d_scatter, plot_2d_projections, plot_radial_distribution,
    plot_grid_count_vs_level,
    plot_single_atom_3d, plot_single_atom_2d,
    plot_single_atom_radial, plot_lebedev_shells,
)

IMG_DIR = "Quadrature_Images"


def mode_multi():
    from pyscf import gto, dft
    mol = gto.M(
        atom='''
            O  0.0000  0.0000  0.0000
            H  0.0000  0.7572  0.5866
            H  0.0000 -0.7572  0.5866
        ''',
        basis='ccpvdz', verbose=0,
    )
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.build()

    coords = grids.coords
    weights = grids.weights
    atm_idx = grids.atm_idx

    print(f"Total grid points: {len(weights)}")
    print(f"Sum of weights: {weights.sum():.4f}")
    for ia in range(mol.natm):
        sym = mol.atom_symbol(ia)
        mask = atm_idx == ia
        print(f"  Atom {ia} ({sym}): {mask.sum()} points, weight sum = {weights[mask].sum():.4f}")

    atom_colors = {0: 'red', 1: 'blue', 2: 'blue'}
    atom_labels = {0: 'O', 1: 'H', 2: 'H'}
    atom_coords = mol.atom_coords()

    fig1 = plot_3d_scatter(coords, atm_idx, atom_labels, atom_colors, atom_coords,
                           grids.level, len(weights))
    fig1.savefig(f"{IMG_DIR}/pyscf_grids_3d.png", dpi=150, bbox_inches='tight')
    plt.show()

    fig2 = plot_2d_projections(coords, atm_idx, atom_labels, atom_colors, atom_coords, grids.level)
    fig2.savefig(f"{IMG_DIR}/pyscf_grids_2d.png", dpi=150, bbox_inches='tight')
    plt.show()

    fig3 = plot_radial_distribution(coords, atom_coords, atm_idx, atom_labels, atom_colors, grids.level)
    fig3.savefig(f"{IMG_DIR}/pyscf_grids_radial.png", dpi=150, bbox_inches='tight')
    plt.show()

    fig4 = plot_grid_count_vs_level(mol)
    fig4.savefig(f"{IMG_DIR}/pyscf_grids_levels.png", dpi=150, bbox_inches='tight')
    plt.show()


def mode_single():
    from pyscf import gto, dft
    from pyscf.dft.gen_grid import gen_atomic_grids, nwchem_prune
    from pyscf.dft import radi

    mol = gto.M(atom='O 0 0 0', basis='ccpvdz', verbose=0)
    atom_grids_tab = gen_atomic_grids(mol, level=3, prune=nwchem_prune)
    coords, vol = atom_grids_tab['O']

    chg = 8
    n_rad = 80
    n_ang = 434
    rad, dr = radi.treutler_ahlrichs(n_rad, chg)
    angs = nwchem_prune(chg, rad, n_ang)

    print(f"O atom grid (level 3):")
    print(f"  Total points: {len(coords)}")
    print(f"  n_rad = {n_rad}, n_ang (max) = {n_ang}")
    print(f"  Radial range: [{rad.min():.4f}, {rad.max():.4f}] Bohr")
    print(f"  Angular grid sizes: {sorted(set(angs))}")

    fig1 = plot_single_atom_3d(coords, n_rad, n_ang)
    fig1.savefig(f"{IMG_DIR}/pyscf_grid_single_3d.png", dpi=150, bbox_inches='tight')
    plt.show()

    fig2 = plot_single_atom_2d(coords)
    fig2.savefig(f"{IMG_DIR}/pyscf_grid_single_2d.png", dpi=150, bbox_inches='tight')
    plt.show()

    fig3 = plot_single_atom_radial(rad, angs, n_rad)
    fig3.savefig(f"{IMG_DIR}/pyscf_grid_single_radial.png", dpi=150, bbox_inches='tight')
    plt.show()

    fig4 = plot_lebedev_shells(rad, angs, n_rad)
    fig4.savefig(f"{IMG_DIR}/pyscf_grid_single_shells.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PySCF grid visualization demos")
    parser.add_argument('--mode', default='all', choices=['multi', 'single', 'all'])
    args = parser.parse_args()

    if args.mode in ('multi', 'all'):
        mode_multi()
    if args.mode in ('single', 'all'):
        mode_single()
