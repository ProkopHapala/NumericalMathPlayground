"""
demo_atomic_radial.py — Plot radial wavefunctions R(r) from PySCF basis sets.

Usage:
    python demo_atomic_radial.py [--basis cc-pVDZ] [--elements H,C,N,O]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto

from BasisFunctions import get_radial_functions, compute_radial_extent
from PySCFGridPlotting import (
    plot_radial_wavefunctions,
    plot_radial_wavefunctions_normalized,
    plot_radial_density,
)

IMG_DIR = "Quadrature_Images"


def main():
    parser = argparse.ArgumentParser(description='Plot atomic radial wavefunctions from PySCF')
    parser.add_argument('--basis', default='cc-pVDZ', help='Basis set name')
    parser.add_argument('--elements', default='H,C,N,O', help='Comma-separated element symbols')
    parser.add_argument('--rmax', type=float, default=8.0, help='Max radial distance (Bohr)')
    parser.add_argument('--nr', type=int, default=1000, help='Number of radial grid points')
    args = parser.parse_args()

    elements = args.elements.split(',')
    r_grid = np.linspace(0.0, args.rmax, args.nr)

    # Build shells once per element (not 4× like the old script)
    elem_shells = []
    for elem in elements:
        nelec = gto.mole.charge(elem)
        spin = nelec % 2
        mol = gto.M(atom=f'{elem} 0 0 0', basis=args.basis, spin=spin)
        shells = get_radial_functions(mol, r_grid)
        elem_shells.append((elem, shells))

    basis_tag = args.basis.replace("-", "_")

    # Figure 1: Raw R(r)
    fig1 = plot_radial_wavefunctions(elem_shells, r_grid, args.basis, args.rmax)
    outfile1 = f'{IMG_DIR}/atomic_radial_{basis_tag}.png'
    fig1.savefig(outfile1, dpi=150, bbox_inches='tight')
    print(f'Saved: {outfile1}')

    # Figure 2: Normalized R(r)
    fig2 = plot_radial_wavefunctions_normalized(elem_shells, r_grid, args.basis, args.rmax)
    outfile2 = f'{IMG_DIR}/atomic_radial_normalized_{basis_tag}.png'
    fig2.savefig(outfile2, dpi=150, bbox_inches='tight')
    print(f'Saved: {outfile2}')

    # Figure 3: Radial density r²|R|²
    fig3 = plot_radial_density(elem_shells, r_grid, args.basis, args.rmax)
    outfile3 = f'{IMG_DIR}/atomic_radial_density_{basis_tag}.png'
    fig3.savefig(outfile3, dpi=150, bbox_inches='tight')
    print(f'Saved: {outfile3}')

    # Print radial extent table
    print(f'\n{"=":=<60s}')
    print(f'Radial extent (|R(r)| < 1% of max) — {args.basis}')
    print(f'{"=":=<60s}')
    for elem, shells in elem_shells:
        shells_sorted = sorted(shells, key=lambda x: (x[1], x[0]))
        print(f'\n{elem}:')
        print(f'  {"orbital":>10s}  {"l":>3s}  {"r_extent (Bohr)":>15s}  {"R_max":>10s}')
        for name, l, R in shells_sorted:
            r_ext, r_max, R_max = compute_radial_extent(r_grid, R)
            print(f'  {name:>10s}  {l:3d}  {r_ext:15.3f}  {R_max:10.4f}')

    plt.close('all')


if __name__ == '__main__':
    main()
