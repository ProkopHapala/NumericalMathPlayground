"""
test_opencl_basis.py — Test OpenCL radial basis evaluation vs PySCF reference.

Compares OpenCLBasis.OpenCLRadialBasis against BasisFunctions.get_radial_functions
for multiple elements and basis sets.

Usage:
    python test_opencl_basis.py
    python test_opencl_basis.py --elements H,C,O --basis cc-pVDZ
    python test_opencl_basis.py --device cpu   # force CPU device
"""
import argparse
import numpy as np
from pyscf import gto
import pyopencl as cl

from BasisFunctions import get_radial_functions
from OpenCLBasis import OpenCLRadialBasis


def test_element(elem, basis, r_grid, device_type=None):
    """
    Test OpenCL vs PySCF for a single element.

    Returns
    -------
    results : list of (label, l, max_abs_err, max_rel_err, R_max)
    """
    nelec = gto.mole.charge(elem)
    spin = nelec % 2
    mol = gto.M(atom=f'{elem} 0 0 0', basis=basis, spin=spin)

    # PySCF reference
    ref_shells = get_radial_functions(mol, r_grid)

    # OpenCL evaluation
    ocl = OpenCLRadialBasis(mol, device_type=device_type)
    ocl_shells = ocl.eval_radial_list(r_grid)

    # Match shells by (l, label)
    ref_map = {(l, label): R for (label, l, R) in ref_shells}
    ocl_map = {(l, label): R for (label, l, R) in ocl_shells}

    results = []
    all_keys = sorted(set(ref_map.keys()) | set(ocl_map.keys()))

    print(f'\n  {"orbital":>10s}  {"l":>3s}  {"max_abs_err":>12s}  {"max_rel_err":>12s}  {"R_max":>10s}  status')
    print(f'  {"─"*10:>10s}  {"─"*3:>3s}  {"─"*12:>12s}  {"─"*12:>12s}  {"─"*10:>10s}  {"─"*6:>6s}')

    for (l, label) in all_keys:
        if (l, label) not in ref_map:
            print(f'  {label:>10s}  {l:3d}  {"—":>12s}  {"—":>12s}  {"—":>10s}  MISSING_REF')
            continue
        if (l, label) not in ocl_map:
            print(f'  {label:>10s}  {l:3d}  {"—":>12s}  {"—":>12s}  {"—":>10s}  MISSING_OCL')
            continue

        R_ref = ref_map[(l, label)]
        R_ocl = ocl_map[(l, label)]

        abs_err = np.abs(R_ocl - R_ref)
        max_abs = np.max(abs_err)
        R_max = np.max(np.abs(R_ref))
        max_rel = max_abs / (R_max + 1e-30)

        status = "OK" if max_rel < 1e-5 else ("WARN" if max_rel < 1e-3 else "FAIL")
        print(f'  {label:>10s}  {l:3d}  {max_abs:12.3e}  {max_rel:12.3e}  {R_max:10.4f}  {status}')
        results.append((label, l, max_abs, max_rel, R_max))

    return results


def main():
    parser = argparse.ArgumentParser(description='Test OpenCL radial basis vs PySCF')
    parser.add_argument('--basis', default='cc-pVDZ', help='Basis set name')
    parser.add_argument('--elements', default='H,C,N,O',
                        help='Comma-separated element symbols')
    parser.add_argument('--rmax', type=float, default=10.0,
                        help='Max radial distance (Bohr)')
    parser.add_argument('--nr', type=int, default=2000,
                        help='Number of radial grid points')
    parser.add_argument('--device', default='gpu', choices=['gpu', 'cpu'],
                        help='OpenCL device type')
    args = parser.parse_args()

    elements = args.elements.split(',')
    r_grid = np.linspace(0.0, args.rmax, args.nr)
    device_type = cl.device_type.GPU if args.device == 'gpu' else cl.device_type.CPU

    # Print device info
    print(f'OpenCL device: {args.device}')
    print(f'Basis set: {args.basis}')
    print(f'Elements: {elements}')
    print(f'Radial grid: {args.nr} points, r_max={args.rmax} Bohr')
    print(f'{"=":=<70s}')

    all_pass = True
    for elem in elements:
        print(f'\n{elem} ({args.basis}):')
        try:
            results = test_element(elem, args.basis, r_grid, device_type)
            for (label, l, max_abs, max_rel, R_max) in results:
                if max_rel >= 1e-3:
                    all_pass = False
        except Exception as e:
            print(f'  ERROR: {e}')
            all_pass = False

    print(f'\n{"=":=<70s}')
    if all_pass:
        print('ALL TESTS PASSED (max_rel_err < 1e-3, float32)')
    else:
        print('SOME TESTS FAILED — see details above')


if __name__ == '__main__':
    main()
