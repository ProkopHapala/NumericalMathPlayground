# TestSystems

Test harnesses for benchmarking all Order-N methods against exact O(N^3)
diagonalization on simple tight-binding systems.

## Files

| File | Role |
|------|------|
| `hydrogen_chain_1d.py` | Comprehensive test harness for Order-N methods on a 1D hydrogen dimer chain. Compares linear-scaling density estimates (FOE, GF, OMM) against exact diagonalization. Supports configurable tight-binding parameters, overlap, and multiple solver options with detailed diagnostics and plotting |
| `hydrogen_lattice_2d.py` | 2D hydrogen lattice test system. Builds a 2D triangular lattice with tight-binding parameters for testing linear-scaling density matrix and orbital solvers in two dimensions |
| `wrong_hopping.md` | Bug note: explains why default parameters in hydrogen_chain_1d.py can cause bonding/antibonding ordering inversion with non-orthogonal overlap |

## Usage

```bash
cd topics/LinearScalingQM/TestSystems

# 1D hydrogen chain — compare all methods
python hydrogen_chain_1d.py --n 32 --method all

# 2D hydrogen lattice
python hydrogen_lattice_2d.py
```

## Dependencies

Imports from sibling folders:
- `DensityMatrix/OrderN.py` — shared utilities (neighbor pairs, matrix assembly, solvers)
- `DensityMatrix/FOE.py` — Fermi Operator Expansion
- `DensityMatrix/GF.py` — Green's Function methods
- `OMM/` — Orbital Minimization Method

The sys.path is automatically configured to find these sibling modules.
