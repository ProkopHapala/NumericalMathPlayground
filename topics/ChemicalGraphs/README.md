# ChemicalGraphs

Python toolkit for building, manipulating, and analyzing molecular graphs —
with a focus on sp2 heterocyclic systems and Kekule pi-bond order optimization.

## Files

| File | Role |
|------|------|
| `__init__.py` | Package exports: AtomicSystem, AtomicGraph, KekulePure, KekuleBackend, elements, atomicUtils, plotUtils, ascii_art_heterocycle, heterocycle_generator |
| `elements.py` | Periodic table data: Z, symbol, period, group, shell, name, covalent/vdW radii, color, valence electrons, mass |
| `atomicUtils.py` | Low-level utilities: bond detection from coordinates, mol/mol2/xyz file I/O, adjacency lists, graph pruning, ring perception |
| `AtomicSystem.py` | Core molecular container — atoms, bonds, positions, charges, lattice vectors. Supports loading from .mol, .mol2, .xyz, .gen, .cif formats |
| `AtomicGraph.py` | Object-graph representation (Atom, Bond, Ring objects with stable identity). No renumbering on deletion; numpy arrays generated on demand via `to_arrays()` |
| `KekulePure.py` | Pure-NumPy pi-bond order optimizer. Parabolic valence penalty, aromatic stabilization, piecewise localization snap. Gradient-descent relaxation |
| `KekuleBackend.py` | Backend state manager for interactive Kekule structure explorer. Honeycomb geometry helpers, persistent AtomicSystem with hexagonal grid metadata |
| `ascii_art_heterocycle.py` | Parse ASCII art drawings of heterocycles into AtomicSystem. Supports single-atom and dimer formats with zig-zag lattice topology inference |
| `heterocycle_generator.py` | Generate heterocyclic structures from sparse rectangular-grid descriptions of hexagonal lattices. E/D layer system with vertical/horizontal dimer modes, heteroatom substitution |
| `plotUtils.py` | Matplotlib plotting helpers: energy/force scans, bond order visualization, system plotting, Kekule phase diagrams |

## Key Concepts

- **Pi-bond order optimization**: Each sp2 atom has a target pi-electron count (n_pi=1). The optimizer balances valence penalties, aromatic stabilization (pi=0.5), and localization snaps (pi=0, 0.5, 1).
- **Heterocycle generation**: Hexagonal lattices described as rows of E (edge) and D (dimer) layers. Heteroatoms specified via sparse compact notation (e.g. `'1N,2.,3-OC'`).
- **ASCII art input**: Two formats — single-atom (bonds inferred from zig-zag topology) and dimer (explicit bond rows with `|` and `-`).

## Usage

```python
from ChemicalGraphs import AtomicSystem, KekulePure, make_n_pi

# Load a molecule
sys = AtomicSystem(fname='molecule.mol')

# Optimize pi-bond orders
n_pi = make_n_pi(sys)  # 1 for sp2, 0 for sp3
kek = KekulePure(sys, n_pi=n_pi)
kek.relax()
print(kek.pi_orders)  # optimized pi-bond orders per bond
```

```python
from ChemicalGraphs import parse_ascii_art, run_kekule_solver

# Build from ASCII art
system, kek = run_kekule_solver("""
    O o O
     c c
     c c
    c c c
    c c c
     c c
    O o O
""")
```
