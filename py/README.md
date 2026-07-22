# py/

Core Python package for molecular modeling — force fields, vibrational analysis, DFTB+ integration, and OpenCL GPU infrastructure. Extracted from [SPAMMM](https://github.com/prokop/SPAMMM) and adapted for standalone use in NumericalMathPlayground.

## Top-level modules

- **AtomicSystem.py** — Array-based molecular representation: flat NumPy arrays for positions, types, bonds, angles, torsions, neighbors. File I/O for `.xyz`, `.mol`, `.mol2`. Bond finding, neighbor lists, PBC cloning, orientation.
- **AtomicGraph.py** — Object-graph molecular topology with stable Python identities (Atom, Bond, Ring objects). Bidirectional references, graph traversal, `to_arrays()`/`from_arrays()` conversion. Used for interactive editing.
- **atomicUtils.py** — Molecular I/O helpers (XYZ, MOL, MOL2 parsing), bond finding by distance, geometry utilities.
- **elements.py** — Periodic table data for all 118 elements: Z, mass, covalent/vdW radii, colors, electronegativity. `ELEMENT_DICT` lookup by symbol.
- **OpenCLBase.py** — Base class for all PyOpenCL computations: device selection by vendor (NVIDIA/AMD/Intel), kernel loading/caching via `load_program_multi`, buffer dictionary (`check_buf`), typed host-device transfer (`toGPU_`/`fromGPU_`), kernel argument binding (`generate_kernel_args`), kernel launch (`run`/`run_vec`).
- **clUtils.py** — OpenCL device selection helpers (`get_nvidia_device`), `GridShape`/`GridCL` for 3D grid dimensions and image buffers.
- **globals.py** — Centralized debug/verbosity controls (`VERBOSITY_LEVEL`, `debug_print`). Environment overrides: `SPAMMM_VERBOSITY`.
- **config_utils.py** — JSON config loading, path resolution for DFTB basis sets and SK parameters.
- **plotUtils.py** — Pure-matplotlib 1D/2D plotting (energies, forces, orbitals, densities, ESP). No Qt dependency.
- **ascii_art_heterocycle.py** — ASCII art heterocycle builder: parse 2D ASCII diagrams into molecular graphs.

## Subpackages

- **FFs/** — Force fields and vibrational analysis (UFF, normal modes, rigid-body pairwise FF)
- **DFTB/** — DFTB+ integration (ctypes wrapper, basis parser, GPU density projection)
- **kernels/** — OpenCL source files (`.cl`), concatenated at build time
- **GUI/** — Interactive visualization: `RigidBodyVispy.py` (Vispy+PyQt5 viewer for RigidBodyPairFF with potential map, mouse picking, parameter controls)

## Usage

```python
from py.AtomicSystem import AtomicSystem
from py.FFs.Vibrations import run_vibrations

mol = AtomicSystem(fname='data/xyz/PTCDA.xyz')
result = run_vibrations(mol, backend='uff', delta=1e-4)
print(result.format_table(unit='cm-1'))
```

See `topics/MultiGridFF/run_vib_PTCDA.py` for a complete example.
