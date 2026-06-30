# IsingMC — Ising / MQCA Ground-State Solver & Degeneracy Analysis

## Problem Statement

This project investigates **Ising-like lattice models** that arise in the
**Molecular Quantum Cellular Automata (MQCA)** framework — a paradigm where
binary logic is implemented by charge configurations on molecular-scale
clusters of quantum dots.

The Hamiltonian for a cluster of `N` sites is:

```
H = sum_i  n_i * E_i  +  sum_{i<j}  n_i * n_j * W_ij
```

where:
- `n_i in {0, 1}` is the occupancy (charge) of site `i`
- `E_i` is the on-site energy (chemical potential / charging energy)
- `W_ij` is the pairwise interaction (coupling) between sites `i` and `j`

On a square lattice the couplings are parameterised by two values:
- `W1` — nearest-neighbour (Manhattan distance 1)
- `W2` — diagonal (Manhattan distance 2)

### Goals

1. **Logic function identification**: For a given cluster geometry with
   input sites (gate pads A, B) and an output site, determine which of the
   16 possible binary logic functions (AND, OR, XOR, …) the cluster
   implements as a function of `(W1, W2)`.

2. **Degenerate ground-state search**: Find cluster geometries and
   parameter regimes `(E0, W1, W2)` where the ground state is **degenerate**
   (2–4 states with nearly equal energy) and **well-separated** from all
   higher excited states. Such systems are highly susceptible to small
   external perturbations (gate voltages) and can be used as controllable
   binary switches.

3. **Phase diagrams**: Produce 2-D maps showing how the output (site
   occupancy) depends on inputs (gate voltages on peripheral sites) across
   the `(W1, W2)` parameter space.

---

## Directory Structure

```
IsingMC/
├── README.md                       ← this file
├── OpenCLBase.py                   ← PyOpenCL utility base class
├── clUtils.py                      ← OpenCL device selection helpers
├── IsingExactSolver.py             ← Exact brute-force solver (≤16 sites)
├── IsingMCSolver.py                ← Monte Carlo solver (large systems)
├── cl/                             ← OpenCL kernel sources
│   ├── ising_exact.cl              ← Gray-code exact ground-state kernel
│   ├── ising_exact_top8.cl         ← Exact solver tracking top-8 states
│   └── ising_mc.cl                 ← Monte Carlo + local-update kernels
├── test_top8.py                    ← Quick test of top8 kernel
├── test_phase_diagrams.py          ← Phase diagram & logic-map generator
├── analyze_degeneracy.py           ← Degeneracy analysis at fixed (W1,W2)
├── analyze_degeneracy_cli.py       ← CLI degeneracy analysis tool
├── find_degenerate_ground_states.py← E0-scan degeneracy search (main script)
├── plot_all_degenerate.py          ← Tetris-style degenerate state plots
├── plot_degenerate_states.py       ← Detailed degenerate state visualization
└── results/                        ← Generated plots and images
```

---

## Solvers

### IsingExactSolver (`IsingExactSolver.py`)

**Exact** brute-force solver for clusters up to **16 sites**. Uses OpenCL
Gray-code traversal to enumerate all `2^N` occupancy configurations and find
the ground state (or top-8 lowest states).

**Key methods:**
- `solve(Esite, W_val, W_idx, nNeigh, nSite)` — find ground state
- `solve_batch_W(Esite, W_val, W_idx, nNeigh, nSite)` — batch with varying W
- `solve_batch_W_top8(Esite, W_val, W_idx, nNeigh, nSite)` — batch, returns top 8 states

**Helper functions:**
- `sq_lattice_sparse(positions, W1, W2, nSite)` — build sparse W matrix from positions
- `apply_input_bias(Eb, pos, inp_pos, inp_neigh, inp_vals, W1, W2)` — apply gate voltage bias
- `eval_logic_table(...)` — evaluate which logic function a cluster implements
- `scan_W1_W2(...)` / `scan_W1_W2_top8(...)` — scan (W1, W2) grid
- `identify_logic(...)` — classify truth table into one of 16 logic functions
- `plot_ground_states(...)` / `plot_logic_map(...)` — visualization helpers

**OpenCL kernels** (`cl/ising_exact.cl`, `cl/ising_exact_top8.cl`):
- `ising_groundstate` — single instance, shared coupling matrix
- `ising_groundstate_batch_W` — batch with per-instance coupling
- `ising_groundstate_top8` — tracks 8 lowest-energy states (for degeneracy detection)

### IsingMCSolver (`IsingMCSolver.py`)

**Monte Carlo** solver for larger systems (>16 sites) where exact enumeration
is infeasible. Uses OpenCL kernels for local updates, Boltzmann sampling,
and global mutation strategies (load best, load neighbor, random reset).

**Key methods:**
- `solve(...)` — high-level solve with tip positions and site coordinates
- Supports screened Coulomb interactions, multipole corrections, mirror charges

**OpenCL kernels** (`cl/ising_mc.cl`):
- `solve_minBrute_fly` — brute-force with on-the-fly energy evaluation
- `solve_minBrute_boltzmann` — Boltzmann-weighted sampling
- `solve_local_updates` — local Monte Carlo with shared-memory occupancy masks
- `solve_MC_neigh` — extended MC with global mutation strategies
- `calculate_currents` — current calculation between sites

### OpenCLBase (`OpenCLBase.py`)

Base class providing OpenCL infrastructure: device selection, context/queue
management, kernel compilation and caching, GPU buffer allocation, and
data transfer utilities. Both solvers inherit from this class.

---

## Scripts & Demos

### `find_degenerate_ground_states.py` — **Main degeneracy search**

Scans the chemical potential `E0` vs coupling `W1` at fixed `W2` values,
**without any gate voltage applied**. The cluster is evaluated by itself —
at certain `E0` the ground state becomes degenerate, making it susceptible
to external stimuli.

**Output per cluster × W2 value:**
- `degeneracy_scan_*.png` — 3-panel map: # degenerate states, gap, good regions + cluster geometry
- `best_degenerate_*.png` — detailed state visualization at best (E0, W1) point
- `spectrum_W1_*.png` — energy spectrum vs W1 with cluster geometry panel
- `spectrum_E0_*.png` — energy spectrum vs E0 with cluster geometry panel

**Usage:**
```bash
python find_degenerate_ground_states.py                          # full scan, output to ./results
python find_degenerate_ground_states.py --clusters T_simple Cross --W2 0.5
python find_degenerate_ground_states.py --outdir ./results_fine --nE 120 --nW 120
```

### `test_phase_diagrams.py` — **Phase diagram & logic map generator**

Produces three sets of images:
1. Ground-state occupancy for each input combination (Tetris-style)
2. 2-D logic map: which logic function is implemented at each (W1, W2)
3. Geometry scan: which (W1, W2) values produce useful logic for multiple cluster shapes

### `analyze_degeneracy.py` — **Fixed-point degeneracy analysis**

Analyzes degeneracy of the `T_extended_output` cluster at a specific `(W1, W2)`
point for all 4 input combinations. Prints top-8 states, ground-state gaps,
and a degeneracy summary.

### `analyze_degeneracy_cli.py` — **CLI degeneracy analysis**

Command-line tool for analyzing degeneracy for arbitrary cluster geometries
and parameters. Supports different input pad positions and plots states in
Tetris style.

```bash
python analyze_degeneracy_cli.py --cluster T_extended_output --W1 2.0 --W2 0.0
python analyze_degeneracy_cli.py --cluster T_extended_inputs --W1 2.0 --W2 1.0 --input-pos side
```

### `test_top8.py` — **Quick kernel test**

Minimal test of the `solve_batch_W_top8` kernel with a 9-site cluster.
Verifies the solver works and checks ground-state gap.

### `plot_all_degenerate.py` — **Tetris-style degenerate state plots**

Plots all 4 input combinations with degenerate states for
`T_extended_output` at `W1=0, W2=0` (no coupling = maximum degeneracy).

### `plot_degenerate_states.py` — **Detailed degenerate state visualization**

Plots the 6 degenerate ground states for inputs (0,1) and (1,0) for
`T_extended_output` at `W1=2.0, W2=1.0`.

---

## Cluster Geometries

All clusters are defined on a square lattice with integer `(x, y)` positions:

| Name | Sites | Description |
|------|-------|-------------|
| `T_simple` | 5 | T-shape: 3-bar + 2-stem |
| `Cross` | 5 | Plus shape: center + 4 arms |
| `Zigzag` | 5 | Zigzag chain of 5 sites |
| `Chain4` | 4 | Linear chain of 4 sites |
| `Chain5` | 5 | Linear chain of 5 sites |
| `L` | 6 | 2×3 rectangle (L-shape) |
| `S` | 6 | S-shape (2×3 staggered) |
| `T_extended_output` | 8 | T-shape with 2-site output stem |
| `T_extended_inputs` | 9 | T-shape with 2-site input stems |

---

## Key Results

The degeneracy search identifies several promising clusters:

- **L cluster** (6 sites): 2-fold checkerboard degeneracy at large W1, gap up to 3.0
- **S cluster** (6 sites): 4-fold degeneracy at W2=0, gap up to 2.77
- **T_simple** (5 sites): 3-fold degeneracy at W2=0, gap up to 2.77
- **Cross** (5 sites): 2-fold degeneracy, gap up to 1.82

All results are saved in `results/`.
