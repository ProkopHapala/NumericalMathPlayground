# KekuleQM — Linear-Scaling π-Electron Solver on GPU

A classical, quantum-motivated bond-order field model for π-electron systems,
solved by gradient descent on GPU via OpenCL. No diagonalization is required;
the method scales linearly with system size (except for the optional O(N²)
Coulomb term, which can be replaced by FFT/FMM).

---

## Table of Contents

1. [What This Is](#1-what-this-is)
2. [Physical Model](#2-physical-model)
3. [Variables and Data Structures](#3-variables-and-data-structures)
4. [Energy Functional](#4-energy-functional)
5. [OpenCL Kernel Architecture](#5-opencl-kernel-architecture)
6. [PyOpenCL Harness](#6-pyopencl-harness)
7. [ChemicalGraphs — Molecule Builder](#7-chemicalgraphs--molecule-builder)
8. [Usage and CLI Reference](#8-usage-and-cli-reference)
9. [Results](#9-results)
10. [Caveats and Known Limitations](#10-caveats-and-known-limitations)
11. [File Layout](#11-file-layout)

---

## 1. What This Is

KekuleQM solves for the ground-state π-electron distribution of conjugated
molecules (graphene flakes, polycyclic aromatic hydrocarbons, heterocycles)
without diagonalizing any Hamiltonian matrix. Instead, it minimizes a
**classical energy functional of bond-order variables** using iterative
gradient descent on GPU.

The key insight is that the quantum one-particle density matrix P can be
parameterized by a small set of local variables — site occupancies and
half-bond electron donations — and the physically important constraints
(electron conservation, Pauli exclusion, valence saturation) can be enforced
locally without global linear algebra.

The hierarchy of approximations is:

```
MO diagonalization (O(N³))
  → density matrix minimization (O(N²)–O(N³))
    → sparse local density-matrix functional (O(N))
      → classical bond-order field model (O(N), this code)
```

---

## 2. Physical Model

### 2.1 The π-system as a graph

Each atom i in the π-system has up to 3 neighbors (sp² hybridization). Each
atom contributes a fixed number of π electrons Z^π_i (1 for carbon, 1–2 for
nitrogen depending on chemistry, 0–2 for oxygen). The molecular graph and
atomic positions are provided by the `ChemicalGraphs` package, which can
build structures from ASCII art representations.

### 2.2 Half-bond electron allocation

The central variables are **half-bond electron donations**:

```
y[i, s, a] ≥ 0
```

where:
- `i` = atom index
- `s` = spin (↑ or ↓)
- `a = 0` = localized (non-bonding) p_z electron density on atom i
- `a = 1, 2, 3` = electron density donated from atom i to its 1st, 2nd, 3rd neighbor

The local spin population is:

```
ρ[i,s] = Σ_a y[i,s,a]
```

The total π-electron population associated with atom i (Mulliken-like) is:

```
ρ_i = Σ_s ρ[i,s] = n_i + ½ Σ_j B_ij
```

where `B_ij` is the bond order between atoms i and j (defined below).

The site charge is:

```
Q_i = Z^π_i - ρ_i
```

This formulation automatically ensures electron conservation: putting electrons
into bonds depletes the site density, and vice versa. A carbon atom with one
π-electron cannot make three full double bonds because that would require
½(B₁ + B₂ + B₃) = 3/2 > 1 electron — the valence constraint is built in.

### 2.3 Bond order

The bond order between atoms i and j is:

```
B_ij = 2 Σ_s √( y[i,s,i→j] · y[j,s,j→i] )
```

This is the classical analogue of the quantum-mechanical Mulliken bond charge
`2·S_ij·c_i·c_j`. The square root of the product of half-bond donations from
both ends captures the quantum amplitude product `c_i · c_j` while keeping all
variables positive and classical.

### 2.4 Local simplex constraint (Pauli exclusion)

A single p_z orbital can hold at most 1 electron per spin (Pauli exclusion).
This is enforced by projecting the local channels onto a simplex:

```
Σ_a y[i,s,a] = ρ_target[i,s]    (with 0 ≤ y[i,s,a] ≤ ρ_target)
```

where `ρ_target[i,s] = Z^π_i / 2` for a closed-shell system. This means the
four local channels (site + 3 bonds) compete for a fixed electron budget per
atom per spin — the classical analogue of idempotency `P² = P`.

---

## 3. Energy Functional

The total energy has six physically motivated terms:

### 3.1 Hopping / resonance energy

```
E_hop = -2 Σ_{⟨ij⟩,s} ε_ij(r) · √(y[i,s,i→j] · y[j,s,j→i])
```

where the hopping integral depends on bond length:

```
ε_ij(r) = ε₀ · exp(-β · (r - r₀))
```

This is the tight-binding / Hückel energy. It favors delocalization: spreading
electrons over multiple bonds lowers the energy. The exponential decay means
shorter bonds have stronger hopping — this is the Peierls mechanism.

### 3.2 Sigma elastic energy

```
E_σ = ½ Σ_{⟨ij⟩} K_r · (r_ij - r₀)²
```

The σ-bond skeleton acts as a harmonic spring with stiffness K_r, pulling all
bonds toward the equilibrium length r₀. This resists the Peierls distortion.

### 3.3 Peierls force

The derivative of E_hop with respect to bond length gives the Peierls force:

```
F_Peierls = -∂E_hop/∂r = -2β · ε_ij(r) · B_ij
```

This force pulls atoms together on bonds with high bond order (the "electron
digs a hole and falls into it" mechanism from SSH theory). Combined with the
sigma restoring force `K_r · (r - r₀)`, the equilibrium distortion is:

```
Δr* = -2β · ε₀ · B_ij / K_r
```

Larger bond order → shorter bond. This is the Peierls/SSH coupling.

### 3.4 Onsite QEq term (electronegativity)

```
E_onsite = Σ_i [ χ_i · (ρ_i - Z^π_i) + ½ η_i · (ρ_i - Z^π_i)² ]
```

This is the charge-equilibration (QEq) term. Electronegativity χ_i controls
whether a site wants to accept or donate π-charge. Nitrogen (χ < 0) is more
electronegative than carbon; oxygen even more so.

### 3.5 Hubbard U

```
E_Hub = Σ_i U_i · ρ[i,↑] · ρ[i,↓]
```

Penalizes double occupation of the same p_z orbital with opposite spins. This
can drive spin polarization, especially at zigzag edges of graphene flakes.

### 3.6 Coulomb electrostatics

```
E_Coul = ½ Σ_{ij} Q_i · J_ij · Q_j
```

where `J_ij = k_Coul / r_ij` (bare Coulomb, optionally with Ohno softening).
This is the only non-local term. Direct evaluation is O(N²); it can be
replaced by FFT, FMM, or treecode without changing the local kernels.

---

## 4. OpenCL Kernel Architecture

The solver uses **three kernels per iteration**, analogous to a classical
force field with local bonding terms and non-local electrostatics:

```
┌─────────────────────────────────────────────────────┐
│  Iteration loop                                      │
│                                                      │
│  1. KekuleQM_gatherLocalBonding                      │
│     - Gather ρ, Q from y                             │
│     - Compute local gradients ∂E/∂y                  │
│     - Compute Peierls + sigma force on atoms         │
│     - Store bond data (ε, B, r)                      │
│                                                      │
│  2. KekuleQM_gatherCoulombDirect                     │
│     - O(N²) tiled direct Coulomb                     │
│     - Compute potential φ_i and force F_Coul         │
│                                                      │
│  3. KekuleQM_updateDOFs                              │
│     - Update y[i,s,a] by gradient descent            │
│     - Project onto local simplex (Pauli)             │
│     - Update positions R with heavy-ball momentum    │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 4.1 Kernel 1: `KekuleQM_gatherLocalBonding`

**Parallelization:** One work item per atom. Each atom reads its own y
variables and those of its neighbors (gather-only, no atomics).

**What it computes:**
- Spin populations `ρ[i,↑], ρ[i,↓]` and charge `Q_i`
- Local gradients `∂E/∂y[i,s,a]` from hopping and onsite terms
- The half-bond gradient contains `√(y_opposite / (y_own + ε))` — this pulls
  up the weaker side of an asymmetric bond
- Force on atom i from sigma springs and Peierls coupling:
  `F = K_r·(r-r₀)·ĥ - 2β·ε(r)·B·ĥ`
- Per-bond diagnostic data: `(ε, √(y_i·y_j), r, B)`

**Performance:** All loops are bounded by `KQ_MAX_NEIGH=3`. Branch divergence
comes only from edge atoms with missing neighbors. Workgroup size is not
critical; occupancy is limited by global-memory latency and exp/sqrt throughput.

### 4.2 Kernel 2: `KekuleQM_gatherCoulombDirect`

**Parallelization:** One work item per atom, workgroup size `KQ_COUL_WG=128`.

**What it computes:**
- Coulomb potential `φ_i = Σ_j J_ij · Q_j`
- Coulomb force `F_Coul = -Σ_j ∂J_ij/∂r · Q_i · Q_j · r̂`

**Tiling:** Atoms are loaded into local memory in tiles of 128. Each global
load is reused by all work items in the group. Synchronization at tile
boundaries.

**Performance:** This is the bottleneck for large systems. Replace with
screened cutoff, FFT, FMM, or treecode for O(N) or O(N log N) scaling.

### 4.3 Kernel 3: `KekuleQM_updateDOFs`

**Parallelization:** One work item per atom.

**Electronic update:**
```
y[i,s,a] ← y[i,s,a] - dtY · (∂E/∂y[i,s,a] - λ_local)
```
where `λ_local = (1/nchan) Σ_a ∂E/∂y[i,s,a]` is a local Lagrange multiplier
that conserves the per-atom electron count (local simplex projection).

After the gradient step, the y variables are projected back onto the simplex
`Σ_a y[i,s,a] = ρ_target[i,s]` with `y ≥ 0`.

**Geometry update (heavy-ball momentum):**
```
v_{t+1} = momentum · v_t + dtR · F
R_{t+1} = R_t + v_{t+1}
```

The heavy-ball (accelerated gradient) method provides inertial smoothing that
stabilizes the coupled electron-geometry relaxation. With `momentum=0` this
reduces to plain Euler integration.

---

## 5. PyOpenCL Harness (`KekuleQM_ocl.py`)

### 5.1 Molecule loading

Two pathways:
- **ASCII art** via `ChemicalGraphs.parse_ascii_art()` — builds molecular
  geometry and bond topology from a text-based hexagonal lattice representation
- **Graphene flake generator** — cuts a circular flake from an infinite
  honeycomb lattice
- **XYZ files** — standard `.xyz` format via `read_xyz()`

### 5.2 Neighbor lists

Fixed-size `neigh[natoms][3]` and `rev[natoms][3]` arrays:
- `neigh[i][k]` = global index of k-th neighbor of atom i (or -1)
- `rev[i][k]` = slot of i inside neighbor j's list

The reverse-slot mapping is critical: during the gather, atom i needs to read
the half-bond donation from its neighbor j back toward i, which is stored at
`y[j, s, 1+rev[i][k]]`.

For ASCII art molecules, bonds are taken from the explicit bond list (not
distance cutoff) to preserve the exact topology.

### 5.3 Iteration and sub-stepping

The electronic system relaxes much faster than geometry. To maintain stability:

- **Electronic sub-stepping:** `n_elec_substeps` electronic-only steps per
  geometry step (default 10). This keeps the electrons near their Born-Oppenheimer
  equilibrium while atoms slowly move.
- **Heavy-ball momentum:** `momentum=0.9` provides damped inertia for the
  geometry update, preventing oscillatory divergence.

### 5.4 Plotting

Three-panel figure:
1. **Bond order convergence** — shows how each bond order evolves over iterations
2. **Initial guess** — the starting state (uniform distribution + small noise)
3. **Final state** — the relaxed molecule with bond colors (red=double,
   green=aromatic, blue=single), bond length labels, atom labels, and
   magenta displacement arrows showing Peierls distortion

Atom label modes: `element`, `index`, `element_pi` (element + π-electron count),
or `none`.

---

## 6. ChemicalGraphs — Molecule Builder

The `ChemicalGraphs` package at `topics/ChemicalGraphs/` provides:

- **`AtomicSystem`** — core molecular container with positions, elements,
  bonds, neighbor lists, capping hydrogen addition
- **`elements`** — periodic table data (covalent radii, VdW radii, colors)
- **`atomicUtils`** — bond finding by distance, XYZ/MOL file I/O
- **`AtomicGraph`** — topological graph operations
- **`KekulePure`** — pure-Python NumPy Kekulé bond-order optimizer (the
  predecessor of this GPU solver)
- **`KekuleBackend`** — backend logic for hexagonal grid management
- **`plotUtils`** — `plotSystem()`, `plotBonds()` for molecular visualization
- **`ascii_art_heterocycle`** — parses ASCII art to build molecular structures;
  includes 28 built-in examples (naphthalene, purine, uracil, perylene, NTCDA, etc.)
- **`heterocycle_generator`** — geometry generation and plotting helpers

### ASCII art format

Molecules are defined as text grids where characters represent atoms (C, N, O)
and bond topology is encoded by spatial adjacency. For example:

```
  C---C
 /     \
C       C
|       |
C       C
 \     /
  C---C
```

The parser supports both "dimer" (bond-row) and "single-atom" formats, and
automatically resolves bond topology, adds capping hydrogens, and computes
π-electron counts per atom.

---

## 7. Usage and CLI Reference

### Basic usage

```bash
# Solve naphthalene (electronic only, default)
python3 KekuleQM_ocl.py -e naphthalene

# Solve with Peierls geometry relaxation
python3 KekuleQM_ocl.py -e naphthalene --updateR 1 --beta 1.0 --Kr 50.0

# Solve a graphene flake with atom indices shown
python3 KekuleQM_ocl.py -e flake --atom_labels index --nsteps 1000

# Solve purine with element + π-electron labels
python3 KekuleQM_ocl.py -e purin --atom_labels element_pi
```

### Available examples

28 ASCII art examples: `naphthalene`, `naphthalene2`, `fulvalene`,
`pentacross`, `biphenyl`, `biphenyl2`, `phenanthrene`, `phenanthrene2`,
`perylene`, `perylene2`, `purin`, `purin_x`, `purin_y`, `7azaindol`,
`karbazol`, `biphenylene`, `uracil`, `cytosin`, `guanin`, `NTCDA`, `NTCDI`,
`TAP`, `Quinolone`, `2Quinolone`, `2purin`, `Quinolinone`, `2Quinolinone`,
`2NCI`.

Plus `flake` for a generated graphene flake.

### Full CLI reference

```
System selection:
  --example, -e EXAMPLE     ASCII art example name or "flake" (default: naphthalene)

Solver parameters:
  --nsteps N                Number of iterations (default: 500)
  --dtY DTY                 Electronic step size (default: 0.05)
  --dtR DTR                 Geometry step size, 0=no relaxation (default: 0.0)
  --momentum M              Heavy-ball momentum for geometry (default: 0.9)
  --elec_substeps N         Electronic sub-steps per geometry step (default: 10)
  --coulomb C               Use Coulomb electrostatics, 1=yes (default: 0)
  --updateR U               Update atomic positions, 1=yes (default: 0)
  --kCoul K                 Coulomb constant in eV·Å/e² (default: 14.3996)

Plotting:
  --atom_labels MODE        Atom label mode: none|element|index|element_pi (default: element)
  --no_bond_lengths         Hide bond length labels on plot

Model parameters — bonding:
  --eps0 EPS0               Hopping strength in eV (default: 2.7)
  --beta BETA               Peierls decay rate in 1/Å (default: 3.0)
  --Kr KR                   Sigma bond stiffness in eV/Å² (default: 20.0)
  --r0 R0                   Equilibrium bond length in Å (default: 1.42)

Model parameters — per-element QEq/Hubbard (C, N, O):
  --chi_X, --eta_X, --U_X, --Zpi_X    Electronegativity, hardness, Hubbard U, π-electron count
```

### Recommended parameter sets

**Electronic-only relaxation** (fast, stable):
```bash
python3 KekuleQM_ocl.py -e naphthalene --nsteps 500
```

**With Peierls distortion** (requires weaker coupling for stability):
```bash
python3 KekuleQM_ocl.py -e naphthalene --updateR 1 --beta 1.0 --Kr 50.0 \
  --dtR 0.0005 --momentum 0.95 --elec_substeps 20 --nsteps 1000
```

**With Coulomb electrostatics:**
```bash
python3 KekuleQM_ocl.py -e purin --coulomb 1 --chi_N -1.5 --U_C 2.0
```

---

## 8. Results

### 8.1 Naphthalene (electronic only)

10 carbon atoms, 11 bonds. Starting from uniform bond order ~0.6, the solver
converges to a Kekulé pattern in ~200 iterations:

| Bond type | Bond order B | Count |
|-----------|-------------|-------|
| Double    | 1.07        | 5     |
| Aromatic  | 0.82        | 4     |
| Single    | 0.50        | 1     |

The central bond (connecting the two rings) has the lowest bond order, matching
chemical intuition.

### 8.2 Naphthalene with Peierls distortion

With `--updateR 1 --beta 1.0 --Kr 50.0`, atoms displace to shorten double bonds
and lengthen single bonds:

| Bond type | Bond order B | Bond length (Å) |
|-----------|-------------|-----------------|
| Double    | 1.08        | 1.358           |
| Aromatic  | 0.82        | 1.374           |
| Single    | 0.49        | 1.393           |

The Peierls distortion amplitude is ~0.03–0.06 Å, consistent with the expected
SSH physics. Bond orders are preserved from the electronic-only solution,
confirming that the geometry relaxation is stable.

### 8.3 Graphene flake (31 atoms)

A circular graphene flake shows a clear Kekulé pattern with alternating double
and single bonds at the edges, and more aromatic character in the interior.

| Bond type | Bond order B | Bond length (Å) |
|-----------|-------------|-----------------|
| Double    | 1.42–1.47   | 1.338–1.340     |
| Aromatic  | 0.76–0.78   | 1.379–1.387     |
| Single    | 0.46–0.54   | 1.398–1.401     |

### 8.4 Heterocycles (purine, uracil)

The solver handles heteroatoms (N, O) through different electronegativity χ
and π-electron count Z^π. Nitrogen draws electron density toward itself,
producing asymmetric bond orders visible in the plots.

---

## 9. Caveats and Known Limitations

### 9.1 Peierls coupling stability

The default parameters (`beta=3.0, Kr=20.0`) are too aggressive for geometry
relaxation — the Peierls force (~8 eV/Å) overwhelms the sigma spring and
causes bond collapse. For `--updateR 1`, use:

```
--beta 1.0 --Kr 50.0 --dtR 0.0005 --momentum 0.95 --elec_substeps 20
```

The physical reason: the Peierls mechanism is a positive feedback loop (shorter
bond → higher bond order → stronger pull → even shorter). The sigma spring must
be stiff enough to counterbalance this. With `beta=3.0` and `eps0=2.7`, the
equilibrium distortion for B=1 would be `Δr = 2·3·2.7/20 = 0.81 Å` — far too
large. With `beta=1.0, Kr=50`, it becomes `Δr = 2·1·2.7/50 = 0.108 Å`, which
is physically reasonable.

### 9.2 Initial guess

The initial state distributes `Z^π/2` per spin equally among all local channels
(site + bonds), plus a small random perturbation (`0.1 * rand`). This produces
non-uniform initial bond orders because atoms with different numbers of
neighbors get different per-channel allocations. The perturbation breaks
symmetry and allows the solver to find a Kekulé pattern rather than remaining
stuck in the symmetric aromatic state.

### 9.3 No explicit Pauli idempotency penalty

The current implementation enforces Pauli exclusion through the local simplex
constraint (fixed per-atom electron count per spin) rather than through an
explicit `P² = P` penalty term. This is simpler and faster but may not capture
all representability constraints. In particular, the off-diagonal idempotency
condition (which couples bond orders around rings) is not enforced. See the
theoretical discussion in `KekuleQM.md` for the full hierarchy of possible
constraints.

### 9.4 Coulomb kernel

The Coulomb kernel uses bare (unscreened) direct summation with no cutoff. For
systems larger than ~1000 atoms, this becomes the bottleneck. The kernel is
tiled through local memory for efficiency, but the scaling is O(N²). FFT, FMM,
or treecode replacements would preserve the local kernel interface.

### 9.5 Spin polarization

The model supports spin-polarized solutions through the Hubbard U term, but the
default parameters (U=0 for all elements) produce closed-shell solutions. To
explore spin polarization (e.g., at graphene zigzag edges), set `--U_C 2.0` or
higher.

### 9.6 No geometry optimization of sigma framework

The current geometry update only applies the Peierls force from π-electrons
plus the harmonic sigma spring. There is no full force-field relaxation (bond
angles, dihedrals, steric repulsion). The atoms move only in response to the
π-electron forces within the harmonic sigma approximation.

### 9.7 Workgroup size for Coulomb kernel

The Coulomb kernel requires `global_size ≥ workgroup_size = 128`. For small
molecules (< 128 atoms), the global size is rounded up to a multiple of 128,
with extra work items returning early. This is handled automatically.

---

## 10. File Layout

```
topics/LinearScalingQM/KekuleQM/
├── KekuleQM.cl          OpenCL kernels (3 kernels + helpers)
├── KekuleQM_ocl.py      PyOpenCL harness (molecule loading, solver, plotting)
├── KekuleQM.md          Theoretical derivation (full discussion with equations)
├── README.md            This file
└── __pycache__/

topics/ChemicalGraphs/   Molecule builder package
├── __init__.py           Package exports
├── elements.py           Periodic table data
├── atomicUtils.py        Bond finding, file I/O
├── AtomicSystem.py       Core molecular container
├── AtomicGraph.py        Graph operations
├── KekulePure.py         Pure-Python Kekulé optimizer (CPU reference)
├── KekuleBackend.py      Hexagonal grid backend
├── plotUtils.py          Molecular plotting utilities
├── ascii_art_heterocycle.py   ASCII art parser + 28 examples
└── heterocycle_generator.py   Geometry generation + plotting
```

---

## 11. Dependencies

- **Python 3** with **NumPy** and **Matplotlib**
- **PyOpenCL** (for GPU acceleration; falls back gracefully if not installed)
- **ChemicalGraphs** package (for ASCII art molecule building; optional)

No other external dependencies. The OpenCL kernels are self-contained and can
be used from C/C++/Fortran harnesses as well.
