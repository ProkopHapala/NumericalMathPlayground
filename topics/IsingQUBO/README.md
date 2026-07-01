# IsingQUBO — Ising/QUBO Solver Hierarchy for Molecular Charge Systems

## Problem Statement

Minimize the classical energy of a 2D molecular charge system:

```
E(n) = sum_i  eps_i * n_i  +  (1/2) * sum_{i!=j}  V_ij * n_i * n_j

n_i in {0,1}   (binary charge occupancy)
```

- `eps_i`: local on-site energy (charging energy + chemical potential + tip/gate field)
- `V_ij > 0`: Coulomb repulsion (dense N×N, or sparse CSR if screened)
- N = 100–1000 sites, 2D rectangular or hexagonal lattice, finite island
- Scan over many tip positions `(r_tip, V_bias)` — need fast, GPU-parallel evaluation
- "Very likely correct" is acceptable; bifurcations and degeneracies are features, not bugs

### Spin representation

Using `s_i = 2*n_i - 1 in {-1,+1}`, the energy becomes:

```
E(s) = -(1/2) sum_{ij} J_ij s_i s_j - sum_i h_i s_i + const

where J_ij = -V_ij/4,  h_i = -eps_i/2 - (1/4) sum_j V_ij
```

Repulsive `V_ij > 0` becomes antiferromagnetic Ising coupling `J_ij < 0` — this matches
checkerboards, stripes, Wigner patterns.

### Ensemble modes

| Mode | Constraint | Allowed moves | Physical scenario |
|------|-----------|---------------|-------------------|
| **Grand-canonical** | `N_q = sum_i n_i` is free | Single flips `0 <-> 1` | Exchange with electrode, substrate, tip, reservoir |
| **Fixed-charge** | `sum_i n_i = N_q` fixed | Pair relocations only | Isolated island, fixed electron count, MQCA propagation |

---

## Directory Structure

```
IsingQUBO/
├── README.md                ← this file
├── IsingQUBO.md             ← design document (full specification)
├── IsingQUBO.chat.md        ← raw LLM synthesis transcript
├── IsingQUBO.ChatGPT55.md   ← ChatGPT 5.5 conversation transcript
├── IsingSolver.py           ← solver hierarchy implementation
└── test_IsingSolver.py      ← benchmark & test suite
```

---

## Solver Hierarchy (`IsingSolver.py`)

The solver hierarchy goes from fastest (least reliable) to most reliable.
All methods return binary occupancy `n` and energy `E`.

### L0: Greedy Local Descent — `greedy_polish()`

Repeatedly applies the best single flip (grand-canonical) or pair relocation
(fixed-charge) until no improving move exists. Used as a **polishing step**
after all other methods.

- Cost: O(Nz) per sweep (sparse) or O(N²) (dense)
- Role: warm-start tip tracking, final discretization

### L1: Mean-Field Annealing — `mfa_anneal()`

Replaces binary `n_i` with probabilities `m_i in [0,1]`. Minimizes the
variational free energy using Fermi-function self-consistent updates:

```
m_i = 1 / (1 + exp(beta * g_i)),  g_i = eps_i + sum_j V_ij m_j
```

For fixed-charge, a chemical potential `mu` is found by bisection at each
iteration to enforce `sum_i m_i = N_q`.

- Cost: O(B·N²) per iteration (dense) or O(B·Nz) (sparse)
- Role: first solver, susceptibility analysis, candidate generation
- Batch size `B` for independent random starts

### L2: Lanczos on MFA Susceptibility — `lanczos_mfa()`

Computes the softest collective modes of the MFA susceptibility matrix
`A = I + beta * D^{1/2} V D^{1/2}` where `D_i = m_i(1-m_i)`. The instability
occurs when `1 + beta * lambda_min -> 0`.

- Cost: O(k·N²) or O(k·Nz), k ~ 10-20
- Role: **diagnostic** — detects where the system is about to bifurcate

### L3: Simulated Bifurcation — `sb_solve()`

Maps to spin variables `s_i = 2*n_i - 1`, then to continuous oscillators
`x_i in R` that undergo a pitchfork bifurcation via a double-well potential:

```
F_i = sum_j J_ij x_j + h_i - 4*lambda * x_i * (x_i^2 - 1)
x_i <- x_i + eta * F_i
```

Linear ramp of `lambda` from 0 to `lambda_max` forces discretization.
Discrete variant uses `sign(x_j)` in the force for better QUBO convergence.

- Cost: O(N_steps · B · Nz), typically ~10⁸ ops for N=1000
- Role: **primary workhorse** for reliable ground states
- Best candidate for GPU parallelization

### L4: Candidate Pool — `CandidatePool` class

Pool of K distinct low-energy configurations for scan reuse. For each new
tip position, re-evaluates all pool energies cheaply:

```
E_a(R) = E_a^V + eps(R) . n_a
```

where `E_a^V = (1/2) n_a^T V n_a` is stored once (drift-free). Prevents
warm-start hysteresis from hiding alternative basins.

- `add(n, eps)` — add unique configuration
- `update_energies(eps)` — recompute for new tip position
- `get_best()` / `get_sorted()` / `get_gap()` — query pool
- `prune(E_best)` — keep only states within `delta_E_keep`

### L5: Exact Enumeration — `exact_ground_state()` / `exact_all_low_energy()`

Brute-force over all `2^N` configurations (grand-canonical) or `C(N, N_q)`
(fixed-charge). Only feasible for N ≤ 24.

- `exact_ground_state(eps, V)` — returns `(n_gs, E_gs)`
- `exact_all_low_energy(eps, V, E_threshold)` — all states within threshold

### Tip Scanner — `scan_tip_positions()`

Scans over tip positions using candidate pool + SB/MFA for ground-state
tracking. Combines hysteretic warm-start with equilibrium pool re-evaluation.

---

## Key Primitives

| Function | Description |
|----------|-------------|
| `compute_energy(n, eps, V)` | `E = eps·n + ½ n^T V n` |
| `compute_fields(n, eps, V)` | Local field `g_i = eps_i + sum_j V_ij n_j` |
| `flip_cost(g, n)` | Cost to flip each site: `Delta_i = (1-2n_i) g_i` |
| `relocate_cost(g, V, i, j)` | Cost to move charge from i to j |
| `n_to_spin(n)` / `spin_to_n(s)` | Convert between {0,1} and {-1,+1} |
| `spin_coefficients(eps, V)` | Map to Ising `(J, h)` |

---

## Test Suite (`test_IsingSolver.py`)

Benchmarks all solver levels against exact enumeration on systems designed
to have degenerate or near-degenerate ground states:

| Test | Description |
|------|-------------|
| 1. No interaction (V=0) | Trivial — catches sign bugs |
| 2. Fixed-charge no interaction | Occupy N_q lowest-eps sites |
| 3. 1D chain NN repulsion | Exact via dynamic programming |
| 4. 2D square checkerboard | Two-fold degeneracy, Lanczos soft modes |
| 5. Tip phase selection | Tip selects one checkerboard sublattice |
| 6. Random small system | Compare all methods vs exact |
| 7. Tip scan | Energy & gap vs tip position |
| 8. Scaling benchmark | Timing vs system size |

**Usage:**
```bash
python test_IsingSolver.py                          # full suite
python test_IsingSolver.py --N 16 --seed 42         # custom size
python test_IsingSolver.py --no-show --skip-scan    # fast mode
```

Output: pass/fail for each test + summary plot (`IsingQUBO_test.png`).

---

## Lattice Builders

| Function | Description |
|----------|-------------|
| `build_square_lattice(nx, ny, a)` | Square lattice with NN edges |
| `build_hex_lattice(nx, ny, a)` | Honeycomb (brick-wall) lattice |
| `build_coulomb_matrix(N, edges, V_nn, long_range)` | NN-only or 1/r Coulomb |
| `build_tip_field(N, pos, tip_R, V_tip, sigma)` | Gaussian AFM tip field |

---

## Design Document

The full design specification, including method comparison, GPU strategy,
and implementation details, is in `IsingQUBO.md`. The raw LLM synthesis
transcripts are in `IsingQUBO.chat.md` and `IsingQUBO.ChatGPT55.md`.
