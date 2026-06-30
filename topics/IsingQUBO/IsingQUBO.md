# Ising/QUBO Solver — Design Document

Synthesis of LLM responses (ChatGPT 5.5, DeepSeek-V4-pro, Gemini-3.1-pro, Grok, Qwen-3.7-Max, Mistral) into a focused implementation specification.

---

## 1. Problem Formulation

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

**Convention notes**:
- This assumes `V_ii = 0` and `V_ij = V_ji` (symmetric)
- The `sum_{ij}` form double-counts pairs with the prefactor `1/2`; equivalently use `sum_{i<j}` without the `1/2`
- Repulsive `V_ij > 0` becomes antiferromagnetic Ising coupling `J_ij < 0` — this matches checkerboards, stripes, Wigner patterns
- Switching between `sum_{i<j}` and `0.5 * sum_{ij}` conventions is a common source of bugs; pick one and be consistent

Both representations are used below; MFA works naturally with `n_i in [0,1]`, SB with `s_i in [-1,+1]`.

### Ensemble modes

Two physically distinct cases:

| Mode | Constraint | Allowed moves | Physical scenario |
|------|-----------|---------------|-------------------|
| **Grand-canonical** | `N_q = sum_i n_i` is free | Single flips `0 <-> 1` | Exchange with electrode, substrate, tip, reservoir |
| **Fixed-charge** | `sum_i n_i = N_q` fixed | Pair relocations only `n_i=1,n_j=0 -> n_i=0,n_j=1` | Isolated island, fixed electron count, MQCA propagation |

For MQCA / cellular-automaton propagation, **fixed-charge** is often more relevant: the signal is not "creating charge from nowhere" but moving or polarizing an existing charge pattern. The solver should have a mode flag:

```
ensemble = grand_canonical | fixed_charge
```

For fixed-charge MFA, a chemical potential `mu` must be adjusted at every iteration (see Section 4).

### Key derived quantities

```
Local field:       g_i = eps_i + sum_j V_ij * n_j    (or m_j for MFA)
Single flip cost:  Delta_i = (1 - 2*n_i) * g_i
Pair relocation:   Delta_{i->j} = -g_i + g_j - V_ij   (for n_i=1, n_j=0)
```

---

## 2. Hierarchy of Methods (Fastest → Most Reliable)

| Level | Method | State | Cost/step (sparse) | Cost/step (dense) | GPU | Reliability | Role |
|-------|--------|-------|---------------------|--------------------|-----|-------------|------|
| **L0** | Greedy local descent | `n[N] int, g[N] float` | O(Nz) per sweep | O(N^2) | Yes | Low (local min) | Polish + tip tracking |
| **L1** | MFA annealing + round + greedy | `m[N] float` | O(Nz) per iter | O(N^2) | Yes | Medium | First solver, susceptibility |
| **L2** | MFA + Lanczos susceptibility | `+ D[N], q_k[N]` | O(kNz) | O(kN^2) | Yes | Diagnostic | Soft-mode detection |
| **L3** | Simulated Bifurcation + greedy | `x[N], p[N] float` | O(Nz) per step | O(N^2) | **Best** | High | Primary GPU workhorse |
| **L4** | Candidate pool / lower-envelope | `K * n[N]` | O(KN) per tip | O(KN) | Yes | High | Scan tracking, degeneracy |
| **L5** | Parallel Tempering MC | R replicas | O(Nz) per sweep | O(N^2) | Poor (branch) | Very high | Validation only |

**Consensus minimum**: L1 (MFA + greedy polish) for quick scans; L3 (SB + greedy polish) for reliable ground states. L2 is diagnostic. The candidate pool (Section 7) is a **core module** from the start, not an optional layer.

---

## 3. Level 0: Greedy Local Descent

### Equations

Given a binary state `n` and local fields `g`:

```
g_i = eps_i + sum_j V_ij * n_j

Single flip cost:   Delta_i = (1 - 2*n_i) * g_i
  - If n_i=0: Delta_i = +g_i    (cost to charge site i)
  - If n_i=1: Delta_i = -g_i    (cost to discharge site i)

Pair relocation (n_i=1, n_j=0 -> n_i=0, n_j=1):
  Delta_{i->j} = -g_i + g_j - V_ij
```

### Pseudocode

```
function greedy_polish(n, eps, V, N, ensemble='grand_canonical'):
    g = eps + V @ n                          # O(N^2) or O(Nz) sparse
    loop:
        # 1. Find best single flip (grand-canonical only)
        if ensemble == 'grand_canonical':
            delta_single = (1 - 2*n) * g     # O(N)
            i_min = argmin(delta_single)

            if delta_single[i_min] < 0:
                dn = 1 - 2*n[i_min]          # Compute BEFORE flip: +1 for 0->1, -1 for 1->0
                n[i_min] = 1 - n[i_min]      # flip
                g += V[:, i_min] * dn        # O(N) or O(z) sparse
                continue

        # 2. Find best pair relocation (active-set version)
        #    Removal costs: r_i = -g_i for occupied sites (n_i=1)
        #    Insertion costs: a_j = g_j for empty sites (n_j=0)
        #    Relocation cost: Delta_{i->j} = r_i + a_j - V_ij
        #    Only test best K_occ removals x K_emp insertions
        K_act = 16  # or 32
        occupied = {k : n[k]==1}
        empty = {k : n[k]==0}

        removal_costs = [( -g[i], i) for i in occupied]   # r_i = -g_i
        insertion_costs = [( g[j], j) for j in empty]      # a_j = g_j
        removal_costs.sort()                               # ascending = most favorable first
        insertion_costs.sort()

        best_delta = 0
        best_ij = None
        for (r_i, i) in removal_costs[:K_act]:
            for (a_j, j) in insertion_costs[:K_act]:
                d = r_i + a_j - V[i,j]
                if d < best_delta:
                    best_delta = d
                    best_ij = (i,j)

        if best_ij is not None:
            i, j = best_ij
            n[i] = 0                        # discharge site i
            n[j] = 1                        # charge site j
            g += -V[:,i] + V[:,j]           # O(N) or O(2z) sparse
            continue

        break  # converged: local minimum
    return n, g
```

**Critical**: The `dn` for single flips must be computed **before** modifying `n[i]`. Computing it after the flip gives the opposite sign. Similarly, for pair relocation, the `g` update uses the **original** assignments (`n[i]` was 1, `n[j]` was 0), so `g += -V[:,i] + V[:,j]`.

### Memory layout

| Array | Size | Type | Notes |
|-------|------|------|-------|
| `n` | N | int8/bool | Binary occupancy |
| `g` | N | float32 | Local fields (maintained incrementally) |
| `eps` | N | float32 | On-site energies (tip-dependent, recomputed per tip position) |
| `V` | N×N or Nz | float32 | Coulomb matrix (dense or CSR sparse) |

### Complexity

- Initial `g = eps + V@n`: O(N^2) dense, O(Nz) sparse
- Each accepted flip: O(N) dense update of `g`, O(z) sparse
- Each accepted pair relocation: O(N) dense, O(2z) sparse
- Typical convergence: 1–20 flips for warm-start tracking

### Parallelizability

- The `g = V @ n` initial computation is a standard SpMV/GEMV — fully parallel
- Finding `argmin(delta_single)` is a parallel reduction
- The pair-relocation search uses an **active-set** approach: sort removal costs `r_i = -g_i` (occupied) and insertion costs `a_j = g_j` (empty), test only the best `K_act` of each, giving O(K_act^2) instead of O(N^2). With K_act=16–32, this is cheap and catches most useful relocation moves
- **Best use**: serial CPU for small N, or as a post-processing step after SB/MFA on GPU

---

## 4. Level 1: Mean-Field Annealing (MFA)

### Physical basis

Replace binary `n_i in {0,1}` with probabilities `m_i in [0,1]`. Minimize the variational free energy:

```
F(m) = sum_i eps_i m_i + (1/2) sum_{i!=j} V_ij m_i m_j
       + T sum_i [m_i ln(m_i) + (1-m_i) ln(1-m_i)]
```

The entropy term `T * [m ln m + (1-m) ln(1-m)]` comes from the factorized variational distribution `P(n) ~ prod_i m_i^{n_i} (1-m_i)^{1-n_i}`.

### Self-consistent update (Fermi function)

Stationarity `dF/dm_i = 0` gives:

```
m_i = 1 / (1 + exp(beta * g_i))

where g_i = eps_i + sum_j V_ij * m_j
      beta = 1/T
```

**Fixed-charge variant**: If `sum_i n_i = N_q` is fixed, introduce a chemical potential `mu`:

```
m_i = 1 / (1 + exp(beta * (g_i - mu)))

with mu chosen so that sum_i m_i = N_q
```

This is typically a small bisection loop per MFA iteration: adjust `mu` until `sum_i m_i = N_q` within tolerance.

### Annealing algorithm

```
function MFA_anneal(eps, V, N, B_replicas, ensemble='grand_canonical', N_q=None):
    # B = batch size (different random seeds or tip positions)
    # State: m[B, N] float32

    # Energy scale for beta schedule (dimensionally correct)
    E_scale = max_i ( |eps_i| + sum_j |V_ij| )   # max local field magnitude
    beta = 0.01 / E_scale                         # High temperature
    m = 0.5 + noise[B, N] * 0.01                  # near-uniform start + small noise

    if ensemble == 'fixed_charge':
        mu = 0.0   # chemical potential, adjusted by bisection

    while beta < 10.0 / E_scale:
        for iter in 1..N_inner:           # N_inner ~ 5-50
            g = eps + V @ m               # Batched SpMV: [B,N] = [N,N] @ [B,N]^T
                                            # O(B*Nz) sparse, O(B*N^2) dense
            if ensemble == 'fixed_charge':
                # Bisection on mu to enforce sum_i m_i = N_q
                mu = bisect_charge_constraint(g, beta, N_q, m)
                m_new = 1.0 / (1.0 + exp(beta * (g - mu)))
            else:
                m_new = 1.0 / (1.0 + exp(beta * g))   # O(B*N), elementwise
            m = (1 - alpha) * m + alpha * m_new    # Damped update, alpha ~ 0.1-0.5

        beta *= beta_factor                # Cool down: beta_factor ~ 1.03-1.15

    # Discretize
    n = (m > 0.5).astype(int)              # Round to binary
    for b in 1..B:
        n[b], _ = greedy_polish(n[b], eps, V, N, ensemble=ensemble)  # L0 polish

    return n, m
```

### Key parameters

| Parameter | Typical value | Notes |
|-----------|--------------|-------|
| `E_scale` | `max_i ( |eps_i| + sum_j |V_ij| )` | Max local field magnitude; sets energy scale |
| `beta_init` | `0.01 / E_scale` | High temperature, smooth start |
| `beta_max` | `10.0 / E_scale` | Near-zero temperature |
| `beta_factor` | 1.03–1.15 | Cooling rate (geometric schedule). Use small steps for fragile bifurcation tracking; 1.5 only for crude tests |
| `N_inner` | 5–50 | Iterations per temperature |
| `alpha` (damping) | 0.1–0.5 | Critical for convergence with repulsive couplings |
| `B` (batch) | 16–64 | Different random seeds or tip positions |

### Memory layout

| Array | Size | Type | Notes |
|-------|------|------|-------|
| `m` | B×N | float32 | Mean-field occupancies (batched) |
| `g` | B×N | float32 | Effective fields (workspace) |
| `eps` | B×N or N | float32 | On-site energies (shared if same tip) |
| `V` | N×N or Nz | float32 | Coulomb matrix (shared across batch) |

### Complexity

- Per inner iteration: `g = eps + V @ m` → O(B·Nz) sparse, O(B·N^2) dense
- Per temperature level: O(N_inner · B · Nz)
- Total: O(N_T · N_inner · B · Nz) where N_T ~ 20–100 temperature levels
- For N=1000, z=6, B=64, N_T=50, N_inner=20: ~4×10^7 edge ops total

### Parallelizability

- **Excellent**: The core operation is batched matrix-vector multiply (GEMM/SpMV)
- Each element of `m_new = sigmoid(-beta * g)` is independent — fully parallel
- No branch divergence, no stochastic acceptance
- GPU: store `V` in read-only global memory, `m` and `g` as B×N row-major tensors

### Failure modes

1. **Fractional trapping**: Sites with `g_i ≈ 0` get stuck at `m_i = 0.5` indefinitely
2. **Oscillation**: Without damping, low-T iterations can cycle for frustrated systems
3. **Smeared degeneracy**: Two competing checkerboard patterns average to `m_i ≈ 0.5` everywhere
4. **No correlation**: MFA assumes `P(n) = prod_i P_i(n_i)`, missing checkerboard/stripe order

### When to use

- Quick smooth susceptibility scans
- Warm-start initialization for SB
- Active-site detection (sites with `m_i ≈ 0.5` are the "soft" ones)
- **Not** as final ground-state oracle for frustrated systems

---

## 5. Level 2: Lanczos on MFA Susceptibility Matrix

### What it does

Lanczos does **not** find the ground state. It analyzes the **stability** of a converged MFA solution to find collective soft modes — directions in which the charge distribution is about to bifurcate.

### Derivation

Linearize the MFA update `m_i = sigma(-beta * g_i)` around a fixed point `m*`:

```
delta_m = -beta * D * (delta_eps + V * delta_m)

where D_i = m_i * (1 - m_i)    (diagonal matrix, local susceptibility)
```

Rearranging:

```
(I + beta * D * V) * delta_m = -beta * D * delta_eps
```

The matrix `D*V` is not symmetric. Use the symmetric similar matrix:

```
A = I + beta * D^{1/2} * V * D^{1/2}
```

### Lanczos procedure

Lanczos finds the smallest eigenvalues/eigenvectors of `A` using only matrix-vector products:

```
function lanczos_MFA(m, V, beta, N, k_eigen):
    D = m * (1 - m)                          # O(N)
    D_sqrt = sqrt(D)                         # O(N)

    # Lanczos iteration for k_eigen smallest eigenvalues
    q = random_unit_vector(N)               # q[N]
    Q = []                                   # Store Lanczos vectors

    for k in 1..k_eigen:
        # Matrix-vector product A @ q:
        #   A @ q = q + beta * D^{1/2} * V * (D^{1/2} * q)
        w = D_sqrt * q                       # O(N) elementwise
        u = V @ w                            # O(Nz) or O(N^2)  ← THE bottleneck
        v = D_sqrt * u                       # O(N)
        Aq = q + beta * v                    # O(N)

        # Standard Lanczos orthogonalization + tridiagonalization
        # (see ARPACK / scipy.sparse.linalg.eigsh)
        ...

    return eigenvalues, eigenvectors
```

### Three distinct Lanczos applications

| Application | Matrix | What it reveals | When to use |
|-------------|--------|-----------------|-------------|
| **A: On V itself** | `V` (N×N) | Intrinsic "wiring modes" — collective charge-density wave patterns (checkerboard, stripes) | Precomputation, once per system geometry |
| **B: On MFA Hessian** | `A = I + beta * D^{1/2} V D^{1/2}` | Which sites are about to bifurcate as T decreases or tip moves | During MFA annealing, at selected temperatures |
| **C: On config graph** | `H_graph = diag(E) + Gamma * Adj` (M×M, M~50-500) | Switching pathways between metastable states | Post-processing, after collecting low-energy states |

### Application B: MFA susceptibility (most useful for live scanning)

```
A = I + beta * D^{1/2} * V * D^{1/2}

D_i = m_i * (1 - m_i)
```

- `D_i` is large (up to 0.25) only when `m_i ≈ 0.5` (active/uncertain sites)
- `D_i → 0` for frozen sites (`m_i ≈ 0` or `1`) — they drop out automatically
- **Smallest eigenvalue of A → 0** means a bifurcation is imminent
- **Corresponding eigenvector** shows which collective charge pattern will nucleate

**Important spectral note**: For repulsive nearest-neighbor systems, `V` can have **negative eigenvalues** because its diagonal is zero and off-diagonal terms are positive. For example, a bipartite adjacency-like `V` has positive and negative spectral branches. The instability occurs when:

```
1 + beta * lambda_min(D^{1/2} V D^{1/2}) -> 0
```

So the relevant soft modes are the **most negative eigenmodes** of `D^{1/2} V D^{1/2}`, or equivalently the smallest eigenvalues of `A`. Do not assume "positive Coulomb matrix = all eigenvalues positive" — this is false for off-diagonal repulsion matrices.

### Application C: Configuration graph (post-processing)

```
1. Collect M ~ 50-500 distinct low-energy integer configurations from SB restarts
2. Build adjacency: connect configs a,b if Hamming distance = 1 or 2
   (i.e., differ by a single flip or pair relocation)
3. Form H_graph[a,b] = E_a * delta_{ab} + Gamma * Adj_{ab}
   (Gamma = small fictitious hopping, e.g. 0.01 * typical energy scale)
4. Lanczos diagonalize H_graph (M×M, tiny)
5. Lowest eigenvectors → collective switching pathways between metastable states
```

### Memory and cost

| Application | Matrix size | Matvec cost | k iterations | Total |
|-------------|------------|-------------|--------------|-------|
| A (on V) | N×N | O(Nz) or O(N^2) | k~20 | O(kNz) |
| B (on A) | N×N (implicit) | O(Nz) or O(N^2) | k~20 | O(kNz) |
| C (on graph) | M×M (M~500) | O(M·d) d~avg degree | k~20 | O(kMd) |

### Parallelizability

- The matvec `V @ w` is the same SpMV/GEMV as everywhere else — fully parallel
- Lanczos orthogonalization is O(kN), small overhead
- Can use `cuSOLVER` or `ARPACK` (via `scipy.sparse.linalg.eigsh`)

---

## 6. Level 3: Simulated Bifurcation (SB)

### Core idea

Map each binary variable to a continuous oscillator that undergoes a pitchfork bifurcation, mechanically snapping from a single-well (unstable center) to a double-well (two stable minima at ±1).

**Important**: SB should be formulated in **spin variables** `s_i = 2*n_i - 1`, using the Ising coupling matrix `J` and field `h`, not the original `eps, V` directly. The spin mapping is:

```
J_ij = -V_ij / 4        (antiferromagnetic for repulsive V)
h_i  = -eps_i / 2 - (1/4) sum_j V_ij

E(s) = -(1/2) s^T J s - h^T s + const
```

A continuous SB state `x_i` gives the spin by `s_i = sign(x_i)`, and the charge by `n_i = (1 + s_i) / 2`.

### Three distinct variants — do not mix them

#### 3A. Penalty relaxation (continuous, n-space)

If you want `x_i in [0,1]` with wells at 0 and 1:

```
E_penalty(x) = eps . x + (1/2) x^T V x + lambda * sum_i x_i^2 * (1 - x_i)^2

Force: F_i = -eps_i - sum_j V_ij x_j - lambda * (2*x_i - 6*x_i^2 + 4*x_i^3)
Update: x_i <- x_i - eta * F_i
```

This is simple but **not true SB** — it is continuous penalty annealing in n-space.

#### 3B. Simulated Bifurcation (spin-space, recommended)

Use spin variables `x_i in [-1,1]` with the Ising coefficients `J, h`:

```
E_cont(x) = -(1/2) x^T J x - h^T x + lambda * sum_i (x_i^2 - 1)^2

Force: F_i = -dE/dx_i = sum_j J_ij x_j + h_i - 4*lambda * x_i * (x_i^2 - 1)
Update: x_i <- x_i + eta * F_i    (note: += because F = -dE/dx, gradient descent is x -= eta*dE/dx = x + eta*F)
```

At `lambda=0`: pure gradient flow on the Ising energy (like MFA at T=0).
At `lambda > V_crit`: double well at `x = ±1`, forcing discretization.

#### 3C. Discrete SB (most robust for QUBO/Ising)

Same as 3B but the force uses `sign(x_j)` instead of `x_j`:

```
F_i = sum_j J_ij * sign(x_j) + h_i - 4*lambda * x_i * (x_i^2 - 1)
```

Forces are computed from the current **discrete** state, not continuous positions. This gives better convergence and less oscillation, at the cost of losing smooth gradient information.

### Schedule

```
lambda(t): start at 0 (pure gradient flow = MFA at T=0)
           ramp up linearly or geometrically to lambda_max
           At lambda=0: single well at x=0
           At lambda > V_crit: double well at x=±1

eta (step size): ~0.01-0.1, may need adaptive tuning
typical steps: 100-1000
```

### Hamiltonian SB (with momentum, better for frustrated systems)

```
dx_i/dt = a0 * p_i
dp_i/dt = -(a0 - a(t)) * x_i - b * x_i^3 + c * (sum_j J_ij * sign(x_j) + h_i) - gamma * p_i

a(t): ramp from 0 to a0 (pump)
b: quartic coefficient (bifurcation strength)
c: coupling scaling
gamma: dissipation (0 = ballistic/conservative, >0 = damped)
```

### Pseudocode (spin-space overdamped SB, batched)

```
function SB_solve(eps, V, N, B_replicas, lambda_max, N_steps, eta):
    # Precompute spin-space coefficients from n-space eps, V
    J = -V / 4.0                              # J_ij = -V_ij/4  (N×N)
    h = -eps / 2.0 - 0.25 * V.sum(axis=1)     # h_i = -eps_i/2 - (1/4) sum_j V_ij

    # State: x[B, N] float32 (spin-like continuous variables)
    x = random_uniform(-0.1, 0.1, shape=(B, N))   # Small random init

    for step in 1..N_steps:
        lambda = lambda_max * step / N_steps       # Linear ramp

        # Force = J@x + h - 4*lambda * x*(x^2-1)
        #   (or discrete: J@sign(x) + h - ...)
        force = J @ x + h                          # Batched GEMV: O(B*Nz) or O(B*N^2)
        force -= 4 * lambda * x * (x*x - 1)        # O(B*N), elementwise

        x += eta * force                            # Gradient descent: x += eta * (-dE/dx)

        # Optional: clamp x to [-2, 2] to prevent blowup

    # Discretize
    s = sign(x)                                     # s_i in {-1,+1}
    n = (s + 1) / 2                                 # n_i in {0,1}

    # Greedy polish each replica (in n-space with original eps, V)
    for b in 1..B:
        n[b], _ = greedy_polish(n[b], eps, V, N)

    return n, x
```

### Memory layout

| Array | Size | Type | Notes |
|-------|------|------|-------|
| `x` | B×N | float32 | Continuous spin-like positions (batched) |
| `p` | B×N | float32 | Momenta (only for Hamiltonian SB) |
| `force` | B×N | float32 | Workspace (gradient) |
| `J` | N×N or Nz | float32 | Ising coupling matrix (precomputed from V) |
| `h` | N | float32 | Ising field (precomputed from eps, V) |
| `eps` | B×N or N | float32 | Original on-site energies (for greedy polish) |
| `V` | N×N or Nz | float32 | Original Coulomb matrix (for greedy polish) |

For N=1000, B=64: `x` + `p` + `force` = 3 × 64 × 1000 × 4 bytes = **768 KB** (trivially fits in GPU memory).

### Complexity

- Per step: `J @ x` → O(B·Nz) sparse, O(B·N^2) dense
- Total: O(N_steps · B · Nz)
- For N=1000, z=6, B=64, N_steps=500: ~2×10^8 edge ops. Potentially **milliseconds to tens of milliseconds** on a good GPU, depending on memory layout, batching, and sparse access pattern. Not sub-millisecond in practice due to kernel launch overhead, reductions, and state polishing.
- For dense N=1000, B=64, N_steps=500: ~3×10^10 flops. Feasible only if expressed as efficient **batched GEMM** (`G = X @ J^T` where `X in R^{B×N}`), not as repeated GEMV. Dense repeated GEMV is memory-bandwidth limited and much worse than GEMM. If many replicas share the same `J`, compute `G = X @ J^T` as one matrix-matrix multiply.

### Is SB combinatorial? NO.

- State: one vector `x in R^N` per replica (plus optional momentum `p in R^N`)
- No `2^N` object is ever constructed
- Each step is a single matrix-vector multiply + elementwise nonlinear update
- The "combinatorial difficulty" is hidden in the number of restarts needed to find the global minimum, not in the per-step cost

### SB vs MFA

| Aspect | MFA | SB |
|--------|-----|-----|
| Variables | `m_i in [0,1]` (probabilities) | `x_i in R` (continuous spin-like) |
| Nonlinearity | Fermi/sigmoid (exponential) | Polynomial `x^3` (cheaper on GPU) |
| Discretization | Threshold at `m>0.5` (can fail at 0.5) | Double-well forces `|x|->1` (more robust) |
| Physical meaning | Free energy, entropy, Fermi-Dirac | Mechanical oscillator, bifurcation |
| Barrier crossing | No (dissipative, gradient-only) | Yes (inertia/momentum helps) |
| GPU friendliness | Good (GEMV + exp) | **Better** (GEMV + polynomial, no exp) |
| Typical steps | 50-500 (inner × temperature levels) | 100-1000 time steps |

**Recommended strategy**: Use MFA for diagnosis and initialization, SB for discrete candidate generation. Both feed into the same greedy polish. SB, MFA, and random starts are all **candidate generators** — the final accepted states should always be greedily polished and ranked by the original binary energy `E(n)`.

---

## 7. Candidate Pool / Lower-Envelope Tracking (Core Module)

### Purpose

The candidate pool is **not** an optional high-level add-on. It is a core module from the beginning. For scan problems, it is the main trick that exploits continuity in tip position and prevents warm-start hysteresis from hiding alternative basins.

During a tip scan, maintain a pool of K distinct low-energy configurations. When the tip moves, re-evaluate all pool energies cheaply and detect degeneracy crossings (Coulomb diamond boundaries).

### Energy evaluation (drift-free)

If `V` does not change between tip positions, store the interaction energy once:

```
E_a^V = (1/2) n_a^T V n_a    (computed once, stored)
```

Then for any new tip position with on-site energies `eps(R)`:

```
E_a(R) = E_a^V + eps(R) . n_a
```

This is cleaner than using `E_old + delta_eps.dot(n)`, because it avoids accumulating numerical drift. It is just a dot product — **O(N) per candidate, O(KN) total**.

### Pseudocode

```
function scan_tip_positions(tip_positions, V, eps0, N):
    pool = []                          # List of (n, E) pairs, max size K
    n_prev = None

    for R in tip_positions:
        eps = eps0 + compute_tip_field(R)
        candidates = []

        # 1. Warm-start from previous solution (hysteretic mode)
        if n_prev is not None:
            n = copy(n_prev)
            n, _ = greedy_polish(n, eps, V, N)
            candidates.append(n)

        # 2. Re-evaluate and polish pool states
        for (n_pool, _) in pool:
            n = copy(n_pool)
            n, _ = greedy_polish(n, eps, V, N)
            candidates.append(n)

        # 3. Batched MFA or SB random restarts
        n_batch = SB_solve(eps, V, N, B=16, ...)   # or MFA_anneal(...)
        candidates.extend(n_batch)

        # 4. Select best and update pool
        E_best = min(E(n) for n in candidates)
        n_best = argmin(...)
        n_prev = n_best

        # Update pool with unique low-energy states
        for n in candidates:
            if E(n) < E_best + delta_E_threshold:
                if n not in pool:   # Check uniqueness (Hamming distance)
                    pool.insert(n, E(n))
        pool.keep_top_K(K=100)

        # 5. Detect degeneracy crossings
        if len(pool) >= 2:
            E_0 = pool[0].E
            E_1 = pool[1].E
            gap = E_1 - E_0
            if gap < gap_threshold:
                # This tip position is on a Coulomb diamond boundary!
                record_degeneracy(R, pool[0], pool[1], gap)

    return results
```

### Memory layout

| Array | Size | Type | Notes |
|-------|------|------|-------|
| `pool_n` | K×ceil(N/64) | uint64 | Bitpacked binary configurations |
| `pool_E_V` | K | float64 | Interaction energy `E_a^V = (1/2) n_a^T V n_a` (computed once) |
| `pool_E` | K | float64 | Total energy at current tip position |
| `pool_hash` | K | uint64 | Hash for fast uniqueness check |
| `pool_last_seen` | K | int | Tip step index when last competitive |
| `pool_basin_count` | K | int | How many times this basin was found |
| `eps` | N | float32 | Current on-site energies |

**Bitpacking**: `n` stored as `uint64[(N+63)/64]` — for N=1000, that's 16 uint64s = 128 bytes per config. For K=100 candidates: 12.8 KB total.

### Complexity

- Energy re-evaluation: O(KN) per tip position (dot products)
- Greedy polish of pool states: O(K · N_flips · z) — typically small
- SB/MFA restarts: O(B · N_steps · Nz) — the main cost
- Degeneracy detection: O(K log K) for sorting

### Two scanning modes

| Mode | Strategy | Captures |
|------|----------|----------|
| **Hysteretic** | Only warm-start from `n_prev` + greedy polish | Physical memory, hysteresis loops |
| **Equilibrium** | Warm-start + pool re-eval + SB/MFA restarts | True ground state, all basins |

For cellular automata: run **both** modes and compare. Hysteretic mode shows the physical evolution; equilibrium mode shows the thermodynamic ground state.

---

## 8. Gumbel-Softmax and Straight-Through Estimator (STE)

### What they are

**Gumbel-Softmax** (for binary: Gumbel-Sigmoid): A differentiable relaxation of discrete sampling.

```
# Binary case:
#   logit a_i = log(p_i) - log(1-p_i)  (log-odds)
#   Add Gumbel noise: g_i ~ Gumbel(0,1) = -log(-log(Uniform(0,1)))
#   Softmax with temperature tau:
m_i = exp((a_i + g_i) / tau) / (1 + exp((a_i + g_i) / tau))
     = sigmoid((a_i + g_i) / tau)

# As tau -> 0: m_i -> {0, 1} (one-hot/discrete)
# As tau -> inf: m_i -> 0.5 (uniform)
```

**Straight-Through Estimator (STE)**: A gradient hack for backprop through discrete steps.

```
Forward pass:  n_i = round(m_i)          # Hard discretization
Backward pass: d(n_i)/d(a_i) ≈ d(m_i)/d(a_i)   # Pretend it was continuous
```

### Relationship to MFA

Yes, they are structurally related:

| Feature | MFA | Gumbel-Softmax |
|---------|-----|----------------|
| Update rule | `m_i = sigmoid(-beta * g_i)` | `m_i = sigmoid((a_i + g_i) / tau)` |
| Temperature | `beta = 1/T` (physical) | `tau` (hyperparameter) |
| Noise | None (deterministic) | Gumbel noise (stochastic) |
| Self-consistency | Yes: `g_i = eps_i + sum_j V_ij m_j` | Only if you put interactions into logits |
| Derivation | Variational free energy | Reparameterization trick for autograd |

**Key insight**: If you make the logits self-consistent (`a_i = -beta * (eps_i + sum_j V_ij m_j)`) and remove the Gumbel noise, Gumbel-Softmax **reduces to MFA**. The Gumbel noise is an extra stochastic perturbation with no physical basis in your system.

### Why they're bad for ground-state physics

1. **Biased gradients**: STE gradient is deliberately fake — it correlates with descent direction but is not exact
2. **Unphysical noise**: Gumbel noise acts like an infinite-temperature heat bath, ignoring actual energy barriers
3. **Smeared transitions**: Differentiable relaxation turns sharp first-order transitions (branch crossings) into fake continuous paths
4. **No variational bound**: MFA minimizes a free energy upper bound; Gumbel/STE optimizes a surrogate loss with no such guarantee
5. **Temperature ambiguity**: `tau` is algorithmic, not physical `T`

### When they might be useful

- **Inverse design**: Later, if you want to optimize molecular geometry or gate parameters via autograd (PyTorch/JAX)
- **Fast stochastic candidate generator**: As a "noisy MFA" — but deterministic MFA is cleaner and faster
- **Prototyping**: Quick to implement in PyTorch, but adds autograd overhead

**Verdict**: Level 0.5 toy method. Do not implement as primary solver. Keep in mind for future inverse-design work.

---

## 9. Recommended Implementation Pipeline

### Phase 1: Core (CPU prototype, then GPU)

```
Module A: Energy kernel
  - compute_energy(n, eps, V) -> float
  - compute_fields(n, eps, V) -> g[N]
  - flip_cost(g, n, i) -> float
  - relocate_cost(g, V, i, j) -> float
  - Spin mapping: J = -V/4, h = -eps/2 - (1/4) sum_j V_ij

Module B: Greedy local descent (L0)
  - greedy_polish(n, eps, V, ensemble='grand_canonical'|'fixed_charge') -> (n, g)
  - Incremental g updates: dn computed BEFORE flip
  - Active-set pair relocation (K_act=16-32)
  - Test: compare with exact enumeration for N <= 20

Module C: Candidate pool (core, not optional)
  - Bitpacked storage: uint64[(N+63)/64] per config
  - Drift-free energy: E_a(R) = E_a^V + eps(R).n_a
  - Uniqueness check via hash + Hamming distance
  - Keep all states with E - E_best < delta_E_keep
```

### Phase 2: MFA solver (L1) + Lanczos diagnostics (L2)

```
Module D: MFA annealing
  - Batched: m[B,N], g[B,N]
  - E_scale = max_i (|eps_i| + sum_j |V_ij|); beta_init = 0.01/E_scale
  - Grand-canonical and fixed-charge modes (mu bisection)
  - Round + greedy polish
  - Test: compare with exact for N <= 20, check susceptibility maps

Module E: Lanczos on MFA susceptibility
  - A = I + beta * D^{1/2} V D^{1/2}
  - Note: V can have negative eigenvalues (off-diagonal repulsion)
  - Use scipy.sparse.linalg.eigsh (ARPACK) for prototyping
  - Visualize lowest eigenvectors as "soft mode" patterns
```

### Phase 3: SB solver (L3)

```
Module F: Simulated Bifurcation (spin-space)
  - Precompute J = -V/4, h = -eps/2 - (1/4) sum_j V_ij
  - Batched: x[B,N], p[B,N] (optional momentum)
  - Lambda/pump ramp schedule
  - Euler or symplectic integrator
  - sign(x) -> greedy polish in n-space with original eps, V
  - Test: compare with MFA + exact for N <= 20
  - Benchmark: speed vs MFA for N=100, 500, 1000
  - Use batched GEMM (X @ J^T) for dense, not repeated GEMV
```

### Phase 4: Tip scanner with candidate pool

```
Module G: Tip scanning
  - Hysteretic mode: warm-start only
  - Equilibrium mode: warm-start + pool re-eval + SB/MFA restarts
  - Degeneracy detection: track energy gaps in float64
  - Output: ground-state map + Coulomb diamond boundaries
  - Precision: float32 for dynamics, float64 for final energy ranking
```

### Phase 5: GPU acceleration

```
Module H: OpenCL/CUDA kernels
  - J @ x batched SpMV/GEMM (reuse KekuleQM.cl patterns)
  - For dense: use GEMM (X @ J^T), not repeated GEMV
  - Elementwise sigmoid / polynomial update
  - Parallel reduction for argmin(flip_cost)
  - Greedy polish: serial per replica, parallel across replicas
```

### Validation strategy

| System size | Validation method |
|-------------|-------------------|
| N <= 20 | Exact enumeration (brute force 2^N) |
| N <= 50 | Branch-and-bound (if available) or exhaustive MC |
| N = 100-1000 | Cross-check: MFA vs SB vs greedy, compare candidate pool diversity |

### Specific validation test cases

1. **Tiny exact enumeration** (N <= 24): Random `V, eps`, compare all methods with brute force
2. **No interaction test** (`V=0`): Ground state is simply `n_i=1 if eps_i<0`. Trivial but catches sign bugs
3. **Fixed-charge no interaction test**: Occupy the `N_q` sites with lowest `eps_i`
4. **1D nearest-neighbor repulsion** (small chain): Compare with dynamic programming / transfer matrix
5. **Bipartite nearest-neighbor square/hex graph**: For pure nearest-neighbor repulsion, after sublattice flip, the problem can be solved by min-cut. Use as exact benchmark for larger N
6. **Degenerate checkerboard test**: Square lattice, zero field, half filling. Ensure solver finds both checkerboard phases under different seeds
7. **Tip perturbation phase-selection test**: Add a weak local field and check whether the solver selects the expected shifted phase rather than creating an unphysical local defect

---

## 10. Precision Policy

Near degeneracy, float32 can reorder nearly equal states. This matters because Coulomb diamonds are exactly energy-crossing phenomena.

```
Use float32 for GPU dynamics: MFA iterations, SB integration, field updates.
Use float64 or compensated summation for final energy ranking of top candidates.
Report energy gaps only after recomputing E in double precision on CPU or high-precision GPU.
```

For candidate selection, store not only the best state but all states within a tolerance:

```
E_a - E_0 < delta_E_keep
```

The tolerance `delta_E_keep` should be larger than numerical noise (e.g. `delta_E_keep ~ 0.01 * E_scale`).

---

## 11. MQCA Output Metrics

The solver's ultimate goal is signal propagation and logic in molecular cellular automata. Track these observables:

```
Ground-state map:       n_i(R, Vbias)
Output response:        y(R, Vbias) = n_output or weighted output polarization
Energy gap:             gap = E_1 - E_0
Hamming distance:       d_H(n_0, n_1) between top two states
Switching locality:     spatial support of n_1 - n_0 (which sites differ)
Input-output gain:      d y_output / d eps_input
Hysteresis:             n_forward(R) != n_backward(R)
Candidate entropy:      number of states within delta_E of ground state
Robustness:             output stable under noise in eps_i, V_ij, tip position
```

These metrics make the solver directly useful for designing molecular cellular automata, not just minimizing a QUBO.

---

## 12. Summary: Which method for which task

| Task | Primary method | Fallback | Diagnostic |
|------|---------------|----------|------------|
| Fast tip scan (smooth maps) | MFA + greedy | — | Lanczos on V (precomputed) |
| Reliable ground state | SB + greedy polish | MFA + greedy | Compare top-K candidates |
| Track hysteresis | Greedy warm-start only | — | Energy gap to pool |
| Find Coulomb diamonds | SB multi-start + pool | — | Degeneracy surface tracking |
| Soft modes / signal propagation | — | — | Lanczos on MFA Hessian (B) |
| Switching pathways | — | — | Lanczos on config graph (C) |
| Validate correctness | — | Exact (N<=20) or PT | — |
| Inverse design (future) | — | — | Gumbel-Softmax / STE |

---

## Appendix: Key Formulas Quick Reference

```
# Energy (n-space)
E(n) = eps . n + (1/2) n^T V n,  V_ii=0, V_ij=V_ji

# Spin mapping
s_i = 2*n_i - 1
J_ij = -V_ij / 4  (antiferromagnetic for repulsive V)
h_i  = -eps_i / 2 - (1/4) sum_j V_ij
E(s) = -(1/2) s^T J s - h^T s + const

# Local field
 g_i = eps_i + sum_j V_ij n_j

# Single flip cost
Delta_i = (1 - 2*n_i) * g_i
# g update: dn = 1-2*n[i] BEFORE flip, then g += V[:,i]*dn

# Pair relocation cost (n_i=1, n_j=0 -> n_i=0, n_j=1)
Delta_{i->j} = -g_i + g_j - V_ij
# g update: g += -V[:,i] + V[:,j]  (using original assignments)

# MFA update (grand-canonical)
m_i = 1 / (1 + exp(beta * g_i)),  g_i = eps_i + sum_j V_ij m_j

# MFA update (fixed-charge)
m_i = 1 / (1 + exp(beta * (g_i - mu))),  sum_i m_i = N_q

# Energy scale for beta schedule
E_scale = max_i ( |eps_i| + sum_j |V_ij| )
beta_init = 0.01 / E_scale,  beta_max = 10.0 / E_scale

# MFA susceptibility matrix for Lanczos
A = I + beta * D^{1/2} V D^{1/2},  D_i = m_i(1-m_i)
# Note: V can have negative eigenvalues; soft modes = smallest eigenvalues of A

# SB (spin-space, overdamped)
E_cont(x) = -(1/2) x^T J x - h^T x + lambda * sum_i (x_i^2 - 1)^2
F_i = sum_j J_ij x_j + h_i - 4*lambda * x_i * (x_i^2 - 1)
x_i <- x_i + eta * F_i   (gradient descent: x += eta * (-dE/dx))

# SB (discrete variant)
F_i = sum_j J_ij * sign(x_j) + h_i - 4*lambda * x_i * (x_i^2 - 1)

# SB Hamiltonian dynamics
dx_i/dt = a0 * p_i
dp_i/dt = -(a0 - a(t)) * x_i - b * x_i^3 + c * (sum_j J_ij sign(x_j) + h_i) - gamma * p_i

# Candidate pool energy (drift-free)
E_a^V = (1/2) n_a^T V n_a   (computed once)
E_a(R) = E_a^V + eps(R) . n_a

# Discretization
s_i = sign(x_i),  n_i = (1 + s_i) / 2

# Precision policy
float32: GPU dynamics (MFA, SB, field updates)
float64: final energy ranking, gap reporting, degeneracy detection
```
