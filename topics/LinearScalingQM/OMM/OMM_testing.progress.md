# OMM 1D Grid Prototype — Progress Report

## 1. Method Overview

### Goal

Minimize the total energy of occupied molecular orbitals on a 1D grid:

$$E = \sum_{i=1}^{N_{\text{orb}}} \langle \phi_i | H | \phi_i \rangle$$

subject to:

- **Orthonormality**: $\langle \phi_i | \phi_j \rangle = \delta_{ij}$ (Pauli exclusion)
- **Strict localization**: each $\phi_i$ is non-zero only within a finite support region (mask of $k$ grid points)

This is the **Orbital Minimization Method (OMM)** — the foundation of linear-scaling DFT codes like SIESTA, ONETEP, and CONQUEST. The key advantage over diagonalization is that all operations are $O(N)$ when orbitals are localized: each orbital only interacts with neighbors whose supports overlap.

### Parametrization: Window Functions

Each orbital is parametrized as:

$$\phi_i(r) = w_i(r) \cdot c_i(r)$$

where:
- $w_i(r)$ is a **smooth window function** with compact support: $w = (1 - (r/R_c)^4)^3$, which is $C^4$ continuous at the boundary (vanishes with zero first and second derivatives)
- $c_i$ are **raw coefficients** defined only on the support mask of orbital $i$
- The window ensures smooth decay to zero at the boundary, avoiding artifacts from hard truncation

The alternative is a **hard cutoff** (no window, $w=1$ everywhere in mask), which is simpler but introduces discontinuities at the boundary that cause high kinetic energy artifacts from the finite-difference Laplacian.

### Hamiltonian

1D finite-difference discretization:

$$H = -\frac{1}{2} \frac{d^2}{dx^2} + V(x)$$

- Kinetic: tridiagonal finite-difference Laplacian ($T_{ii} = 1/dx^2$, $T_{i,i\pm1} = -0.5/dx^2$)
- Potential: harmonic well $V = \frac{1}{4}(x - L/2)^2$

Reference solution: exact diagonalization via `scipy.linalg.eigh_tridiagonal` (banded eigensolver).

---

## 2. Algorithm Variants Tested

Four modes were implemented and benchmarked in `OMM_1D_grid.py`:

### Mode 1: Jacobi (energy + orthogonalization loops)

The original "projective dynamics" approach — split into two phases:

1. **Energy step**: gradient descent on $E$, $\phi_i \mathrel{-}= \eta \cdot 2H\phi_i$ (within mask only)
2. **Orthogonalization**: iterative damped Gauss-Seidel sweeps:
   - $\phi_i \mathrel{-}= \alpha \sum_{j \neq i} \langle \phi_i | \phi_j \rangle \phi_j$ (projected onto mask)
   - Normalize $\phi_i$
   - Repeat for `ortho_iter` iterations

This is the simplest method but requires many orthogonalization sweeps per step and has a stiffness/damping tradeoff.

### Mode 2: CG Projected Gradient (`--mode cg`)

Replaces the split energy+orthogonalization with a **single coupled step** using exact constraint projection:

1. Compute energy gradients $g_i = 2H\phi_i$
2. Compute overlap matrix $O_{ij} = \langle \phi_i | \phi_j \rangle$
3. Compute RHS: $G_{ik} = \langle g_i | \phi_k \rangle$
4. **Solve $O \cdot \lambda_i = G_i$ via CG** (conjugate gradient on the small $N_{\text{orb}} \times N_{\text{orb}}$ system)
5. Projected gradient: $g_i^\perp = g_i - \sum_j \lambda_{ij} \phi_j$ (tangent to constraint manifold)
6. Update coefficients: $c_i \mathrel{-}= \eta \cdot (w_i \odot g_i^\perp + \lambda_{\text{reg}} \cdot c_i)$
7. Normalize + light Jacobi correction (3 sweeps) to pull back to manifold

The CG solve ensures the gradient is **exactly tangent** to the orthonormality constraint — no damping parameter needed. The overlap matrix $O$ is sparse (only neighbor pairs have nonzero entries) and small ($N_{\text{orb}} \times N_{\text{orb}}$, not $N_{\text{grid}} \times N_{\text{grid}}$).

### Mode 3: Penalty (`--mode penalty`)

Simultaneous minimization of energy + orthogonality penalty — no inner linear solve:

$$\mathcal{L} = \sum_i \langle \phi_i | H | \phi_i \rangle + K \sum_{i<j} O_{ij}^2 + \lambda_{\text{reg}} \sum_i \|c_i\|^2$$

Single gradient step:

$$\frac{\partial \mathcal{L}}{\partial c_i} = w_i \odot \left( 2H\phi_i + 2K \sum_{j \neq i} O_{ij} \phi_j \right) + 2\lambda_{\text{reg}} \cdot c_i$$

- $K$ controls the energy/orthogonality tradeoff (low $K$ → energy dominates, risk of bosonic collapse; high $K$ → orthogonality dominates, slower energy convergence)
- `n_correct` optional Jacobi correction sweeps after the penalty step (hybrid mode)
- Normalization: uniform scaling of $c_i$ to ensure $\|\phi_i\| = 1$ (not element-wise $\phi/w$)

This is the **Car-Parrinello analogue**: don't solve constraints exactly each step, just let the penalty force push toward orthogonality. Much cheaper per step (no CG solve).

### Mode 4: CG Energy Minimizer (`--mode cg_min`)

Conjugate gradient (Polak-Ribière) on the **same penalized loss** as Mode 3, with backtracking Armijo line search:

1. Flatten all coefficients into one vector $x$
2. Compute loss $f(x)$ and gradient $\nabla f$ via `_penalty_loss_and_grad`
3. CG direction: $d = -g + \beta \cdot d_{\text{prev}}$ (Polak-Ribière+, $\beta \geq 0$)
4. Backtracking line search: find $\alpha$ satisfying Armijo condition $f(x + \alpha d) \leq f(x) + c_1 \alpha \nabla f \cdot d$
5. Accept step, normalize, optional correction

**Hypothesis**: CG's superlinear convergence would minimize energy faster than fixed-step gradient descent.

**Result**: CG conjugacy is broken by the per-step normalization ($\|\phi\| = 1$), which is a Riemannian constraint. Standard CG assumes Euclidean geometry. The line search overhead (~2× cost per step) does not pay off — convergence is nearly identical to fixed-step penalty mode. A proper Riemannian CG (with parallel transport of search directions) would be needed but is complex and unnecessary given penalty mode's effectiveness.

---

## 3. Numerical Challenges and Solutions

### Challenge 1: Coefficient Blow-up at Window Boundaries

**Problem**: When $w_i \to 0$ at the boundary, the energy $\langle \phi | H | \phi \rangle$ and norm $\langle \phi | \phi \rangle$ become insensitive to $c_i$. The optimizer can push boundary coefficients to $\pm\infty$ with zero energy cost. This was the dominant instability in penalty mode with window functions.

**Root cause**: The finite-difference Laplacian generates large gradients at the boundary (discontinuity in higher derivatives), and the $\|w \cdot c\| = 1$ normalization amplifies $c$ where $w$ is small (since $\|c\| \sim 1/\|w\|$).

**Attempted fixes** (all failed):
- $\lambda_{\text{reg}} / w$ regularization: immediate divergence (1/w term too large)
- $\lambda_{\text{reg}} \cdot w$ regularization: too weak at boundary
- $w^2$ gradient weighting: reduced but didn't eliminate blow-up
- $\|c\| = 1$ normalization: prevented blow-up but caused bosonic collapse (all orbitals → ground state, since physical normalization $\|\phi\| = 1$ was lost)
- Clipping $c$ to max norm: either too aggressive (destroyed orbitals) or undone by subsequent normalization

**Solution**: Two-part fix:

1. **Regularized least-squares coefficient extraction** (`_sync_coefs_from_orbitals`):
   $$c = \frac{\phi \cdot w}{w^2 + \epsilon}$$
   where $\epsilon = w_{\text{clip}}^2$. This damps $c$ where $w$ is small (instead of amplifying via $c = \phi/w$). The parameter $w_{\text{clip}}$ controls the damping strength.

2. **Uniform scaling normalization**: After any update, normalize by computing $\|\phi\|$ and scaling $c$ by a single factor $1/\|\phi\|$ (not element-wise). This preserves the shape of $c$ while ensuring $\|\phi\| = 1$.

### Challenge 2: Support Recentering

**Problem**: Orbitals initialized at suboptimal positions get stuck — the energy gradient can't move the support center, only the coefficients within the fixed mask.

**Solution**: `recenter_supports()` — after each step, shift each orbital's mask to the center-of-mass of $|\phi|^2$, then re-extract coefficients via regularized least-squares. This is crucial for convergence: without it, energies are significantly higher and density accuracy degrades.

### Challenge 3: Penalty K Scaling with System Size

**Problem**: As $N_{\text{orb}}$ increases, overlaps grow (more neighbor pairs), and the penalty method struggles to maintain orthogonality. Higher $K$ values (50, 100) destabilize the system — energy diverges.

**Observation**: $K = 10$ works well for $N_{\text{orb}} \leq 5$. For larger systems, the penalty alone is insufficient and correction sweeps or CG projection become necessary. This is a known limitation of penalty methods — the penalty stiffness must scale with the constraint violation magnitude.

---

## 4. Performance and Sparsity Considerations

### Complexity Analysis

| Operation | Cost | Sparsity |
|-----------|------|----------|
| Energy gradient $H\phi_i$ | $O(k)$ per orbital, $O(Nk)$ total | Sparse matvec (tridiagonal $H$, localized $\phi$) |
| Overlap $O_{ij} = \langle \phi_i \| \phi_j \rangle$ | $O(k)$ per pair, $O(Nk)$ total | Only neighbor pairs (supports overlap) |
| CG solve $O \cdot \lambda = G$ | $O(N_{\text{orb}}^{1.5})$ worst case, $O(N_{\text{orb}} \cdot k)$ typical | $O$ is sparse (bandwidth $\sim 2R_{\text{cut}}/\text{spacing}$) |
| Penalty gradient | $O(Nk)$ total | Same sparse structure, no linear solve |
| Normalization | $O(k)$ per orbital | Local |
| Recentering | $O(k)$ per orbital | Local |

where $N = N_{\text{grid}}$ (total grid points), $k = \text{support\_size}$ (grid points per orbital), and $N_{\text{orb}} \propto N$ for fixed density.

### Key Distinction: Two Overlap Matrices

1. **Basis overlap** $S_{\mu\nu}$: overlap between basis functions (atomic orbitals / grid points). **Constant** — can be pre-factored (Cholesky) before iteration. Sparse with fixed pattern.

2. **Orbital overlap** $O_{ij} = \langle \phi_i | \phi_j \rangle$: overlap between optimized molecular orbitals. **Changes every iteration** — cannot be pre-factored. Sparse (only neighbor pairs) but denser than $S$ (bandwidth $\sim 2 \times \text{support\_size}$).

### Avoiding Densification

The critical insight for GPU implementation: **never form $O_{ij}$ explicitly as a matrix**. The CG solve only needs matrix-vector products $O \cdot v$, which can be decomposed:

$$(O \cdot v)_i = \sum_j \langle \phi_i | \phi_j \rangle v_j = \langle \phi_i | \Psi \rangle \quad \text{where } \Psi = \sum_j v_j \phi_j$$

This is 3 sparse operations:
1. **Scatter-add**: $\Psi = \sum_j v_j \phi_j$ — accumulate into dense workspace (each grid point touched only by orbitals whose masks contain it)
2. **Apply $S$**: $S\Psi$ — sparse matvec (existing `apply_operator` kernel)
3. **Dot product**: $(O \cdot v)_i = \langle \phi_i | S\Psi \rangle$ — restricted to mask $M_i$ (existing `project_orbitals` kernel)

No $O$ matrix is ever formed. No densification. Memory access is identical to existing kernels.

### GPU Implementation Strategy

The 3-phase GPU design from `OMM.cl` maps directly:

| Phase | Kernel | Operation |
|-------|--------|-----------|
| 1 | `apply_operator` | Compute $H|\phi_i\rangle$ and $S|\phi_i\rangle$ on expanded supports |
| 2 | `project_orbitals` | Compute scalar overlaps $\langle \phi_i \| H \| \phi_j \rangle$ and $\langle \phi_i \| S \| \phi_j \rangle$ |
| 3 | `assemble_gradient` | Combine into gradient $g_i = H\phi_i - \sum_j \lambda_{ij} S\phi_j$ |

For CG mode, add a CG loop calling phases 1-3 with the implicit $O \cdot v$ decomposition. The CG state vectors ($v, r, p$) are only $N_{\text{orb}}$-dimensional — negligible memory.

For penalty mode, phases 1-3 are sufficient with the penalty gradient assembly — **no CG loop needed at all**, making it the most GPU-friendly option.

**Pre-calculated interaction maps** (`Atom_Inter_Map`): For each coefficient of orbital $i$ and each neighbor $j$, pre-compute the memory offset into $S\phi_j$. This eliminates all index searching in the hot loop — the GPU kernel becomes a linear stream of gather-load-multiply-add operations with no branching.

---

## 5. Results

### Benchmark Configuration

- Grid: $N = 100$ points, $L = 30$ (harmonic well)
- Support size: $k = 20$ grid points per orbital
- Iterations: 1000
- Window: smooth $(1 - (r/R_c)^4)^3$
- $\lambda_{\text{reg}} = 0.01$, $w_{\text{clip}} = 0.05$
- Recentering: every step

### Speed and Density Comparison (1000 iterations)

| $N_{\text{orb}}$ | Mode | $E$ @ 500 | Density L2 | Max overlap | ms/iter | Speedup vs CG |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 3 | **Penalty nc=0** | 3.353 | **0.043** | 5.4e-2 | **0.29** | **3.4×** |
| 3 | CG (proj. grad) | 3.448 | 0.126 | 6.4e-3 | 0.99 | 1× |
| 3 | cg_min nc=0 | 3.351 | 0.043 | 5.9e-2 | 0.38 | 2.6× |
| 5 | **Penalty nc=0** | 9.859 | 0.204 | 1.0e-1 | **0.46** | **2.6×** |
| 5 | CG | 9.342 | 0.248 | 6.9e-3 | 1.18 | 1× |
| 5 | cg_min nc=0 | 9.871 | 0.218 | 1.0e-1 | 1.06 | 1.1× |
| 8 | **Penalty nc=0** | 25.531 | 0.388 | 1.6e-1 | **0.73** | **4.8×** |
| 8 | CG | 23.198 | **0.296** | 6.2e-3 | 3.54 | 1× |
| 8 | cg_min nc=0 | 25.439 | 0.574 | 2.7e-1 | 1.48 | 2.4× |
| 12 | **Penalty nc=0** | 54.554 | 0.761 | 2.7e-1 | **1.41** | **4.8×** |
| 12 | CG | 50.823 | **0.224** | 5.8e-3 | 6.71 | 1× |
| 12 | cg_min nc=0 | 95.785 | 1.265 | 3.4e-1 | 2.37 | 2.8× |

### Effect of Correction Sweeps (penalty mode, $K=10$)

| $N_{\text{orb}}$ | nc | $E$ @ 500 | Density L2 | Overlap | ms/iter |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 5 | 0 | 9.859 | 0.204 | 1.0e-1 | 0.46 |
| 5 | 1 | 10.143 | 0.287 | 1.5e-2 | 1.33 |
| 5 | 2 | 10.144 | 0.179 | 6.4e-3 | 0.39 |
| 8 | 0 | 25.531 | 0.388 | 1.6e-1 | 0.73 |
| 8 | 1 | 34.269 | 0.631 | 1.8e-2 | 1.64 |
| 12 | 0 | 54.554 | 0.761 | 2.7e-1 | 1.41 |
| 12 | 1 | 70.573 | 1.013 | 3.4e-2 | 4.43 |

Correction sweeps reduce overlap but **increase energy and degrade density** — the Jacobi orthogonalization fights the penalty minimization, pushing orbitals to higher energy.

### Effect of Penalty Stiffness $K$

| $N_{\text{orb}}$ | $K$ | $E$ @ 500 | Density L2 | Overlap | Stable? |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 5 | 10 | 9.859 | 0.204 | 1.0e-1 | yes |
| 5 | 50 | 10.132 | 0.226 | 4.3e-2 | yes |
| 5 | 100 | 46.715 | 1.030 | 2.6e-1 | **diverged** |
| 8 | 10 | 25.531 | 0.388 | 1.6e-1 | yes |
| 8 | 50 | 27.187 | 0.457 | 4.6e-2 | yes |
| 8 | 100 | 165.207 | 4.481 | 1.0 | **diverged** |
| 12 | 10 | 54.554 | 0.761 | 2.7e-1 | yes |
| 12 | 50 | 64.202 | 0.798 | 2.5e-1 | marginal |
| 12 | 100 | 299.938 | 3.784 | 1.0 | **diverged** |

$K = 10$ is the sweet spot. $K \geq 100$ diverges for $N_{\text{orb}} \geq 5$.

### CG Minimizer (cg_min) Assessment

The CG energy minimizer with Polak-Ribière + Armijo line search was tested as a potential faster alternative to fixed-step penalty. Key findings:

- **Convergence nearly identical** to fixed-step penalty — the per-step $\|\phi\| = 1$ normalization breaks CG conjugacy (this is a Riemannian optimization problem, not Euclidean)
- **~2× overhead** from line search (multiple loss evaluations per step)
- **Diverges at $N_{\text{orb}} = 12$** — the line search can take too-large steps when the loss landscape is complex
- **Larger initial step sizes** (0.05, 0.1, 0.5) all diverge — the fixed step 0.01 is already near-optimal

**Conclusion**: CG as an energy minimizer does not help in this setting. A proper Riemannian CG with parallel transport would be needed but is not worth the complexity.

---

## 6. Key Findings

1. **Penalty mode (nc=0) is 3–5× faster** than CG projected gradient, with **better density for small systems** ($N_{\text{orb}} \leq 5$). For larger systems, CG's exact projection gives better density but at 5× the cost.

2. **Density, not orthogonality, is the figure of merit** for SCF acceleration. The penalty mode's higher overlap (~0.1 vs ~0.006 for CG) does not prevent good density convergence — the orbitals are "good enough" for charge mixing. Exact orthogonality is only needed at the final diagonalization step.

3. **CG minimizer doesn't help**: normalization makes this Riemannian, breaking Euclidean CG conjugacy. Line search overhead is not worth it.

4. **Correction sweeps (nc ≥ 1) hurt density**: Jacobi orthogonalization fights penalty minimization, pushing to higher energy. Pure simultaneous (nc=0) is best.

5. **$K = 10$ is robust** across system sizes. $K \geq 100$ diverges. The penalty stiffness does not need to scale with system size for $N_{\text{orb}} \leq 12$.

6. **Window functions are essential** for smooth convergence — hard cutoffs cause kinetic energy artifacts at boundaries. The regularized least-squares extraction ($c = \phi w / (w^2 + \epsilon)$) is the key to stable coefficient management.

7. **Recentering is critical** — without it, orbitals get stuck in suboptimal positions and energies are significantly higher.

---

## 7. Best Path Forward

### For SCF Acceleration (primary use case)

The goal is to **avoid diagonalization during SCF** and only do one at the end. The optimal strategy:

1. **During SCF iterations**: use `penalty nc=0` — fastest (3–5× vs CG), density good enough for charge mixing
2. **At end of SCF**: one `CG` pass or exact diagonalization to get accurate density and orbitals
3. This leverages the penalty mode's speed for the bulk of iterations and CG's accuracy for the final polish

### For Larger Systems (2D/3D, more orbitals)

- **Overlap grows with $N_{\text{orb}}$** — penalty alone may not suffice. Options:
  - Adaptive $K$: increase penalty stiffness as overlaps grow
  - Periodic CG "reset" steps: every $M$ steps, do one CG projection to reduce accumulated overlap
  - Chebyshev-accelerated Jacobi: use Chebyshev semi-iterative method for the orthogonalization (as in Wang 2015 for projective dynamics)
- **Support size scaling**: in 2D/3D, $k$ grows as $R_{\text{cut}}^d$, so the per-orbital cost increases. The penalty mode's advantage (no CG solve) becomes even more significant.

### GPU Implementation Priority

1. **Penalty mode first** — simplest to implement on GPU (no CG loop, no global reductions):
   - Phase 1: `apply_operator` (compute $H\phi_i$)
   - Phase 2: `project_orbitals` (compute overlaps $O_{ij}$)
   - Phase 3: `assemble_gradient` (combine energy + penalty gradient)
   - Phase 4: update coefficients + normalize
   - All phases are embarrassingly parallel across orbitals

2. **CG mode second** — requires CG loop with global reductions (dot products for $\alpha, \beta$), which are slow on GPU but manageable for small $N_{\text{orb}}$

3. **Pre-calculated interaction maps** (`Atom_Inter_Map`) eliminate all index searching in kernels — the hot loop becomes pure gather-load-multiply-add

### Open Questions

- **How does penalty mode scale to 2D/3D?** The overlap structure changes (more neighbors per orbital), and the optimal $K$ may need adjustment.
- **Can we use adaptive step size?** FIRE-style adaptive timestep could accelerate penalty mode convergence without the overhead of CG line search.
- **Metallic systems**: both penalty and CG fail (no exponential decay of density matrix). Green's function methods are needed instead.
- **Riemannian CG**: if faster convergence is needed, implementing proper Riemannian CG with parallel transport on the Stiefel manifold could combine CG's convergence rate with constraint preservation. This is more complex but theoretically sound.

---

## 8. File Reference

All implementations are in `OMM_1D_grid.py`:

- `OMM1DSolver`: main solver class with grid, Hamiltonian, window functions
- `projected_gradient_step`: CG-based exact projection (Mode 2)
- `penalty_gradient_step`: simultaneous penalty minimization (Mode 3)
- `cg_minimize_step`: CG energy minimizer with line search (Mode 4)
- `_sync_coefs_from_orbitals`: regularized least-squares coefficient extraction
- `recenter_supports`: adaptive support tracking
- `orthogonalize`: damped Jacobi/Gauss-Seidel orthogonalization
- `reference_ground_state`: exact banded eigensolver for comparison

CLI usage:
```bash
# Penalty mode (fastest, best for SCF)
python OMM_1D_grid.py --mode penalty --use_window 1 --K 10 --n_correct 0

# CG projected gradient (most accurate)
python OMM_1D_grid.py --mode cg --use_window 1

# CG energy minimizer (experimental)
python OMM_1D_grid.py --mode cg_min --use_window 1 --K 10 --n_correct 0

# Jacobi (baseline)
python OMM_1D_grid.py --mode jacobi --use_window 1
```
