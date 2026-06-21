# Spectral Filtering + Subspace Iteration with Pruning

## Overview

The method combines **Chebyshev polynomial band-pass filtering** with **Rayleigh-Ritz subspace iteration** to compute eigenvalues inside prescribed frequency bands.  Trial vectors are started in **redundant numbers** (2x the expected count) and pruned gradually using residual-based outlier detection.

---

## 1. Chebyshev Band-Pass Filter

### 1a. Scalar polynomial

For a target band $[f_{\text{lo}}, f_{\text{hi}}] \subset [-1,1]$ we construct a Chebyshev polynomial $p(x)$ of degree $d$ that approximates the **indicator (rectangle) function** of the band.

In the Chebyshev basis $T_m(x)$:

$$
p(x) = \sum_{m=0}^{d} c_m \, T_m(x)
$$

The coefficients are obtained from the exact Fourier-Chebyshev expansion of the step function:

$$
c_0 = \frac{\theta_{\text{lo}} - \theta_{\text{hi}}}{\pi}, \qquad
c_m = \frac{2}{\pi} \frac{\sin(m\theta_{\text{hi}}) - \sin(m\theta_{\text{lo}})}{m} \quad (m \ge 1)
$$

where $\theta = \arccos(x)$.  Because the step function is discontinuous, the raw Chebyshev series suffers from **Gibbs oscillations**.  We therefore apply **Jackson damping**:

$$
c_m \leftarrow g_m^{(d)} \, c_m, \qquad
g_m^{(d)} = \frac{(d+1-m)\cos\frac{\pi m}{d+1} + \sin\frac{\pi m}{d+1}\cot\frac{\pi}{d+1}}{d+1}
$$

Finally we rescale so that $p(x_{\text{mid}}) \approx 1$ inside the band.

### 1b. Matrix polynomial (the actual filter)

For a Hamiltonian $H$ with spectrum in $[-1,1]$ and a set of $k$ random trial vectors $V \in \mathbb{R}^{N \times k}$:

$$
V_{\text{filt}} = p(H)\,V = c_0 V + c_1 H V + \sum_{m=2}^{d} c_m \bigl(2 H T_{m-1} - T_{m-2}\bigr)
$$

This is evaluated by the three-term recurrence

$$
T_0 = V, \quad T_1 = H V, \quad T_m = 2 H T_{m-1} - T_{m-2}
$$

so the cost is **$d$ matrix-vector products** (SpMVs) for all $k$ vectors simultaneously.

### Why this works

If $H = \sum_i \lambda_i \, |\psi_i\rangle\langle\psi_i|$ is the spectral decomposition, then

$$
p(H)\,|\psi_i\rangle = p(\lambda_i)\,|\psi_i\rangle
$$

- Inside the band: $p(\lambda_i) \approx 1$  →  eigenvectors are **preserved**
- Outside the band: $|p(\lambda_i)| \ll 1$  →  eigenvectors are **suppressed**

As $d \to \infty$ (without Jackson damping) $p(x)$ converges to the exact rectangle; with Jackson damping the transition width is $\mathcal{O}(1/d)$ but oscillations are suppressed.

### Optional: squared filter

Applying the filter twice gives $p(H)^2$, which is nonnegative and has a steeper transition (the squared polynomial has half the transition width).  This is the `--square_filter` option.

---

## 2. KPM Chebyshev Point Filter (Spectral Filter Mode)

In addition to the band-pass rectangle filter, we implement a **point filter** that evaluates the Chebyshev spectral response at many individual frequencies across the whole spectrum.  This is the classical **Kernel Polynomial Method (KPM)** used for density-of-states estimation, adapted here to visualize how the filter converges as iteration count grows.

### 2a. Algorithm

For each target frequency $f \in [-1, 1]$ we form the Jackson-damped Chebyshev series of the **point (delta) filter**:

$$
V_f = \sum_{m=0}^{n} g_m^{(n)} \, 2\cos(m\theta_f) \; T_m(H)\,V_0
$$

where $\theta_f = \arccos(f)$ and $g_m^{(n)}$ is the Jackson kernel of degree $n$.  The key optimization is that the basis $T_m(H)V_0$ is built **once** (frequency-independent) and then combined with cheap scalar weights for each $f$:

```
1. Build basis:  B_m = T_m(H) V_0   for m = 0 ... n_max
2. For each frequency f:
       V_f = sum_m weight_m(f) * B_m
       amplitude[f, j] = ||V_f[:, j]||   (per-vector norm)
```

Cost: $n_{\max}$ SpMVs to build the basis (shared across all frequencies), then $\mathcal{O}(n_{\max} \cdot N \cdot k \cdot n_{\text{freq}})$ cheap scalar operations for the frequency sweep.

### 2b. What the plot shows

- **Top panel:** total amplitude $\sum_j \|V_f^{(:,j)}\|$ vs frequency $f$ for several iteration counts $n = 4, 8, 16, 32, 64, 128$
- **Bottom strips:** per-vector amplitudes as a 2D image (frequency $\times$ vector index)

As $n$ increases, sharp **peaks emerge at the true eigenvalues** — the Chebyshev filter acts as a tunable spectrometer.

### 2c. Jackson damping and Gibbs ripples

Without Jackson damping, the point filter converges to a Dirac comb but exhibits strong **Gibbs oscillations** — spurious peaks and sign flips between eigenvalues.  With Jackson damping the peaks are slightly broadened (width $\sim 1/n$) but the baseline is clean.  The comparison is instructive:

| | With Jackson | Without Jackson |
|---|---|---|
| Peak width | $\mathcal{O}(1/n)$ | narrower |
| Baseline | flat | Gibbs ripples ($\sim 10\%$) |
| Use case | reliable DOS / spectrum | educational demo |

Run `--spectral_no_jackson 1` to see the ripples.

### 2d. Power iteration alternative

The `--spectral_method power` mode replaces the Chebyshev construction with shifted power iteration:

$$
V_f^{(n)} = (f I - H)^n V_0
$$

and plots $1 / \|V_f^{(n)}\|$ so that resonances (near-eigenvalues) appear as **peaks** rather than notches.  This is slower per frequency ($n_{\text{freq}}$ separate matrix powers) but illustrates the same spectral convergence principle.

---

## 3. Ritz Subspace Iteration

### 3a. QR + Rayleigh-Ritz

After filtering, the columns of $V_{\text{filt}}$ span a subspace dominated by the target band's eigenvectors.  We orthonormalize:

$$
Q = \text{QR}(V_{\text{filt}})
$$

and perform the **Rayleigh-Ritz** projection:

$$
T = Q^T H Q \quad (\text{small } k \times k \text{ matrix})
$$

Solving the small eigenproblem

$$
T \, z_j = \omega_j \, z_j
$$

gives Ritz values $\omega_j$ and Ritz vectors $u_j = Q z_j$.  The Ritz residuals are

$$
r_j = \|H u_j - \omega_j u_j\|
$$

which measure how well each Ritz pair approximates a true eigenpair of $H$.

### 3b. Repeated iteration with spectral filtering

The Ritz vectors for eigenvalues inside the band are used as the new trial vectors for the **next filter step**:

$$
V^{(n+1)} = p(H) \, U^{(n)}_{\text{in-band}}
$$

This is **subspace iteration** (also called "orthogonal iteration" or "simultaneous iteration") with the filter polynomial $p(H)$ playing the role of the power-method operator.  Each iteration:

1. $p(H)$ amplifies the in-band components
2. Rayleigh-Ritz extracts approximate eigenpairs
3. Keeping only in-band Ritz vectors refocuses the subspace

Convergence rate: for a Ritz value $\omega_j^{(n)}$ approaching true eigenvalue $\lambda_j$, the error decays as

$$
|\omega_j^{(n)} - \lambda_j| \sim \mathcal{O}\left(\left|\frac{p(\lambda_{\text{nearest outside}})}{p(\lambda_j)}\right|^n\right)
$$

Because $p(\lambda_j) \approx 1$ inside the band and $|p| \ll 1$ outside, convergence is fast provided the filter degree is large enough.

### Exact SpMV count per subspace iteration

Let $k_n$ be the current subspace dimension at iteration $n$ (starts at $2 \times$ expected count, shrinks due to pruning).  Let $d$ be the Chebyshev degree (`coarse_iters`) and $r$ the number of filter applications (`filter_reps`).  The `--square_filter` flag doubles the cost per application because $p(H)^2$ requires two polynomial evaluations.

**Step 1 — Filtering:**

For each of the $r$ applications:

$$
\text{SpMVs}_{\text{filter}} = r \times d \times (1 + \mathbb{1}_{\text{square}}) \times k_n
$$

- `apply_cheb_poly(H, V, c)` costs exactly $d$ multiplications `H @ M` where $M$ is $N \times k_n$
- If `square_filter=True`, we call it twice per application (total $2d$ per application)
- In vector-equivalent terms: each `H @ M` with $k_n$ columns counts as $k_n$ SpMVs

**Step 2 — QR:**  zero SpMVs (pure linear algebra on $N \times k_n$)

**Step 3 — Rayleigh-Ritz:**

```python
T = Q.T @ (H @ Q)
```

`H @ Q` is one $N \times k_n$ multiply:

$$
\text{SpMVs}_{\text{Ritz}} = k_n
$$

**Step 3 — Lanczos (alternative):**

Single-vector Lanczos builds a Krylov subspace of size $m = \max(2, k_n)$ with one `H @ q` per step, plus an extra `H @ U` for residual computation:

$$
\text{SpMVs}_{\text{Lanczos}} \approx m + m = 2m
$$

**Total per iteration:**

| Mode | SpMVs per iteration |
|------|---------------------|
| Ritz, no square, $r=1$ | $d \cdot k_n + k_n = (d+1) k_n$ |
| Ritz, square, $r=1$ | $2d \cdot k_n + k_n = (2d+1) k_n$ |
| Ritz, no square, $r=0$ | $k_n$ (pure refinement, no filtering) |
| Lanczos, no square, $r=1$ | $d \cdot k_n + 2m$ |

**Total for a full run:** summed over all bands and all iterations.  Because pruning shrinks $k_n$, the cost decreases as the subspace converges.

### Literature connection

This is essentially the **FEAST** algorithm (Polizzi 2009) without the contour integral — FEAST uses rational filters $(zI-H)^{-1}$ while we use a Chebyshev polynomial filter.  The Ritz iteration step is identical.  Other related methods: **Sakurai-Sugiura** (SS) contour integration, **FiltLan** (filtered Lanczos), and the **Kernel Polynomial Method** (KPM) for DOS estimation.

---

## 4. Pruning Strategies

We start with $2\times$ more trial vectors than expected eigenvalues per band.  The subspace must **shrink** to the true dimension.  Several strategies are available:

### 4a. Residual pruning (`--prune_mode residual`)

After a warm-up of 2 iterations, drop **all** Ritz vectors with residual above a tolerance:

$$
\text{keep}_j = \bigl(r_j < \tau_{\text{prune}}\bigr)
$$

**Problem:** if the tolerance is too tight, good but slowly-converging vectors are killed.

### 4b. Cluster pruning (`--prune_mode cluster`)

Two vectors chasing the **same eigenvalue** become nearly collinear after filtering.  We detect this on the **raw filtered vectors** (before QR orthonormalization, which would destroy the collinearity signal):

1. Normalize columns: $\tilde{v}_j = v_j / \|v_j\|$
2. Compute cheap Rayleigh quotients: $\hat{\lambda}_j = \tilde{v}_j^T H \tilde{v}_j$
3. Sort by $\hat{\lambda}_j$ and check **adjacent neighbors**
4. If both $|\hat{\lambda}_{j+1} - \hat{\lambda}_j| < \delta_\lambda$ **and** $|\tilde{v}_j^T \tilde{v}_{j+1}| > \tau_{\text{dot}}$:
   - Same mode → prune the one with **larger estimated residual**

**Advantage:** identifies the *root cause* (redundancy), not just a symptom (large residual).

### 4c. Gradual pruning (`--prune_mode gradual`) — **recommended**

Each iteration, drop **at most one** vector — the single worst outlier.  A vector is pruned only if **all three** conditions hold:

| # | Condition | Meaning |
|---|-----------|---------|
| 1 | $r_{\text{worst}} > \tau_{\text{prune}}$ | Has not converged yet |
| 2 | $r_{\text{worst}} > f \cdot \text{median}(\{r_j\})$ | Is a genuine outlier ($f=3$ by default) |
| 3 | $r_{\text{worst}} \ge 0.9 \, r_{\text{prev}}$ | Is **not improving** (residual stagnant or growing) |

Condition 3 is the key: a vector near a true eigenvalue may have large residual simply because it converges slowly, but its residual drops iteration-over-iteration.  Such vectors are **preserved**.  Only vectors whose residuals are flat or increasing are pruned.

### 4d. Hybrid (`--prune_mode hybrid`)

1. Cluster prune to remove redundant pairs
2. If the subspace is still too large, apply one gradual prune step

### Subspace shrinking

After pruning, the next iterate uses only the kept vectors:

$$
V^{(n+1)} = U^{(n)}_{\text{keep}} \quad \text{(dimension strictly decreases or stays)}
$$

If **all** vectors would be pruned, we keep the single best one (smallest residual) rather than restarting with full random vectors.  The subspace **never regrows**.

---

## 5. Cheap Rank Estimate

After the **first** filter application, we can estimate the number of eigenvalues in the band via an SVD of $V_{\text{filt}}$:

$$
V_{\text{filt}} = U \Sigma W^T \quad\Rightarrow\quad
\text{rank}_{\text{est}} = \#\{\sigma_i / \sigma_0 > 0.1\}
$$

Because $p(H)$ projects onto the band's eigenspace, the number of significant singular values approximates the number of eigenvalues in $[f_{\text{lo}}, f_{\text{hi}}]$.  Cost: $\mathcal{O}(N k^2)$ with $k \ll N$ — negligible.

---

## 6. Summary of Algorithm

```
for each band [f_lo, f_hi]:
    compute Chebyshev coefficients c for rectangle indicator
    V = random(N, 2*nvec)          # redundant trial vectors
    for iteration = 1 ... n_iter:
        V_filt = p(H) * V          # Chebyshev band-pass filter
        if square_filter: V_filt = p(H) * V_filt
        Q, R = qr(V_filt)
        w, U, r = rayleigh_ritz(H, Q)   # small eigenproblem
        select vectors inside band
        apply pruning strategy (gradual recommended)
        V = U_kept                   # shrink subspace
    end
end
```

---

## 7. Notes and Alternatives

### Filter choices
- **Rational filter** (FEAST): $(zI-H)^{-1}$ gives exact rectangle but requires linear solves (expensive)
- **Chebyshev filter** (this work): matrix-vector products only, cheap but approximate rectangle
- **Jackson damping**: trade-off — wider transition but no Gibbs ringing; without it the filter has $\sim 10\%$ overshoot at band edges

### Subspace extraction
- **Rayleigh-Ritz** (used here): stable, works well for dense spectra
- **Lanczos**: cheaper per iteration (no small dense eigenproblem) but less robust for multiple eigenvalues; we implemented single-vector Lanczos as an alternative (`--method lanczos`)
- **Harmonic Ritz**: better for interior eigenvalues near the band edges

### Pruning alternatives from literature
- **Locking & Purging** (ARPACK, PRIMME): converged vectors are deflated from the active subspace
- **Restarting with implicit filtering**: discard the subspace and restart with Ritz vectors only (no gradual pruning)
- **Rank-revealing QR**: directly estimate the subspace dimension and truncate
- **Krylov-Schur** (Sorensen): uses a Schur form to enable arbitrary vector deletion without losing convergence information

### When this method excels
- **Dense spectra** with many eigenvalues: band partitioning isolates groups
- **Large matrices**: only matrix-vector products needed (SpMV), no factorization
- **Targeted eigenvalues**: you only compute eigenvalues in prescribed bands, not the full spectrum

### When other methods are better
- **Sparse spectra** with well-separated eigenvalues: Lanczos or shift-invert is faster
- **Need high accuracy** inside a narrow band: FEAST with rational filter converges faster per iteration
- **Very large $N$**: the $\mathcal{O}(N k^2)$ QR/Ritz step may dominate; block Lanczos or LOBPCG are better

---

## 8. Tutorial: Running the Demos

All demos use the same script `chebyshev_prefiltered_jacobi_ocl.py`.  The script is modular: all linear algebra lives in `spectral_solvers.py` and all plotting in `spectral_plotting.py`.

### 8a. Debug: Chebyshev rectangle convergence

Show how the scalar band-pass polynomial $p(x)$ converges to a rectangle as degree increases:

```bash
python chebyshev_prefiltered_jacobi_ocl.py --debug_rect --size 60 --nbands 6 --coarse_iters 40 --save rect.png
```

Output: a 2-panel plot with the polynomial and its square for each band.

### 8b. Spectral filter (KPM point filter)

Visualize how the Chebyshev point filter sharpens around eigenvalues as iteration count grows:

```bash
# With Jackson damping (smooth peaks)
python chebyshev_prefiltered_jacobi_ocl.py --spectral_filter --size 60 --nvec 8 --save spectral_cheb.png

# Without Jackson damping (observe Gibbs ripples)
python chebyshev_prefiltered_jacobi_ocl.py --spectral_filter --spectral_no_jackson 1 --size 60 --nvec 8 --save spectral_gibbs.png

# Power iteration instead of Chebyshev
python chebyshev_prefiltered_jacobi_ocl.py --spectral_filter --spectral_method power --size 60 --nvec 8 --save spectral_power.png
```

Output: top panel shows total amplitude vs frequency for iterations 4, 8, 16, 32, 64, 128; bottom panels show per-vector amplitude strips.

### 8c. Converge spectrum (band-pass filter + subspace iteration)

Compute approximate eigenvalues in all bands with pruning and plot convergence:

```bash
# Ritz + gradual pruning (recommended)
python chebyshev_prefiltered_jacobi_ocl.py --converge_spectrum --size 60 --nvec 20 --nbands 6 \
    --coarse_iters 40 --conv_iters 8 --square_filter --filter_reps 1 \
    --prune_mode gradual --prune_tol 1e-3 --gradual_factor 3.0 \
    --method ritz --save conv.png

# Lanczos variant (cheaper per iteration)
python chebyshev_prefiltered_jacobi_ocl.py --converge_spectrum --size 60 --nvec 20 --nbands 6 \
    --coarse_iters 40 --conv_iters 8 --method lanczos --save conv_lanczos.png

# No pruning baseline (all vectors kept)
python chebyshev_prefiltered_jacobi_ocl.py --converge_spectrum --size 60 --nvec 20 --nbands 6 \
    --coarse_iters 40 --conv_iters 8 --prune_mode none --method ritz --save no_prune.png

# Aggressive residual pruning
python chebyshev_prefiltered_jacobi_ocl.py --converge_spectrum --size 60 --nvec 20 --nbands 6 \
    --coarse_iters 40 --conv_iters 8 --prune_mode residual --prune_tol 1e-3 --method ritz --save residual_prune.png

# Hybrid pruning (cluster + gradual)
python chebyshev_prefiltered_jacobi_ocl.py --converge_spectrum --size 60 --nvec 20 --nbands 6 \
    --coarse_iters 40 --conv_iters 8 --prune_mode hybrid --cluster_tol 0.85 --prune_tol 1e-3 --method ritz --save hybrid_prune.png
```

Output: a 3-panel plot (eigenvalue convergence, subspace dimension, residual history) and a terminal summary of SpMV costs and converged eigenvalues.

### 8d. Solve single band

Extract eigenpairs in one specific band without convergence plot:

```bash
python chebyshev_prefiltered_jacobi_ocl.py --solve_band 2 --size 60 --nvec 10 \
    --coarse_iters 40 --solve_iters 4 --square_filter
```

Output: Ritz eigenvalues and residuals for band index 2, plus exact eigenvalues for comparison (if $N \le 200$).

### 8e. Parameter quick-reference

| Parameter | Default | Meaning |
|---|---|---|
| `--size` | 20 | Matrix dimension $N$ |
| `--nvec` | 8 | Random probe vectors per band |
| `--coarse_iters` | 32 | Chebyshev degree $d$ (filter sharpness) |
| `--conv_iters` | 6 | Subspace iteration count |
| `--square_filter` | False | Apply $p(H)^2$ for steeper transition |
| `--filter_reps` | 1 | Filter applications per iteration |
| `--prune_mode` | none | none / residual / cluster / gradual / hybrid |
| `--method` | ritz | ritz (subspace iteration) or lanczos |
| `--res_tol` | 1e-6 | Residual tolerance for "converged" report |
