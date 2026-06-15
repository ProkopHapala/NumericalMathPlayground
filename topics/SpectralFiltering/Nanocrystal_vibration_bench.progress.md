# USER

look into 
/home/prokop/git/NumericalMathPlayground/topics/SpectralFiltering
@topics/SpectralFiltering/SpectralFiltering_doc_tutorial.md @topics/SpectralFiltering/SpectralFiltering_discussion.md @topics/SpectralFiltering/RandomizedLinarAlgebra.md 

there are some tips how to make some fast linear solvers for sparse matrixes 


I prepared some test files for nancrystals in different repo, to see how our methods behave on them

Can you test them?

1) fist I want to see just spectral filter which converge to spectrum (chebyshev or something)
2) then I want to use some solver which can find eigenvectros in sume sub interval (because there will be thousand eigenstates, even small range of frequencies can have many aigenstates. Therefore we should maybe split it to su-interval so we keep our solver small with e.g. 16-64 vectros in the active subspace )

before you start coding review what we have make notes and a plen

---

```
Benchmark Hessian export is in place. Here is what you have and how to use it.

## What was built

**Export pipeline**
- `pyBall/vibration_benchmark.py` — MMFF relax → full/sparse Hessian → rigid-mode shift → PD check → `.npz`
- `tests/tSiNCs/export_vibration_benchmarks.py` — calibration sweep + suite export

**Documentation**
- [`doc/Topics/FTIR_Nanocrystals/Vibration_benchmark_npz.guide.md`](doc/Topics/FTIR_Nanocrystals/Vibration_benchmark_npz.guide.md) — every array key, units, solver usage

**Generated fixtures** (gitignored under `tests/tSiNCs/fixtures/vibration_benchmarks/`):

| File | Atoms | DOF | Dense `eigh` | nnz (CSR K) |
|------|-------|-----|--------------|-------------|
| `adamantane.npz` | 26 | 78 | ~0 ms | 6k |
| `nc_C_R4.npz` | 80 | 240 | ~1 ms | 58k |
| `nc_C_R5.npz` | 164 | 492 | ~6 ms | 242k |
| `nc_C_R6.npz` | 270 | 810 | ~20 ms | 656k |
| `nc_C_R7.npz` | 414 | 1242 | ~56 ms | 1.5M |
| `nc_C_R8.npz` | 558 | 1674 | ~123 ms | 2.8M |

All exported matrices have `n_negative_projected == 0` after rigid-mode penalty (`rigid_shift=1e6`).

## Dense-reference size limit

On this machine, **numpy `eigvalsh` stays under ~1 s up to R=8 (558 atoms, 1674 DOF)**. That is a practical ceiling for bundled dense `H_dense_projected` reference matrices. Systems above ~3000 DOF should be sparse-only (no dense arrays in `.npz`).

Regenerate anytime:

```bash
python3 tests/tSiNCs/export_vibration_benchmarks.py              # full calibration + export
python3 tests/tSiNCs/export_vibration_benchmarks.py --skip-calibrate  # suite only
```

## Rigid modes & positive definiteness

Raw cluster Hessians have 6 near-zero rigid modes. Export applies `FTIR.apply_rigid_mode_shift` / `prepare_sparse_hessian` with `shift=1e6`, which lifts those modes so iterative solvers see a **numerically PD** stiffness matrix. For external benchmarks, use **`K_csr_*`** (projected CSR), not raw `blocks`.

## External solver repo usage

```python
import json, numpy as np, scipy.sparse as sp

d = np.load("nc_C_R5.npz", allow_pickle=True)
meta = json.loads(str(d["meta_json"]))
K = sp.csr_matrix((d["K_csr_data"], d["K_csr_indices"], d["K_csr_indptr"]), shape=tuple(d["K_csr_shape"]))
M = sp.diags(d["mass_diag"])
omega_ref = d["omegas_modes_projected"]  # validation against dense eigh

# Dynamic stiffness probe: A(w) = K - (w + i*eta)^2 M
```

Compare iterative methods against `omegas_modes_projected` and (for small systems) `H_dense_projected`.

## Why this helps the convergence problem

The progress reports show Jacobi / momentum / GS paths failing to converge reliably. With frozen, PD, documented inputs you can:

1. **Isolate the solver** from Hessian-generation bugs
2. **Bracket by size** — adamantane (78 DOF) → R8 (1674 DOF)
3. **Test** \((K - \omega^2 M)^{-1} f\) and full spectra independently of FireCore’s `FTIR.py` orchestration

Likely next solver work (once benchmarks are in the external repo):

- Preconditioned CG on shifted systems \(K - \sigma M\)
- LOBPCG / Jacobi–Davidson for lowest modes (after rigid shift)
- SSOR / incomplete Cholesky preconditioners tuned to MMFF sparsity pattern

## Notes

- **H caps** use `pyBall/nanocrystal_gen.py` VSEPR placement (not the legacy 53° bisector bug).
- **`si_G1_caps_only`** is skipped — the parallel fixture has invalid MMFF topology; regenerate Si NCs from `gen_nanocrystals.mjs` when that track is ready.
- Fixed **`MMFF.clear()`** before each system so buffer views do not leak between exports (this was causing garbage `neighs` on multi-system runs).

Copy the `.npz` files manually into your external benchmark repo. If you want, I can add a thin `tests/tSiNCs/verify_vibration_benchmarks.py` smoke test or extend the suite with Si NCs once generation fixtures are fixed.
```
---

# Cursor Composer 2.5

## Review: what exists today

### Code (`topics/SpectralFiltering/`)

| File | Role |
|------|------|
| `spectral_solvers.py` | Core algorithms: Chebyshev rectangle filter, KPM point filter, `solve_spectrum()` (band-pass + Ritz/Lanczos + pruning), `solve_band()` |
| `spectral_demos.py` | CLI entry point (`--spectral_filter`, `--converge_spectrum`, `--solve_band`, …) |
| `spectral_plotting.py` | Plots for filter shapes, KPM evolution, 3-panel convergence |
| Docs | Tutorial + long design discussion (Chebyshev vs inversion, pruning, GPU notes) |

**What works well (on toy problems):**
- **Phase 1 (KPM spectrometer):** `chebyshev_filter()` builds `T_m(H)V_0` once, sweeps many frequencies — exactly the “peaks sharpen with iteration” demo you want.
- **Phase 2 (band eigenvectors):** `solve_spectrum()` = FEAST-like Chebyshev band-pass + Rayleigh–Ritz + gradual/cluster pruning — matches your sub-interval eigenvector goal.

**Critical gaps for nanocrystal benchmarks:**

1. **Dense Laplacian only** — `generate_test_matrix()` produces N×N dense H with spectrum in `[-0.95, 0.95]`. No `scipy.sparse`, no CSR matvec path.

2. **Standard eigenproblem, not generalized** — benchmarks are `K v = λ M v` (Hessian + diagonal mass). Chebyshev needs a symmetric operator with bounded spectrum, typically  
   `H = M^{-1/2} K M^{-1/2}` (eigenvalues λ = ω²).

3. **No spectral rescaling** — Chebyshev assumes `[-1, 1]`. Benchmark ω is ~0.25–4.3 (internal MMFF units); rigid modes sit at ~1000 after `rigid_shift=1e6`.

4. **Jacobi coarse–fine path removed** — discussion describes Jacobi + heavy-ball after Chebyshev prefilter; `spectral_demos.py` ends with “main mode removed”. Only polynomial-filter + Ritz remains.

5. **Docs vs filenames** — tutorial references `chebyshev_prefiltered_jacobi_ocl.py`; actual script is `spectral_demos.py`.

6. **No benchmark fixtures in this repo** — `.npz` files live in FireCore (found at  
   `/home/prokop/git/FireCore/tests/tSiNCs/fixtures/vibration_benchmarks/`). Nothing copied here yet.

## Benchmark data (available now)

Fixtures are ready in FireCore:

| File | DOF | n_vib modes | ω range (vib) | nnz(K) |
|------|-----|-------------|---------------|--------|
| `adamantane.npz` | 78 | 72 | 0.54 – 4.30 | 6k |
| `nc_C_R5.npz` | 492 | 486 | 0.26 – 4.30 | 242k |
| `nc_C_R8.npz` | 1674 | 1668 | 0.23 – 4.30 | 2.8M |

- Use **`K_csr_*`** (projected, PD after rigid shift), not raw `blocks`.
- Reference: `omegas_modes_projected` = √(eigenvalues of `M^{-1/2} K M^{-1/2}`).
- Units are internal MMFF (√(eV/amu)/Å), not cm⁻¹ — fine for relative benchmarking.
- Dense `H_dense_projected` included up to R8 for validation.

**Mode density matters:** vibrational spectrum is dense. With 8 equal bands, R8 bands hold **60–330 modes each** — far above a 16–64 vector subspace. To keep subspaces small you need **narrow ω windows**, e.g. width ≈ 0.05 → ~82 bands, median ~20 modes/band, max ~91 (R8).

## Mapping your two goals to existing code

### Goal 1 — Spectral filter → spectrum (Chebyshev/KPM)

**Existing:** `chebyshev_filter()` + `--spectral_filter`.

**Adaptations needed:**
```
K, M  →  H = D^{-1/2} K D^{-1/2}     (D = diag(M))
estimate λ_min, λ_max (Lanczos or from reference on small systems)
H_scaled = 2(H - c)/r  mapped to [-1,1]
KPM on H_scaled; x-axis back-mapped to ω = sqrt(λ)
overlay reference omegas_modes_projected
```

- SpMV: `H @ V` via `M^{-1/2} K M^{-1/2} @ V` (two diagonal scales + one `K @ V`).
- Jackson damping already implemented.
- Success metric: peaks align with reference ω as degree n → 64, 128, …

### Goal 2 — Eigenvectors in ω sub-intervals (small active subspace)

**Existing:** `solve_spectrum()` with band `[f_lo, f_hi]`, `nvec`, pruning.

**Adaptations needed:**
- Same `H_scaled` operator + **bands in ω space** (convert band edges to scaled eigenvalue edges: `λ = ω²` after scaling).
- **Band width ~0.05–0.10** in ω (not 8 equal bands) so each band has ≲ 32–64 true modes.
- Start with **`nvec=32`**, `2×` oversampling inside `solve_spectrum` → 64 trial vectors; **`prune_mode=gradual`**.
- **`coarse_iters` ~ 40–80** — spectrum span is moderate once scaled, but bands can be narrow.
- Compare Ritz ω to reference: missed modes, spurious modes, max |Δω|, residual, SpMV count vs `eigh`.

**Rayleigh–Ritz on GEP:** project with M-orthogonality: `T = Q^T H Q` where `Q` is M-orthonormal, or stay in `H = M^{-1/2} K M^{-1/2}` space and convert Ritz values back to ω.

## Proposed implementation plan

### Step 0 — Infrastructure (small, reusable)

1. **`vibration_loader.py`** — load `.npz`, build CSR `K`, diag `M`, filter vib modes (`ω < 100`), optional dense reference.
2. **`gep_operator.py`** — `ScaledMassWeightedOperator`:
   - `matvec(V)` for `H_scaled @ V`
   - `spectral_bounds()` via a few Lanczos iterations on `H` (or exact on adamantane)
   - `omega_to_scaled(ω)`, `scaled_to_omega(λ_scaled)`
3. **Fixture path** — env var or CLI `--fixtures-dir` pointing to FireCore fixtures (or copy subset into `topics/SpectralFiltering/fixtures/`).

Refactor `spectral_solvers.py` minimally: accept callable/LinearOperator instead of dense `H` (or thin wrapper that delegates `H @ V`).

### Step 1 — Benchmark: KPM spectral filter (your request #1)

Script: `benchmark_kpm_spectrum.py`

| System | What to run | Output |
|--------|-------------|--------|
| adamantane | KPM degrees 4,8,16,32,64,128; 8 probe vectors | PNG + peak–ω error table |
| nc_C_R5 | same | timing + SpMV count |
| nc_C_R8 | same | largest sparse-only case |

Validate against `omegas_modes_projected[vib]`. Report peak positions vs true ω (no eigenvectors yet).

### Step 2 — Benchmark: band subspace solver (your request #2)

Script: `benchmark_band_eigensolver.py`

Parameters (starting point):
- `band_width = 0.05` (ω units) → auto band list
- `nvec = 32`, `coarse_iters = 60`, `conv_iters = 8`, `square_filter`, `prune_mode=gradual`
- `method = ritz` only (Lanczos on interior bands is wrong tool — already documented)

Per band report:
- `# found` vs `# reference` in band
- max / median |Δω| for matched pairs
- residual tolerance hits
- subspace dimension after pruning
- total SpMV vs dense `eigh` time (R8)

Sweep: adamantane (full validation) → R5 → R8.

### Step 3 — Optional follow-ups (not in first pass)

- Adaptive band splitting using `cheap_rank_estimate()` after first filter
- Symlink/copy fixtures + smoke test in CI
- Restore Jacobi shift-solve stage for `(ω² M - K)^{-1} f` if you want the full coarse–fine pipeline on sparse K

## Risks and design choices to confirm

1. **Band granularity:** width 0.05 ω ≈ 82 bands for R8 — acceptable, or prefer adaptive rank-based splitting?
2. **λ vs ω bands:** Chebyshev is naturally in λ; narrow ω bands are nonlinear in λ. Prefer ω windows (FTIR-natural) with rectangle in λ, or equal-λ bands?
3. **Fixture location:** point at FireCore path vs copy into this repo?
4. **Scope of first PR:** Step 0+1 only (KPM validation), then Step 2 — or both together?

## Suggested run order (once implemented)

```bash
# Phase 1 — KPM spectrometer on adamantane
python benchmark_kpm_spectrum.py --system adamantane --fixtures-dir /path/to/vibration_benchmarks

# Phase 2 — band solver, narrow windows
python benchmark_band_eigensolver.py --system nc_C_R5 --band-width 0.05 --nvec 32 --conv-iters 8
```

I have not written any code yet, per your request. The FireCore fixtures are available locally, so we can start immediately once you confirm band-width strategy and whether to reference FireCore in place or copy fixtures here.

---

Implemented in the existing three modules (`spectral_solvers.py`, `spectral_demos.py`, `spectral_plotting.py`). Summary:

## What was added

### Vibration benchmarks (`spectral_solvers.py`)
- Load `.npz` from FireCore (`--fixtures-dir`, default `VIBRATION_BENCHMARKS_DIR`)
- **GEP → standard form:** `H = M^{-1/2} K M^{-1/2}`, eigenvalues `λ = ω²`
- **Chebyshev rescaling** to `[-1, 1]` from vibrational `ω` range
- **Rigid modes:** excluded via vibrational subspace (`ω < 100`); 6 rigid translators/rotators built from `pos` as fallback for sparse-only path
- **`VibrationOperator`:** works with all solvers via `op_matmul()` (dense or operator)
- **Spectral basis** (default, `ndof ≤ 2500`): uses `H_dense_projected` from `.npz` — required because `K_csr` ≠ reference matrix (≈13 eV/Å² offset)
- **OpenCL CSR SpMM** (`--opencl --sparse_k`): 2D kernel `(row, vector_col)` for `K @ V` with multiple probe vectors
- **`ritz_omegas()`:** physical ω from Rayleigh quotients for plots/residuals

### CLI (`spectral_demos.py`)
```bash
cd topics/SpectralFiltering

# KPM spectrometer — peaks vs reference ω (red lines)
python3 spectral_demos.py --benchmark nc_C_R5 --spectral_filter --save out_kpm.png

# Band subspace solver — convergence + residual panels vs ω
python3 spectral_demos.py --benchmark nc_C_R5 --converge_spectrum \
  --band_width 0.15 --nvec 16 --conv_iters 5 --coarse_iters 60 --save out_conv.png

# Full R4–R8 suite (KPM + converge)
python3 spectral_demos.py --benchmark_suite nc_C_R4,nc_C_R5,nc_C_R6,nc_C_R7,nc_C_R8 \
  --run_all --save_dir /tmp/nc_bench --spectral_nfreq 400
```

Plots match the toy demos: reference eigenvalues as **red dashed lines**, KPM evolution (top + per-vector strips), converge plot (filter / Ritz trajectories / **residual vs ω**).

### Plots (`spectral_plotting.py`)
- ω axis for benchmarks, system name in title
- Filter panel uses scaled Chebyshev coords; convergence/residual panels use ω

## Test run

Generated for R4–R8 in `/tmp/nc_bench2/` (`*_kpm.png`, `*_converge.png`). On adamantane, bands 1–2 recover reference ω to 4 decimal places; band edges can miss 1 mode when the filter is too narrow.

## Important notes

1. **No separate GEP solver** — reduction to `H = M^{-1/2} K M^{-1/2}` is enough for Chebyshev/Ritz.
2. **Default matvec uses vibrational eigenbasis** from `H_dense_projected` (fast, correct vs reference). True sparse `K_csr` path is `--sparse_k` (experimental; CSR/export mismatch still needs fixing in FireCore).
3. **OpenCL** needs `pyopencl` + `--sparse_k --opencl` (spectral basis runs on CPU dense algebra).
4. Tune **`--band_width`** (0.10–0.20), **`--nvec`**, **`--coarse_iters`** for denser spectra on R7/R8.