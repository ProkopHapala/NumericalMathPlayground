# Stiffness Fitting for Blended Rigid-Body Frames Force-Field (BRBFFF) — Implementation Plan

> **Implementation status (2026-07-16):** The executable implementation now uses the eta-coordinate contract described in the corrected results at the end of this document.  Earlier pseudocode and design decisions are retained as provenance; when they disagree with `fit_stiffness.py`, the executable code and corrected report are authoritative.

## Context

We have:
- **PTCDA molecule** loaded via `AtomicSystem` (38 atoms, planar, ~11.5 Å long)
- **Two frame centers** at bridge oxygens: `c1 ≈ [-5.73, 0, 0]`, `c2 ≈ [5.73, 0, 0]`
- **Fixed interpolation weights** `s_i ∈ [0,1]` from `blended_rigid_frames.py` (radial exp or smoothstep)
- **UFF Hessian** available via `py/FFs/Vibrations.py` → `run_vibrations(mol, backend='uff')`
- **UFF energy/force evaluator** via `py/FFs/UFF_cl.py` → `make_ff_eval_fn(mol)`

## Design Goals

**Primary — Mechanics:** Fit the **6×6 symmetric stiffness matrix K** (21 independent params) for the relative frame deformation, assuming weights and frame centers are fixed. The current training samples evaluate the full UFF energy on the **restricted blended geometry**; they do not relax hard Cartesian DOFs. A separate Schur-complement diagnostic estimates the relaxed target. Constrained-relaxation samples remain a TODO if the final model is meant to represent the relaxed potential. Mass plays **no role** in pure mechanics.

**Secondary — Dynamics (vibrational frequencies):** Once K is fitted, mass-weighted eigenvalue analysis can be used to predict vibrational frequencies and compare against UFF normal modes. This is a validation tool, not the fitting objective. Mass enters only here.

**Implication:** The core fitting methods (Galerkin, relaxed, energy/force LS) work in **pure stiffness space** (eV/Å², eV/rad²) without any mass weighting. Mass weighting is applied only as a post-processing step for frequency comparison.

---

## Mathematical Setup (from BlendedRigidBodyFrames.chat.md)

### Frame basis B(W)

For K=2 frames, atom i, frame a:

```
B_{ia} = w_{ia} * [ I_3 | -[r_i^0 - c_a^0]_× ]    (3×6 block)
```

Stacked: `B ∈ R^{3N × 12}` (12 = 6×2 frame DOFs).

For two frames with weights `w_{i1}=1-s_i`, `w_{i2}=s_i`:

```
δr_i = (1-s_i)[δt_1 + δθ_1 × (r_i - c1)] + s_i[δt_2 + δθ_2 × (r_i - c2)]
```

Frame coordinate vector: `δq = [δt_1, δθ_1, δt_2, δθ_2]^T ∈ R^12`

### Internal coordinates

12 frame DOFs minus 6 global rigid = **6 internal DOFs**.

Relative twist for two frames:
```
ξ = log(D_1^{-1} D_2) ∈ R^6    [ρ; φ]  (3 translation + 3 rotation)
```

Scaled to same units: `η = S·ξ`, `S = diag(1,1,1, ℓ,ℓ,ℓ)` where `ℓ = |c2-c1|`.

Harmonic frame energy:
```
V_frame = ½ η^T K η
```

K is symmetric 6×6 → **21 independent parameters**.

### Implementation note

The original plan used an orthonormal QR basis as if it were the physical internal coordinate map.  The implementation now keeps the geometry Jacobian explicit: `P_eta = ∂r/∂η`, and Galerkin stiffness is `P_eta.T @ H @ P_eta`.  This is a coordinate congruence, so energies and forces remain consistent.  QR is retained only for rank and overlap diagnostics.  The nonlinear sample generator also subtracts the reference generalized force before fitting curvature; otherwise float32 UFF residual forces masquerade as a linear term.

### Relation between B and internal coordinates

The 12 frame DOFs `δq` map to 6 internal relative coordinates `δξ` via a linear operator
(incidence matrix) `C ∈ R^{6×12}`:

```
δξ = C · δq     where C extracts relative motion: δξ = [-I, +I]·δq (approximately)
```

More precisely, for infinitesimal deformations:
```
δξ = δq_2 - δq_1   (relative twist = frame2 - frame1)
```

So `C = [-I_6, I_6]` (6×12), and the internal stiffness is:
```
K_internal = C^T K_edge C   (but K_edge is what we want to fit)
```

---

## Methods to Implement

### Method 1: Galerkin (Ritz) Stiffness — Direct, No Fitting

**Source:** Section 10.1 / 13 of chat.md

**Formula (mechanics — no mass needed):**
```
K_f = B^T H B       (12×12, pure stiffness in eV/Å²)
```

Remove 6 global rigid modes from K_f → project to internal 6×6:

1. Build rigid basis `G ∈ R^{3N×6}` — can use geometric rigid basis (translations + rotations about COM), **no mass weighting needed for mechanics**
2. Project: `B_perp = (I - G (G^T G)^{-1} G^T) B`  (Euclidean projection)
3. Orthonormalize columns of B_perp → `Q_B ∈ R^{3N×6}`
4. `K_Ritz = Q_B^T H Q_B` (6×6, in eV/Å²)

**This is pure linear algebra — no optimization, no mass.** The result is the exact restriction of the atomistic harmonic energy to the frame manifold.

**For dynamics (secondary):** Mass-weighted version uses `M_f = B^T M B` and mass-orthonormalization, but this only affects frequency prediction, not the stiffness itself.

**Code reuse:**
- `Vibrations.hessian_from_ff(mol, ff='uff')` → H, pos
- Rigid basis: can use `Vibrations.rigid_body_basis(pos, masses)` (mass-weighted) for dynamics, or build a simple geometric one for pure mechanics
- `np.linalg.qr` for orthonormalization

**Pros:** Exact, no fitting, single matrix product. No mass dependency.
**Cons:** Overestimates stiffness (atoms can't relax outside frame space).

---

### Method 2: Spectrally Calibrated Stiffness — Linear Algebra

**Source:** Section 10.2 of chat.md

**Goal:** Match the 6 lowest UFF eigenvalues exactly.

**Note:** This method is inherently **dynamics-oriented** — it matches vibrational frequencies, which requires mass weighting. For pure mechanics, Methods 1, 3, 4, 5 are more appropriate.

**Steps:**
1. Get target modes `U_6 = [u_1,...,u_6]` and eigenvalues `Λ = diag(λ_1,...,λ_6)` from `VibrationResult` (these are mass-weighted)
2. Map to coarse coordinates: `A = M_η^{-1} P^T M U_6` where `P = Q_B` (from Method 1, mass-weighted version)
3. Mass-orthonormalize: `S = A^T M_η A`, `Z = A S^{-1/2}`
4. `K_spec = M_η Z Λ Z^T M_η`

**Code reuse:**
- `VibrationResult.modes[:, :6]` and `VibrationResult.frequencies_cm1[:6]` → U_6, Λ
- `np.linalg.eigh` for S^{-1/2}

**Pros:** Exact frequency matching for 6 softest modes.
**Cons:** May give wrong forces for arbitrary frame displacements not aligned with modes. Requires mass — not a pure mechanics method.

---

### Method 3: Edge-Sparse Stiffness via Linear Least Squares

**Source:** Section 13 of chat.md (generalized)

**Goal:** Fit K_edge (6×6 symmetric, 21 params) such that:
```
K_model = C^T K_edge C  ≈  K_f (the 12×12 Galerkin stiffness)
```

where `C = [-I_6, I_6]` (6×12 incidence for single edge).

**Setup as linear LS:**

K_edge is symmetric 6×6 with 21 free parameters. Vectorize:
```
k = vec(K_edge) ∈ R^{21}    (upper triangle)
```

For each entry (i,j) of the 12×12 K_f:
```
K_f[i,j] = sum_{α≤β} k_{αβ} * (C^T E_{αβ} C)[i,j]
```

where E_{αβ} is the symmetric basis matrix for parameter k_{αβ}.

Build design matrix `X ∈ R^{144×21}` and solve:
```
min_k || X·k - vec(K_f) ||^2 + λ_K ||k||^2
```

**Positive semidefiniteness:** Either:
- (a) Fit unconstrained, then project K_edge to nearest PSD matrix (clamp negative eigenvalues)
- (b) Parameterize as `K_edge = L L^T + εI` (nonlinear, 21→21 params but guaranteed PSD)

**Code reuse:**
- `scipy.linalg.lstsq` or `np.linalg.lstsq` for unconstrained fit
- `np.linalg.eigh` for PSD projection

**Pros:** Sparse, transferable, physically interpretable per-edge stiffness.
**Cons:** Single edge for K=2 is trivial — more useful for K≥3.

---

### Method 4: Nonlinear Energy/Force Fitting via scipy.optimize

**Source:** Sections 20-22 of chat.md

**Goal:** Sample frame deformations, reconstruct atom positions, evaluate UFF E/F, fit K.

**Procedure:**

1. **Generate training samples:**
   - For each sample s, pick random relative twist `ξ_s ∈ R^6` (small amplitude, e.g. |φ| < 0.1 rad, |ρ| < 0.1 Å)
   - Build D_1 = I, D_2 = exp(ξ_s) (using SE(3) exponential map)
   - Reconstruct atom positions: `r_i = (1-s_i)·D_1·r_i^0 + s_i·D_2·r_i^0` (linear blend)
   - Evaluate UFF: `E_s, F_s = eval_fn(r)` via `make_ff_eval_fn`

2. **Compute generalized frame forces from atom forces:**
   - `F_a = Σ_i w_{ia} f_i`  (frame force)
   - `τ_a = Σ_i w_{ia} (y_{ia} × f_i)`  (frame torque)
   - where `y_{ia} = A_a (r_i^0 - c_a^0)` is the rotated reference offset
   - Generalized force on relative twist: `Q_s = [W_2 - W_1]` (approximately, for small deformations)

3. **Fit K (21 params) to energy and force data:**

   Energy model: `V_pred(ξ) = ½ η^T K η = ½ ξ^T S^T K S ξ`
   
   Force model: `Q_pred = -S^T K S ξ`

   Loss:
   ```
   L(K) = Σ_s [ w_E (V_pred(ξ_s) - E_s)^2 + w_F ||Q_pred(ξ_s) - Q_s||^2 ] + λ ||K||_F^2
   ```

   This is **linear least squares** in K entries if we only use energy+force (quadratic potential):
   ```
   V_pred = ½ Σ_{α≤β} k_{αβ} (Sξ)_α (Sξ)_β
   Q_pred,γ = -Σ_{α≤β} k_{αβ} S_{γγ} S_{αα} ξ_α   (for diagonal) ...
   ```

   Actually, for the harmonic case, both energy and force are **linear in K entries**, so this is still a linear LS problem!

   For **anharmonic** extensions (quartic, cos(φ), etc.), it becomes nonlinear → use `scipy.optimize.minimize` with L-BFGS-B.

4. **PSD constraint:** Use `K = L L^T + εI` parameterization with `scipy.optimize.minimize(method='L-BFGS-B')`.

**Code reuse:**
- `make_ff_eval_fn(mol)` → eval_fn for energy/force evaluation
- `scipy.optimize.minimize` or `scipy.linalg.lstsq`
- `blended_rigid_frames.py` weights and frame centers
- SE(3) exp map: implement small `exp_se3(ξ)` helper (Rodrigues formula + translation)

**Pros:** Can fit to arbitrary distorted geometries, not just Hessian. Extensible to anharmonic.
**Cons:** Requires multiple UFF evaluations (but N=38, so each is fast).

---

### Method 5: Relaxed (Statically Condensed) Stiffness — Schur Complement

**Source:** Section 10.3 of chat.md

**Goal:** Allow hard atomistic DOFs to relax for each frame deformation.

**Formula:**
```
K_relax = (C H_⊥^{-1} C^T)^{-1}
```

where:
- `C = (P^T P)^{-1} P^T` is the coarse extraction operator (P = Q_B from Method 1, **no mass needed for mechanics**)
- `H_⊥` is the Hessian projected to non-rigid space (using geometric rigid basis, not mass-weighted)

**Implementation as saddle-point solve (mechanics version — no mass):**

For each unit coarse coordinate e_α (α=1..6), solve:
```
[ H    C^T   G ] [ p_α     ]   [ 0    ]
[ C    0     0 ] [ λ_α     ] = [ e_α  ]
[ G^T  0     0 ] [ μ_α     ]   [ 0    ]
```

where G is the **geometric** rigid basis (translations + rotations about COM, Euclidean-orthonormalized). No mass matrix in the system.

The 6 vectors p_α form the relaxed basis P_*. Then:
```
K_relax = (C H_⊥^{-1} C^T)^{-1}
```

**For dynamics (secondary):** Replace G with mass-weighted rigid basis and add M to the saddle-point system if mass-weighted relaxation is desired.

**Code reuse:**
- `scipy.linalg.block_solve` or direct `np.linalg.solve` on the 3N+6+6 system
- H from `Vibrations.hessian_from_ff`
- Rigid basis: geometric version (simple to build) or `Vibrations.rigid_body_basis` for dynamics

**Pros:** Most physically rigorous for mechanics — accounts for relaxation of hard modes. No mass dependency.
**Cons:** Requires solving (3N+12) linear system 6 times. For N=38 → 126×126, trivial.

---

## Implementation Plan — Step by Step

### Step 0: Reuse existing data

```python
# From run_vib_PTCDA.py pattern:
from py.AtomicSystem import AtomicSystem
from py.FFs.Vibrations import run_vibrations, hessian_from_ff, atomic_masses, rigid_body_basis

mol = AtomicSystem(fname=PTCDA_FILE)
result = run_vibrations(mol, backend='uff', delta=1e-4, do_nonbond=False)

# Available:
# result.hessian    — (3N, 3N) eV/Å², rigid-projected
# result.modes      — (3N, n_modes) mass-normalized
# result.frequencies_cm1
# result.masses     — (N,) amu
# result.pos        — (N, 3) Å
```

Also from `blended_rigid_frames.py`:
```python
# Frame centers c1, c2 and weights s (either radial or smoothstep)
# mol.apos, mol.bonds, mol.ngs, mol.enames
```

### Step 1: Build frame basis B(W) — `build_frame_basis()`

**New function** (~20 lines):

```python
def build_frame_basis(pos, centers, weights):
    """Build B ∈ R^{3N × 6K} from atom positions, frame centers, weights.
    centers: (K, 3), weights: (N, K)
    """
    N, K = pos.shape[0], centers.shape[0]
    B = np.zeros((3*N, 6*K))
    for a in range(K):
        for i in range(N):
            r = pos[i] - centers[a]
            # [I | -r×] is 3×6
            skew = np.array([[0, -r[2], r[1]],
                            [r[2], 0, -r[0]],
                            [-r[1], r[0], 0]])
            Bia = np.hstack([np.eye(3), -skew])  # 3×6
            B[3*i:3*i+3, 6*a:6*a+6] = weights[i, a] * Bia
    return B
```

**No external dependencies beyond numpy.**

### Step 2: Remove rigid modes, extract internal basis — `extract_internal_basis()`

**New function** (~15 lines). Two variants: mechanics (no mass) and dynamics (mass-weighted):

```python
def rigid_basis_geometric(pos):
    """Build geometric rigid basis (no mass) — 3 translations + 3 rotations about COM."""
    N = pos.shape[0]
    com = pos.mean(axis=0)
    G = np.zeros((3*N, 6))
    for i in range(N):
        G[3*i:3*i+3, 0] = [1,0,0]  # Tx
        G[3*i:3*i+3, 1] = [0,1,0]  # Ty
        G[3*i:3*i+3, 2] = [0,0,1]  # Tz
        G[3*i:3*i+3, 3] = np.cross([1,0,0], pos[i]-com)  # Rx
        G[3*i:3*i+3, 4] = np.cross([0,1,0], pos[i]-com)  # Ry
        G[3*i:3*i+3, 5] = np.cross([0,0,1], pos[i]-com)  # Rz
    # Orthonormalize
    Q, _ = np.linalg.qr(G)
    return Q

def extract_internal_basis(B, pos, masses=None, bMassWeight=False):
    """Project B to remove 6 global rigid modes, return orthonormal Q_B ∈ R^{3N×6}.
    If bMassWeight=False (default, for mechanics): Euclidean projection, no mass.
    If bMassWeight=True (for dynamics): mass-weighted projection.
    """
    if bMassWeight:
        G = rigid_body_basis(pos, masses)  # reuse from Vibrations.py (mass-weighted)
        M_sqrt = np.diag(np.repeat(np.sqrt(masses), 3))
        B_proj = M_sqrt @ B
        G_proj = M_sqrt @ G
    else:
        G = rigid_basis_geometric(pos)  # no mass
        B_proj = B
        G_proj = G
    # Project out rigid modes
    B_perp = B_proj - G_proj @ (G_proj.T @ B_proj)
    # QR to orthonormalize
    Q_B, _ = np.linalg.qr(B_perp)
    # Remove zero columns (rank deficiency)
    norms = np.linalg.norm(Q_B, axis=0)
    Q_B = Q_B[:, norms > 1e-10]
    return Q_B
```

### Step 3: Method 1 — Galerkin stiffness — `galerkin_stiffness()`

**New function** (~5 lines). **No mass needed for mechanics:**

```python
def galerkin_stiffness(H, Q_B):
    """K_Ritz = Q_B^T H Q_B (6×6). Pure mechanics — no mass weighting."""
    K = Q_B.T @ H @ Q_B
    return 0.5 * (K + K.T)  # symmetrize
```

**For dynamics:** If Q_B was built with mass weighting, the result is the mass-weighted stiffness. To get frequencies, solve `K z = ω² M_η z` where `M_η = Q_B^T M Q_B`.

### Step 4: Method 2 — Spectrally calibrated stiffness — `spectral_stiffness()`

**Dynamics-only method** (requires mass weighting to match frequencies). Skipped for pure mechanics focus. Included for completeness.

**New function** (~15 lines):

```python
def spectral_stiffness(H, Q_B_mw, masses, target_modes, target_eigenvalues):
    """Match lowest mass-weighted eigenvalues exactly. DYNAMICS ONLY — requires mass."""
    M_sqrt = np.diag(np.repeat(np.sqrt(masses), 3))
    M_eta = Q_B_mw.T @ (M_sqrt @ M_sqrt) @ Q_B_mw  # should be I if mass-orthonormal
    U_mw = M_sqrt @ target_modes  # (3N, r)
    A = np.linalg.pinv(M_eta) @ (Q_B_mw.T @ U_mw)  # (6, r)
    S = A.T @ M_eta @ A
    S_sqrt_inv = np.linalg.inv(np.linalg.cholesky(S))
    Z = A @ S_sqrt_inv
    Lambda = np.diag(target_eigenvalues)
    K_spec = M_eta @ Z @ Lambda @ Z.T @ M_eta
    return 0.5 * (K_spec + K_spec.T)
```

### Step 5: Method 3 — Edge-sparse LS fit — `fit_edge_stiffness_ls()`

**New function** (~30 lines):

```python
def fit_edge_stiffness_ls(K_f_12x12, C, lambda_reg=1e-6):
    """Fit K_edge (6×6 symmetric, 21 params) via linear LS.
    K_model = C^T K_edge C ≈ K_f
    C: (6, 12) incidence matrix
    """
    # Build basis matrices E_{αβ} for symmetric 6×6
    basis_mats = []
    for a in range(6):
        for b in range(a, 6):
            E = np.zeros((6, 6))
            E[a, b] = E[b, a] = 1.0
            basis_mats.append(E)
    
    # Design matrix: each row = vec(C^T E_k C) for k=1..21
    X = np.zeros((144, 21))
    for k, E in enumerate(basis_mats):
        X[:, k] = (C.T @ E @ C).ravel()
    
    y = K_f_12x12.ravel()
    # Regularized LS
    XtX = X.T @ X + lambda_reg * np.eye(21)
    k_vec = np.linalg.solve(XtX, X.T @ y)
    
    # Reconstruct K_edge
    K_edge = np.zeros((6, 6))
    for k, E in enumerate(basis_mats):
        K_edge += k_vec[k] * E
    
    # PSD projection
    w, V = np.linalg.eigh(K_edge)
    w = np.maximum(w, 0)
    K_edge_psd = V @ np.diag(w) @ V.T
    return K_edge_psd, K_edge
```

### Step 6: Method 4 — Nonlinear energy/force fitting — `fit_stiffness_nonlinear()`

**New functions** (~80 lines total):

#### 6a: SE(3) exponential map helper

```python
def exp_se3(xi):
    """Exp map for se(3) twist xi=[rho(3), phi(3)] -> 4x4 homogeneous matrix."""
    rho = xi[:3]
    phi = xi[3:]
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        R = np.eye(3)
        t = rho
    else:
        # Rodrigues formula
        k = phi / theta
        K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
        R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*K@K
        # Translation (SE(3) exact)
        t = (np.eye(3)/theta * np.sin(theta) + 
             K/theta**2 * (1-np.cos(theta))) @ rho  # simplified
    H = np.eye(4)
    H[:3,:3] = R
    H[:3,3] = t
    return H
```

#### 6b: Reconstruct atom positions from frame deformations

```python
def reconstruct_positions(pos0, c1, c2, weights_s, D1, D2):
    """Linear blend: r_i = (1-s_i)*D1*r_i^0 + s_i*D2*r_i^0
    D1, D2: 4x4 homogeneous transforms (applied to absolute positions)
    """
    N = pos0.shape[0]
    pos = np.zeros((N, 3))
    for i in range(N):
        p1 = (D1[:3,:3] @ (pos0[i] - c1) + D1[:3,3] + c1)
        p2 = (D2[:3,:3] @ (pos0[i] - c2) + D2[:3,3] + c2)
        pos[i] = (1 - weights_s[i]) * p1 + weights_s[i] * p2
    return pos
```

#### 6c: Generalized frame forces from atom forces

```python
def atom_to_frame_forces(F_atoms, pos0, c1, c2, D1, D2, weights_s):
    """Reduce atom forces to frame wrenches via Jacobian transpose."""
    N = pos0.shape[0]
    F1 = np.zeros(3); F2 = np.zeros(3)
    tau1 = np.zeros(3); tau2 = np.zeros(3)
    for i in range(N):
        f = F_atoms[i]
        s = weights_s[i]
        y1 = D1[:3,:3] @ (pos0[i] - c1)
        y2 = D2[:3,:3] @ (pos0[i] - c2)
        F1 += (1-s) * f
        F2 += s * f
        tau1 += (1-s) * np.cross(y1, f)
        tau2 += s * np.cross(y2, f)
    return F1, tau1, F2, tau2
```

#### 6d: Fit K via LS on energy+force samples

```python
def fit_stiffness_from_samples(samples, ell, lambda_reg=1e-6):
    """samples: list of (xi_6, E_uff, Q_6) where Q = generalized force on relative twist.
    Fit K (6x6 symmetric) to V=½ η^T K η and Q = -S^T K S ξ.
    """
    S = np.diag([1,1,1, ell,ell,ell])
    # Build LS system: both energy and force equations are linear in K entries
    rows_A = []
    rows_b = []
    for xi, E, Q in samples:
        eta = S @ xi
        # Energy: E = ½ η^T K η → linear in k_{αβ}
        for a in range(6):
            for b in range(a, 6):
                coeff = 0.5 * eta[a] * eta[b] * (2.0 if a != b else 1.0)
                rows_A.append(coeff_row...)  # build design row
        rows_b.append(E)
        # Force: Q_γ = -Σ_{αβ} K_{αβ} S_{γγ} S_{αα} ξ_α ... (derive from -∂V/∂ξ)
        # Q = -S^T K S ξ  →  Q_γ = -Σ_{α,β} S_{γγ} K_{γβ} S_{ββ} ξ_β
        # This gives 6 more equations per sample
        ...
    # Solve LS
    k_vec, *_ = np.linalg.lstsq(A, b, rcond=None)
    # Reconstruct K
    ...
```

**Note:** The energy+force fitting for a **harmonic** (quadratic) potential is actually **linear LS** in K entries. Nonlinear optimization (scipy.optimize) is only needed for:
- PSD-constrained fit via `K = L L^T + εI`
- Anharmonic potential terms (cubic, quartic, cos)

#### 6e: PSD-constrained nonlinear fit (optional)

```python
from scipy.optimize import minimize

def fit_stiffness_nonlinear_psd(samples, ell, eps=1e-6):
    """Fit K = L L^T + eps*I via L-BFGS-B to guarantee PSD."""
    n_params = 21  # lower triangular L entries
    S = np.diag([1,1,1, ell,ell,ell])
    
    def unpack(params):
        L = np.zeros((6,6))
        idx = 0
        for i in range(6):
            for j in range(i+1):
                L[i,j] = params[idx]; idx += 1
        return L
    
    def loss(params):
        L = unpack(params)
        K = L @ L.T + eps * np.eye(6)
        total = 0
        for xi, E, Q in samples:
            eta = S @ xi
            V_pred = 0.5 * eta @ K @ eta
            Q_pred = -S.T @ K @ S @ xi
            total += (V_pred - E)**2 + np.sum((Q_pred - Q)**2)
        return total
    
    res = minimize(loss, x0=np.zeros(n_params), method='L-BFGS-B')
    L = unpack(res.x)
    return L @ L.T + eps * np.eye(6)
```

### Step 7: Method 5 — Relaxed stiffness (Schur complement) — `relaxed_stiffness()`

**New function** (~30 lines). **Mechanics version — no mass in the saddle-point system:**

```python
def relaxed_stiffness(H, pos, Q_B):
    """Schur complement: K_relax = (C H_⊥^{-1} C^T)^{-1}
    Mechanics version — no mass weighting. Solve saddle-point for each unit coarse coordinate.
    """
    N = pos.shape[0]
    ndof = 3 * N
    
    # Coarse extraction: C = (Q_B^T Q_B)^{-1} Q_B^T  (Euclidean, no mass)
    C = np.linalg.solve(Q_B.T @ Q_B, Q_B.T)  # (6, 3N)
    
    # Geometric rigid basis (no mass)
    G = rigid_basis_geometric(pos)  # (3N, 6)
    
    # Saddle-point system: [H, C^T, G; C, 0, 0; G^T, 0, 0]  — NO mass matrix
    n_coarse = C.shape[0]  # 6
    n_rigid = G.shape[1]   # 6
    n_total = ndof + n_coarse + n_rigid
    
    A = np.zeros((n_total, n_total))
    A[:ndof, :ndof] = H
    A[:ndof, ndof:ndof+n_coarse] = C.T
    A[:ndof, ndof+n_coarse:] = G
    A[ndof:ndof+n_coarse, :ndof] = C
    A[ndof+n_coarse:, :ndof] = G.T
    
    # Solve for 6 unit coarse coordinates
    P_star = np.zeros((ndof, n_coarse))
    for alpha in range(n_coarse):
        rhs = np.zeros(n_total)
        rhs[ndof + alpha] = 1.0
        sol = np.linalg.solve(A, rhs)
        P_star[:, alpha] = sol[:ndof]
    
    K_relax = np.linalg.inv(C @ P_star)
    return 0.5 * (K_relax + K_relax.T)
```

### Step 8: Compare and validate

**Two levels of comparison:**

**8a — Mechanics (primary): Compare stiffness eigenvalues directly (no mass)**

```python
def compare_stiffness_mechanics(K_galerkin, K_relax, K_edge, K_samples):
    """Compare stiffness eigenvalues (eV/Å²) — pure mechanics, no mass."""
    results = {}
    for name, K in [('Galerkin', K_galerkin), ('Relaxed', K_relax),
                     ('Edge-LS', K_edge), ('E/F-fit', K_samples)]:
        w = np.linalg.eigvalsh(K)
        results[name] = w  # in eV/Å² (or eV/rad² for rotational components)
    return results
```

**8b — Dynamics (secondary): Compare vibrational frequencies (mass-weighted)**

```python
def compare_stiffness_frequencies(K_galerkin, K_spectral, K_relax, K_edge, masses, Q_B):
    """Diagonalize each K in mass-weighted frame space, compare frequencies in cm^-1."""
    M = np.diag(np.repeat(masses, 3))
    M_eta = Q_B.T @ M @ Q_B
    M_eta_inv_sqrt = np.linalg.inv(np.linalg.cholesky(M_eta))
    
    results = {}
    for name, K in [('Galerkin', K_galerkin), ('Spectral', K_spectral), 
                     ('Relaxed', K_relax), ('Edge-LS', K_edge)]:
        D = M_eta_inv_sqrt @ K @ M_eta_inv_sqrt
        w = np.linalg.eigvalsh(D)
        # Convert to cm^-1
        eV_to_J = 1.602176634e-19
        amu_to_kg = 1.66053906660e-27
        ang_to_m = 1e-10
        c_cm = 2.99792458e10
        freqs = np.sqrt(np.maximum(w, 0) * eV_to_J / (amu_to_kg * ang_to_m**2)) / c_cm
        results[name] = freqs
    return results
```

**8c — Force matching (primary mechanics validation):**

For each method, apply a known frame displacement ξ, compute `Q_pred = -S^T K S ξ`,
and compare against UFF generalized forces `Q_uff` obtained from `atom_to_frame_forces()`.
Report RMS force residual per method.

### Step 9: Plot comparison

Plot frame-mode frequencies vs UFF target frequencies as bar chart or scatter.
Reuse matplotlib pattern from `blended_rigid_frames.py`.

---

## File Structure

**Single new file:** `topics/MultiGridFF/fit_stiffness.py`

```
# Imports from existing code:
from py.FFs.Vibrations import run_vibrations, hessian_from_ff, rigid_body_basis, atomic_masses
from py.AtomicSystem import AtomicSystem
# (reuse blended_rigid_frames.py for weights and frame centers)

# New functions (all pure numpy/scipy):
# 1. build_frame_basis()
# 2. extract_internal_basis()
# 3. galerkin_stiffness()
# 4. spectral_stiffness()
# 5. fit_edge_stiffness_ls()
# 6. exp_se3()
# 7. reconstruct_positions()
# 8. atom_to_frame_forces()
# 9. fit_stiffness_from_samples()       — linear LS on E+F
# 10. fit_stiffness_nonlinear_psd()     — scipy L-BFGS-B with K=LL^T
# 11. relaxed_stiffness()               — Schur complement
# 12. compare_stiffness_methods()
# 13. main() — orchestrate all methods, print comparison table, plot
```

**Estimated total new code:** ~300-400 lines (all pure numpy/scipy, no new dependencies).

---

## Data Flow Summary

```
PTCDA.xyz
    ↓
AtomicSystem (existing)
    ↓
blended_rigid_frames.py → weights s_i, frame centers c1, c2 (existing)
    ↓
run_vibrations(mol, 'uff') → H, modes, freqs, masses (existing)
    ↓
build_frame_basis(pos, [c1,c2], [[1-s],[s]]) → B (12×3N)  [NEW]
    ↓
extract_internal_basis(B, pos, masses) → Q_B (3N×6)  [NEW]
    ↓
┌─────────────────────────────────────────────────────┐
│ Method 1: galerkin_stiffness(H, Q_B) → K_Ritz 6×6  │  [NEW, ~5 lines]
│ Method 2: spectral_stiffness(H, Q_B, U, Λ) → K_spec │  [NEW, ~10 lines]
│ Method 3: fit_edge_stiffness_ls(K_f, C) → K_edge    │  [NEW, ~30 lines]
│ Method 4: fit_stiffness_from_samples() → K_nl       │  [NEW, ~80 lines]
│ Method 5: relaxed_stiffness(H, pos, masses, Q_B)    │  [NEW, ~30 lines]
└─────────────────────────────────────────────────────┘
    ↓
compare_stiffness_methods() → frequency comparison table  [NEW, ~20 lines]
    ↓
plot comparison  [NEW, ~30 lines]
```

---

## Key Design Decisions

1. **Mechanics first, dynamics second:** The primary goal is fitting the stiffness matrix K for static relaxation and force reproduction. Mass is irrelevant for mechanics. Mass-weighted analysis (frequencies, mode shapes) is a secondary validation step.

2. **Units:** H from UFF is in eV/Å². All K matrices are in eV/Å² (for translation) and eV/rad² (for rotation). Use scaling matrix S = diag(1,1,1,ℓ,ℓ,ℓ) to equalize units. No mass units in K itself.

3. **No mass in core methods:** Methods 1 (Galerkin), 3 (edge LS), 4 (E/F fitting), and 5 (relaxed) operate in pure stiffness space. Only Method 2 (spectral) requires mass because it matches vibrational frequencies.

4. **Rigid mode removal:** Use geometric rigid basis (translations + rotations about COM, Euclidean-orthonormalized) for mechanics. Mass-weighted rigid basis from `Vibrations.rigid_body_basis()` only for dynamics validation.

5. **SE(3) exp map:** Implement minimal `exp_se3(xi)` using Rodrigues formula. Only needed for Method 4 (nonlinear fitting). For small deformations, can also use linear approximation.

6. **Linear vs nonlinear:** Methods 1-3 and 5 are **pure linear algebra** (no optimization). Method 4 is linear LS for harmonic potential, nonlinear only for PSD constraint or anharmonic extensions.

7. **scipy usage:** `scipy.optimize.minimize(method='L-BFGS-B')` for PSD-constrained fit. `scipy.linalg.lstsq` as alternative to `np.linalg.lstsq`. No other scipy dependencies.

8. **Validation — two levels:**
   - **Mechanics (primary):** Compare stiffness eigenvalues (eV/Å²) across methods. Check force residuals on test frame displacements. Verify energy matching on distorted geometries.
   - **Dynamics (secondary):** Compare mass-weighted frequencies (cm⁻¹) against UFF normal modes. Compute principal angles between frame subspace and target mode subspace.

---

## What NOT to Implement (Yet)

- Weight optimization (FIRE/Adam/BFGS) — weights are fixed for now
- Anharmonic potential fitting (quartic, cos terms) — start with harmonic
- Multi-frame (K>2) generalization — start with K=2
- Dual-quaternion or SE(3) nonlinear interpolation — use linear blend for position reconstruction
- Graph biharmonic weights — geometric weights are sufficient
- Constrained relaxation for training data generation — use direct frame displacement + UFF evaluation

---

## Testing Strategy

### Mechanics (primary)

1. **Stiffness eigenvalue ordering:** K_galerkin eigenvalues (eV/Å²) should be ≥ K_relax eigenvalues (Galerkin overestimates, relaxation softens)
2. **Force consistency:** For each restricted method, apply a test displacement, form `η = S ξ`, and compare `Q_η = -K_η η` with the UFF virtual-work force `J_ξ^T F` transformed into eta coordinates.  Do not score the statically condensed `Relaxed` matrix against these unrelaxed samples.
3. **Energy matching:** For distorted geometries, compare `V_frame = ½ η^T K η` against actual UFF energy difference `E(ξ) - E(0)`.
4. **PSD check:** All K matrices should be positive semidefinite (no negative eigenvalues).
5. **Symmetry:** K should respect PTCDA's D₂h symmetry (block-diagonal structure when expressed in symmetry-adapted coordinates).

### Dynamics (secondary)

6. **Frequency comparison:** Mass-weighted eigenvalues of K should be close to 6 lowest UFF vibrational frequencies.
7. **Spectral match:** Compare the spectral fit only on the represented subspace.  A rank-deficient overlap is expected when target UFF modes lie outside the frame tangent space; in the current two-frame test the rank is 4/6.
8. **Rayleigh-Ritz diagnostic:** A Galerkin restriction is an upper-bound construction only when the same generalized eigenproblem and nested subspaces are being compared.  It is not a promise that each sorted frame frequency equals or bounds the corresponding molecular frequency.

---

## Historical results — pre-fix implementation (2025-07-16)

> The numbers and conclusions in this historical section describe the broken coordinate/extraction implementation.  They are retained as a debugging record only; use the corrected report below.

### Setup

- PTCDA (38 atoms), UFF force field, relaxed to minimum (E=0.038 eV)
- 2 frames at bridge oxygens: c1≈[-5.73, 0, 0], c2≈[5.73, 0, 0], ℓ=11.47 Å
- Smoothstep weights s_i ∈ [0, 1]
- Absolute position transforms (fixed rigid mode invariance bug)
- Energy scans: frame 1 fixed, frame 2 deformed along each of 6 DOFs

### Stiffness eigenvalues (eV/Å²)

| Method | λ₁ | λ₂ | λ₃ | λ₄ | λ₅ | λ₆ |
|--------|-----|-----|-----|-----|-----|-----|
| Galerkin | 0.109 | 0.181 | 0.917 | 1.419 | 2.872 | 4.546 |
| Edge-LS | 0.080 | 3.293 | 3.840 | 6.481 | 13.744 | 76.018 |
| E/F-fit | 0.016 | 0.056 | 0.065 | 1.276 | 12.005 | 23.713 |
| Relaxed | 0.094 | 0.136 | 0.542 | 0.733 | 2.297 | 3.866 |

### Force residuals (RMS per sample)

| Method | RMS force residual |
|--------|-------------------|
| Galerkin | 0.74 |
| Edge-LS | 23.4 |
| E/F-fit | 0.38 |
| Relaxed | 0.78 |

### Per-DOF energy scan: Galerkin diagonal vs UFF effective stiffness

| DOF | Galerkin K_diag | UFF effective | Ratio (G/UFF) | c4/c2 | Assessment |
|-----|----------------|---------------|---------------|-------|------------|
| tx | 1.44 | 6.87 | 0.21× | 0.001 | Galerkin 5× too soft |
| ty | 2.27 | 3.24 | 0.70× | 0.003 | Galerkin 1.4× too soft |
| tz | 0.46 | 0.04 | 11.5× | 0.59 | Galerkin 11× too stiff |
| rx | 10.84 | 1.65 | 6.6× | 2.79 | Galerkin 6.6× too stiff |
| ry | 6.49 | 1.91 | 3.4× | 1.94 | Galerkin 3.4× too stiff |
| rz | 84.90 | 38.04 | 2.2× | 0.02 | Galerkin 2.2× too stiff |

### Rigid mode invariance — FIXED

All 6 rigid modes (both true rigid and frame-blended) now give max|ΔE| < 1.5e-6 eV. Previously, frame-blended rotations gave ΔE up to 1.085 eV due to center-relative transforms. Fixed by using absolute position transforms in `build_frame_basis` and `reconstruct_positions`.

---

### Analysis: Why Translations Are Somewhat OK but Rotations Are Bad

#### 1. Galerkin overestimates rotational stiffness (3-11×)

The Galerkin method computes K = Q_B^T H Q_B, which is the exact restriction of the Hessian to the frame subspace. This means atoms are **constrained to move as blended rigid-body motions** — no internal relaxation is allowed.

For **translations** (tx, ty): frame translation displaces atoms proportionally to their weight. Atoms near the deformed frame move more. This is a shearing motion that partially matches real internal deformations, so the stiffness is in the right ballpark (within 5×).

For **rotations** (rx, ry, rz): frame rotation rotates atoms about the **origin** with rotation arm |r_i| up to ~6 Å. This creates large displacements for distant atoms. In reality, the molecule can relax by bond stretching and angle bending, which absorbs the deformation much more softly. The Galerkin method cannot capture this relaxation, so it overestimates rotational stiffness by 3-11×.

The **Relaxed** method (Schur complement) should fix this by allowing hard DOFs to relax, and indeed it gives softer stiffness (eigenvalues 0.094-3.866 vs Galerkin 0.109-4.546). But the improvement is modest — the relaxation only helps partially because the frame subspace itself is a poor approximation of the soft modes.

#### 2. Edge-LS is especially bad for rotations (3-70× too stiff)

The Edge-LS method fits K_edge such that C^T K_edge C ≈ K_f (12×12), where C = [-I, I]. The diagnostic shows:

- `C @ q_rigid = 0.000` — rigid modes are correctly in null(C) (after the absolute-position fix)
- But C^T K_edge C has **rank 6** and **zero rigid-internal coupling** by construction
- K_f = B^T H B (12×12) has significant **off-diagonal coupling** between rigid and internal blocks (especially for rotations, where the rotation arm creates large cross-terms)
- The LS fit tries to match all 144 entries of K_f with only 21 parameters, but the model can only represent the 6×6 internal block — the rigid block and coupling are forced to zero
- The fit **inflates the internal stiffness** to compensate for the missing coupling, producing eigenvalues 3-70× larger than the correct Q_B-based extraction
- The effect is worse for rotations because rigid-internal coupling scales with rotation arm (~5-6 Å), making the missing coupling terms much larger

**The naive C-based extraction** `K_edge_naive = C K_f C^T` gives eigenvalues [0.32, 13.17, 15.36, 25.92, 54.97, 304.07] — already 3-94× too stiff. The LS fit can't fix this because the problem is in the extraction operator C itself, not in the fitting.

#### 3. Galerkin doesn't work well even for translations

Looking at the per-DOF comparison:
- **tx**: Galerkin K_diag = 1.44, but UFF effective = 6.87 → Galerkin is **5× too soft**
- **ty**: Galerkin = 2.27, UFF = 3.24 → 1.4× too soft
- **tz**: Galerkin = 0.46, UFF = 0.04 → **11× too stiff**

This is inconsistent — sometimes too stiff, sometimes too soft. The reason is that the frame basis B mixes translation and rotation components through the weights. A pure tx deformation of frame 2 doesn't produce a pure translation of atoms — it produces a weighted translation that looks like a shear. The UFF Hessian sees this shear differently than a simple translation.

The **diagonal elements** of K in the frame coordinate basis are not physically meaningful individual stiffnesses — they mix different physical motions. Only the eigenvalues of K are invariant, and those don't map cleanly to individual DOF scans.

#### 4. E/F-fit: best force matching but wrong eigenvalue spectrum

E/F-fit has the lowest force residual (0.38 vs 0.74 for Galerkin), but its eigenvalue spectrum is very different: two near-zero modes (0.016, 0.056) and two very stiff modes (12.0, 23.7). This suggests the fit is:
- Capturing **anharmonic** effects (the training samples at ±0.05 Å / ±0.005 rad sample beyond the harmonic regime)
- Fitting to **force directions** that don't align with the eigenvectors of the true harmonic stiffness
- The 6×6 K has enough freedom to fit forces well at sample points but not to reproduce the correct curvature

#### 5. Fundamental problem: 6 DOFs is too few for PTCDA

PTCDA has 38 atoms → 108 internal DOFs. Compressing to 6 frame DOFs means the model can only represent 6 collective modes. The frame basis B maps frame motions to atom displacements as:

```
δr_i = (1-s_i) * (δt₁ + δθ₁ × r_i) + s_i * (δt₂ + δθ₂ × r_i)
```

This is a **very specific** 6-dimensional subspace of the 108-dimensional internal space. The actual soft modes of PTCDA involve:
- In-plane bending (C-C-C angle deformation)
- Out-of-plane bending (z-displacement of the perylene core)
- Bond stretching along the long axis
- Localized vibrations of the anhydride groups

These don't map cleanly to frame translations and rotations. The frame model's 6 modes are:
1. Relative translation along x (stretching) — partially matches bond stretching
2. Relative translation along y (shearing) — partially matches in-plane bending
3. Relative translation along z (out-of-plane shear) — partially matches out-of-plane bending
4. Relative rotation about x (twisting) — doesn't match any soft mode well
5. Relative rotation about y (bending) — partially matches long-axis bending
6. Relative rotation about z (in-plane rotation) — doesn't match any soft mode

The **principal angles** between the frame subspace and the 6 softest UFF modes would quantify this mismatch. When the subspaces are nearly orthogonal, no fitting method can produce correct results.

### Conclusions

1. **Rigid mode invariance** is now correctly preserved (absolute transforms fix the blending bug)
2. **No single method reproduces all 6 modes correctly** because the frame basis is too restrictive
3. **Galerkin** gives an upper bound but overestimates rotational stiffness 3-11× (no relaxation)
4. **Relaxed** method is the most physically correct but still limited by the 6-DOF restriction
5. **Edge-LS** is fundamentally flawed for K=2: the incidence matrix C cannot represent rigid-internal coupling, inflating stiffness 3-70×
6. **E/F-fit** gives best force matching but wrong eigenvalue spectrum due to anharmonic contamination
7. **The core issue** is that 6 frame DOFs cannot capture the 108 internal DOFs of PTCDA. The frame basis is not a good reduced basis for the molecule's soft modes.

### Next steps

- Compute principal angles between frame subspace and UFF soft modes to quantify mismatch
- Consider increasing the number of frames (K=3 or 4) to get more DOFs
- Consider adding per-atom correction terms (hard springs) to the frame model
- Consider fitting anharmonic terms (quartic) for rotations where c4/c2 > 1
- Consider weight optimization to better align the frame subspace with soft modes

## Corrected implementation results (2026-07-16)

The first report above is a debugging record, not the current result.  The central repair was to make the internal-coordinate contract explicit:

\[
  \eta = S\,\xi, \qquad V_2(\eta)=\tfrac12\eta^T K_\eta\eta .
\]

Here \(\xi\) is the six-component relative frame motion and \(S\) is the translation/rotation scaling matrix.  Every fitted or projected stiffness in `fit_stiffness.py` is now a stiffness in **eta coordinates**.  A QR factorization is used only to diagnose the span; its orthonormal coordinates are not physical coordinates and must not be used as if they were a congruence transform.

### What is compared

The restricted models (Galerkin, Edge-LS, and energy/force fit) all predict forces for the same unrelaxed blended geometry.  `Relaxed` is a different target: it is a Schur-complement/static-condensation estimate in which eliminated Cartesian coordinates are allowed to respond.  Its eigenvalues must not be scored against the restricted force samples.

| Method | η-space eigenvalues (eV/Å²) | Interpretation |
|---|---|---|
| Galerkin | 0.0275, 0.0321, 0.0793, 0.6352, 6.4803, 13.7437 | Exact Hessian restriction to the blended frame subspace |
| Edge-LS | 0.0275, 0.0321, 0.0801, 0.6353, 6.4806, 13.7437 | Equivalent internal-edge extraction when the right inverse is used |
| E/F-fit | 0.0267, 0.0324, 0.0799, 0.6363, 6.4822, 13.7487 | Direct least-squares fit to restricted energy and forces |
| Relaxed | 0.0206, 0.0279, 0.0469, 0.3281, 5.5111, 10.9916 | Softened response after Cartesian relaxation; different observable |

The Edge-LS/direct relative error is \(2.5\times10^{-7}\).  The independent virtual-work check has relative L2 error \(1.8\times10^{-3}\) (absolute force error about \(4.1\times10^{-4}\) eV/Å).  These are the useful parity checks; comparing raw matrix entries in different coordinates is not.

### Frequencies and representability

The generalized eigenproblem is \(K_\eta v=\omega^2 M_\eta v\), with reduced mass metric \(M_\eta=P_\eta^T M P_\eta\).  Wavenumbers use \(\omega/(2\pi c)\).  The restricted models give approximately 50, 76, 149, 191, 259, and 406 cm⁻¹, whereas the six lowest UFF modes are 46, 65, 123, 125, 139, and 164 cm⁻¹.

The spectral target has numerical rank **4/6**: two UFF low-mode directions have almost zero overlap with the two-frame tangent space.  This is a representation limit, not a failed least-squares solve.  Adding more frames or residual internal basis vectors is required if those modes are important.

### Non-obvious caveats

- `Edge-LS` needs the factor \(\tfrac14 C K_f C^T\), because the incidence matrix satisfies \(CC^T=2I\).  The naive product is systematically too stiff.
- Absolute position transforms are essential.  Rotating each frame about its own center before blending breaks global rigid-mode invariance; the corrected checks give maximum energy drift below \(1.4\times10^{-6}\) eV.
- The UFF/OpenCL path uses float32 internally.  Near the minimum its force floor is about \(2\times10^{-3}\) eV/Å, so a relaxation message saying “not converged” can reflect numerical precision rather than a bad Hessian.
- Finite-amplitude scans are not purely harmonic.  For the reported test, the out-of-plane translation has a quartic fraction of about 0.15; use the Hessian/near-zero fit when the goal is harmonic stiffness.
- A diagonal entry in the frame basis is coordinate-dependent and should not be interpreted as “the translation stiffness”.  Compare generalized eigenvalues or energies along a specified displacement instead.

### Remaining TODOs

1. Generate genuinely relaxed training samples (constrained minimization) if the desired model is a relaxed potential rather than a restricted one.
2. Add three or more frames, or append residual internal modes, and re-measure principal angles/rank before fitting more parameters.
3. Optimize blending weights against the mass-weighted low-mode subspace; the current smoothstep weights are geometric, not learned.
4. Replace diagnostic finite-difference geometry Jacobians with an analytic SE(3) Jacobian in a production implementation.
5. Decide whether finite-temperature/environment-dependent corrections or explicit quartic terms are needed for the intended use.
