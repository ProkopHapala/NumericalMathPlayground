// =============================================================================
// kekule_fluid.cl — OpenCL kernels for Model A: Kekulé fluid solver
//
// =============================================================================
// PHYSICAL OVERVIEW
// =============================================================================
//
// This file implements the "Kekulé fluid" model (Model A) for the time
// evolution of bond-order alternation (Kekulé pattern) on a honeycomb graph
// (e.g. graphene).  The central object is a complex Kekulé order parameter
// z_i defined on each atom i, which encodes which of the three possible
// Kekulé dimer coverings is locally preferred.
//
// --- Physical motivation ---
//
// In a sp² carbon network (graphene, nanotubes, polycyclic aromatics) each
// atom contributes one p_z electron to the π system.  The π bond order
// x_b ∈ [0, 1] on bond b measures the fraction of π electron density on
// that bond.  In a perfect Kekulé structure, each atom has exactly one
// "double bond" (x_b = 1) and two "single bonds" (x_b = 0), so the three
// bond orders around an atom sum to 1 (the valence constraint):
//
//   Σ_{b ∈ N(i)} x_b = targetVal_i   (typically 1.0 for sp² carbon)
//
// On a honeycomb lattice there are three degenerate Kekulé patterns,
// related by 120° rotation.  We label them by a phase φ ∈ {0, 2π/3, 4π/3}.
// The complex order parameter z_i captures the local preference:
//
//   z_i = Σ_{b ∈ N(i)} (x_b - x̄_i) · ω_{dir(b)}
//
// where ω_d = exp(i·θ_d) with θ_d = {0, 2π/3, 4π/3} the three bond
// directions, and x̄_i is the mean bond order around atom i.  Thus z_i is
// the threefold Fourier component of the local bond-order pattern.
//
//   |z_i| = 0  →  uniform (aromatic, no Kekulé alternation)
//   |z_i| > 0  →  Kekulé dimerization present
//   arg(z_i)   →  selects which of the 3 Kekulé patterns
//
// --- Dynamics ---
//
// The order parameter evolves according to a damped complex
// Ginzburg–Landau equation:
//
//   dz_i/dt = -η·g_i - i·Ω·g_i
//
// where g_i is the "force" (gradient of a free-energy-like functional):
//
//   g_i = κ·∇²z_i                          (spatial stiffness)
//       + r·z_i                             (linear: r<0 → wants |z|>0)
//       + u·|z_i|²·z_i                      (cubic saturation)
//       - λ·(z_i*)²                         (Z₃ locking: selects 3 phases)
//       + k_core·defect_i·z_i               (defect core suppression)
//       + k_pin·p_i·(z_i - z_i^pin)         (external phase pinning)
//
// The -η·g term is dissipative (relaxes toward free-energy minima).
// The -i·Ω·g term is conservative (Schrödinger-like phase rotation).
//
// --- Time step pipeline (stepModelA) ---
//
//   1. bondsToZ:           x_b → z_i        (Fourier reconstruction)
//   2. applyPinsToZ:       hard defect/pin  (boundary conditions on z)
//   3. evolveZ_RK4:        z^n → z^{n+1}   (4th-order Runge–Kutta)
//   4. zToRawBonds:        z_i → x_b^raw   (inverse Fourier + amplitude)
//   5. copyRawToX:         x_b = x_b^raw
//   6. projectBondOrders:  enforce Σ x_b = targetVal  (red-black Gauss-Seidel)
//   7. bondsToZ:           recompute z for diagnostics
//
// =============================================================================
// STATE VARIABLES (buffers)
// =============================================================================
//
// z_real[NATOM], z_imag[NATOM]   — complex Kekulé order parameter z_i
//   Physical meaning: local Kekulé dimerization amplitude and phase.
//   |z_i| ∈ [0,1] is the alternation strength; arg(z_i) ∈ [0,2π) selects
//   the Kekulé pattern.  Stored as two float arrays (real, imag).
//
// bond_x[NBOND]                  — projected π bond order x_b ∈ [0,1]
//   Physical meaning: fraction of π electron density on bond b.
//   Satisfies the valence constraint Σ x_b = targetVal around each atom.
//
// bond_xRaw[NBOND]               — unprojected bond order (before valence projection)
//   Physical meaning: what the z-field "wants" the bond order to be,
//   before enforcing the chemical valence constraint.
//
// defect[NATOM]                  — defect strength ∈ [0,1]
//   Physical meaning: 0 = normal sp² carbon, 1 = fully protonated/sp³
//   (e.g. H+ addition).  Suppresses Kekulé order at the site.
//
// pinStrength[NATOM]             — phase pinning strength ∈ [0,1]
//   Physical meaning: 0 = free, 1 = hard-pinned to pinPhase.
//   Used to fix the Kekulé phase at defect/boundary sites.
//
// pinPhase[NATOM]                — preferred Kekulé phase (radians)
//   Physical meaning: the desired arg(z) at pinned sites.
//   Typically set to θ_dir[d] to prefer dimerization along direction d.
//
// targetVal[NATOM]               — target π valence (sum of bond orders)
//   Physical meaning: 1.0 for normal sp² carbon, 0.0 for protonated (H+).
//
// bondBase[NBOND]                — baseline bond order x_b^0
//   Physical meaning: uniform/aromatic bond order before Kekulé
//   alternation.  Computed as 0.5·(targetVal_i/deg_i + targetVal_j/deg_j).
//
// atom_bonds[NATOM×3]            — bond indices incident to each atom (-1 if absent)
// bond_dir[NBOND]                — direction index {0,1,2} for each bond
// bond_iA[NBOND], bond_iB[NBOND] — A/B sublattice atom indices of each bond
// neighbors[NATOM×3]             — neighbor atom indices (-1 if absent)
//
// =============================================================================
// PARAMETERS (Params struct)
// =============================================================================
//
// kappa   = 0.20   spatial stiffness (graph Laplacian coupling)
// r       = -1.00  linear coefficient (r<0 → spontaneous Kekulé order)
// u       = 1.00   cubic saturation (prevents |z|→∞)
// lambda  = 0.15   Z₃ anisotropy (locks arg(z) to 0, 2π/3, 4π/3)
// eta     = 0.08   dissipative relaxation rate
// Omega   = 1.00   conservative (Schrödinger-like) phase rotation frequency
// k_pin   = 2.00   phase pinning stiffness
// k_core  = 5.00   defect core suppression strength
// dtA     = 0.02   time step
// A_pin   = 1.00   amplitude of pinned z target
// nProj   = 10     number of ABBA projection iterations (4 sweeps each = 40 total)
//
// =============================================================================

// Complex numbers are stored as pairs of floats: (real, imag)

// Parameters struct passed as a single buffer
typedef struct {
    float kappa;     // spatial stiffness
    float r;         // linear coefficient (r<0 → spontaneous order)
    float u;         // cubic saturation
    float lambda;    // Z₃ anisotropy (Kekulé phase locking)
    float eta;       // dissipative relaxation rate
    float Omega;     // conservative phase rotation frequency
    float k_pin;     // phase pinning stiffness
    float k_core;    // defect core suppression
    float dtA;       // time step
    float A_pin;     // pinned z amplitude target
    float kekule_amp; // bond modulation amplitude factor (0.8 default, 2.0 for full dimer)
    float omega_proj; // projection relaxation parameter (0.5 default, 1.0 for exact)
} Params;

// Bond directions: θ_d for d = 0, 1, 2
// These are the angles of the three honeycomb bond directions in the plane.
// ω_d = exp(i·θ_d) maps bond direction d to a complex phase.
// The three Kekulé patterns correspond to arg(z) = θ_0, θ_1, θ_2.
__constant float thetaDir[3] = {0.0f, 2.09439510239f, 4.18879020479f}; // 0, 2π/3, 4π/3

// =============================================================================
// Step 1: bondsToZ — Reconstruct complex Kekulé order z_i from bond orders
// =============================================================================
//
// Given the projected bond orders x_b, compute the complex Kekulé order
// parameter z_i on each atom.  This is the threefold Fourier component
// of the local bond pattern:
//
//   x̄_i = (1/n_i) · Σ_{b ∈ N(i)} x_b          (mean bond order)
//
//   z_i = Σ_{b ∈ N(i)} (x_b - x̄_i) · ω_{dir(b)}
//
// where ω_d = exp(i·θ_d) and n_i is the number of bonds at atom i.
//
// Interpretation:
//   - If all bonds are equal (x_b = x̄), then z_i = 0 (aromatic, no dimerization).
//   - If one bond is strong and two are weak, z_i points in the direction
//     of the strong bond → |z| > 0, arg(z) = θ_d.
//
// This is the inverse of zToRawBonds (Step 4).
//
// =============================================================================
__kernel void bondsToZ(
    __global const int*   atom_bonds,   // (NATOM, 3) bond indices, -1 if absent
    __global const int*   bond_dir,     // (NBOND) direction 0,1,2
    __global const float* bond_x,       // (NBOND) projected π bond order x_b ∈ [0,1]
    __global float* z_real,             // (NATOM) output: Re(z_i)
    __global float* z_imag              // (NATOM) output: Im(z_i)
) {
    int i = get_global_id(0);
    if (i >= NATOM) return;

    // Pass 1: compute mean bond order x̄_i = (1/n_i) Σ x_b
    float xavg = 0.0f;
    int nb = 0;
    for (int d = 0; d < 3; d++) {
        int b = atom_bonds[i * 3 + d];
        if (b < 0) continue;
        xavg += bond_x[b];
        nb++;
    }

    // Isolated atom (no bonds) → z = 0
    if (nb == 0) {
        z_real[i] = 0.0f;
        z_imag[i] = 0.0f;
        return;
    }

    xavg /= (float)nb;

    // Pass 2: z_i = Σ (x_b - x̄_i) · exp(i·θ_{dir(b)})
    //         = Σ (x_b - x̄_i) · [cos(θ_d) + i·sin(θ_d)]
    float zr = 0.0f, zi = 0.0f;
    for (int d = 0; d < 3; d++) {
        int b = atom_bonds[i * 3 + d];
        if (b < 0) continue;
        int dir = bond_dir[b];
        float dx = bond_x[b] - xavg;               // deviation from mean
        float ca = cos(thetaDir[dir]);              // Re(ω_d)
        float sa = sin(thetaDir[dir]);              // Im(ω_d)
        zr += dx * ca;                              // accumulate Re(z)
        zi += dx * sa;                              // accumulate Im(z)
    }

    z_real[i] = zr;
    z_imag[i] = zi;
}

// =============================================================================
// Step 2: applyPinsToZ — Hard defect suppression and phase pinning
// =============================================================================
//
// Modify z_i before time evolution to impose boundary/defect conditions.
//
// Defect suppression (e.g. H+ protonation, sp³ conversion):
//   z_i ← (1 - defect_i) · z_i
//
//   A fully protonated carbon (defect=1) has no π electron, so z_i → 0.
//   Partial defects scale z proportionally.
//
// Phase pinning (fix the Kekulé pattern at a site):
//   z_i ← (1 - p_i) · z_i + p_i · A_pin · exp(i·φ_pin)
//
//   This blends the current z toward a target z_pin = A_pin·e^{iφ_pin}.
//   p_i = 1 means hard pin; p_i = 0 means no pinning.
//   The pin phase φ_pin = θ_dir[d] selects which Kekulé pattern to enforce,
//   i.e. which bond direction should carry the double bond.
//
// Note: this is a HARD constraint applied before RK4.  A SOFT pinning
// force (k_pin term) is also present in the RHS during RK4 evolution.
//
// =============================================================================
__kernel void applyPinsToZ(
    __global float* z_real,              // (NATOM) in/out: Re(z_i)
    __global float* z_imag,              // (NATOM) in/out: Im(z_i)
    __global const float* defect,        // (NATOM) defect strength ∈ [0,1]
    __global const float* pinStrength,   // (NATOM) pin strength p_i ∈ [0,1]
    __global const float* pinPhase,      // (NATOM) pin phase φ_pin (radians)
    __global const Params* params        // scalar parameters
) {
    int i = get_global_id(0);
    if (i >= NATOM) return;

    // Defect suppression: z *= (1 - defect)
    float def = defect[i];
    float zr = z_real[i] * (1.0f - def);
    float zi = z_imag[i] * (1.0f - def);

    // Phase pinning: blend toward z_pin = A_pin · exp(i·φ_pin)
    float p = pinStrength[i];
    if (p > 0.0f) {
        float A_pin = params[0].A_pin;
        float phi = pinPhase[i];
        float zpr = A_pin * cos(phi);    // Re(z_pin)
        float zpi = A_pin * sin(phi);    // Im(z_pin)
        zr = (1.0f - p) * zr + p * zpr;  // linear blend
        zi = (1.0f - p) * zi + p * zpi;
    }

    z_real[i] = zr;
    z_imag[i] = zi;
}

// =============================================================================
// Step 3: rhsZ — Compute right-hand side of the Ginzburg–Landau equation
// =============================================================================
//
// Evaluates dz_i/dt = -η·g_i - i·Ω·g_i, where g_i is the complex "force":
//
//   g_i = κ·∇²z_i + r·z_i + u·|z_i|²·z_i - λ·(z_i*)²
//         + k_core·defect_i·z_i + k_pin·p_i·(z_i - z_i^pin)
//
// Term-by-term physics:
//
//   κ·∇²z_i     — Graph Laplacian: ∇²z_i = Σ_{j∈N(i)} (z_i - z_j).
//                 Penalizes spatial variation of Kekulé order.
//                 Acts as surface tension for domain walls / nodal lines.
//
//   r·z_i       — Linear term.  r < 0 means the free energy F = r|z|² + ...
//                 is minimized at |z| > 0, driving spontaneous Kekulé
//                 dimerization (analogous to r<0 in Landau theory).
//
//   u·|z|²·z    — Cubic saturation.  Prevents |z| → ∞.
//                 Free energy: (r|z|² + u|z|⁴/2) → minimum at |z|² = -r/u.
//                 With r=-1, u=1: |z|_eq = 1.
//
//   -λ·(z*)²    — Z₃ anisotropy (Kekulé locking).  This term is the derivative
//                 of the free-energy term -λ·Re(z³) = -λ·|z|³·cos(3φ),
//                 which creates three degenerate minima at arg(z) = 0, 2π/3, 4π/3,
//                 corresponding to the three Kekulé dimer coverings.
//
//                 Derivation: F_aniso = -λ·Re(z³) = -λ·(z³ + z*³)/2
//                 ∂F/∂z* = -(3λ/2)·(z*)²
//
//                 The code absorbs the factor 3/2 into λ, so the force term
//                 is simply -λ·(z*)².
//
//                 cos(3φ) has minima at φ = 0, 2π/3, 4π/3 (mod 2π/3),
//                 which is exactly the Z₃ symmetry of the Kekulé patterns.
//
//   k_core·d·z  — Defect core suppression.  At protonated/removed sites
//                 (defect=1), this adds a positive restoring force toward
//                 z=0, suppressing Kekulé order locally.
//
//   k_pin·p·(z - z_pin) — Soft phase pinning.  Pulls z toward z_pin
//                 with stiffness k_pin·p.  Complements the hard pinning
//                 in applyPinsToZ.
//
// The time derivative combines dissipative and conservative parts:
//
//   dz/dt = -η·g  (dissipative: relaxes to free-energy minimum)
//         - i·Ω·g (conservative: Schrödinger-like phase rotation)
//
// Expanding:  (-η - iΩ)·(gr + i·gi) = (-η·gr + Ω·gi) + i·(-η·gi - Ω·gr)
//
// So:  Re(dz/dt) = -η·gr + Ω·gi
//      Im(dz/dt) = -η·gi - Ω·gr
//
// This kernel is called 4 times per RK4 step with different z arrays.
//
// =============================================================================
__kernel void rhsZ(
    __global const float* z_real_in,     // (NATOM) input Re(z) for this RK4 stage
    __global const float* z_imag_in,     // (NATOM) input Im(z)
    __global float* rhs_real,            // (NATOM) output: Re(dz/dt)
    __global float* rhs_imag,            // (NATOM) output: Im(dz/dt)
    __global const int*   neighbors,     // (NATOM, 3) neighbor atom indices, -1 if absent
    __global const float* defect,        // (NATOM) defect strength
    __global const float* pinStrength,   // (NATOM) pin strength
    __global const float* pinPhase,      // (NATOM) pin phase
    __global const Params* params        // scalar parameters
) {
    int i = get_global_id(0);
    if (i >= NATOM) return;

    // Load parameters
    float kappa  = params[0].kappa;
    float r      = params[0].r;
    float u      = params[0].u;
    float lambda = params[0].lambda;
    float eta    = params[0].eta;
    float Omega  = params[0].Omega;
    float k_pin  = params[0].k_pin;
    float k_core = params[0].k_core;

    float zr = z_real_in[i];
    float zi = z_imag_in[i];

    // --- Graph Laplacian: ∇²z_i = Σ_{j∈N(i)} (z_i - z_j) ---
    float lap_r = 0.0f, lap_i = 0.0f;
    for (int d = 0; d < 3; d++) {
        int j = neighbors[i * 3 + d];
        if (j < 0) continue;
        lap_r += zr - z_real_in[j];
        lap_i += zi - z_imag_in[j];
    }

    // --- |z|² ---
    float z2 = zr * zr + zi * zi;

    // --- Compute g = κ·∇²z + r·z + u·|z|²·z - λ·(z*)² + k_core·d·z ---
    //
    // (z*)² = (zr - i·zi)² = (zr² - zi²) - i·(2·zr·zi)
    // So:  Re[(z*)²] = zr² - zi²
    //      Im[(z*)²] = -2·zr·zi
    // And: -λ·(z*)² has  Re = -λ·(zr² - zi²)
    //                      Im = -λ·(-2·zr·zi) = +2λ·zr·zi
    float def = defect[i];
    float gr = kappa * lap_r                           // κ·∇²z (real)
             + r * zr                                  // r·z  (real)
             + u * z2 * zr                             // u·|z|²·z (real)
             - lambda * (zr * zr - zi * zi)            // -λ·Re[(z*)²]
             + k_core * def * zr;                      // k_core·defect·z (real)

    float gi = kappa * lap_i                           // κ·∇²z (imag)
             + r * zi                                  // r·z  (imag)
             + u * z2 * zi                             // u·|z|²·z (imag)
             - lambda * (-2.0f * zr * zi)              // -λ·Im[(z*)²] = +2λ·zr·zi
             + k_core * def * zi;                      // k_core·defect·z (imag)

    // --- Soft pinning: g += k_pin·p·(z - z_pin) ---
    float p = pinStrength[i];
    if (p > 0.0f) {
        float A_pin = params[0].A_pin;
        float phi = pinPhase[i];
        float zpr = A_pin * cos(phi);                  // Re(z_pin)
        float zpi = A_pin * sin(phi);                  // Im(z_pin)
        gr += k_pin * p * (zr - zpr);
        gi += k_pin * p * (zi - zpi);
    }

    // --- dz/dt = (-η - iΩ)·g ---
    // (-η - iΩ)·(gr + i·gi) = -η·gr + Ω·gi + i·(-η·gi - Ω·gr)
    rhs_real[i] = -eta * gr + Omega * gi;
    rhs_imag[i] = -eta * gi - Omega * gr;
}

// =============================================================================
// Step 3a: rk4IntermediateHalf — z_temp = z0 + (dt/2)·k
// =============================================================================
//
// Used for RK4 stages k2 and k3:
//   k1 = rhs(z0)
//   k2 = rhs(z0 + 0.5·dt·k1)    ← this kernel
//   k3 = rhs(z0 + 0.5·dt·k2)    ← this kernel
//   k4 = rhs(z0 + dt·k3)         ← rk4Intermediate (full dt)
//
// =============================================================================
__kernel void rk4IntermediateHalf(
    __global const float* z0_real,       // (NATOM) Re(z0) — saved initial state
    __global const float* z0_imag,       // (NATOM) Im(z0)
    __global const float* k_real,        // (NATOM) Re(k) — RHS from previous stage
    __global const float* k_imag,        // (NATOM) Im(k)
    __global float* z_out_real,          // (NATOM) output: z0 + 0.5·dt·k
    __global float* z_out_imag,          // (NATOM)
    __global const Params* params
) {
    int i = get_global_id(0);
    if (i >= NATOM) return;

    float half_dt = params[0].dtA * 0.5f;
    z_out_real[i] = z0_real[i] + half_dt * k_real[i];
    z_out_imag[i] = z0_imag[i] + half_dt * k_imag[i];
}

// =============================================================================
// Step 3a-full: rk4Intermediate — z_temp = z0 + dt·k
// =============================================================================
//
// Used for RK4 stage k4:
//   k4 = rhs(z0 + dt·k3)   ← this kernel
//
// =============================================================================
__kernel void rk4Intermediate(
    __global const float* z0_real,       // (NATOM) Re(z0)
    __global const float* z0_imag,       // (NATOM) Im(z0)
    __global const float* k_real,        // (NATOM) Re(k)
    __global const float* k_imag,        // (NATOM) Im(k)
    __global float* z_out_real,          // (NATOM) output: z0 + dt·k
    __global float* z_out_imag,          // (NATOM)
    __global const Params* params
) {
    int i = get_global_id(0);
    if (i >= NATOM) return;

    float dt = params[0].dtA;
    z_out_real[i] = z0_real[i] + dt * k_real[i];
    z_out_imag[i] = z0_imag[i] + dt * k_imag[i];
}

// =============================================================================
// Step 3b: rk4Combine — Final RK4 combination
// =============================================================================
//
//   z^{n+1} = z0 + dt·(k1 + 2·k2 + 2·k3 + k4) / 6
//
// This is the standard 4th-order Runge–Kutta weighted average.
// RK4 is used instead of Euler because the -i·Ω·g term is oscillatory
// and requires accurate phase evolution.
//
// =============================================================================
__kernel void rk4Combine(
    __global const float* z0_real,       // (NATOM) Re(z0) — initial state at t_n
    __global const float* z0_imag,       // (NATOM) Im(z0)
    __global const float* k1_real,       // (NATOM) Re(k1) = Re(rhs(z0))
    __global const float* k1_imag,       // (NATOM) Im(k1)
    __global const float* k2_real,       // (NATOM) Re(k2) = Re(rhs(z0+0.5dt·k1))
    __global const float* k2_imag,       // (NATOM) Im(k2)
    __global const float* k3_real,       // (NATOM) Re(k3) = Re(rhs(z0+0.5dt·k2))
    __global const float* k3_imag,       // (NATOM) Im(k3)
    __global const float* k4_real,       // (NATOM) Re(k4) = Re(rhs(z0+dt·k3))
    __global const float* k4_imag,       // (NATOM) Im(k4)
    __global float* z_out_real,          // (NATOM) output: z at t_{n+1}
    __global float* z_out_imag,          // (NATOM)
    __global const Params* params
) {
    int i = get_global_id(0);
    if (i >= NATOM) return;

    float dt = params[0].dtA;
    float dt6 = dt / 6.0f;

    // z^{n+1} = z0 + (dt/6)·(k1 + 2k2 + 2k3 + k4)
    float r = z0_real[i] + dt6 * (k1_real[i] + 2.0f * k2_real[i] + 2.0f * k3_real[i] + k4_real[i]);
    float im = z0_imag[i] + dt6 * (k1_imag[i] + 2.0f * k2_imag[i] + 2.0f * k3_imag[i] + k4_imag[i]);

    z_out_real[i] = r;
    z_out_imag[i] = im;
}

// =============================================================================
// Step 4: zToRawBonds — Convert z back to desired (raw) bond orders
// =============================================================================
//
// Given the evolved complex order parameter z_i, compute the desired bond
// orders x_b^raw before valence projection.
//
// For each bond b = (i, j) with direction d:
//
//   z_b = ½·(z_i + z_j)                    (bond-averaged order parameter)
//   A_b = min(|z_b|, 1)                     (amplitude, clamped to [0,1])
//   φ_b = arg(z_b)                          (phase)
//   a_b = kekule_amp·min(x_b^0, 1 - x_b^0)  (max alternation amplitude)
//
//   x_b^raw = x_b^0 + a_b·A_b·cos(φ_b - θ_d)
//
// where x_b^0 = bondBase[b] is the uniform/aromatic baseline.
//
// Physics: the cos(φ_b - θ_d) factor projects the Kekulé phase onto the
// bond direction.  When arg(z_b) = θ_d, bond b is maximally strengthened
// (x_b > x_b^0); when arg(z_b) = θ_d + π, it is maximally weakened.
// The three bonds around an atom see cos(φ - θ_0), cos(φ - θ_1), cos(φ - θ_2),
// which sum to zero — so the valence constraint is approximately satisfied
// even before projection.
//
// Result is clamped to [0, 1] (physical bond order range).
//
// =============================================================================
__kernel void zToRawBonds(
    __global const int*   bond_iA,       // (NBOND) A-sublattice atom index
    __global const int*   bond_iB,       // (NBOND) B-sublattice atom index
    __global const int*   bond_dir,      // (NBOND) direction {0,1,2}
    __global const float* z_real,        // (NATOM) Re(z_i) after RK4
    __global const float* z_imag,        // (NATOM) Im(z_i)
    __global const float* bondBase,      // (NBOND) baseline bond order x_b^0
    __global float* bond_xRaw,           // (NBOND) output: raw bond order
    __global const Params* params
) {
    int b = get_global_id(0);
    if (b >= NBOND) return;

    int i = bond_iA[b];
    int j = bond_iB[b];
    int d = bond_dir[b];

    // Bond-averaged order parameter: z_b = ½·(z_i + z_j)
    float zr = 0.5f * (z_real[i] + z_real[j]);
    float zi = 0.5f * (z_imag[i] + z_imag[j]);

    // Amplitude A_b = min(|z_b|, 1)
    float A = sqrt(zr * zr + zi * zi);
    if (A > 1.0f) A = 1.0f;

    // Phase φ_b = arg(z_b)
    float phi = atan2(zi, zr);

    // Baseline and max alternation amplitude
    float x0 = bondBase[b];
    float amp = params[0].kekule_amp * fmin(x0, 1.0f - x0);

    // Raw bond order: x_b^0 + a_b·A_b·cos(φ_b - θ_d)
    float xraw = x0 + amp * A * cos(phi - thetaDir[d]);

    // Clamp to physical range [0, 1]
    if (xraw < 0.0f) xraw = 0.0f;
    if (xraw > 1.0f) xraw = 1.0f;

    bond_xRaw[b] = xraw;
}

// =============================================================================
// Step 5: copyRawToX — Copy raw bond orders to working array
// =============================================================================
//
//   x_b ← x_b^raw
//
// Simple copy.  The raw bond orders from zToRawBonds become the starting
// point for the valence projection (Step 6).
//
// =============================================================================
__kernel void copyRawToX(
    __global float* bond_x,              // (NBOND) output: working bond orders
    __global const float* bond_xRaw      // (NBOND) input: raw bond orders
) {
    int b = get_global_id(0);
    if (b >= NBOND) return;
    bond_x[b] = bond_xRaw[b];
}

// =============================================================================
// Step 6a: copyBonds — Copy all bond orders from src to dst (race-free)
// =============================================================================
//
// Used before each sublattice projection pass to initialize the output buffer
// with the current values.  Bonds not touched by the active sublattice retain
// their input values.
//
// =============================================================================
__kernel void copyBonds(
    __global const float* bond_x_src,  // (NBOND) input
    __global float* bond_x_dst         // (NBOND) output
) {
    int b = get_global_id(0);
    if (b >= NBOND) return;
    bond_x_dst[b] = bond_x_src[b];
}

// =============================================================================
// Step 6b: projectBondOrdersSub — Race-free valence projection (one sublattice)
// =============================================================================
//
// Enforce the local valence (dimer) constraint:
//
//   Σ_{b ∈ N(i)} x_b = targetVal_i   for each atom i
//
// This is the "Pauli incompressibility" of the model: each atom has a fixed
// π electron budget, and bond orders must sum to that budget.
//
// --- Race-free strategy: bipartite red-black (ping-pong) ---
//
// The honeycomb graph is bipartite: every bond connects one A-sublattice
// atom (sub=+1) to one B-sublattice atom (sub=-1).  Therefore:
//   - No two A atoms share a bond → A atoms can write bonds in parallel
//   - No two B atoms share a bond → B atoms can write bonds in parallel
//
// We exploit this with a ping-pong buffer scheme:
//
//   Pass 1 (A sublattice):
//     1a. copyBonds:  bond_x_out = bond_x_in          (all bonds copied)
//     1b. project_A:  bond_x_out += corrections        (only A atoms write)
//     - Every bond has exactly one A endpoint → every bond is written at most once
//     - Bonds not connected to any A atom (should not exist on honeycomb) keep input value
//     - No race condition
//
//   Pass 2 (B sublattice):
//     2a. copyBonds:  bond_x_in = bond_x_out           (all bonds copied)
//     2b. project_B:  bond_x_in += corrections         (only B atoms write)
//     - Same logic for B sublattice
//     - No race condition
//
// This is red-black Gauss-Seidel: the B pass sees the already-corrected
// values from the A pass, giving faster convergence than plain Jacobi.
//
// --- Update rule (per atom i of the active sublattice) ---
//
//   s_i = Σ_{b ∈ N(i)} x_b              (current sum, read from input buffer)
//   corr_i = (targetVal_i - s_i) / n_i  (correction per bond)
//   x_b ← clamp(x_b + omega_proj·corr_i, 0, 1)   for each b ∈ N(i)
//
// omega_proj (default 0.5) is the relaxation factor.  The ½ under-relaxation
// accounts for each bond being shared by two atoms: both endpoints
// (A in pass 1, B in pass 2) contribute a partial correction.
//
// =============================================================================
__kernel void projectBondOrdersSub(
    __global const float* bond_x_in,    // (NBOND) input bond orders (read-only)
    __global float* bond_x_out,         // (NBOND) output bond orders (write-only)
    __global const int*   atom_bonds,   // (NATOM, 3) bond indices per atom
    __global const float* targetVal,    // (NATOM) target π valence per atom
    __global const int*   sub,          // (NATOM) sublattice: +1=A, -1=B
    int target_sub,                      // +1 to process A atoms, -1 for B atoms
    float omega_proj                     // relaxation factor (0.5 under-relaxed, 1.0 exact)
) {
    int i = get_global_id(0);
    if (i >= NATOM) return;

    // Skip atoms not in the target sublattice (red-black sweep)
    if (sub[i] != target_sub) return;

    // Compute current valence sum s_i = Σ x_b (from input buffer) and bond count
    float s = 0.0f;
    int nb = 0;
    for (int d = 0; d < 3; d++) {
        int b = atom_bonds[i * 3 + d];
        if (b < 0) continue;
        s += bond_x_in[b];
        nb++;
    }
    if (nb == 0) return;

    // Per-bond correction: corr = (target - current_sum) / n_bonds
    float target = targetVal[i];
    float corr = (target - s) / (float)nb;

    // Write corrected bond values to output buffer (race-free: no two atoms
    // of the same sublattice share a bond)
    for (int d = 0; d < 3; d++) {
        int b = atom_bonds[i * 3 + d];
        if (b < 0) continue;
        float newx = bond_x_in[b] + omega_proj * corr;
        if (newx < 0.0f) newx = 0.0f;    // clamp to physical range
        if (newx > 1.0f) newx = 1.0f;
        bond_x_out[b] = newx;
    }
}
