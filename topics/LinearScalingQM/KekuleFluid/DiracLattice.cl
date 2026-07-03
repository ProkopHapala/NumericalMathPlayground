// =============================================================================
// DiracLattice.cl — Tight-binding π-electron propagation on a honeycomb lattice
//
// =============================================================================
// PHYSICAL OVERVIEW
// =============================================================================
//
// This file implements Model B: the tight-binding Schrödinger equation for a
// single π electron on a honeycomb graph (e.g. a PAH molecule or graphene
// flake).  The Hamiltonian is constructed from the Kekulé bond orders
// produced by Model A (kekule_fluid.cl), making the π-electron spectrum
// sensitive to the Kekulé texture.
//
// --- Physical motivation ---
//
// Each sp² carbon atom contributes one p_z orbital to the π system.  In a
// tight-binding (Hückel) approximation, the π-electron Hamiltonian is:
//
//   H = Σ_i  ε_i |i⟩⟨i|  +  Σ_{⟨ij⟩}  t_ij |i⟩⟨j|
//
// where:
//   ε_i  = onsite energy at atom i (zero for normal sp² carbon, large for
//          protonated/defect sites where the p_z orbital is removed)
//   t_ij = hopping integral between nearest-neighbor atoms i and j
//
// --- Kekulé modulation of hopping ---
//
// In a uniform (aromatic) system, all hoppings are equal: t_ij = t₀.
// The Kekulé bond order x_ij from Model A measures the π-electron density
// on bond ⟨ij⟩.  Stronger bonds (larger x_ij) have larger hopping:
//
//   t_ij = t₀ + δt · (x_ij - x₀_ij)
//
// where x₀_ij = bondBase is the uniform/aromatic baseline bond order and
// δt = dt_kek is the Kekulé coupling strength.
//
// In a perfect Kekulé pattern, bonds alternate between strong (x≈1) and
// weak (x≈0), so t_ij alternates between t₀+δt·(1-x₀) and t₀-δt·x₀.
// This dimerization opens a gap at the Dirac points — the "Kekulé mass."
//
// --- Defect onsite potential ---
//
// When a carbon is protonated (H+ addition), its p_z orbital is removed
// (converted to sp³).  We model this with a large onsite potential:
//
//   ε_i = V_defect · defect_i
//
// where defect_i ∈ [0,1] and V_defect is the defect potential strength.
// For defect_i = 1, the atom is effectively removed from the π system.
//
// --- Time evolution ---
//
// The time-dependent Schrödinger equation:
//
//   i ℏ ∂ψ/∂t = H ψ
//
// In natural units (ℏ = 1):
//
//   i ∂ψ/∂t = H ψ   →   ∂ψ/∂t = -i H ψ
//
// Splitting ψ = ψ_r + i ψ_i and H = H_r (real, since t_ij and ε_i are real):
//
//   H ψ = H_r (ψ_r + i ψ_i) = H_r ψ_r + i H_r ψ_i
//
//   ∂ψ_r/∂t = Re(-i H ψ) = Re(-i H_r ψ_r + H_r ψ_i) = H_r ψ_i
//   ∂ψ_i/∂t = Im(-i H ψ) = Im(-i H_r ψ_r + H_r ψ_i) = -H_r ψ_r
//
// So:
//   dψ_r/dt =  Σ_j t_ij ψ_i_j + ε_i ψ_i_i    (= Im(Hψ))
//   dψ_i/dt = -Σ_j t_ij ψ_r_j - ε_i ψ_r_i    (= -Re(Hψ))
//
// Integration uses 4th-order Runge-Kutta (RK4) for accuracy of the
// oscillatory phase evolution.
//
// --- Connection to continuum Dirac equation ---
//
// Near the K and K' points of the Brillouin zone, the tight-binding
// Hamiltonian reduces to the 2-component Dirac Hamiltonian:
//
//   H_D = v_F σ·k + Δ(r) σ_z
//
// where v_F is the Fermi velocity, σ are Pauli matrices acting on the
// A/B sublattice space, and Δ(r) is the Kekulé mass gap proportional to
// the Kekulé order parameter z(r).  Vortices in z(r) bind topological
// zero modes (Jackiw-Rossi states), which appear as in-gap states in the
// lattice model.
//
// =============================================================================
// STATE VARIABLES (buffers)
// =============================================================================
//
// psi_r[NATOM], psi_i[NATOM]  — complex wavefunction ψ_i = ψ_r + i·ψ_i
//   Physical meaning: amplitude of the π-electron wavefunction on atom i.
//   |ψ_i|² is the probability density on atom i.
//   Σ_i |ψ_i|² = 1 (normalization, conserved by unitary evolution).
//   Dimensions: NATOM (one per atom).
//
// bond_x[NBOND]               — π bond order x_ij from Model A
//   Physical meaning: fraction of π-electron density on bond ⟨ij⟩.
//   Modulates the hopping: t_ij = t₀ + δt·(x_ij - x₀_ij).
//   Dimensions: NBOND (one per bond).
//
// bondBase[NBOND]             — baseline (aromatic) bond order x₀_ij
//   Physical meaning: uniform bond order before Kekulé dimerization.
//   Computed as 0.5·(targetVal_i/deg_i + targetVal_j/deg_j).
//   Dimensions: NBOND.
//
// defect[NATOM]               — defect strength ∈ [0,1]
//   Physical meaning: 0 = normal sp² carbon, 1 = fully protonated (sp³).
//   Sets the onsite potential: ε_i = V_defect · defect_i.
//   Dimensions: NATOM.
//
// neighbors[NATOM×3]          — neighbor atom indices (-1 if absent)
//   Used to iterate over nearest neighbors in the Hamiltonian.
//   Dimensions: NATOM × 3 (max 3 neighbors per atom on honeycomb).
//
// atom_bonds[NATOM×3]         — bond indices incident to each atom (-1 if absent)
//   Used to look up the bond order x_ij for each neighbor direction.
//   Dimensions: NATOM × 3.
//
// =============================================================================
// PARAMETERS (DiracParams struct)
// =============================================================================
//
// t0      = 1.0    baseline hopping integral (sets energy unit)
// dt_kek  = 0.5    Kekulé coupling strength (δt in t_ij = t₀ + δt·Δx)
// dt      = 0.05   time step for RK4 integration
//
// Note: V_def is NOT in the struct — the defect array is pre-scaled by V_def
// on the host side (defect_arr *= V_def before upload).  This keeps the
// kernel simple and allows different defect types with different potentials.
// V_def should be >> bandwidth (~6*t0) to truly remove a site from the π system.
//
// =============================================================================
// RK4 BUFFER MANAGEMENT (race-free)
// =============================================================================
//
// The RK4 integration uses separate buffers for each stage's RHS (k1..k4)
// and a temporary buffer (tmp) for the saved initial state ψ₀:
//
//   k1 = rhs(ψ)          → reads b_psi, writes b_k1
//   ψ* = ψ₀ + ½dt·k1    → reads b_tmp + b_k1, writes b_psi
//   k2 = rhs(ψ*)         → reads b_psi, writes b_k2
//   ψ* = ψ₀ + ½dt·k2    → reads b_tmp + b_k2, writes b_psi
//   k3 = rhs(ψ*)         → reads b_psi, writes b_k3
//   ψ* = ψ₀ + dt·k3     → reads b_tmp + b_k3, writes b_psi
//   k4 = rhs(ψ*)         → reads b_psi, writes b_k4
//   ψ_new = ψ₀ + dt·(k1+2k2+2k3+k4)/6  → reads b_tmp + all k's, writes b_psi
//
// Each kernel reads from one set of buffers and writes to a disjoint set.
// The OpenCL in-order queue guarantees that each kernel completes before
// the next starts.  No race conditions.
//
// =============================================================================

#ifndef NATOM
#define NATOM 0
#endif
#ifndef NBOND
#define NBOND 0
#endif

// Parameters struct — must match the float32 packing in DiracLattice_ocl.py
typedef struct {
    float t0;        // baseline hopping integral t₀
    float dt_kek;    // Kekulé coupling strength δt
    float dt;        // time step Δt for RK4
} DiracParams;

// =============================================================================
// Kernel: rhs_psi — Compute right-hand side of i ∂ψ/∂t = H ψ
// =============================================================================
//
// For each atom i, compute (Hψ)_i = ε_i ψ_i + Σ_{j∈N(i)} t_ij ψ_j,
// then convert to time derivative:
//
//   dψ_r/dt =  Im(Hψ) =  ε_i ψ_i_i + Σ_j t_ij ψ_i_j
//   dψ_i/dt = -Re(Hψ) = -ε_i ψ_r_i - Σ_j t_ij ψ_r_j
//
// The hopping is Kekulé-modulated:
//   t_ij = t₀ + δt · (x_ij - x₀_ij)
//
// where x_ij = bond_x[b] and x₀_ij = bondBase[b] for bond b = ⟨i,j⟩.
//
// Input:  psi_r_in, psi_i_in  (read-only, current ψ or RK4 intermediate)
// Output: rhs_r, rhs_i        (write-only, dψ/dt for this RK4 stage)
//
// No race condition: each thread writes to rhs_r[i], rhs_i[i] for its own
// atom i only.  Reads from psi_*_in[j] for neighbors j are read-only.
//
// =============================================================================
__kernel void rhs_psi(
    __global const float* psi_r_in,    // (NATOM) Re(ψ) — input
    __global const float* psi_i_in,    // (NATOM) Im(ψ) — input
    __global float* rhs_r,             // (NATOM) output: Re(dψ/dt) = Im(Hψ)
    __global float* rhs_i,             // (NATOM) output: Im(dψ/dt) = -Re(Hψ)
    __global const int*   neighbors,   // (NATOM, 3) neighbor atom indices, -1 if absent
    __global const int*   atom_bonds,  // (NATOM, 3) bond indices, -1 if absent
    __global const float* bond_x,      // (NBOND) Kekulé bond order from Model A
    __global const float* bondBase,    // (NBOND) baseline (aromatic) bond order
    __global const float* defect,      // (NATOM) onsite potential ε_i (already scaled)
    __global const DiracParams* params
) {
    int i = get_global_id(0);
    if (i >= NATOM) return;

    float t0     = params[0].t0;
    float dt_kek = params[0].dt_kek;

    // Onsite potential: ε_i ψ_i
    float eps = defect[i];
    float Hr = eps * psi_r_in[i];      // Re(Hψ)_i from onsite
    float Hi = eps * psi_i_in[i];      // Im(Hψ)_i from onsite

    // Nearest-neighbor hopping: Σ_j t_ij ψ_j
    for (int d = 0; d < 3; d++) {
        int j = neighbors[i * 3 + d];
        if (j < 0) continue;
        int b = atom_bonds[i * 3 + d];

        // Kekulé-modulated hopping: t_ij = t₀ + δt·(x_ij - x₀_ij)
        float t_ij = t0 + dt_kek * (bond_x[b] - bondBase[b]);

        Hr += t_ij * psi_r_in[j];      // accumulate Re(Hψ)
        Hi += t_ij * psi_i_in[j];      // accumulate Im(Hψ)
    }

    // dψ/dt = -i H ψ
    // Re(-i(Hr + iHi)) = Hi
    // Im(-i(Hr + iHi)) = -Hr
    rhs_r[i] =  Hi;                    // dψ_r/dt = Im(Hψ)
    rhs_i[i] = -Hr;                    // dψ_i/dt = -Re(Hψ)
}

// =============================================================================
// Kernel: rk4_intermediate_half — ψ* = ψ₀ + (Δt/2)·k
// =============================================================================
//
// RK4 stages k2 and k3:
//   k2 = rhs(ψ₀ + ½Δt·k1)   ← this kernel with k = k1
//   k3 = rhs(ψ₀ + ½Δt·k2)   ← this kernel with k = k2
//
// Input:  p0_r, p0_i  (saved initial state ψ₀, read-only)
//         k_r, k_i    (RHS from previous stage, read-only)
// Output: out_r, out_i (intermediate ψ* for next rhs call, write-only)
//
// No race condition: each thread writes only out_r[i], out_i[i].
//
// =============================================================================
__kernel void rk4_intermediate_half(
    __global const float* p0_r,        // (NATOM) Re(ψ₀) — saved initial state
    __global const float* p0_i,        // (NATOM) Im(ψ₀)
    __global const float* k_r,         // (NATOM) Re(k) — RHS from previous stage
    __global const float* k_i,         // (NATOM) Im(k)
    __global float* out_r,             // (NATOM) output: ψ₀ + ½Δt·k
    __global float* out_i,             // (NATOM)
    __global const DiracParams* params
) {
    int i = get_global_id(0);
    if (i >= NATOM) return;

    float half_dt = params[0].dt * 0.5f;
    out_r[i] = p0_r[i] + half_dt * k_r[i];
    out_i[i] = p0_i[i] + half_dt * k_i[i];
}

// =============================================================================
// Kernel: rk4_intermediate_full — ψ* = ψ₀ + Δt·k
// =============================================================================
//
// RK4 stage k4:
//   k4 = rhs(ψ₀ + Δt·k3)   ← this kernel with k = k3
//
// =============================================================================
__kernel void rk4_intermediate_full(
    __global const float* p0_r,        // (NATOM) Re(ψ₀)
    __global const float* p0_i,        // (NATOM) Im(ψ₀)
    __global const float* k_r,         // (NATOM) Re(k)
    __global const float* k_i,         // (NATOM) Im(k)
    __global float* out_r,             // (NATOM) output: ψ₀ + Δt·k
    __global float* out_i,             // (NATOM)
    __global const DiracParams* params
) {
    int i = get_global_id(0);
    if (i >= NATOM) return;

    float dt = params[0].dt;
    out_r[i] = p0_r[i] + dt * k_r[i];
    out_i[i] = p0_i[i] + dt * k_i[i];
}

// =============================================================================
// Kernel: rk4_combine — Final RK4 combination
// =============================================================================
//
//   ψ^{n+1} = ψ₀ + (Δt/6)·(k1 + 2·k2 + 2·k3 + k4)
//
// This is the standard 4th-order Runge-Kutta weighted average.
// RK4 is used because the Schrödinger equation is oscillatory and
// requires accurate phase evolution (Euler would accumulate phase errors).
//
// Input:  p0_r, p0_i  (ψ₀, read-only)
//         k1_r..k4_r, k1_i..k4_i  (four RK4 slopes, read-only)
// Output: out_r, out_i (ψ at t_{n+1}, write-only)
//
// =============================================================================
__kernel void rk4_combine(
    __global const float* p0_r,        // (NATOM) Re(ψ₀)
    __global const float* p0_i,        // (NATOM) Im(ψ₀)
    __global const float* k1_r,        // (NATOM) Re(k1) = Re(rhs(ψ₀))
    __global const float* k1_i,        // (NATOM) Im(k1)
    __global const float* k2_r,        // (NATOM) Re(k2) = Re(rhs(ψ₀+½Δt·k1))
    __global const float* k2_i,        // (NATOM) Im(k2)
    __global const float* k3_r,        // (NATOM) Re(k3) = Re(rhs(ψ₀+½Δt·k2))
    __global const float* k3_i,        // (NATOM) Im(k3)
    __global const float* k4_r,        // (NATOM) Re(k4) = Re(rhs(ψ₀+Δt·k3))
    __global const float* k4_i,        // (NATOM) Im(k4)
    __global float* out_r,             // (NATOM) output: ψ^{n+1}
    __global float* out_i,             // (NATOM)
    __global const DiracParams* params
) {
    int i = get_global_id(0);
    if (i >= NATOM) return;

    float dt = params[0].dt;
    float dt6 = dt / 6.0f;

    // ψ^{n+1} = ψ₀ + (Δt/6)·(k1 + 2k2 + 2k3 + k4)
    out_r[i] = p0_r[i] + dt6 * (k1_r[i] + 2.0f * k2_r[i] + 2.0f * k3_r[i] + k4_r[i]);
    out_i[i] = p0_i[i] + dt6 * (k1_i[i] + 2.0f * k2_i[i] + 2.0f * k3_i[i] + k4_i[i]);
}
