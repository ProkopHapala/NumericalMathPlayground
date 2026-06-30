#!/usr/bin/env python3
"""
test_IsingSolver.py — Benchmark Ising/QUBO solver hierarchy on degenerate systems.

Strategy: start with small systems where exact enumeration is possible,
benchmark cheaper methods (greedy, MFA, SB) against exact, then climb
to larger systems where exact is infeasible.

Test systems (all designed to have degenerate or near-degenerate ground states):
  1. No interaction (V=0) — trivial, catches sign bugs
  2. 1D chain with NN repulsion — exact via DP
  3. 2D square lattice NN repulsion — checkerboard degeneracy
  4. 2D hex lattice NN repulsion — 3-fold degeneracy
  5. Square lattice + weak tip field — phase selection
  6. Random small system — exact enumeration comparison

Usage:
    python test_IsingSolver.py
    python test_IsingSolver.py --N 16 --seed 42
    python test_IsingSolver.py --no-show
"""

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from IsingSolver import (
    compute_energy, compute_fields, greedy_polish,
    mfa_anneal, sb_solve, lanczos_mfa,
    exact_ground_state, exact_all_low_energy,
    CandidatePool, scan_tip_positions,
    spin_coefficients, n_to_spin, spin_to_n,
)


# ---------------------------------------------------------------------------
# Lattice builders
# ---------------------------------------------------------------------------

def build_square_lattice(nx, ny, a=1.0):
    """Square lattice positions and nearest-neighbor edges."""
    pos = np.array([[i * a, j * a] for j in range(ny) for i in range(nx)])
    N = nx * ny
    edges = []
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            if i + 1 < nx:
                edges.append((idx, j * nx + i + 1))  # right
            if j + 1 < ny:
                edges.append((idx, (j + 1) * nx + i))  # up
    return pos, edges, N

def build_hex_lattice(nx, ny, a=1.0):
    """Honeycomb lattice (2 sublattices, 3 NN per site).
    Simple brick-wall representation: rectangular with shifted rows."""
    pos = []
    for j in range(ny):
        for i in range(nx):
            x = i * a + (j % 2) * a * 0.5
            y = j * a * np.sqrt(3) / 2
            pos.append([x, y])
    pos = np.array(pos)
    N = nx * ny
    edges = []
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            if i + 1 < nx:
                edges.append((idx, j * nx + i + 1))  # right
            if j + 1 < ny:
                edges.append((idx, (j + 1) * nx + i))  # up
    # Add diagonal bonds for honeycomb (shifted rows)
    for j in range(ny - 1):
        for i in range(nx):
            idx = j * nx + i
            if j % 2 == 0 and i > 0:
                edges.append((idx, (j + 1) * nx + i - 1))  # down-left
            elif j % 2 == 1 and i + 1 < nx:
                edges.append((idx, (j + 1) * nx + i + 1))  # down-right
    return pos, edges, N

def build_coulomb_matrix(N, edges, V_nn=1.0, long_range=False, pos=None):
    """Build Coulomb interaction matrix.
    V_ii = 0 (no self-interaction), V_ij = V_ji (symmetric).
    For NN-only: V_ij = V_nn for nearest neighbors, 0 otherwise.
    For long-range: V_ij = V_nn / |r_i - r_j| (Coulomb 1/r)."""
    V = np.zeros((N, N))
    for i, j in edges:
        V[i, j] = V_nn
        V[j, i] = V_nn
    if long_range and pos is not None:
        for i in range(N):
            for j in range(i + 1, N):
                r = np.linalg.norm(pos[i] - pos[j])
                if r > 1e-10 and V[i, j] == 0:  # don't override NN
                    V[i, j] = V_nn / r
                    V[j, i] = V_nn / r
    return V

def build_tip_field(N, pos, tip_R, V_tip=0.5, sigma=2.0):
    """AFM tip local field: eps_i = -V_tip * exp(-|r_i - R_tip|^2 / (2*sigma^2)).
    Negative eps attracts electrons (lowers energy when n_i=1).
    The Gaussian profile models the tip's local influence; sigma controls
    the spatial extent (modest sigma preserves long-range correlations)."""
    r = np.linalg.norm(pos - tip_R[None, :], axis=1)
    return -V_tip * np.exp(-r**2 / (2.0 * sigma**2))


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_no_interaction(N=10, seed=42):
    """V=0: ground state is n_i=1 if eps_i<0. Trivial but catches sign bugs."""
    print("\n=== Test 1: No interaction (V=0) ===")
    rng = np.random.RandomState(seed)
    eps = rng.randn(N) * 0.5
    V = np.zeros((N, N))

    n_exact, E_exact = exact_ground_state(eps, V, verbose=True)
    n_expected = (eps < 0).astype(int)
    E_expected = compute_energy(n_expected, eps, V)

    assert np.array_equal(n_exact, n_expected), f"Exact mismatch: {n_exact} vs {n_expected}"
    assert abs(E_exact - E_expected) < 1e-12, f"Energy mismatch: {E_exact} vs {E_expected}"
    print(f"  PASS: exact matches trivial solution, E={E_exact:.6f}")

    # Greedy from random start
    n0 = rng.randint(0, 2, N)
    n_greedy, _ = greedy_polish(n0, eps, V)
    E_greedy = compute_energy(n_greedy, eps, V)
    print(f"  Greedy: E={E_greedy:.6f} ({'PASS' if abs(E_greedy - E_exact) < 1e-10 else 'FAIL'})")

    # MFA
    n_mfa, _, E_mfa = mfa_anneal(eps, V, B=8, seed=seed)
    E_mfa_best = E_mfa.min()
    print(f"  MFA:    E={E_mfa_best:.6f} ({'PASS' if abs(E_mfa_best - E_exact) < 1e-10 else 'FAIL'})")

    # SB
    n_sb, _, E_sb = sb_solve(eps, V, B=8, seed=seed)
    E_sb_best = E_sb.min()
    print(f"  SB:     E={E_sb_best:.6f} ({'PASS' if abs(E_sb_best - E_exact) < 1e-10 else 'FAIL'})")

    return abs(E_greedy - E_exact) < 1e-10 and abs(E_mfa_best - E_exact) < 1e-10 and abs(E_sb_best - E_exact) < 1e-10


def test_fixed_charge_no_interaction(N=10, N_q=5, seed=42):
    """V=0, fixed charge: occupy N_q sites with lowest eps_i."""
    print(f"\n=== Test 2: Fixed-charge no interaction (N={N}, N_q={N_q}) ===")
    rng = np.random.RandomState(seed)
    eps = rng.randn(N) * 0.5
    V = np.zeros((N, N))

    n_exact, E_exact = exact_ground_state(eps, V, ensemble='fixed_charge', N_q=N_q, verbose=True)
    n_expected = np.zeros(N, dtype=int)
    n_expected[np.argsort(eps)[:N_q]] = 1
    E_expected = compute_energy(n_expected, eps, V)

    assert np.array_equal(n_exact, n_expected), f"Exact mismatch"
    assert abs(E_exact - E_expected) < 1e-12
    print(f"  PASS: exact matches trivial fixed-charge solution, E={E_exact:.6f}")

    # Greedy fixed-charge from random start
    n0 = np.zeros(N, dtype=int)
    n0[rng.choice(N, N_q, replace=False)] = 1
    n_greedy, _ = greedy_polish(n0, eps, V, ensemble='fixed_charge')
    E_greedy = compute_energy(n_greedy, eps, V)
    print(f"  Greedy (fixed): E={E_greedy:.6f} ({'PASS' if abs(E_greedy - E_exact) < 1e-10 else 'FAIL'})")

    # MFA fixed-charge
    n_mfa, _, E_mfa = mfa_anneal(eps, V, B=8, ensemble='fixed_charge', N_q=N_q, seed=seed)
    E_mfa_best = E_mfa.min()
    print(f"  MFA (fixed):    E={E_mfa_best:.6f} ({'PASS' if abs(E_mfa_best - E_exact) < 1e-10 else 'FAIL'})")

    return abs(E_greedy - E_exact) < 1e-10 and abs(E_mfa_best - E_exact) < 1e-10


def test_1d_chain(N=16, V_nn=1.0, seed=42):
    """1D chain with NN repulsion: exact via dynamic programming."""
    print(f"\n=== Test 3: 1D chain NN repulsion (N={N}) ===")
    rng = np.random.RandomState(seed)
    eps = rng.randn(N) * 0.1  # weak on-site disorder
    V = np.zeros((N, N))
    for i in range(N - 1):
        V[i, i + 1] = V_nn
        V[i + 1, i] = V_nn

    # Exact via DP: for 1D NN, optimal substructure
    # dp[i][s] = min energy for sites 0..i with site i in state s
    dp = np.full((N, 2), np.inf)
    parent = np.zeros((N, 2), dtype=int)
    dp[0, 0] = eps[0] * 0  # n_0=0
    dp[0, 1] = eps[0] * 1  # n_0=1
    for i in range(1, N):
        for s in range(2):
            for s_prev in range(2):
                E = dp[i - 1, s_prev] + eps[i] * s + V_nn * s * s_prev
                if E < dp[i, s]:
                    dp[i, s] = E
                    parent[i, s] = s_prev
    # Backtrack
    n_dp = np.zeros(N, dtype=int)
    s_best = int(np.argmin(dp[N - 1]))
    E_dp = dp[N - 1, s_best]
    for i in range(N - 1, -1, -1):
        n_dp[i] = s_best
        s_best = parent[i, s_best]
    print(f"  DP: E={E_dp:.10f}, n={n_dp}")

    # Exact enumeration (for small N)
    if N <= 20:
        n_exact, E_exact = exact_ground_state(eps, V, verbose=True)
        assert abs(E_dp - E_exact) < 1e-12, f"DP vs exact: {E_dp} vs {E_exact}"
        print(f"  DP matches exact enumeration")

    # Greedy
    n0 = rng.randint(0, 2, N)
    n_greedy, _ = greedy_polish(n0, eps, V)
    E_greedy = compute_energy(n_greedy, eps, V)
    status = 'PASS' if abs(E_greedy - E_dp) < 1e-10 else f'FAIL (dE={E_greedy - E_dp:.6f})'
    print(f"  Greedy: E={E_greedy:.10f} ({status})")

    # MFA
    n_mfa, _, E_mfa = mfa_anneal(eps, V, B=16, seed=seed)
    E_mfa_best = E_mfa.min()
    status = 'PASS' if abs(E_mfa_best - E_dp) < 1e-10 else f'FAIL (dE={E_mfa_best - E_dp:.6f})'
    print(f"  MFA:    E={E_mfa_best:.10f} ({status})")

    # SB
    n_sb, _, E_sb = sb_solve(eps, V, B=16, seed=seed)
    E_sb_best = E_sb.min()
    status = 'PASS' if abs(E_sb_best - E_dp) < 1e-10 else f'FAIL (dE={E_sb_best - E_dp:.6f})'
    print(f"  SB:     E={E_sb_best:.10f} ({status})")

    return abs(E_greedy - E_dp) < 1e-10 and abs(E_mfa_best - E_dp) < 1e-10 and abs(E_sb_best - E_dp) < 1e-10


def test_square_checkerboard(nx=4, ny=4, V_nn=1.0, seed=42):
    """2D square lattice NN repulsion, zero field: checkerboard degeneracy.
    Two degenerate ground states (A/B sublattice filling at half filling).
    With weak disorder, one is selected."""
    N = nx * ny
    print(f"\n=== Test 4: Square lattice checkerboard (nx={nx}, ny={ny}, N={N}) ===")
    rng = np.random.RandomState(seed)
    pos, edges, N_check = build_square_lattice(nx, ny)
    assert N == N_check
    V = build_coulomb_matrix(N, edges, V_nn=V_nn)
    eps = np.zeros(N)  # no field -> perfect degeneracy

    # Exact
    n_exact, E_exact = exact_ground_state(eps, V, verbose=True)

    # Check it's a checkerboard
    n_checker_A = np.zeros(N, dtype=int)
    n_checker_B = np.zeros(N, dtype=int)
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            if (i + j) % 2 == 0:
                n_checker_A[idx] = 1
            else:
                n_checker_B[idx] = 1
    E_A = compute_energy(n_checker_A, eps, V)
    E_B = compute_energy(n_checker_B, eps, V)
    print(f"  Checkerboard A: E={E_A:.10f}")
    print(f"  Checkerboard B: E={E_B:.10f}")
    assert abs(E_A - E_exact) < 1e-12 and abs(E_B - E_exact) < 1e-12, "Checkerboards should be ground states"
    print(f"  Both checkerboards are ground states (degenerate)")

    # Find all degenerate states
    states = exact_all_low_energy(eps, V, E_threshold=1e-6)
    print(f"  Degenerate ground states: {len(states)}")
    for n_s, E_s in states[:5]:
        print(f"    E={E_s:.10f}, n={n_s}")

    # MFA should find one of them
    n_mfa, m, E_mfa = mfa_anneal(eps, V, B=32, seed=seed)
    E_mfa_best = E_mfa.min()
    status = 'PASS' if abs(E_mfa_best - E_exact) < 1e-10 else f'FAIL (dE={E_mfa_best - E_exact:.6f})'
    print(f"  MFA:    E={E_mfa_best:.10f} ({status})")

    # SB
    n_sb, _, E_sb = sb_solve(eps, V, B=32, seed=seed)
    E_sb_best = E_sb.min()
    status = 'PASS' if abs(E_sb_best - E_exact) < 1e-10 else f'FAIL (dE={E_sb_best - E_exact:.6f})'
    print(f"  SB:     E={E_sb_best:.10f} ({status})")

    # Lanczos on MFA susceptibility (at the MFA fixed point, m~0.5 everywhere)
    # The soft mode should reveal the checkerboard pattern
    m_mean = m[np.argmin(E_mfa)]  # take the best replica's m
    if np.allclose(m_mean, 0.5, atol=0.1):
        E_scale = np.max(np.abs(eps) + np.abs(V).sum(axis=1))
        beta_analysis = 1.0 / E_scale  # moderate beta
        eigvals, eigvecs = lanczos_mfa(m_mean, V, beta_analysis, k=5)
        print(f"  Lanczos soft modes (beta={beta_analysis:.4f}):")
        for i in range(min(5, len(eigvals))):
            v = eigvecs[:, i]
            # Check if soft mode looks like checkerboard
            corr_A = abs(np.dot(v, n_checker_A.astype(float) - 0.5))
            corr_B = abs(np.dot(v, n_checker_B.astype(float) - 0.5))
            print(f"    lambda_{i}={eigvals[i]:.6f}, |<v|A>|={corr_A:.4f}, |<v|B>|={corr_B:.4f}")

    return abs(E_mfa_best - E_exact) < 1e-10 and abs(E_sb_best - E_exact) < 1e-10


def test_tip_phase_selection(nx=4, ny=4, V_nn=1.0, V_tip=0.3, sigma=1.5, seed=42):
    """Square lattice + weak tip field: the tip should select one checkerboard
    phase, not break the pattern. Tests whether the solver finds the correct
    shifted phase rather than creating an unphysical local defect."""
    N = nx * ny
    print(f"\n=== Test 5: Tip phase selection (nx={nx}, ny={ny}, V_tip={V_tip}) ===")
    rng = np.random.RandomState(seed)
    pos, edges, _ = build_square_lattice(nx, ny)
    V = build_coulomb_matrix(N, edges, V_nn=V_nn)

    # Place tip at center
    tip_R = pos.mean(axis=0)
    eps = build_tip_field(N, pos, tip_R, V_tip=V_tip, sigma=sigma)

    n_exact, E_exact = exact_ground_state(eps, V, verbose=True)

    # Check which checkerboard the tip selects
    n_checker_A = np.zeros(N, dtype=int)
    n_checker_B = np.zeros(N, dtype=int)
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            if (i + j) % 2 == 0:
                n_checker_A[idx] = 1
            else:
                n_checker_B[idx] = 1
    # Tip at center should favor the sublattice that has a site at the tip position
    tip_idx = np.argmin(np.linalg.norm(pos - tip_R[None, :], axis=1))
    favored = 'A' if n_checker_A[tip_idx] == 1 else 'B'
    n_favored = n_checker_A if favored == 'A' else n_checker_B
    E_favored = compute_energy(n_favored, eps, V)
    print(f"  Tip at site {tip_idx}, favors checkerboard {favored}: E={E_favored:.10f}")

    # The exact ground state should be one of the two checkerboards (possibly shifted by tip)
    d_A = np.sum(n_exact != n_checker_A)  # Hamming distance to A
    d_B = np.sum(n_exact != n_checker_B)  # Hamming distance to B
    d_min = min(d_A, d_B)
    favored_actual = 'A' if d_A < d_B else 'B'
    print(f"  Exact ground state: closest to checkerboard {favored_actual} (d_H={d_min})")
    if d_min > 0:
        print(f"  (Tip broke the checkerboard pattern — {d_min} sites differ)")
        print(f"  E_exact={E_exact:.10f}, E_favored={E_favored:.10f}, dE={E_favored - E_exact:.10f}")

    # SB
    n_sb, _, E_sb = sb_solve(eps, V, B=32, seed=seed)
    E_sb_best = E_sb.min()
    status = 'PASS' if abs(E_sb_best - E_exact) < 1e-10 else f'FAIL (dE={E_sb_best - E_exact:.6f})'
    print(f"  SB:     E={E_sb_best:.10f} ({status})")

    # MFA
    n_mfa, _, E_mfa = mfa_anneal(eps, V, B=32, seed=seed)
    E_mfa_best = E_mfa.min()
    status = 'PASS' if abs(E_mfa_best - E_exact) < 1e-10 else f'FAIL (dE={E_mfa_best - E_exact:.6f})'
    print(f"  MFA:    E={E_mfa_best:.10f} ({status})")

    return abs(E_sb_best - E_exact) < 1e-10 and abs(E_mfa_best - E_exact) < 1e-10


def test_random_small(N=16, seed=42):
    """Random small system: compare all methods against exact enumeration."""
    print(f"\n=== Test 6: Random small system (N={N}, seed={seed}) ===")
    rng = np.random.RandomState(seed)
    eps = rng.randn(N) * 0.5
    # Random sparse interaction matrix (symmetric, V_ii=0, V_ij>0)
    V = np.zeros((N, N))
    n_edges = N * 2
    for _ in range(n_edges):
        i, j = rng.randint(0, N, 2)
        if i != j:
            v = rng.uniform(0.5, 2.0)
            V[i, j] = v
            V[j, i] = v

    n_exact, E_exact = exact_ground_state(eps, V, verbose=True)

    # Find all low-energy states
    states = exact_all_low_energy(eps, V, E_threshold=0.5)
    print(f"  Low-energy states (dE<0.5): {len(states)}")
    if len(states) >= 2:
        print(f"    E_0={states[0][1]:.10f}, E_1={states[1][1]:.10f}, gap={states[1][1] - states[0][1]:.10f}")

    # Greedy from multiple random starts
    n_greedy_starts = 50
    E_greedy_all = []
    for _ in range(n_greedy_starts):
        n0 = rng.randint(0, 2, N)
        n_g, _ = greedy_polish(n0, eps, V)
        E_greedy_all.append(compute_energy(n_g, eps, V))
    E_greedy_best = min(E_greedy_all)
    status = 'PASS' if abs(E_greedy_best - E_exact) < 1e-10 else f'FAIL (dE={E_greedy_best - E_exact:.6f})'
    print(f"  Greedy ({n_greedy_starts} starts): E={E_greedy_best:.10f} ({status})")

    # MFA
    n_mfa, _, E_mfa = mfa_anneal(eps, V, B=32, seed=seed)
    E_mfa_best = E_mfa.min()
    status = 'PASS' if abs(E_mfa_best - E_exact) < 1e-10 else f'FAIL (dE={E_mfa_best - E_exact:.6f})'
    print(f"  MFA (B=32): E={E_mfa_best:.10f} ({status})")

    # SB
    n_sb, _, E_sb = sb_solve(eps, V, B=32, seed=seed)
    E_sb_best = E_sb.min()
    status = 'PASS' if abs(E_sb_best - E_exact) < 1e-10 else f'FAIL (dE={E_sb_best - E_exact:.6f})'
    print(f"  SB (B=32):  E={E_sb_best:.10f} ({status})")

    # SB discrete
    n_sb_d, _, E_sb_d = sb_solve(eps, V, B=32, seed=seed, discrete=True)
    E_sb_d_best = E_sb_d.min()
    status = 'PASS' if abs(E_sb_d_best - E_exact) < 1e-10 else f'FAIL (dE={E_sb_d_best - E_exact:.6f})'
    print(f"  SB discrete (B=32): E={E_sb_d_best:.10f} ({status})")

    return abs(E_mfa_best - E_exact) < 1e-10 and abs(E_sb_best - E_exact) < 1e-10


def test_tip_scan(nx=4, ny=4, V_nn=1.0, V_tip=0.3, sigma=1.5, n_tips=20, seed=42):
    """Tip scan with candidate pool: track ground state as tip moves across the lattice.
    Check for degeneracy crossings (Coulomb diamond boundaries)."""
    N = nx * ny
    print(f"\n=== Test 7: Tip scan with candidate pool (nx={nx}, ny={ny}, n_tips={n_tips}) ===")
    pos, edges, _ = build_square_lattice(nx, ny)
    V = build_coulomb_matrix(N, edges, V_nn=V_nn)

    # Scan tip along x-axis through center
    y_center = pos[:, 1].mean()
    x_range = np.linspace(pos[:, 0].min() - 0.5, pos[:, 0].max() + 0.5, n_tips)

    def eps_fn(i):
        return build_tip_field(N, pos, np.array([x_range[i], y_center]), V_tip=V_tip, sigma=sigma)

    t0 = time.time()
    results = scan_tip_positions(eps_fn, V, N, n_tips=n_tips, B_sb=8, B_mfa=4, seed=seed, verbose=True)
    t_scan = time.time() - t0
    print(f"  Scan time: {t_scan:.3f}s ({n_tips} positions)")

    # Verify each tip position against exact
    n_correct = 0
    for r in results:
        n_exact, E_exact = exact_ground_state(eps_fn(r['tip_idx']), V)
        if abs(r['E_best'] - E_exact) < 1e-10:
            n_correct += 1
        else:
            print(f"  tip {r['tip_idx']}: E_solver={r['E_best']:.10f}, E_exact={E_exact:.10f}, dE={r['E_best'] - E_exact:.6f}")
    print(f"  Correct: {n_correct}/{n_tips}")

    # Report degeneracy crossings
    gaps = [r['gap'] for r in results]
    small_gaps = [(i, g) for i, g in enumerate(gaps) if g < 0.1]
    if small_gaps:
        print(f"  Near-degeneracies (gap < 0.1): {len(small_gaps)} positions")
        for i, g in small_gaps:
            print(f"    tip {i}: gap={g:.6f}")

    return n_correct == n_tips


def test_benchmark_scaling(seed=42):
    """Benchmark scaling: measure time and accuracy for increasing N.
    Uses square lattice with NN repulsion + weak disorder."""
    print("\n=== Test 8: Benchmark scaling ===")
    rng = np.random.RandomState(seed)
    sizes = [(4, 4), (6, 6), (8, 8), (10, 10)]
    results = []

    for nx, ny in sizes:
        N = nx * ny
        pos, edges, _ = build_square_lattice(nx, ny)
        V = build_coulomb_matrix(N, edges, V_nn=1.0)
        eps = rng.randn(N) * 0.1  # weak disorder to break degeneracy

        # Exact (if feasible)
        E_exact = None
        if N <= 20:
            _, E_exact = exact_ground_state(eps, V)
            print(f"\n  N={N} ({nx}x{ny}): E_exact={E_exact:.10f}")
        else:
            print(f"\n  N={N} ({nx}x{ny}): exact infeasible")

        # Greedy
        t0 = time.time()
        E_greedy_best = np.inf
        for _ in range(20):
            n0 = rng.randint(0, 2, N)
            n_g, _ = greedy_polish(n0, eps, V)
            E_g = compute_energy(n_g, eps, V)
            E_greedy_best = min(E_greedy_best, E_g)
        t_greedy = time.time() - t0

        # MFA
        t0 = time.time()
        n_mfa, _, E_mfa = mfa_anneal(eps, V, B=64, seed=rng.randint(0, 2**31), N_inner=30, beta_factor=1.08)
        t_mfa = time.time() - t0
        E_mfa_best = E_mfa.min()

        # SB
        t0 = time.time()
        n_sb, _, E_sb = sb_solve(eps, V, B=64, seed=rng.randint(0, 2**31), N_steps=1000, discrete=True)
        t_sb = time.time() - t0
        E_sb_best = E_sb.min()

        # Report
        ref = E_exact if E_exact is not None else min(E_greedy_best, E_mfa_best, E_sb_best)
        print(f"    Greedy (20x): E={E_greedy_best:.10f}, dE={E_greedy_best - ref:.2e}, t={t_greedy:.4f}s")
        print(f"    MFA (B=64):    E={E_mfa_best:.10f}, dE={E_mfa_best - ref:.2e}, t={t_mfa:.4f}s")
        print(f"    SB (B=64):     E={E_sb_best:.10f}, dE={E_sb_best - ref:.2e}, t={t_sb:.4f}s")

        results.append({
            'N': N, 'nx': nx, 'ny': ny,
            'E_exact': E_exact, 'E_greedy': E_greedy_best, 'E_mfa': E_mfa_best, 'E_sb': E_sb_best,
            't_greedy': t_greedy, 't_mfa': t_mfa, 't_sb': t_sb,
        })

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(save_path="IsingQUBO_test.png"):
    """Plot test results: ground state visualization and energy comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Panel 1: Square checkerboard ground state
    ax = axes[0, 0]
    nx, ny = 4, 4
    pos, edges, N = build_square_lattice(nx, ny)
    V = build_coulomb_matrix(N, edges, V_nn=1.0)
    eps = np.zeros(N)
    n_exact, _ = exact_ground_state(eps, V)
    colors = ['blue' if n == 1 else 'lightgray' for n in n_exact]
    ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=200, edgecolors='black', zorder=3)
    for i, j in edges:
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 'k-', alpha=0.3)
    ax.set_title(f"Checkerboard ground state ({nx}x{ny})")
    ax.set_aspect('equal')
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # Panel 2: Tip phase selection
    ax = axes[0, 1]
    tip_R = pos.mean(axis=0)
    eps_tip = build_tip_field(N, pos, tip_R, V_tip=0.3, sigma=1.5)
    n_tip, _ = exact_ground_state(eps_tip, V)
    colors = ['red' if n == 1 else 'lightgray' for n in n_tip]
    ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=200, edgecolors='black', zorder=3)
    ax.scatter(*tip_R, c='green', marker='x', s=300, zorder=4, label='tip')
    for i, j in edges:
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 'k-', alpha=0.3)
    ax.set_title("Tip phase selection")
    ax.set_aspect('equal')
    ax.legend()

    # Panel 3: MFA soft mode (Lanczos eigenvector)
    ax = axes[0, 2]
    n_mfa, m, E_mfa = mfa_anneal(eps, V, B=1, seed=42)
    m_mean = m[0]
    E_scale = np.max(np.abs(eps) + np.abs(V).sum(axis=1))
    eigvals, eigvecs = lanczos_mfa(m_mean, V, 1.0 / E_scale, k=4)
    v = eigvecs[:, 0]
    ax.scatter(pos[:, 0], pos[:, 1], c=v, s=200, edgecolors='black', cmap='RdBu_r', zorder=3)
    for i, j in edges:
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 'k-', alpha=0.3)
    ax.set_title(f"Soft mode (lambda={eigvals[0]:.4f})")
    ax.set_aspect('equal')

    # Panel 4: Energy comparison bar chart
    ax = axes[1, 0]
    test_names = ['No V', '1D chain', 'Square', 'Tip', 'Random']
    test_Es = []
    for name, nx, ny, V_nn, eps_fn in [
        ("No V", 4, 4, 0.0, lambda N: np.random.RandomState(42).randn(N) * 0.5),
        ("1D chain", 16, 1, 1.0, lambda N: np.random.RandomState(42).randn(N) * 0.1),
        ("Square", 4, 4, 1.0, lambda N: np.zeros(N)),
        ("Tip", 4, 4, 1.0, lambda N: None),  # special
        ("Random", 16, 1, 1.0, lambda N: np.random.RandomState(42).randn(N) * 0.5),
    ]:
        pass  # placeholder — actual values filled from test runs
    ax.text(0.5, 0.5, "See console output\nfor energy comparison", ha='center', va='center', transform=ax.transAxes)
    ax.set_title("Energy comparison")

    # Panel 5: Tip scan energy profile
    ax = axes[1, 1]
    nx, ny = 4, 4
    pos2, edges2, N2 = build_square_lattice(nx, ny)
    V2 = build_coulomb_matrix(N2, edges2, V_nn=1.0)
    n_tips = 20
    y_center = pos2[:, 1].mean()
    x_range = np.linspace(pos2[:, 0].min() - 0.5, pos2[:, 0].max() + 0.5, n_tips)
    E_scan = []
    gap_scan = []
    for i in range(n_tips):
        eps_i = build_tip_field(N2, pos2, np.array([x_range[i], y_center]), V_tip=0.3, sigma=1.5)
        n_ex, E_ex = exact_ground_state(eps_i, V2)
        E_scan.append(E_ex)
        # Simple gap: difference to second-best from greedy multi-start
        E_second = np.inf
        for _ in range(10):
            n0 = np.random.RandomState(i * 100 + _).randint(0, 2, N2)
            n_g, _ = greedy_polish(n0, eps_i, V2)
            E_g = compute_energy(n_g, eps_i, V2)
            if E_g > E_ex + 1e-12 and E_g < E_second:
                E_second = E_g
        gap_scan.append(E_second - E_ex if E_second < np.inf else 1.0)
    ax.plot(x_range, E_scan, 'b-o', label='E_ground')
    ax2 = ax.twinx()
    ax2.plot(x_range, gap_scan, 'r--s', label='gap')
    ax.set_xlabel("Tip x position")
    ax.set_ylabel("Ground state energy", color='blue')
    ax2.set_ylabel("Energy gap", color='red')
    ax.set_title("Tip scan: energy and gap")

    # Panel 6: Scaling benchmark
    ax = axes[1, 2]
    bench = test_benchmark_scaling(seed=42)
    Ns = [r['N'] for r in bench]
    ax.semilogy(Ns, [r['t_greedy'] for r in bench], 'o-', label='Greedy (20x)')
    ax.semilogy(Ns, [r['t_mfa'] for r in bench], 's-', label='MFA (B=64)')
    ax.semilogy(Ns, [r['t_sb'] for r in bench], '^-', label='SB (B=64)')
    ax.set_xlabel("N (system size)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Scaling benchmark")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved plot to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Test Ising/QUBO solver hierarchy")
    p.add_argument("--N", type=int, default=16, help="Default system size for tests")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--no-show", action="store_true", help="Do not call plt.show()")
    p.add_argument("--save", default="IsingQUBO_test.png", help="Save plot path")
    p.add_argument("--skip-scan", action="store_true", help="Skip tip scan test (slow)")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("=" * 70)
    print("Ising/QUBO Solver Test Suite")
    print("=" * 70)

    all_pass = True

    # Test 1: No interaction
    all_pass &= test_no_interaction(N=10, seed=args.seed)

    # Test 2: Fixed-charge no interaction
    all_pass &= test_fixed_charge_no_interaction(N=10, N_q=5, seed=args.seed)

    # Test 3: 1D chain
    all_pass &= test_1d_chain(N=min(args.N, 16), seed=args.seed)

    # Test 4: Square checkerboard
    all_pass &= test_square_checkerboard(nx=4, ny=4, seed=args.seed)

    # Test 5: Tip phase selection
    all_pass &= test_tip_phase_selection(nx=4, ny=4, seed=args.seed)

    # Test 6: Random small system
    all_pass &= test_random_small(N=min(args.N, 16), seed=args.seed)

    # Test 7: Tip scan
    if not args.skip_scan:
        all_pass &= test_tip_scan(nx=4, ny=4, n_tips=20, seed=args.seed)

    # Test 8: Benchmark scaling
    test_benchmark_scaling(seed=args.seed)

    # Summary
    print("\n" + "=" * 70)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED (see details above)")
    print("=" * 70)

    # Plot
    plot_results(save_path=args.save)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
