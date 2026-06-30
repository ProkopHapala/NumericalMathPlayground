"""
IsingSolver.py — Ising/QUBO solver hierarchy for molecular charge systems.

Minimizes E(n) = eps . n + (1/2) n^T V n, n_i in {0,1}, for 2D molecular
cellular automata with Coulomb repulsion and AFM tip fields.

Hierarchy (fastest → most reliable):
  L0: Greedy local descent (single flips + pair relocations)
  L1: Mean-Field Annealing (Fermi-function self-consistent iteration)
  L2: Lanczos on MFA susceptibility matrix (soft-mode diagnostics)
  L3: Simulated Bifurcation (spin-space nonlinear oscillator dynamics)
  L4: Candidate pool / lower-envelope tracking (scan reuse)
  L5: Exact enumeration (brute force, N <= 24)

Two ensemble modes:
  grand_canonical: N_q = sum n_i is free, single flips allowed
  fixed_charge:    sum n_i = N_q fixed, pair relocations only

References: IsingQUBO.md design document.
"""

import numpy as np
from scipy.sparse.linalg import eigsh


# ---------------------------------------------------------------------------
# Energy and field primitives
# ---------------------------------------------------------------------------

def compute_energy(n, eps, V):
    """E(n) = eps . n + (1/2) n^T V n.  V is dense or sparse (N×N), n is (N,) int."""
    n = n.astype(float)
    return float(eps @ n + 0.5 * n @ (V @ n))

def compute_fields(n, eps, V):
    """g_i = eps_i + sum_j V_ij n_j.  Local field at each site."""
    return eps + V @ n.astype(float)

def flip_cost(g, n):
    """Delta_i = (1 - 2*n_i) * g_i.  Cost to flip site i.
    If n_i=0: Delta = +g_i (cost to charge).  If n_i=1: Delta = -g_i (cost to discharge).
    Negative Delta means the flip lowers energy."""
    return (1.0 - 2.0 * n) * g

def relocate_cost(g, V, i, j):
    """Delta_{i->j} = -g_i + g_j - V_ij.  Cost to move charge from i (occupied) to j (empty).
    Negative means the relocation lowers energy."""
    return -g[i] + g[j] - V[i, j]


# ---------------------------------------------------------------------------
# Spin mapping: n_i in {0,1} <-> s_i = 2*n_i - 1 in {-1,+1}
# ---------------------------------------------------------------------------

def n_to_spin(n):
    """s_i = 2*n_i - 1."""
    return 2.0 * n.astype(float) - 1.0

def spin_to_n(s):
    """n_i = (1 + s_i) / 2, rounded to binary."""
    return ((s + 1.0) / 2.0 > 0.5).astype(int)

def spin_coefficients(eps, V):
    """Map n-space (eps, V) to spin-space (J, h).
    J_ij = -V_ij/4  (antiferromagnetic for repulsive V > 0)
    h_i  = -eps_i/2 - (1/4) sum_j V_ij
    E(s) = -(1/2) s^T J s - h^T s + const
    Assumes V_ii = 0, V_ij = V_ji."""
    n = eps.shape[0]
    J = -0.25 * V
    if hasattr(V, 'toarray'):
        V_dense = V.toarray()
    else:
        V_dense = V
    h = -0.5 * eps - 0.25 * V_dense.sum(axis=1)
    return J, h


# ---------------------------------------------------------------------------
# L0: Greedy local descent
# ---------------------------------------------------------------------------

def greedy_polish(n, eps, V, ensemble='grand_canonical', K_act=32, max_sweeps=1000):
    """Greedy local descent: repeatedly apply the best single flip or pair relocation
    until no improving move exists.

    Parameters
    ----------
    n : (N,) int array — binary occupancy (modified in-place and returned)
    eps : (N,) float — on-site energies
    V : (N,N) dense or sparse — Coulomb matrix (V_ii=0, V_ij=V_ji, V_ij>0)
    ensemble : 'grand_canonical' (single flips allowed) or 'fixed_charge' (relocations only)
    K_act : int — active-set size for pair relocation search (test only K_act×K_act best pairs)
    max_sweeps : int — safety limit on number of accepted moves

    Returns
    -------
    n : (N,) int — polished binary state
    g : (N,) float — local fields at the final state

    Physical motivation:
        At zero temperature, the system relaxes to the nearest local minimum
        by accepting all energy-lowering moves. For warm-start tip tracking,
        this is the fastest way to follow a local basin as the tip moves.

    Stability notes:
        The dn for single flips MUST be computed before modifying n[i].
        Computing (1-2*n[i]) after the flip gives the opposite sign.
        For pair relocation, g update uses original assignments: g += -V[:,i] + V[:,j].
    """
    n = n.copy().astype(int)
    g = compute_fields(n, eps, V)
    N = len(n)

    for _ in range(max_sweeps):
        improved = False

        # --- Single flips (grand-canonical only) ---
        if ensemble == 'grand_canonical':
            deltas = flip_cost(g, n)  # O(N)
            i_min = int(np.argmin(deltas))
            if deltas[i_min] < -1e-14:
                dn = 1 - 2 * n[i_min]  # +1 for 0->1, -1 for 1->0; compute BEFORE flip
                n[i_min] = 1 - n[i_min]
                g = g + V[:, i_min] * dn  # O(N) dense, O(z) sparse
                improved = True
                continue

        # --- Pair relocation (active-set: only test K_act best removals × K_act best insertions) ---
        # Removal cost r_i = -g_i for occupied sites (lower = more favorable to remove)
        # Insertion cost a_j = g_j for empty sites (lower = more favorable to fill)
        # Relocation cost Delta_{i->j} = r_i + a_j - V_ij
        occ = np.where(n == 1)[0]
        emp = np.where(n == 0)[0]
        if len(occ) == 0 or len(emp) == 0:
            break

        r = -g[occ]  # removal costs
        a = g[emp]   # insertion costs
        k_r = min(K_act, len(occ))
        k_a = min(K_act, len(emp))
        idx_r = np.argpartition(r, k_r - 1)[:k_r]  # K_act most favorable removals
        idx_a = np.argpartition(a, k_a - 1)[:k_a]  # K_act most favorable insertions

        best_delta = -1e-14
        best_ij = None
        for ii in idx_r:
            i = occ[ii]
            for jj in idx_a:
                j = emp[jj]
                d = r[ii] + a[jj] - V[i, j]
                if d < best_delta:
                    best_delta = d
                    best_ij = (i, j)

        if best_ij is not None:
            i, j = best_ij
            n[i] = 0
            n[j] = 1
            g = g - V[:, i] + V[:, j]  # original: n[i] was 1, n[j] was 0
            improved = True
            continue

        if not improved:
            break

    return n, g


# ---------------------------------------------------------------------------
# L1: Mean-Field Annealing (MFA)
# ---------------------------------------------------------------------------

def mfa_anneal(eps, V, B=16, ensemble='grand_canonical', N_q=None,
               beta_init=None, beta_max=None, beta_factor=1.1,
               N_inner=20, alpha=0.3, seed=None, verbose=False):
    """Mean-Field Annealing with Fermi-function self-consistent updates.

    Replaces binary n_i in {0,1} with probabilities m_i in [0,1].
    Minimizes the variational free energy:
        F(m) = eps . m + (1/2) m^T V m + T sum_i [m_i ln m_i + (1-m_i) ln(1-m_i)]
    Stationarity gives the self-consistent Fermi update:
        m_i = 1 / (1 + exp(beta * g_i)),  g_i = eps_i + sum_j V_ij m_j

    For fixed_charge, a chemical potential mu is introduced:
        m_i = 1 / (1 + exp(beta * (g_i - mu))),  sum_i m_i = N_q
    mu is found by bisection at each iteration.

    Annealing: start at high T (small beta), converge, then cool (increase beta).
    As beta -> infinity, sigmoid -> step function, m_i -> {0,1}.

    Parameters
    ----------
    B : int — batch size (independent random starts)
    beta_init, beta_max : float — if None, auto-computed from E_scale
    beta_factor : float — geometric cooling rate (1.03-1.15 for fragile bifurcations)
    N_inner : int — iterations per temperature level
    alpha : float — damping factor (0.1-0.5; critical for repulsive couplings to avoid oscillation)

    Returns
    -------
    n_list : (B, N) int — rounded + greedy-polished binary states
    m_list : (B, N) float — final mean-field probabilities
    E_list : (B,) float — energies of polished states

    Physical motivation:
        MFA is a controlled approximation to the Gibbs free energy. It gives
        smooth susceptibility maps and predicts where the system is about to
        bifurcate. However, it can get trapped in fractional states (m_i=0.5)
        for frustrated systems, so it's a candidate generator, not a final solver.

    Stability notes:
        - Damping (alpha < 1) is essential for repulsive V: without it, the
          Fermi update oscillates because a flip at site i changes the field
          at neighbor j, which changes m_j, which changes the field back at i.
        - beta must have units of inverse energy. E_scale = max local field
          magnitude ensures the initial state is near-uniform (m~0.5) and
          the final state is well-discretized.
    """
    rng = np.random.RandomState(seed)
    N = len(eps)

    # Energy scale: max local field magnitude = max(|eps_i| + sum_j |V_ij|)
    if hasattr(V, 'toarray'):
        V_abs_row_sum = np.abs(V.toarray()).sum(axis=1)
    else:
        V_abs_row_sum = np.abs(V).sum(axis=1)
    E_scale = np.max(np.abs(eps) + V_abs_row_sum)
    if E_scale < 1e-30:
        E_scale = 1.0

    if beta_init is None:
        beta_init = 0.01 / E_scale
    if beta_max is None:
        beta_max = 10.0 / E_scale

    # Initialize: near-uniform + small noise to break symmetry
    m = 0.5 + rng.randn(B, N) * 0.01
    m = np.clip(m, 0.01, 0.99)

    beta = beta_init
    while beta < beta_max:
        for _ in range(N_inner):
            g = eps[None, :] + m @ V.T  # (B,N) = (B,N) @ (N,N)^T, O(B*N^2) or O(B*Nz)
            if ensemble == 'fixed_charge':
                # Bisection on mu to enforce sum_i m_i = N_q per replica
                m_new = np.empty_like(m)
                for b in range(B):
                    mu = _bisect_mu(g[b], beta, N_q)
                    m_new[b] = 1.0 / (1.0 + np.exp(beta * (g[b] - mu)))
            else:
                m_new = 1.0 / (1.0 + np.exp(beta * g))  # Fermi-Dirac, O(B*N)
            m = (1.0 - alpha) * m + alpha * m_new  # damped update
        beta *= beta_factor

    # Discretize and polish
    n_list = np.zeros((B, N), dtype=int)
    E_list = np.zeros(B)
    for b in range(B):
        if ensemble == 'fixed_charge' and N_q is not None:
            # Select top-N_q sites by m_i (enforce charge constraint)
            n = np.zeros(N, dtype=int)
            n[np.argpartition(m[b], -N_q)[-N_q:]] = 1
        else:
            n = (m[b] > 0.5).astype(int)
        n, _ = greedy_polish(n, eps, V, ensemble=ensemble)
        n_list[b] = n
        E_list[b] = compute_energy(n, eps, V)

    if verbose:
        print(f"#MFA: B={B}, beta=[{beta_init:.4f},{beta_max:.4f}], E_range=[{E_list.min():.6f},{E_list.max():.6f}]")

    return n_list, m, E_list


def _bisect_mu(g, beta, N_q, mu_lo=None, mu_hi=None, tol=1e-8, max_iter=50):
    """Find mu such that sum_i 1/(1+exp(beta*(g_i-mu))) = N_q.
    Monotone in mu, so bisection is guaranteed to converge."""
    if mu_lo is None:
        mu_lo = g.min() - 10.0 / beta
    if mu_hi is None:
        mu_hi = g.max() + 10.0 / beta
    for _ in range(max_iter):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        n_q = np.sum(1.0 / (1.0 + np.exp(beta * (g - mu_mid))))
        if n_q > N_q:
            mu_hi = mu_mid  # too many occupied -> lower mu to reduce occupancy
        else:
            mu_lo = mu_mid  # too few occupied -> raise mu to increase occupancy
        if abs(mu_hi - mu_lo) < tol:
            break
    return 0.5 * (mu_lo + mu_hi)


# ---------------------------------------------------------------------------
# L2: Lanczos on MFA susceptibility matrix
# ---------------------------------------------------------------------------

def lanczos_mfa(m, V, beta, k=10):
    """Lanczos on the MFA susceptibility matrix A = I + beta * D^{1/2} V D^{1/2}
    to find the softest collective modes (smallest eigenvalues).

    D_i = m_i * (1 - m_i)  — local susceptibility (charge variance).
    D_i -> 0 for frozen sites (m~0 or 1), D_i = 0.25 for active sites (m~0.5).

    The instability occurs when 1 + beta * lambda_min(D^{1/2} V D^{1/2}) -> 0.
    Note: V can have NEGATIVE eigenvalues (off-diagonal repulsion, zero diagonal),
    so the smallest eigenvalues of A correspond to the most negative eigenmodes
    of D^{1/2} V D^{1/2}. These are the "soft modes" — collective charge
    rearrangements that are about to bifurcate.

    Parameters
    ----------
    m : (N,) float — converged MFA probabilities
    V : (N,N) — Coulomb matrix
    beta : float — inverse temperature at which to analyze stability
    k : int — number of eigenvalues to compute

    Returns
    -------
    eigvals : (k,) float — smallest eigenvalues of A
    eigvecs : (N, k) float — corresponding eigenvectors (soft mode patterns)

    Cost: O(k * N^2) or O(k * Nz) for sparse V. k ~ 10-20 is sufficient.
    """
    N = len(m)
    D = m * (1.0 - m)
    D_sqrt = np.sqrt(D)

    # Build the symmetric matrix A = I + beta * D^{1/2} V D^{1/2}
    # For sparse V, we could do this implicitly, but for N <= 1000 dense is fine
    if hasattr(V, 'toarray'):
        V_dense = V.toarray()
    else:
        V_dense = V
    A = np.eye(N) + beta * (D_sqrt[:, None] * V_dense * D_sqrt[None, :])

    # Use scipy's eigsh for smallest eigenvalues
    k = min(k, N - 1)
    eigvals, eigvecs = eigsh(A, k=k, which='SA')  # 'SA' = smallest algebraic
    idx = np.argsort(eigvals)
    return eigvals[idx], eigvecs[:, idx]


# ---------------------------------------------------------------------------
# L3: Simulated Bifurcation (SB)
# ---------------------------------------------------------------------------

def sb_solve(eps, V, B=32, lambda_max=None, N_steps=500, eta=0.05,
             discrete=False, seed=None, verbose=False):
    """Simulated Bifurcation in spin variables.

    Maps n_i in {0,1} to spin s_i = 2*n_i-1 in {-1,+1}, then to continuous
    oscillator x_i in R that undergoes a pitchfork bifurcation.

    Energy in spin-space:
        E_cont(x) = -(1/2) x^T J x - h^T x + lambda * sum_i (x_i^2 - 1)^2
    where J = -V/4, h = -eps/2 - (1/4) sum_j V_ij.

    Force: F_i = -dE/dx_i = sum_j J_ij x_j + h_i - 4*lambda * x_i * (x_i^2 - 1)
    Update: x_i <- x_i + eta * F_i  (gradient descent: x += eta * (-dE/dx))

    At lambda=0: pure gradient flow on Ising energy (like MFA at T=0).
    At lambda > V_crit: double well at x = ±1, forcing discretization.

    Discrete SB variant: force uses sign(x_j) instead of x_j — forces computed
    from the current discrete state, giving better convergence for QUBO.

    Parameters
    ----------
    B : int — batch size (independent random starts for basin exploration)
    lambda_max : float — if None, auto-computed from V spectral range
    N_steps : int — number of time steps (100-1000)
    eta : float — step size (0.01-0.1, may need tuning)
    discrete : bool — use discrete SB (sign(x) in force) vs continuous SB

    Returns
    -------
    n_list : (B, N) int — sign(x) discretized + greedy-polished binary states
    x_list : (B, N) float — final continuous positions
    E_list : (B,) float — energies of polished states

    Physical motivation:
        SB is NOT combinatorial — it operates on one vector x in R^N per replica.
        The pitchfork bifurcation mechanically forces variables toward ±1,
        naturally exploring different basins. The nonlinear x^3 term is a
        physical "enforcer" that pushes variables away from the unstable
        fractional center (unlike MFA which can get stuck at m=0.5).

    Performance:
        Per step: J @ x is O(B*N^2) dense or O(B*Nz) sparse.
        For dense, use batched GEMM (X @ J^T) not repeated GEMV.
        Total: O(N_steps * B * Nz). For N=1000, B=64, N_steps=500: ~10^8 ops.
        On GPU: milliseconds to tens of milliseconds (not sub-millisecond due to
        kernel launch overhead and state polishing).

    Stability notes:
        - x should be clamped to [-2, 2] to prevent blowup at large lambda
        - eta may need reduction if lambda_max is large (stiff ODE)
        - Multiple random starts (B >= 16) are essential for finding all basins
    """
    rng = np.random.RandomState(seed)
    N = len(eps)

    # Precompute spin-space coefficients
    J, h = spin_coefficients(eps, V)

    # Auto-compute lambda_max from spectral range of J
    if lambda_max is None:
        if hasattr(V, 'toarray'):
            V_dense = V.toarray()
        else:
            V_dense = V
        # Spectral radius estimate via Gershgorin: |lambda| <= max row sum of |J|
        J_abs_row_sum = np.abs(J).sum(axis=1) if not hasattr(J, 'toarray') else np.abs(J.toarray()).sum(axis=1)
        lambda_max = 2.0 * np.max(J_abs_row_sum) + 1.0

    # Initialize: small random positions near x=0 (unstable fixed point)
    x = rng.uniform(-0.1, 0.1, size=(B, N))

    for step in range(N_steps):
        lam = lambda_max * (step + 1) / N_steps  # linear ramp

        if discrete:
            # Discrete SB: force from sign(x), more robust for QUBO
            s = np.sign(x)
            force = s @ J.T + h[None, :]  # (B,N) = (B,N) @ (N,N)^T
        else:
            # Continuous SB: force from x itself
            force = x @ J.T + h[None, :]  # batched GEMV/GEMM

        force -= 4.0 * lam * x * (x * x - 1.0)  # bifurcation potential, O(B*N)
        x += eta * force  # gradient descent
        x = np.clip(x, -2.0, 2.0)  # stability clamp

    # Discretize: s = sign(x), n = (1+s)/2
    n_list = np.zeros((B, N), dtype=int)
    E_list = np.zeros(B)
    for b in range(B):
        n = spin_to_n(x[b])
        n, _ = greedy_polish(n, eps, V)
        n_list[b] = n
        E_list[b] = compute_energy(n, eps, V)

    if verbose:
        print(f"#SB: B={B}, steps={N_steps}, lambda_max={lambda_max:.4f}, E_range=[{E_list.min():.6f},{E_list.max():.6f}]")

    return n_list, x, E_list


# ---------------------------------------------------------------------------
# L5: Exact enumeration (brute force, N <= 24)
# ---------------------------------------------------------------------------

def exact_ground_state(eps, V, ensemble='grand_canonical', N_q=None, verbose=False):
    """Brute-force exact ground state by enumerating all 2^N configurations.
    Only feasible for N <= ~24 (2^24 = 16M configurations).

    For fixed_charge, only enumerates configurations with sum n_i = N_q
    (C(N, N_q) configurations, which may be much fewer than 2^N).
    """
    N = len(eps)
    if N > 24:
        raise ValueError(f"Exact enumeration infeasible for N={N} (>24)")

    if hasattr(V, 'toarray'):
        V_dense = V.toarray()
    else:
        V_dense = V

    best_E = np.inf
    best_n = None

    if ensemble == 'fixed_charge' and N_q is not None:
        # Enumerate only fixed-charge configurations
        from itertools import combinations
        for occ in combinations(range(N), N_q):
            n = np.zeros(N, dtype=int)
            n[list(occ)] = 1
            E = compute_energy(n, eps, V_dense)
            if E < best_E:
                best_E = E
                best_n = n.copy()
    else:
        # Full enumeration
        for mask in range(1 << N):
            n = np.array([(mask >> i) & 1 for i in range(N)], dtype=int)
            E = compute_energy(n, eps, V_dense)
            if E < best_E:
                best_E = E
                best_n = n.copy()

    if verbose:
        print(f"#EXACT: N={N}, E_gs={best_E:.10f}, n_gs={best_n}")

    return best_n, best_E


def exact_all_low_energy(eps, V, E_threshold=None, ensemble='grand_canonical', N_q=None, max_states=1000):
    """Enumerate all states within E_threshold of the ground state energy.
    Useful for finding degenerate states and building the configuration graph.
    Returns list of (n, E) sorted by energy.
    """
    N = len(eps)
    if N > 24:
        raise ValueError(f"Exact enumeration infeasible for N={N} (>24)")

    if hasattr(V, 'toarray'):
        V_dense = V.toarray()
    else:
        V_dense = V

    # First find ground state energy
    n_gs, E_gs = exact_ground_state(eps, V_dense, ensemble=ensemble, N_q=N_q)
    if E_threshold is None:
        E_threshold = 0.1 * abs(E_gs) + 0.01

    states = []
    if ensemble == 'fixed_charge' and N_q is not None:
        from itertools import combinations
        for occ in combinations(range(N), N_q):
            n = np.zeros(N, dtype=int)
            n[list(occ)] = 1
            E = compute_energy(n, eps, V_dense)
            if E <= E_gs + E_threshold:
                states.append((n.copy(), E))
    else:
        for mask in range(1 << N):
            n = np.array([(mask >> i) & 1 for i in range(N)], dtype=int)
            E = compute_energy(n, eps, V_dense)
            if E <= E_gs + E_threshold:
                states.append((n.copy(), E))

    states.sort(key=lambda s: s[1])
    return states[:max_states]


# ---------------------------------------------------------------------------
# Candidate pool (core module for tip scanning)
# ---------------------------------------------------------------------------

class CandidatePool:
    """Pool of K distinct low-energy configurations for scan reuse.

    For each new tip position, re-evaluate all pool energies cheaply:
        E_a(R) = E_a^V + eps(R) . n_a
    where E_a^V = (1/2) n_a^T V n_a is stored once (drift-free).

    This prevents warm-start hysteresis from hiding alternative basins.
    The pool is updated with unique low-energy states found by MFA/SB/greedy.

    Precision policy:
        E_a^V and E_a(R) stored in float64 to avoid float32 reordering
        of nearly-degenerate states (Coulomb diamonds are energy-crossing phenomena).
    """

    def __init__(self, V, max_size=100, delta_E_keep=0.1):
        self.V = V
        self.max_size = max_size
        self.delta_E_keep = delta_E_keep
        self.n_list = []   # list of (N,) int arrays
        self.E_V = []      # list of float64: interaction energy (1/2) n^T V n
        self.E_curr = []   # list of float64: total energy at current tip position
        self.hashes = []   # list of int: hash for fast uniqueness
        self.basin_count = []  # how many times each basin was found

    def _hash(self, n):
        return hash(n.tobytes())

    def add(self, n, eps, E=None):
        """Add a configuration to the pool if it's unique and low-energy enough."""
        h = self._hash(n)
        if h in self.hashes:
            idx = self.hashes.index(h)
            self.basin_count[idx] += 1
            return False  # already in pool

        E_V = 0.5 * float(n.astype(float) @ (self.V @ n.astype(float)))
        if E is None:
            E = E_V + float(eps @ n.astype(float))

        self.n_list.append(n.copy())
        self.E_V.append(E_V)
        self.E_curr.append(E)
        self.hashes.append(h)
        self.basin_count.append(1)
        return True

    def update_energies(self, eps):
        """Recompute E_curr for all pool members at new tip position (drift-free)."""
        for a in range(len(self.n_list)):
            self.E_curr[a] = self.E_V[a] + float(eps @ self.n_list[a].astype(float))

    def get_best(self):
        """Return (n_best, E_best) — lowest energy configuration in pool."""
        if len(self.n_list) == 0:
            return None, np.inf
        idx = int(np.argmin(self.E_curr))
        return self.n_list[idx].copy(), self.E_curr[idx]

    def get_sorted(self):
        """Return all (n, E) sorted by energy."""
        idx = np.argsort(self.E_curr)
        return [(self.n_list[i].copy(), self.E_curr[i]) for i in idx]

    def get_gap(self):
        """Energy gap E_1 - E_0 between best and second-best. inf if < 2 states."""
        if len(self.n_list) < 2:
            return np.inf
        sorted_E = np.sort(self.E_curr)
        return sorted_E[1] - sorted_E[0]

    def prune(self, E_best):
        """Keep only states with E - E_best < delta_E_keep, up to max_size."""
        keep = [(a, self.E_curr[a]) for a in range(len(self.n_list))
                if self.E_curr[a] <= E_best + self.delta_E_keep]
        keep.sort(key=lambda x: x[1])
        keep = keep[:self.max_size]
        self.n_list = [self.n_list[a] for a, _ in keep]
        self.E_V = [self.E_V[a] for a, _ in keep]
        self.E_curr = [self.E_curr[a] for a, _ in keep]
        self.hashes = [self.hashes[a] for a, _ in keep]
        self.basin_count = [self.basin_count[a] for a, _ in keep]

    def __len__(self):
        return len(self.n_list)


# ---------------------------------------------------------------------------
# Tip scanner with candidate pool
# ---------------------------------------------------------------------------

def scan_tip_positions(eps_fn, V, N, n_tips=100, B_sb=16, B_mfa=8,
                       ensemble='grand_canonical', pool_size=50, seed=None,
                       verbose=False):
    """Scan over tip positions, using candidate pool + SB/MFA for ground-state tracking.

    Parameters
    ----------
    eps_fn : callable(i_tip) -> (N,) float — on-site energies for tip position i
    V : (N,N) — Coulomb matrix (constant across tip positions)
    n_tips : int — number of tip positions to scan
    B_sb, B_mfa : int — batch sizes for SB and MFA restarts per tip position

    Returns
    -------
    results : list of dict with keys: n_best, E_best, gap, pool_size, tip_idx

    Two modes run simultaneously:
        - Hysteretic: warm-start from previous solution + greedy polish
        - Equilibrium: warm-start + pool re-eval + SB/MFA restarts
    The equilibrium result is returned; hysteretic can be recovered by
    only using the warm-start candidate.
    """
    rng = np.random.RandomState(seed)
    pool = CandidatePool(V, max_size=pool_size, delta_E_keep=0.5)
    n_prev = None
    results = []

    for it in range(n_tips):
        eps = eps_fn(it)
        pool.update_energies(eps)
        candidates = []

        # 1. Hysteretic warm-start
        if n_prev is not None:
            n, _ = greedy_polish(n_prev.copy(), eps, V, ensemble=ensemble)
            candidates.append(n)

        # 2. Pool re-evaluation + polish
        for n_pool, _ in pool.get_sorted()[:10]:  # polish top 10 pool states
            n, _ = greedy_polish(n_pool.copy(), eps, V, ensemble=ensemble)
            candidates.append(n)

        # 3. Fresh SB restarts
        n_sb, _, _ = sb_solve(eps, V, B=B_sb, seed=rng.randint(0, 2**31), discrete=True)
        candidates.extend(n_sb)

        # 4. MFA restarts (fewer, cheaper)
        n_mfa, _, _ = mfa_anneal(eps, V, B=B_mfa, ensemble=ensemble, seed=rng.randint(0, 2**31))
        candidates.extend(n_mfa)

        # 5. Select best, update pool
        E_all = np.array([compute_energy(n, eps, V) for n in candidates])
        i_best = int(np.argmin(E_all))
        n_best = candidates[i_best]
        E_best = E_all[i_best]
        n_prev = n_best

        for i, n in enumerate(candidates):
            pool.add(n, eps, E=E_all[i])
        pool.prune(E_best)

        gap = pool.get_gap()
        results.append({
            'tip_idx': it, 'n_best': n_best.copy(), 'E_best': E_best,
            'gap': gap, 'pool_size': len(pool),
        })

        if verbose and it % max(1, n_tips // 10) == 0:
            print(f"  tip {it}/{n_tips}: E={E_best:.6f}, gap={gap:.6f}, pool={len(pool)}")

    return results
