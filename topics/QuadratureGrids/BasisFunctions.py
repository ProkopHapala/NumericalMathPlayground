"""
BasisFunctions.py — Gaussian basis set functions and analytic integrals.

Core module: STO-nG orbital evaluation and exact 2D integrals.
No I/O, no plotting.
"""
import numpy as np


# ── STO-nG basis data ─────────────────────────────────────────────────────────

STO3G_1S = {
    'alpha': np.array([3.42525091, 0.62391373, 0.1688554]),
    'coef':  np.array([0.15432897, 0.53532814, 0.44463454]),
}

STO6G_1S = {
    'alpha': np.array([35.52322122, 6.513143725, 1.822142904,
                       0.625955266, 0.243076747, 0.100112428]),
    'coef':  np.array([0.00916359628, 0.04936149294, 0.1685383049,
                       0.3705627997, 0.4164915298, 0.1303340841]),
}

STO3G_2P_C = {
    'alpha': np.array([2.9412494, 0.6834831, 0.2222899]),
    'coef':  np.array([0.15591627, 0.60768372, 0.39195739]),
}


# ── Orbital evaluation ────────────────────────────────────────────────────────

def gaussian_1s(r2, basis, zeta=1.0):
    """1s orbital: Σ c_i * exp(-α_i * ζ² * r²). r2 = r² (squared distance)."""
    phi = np.zeros_like(r2)
    for a, c in zip(basis['alpha'], basis['coef']):
        phi += c * np.exp(-a * zeta**2 * r2)
    return phi


def gaussian_2p(x, y, basis, zeta=1.0):
    """2p_x orbital: x * Σ c_i * exp(-α_i * ζ² * r²)."""
    r2 = x**2 + y**2
    radial = np.zeros_like(r2)
    for a, c in zip(basis['alpha'], basis['coef']):
        radial += c * np.exp(-a * zeta**2 * r2)
    return x * radial


def gaussian_2py(x, y, basis, zeta=1.0):
    """2p_y orbital: y * Σ c_i * exp(-α_i * ζ² * r²)."""
    r2 = x**2 + y**2
    radial = np.zeros_like(r2)
    for a, c in zip(basis['alpha'], basis['coef']):
        radial += c * np.exp(-a * zeta**2 * r2)
    return y * radial


# ── Analytic integrals (2D) ───────────────────────────────────────────────────

def analytic_1s_norm_sq(basis, zeta):
    """Analytic ∫|φ_1s|² dA in 2D. ∫ exp(-βr²) dA = π/β."""
    total = 0.0
    for i in range(len(basis['alpha'])):
        for j in range(len(basis['alpha'])):
            beta = (basis['alpha'][i] + basis['alpha'][j]) * zeta**2
            total += basis['coef'][i] * basis['coef'][j] * np.pi / beta
    return total


def analytic_2p_norm_sq(basis, zeta):
    """Analytic ∫|φ_2p|² dA in 2D. ∫ x² exp(-βr²) dA = π/(2β²)."""
    total = 0.0
    for i in range(len(basis['alpha'])):
        for j in range(len(basis['alpha'])):
            beta = (basis['alpha'][i] + basis['alpha'][j]) * zeta**2
            total += basis['coef'][i] * basis['coef'][j] * np.pi / (2 * beta**2)
    return total


# ── Test function generation for weight optimization ──────────────────────────

def eval_test_func(xy, kind, zeta, x0, y0):
    """Evaluate a test function at (x, y) points."""
    x, y = xy[:, 0], xy[:, 1]
    r2 = (x - x0)**2 + (y - y0)**2
    if kind == 'gauss':
        return np.exp(-zeta**2 * r2)
    elif kind == 'px':
        return (x - x0) * np.exp(-zeta**2 * r2)
    elif kind == 'x2g':
        return (x - x0)**2 * np.exp(-zeta**2 * r2)
    elif kind == 'xyg':
        return (x - x0) * (y - y0) * np.exp(-zeta**2 * r2)
    elif kind == 'const':
        return np.ones(len(x))
    else:
        raise ValueError(f"Unknown test function kind: {kind}")


def build_test_function_set():
    """
    Build the standard set of test functions for weight optimization.

    Returns
    -------
    funcs : list of (label, (kind, zeta, x0, y0)) tuples
    integrals_total : list of float — analytic integral over all space
    """
    funcs = []
    integrals = []

    # Constant (area constraint)
    funcs.append(('const', ('const', 0, 0, 0)))
    integrals.append(None)  # handled specially: inner domain area

    # Centered Gaussians
    for zeta in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
        funcs.append((f'G(ζ={zeta})', ('gauss', zeta, 0.0, 0.0)))
        integrals.append(np.pi / zeta**2)

    # Off-center Gaussians
    for zeta in [0.5, 1.0, 2.0, 3.0]:
        for (x0, y0) in [(0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0), (1.5, 0.0),
                         (0.25, 0.25), (0.5, 0.5), (0.75, 0.25), (1.0, 0.5), (1.5, 0.5),
                         (0.25, 0.5), (0.5, 0.75), (1.0, 1.0), (1.5, 1.0), (2.0, 0.5)]:
            funcs.append((f'G(ζ={zeta},@({x0},{y0}))', ('gauss', zeta, x0, y0)))
            integrals.append(np.pi / zeta**2)

    # p-orbital-like (odd, integral = 0)
    for zeta in [1.0, 2.0, 3.0, 5.0]:
        for (x0, y0) in [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.5, 0.0)]:
            funcs.append((f'px(ζ={zeta},@({x0},{y0}))', ('px', zeta, x0, y0)))
            integrals.append(0.0)

    # d-orbital-like
    for zeta in [1.0, 2.0, 3.0, 5.0]:
        for (x0, y0) in [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)]:
            funcs.append((f'x²G(ζ={zeta},@({x0},{y0}))', ('x2g', zeta, x0, y0)))
            integrals.append(np.pi / (2 * zeta**4))

    return funcs, integrals


# ── PySCF radial wavefunction extraction ──────────────────────────────────────

def get_radial_functions(mol, r_grid):
    """
    Extract radial wavefunctions R(r) for each AO shell from a PySCF molecule.

    Evaluates AOs along the z-axis and factors out the angular part:
      s (l=0):  R(r) = AO(r)
      p (l=1):  R(r) = AO_pz(r) / r
      d (l=2):  R(r) = AO_dz2(r) / (2*r^2)

    Parameters
    ----------
    mol : pyscf.gto.Mole
    r_grid : (N,) array — radial distances (Bohr)

    Returns
    -------
    list of (label, l, R_values) for each unique shell
    """
    N = len(r_grid)
    coords = np.zeros((N, 3))
    coords[:, 2] = r_grid
    ao_vals = mol.eval_ao('GTOval', coords)

    labels = mol.ao_labels()
    shells = {}

    for iao, label in enumerate(labels):
        parts = label.split()
        orb_name = parts[-1]

        if 's' in orb_name and 'p' not in orb_name and 'd' not in orb_name:
            l = 0
            key = (l, orb_name)
            angular = np.ones(N)
        elif 'pz' in orb_name:
            l = 1
            key = (l, orb_name.replace('pz', 'p'))
            angular = r_grid.copy()
        elif 'px' in orb_name or 'py' in orb_name:
            continue
        elif 'dz^2' in orb_name or 'dz2' in orb_name:
            l = 2
            key = (l, orb_name.replace('dz^2', 'd').replace('dz2', 'd'))
            angular = 2 * r_grid**2
        elif 'f+0' in orb_name or 'f0' in orb_name:
            l = 3
            key = (l, orb_name)
            angular = r_grid**3
        elif 'd' in orb_name:
            continue
        else:
            continue

        r_vals = ao_vals[:, iao]
        mask = np.abs(angular) > 1e-15
        R = np.zeros(N)
        R[mask] = r_vals[mask] / angular[mask]
        if l > 0:
            R[~mask] = 0.0
        else:
            R[~mask] = r_vals[~mask]

        if key not in shells:
            shells[key] = (orb_name, l, R)

    return list(shells.values())


def compute_radial_extent(r_grid, R, threshold_frac=0.01, r_min=0.01):
    """
    Find radial extent where |R(r)| drops below threshold_frac * |R|_max.

    Returns
    -------
    r_extent : float — radial distance where R falls below threshold
    r_max : float — radial distance of |R| maximum
    R_max : float — maximum |R| value
    """
    R_abs = np.abs(R)
    if R_abs.max() <= 0:
        return r_grid[-1], 0.0, 0.0
    threshold = threshold_frac * R_abs.max()
    mask_nz = r_grid > r_min
    r_nz = r_grid[mask_nz]
    R_nz = R_abs[mask_nz]
    beyond = np.where(R_nz < threshold)[0]
    r_extent = r_nz[beyond[0]] if len(beyond) > 0 else r_grid[-1]
    r_max = r_nz[np.argmax(R_nz)]
    return r_extent, r_max, R_abs.max()


# ── Atomic orbital trial functions (from pySCF basis sets) ────────────────────

def _double_factorial(n):
    """n!! = n·(n-2)·...·1 or 2. (-1)!! = 0!! = 1."""
    if n <= 0:
        return 1
    r = 1
    while n > 0:
        r *= n
        n -= 2
    return r


def _gaussian_moment_2d(nx, ny, gamma):
    """∫ x^nx y^ny exp(-γ r²) dA over 2D. Odd powers → 0."""
    if nx % 2 == 1 or ny % 2 == 1:
        return 0.0
    m, n = nx // 2, ny // 2
    return _double_factorial(2*m - 1) * _double_factorial(2*n - 1) * np.pi / (2**(m+n) * gamma**(m+n+1))


def _angular_poly_matrix(kind, a, b):
    """Polynomial coefficient matrix for angular part. poly[i,j] = coeff of x^i y^j."""
    if kind == 's':
        p = np.zeros((1, 1)); p[0, 0] = 1.0
    elif kind == 'p_v':
        p = np.zeros((2, 2)); p[1, 0] = a; p[0, 1] = b
    elif kind == 'd_vv':
        p = np.zeros((3, 3)); p[2, 0] = a*a; p[1, 1] = 2*a*b; p[0, 2] = b*b
    elif kind == 'd_vu':
        p = np.zeros((3, 3)); p[2, 0] = -a*b; p[1, 1] = a*a - b*b; p[0, 2] = a*b
    else:
        raise ValueError(f"Unknown angular kind: {kind}")
    return p


def _poly_product(p1, p2):
    """Multiply two polynomial coefficient matrices."""
    n1, m1 = p1.shape
    n2, m2 = p2.shape
    result = np.zeros((n1 + n2 - 1, m1 + m2 - 1))
    for i1 in range(n1):
        for j1 in range(m1):
            if p1[i1, j1] == 0:
                continue
            for i2 in range(n2):
                for j2 in range(m2):
                    if p2[i2, j2] == 0:
                        continue
                    result[i1+i2, j1+j2] += p1[i1, j1] * p2[i2, j2]
    return result


def _overlap_2d(desc_i, desc_j):
    """Analytic ∫ φ_i φ_j dA over 2D for two basis function descriptors."""
    pi = _angular_poly_matrix(desc_i['angular_kind'], *desc_i['v'])
    pj = _angular_poly_matrix(desc_j['angular_kind'], *desc_j['v'])
    pp = _poly_product(pi, pj)
    total = 0.0
    for a_i, c_i in zip(desc_i['exps'], desc_i['coefs']):
        for a_j, c_j in zip(desc_j['exps'], desc_j['coefs']):
            gamma = a_i + a_j
            for ix in range(pp.shape[0]):
                for iy in range(pp.shape[1]):
                    if pp[ix, iy] != 0:
                        total += c_i * c_j * pp[ix, iy] * _gaussian_moment_2d(ix, iy, gamma)
    return total


def extract_pyscf_shells(elements, basis_name='cc-pVDZ'):
    """Extract contracted Gaussian shells from pySCF for given elements.

    Returns list of dicts: {element, l, exps, coefs, label}
    """
    from pyscf import gto
    shells = []
    for elem in elements:
        nelec = gto.mole.charge(elem)
        spin = nelec % 2
        mol = gto.M(atom=f'{elem} 0 0 0', basis=basis_name, spin=spin)
        for ib in range(mol.nbas):
            l = mol.bas_angular(ib)
            nctr = mol.bas_nctr(ib)
            exps = mol.bas_exp(ib)
            coefs = mol.bas_ctr_coeff(ib)  # (nprim, nctr)
            for j in range(nctr):
                shells.append(dict(
                    element=elem, l=l,
                    exps=exps.copy(), coefs=coefs[:, j].copy(),
                    label=f'{elem} l={l} ctr={j}',
                ))
    return shells


def build_atomic_basis(elements, basis_name='cc-pVDZ', n_angular_dirs=8, seed=42):
    """Build atomic basis function set with angular variants.

    For each radial shell:
      l=0 (s): 1 angular variant (isotropic)
      l=1 (p): n_angular_dirs variants (p along random directions)
      l=2 (d): 2*n_angular_dirs variants (d_vv and d_vu per direction)

    Returns list of basis function descriptors.
    """
    raw_shells = extract_pyscf_shells(elements, basis_name)
    rng = np.random.RandomState(seed)
    descs = []
    for shell in raw_shells:
        l = shell['l']
        if l == 0:
            descs.append(dict(**shell, angular_kind='s', v=(1.0, 0.0), u=(0.0, 1.0)))
        elif l == 1:
            for _ in range(n_angular_dirs):
                theta = rng.uniform(0, 2*np.pi)
                a, b = np.cos(theta), np.sin(theta)
                descs.append(dict(**shell, angular_kind='p_v', v=(a, b), u=(-b, a)))
        elif l == 2:
            for _ in range(n_angular_dirs):
                theta = rng.uniform(0, 2*np.pi)
                a, b = np.cos(theta), np.sin(theta)
                descs.append(dict(**shell, angular_kind='d_vv', v=(a, b), u=(-b, a)))
                descs.append(dict(**shell, angular_kind='d_vu', v=(a, b), u=(-b, a)))
    return descs


def eval_basis_functions(xy, basis_descs):
    """Evaluate all basis functions at xy points. Returns (n_basis, Npts) array."""
    x = xy[:, 0]
    y = xy[:, 1]
    r2 = x**2 + y**2
    n_basis = len(basis_descs)
    Npts = len(xy)
    vals = np.zeros((n_basis, Npts))
    for i, desc in enumerate(basis_descs):
        radial = np.zeros(Npts)
        for a, c in zip(desc['exps'], desc['coefs']):
            radial += c * np.exp(-a * r2)
        kind = desc['angular_kind']
        a_d, b_d = desc['v']
        if kind == 's':
            angular = 1.0
        elif kind == 'p_v':
            angular = a_d * x + b_d * y
        elif kind == 'd_vv':
            angular = (a_d * x + b_d * y)**2
        elif kind == 'd_vu':
            angular = (a_d * x + b_d * y) * (-b_d * x + a_d * y)
        vals[i] = angular * radial
    return vals


def compute_overlap_matrix(basis_descs):
    """Compute analytic overlap matrix S_ij = ∫φ_i φ_j dA over 2D."""
    n = len(basis_descs)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            S[i, j] = _overlap_2d(basis_descs[i], basis_descs[j])
            S[j, i] = S[i, j]
    return S


def build_atomic_trial_set(elements=['H', 'C', 'N', 'O'], basis_name='cc-pVDZ',
                           n_angular_dirs=8, n_combos=200, seed=42):
    """Build atomic orbital trial function set.

    Pipeline:
      1. Extract radial shells from pySCF
      2. Generate angular variants (s, p_v, d_vv, d_vu)
      3. Compute analytic overlap matrix
      4. Generate random linear combinations (normalized to unit integral)

    Returns dict with:
        basis_descs : list of basis function descriptors
        S : (n_basis, n_basis) overlap matrix
        C : (n_combos, n_basis) random combination coefficients (normalized)
        n_radial : number of radial shells (step 1)
        n_basis : number of angular variants (step 2)
        n_combos : number of random combinations (step 4)
    """
    raw_shells = extract_pyscf_shells(elements, basis_name)
    n_radial = len(raw_shells)

    basis_descs = build_atomic_basis(elements, basis_name, n_angular_dirs, seed)
    n_basis = len(basis_descs)

    S = compute_overlap_matrix(basis_descs)

    rng = np.random.RandomState(seed + 1)
    C = rng.randn(n_combos, n_basis)
    # Normalize each combination so ∫|Σ c_j φ_j|² dA = 1
    norms = np.sqrt(np.maximum(np.einsum('ki,ij,kj->k', C, S, C), 1e-30))
    C = C / norms[:, np.newaxis]

    # Count angular forks per l
    n_s = sum(1 for d in basis_descs if d['angular_kind'] == 's')
    n_p = sum(1 for d in basis_descs if d['angular_kind'] == 'p_v')
    n_d = sum(1 for d in basis_descs if d['angular_kind'].startswith('d'))

    print(f"\n── Atomic trial function set ──────────────────────────────────")
    print(f"  Elements: {elements}, basis: {basis_name}")
    print(f"  Radial shells (step 1): {n_radial}")
    print(f"  Angular variants (step 2): {n_basis}  (s={n_s}, p={n_p}, d={n_d})")
    print(f"  Random combinations (step 3): {n_combos}")
    print(f"  Total trial functions: {n_combos}")

    return dict(
        basis_descs=basis_descs, S=S, C=C,
        n_radial=n_radial, n_basis=n_basis, n_combos=n_combos,
        raw_shells=raw_shells,
    )


def eval_trial_functions(xy, trial_set):
    """Evaluate trial functions at xy points. Returns (n_combos, Npts) array of f = (Σ c_j φ_j)²."""
    B = eval_basis_functions(xy, trial_set['basis_descs'])
    F = (trial_set['C'] @ B)**2
    return F


def trial_analytic_integrals(trial_set):
    """Compute analytic integrals ∫f_k dA for all trial functions. Returns (n_combos,) array."""
    C = trial_set['C']
    S = trial_set['S']
    return np.einsum('ki,ij,kj->k', C, S, C)
