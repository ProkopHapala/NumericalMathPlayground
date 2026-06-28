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
