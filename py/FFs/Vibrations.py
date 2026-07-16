"""
Vibrations.py — Normal-mode analysis from molecular Hessians (DFTB or GPU force-field FD).

Rigid-body modes are projected from the Hessian before diagonalization — required for
floating molecules in the GUI where translations/rotations would otherwise appear as
near-zero "vibrations". In-plane vs out-of-plane classification uses displacement
energy fractions in the lab xy/z axes (suited to planar adsorbates on z≈0).

Design:
- Rigid projection is performed in the mass-weighted dynamical-matrix space and then
  transformed back to Cartesian Hessian coordinates.
- Returned physical modes satisfy ``U.T @ M @ U = I`` and frequencies are spectroscopic
  wavenumbers ``omega/(2*pi*c)``.
- ``ReducedPolynomialPotential`` fits a local reduced-coordinate energy and its exact
  analytic force gradient.  The accompanying sampled radial-quartic stabilizer and
  bounded load-relaxation helper make such models safe to minimize *inside an
  explicitly stated trust region*; they do not turn a local fit into a validated
  global molecular potential.
- ``BlendedFrameDynamics`` implements two-frame molecular skinning: atomic external
  forces are transferred by ``J.T @ F`` and the instantaneous mass is ``J.T @ M @ J``.
  Its finite-difference Jacobian and damped velocity-Verlet path are correctness-first
  local dynamics, not an energy-conserving production NVE integrator.

- **Backends:** DFTB+ `SecondDerivatives`; UFF/SPFF via `FFEvaluator.make_ff_eval_fn` + central FD
- **Entry point:** `run_vibrations(mol, backend=...)`
- **Units:** internal SSOT is cm⁻¹; display via `freq_cm1_to_unit` (meV, THz, kcal/mol)

Open issues:
- SPFF: pi-orbitals frozen during FD (nuclear-position modes only)
- Absolute frequencies from UFF/DFTB not calibrated to experiment
- Phonon bands / PBC: not implemented (see doc/Topics/Vibrations.md)
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Literal, Optional

import numpy as np

from .. import elements
from ..AtomicSystem import AtomicSystem
from .UFF_cl import make_ff_eval_fn

# DFTB imports are optional (only needed for backend='dftb')
try:
    from ..DFTB.DFTB_utils import (
        write_dftb_input_hessian,
        read_hessian,
        hessian_hartree_bohr_to_eV_angstrom,
        DFTB_EXE,
        DEFAULT_SK_SET,
    )
except (ImportError, RuntimeError, Exception):
    write_dftb_input_hessian = None
    read_hessian = None
    hessian_hartree_bohr_to_eV_angstrom = None
    DFTB_EXE = None
    DEFAULT_SK_SET = None

# h*c in eV·cm (for nu in cm^-1): E = HC_EV_CM * nu; E_zpe = 0.5 * HC_EV_CM * nu
HC_EV_CM = 1.239841984e-4
C_CM_S = 2.99792458e10
EV_TO_KCAL_MOL = 23.060548025
# (eV / (amu Å)) -> Å / fs².  The same factor applies to rotational generalized
# coordinates because their projected mass/force units contain matching Å² factors.
EV_PER_AMU_ANG2_TO_ANG_PER_FS2 = 9.648533212e-3

Backend = Literal['uff', 'spff', 'dftb']
FreqUnit = Literal['cm-1', 'meV', 'THz', 'kcal/mol']

FREQ_UNIT_LABELS = {
    'cm-1': 'cm⁻¹',
    'meV': 'meV',
    'THz': 'THz',
    'kcal/mol': 'kcal/mol',
}


def freq_cm1_to_unit(nu_cm1: float, unit: FreqUnit = 'cm-1') -> float:
    """Convert wavenumber (cm⁻¹) to display unit. Energy units use quantum E = h c ν."""
    nu = float(nu_cm1)
    if unit == 'cm-1':
        return nu
    if unit == 'THz':
        return nu * C_CM_S / 1e12
    E_eV = HC_EV_CM * abs(nu)
    sign = 1.0 if nu >= 0 else -1.0
    if unit == 'meV':
        return sign * E_eV * 1000.0
    if unit == 'kcal/mol':
        return sign * E_eV * EV_TO_KCAL_MOL
    raise ValueError(f"Unknown freq unit: {unit!r}")


def format_freq(nu_cm1: float, unit: FreqUnit = 'cm-1') -> str:
    v = freq_cm1_to_unit(nu_cm1, unit)
    if unit == 'cm-1':
        return f'{v:.2f}'
    if unit == 'THz':
        return f'{v:.4f}'
    if unit == 'meV':
        return f'{v:.2f}'
    return f'{v:.3f}'


def format_E_zpe(nu_cm1: float, unit: FreqUnit = 'cm-1') -> str:
    """Zero-point energy ½ h c ν in display-consistent units."""
    nu = float(nu_cm1)
    if unit in ('meV', 'kcal/mol'):
        return format_freq(0.5 * abs(nu), unit) if nu >= 0 else '—'
    return f'{0.5 * HC_EV_CM * abs(nu):.5f}'


@dataclass
class ModeInfo:
    index: int
    freq_cm1: float
    E_zpe_eV: float
    f_xy: float
    f_z: float
    character: str = ''


@dataclass
class VibrationResult:
    enames: list
    pos: np.ndarray           # (N, 3) Å
    masses: np.ndarray        # (N,) amu
    hessian: np.ndarray       # (3N, 3N) eV/Å² after rigid projection
    frequencies_cm1: np.ndarray  # (n_modes,) sorted ascending
    modes: np.ndarray         # (3N, n_modes) mass-normalized displacement vectors
    mode_info: list           # list[ModeInfo]
    backend: str
    n_rigid_removed: int = 6
    perm: Optional[list] = None

    def softest_indices(self, n=6):
        return np.argsort(np.abs(self.frequencies_cm1))[:n]

    def format_table(self, unit: FreqUnit = 'cm-1') -> str:
        flab = FREQ_UNIT_LABELS[unit]
        elab = flab if unit in ('meV', 'kcal/mol') else 'eV'
        lines = [f"{'#':>4} {('freq/' + flab):>14} {('E_zpe/' + elab):>14} {'f_xy':>8} {'f_z':>8}  character"]
        for m in self.mode_info:
            lines.append(f"{m.index:4d} {format_freq(m.freq_cm1, unit):>14} {format_E_zpe(m.freq_cm1, unit):>14} {m.f_xy:8.3f} {m.f_z:8.3f}  {m.character}")
        return '\n'.join(lines)

    def mode_freq_label(self, mode_index: int, unit: FreqUnit = 'cm-1') -> str:
        m = self.mode_info[mode_index]
        return f"{format_freq(m.freq_cm1, unit)} {FREQ_UNIT_LABELS[unit]}"


def reduced_polynomial_powers(ndim: int, min_degree: int = 3, max_degree: int = 4) -> np.ndarray:
    """Return unique monomial powers for total degrees ``min_degree..max_degree``.

    A power row ``[2, 0, 1]`` represents ``z[0]**2 * z[2]``.  Enumerating
    combinations with replacement is equivalent to storing fully symmetric cubic and
    quartic tensors without duplicate permutation coefficients.
    """
    if ndim < 1 or min_degree < 0 or max_degree < min_degree:
        raise ValueError(
            f"invalid polynomial dimensions/degrees: ndim={ndim}, "
            f"min_degree={min_degree}, max_degree={max_degree}"
        )
    powers = []
    for degree in range(min_degree, max_degree + 1):
        for indices in combinations_with_replacement(range(ndim), degree):
            powers.append(np.bincount(indices, minlength=ndim))
    return np.asarray(powers, dtype=np.int16)


def _monomial_values_gradients(z: np.ndarray, powers: np.ndarray):
    """Evaluate monomials and derivatives with respect to dimensionless ``z``."""
    z = np.asarray(z, dtype=np.float64)
    powers = np.asarray(powers, dtype=np.int16)
    values = np.prod(z[:, None, :] ** powers[None, :, :], axis=2)
    gradients = np.zeros((z.shape[0], powers.shape[0], z.shape[1]), dtype=np.float64)
    for i in range(z.shape[1]):
        active = powers[:, i] > 0
        if not np.any(active):
            continue
        p = powers[active].copy()
        prefactor = p[:, i].astype(np.float64)
        p[:, i] -= 1
        gradients[:, active, i] = prefactor[None, :] * np.prod(
            z[:, None, :] ** p[None, :, :], axis=2
        )
    return values, gradients


@dataclass
class ReducedPolynomialPotential:
    """Local reduced potential with an anchored harmonic Hessian.

    ``coordinate_scale`` nondimensionalizes the reduced coordinates before evaluating
    the higher-order monomials.  The stored coefficients therefore have energy units.
    The model is a Taylor-like approximation around ``eta=0``; a quartic polynomial
    with a negative homogeneous direction is not globally bounded and must only be
    used inside its fitted coordinate range.
    """

    K: np.ndarray
    coordinate_scale: np.ndarray
    powers: np.ndarray
    coefficients: np.ndarray

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=np.float64)
        self.coordinate_scale = np.asarray(self.coordinate_scale, dtype=np.float64)
        self.powers = np.asarray(self.powers, dtype=np.int16)
        self.coefficients = np.asarray(self.coefficients, dtype=np.float64)
        ndim = self.K.shape[0] if self.K.ndim == 2 else -1
        if (
            self.K.shape != (ndim, ndim)
            or self.coordinate_scale.shape != (ndim,)
            or self.powers.ndim != 2
            or self.powers.shape[1] != ndim
            or self.coefficients.shape != (self.powers.shape[0],)
        ):
            raise ValueError("incompatible reduced-polynomial model array shapes")
        if np.any(self.coordinate_scale <= 0) or np.any(self.powers < 0):
            raise ValueError("coordinate scales must be positive and monomial powers nonnegative")
        if not all(np.all(np.isfinite(a)) for a in (self.K, self.coordinate_scale, self.coefficients)):
            raise ValueError("reduced-polynomial model contains NaN or infinite values")
        if not np.allclose(self.K, self.K.T, rtol=1e-10, atol=1e-12):
            raise ValueError("harmonic anchor K must be symmetric")
        self.K = 0.5 * (self.K + self.K.T)

    def save_npz(self, path):
        """Save the complete scaled-coordinate model without losing its basis contract."""
        np.savez(
            path,
            K=np.asarray(self.K, dtype=np.float64),
            coordinate_scale=np.asarray(self.coordinate_scale, dtype=np.float64),
            powers=np.asarray(self.powers, dtype=np.int16),
            coefficients=np.asarray(self.coefficients, dtype=np.float64),
        )

    @classmethod
    def load_npz(cls, path):
        """Load a model written by ``save_npz``."""
        with np.load(path) as data:
            return cls(
                K=data['K'],
                coordinate_scale=data['coordinate_scale'],
                powers=data['powers'],
                coefficients=data['coefficients'],
            )

    def energy_force(self, eta):
        """Return model energy and conjugate force ``Q=-dV/deta``."""
        eta = np.asarray(eta, dtype=np.float64)
        scalar_input = eta.ndim == 1
        eta_batch = eta[None, :] if scalar_input else eta
        if eta_batch.ndim != 2 or eta_batch.shape[1] != self.K.shape[0]:
            raise ValueError(f"eta must have shape ({self.K.shape[0]},) or (n,{self.K.shape[0]})")
        z = eta_batch / self.coordinate_scale[None, :]
        values, grad_z = _monomial_values_gradients(z, self.powers)
        energy = 0.5 * np.einsum('ni,ij,nj->n', eta_batch, self.K, eta_batch)
        energy += values @ self.coefficients
        force = -(eta_batch @ self.K.T)
        force -= np.einsum('ntd,t->nd', grad_z, self.coefficients) / self.coordinate_scale[None, :]
        if scalar_input:
            return float(energy[0]), force[0]
        return energy, force

    def error_metrics(self, samples) -> dict:
        """Return energy and force errors for ``(eta, dE, Q)`` samples."""
        eta = np.asarray([s[0] for s in samples], dtype=np.float64)
        energy_ref = np.asarray([s[1] for s in samples], dtype=np.float64)
        force_ref = np.asarray([s[2] for s in samples], dtype=np.float64)
        energy, force = self.energy_force(eta)
        de = energy - energy_ref
        dq = force - force_ref
        return {
            'energy_rmse': float(np.sqrt(np.mean(de * de))),
            'energy_max_abs': float(np.max(np.abs(de))),
            'force_component_rmse': float(np.sqrt(np.mean(dq * dq))),
            'force_vector_rms': float(np.sqrt(np.mean(np.sum(dq * dq, axis=1)))),
            'force_max_abs': float(np.max(np.abs(dq))),
        }

    def homogeneous_direction_range(self, degree: int = 4, n_directions: int = 4096, seed: int = 123):
        """Sample the homogeneous term on the dimensionless unit sphere."""
        if n_directions < 1:
            raise ValueError("n_directions must be positive")
        select = np.sum(self.powers, axis=1) == degree
        if not np.any(select):
            return 0.0, 0.0
        rng = np.random.RandomState(seed)
        z = rng.normal(size=(n_directions, self.K.shape[0]))
        z /= np.linalg.norm(z, axis=1)[:, None]
        values, _ = _monomial_values_gradients(z, self.powers[select])
        directional = values @ self.coefficients[select]
        return float(np.min(directional)), float(np.max(directional))


def stabilize_reduced_quartic_sampled(
    model: ReducedPolynomialPotential,
    margin: float = 1e-8,
    n_directions: int = 32768,
    safety_factor: float = 1.10,
    seed: int = 123,
):
    """Add ``lambda*||eta/scale||^4`` when sampled quartic directions are unsafe.

    This is an operational stabilization for a cubic/quartic Taylor fit.  It probes
    the quartic term on random unit-sphere directions, then adds an isotropic positive
    quartic contribution sufficient to lift the *sampled* minimum above ``margin``.
    The random scan is deliberately reported in the returned diagnostics: it is not
    a proof of global positivity.  Pair this helper with ``relax_reduced_potential``
    and a finite trust radius for a runtime-safe relaxation contract.
    """
    if margin <= 0 or not np.isfinite(margin):
        raise ValueError("margin must be finite and positive")
    if n_directions < 1 or safety_factor < 1 or not np.isfinite(safety_factor):
        raise ValueError("n_directions must be positive and safety_factor must be >= 1")

    ndim = model.K.shape[0]
    quartic_min, quartic_max = model.homogeneous_direction_range(
        degree=4, n_directions=n_directions, seed=seed,
    )
    # lambda ||z||^4 = lambda [sum_i z_i^4 + 2 sum_i<j z_i^2 z_j^2].
    # It is exactly lambda on the dimensionless unit sphere.
    lam = max(0.0, safety_factor * (margin - quartic_min))
    coefficients = model.coefficients.copy()
    by_power = {tuple(p.tolist()): i for i, p in enumerate(model.powers)}
    if lam > 0:
        required = []
        for i in range(ndim):
            p = np.zeros(ndim, dtype=np.int16); p[i] = 4
            required.append((tuple(p.tolist()), 1.0))
        for i in range(ndim):
            for j in range(i + 1, ndim):
                p = np.zeros(ndim, dtype=np.int16); p[i] = p[j] = 2
                required.append((tuple(p.tolist()), 2.0))
        missing = [power for power, _ in required if power not in by_power]
        if missing:
            raise ValueError(
                "quartic stabilization requires a complete degree-4 monomial basis; "
                f"missing {len(missing)} terms"
            )
        for power, factor in required:
            coefficients[by_power[power]] += lam * factor

    stabilized = ReducedPolynomialPotential(
        K=model.K.copy(),
        coordinate_scale=model.coordinate_scale.copy(),
        powers=model.powers.copy(),
        coefficients=coefficients,
    )
    stabilized_min, stabilized_max = stabilized.homogeneous_direction_range(
        degree=4, n_directions=n_directions, seed=seed,
    )
    return stabilized, {
        'quartic_min_before': quartic_min,
        'quartic_max_before': quartic_max,
        'quartic_min_after': stabilized_min,
        'quartic_max_after': stabilized_max,
        'radial_quartic_lambda': float(lam),
        'margin': float(margin),
        'n_directions': int(n_directions),
        'seed': int(seed),
    }


@dataclass
class ReducedRelaxationResult:
    """Result of a bounded minimization of ``V(eta) - f_ext dot eta``."""

    eta: np.ndarray
    internal_energy: float
    total_energy: float
    restoring_force: np.ndarray
    external_force: np.ndarray
    residual_norm: float
    success: bool
    hit_boundary: bool
    message: str
    n_iterations: int


def _reduced_trust_radius(model: ReducedPolynomialPotential, trust_radius):
    """Normalize dimensionless scalar/vector trust radii to physical eta bounds."""
    ndim = model.K.shape[0]
    radius = np.asarray(trust_radius, dtype=np.float64)
    if radius.ndim == 0:
        radius = np.full(ndim, float(radius))
    if radius.shape != (ndim,) or np.any(radius <= 0) or not np.all(np.isfinite(radius)):
        raise ValueError("trust_radius must be a finite positive scalar or one value per coordinate")
    return radius * model.coordinate_scale


def relax_reduced_potential(
    model: ReducedPolynomialPotential,
    external_force,
    eta0=None,
    trust_radius=1.0,
    maxiter: int = 1000,
    gtol: float = 1e-10,
) -> ReducedRelaxationResult:
    """Relax a stable reduced potential under a constant generalized load.

    The minimized potential is ``V(eta) - f_ext dot eta``.  Bounds are in units of
    ``model.coordinate_scale`` and are part of the physical API: a boundary hit says
    that the requested load left the fit's validated coordinate box, not that the
    boundary configuration is a trustworthy prediction.
    """
    try:
        from scipy.optimize import minimize, root
    except ImportError as exc:  # pragma: no cover - depends on optional SciPy install
        raise ImportError("relax_reduced_potential requires scipy.optimize") from exc

    ndim = model.K.shape[0]
    external_force = np.asarray(external_force, dtype=np.float64)
    if external_force.shape != (ndim,) or not np.all(np.isfinite(external_force)):
        raise ValueError("external_force must be finite with one value per reduced coordinate")
    bounds_abs = _reduced_trust_radius(model, trust_radius)
    if eta0 is None:
        eta0 = np.zeros(ndim, dtype=np.float64)
    eta0 = np.asarray(eta0, dtype=np.float64)
    if eta0.shape != (ndim,) or not np.all(np.isfinite(eta0)):
        raise ValueError("eta0 must be finite with one value per reduced coordinate")
    eta0 = np.clip(eta0, -bounds_abs, bounds_abs)

    def objective(eta):
        energy, restoring_force = model.energy_force(eta)
        total = energy - np.dot(external_force, eta)
        # model Q = -dV/deta, so d(V-f.eta)/deta = -Q-f.
        gradient = -restoring_force - external_force
        return total, gradient

    result = minimize(
        objective, eta0, jac=True,
        method='L-BFGS-B', bounds=list(zip(-bounds_abs, bounds_abs)),
        options={'maxiter': int(maxiter), 'gtol': float(gtol), 'ftol': 1e-15, 'maxls': 100},
    )
    eta = np.asarray(result.x, dtype=np.float64)
    boundary_tolerance = 1e-7 * np.maximum(bounds_abs, model.coordinate_scale)
    hit_boundary = bool(np.any(np.abs(np.abs(eta) - bounds_abs) <= boundary_tolerance))
    # L-BFGS-B often stops on its relative-energy criterion while the configuration is
    # still a few ulps away from force balance.  For an interior solution, polish its
    # stationarity equation with a root solve.  Do not polish a boundary solution:
    # there the correct condition is KKT complementarity, not zero full gradient.
    root_result = None
    if not hit_boundary:
        root_result = root(
            lambda x: -model.energy_force(x)[1] - external_force,
            eta, method='hybr', options={'xtol': min(float(gtol), 1e-11)},
        )
        candidate = np.asarray(root_result.x, dtype=np.float64)
        if (
            root_result.success and np.all(np.isfinite(candidate))
            and np.all(np.abs(candidate) < bounds_abs - boundary_tolerance)
        ):
            eta = candidate
    internal_energy, restoring_force = model.energy_force(eta)
    residual = -restoring_force - external_force
    hit_boundary = bool(np.any(np.abs(np.abs(eta) - bounds_abs) <= boundary_tolerance))
    residual_norm = float(np.linalg.norm(residual))
    polished_success = root_result is not None and root_result.success and not hit_boundary
    success = bool((result.success or polished_success) and np.isfinite(internal_energy) and np.isfinite(residual_norm))
    message = str(result.message)
    n_iterations = int(result.nit)
    if root_result is not None:
        message += f"; stationarity polish: {root_result.message}"
        n_iterations += int(getattr(root_result, 'nfev', 0))
    return ReducedRelaxationResult(
        eta=eta,
        internal_energy=float(internal_energy),
        total_energy=float(internal_energy - np.dot(external_force, eta)),
        restoring_force=np.asarray(restoring_force, dtype=np.float64),
        external_force=external_force.copy(),
        residual_norm=residual_norm,
        success=success,
        hit_boundary=hit_boundary,
        message=message,
        n_iterations=n_iterations,
    )


def relax_reduced_load_path(
    model: ReducedPolynomialPotential,
    external_force_final,
    load_fractions=None,
    n_steps: int = 21,
    eta0=None,
    trust_radius=1.0,
    maxiter: int = 1000,
    gtol: float = 1e-10,
):
    """Quasistatically relax a continuation path of proportional external loads."""
    external_force_final = np.asarray(external_force_final, dtype=np.float64)
    if load_fractions is None:
        if n_steps < 2:
            raise ValueError("n_steps must be at least two")
        load_fractions = np.linspace(0.0, 1.0, n_steps)
    load_fractions = np.asarray(load_fractions, dtype=np.float64)
    if load_fractions.ndim != 1 or len(load_fractions) < 1 or not np.all(np.isfinite(load_fractions)):
        raise ValueError("load_fractions must be a non-empty finite one-dimensional array")
    current_eta = None if eta0 is None else np.asarray(eta0, dtype=np.float64)
    path = []
    for fraction in load_fractions:
        relaxed = relax_reduced_potential(
            model, fraction * external_force_final, eta0=current_eta,
            trust_radius=trust_radius, maxiter=maxiter, gtol=gtol,
        )
        path.append(relaxed)
        current_eta = relaxed.eta
    return load_fractions, path


def reduced_potential_hessian(model: ReducedPolynomialPotential, eta=None, relative_step: float = 1e-5):
    """Central-difference reduced Hessian ``d2V/deta2`` from analytic forces."""
    if relative_step <= 0 or not np.isfinite(relative_step):
        raise ValueError("relative_step must be finite and positive")
    ndim = model.K.shape[0]
    if eta is None:
        eta = np.zeros(ndim, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    if eta.shape != (ndim,) or not np.all(np.isfinite(eta)):
        raise ValueError("eta must be finite with one value per reduced coordinate")
    hessian = np.empty((ndim, ndim), dtype=np.float64)
    for i, step in enumerate(relative_step * model.coordinate_scale):
        eta_p = eta.copy(); eta_p[i] += step
        eta_m = eta.copy(); eta_m[i] -= step
        _, force_p = model.energy_force(eta_p)
        _, force_m = model.energy_force(eta_m)
        hessian[:, i] = -(force_p - force_m) / (2.0 * step)
    return 0.5 * (hessian + hessian.T)


def exp_se3(xi):
    """Map a six-vector twist ``[translation, rotation]`` to an SE(3) transform."""
    xi = np.asarray(xi, dtype=np.float64)
    if xi.shape != (6,) or not np.all(np.isfinite(xi)):
        raise ValueError("xi must be a finite six-vector")
    rho, phi = xi[:3], xi[3:]
    theta = np.linalg.norm(phi)
    omega = np.array([
        [0.0, -phi[2], phi[1]],
        [phi[2], 0.0, -phi[0]],
        [-phi[1], phi[0], 0.0],
    ])
    omega2 = omega @ omega
    if theta < 1e-6:
        omega3 = omega2 @ omega
        rotation = np.eye(3) + omega + 0.5 * omega2 + omega3 / 6.0
        V = np.eye(3) + 0.5 * omega + omega2 / 6.0 + omega3 / 24.0
    else:
        rotation = np.eye(3) + np.sin(theta) / theta * omega + (1.0 - np.cos(theta)) / theta**2 * omega2
        V = np.eye(3) + (1.0 - np.cos(theta)) / theta**2 * omega + (theta - np.sin(theta)) / theta**3 * omega2
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = V @ rho
    return transform


def log_se3(transform):
    """Return the principal six-vector logarithm of an SE(3) transform."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("transform must be a finite 4x4 matrix")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    cos_theta = np.clip(0.5 * (np.trace(rotation) - 1.0), -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    vee = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    if theta < 1e-7:
        phi = 0.5 * vee
    elif np.pi - theta < 1e-5:
        # The skew part loses accuracy near pi.  The unit eigenvector of R with
        # eigenvalue one remains well conditioned; orient it with the skew part.
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        axis /= np.linalg.norm(axis)
        if np.dot(axis, vee) < 0:
            axis *= -1.0
        phi = theta * axis
    else:
        phi = theta / (2.0 * np.sin(theta)) * vee
    omega = np.array([
        [0.0, -phi[2], phi[1]],
        [phi[2], 0.0, -phi[0]],
        [-phi[1], phi[0], 0.0],
    ])
    omega2 = omega @ omega
    if theta < 1e-6:
        V_inv = np.eye(3) - 0.5 * omega + omega2 / 12.0
    else:
        coefficient = 1.0 / theta**2 - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
        V_inv = np.eye(3) - 0.5 * omega + coefficient * omega2
    return np.concatenate([V_inv @ translation, phi])


def reconstruct_blended_frame_positions(pos0, frame_transforms, weights):
    """Skin a reference molecule by partition-of-unity weighted absolute SE(3) frames."""
    pos0 = np.asarray(pos0, dtype=np.float64)
    frame_transforms = np.asarray(frame_transforms, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if pos0.ndim != 2 or pos0.shape[1] != 3:
        raise ValueError("pos0 must have shape (n_atoms, 3)")
    if frame_transforms.ndim != 3 or frame_transforms.shape[1:] != (4, 4):
        raise ValueError("frame_transforms must have shape (n_frames, 4, 4)")
    if weights.shape != (pos0.shape[0], frame_transforms.shape[0]):
        raise ValueError("weights must have one row per atom and one column per frame")
    if not all(np.all(np.isfinite(a)) for a in (pos0, frame_transforms, weights)):
        raise ValueError("frame reconstruction inputs contain NaN or infinite values")
    if not np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-10):
        raise ValueError("frame weights must satisfy partition of unity for every atom")
    rotated = np.einsum('kab,nb->nka', frame_transforms[:, :3, :3], pos0)
    transformed = rotated + frame_transforms[None, :, :3, 3]
    return np.einsum('nk,nka->na', weights, transformed)


def blended_frame_jacobian(pos0, frame_transforms, weights, step: float = 1e-6):
    """Differentiate blended positions with respect to left SE(3) frame increments."""
    if step <= 0 or not np.isfinite(step):
        raise ValueError("step must be finite and positive")
    frame_transforms = np.asarray(frame_transforms, dtype=np.float64)
    n_frames = frame_transforms.shape[0]
    reference = reconstruct_blended_frame_positions(pos0, frame_transforms, weights)
    jacobian = np.empty((reference.size, 6 * n_frames), dtype=np.float64)
    for frame in range(n_frames):
        for component in range(6):
            delta = np.zeros(6); delta[component] = step
            plus = frame_transforms.copy(); plus[frame] = exp_se3(delta) @ plus[frame]
            minus = frame_transforms.copy(); minus[frame] = exp_se3(-delta) @ minus[frame]
            pos_plus = reconstruct_blended_frame_positions(pos0, plus, weights)
            pos_minus = reconstruct_blended_frame_positions(pos0, minus, weights)
            jacobian[:, 6 * frame + component] = ((pos_plus - pos_minus) / (2.0 * step)).ravel()
    return jacobian


def project_atomic_forces_to_frames(atom_forces, position_jacobian):
    """Project Cartesian atomic forces to frame wrenches by exact discrete virtual work."""
    atom_forces = np.asarray(atom_forces, dtype=np.float64)
    position_jacobian = np.asarray(position_jacobian, dtype=np.float64)
    if atom_forces.ndim != 2 or atom_forces.shape[1] != 3:
        raise ValueError("atom_forces must have shape (n_atoms, 3)")
    if position_jacobian.shape[0] != atom_forces.size:
        raise ValueError("position_jacobian rows must match the flattened atomic-force vector")
    if not all(np.all(np.isfinite(a)) for a in (atom_forces, position_jacobian)):
        raise ValueError("force projection inputs contain NaN or infinite values")
    return position_jacobian.T @ atom_forces.ravel()


def relative_frame_eta(frame_transforms, length_scale: float):
    """Return scaled relative twist ``eta=S log(D1^-1 D2)`` for exactly two frames."""
    frame_transforms = np.asarray(frame_transforms, dtype=np.float64)
    if frame_transforms.shape != (2, 4, 4):
        raise ValueError("relative_frame_eta currently requires exactly two frames")
    if length_scale <= 0 or not np.isfinite(length_scale):
        raise ValueError("length_scale must be finite and positive")
    xi = log_se3(np.linalg.inv(frame_transforms[0]) @ frame_transforms[1])
    return np.array([1.0, 1.0, 1.0, length_scale, length_scale, length_scale]) * xi


def relative_frame_eta_jacobian(frame_transforms, length_scale: float, step: float = 1e-6):
    """Differentiate the two-frame relative eta coordinate by left pose increments."""
    frame_transforms = np.asarray(frame_transforms, dtype=np.float64)
    if frame_transforms.shape != (2, 4, 4):
        raise ValueError("relative_frame_eta_jacobian currently requires exactly two frames")
    if step <= 0 or not np.isfinite(step):
        raise ValueError("step must be finite and positive")
    jacobian = np.empty((6, 12), dtype=np.float64)
    for frame in range(2):
        for component in range(6):
            delta = np.zeros(6); delta[component] = step
            plus = frame_transforms.copy(); plus[frame] = exp_se3(delta) @ plus[frame]
            minus = frame_transforms.copy(); minus[frame] = exp_se3(-delta) @ minus[frame]
            jacobian[:, 6 * frame + component] = (
                relative_frame_eta(plus, length_scale) - relative_frame_eta(minus, length_scale)
            ) / (2.0 * step)
    return jacobian


@dataclass
class BlendedFrameEvaluation:
    """One conservative two-frame state evaluation, including force-transfer matrices."""

    positions: np.ndarray
    eta: np.ndarray
    internal_energy: float
    external_energy: float
    atom_forces: np.ndarray
    position_jacobian: np.ndarray
    frame_forces: np.ndarray
    mass_matrix: np.ndarray

    @property
    def potential_energy(self):
        return self.internal_energy + self.external_energy


@dataclass
class BlendedFrameDynamics:
    """Two-frame skinning dynamics with virtual-work force transfer and a reduced potential.

    Frame poses are updated by left SE(3) increments.  ``relax_step`` is the robust
    conservative choice for molecular relaxation because it backtracks on the total
    potential supplied by the external evaluator.  ``step_velocity_verlet`` provides
    damped inertial motion with the instantaneous projected mass matrix; it is a local
    reduced integrator, not a replacement for a fully variational Lie-group MD scheme.
    Both routes enforce the polynomial potential's relative-coordinate trust box.
    """

    pos0: np.ndarray
    weights: np.ndarray
    masses: np.ndarray
    potential: ReducedPolynomialPotential
    length_scale: float
    frame_transforms: np.ndarray | None = None
    velocities: np.ndarray | None = None
    trust_radius: float | np.ndarray = 1.0

    def __post_init__(self):
        self.pos0 = np.asarray(self.pos0, dtype=np.float64)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        self.masses = np.asarray(self.masses, dtype=np.float64)
        if self.pos0.ndim != 2 or self.pos0.shape[1] != 3 or self.masses.shape != (len(self.pos0),):
            raise ValueError("pos0/masses must describe one Cartesian mass per atom")
        if np.any(self.masses <= 0) or not np.all(np.isfinite(self.masses)):
            raise ValueError("masses must be finite and positive")
        if self.weights.shape != (len(self.pos0), 2):
            raise ValueError("BlendedFrameDynamics currently requires two frame weights per atom")
        if not np.allclose(np.sum(self.weights, axis=1), 1.0, atol=1e-10):
            raise ValueError("weights must satisfy partition of unity")
        if self.frame_transforms is None:
            self.frame_transforms = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
        self.frame_transforms = np.asarray(self.frame_transforms, dtype=np.float64)
        if self.frame_transforms.shape != (2, 4, 4):
            raise ValueError("frame_transforms must have shape (2, 4, 4)")
        if self.velocities is None:
            self.velocities = np.zeros(12, dtype=np.float64)
        self.velocities = np.asarray(self.velocities, dtype=np.float64)
        if self.velocities.shape != (12,) or not np.all(np.isfinite(self.velocities)):
            raise ValueError("velocities must be a finite 12-vector")
        if self.potential.K.shape != (6, 6):
            raise ValueError("two-frame dynamics requires a six-coordinate reduced potential")
        if self.length_scale <= 0 or not np.isfinite(self.length_scale):
            raise ValueError("length_scale must be finite and positive")
        self._eta_bounds = _reduced_trust_radius(self.potential, self.trust_radius)

    def positions(self):
        """Return current skinned atom positions."""
        return reconstruct_blended_frame_positions(self.pos0, self.frame_transforms, self.weights)

    def within_trust_region(self, eta=None):
        """Report whether the current relative deformation remains inside the fitted eta box."""
        if eta is None:
            eta = relative_frame_eta(self.frame_transforms, self.length_scale)
        eta = np.asarray(eta, dtype=np.float64)
        return bool(eta.shape == (6,) and np.all(np.abs(eta) <= self._eta_bounds))

    def position_jacobian(self):
        """Return current Cartesian-position Jacobian with respect to frame increments."""
        return blended_frame_jacobian(self.pos0, self.frame_transforms, self.weights)

    def mass_matrix(self, position_jacobian=None):
        """Return the instantaneous projected mass matrix ``J.T M J``."""
        if position_jacobian is None:
            position_jacobian = self.position_jacobian()
        mass3 = np.repeat(self.masses, 3)
        mass_matrix = position_jacobian.T @ (mass3[:, None] * position_jacobian)
        mass_matrix = 0.5 * (mass_matrix + mass_matrix.T)
        eigenvalues = np.linalg.eigvalsh(mass_matrix)
        if eigenvalues[0] <= max(eigenvalues[-1], 1.0) * 1e-12:
            raise RuntimeError("projected two-frame mass matrix is singular or ill-conditioned")
        return mass_matrix

    def evaluate(self, external_evaluator=None) -> BlendedFrameEvaluation:
        """Evaluate conservative internal/external forces and project them to frame space."""
        positions = self.positions()
        eta = relative_frame_eta(self.frame_transforms, self.length_scale)
        if not self.within_trust_region(eta):
            raise RuntimeError("two-frame relative deformation left the reduced-potential trust region")
        internal_energy, internal_eta_force = self.potential.energy_force(eta)
        eta_jacobian = relative_frame_eta_jacobian(self.frame_transforms, self.length_scale)
        internal_frame_force = eta_jacobian.T @ internal_eta_force
        if external_evaluator is None:
            external_energy = 0.0
            atom_forces = np.zeros_like(positions)
        else:
            external_energy, atom_forces = external_evaluator(positions)
            external_energy = float(external_energy)
            atom_forces = np.asarray(atom_forces, dtype=np.float64)
            if atom_forces.shape != positions.shape:
                raise ValueError("external_evaluator must return forces with shape (n_atoms, 3)")
        position_jacobian = self.position_jacobian()
        external_frame_force = project_atomic_forces_to_frames(atom_forces, position_jacobian)
        if not all(np.all(np.isfinite(a)) for a in (positions, eta, atom_forces, internal_frame_force, external_frame_force)):
            raise FloatingPointError("non-finite value in blended-frame force evaluation")
        return BlendedFrameEvaluation(
            positions=positions,
            eta=eta,
            internal_energy=float(internal_energy),
            external_energy=external_energy,
            atom_forces=atom_forces,
            position_jacobian=position_jacobian,
            frame_forces=internal_frame_force + external_frame_force,
            mass_matrix=self.mass_matrix(position_jacobian),
        )

    def apply_increment(self, increment):
        """Advance each frame pose by a left SE(3) increment."""
        increment = np.asarray(increment, dtype=np.float64)
        if increment.shape != (12,) or not np.all(np.isfinite(increment)):
            raise ValueError("increment must be a finite 12-vector")
        for frame in range(2):
            self.frame_transforms[frame] = exp_se3(increment[6 * frame:6 * frame + 6]) @ self.frame_transforms[frame]

    def kinetic_energy(self, mass_matrix=None):
        """Return instantaneous projected kinetic energy."""
        if mass_matrix is None:
            mass_matrix = self.mass_matrix()
        return 0.5 * float(self.velocities @ mass_matrix @ self.velocities)

    def relax_step(self, external_evaluator=None, mobility: float = 1.0, max_atom_step: float = 0.02, max_backtracks: int = 16):
        """Take an energy-decreasing projected-force relaxation step with backtracking."""
        if mobility <= 0 or max_atom_step <= 0 or max_backtracks < 0:
            raise ValueError("mobility/max_atom_step must be positive and max_backtracks nonnegative")
        current = self.evaluate(external_evaluator)
        direction = np.linalg.solve(current.mass_matrix, current.frame_forces)
        atom_direction = current.position_jacobian @ direction
        max_displacement = np.max(np.linalg.norm(atom_direction.reshape(-1, 3), axis=1))
        alpha = mobility if max_displacement == 0 else min(mobility, max_atom_step / max_displacement)
        old_transforms = self.frame_transforms.copy()
        for _ in range(max_backtracks + 1):
            self.frame_transforms = old_transforms.copy()
            self.apply_increment(alpha * direction)
            try:
                trial = self.evaluate(external_evaluator)
            except RuntimeError:
                alpha *= 0.5
                continue
            if np.isfinite(trial.potential_energy) and trial.potential_energy <= current.potential_energy:
                self.velocities.fill(0.0)
                return trial
            alpha *= 0.5
        self.frame_transforms = old_transforms
        raise RuntimeError("projected relaxation step could not find an energy-decreasing increment")

    def step_velocity_verlet(self, external_evaluator=None, dt: float = 0.1, damping: float = 0.0, max_atom_step: float = 0.02):
        """Advance local reduced inertial dynamics in fs using damped projected velocity-Verlet."""
        if dt <= 0 or damping < 0 or max_atom_step <= 0:
            raise ValueError("dt/max_atom_step must be positive and damping nonnegative")
        current = self.evaluate(external_evaluator)
        acceleration = EV_PER_AMU_ANG2_TO_ANG_PER_FS2 * np.linalg.solve(
            current.mass_matrix, current.frame_forces,
        )
        damping_factor = np.exp(-0.5 * damping * dt)
        velocity_half = damping_factor * self.velocities + 0.5 * dt * acceleration
        increment = dt * velocity_half
        atom_increment = current.position_jacobian @ increment
        max_displacement = np.max(np.linalg.norm(atom_increment.reshape(-1, 3), axis=1))
        if max_displacement > max_atom_step:
            increment *= max_atom_step / max_displacement
            velocity_half = increment / dt
        old_transforms = self.frame_transforms.copy()
        self.apply_increment(increment)
        try:
            updated = self.evaluate(external_evaluator)
        except RuntimeError as exc:
            self.frame_transforms = old_transforms
            self.velocities.fill(0.0)
            if "trust region" not in str(exc):
                raise
            # A rejected impulse must not create a latent velocity that tunnels
            # through the local model on the following step.
            return current
        updated_acceleration = EV_PER_AMU_ANG2_TO_ANG_PER_FS2 * np.linalg.solve(
            updated.mass_matrix, updated.frame_forces,
        )
        self.velocities = damping_factor * velocity_half + 0.5 * dt * updated_acceleration
        if not np.all(np.isfinite(self.velocities)):
            raise FloatingPointError("non-finite reduced velocity after velocity-Verlet step")
        return updated


def fit_reduced_polynomial(
    samples,
    K,
    min_degree: int = 3,
    max_degree: int = 4,
    coordinate_scale=None,
    ridge: float = 1e-8,
):
    """Fit higher-order energy and force residuals around a fixed harmonic ``K``.

    The regression is linear in unique monomial coefficients.  Coordinates are scaled
    to order unity, and the energy and force equation groups are RMS-normalized so six
    force components do not automatically outweigh one energy observation.
    """
    if not samples:
        raise ValueError("at least one sample is required")
    if ridge < 0:
        raise ValueError("ridge regularization must be nonnegative")
    K = np.asarray(K, dtype=np.float64)
    eta = np.asarray([s[0] for s in samples], dtype=np.float64)
    energy = np.asarray([s[1] for s in samples], dtype=np.float64)
    force = np.asarray([s[2] for s in samples], dtype=np.float64)
    ndim = eta.shape[1]
    if K.shape != (ndim, ndim) or force.shape != eta.shape:
        raise ValueError(f"incompatible K/eta/force shapes: {K.shape}, {eta.shape}, {force.shape}")
    if not all(np.all(np.isfinite(a)) for a in (K, eta, energy, force)):
        raise ValueError("polynomial fit inputs contain NaN or infinite values")
    if coordinate_scale is None:
        coordinate_scale = np.max(np.abs(eta), axis=0)
    coordinate_scale = np.asarray(coordinate_scale, dtype=np.float64)
    if coordinate_scale.shape != (ndim,) or np.any(coordinate_scale <= 0):
        raise ValueError("coordinate_scale must be positive and have one entry per coordinate")

    powers = reduced_polynomial_powers(ndim, min_degree=min_degree, max_degree=max_degree)
    z = eta / coordinate_scale[None, :]
    values, grad_z = _monomial_values_gradients(z, powers)
    harmonic_energy = 0.5 * np.einsum('ni,ij,nj->n', eta, K, eta)
    harmonic_force = -(eta @ K.T)
    energy_residual = energy - harmonic_energy
    force_residual = force - harmonic_force

    energy_scale = max(float(np.sqrt(np.mean(energy_residual ** 2))), 1e-12)
    force_scale = max(float(np.sqrt(np.mean(force_residual ** 2))), 1e-12)
    force_features = -grad_z / coordinate_scale[None, None, :]
    group_balance = np.sqrt(ndim)
    A = np.vstack([
        values / energy_scale,
        force_features.transpose(0, 2, 1).reshape(-1, powers.shape[0]) / (force_scale * group_balance),
    ])
    b = np.concatenate([
        energy_residual / energy_scale,
        force_residual.reshape(-1) / (force_scale * group_balance),
    ])

    degrees = np.sum(powers, axis=1)
    ridge_weights = ridge * (degrees / max(min_degree, 1)) ** 2
    A_aug = np.vstack([A, np.diag(np.sqrt(ridge_weights))])
    b_aug = np.concatenate([b, np.zeros(powers.shape[0])])
    coefficients, _, _, _ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    data_singular = np.linalg.svd(A, compute_uv=False)
    rank_tol = np.finfo(np.float64).eps * max(A.shape) * data_singular[0]
    data_rank = int(np.count_nonzero(data_singular > rank_tol))
    model = ReducedPolynomialPotential(
        K=0.5 * (K + K.T),
        coordinate_scale=coordinate_scale,
        powers=powers,
        coefficients=coefficients,
    )
    diagnostics = {
        'n_terms': int(powers.shape[0]),
        'rank': data_rank,
        'condition': (
            float(data_singular[0] / data_singular[data_rank-1])
            if data_rank == powers.shape[0] else float('inf')
        ),
        'energy_scale': energy_scale,
        'force_scale': force_scale,
        'ridge': float(ridge),
    }
    return model, diagnostics


def atomic_masses(enames) -> np.ndarray:
    return np.array([elements.ELEMENT_DICT[e][elements.index_mass] for e in enames], dtype=np.float64)


def hessian_fd_forces(eval_fn, pos0, delta=1e-4):
    """Central finite-difference Hessian from eval_fn(pos) -> (E, F). F in eV/Å."""
    pos0 = np.asarray(pos0, dtype=np.float64)
    natoms = pos0.shape[0]
    ndof = 3 * natoms
    H = np.zeros((ndof, ndof), dtype=np.float64)
    for i in range(ndof):
        ia, c = divmod(i, 3)
        pos_p = pos0.copy(); pos_p[ia, c] += delta
        pos_m = pos0.copy(); pos_m[ia, c] -= delta
        _, Fp = eval_fn(pos_p.astype(np.float32))
        _, Fm = eval_fn(pos_m.astype(np.float32))
        H[i, :] = -(Fp - Fm).ravel() / (2.0 * delta)
    return 0.5 * (H + H.T)


def rigid_body_basis(pos, masses):
    """Mass-weighted translation + rotation basis vectors, shape (3N, 6)."""
    pos = np.asarray(pos, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    N = len(masses)
    com = np.average(pos, axis=0, weights=masses)
    r = pos - com
    R = np.zeros((3 * N, 6), dtype=np.float64)
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        for i in range(N):
            R[3 * i:3 * i + 3, k] = np.sqrt(masses[i]) * e
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        for i in range(N):
            R[3 * i:3 * i + 3, 3 + k] = np.sqrt(masses[i]) * np.cross(r[i], e)
    Q, _ = np.linalg.qr(R)
    return Q


def project_rigid_modes(hessian, pos, masses):
    """Remove rigid motion in mass-weighted space; return Cartesian ``H'`` and projector.

    ``rigid_body_basis`` returns vectors in ``y=M**1/2 x`` coordinates.  Applying its
    Euclidean projector directly to the Cartesian Hessian mixes coordinate systems.
    Here ``P_mw`` projects the dynamical matrix, while ``P_phys`` is the corresponding
    mass-orthogonal projector acting on Cartesian displacements.
    """
    masses = np.asarray(masses, dtype=np.float64)
    sqrt_m = np.repeat(np.sqrt(masses), 3)
    inv_sqrt_m = 1.0 / sqrt_m
    Q_mw = rigid_body_basis(pos, masses)
    P_mw = np.eye(hessian.shape[0]) - Q_mw @ Q_mw.T
    D = inv_sqrt_m[:, None] * hessian * inv_sqrt_m[None, :]
    Dp = P_mw @ D @ P_mw
    Hp = sqrt_m[:, None] * Dp * sqrt_m[None, :]
    P_phys = inv_sqrt_m[:, None] * P_mw * sqrt_m[None, :]
    return 0.5 * (Hp + Hp.T), P_phys


def hessian_to_modes(hessian, masses):
    """Diagonalize a Cartesian Hessian; return cm⁻¹ and mass-normalized physical modes."""
    masses = np.asarray(masses, dtype=np.float64)
    im = np.repeat(masses ** -0.5, 3)
    D = im[:, None] * hessian * im[None, :]
    omega2, eigvec_mw = np.linalg.eigh(D)
    eV_to_J = 1.602176634e-19
    amu_to_kg = 1.66053906660e-27
    ang_to_m = 1e-10
    c_cm = 2.99792458e10
    omega_abs = np.sqrt(np.abs(omega2) * eV_to_J / (amu_to_kg * ang_to_m ** 2))
    freq_cm1 = np.sign(omega2) * omega_abs / (2.0 * np.pi * c_cm)
    modes_phys = im[:, None] * eigvec_mw
    for k in range(modes_phys.shape[1]):
        norm = np.sqrt(np.sum(np.repeat(masses, 3) * modes_phys[:, k] ** 2))
        if norm > 0:
            modes_phys[:, k] /= norm
    return freq_cm1, modes_phys


def mode_plane_fractions(mode_3n: np.ndarray):
    """Return (f_xy, f_z) displacement energy fractions for one mode vector (3N,)."""
    u = mode_3n.reshape(-1, 3)
    E_xy = float(np.sum(u[:, 0] ** 2 + u[:, 1] ** 2))
    E_z = float(np.sum(u[:, 2] ** 2))
    tot = E_xy + E_z
    if tot < 1e-30:
        return 0.5, 0.5
    return E_xy / tot, E_z / tot


def _character_label(f_xy, f_z):
    if f_z > 0.75:
        return 'out-of-plane'
    if f_xy > 0.75:
        return 'in-plane'
    return 'mixed'


def hessian_from_ff(mol, ff='uff', delta=1e-4, do_nonbond=False):
    eval_fn, pos0, natoms, perm, enames = make_ff_eval_fn(mol, ff=ff, do_nonbond=do_nonbond)
    H = hessian_fd_forces(eval_fn, pos0, delta=delta)
    masses = atomic_masses(enames)
    return H, pos0.astype(np.float64), masses, enames, perm


def hessian_from_dftb(enames, apos, workdir=None, sk_set=None, delta=1e-4):
    """Run DFTB+ SecondDerivatives; return H in eV/Å²."""
    sk_set = sk_set or DEFAULT_SK_SET
    apos = np.asarray(apos, dtype=np.float64)
    cwd = os.getcwd()
    own_tmp = workdir is None
    if own_tmp:
        workdir = tempfile.mkdtemp(prefix='spammm_vib_')
    try:
        os.chdir(workdir)
        with open('geo.xyz', 'w') as f:
            f.write(f"{len(enames)}\n")
            f.write("vibrations\n")
            for e, p in zip(enames, apos):
                f.write(f"{e} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        write_dftb_input_hessian(enames, gname='geo.xyz', sk_set=sk_set, delta=delta)
        ierr = os.system(f'{DFTB_EXE} > OUT 2> ERR')
        if ierr != 0:
            raise RuntimeError(f"DFTB+ Hessian failed with code {ierr} in {workdir}")
        H_hb = read_hessian('hessian.out', n_atoms=len(enames))
        H = hessian_hartree_bohr_to_eV_angstrom(H_hb)
        return H
    finally:
        os.chdir(cwd)
        if own_tmp:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)


def analyze_vibrations(hessian, pos, masses, enames, backend='uff', perm=None, rigid_freq_tol=20.0):
    Hp, _ = project_rigid_modes(hessian, pos, masses)
    freq_all, modes_all = hessian_to_modes(Hp, masses)
    vibr_mask = np.abs(freq_all) >= rigid_freq_tol
    freq = freq_all[vibr_mask]
    modes = modes_all[:, vibr_mask]
    order = np.argsort(np.abs(freq))
    freq = freq[order]
    modes = modes[:, order]
    mode_info = []
    for i, k in enumerate(range(modes.shape[1])):
        f = float(freq[k])
        f_xy, f_z = mode_plane_fractions(modes[:, k])
        mode_info.append(ModeInfo(index=i, freq_cm1=f, E_zpe_eV=0.5 * HC_EV_CM * abs(f), f_xy=f_xy, f_z=f_z, character=_character_label(f_xy, f_z)))
    return VibrationResult(enames=list(enames), pos=np.asarray(pos), masses=np.asarray(masses), hessian=Hp, frequencies_cm1=freq, modes=modes, mode_info=mode_info, backend=backend, perm=perm)


def run_vibrations(mol, backend: Backend = 'uff', delta=1e-4, do_nonbond=False, sk_set=None, workdir=None, rigid_freq_tol=20.0):
    """End-to-end vibrational analysis.

    Args:
        mol: AtomicSystem or path to .xyz
        backend: 'uff', 'spff', or 'dftb'
    """
    if isinstance(mol, str):
        mol = AtomicSystem(fname=mol)
    enames = list(mol.enames)
    pos = np.asarray(mol.apos[:, :3], dtype=np.float64)
    masses = atomic_masses(enames)
    perm = None
    if backend in ('uff', 'spff'):
        H, pos, masses, enames, perm = hessian_from_ff(mol, ff=backend, delta=delta, do_nonbond=do_nonbond)
    elif backend == 'dftb':
        H = hessian_from_dftb(enames, pos, workdir=workdir, sk_set=sk_set, delta=delta)
    else:
        raise ValueError(f"Unknown backend={backend!r}")
    return analyze_vibrations(H, pos, masses, enames, backend=backend, perm=perm, rigid_freq_tol=rigid_freq_tol)
