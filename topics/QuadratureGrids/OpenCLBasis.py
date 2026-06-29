"""
OpenCLBasis.py — OpenCL-accelerated evaluation of radial Gaussian basis functions.

Core module: extracts basis parameters from PySCF and evaluates R(r) on GPU.
No I/O, no plotting.

The radial function convention matches BasisFunctions.get_radial_functions:
  R(r) = C_l * Σ_i c_i * exp(-α_i * r²)

where C_l = Y_l0(θ=0) / angular_divisor, matching the extraction along the z-axis:
  l=0: C = 1/sqrt(4π)        (divisor = 1)
  l=1: C = sqrt(3/(4π))      (divisor = r)
  l=2: C = sqrt(5/(16π))     (divisor = 2r²)
"""
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

# ── OpenCL kernel source ──────────────────────────────────────────────────────

_KERNEL_SOURCE = r"""
// Single precision (float32) — ~10x faster on consumer GPUs.
// Accuracy ~1e-7 is well below quadrature grid error.

// Evaluate radial basis functions R(r) for all shells at all radial points.
// Global size: (N_radial, N_shells)
// Each work item computes R[ir, ishell]
__kernel void eval_radial(
    __global const float *r_grid,      // N_radial radial distances
    __global const float *exps,        // flat: all exponents
    __global const float *coefs,       // flat: all contraction coefficients
    __global const int  *shell_offsets, // N_shells: start index in exps/coefs
    __global const int  *shell_nprim,   // N_shells: number of primitives
    __global const int  *shell_l,       // N_shells: angular momentum l
    __global const float *shell_norm,    // N_shells: normalization constant C_l
    __global float       *R_out,        // N_radial × N_shells output (row-major)
    const int N_radial,
    const int N_shells
)
{
    int ir = get_global_id(0);
    int ishell = get_global_id(1);

    if (ir >= N_radial || ishell >= N_shells) return;

    int nprim  = shell_nprim[ishell];
    int offset = shell_offsets[ishell];
    int l      = shell_l[ishell];
    float norm = shell_norm[ishell];

    float r_val = r_grid[ir];
    float r2 = r_val * r_val;

    // For l > 0, R(0) = 0 (angular divisor vanishes)
    if (l > 0 && r_val < 1e-15f) {
        R_out[ir * N_shells + ishell] = 0.0f;
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < nprim; i++) {
        float alpha = exps[offset + i];
        float coef  = coefs[offset + i];
        sum += coef * exp(-alpha * r2);
    }

    R_out[ir * N_shells + ishell] = norm * sum;
}

// Evaluate radial basis functions AND their derivatives dR/dr.
// Uses finite-difference-free analytic derivative:
//   dR/dr = C_l * Σ_i c_i * (-2*α_i*r) * exp(-α_i * r²)
__kernel void eval_radial_deriv(
    __global const float *r_grid,
    __global const float *exps,
    __global const float *coefs,
    __global const int  *shell_offsets,
    __global const int  *shell_nprim,
    __global const int  *shell_l,
    __global const float *shell_norm,
    __global float       *R_out,
    __global float       *dR_out,
    const int N_radial,
    const int N_shells
)
{
    int ir = get_global_id(0);
    int ishell = get_global_id(1);

    if (ir >= N_radial || ishell >= N_shells) return;

    int nprim  = shell_nprim[ishell];
    int offset = shell_offsets[ishell];
    int l      = shell_l[ishell];
    float norm = shell_norm[ishell];

    float r_val = r_grid[ir];
    float r2 = r_val * r_val;

    if (l > 0 && r_val < 1e-15f) {
        R_out[ir * N_shells + ishell]  = 0.0f;
        dR_out[ir * N_shells + ishell] = 0.0f;
        return;
    }

    float sum  = 0.0f;
    float dsum = 0.0f;
    for (int i = 0; i < nprim; i++) {
        float alpha = exps[offset + i];
        float coef  = coefs[offset + i];
        float e = exp(-alpha * r2);
        sum  += coef * e;
        dsum += coef * (-2.0f * alpha * r_val) * e;
    }

    R_out[ir * N_shells + ishell]  = norm * sum;
    dR_out[ir * N_shells + ishell] = norm * dsum;
}
"""

# ── Angular normalization constants C_l ──────────────────────────────────────

def _angular_norm(l):
    """C_l = Y_l0(θ=0) / angular_divisor, matching get_radial_functions convention.

    The angular divisor along the z-axis is the solid harmonic at θ=0:
      l=0: 1, l=1: r, l=2: 2r², l=3: r³
    """
    if l == 0:
        return 1.0 / np.sqrt(4.0 * np.pi)
    elif l == 1:
        return np.sqrt(3.0 / (4.0 * np.pi))
    elif l == 2:
        return np.sqrt(5.0 / (16.0 * np.pi))
    elif l == 3:
        return np.sqrt(7.0 / (4.0 * np.pi))
    else:
        raise ValueError(f"Angular momentum l={l} not supported (only 0-3)")


# ── Basis parameter extraction ────────────────────────────────────────────────

def extract_basis_params(mol):
    """
    Extract basis parameters from a PySCF molecule.

    Returns
    -------
    shells : list of dict, each with keys:
        'label'    : str — AO label (e.g. '1s', '2pz')
        'l'        : int — angular momentum
        'exps'     : (nprim,) array — exponents
        'coefs'    : (nprim,) array — normalized contraction coefficients
        'norm'     : float — angular normalization C_l
    """
    ao_labels = mol.ao_labels()
    shells = []

    iao = 0
    for ibas in range(len(mol._bas)):
        l = int(mol._bas[ibas, 1])
        nprim = int(mol._bas[ibas, 2])
        nctr = int(mol._bas[ibas, 3])
        ptr_exp = int(mol._bas[ibas, 5])
        ptr_coeff = int(mol._bas[ibas, 6])

        exps = mol._env[ptr_exp:ptr_exp + nprim].copy()
        n_components = 2 * l + 1

        for ictr in range(nctr):
            coeff_start = ptr_coeff + ictr * nprim
            coefs = mol._env[coeff_start:coeff_start + nprim].copy()

            # Pick the z-axis AO label (m=0 component)
            # l=0: offset 0; l=1: pz at offset 2; l>=2: m=0 at offset l
            z_offset = 0 if l == 0 else (2 if l == 1 else l)
            label = ao_labels[iao + z_offset]
            iao += n_components

            shells.append({
                'label': label,
                'l': l,
                'exps': exps,
                'coefs': coefs,
                'norm': _angular_norm(l),
            })

    return shells


def filter_radial_shells(shells):
    """
    Filter to keep only the shells used in radial extraction
    (s, pz, dz2 — matching get_radial_functions).

    Returns
    -------
    filtered : list of dict with added 'radial_label' key
    """
    filtered = []
    for sh in shells:
        label = sh['label']
        parts = label.split()
        orb_name = parts[-1]

        if 's' in orb_name and 'p' not in orb_name and 'd' not in orb_name and 'f' not in orb_name:
            filtered.append({**sh, 'radial_label': orb_name})
        elif 'pz' in orb_name:
            filtered.append({**sh, 'radial_label': orb_name})
        elif 'dz^2' in orb_name or 'dz2' in orb_name:
            filtered.append({**sh, 'radial_label': orb_name})
        elif 'f+0' in orb_name or 'f0' in orb_name:
            filtered.append({**sh, 'radial_label': orb_name})
        # Skip px, py, dxy, dxz, dyz, dx2-y2, f±1, f±2, f±3

    return filtered


# ── OpenCL radial evaluator ───────────────────────────────────────────────────

class OpenCLRadialBasis:
    """
    OpenCL-accelerated radial basis function evaluator.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        PySCF molecule with basis set
    device_type : cl.device_type
        Device type (GPU by default, can be CPU for testing)
    """

    def __init__(self, mol, device_type=None):
        # Extract basis parameters
        all_shells = extract_basis_params(mol)
        self.shells = filter_radial_shells(all_shells)
        self.n_shells = len(self.shells)

        if self.n_shells == 0:
            raise ValueError("No radial shells found in molecule")

        # Flatten basis data for OpenCL
        self._exps_flat = np.concatenate([sh['exps'] for sh in self.shells])
        self._coefs_flat = np.concatenate([sh['coefs'] for sh in self.shells])
        self._shell_offsets = np.array(
            [0] + [len(sh['exps']) for sh in self.shells[:-1]]
        )
        # Fix: cumulative offsets
        offsets = [0]
        for sh in self.shells[:-1]:
            offsets.append(offsets[-1] + len(sh['exps']))
        self._shell_offsets = np.array(offsets, dtype=np.int32)

        self._shell_nprim = np.array(
            [len(sh['exps']) for sh in self.shells], dtype=np.int32
        )
        self._shell_l = np.array(
            [sh['l'] for sh in self.shells], dtype=np.int32
        )
        self._shell_norm = np.array(
            [sh['norm'] for sh in self.shells], dtype=np.float32
        )

        # Set up OpenCL — pick first available device of requested type
        if device_type is None:
            device_type = cl.device_type.GPU

        candidate_devices = []
        for plat in cl.get_platforms():
            for dev in plat.get_devices(device_type):
                candidate_devices.append(dev)

        if not candidate_devices:
            # Fallback: any device
            for plat in cl.get_platforms():
                candidate_devices.extend(plat.get_devices())

        if not candidate_devices:
            raise RuntimeError("No OpenCL devices found")

        self.device = candidate_devices[0]

        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, _KERNEL_SOURCE).build()

    def eval_radial(self, r_grid, return_deriv=False):
        """
        Evaluate R(r) for all shells at radial grid points.

        Parameters
        ----------
        r_grid : (N,) array — radial distances (Bohr)
        return_deriv : bool — also return dR/dr

        Returns
        -------
        R : (N, n_shells) array — radial function values
        dR : (N, n_shells) array — derivatives (only if return_deriv=True)
        """
        r = np.asarray(r_grid, dtype=np.float32)
        N = len(r)

        # Transfer to device
        d_r = cl_array.to_device(self.queue, r)
        d_exps = cl_array.to_device(self.queue, self._exps_flat.astype(np.float32))
        d_coefs = cl_array.to_device(self.queue, self._coefs_flat.astype(np.float32))
        d_offsets = cl_array.to_device(self.queue, self._shell_offsets)
        d_nprim = cl_array.to_device(self.queue, self._shell_nprim)
        d_l = cl_array.to_device(self.queue, self._shell_l)
        d_norm = cl_array.to_device(self.queue, self._shell_norm)

        R_out = cl_array.empty(self.queue, (N, self.n_shells), dtype=np.float32)

        global_size = (N, self.n_shells)

        if return_deriv:
            dR_out = cl_array.empty(self.queue, (N, self.n_shells), dtype=np.float32)
            self.prg.eval_radial_deriv(
                self.queue, global_size, None,
                d_r.data, d_exps.data, d_coefs.data,
                d_offsets.data, d_nprim.data, d_l.data, d_norm.data,
                R_out.data, dR_out.data,
                np.int32(N), np.int32(self.n_shells)
            )
            self.queue.finish()
            return R_out.get(), dR_out.get()
        else:
            self.prg.eval_radial(
                self.queue, global_size, None,
                d_r.data, d_exps.data, d_coefs.data,
                d_offsets.data, d_nprim.data, d_l.data, d_norm.data,
                R_out.data,
                np.int32(N), np.int32(self.n_shells)
            )
            self.queue.finish()
            return R_out.get()

    def eval_radial_list(self, r_grid):
        """
        Evaluate R(r) and return as list of (label, l, R) tuples,
        matching the convention of BasisFunctions.get_radial_functions.

        Returns
        -------
        list of (radial_label, l, R_array)
        """
        R = self.eval_radial(r_grid)
        result = []
        for ish, sh in enumerate(self.shells):
            result.append((sh['radial_label'], sh['l'], R[:, ish]))
        return result

    def get_shell_info(self):
        """Return list of (label, l, nprim) for all shells."""
        return [(sh['radial_label'], sh['l'], len(sh['exps']))
                for sh in self.shells]
