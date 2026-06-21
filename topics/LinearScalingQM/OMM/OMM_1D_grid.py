"""
OMM_1D_grid.py — 1D grid-based Orbital Minimization Method solver.
Optimizes localized orbitals on a 1D chain with Jacobi-style orthogonalization
and optional support truncation. Serves as a simple CPU prototype for the GPU version.
"""
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# unlimited line lenght in numpy when printing
np.set_printoptions(linewidth=np.inf)

class OMM1DSolver:
    w_clip = 0.05  # default window clip threshold for coef sync
    def __init__(self, n_grid=200, l_max=20.0):
        self.n = n_grid
        self.x = np.linspace(0, l_max, n_grid)
        self.dx = self.x[1] - self.x[0]
        
        # 1. Setup Hamiltonian: Kinetic (Finite Difference) + Potential
        # T = -0.5 * d^2/dx^2
        self.main_diag = np.ones(self.n) / (self.dx**2)
        self.off_diag = -0.5 * np.ones(self.n - 1) / (self.dx**2)
        self.H = np.diag(self.main_diag) + np.diag(self.off_diag, k=1) + np.diag(self.off_diag, k=-1)
        
        # V = harmonic well in the middle
        self.V = 0.5 * 0.5 * (self.x - l_max/2)**2
        np.fill_diagonal(self.H, np.diag(self.H) + self.V)

    def build_window(self, mask, center_idx):
        """Smooth window: w = (1 - (r/Rc)^4)^3, C^4 continuous at boundary."""
        r = np.abs(self.x[mask] - self.x[center_idx])
        Rc = (self.x[mask[-1]] - self.x[mask[0]]) / 2.0
        t = np.clip(r / Rc, 0, 1)
        return (1.0 - t**4)**3

    def setup_orbitals(self, n_orb=4, support_size=40, use_window=True, lambda_reg=0.01):
        """Initialize localized orbitals with fixed masks and optional window function."""
        self.n_orb = n_orb
        self.support_size = support_size
        self.use_window = use_window
        self.lambda_reg = lambda_reg
        self.orbitals = []
        self.masks = []
        self.windows = []
        self.coefs = []
        
        # Space centers evenly
        centers = np.linspace(self.n*0.3, self.n*0.7, n_orb).astype(int)
        
        for c in centers:
            mask = np.arange(max(0, c - support_size//2), min(self.n, c + support_size//2))
            self.masks.append(mask)
            
            # Window function (or flat if disabled)
            w = self.build_window(mask, c) if use_window else np.ones(len(mask))
            self.windows.append(w)
            
            # Raw coefficients (Gaussian blob), normalized with windowed norm
            c_vals = np.exp(-0.5 * ((self.x[mask] - self.x[c])/(self.dx*5))**2)
            c_vals /= np.sqrt(np.sum(w**2 * c_vals**2))
            self.coefs.append(c_vals)
            
            # phi = w * c on full grid
            phi = np.zeros(self.n)
            phi[mask] = w * c_vals
            self.orbitals.append(phi)
    
    def sync_orbitals(self):
        """Reconstruct phi_i = w_i * c_i on full grid from raw coefficients."""
        for i in range(self.n_orb):
            self.orbitals[i][:] = 0.0
            self.orbitals[i][self.masks[i]] = self.windows[i] * self.coefs[i]
    
    def _sync_coefs_from_orbitals(self):
        """Extract c_i from phi_i via regularized least-squares: c = phi*w/(w²+eps).
        This damps c where w is small, preventing boundary blow-up."""
        eps = self.w_clip**2  # regularization proportional to clip threshold
        for i in range(self.n_orb):
            mi = self.masks[i]
            w = self.windows[i]
            phi = self.orbitals[i][mi]
            self.coefs[i] = phi * w / (w**2 + eps)
    
    def normalize_coefs(self, mode="windowed"):
        """Normalize c_i. mode='windowed': ||w*c||=1, mode='raw': ||c||=1."""
        for i in range(self.n_orb):
            w = self.windows[i]
            c = self.coefs[i]
            if mode == "raw":
                norm = np.linalg.norm(c)
            else:
                norm = np.sqrt(np.sum(w**2 * c**2))
            if norm > 1e-12:
                self.coefs[i] = c / norm
    
    def compute_overlap_matrix(self):
        """Compute O_{ij} = <phi_i|phi_j> for all orbital pairs."""
        O = np.zeros((self.n_orb, self.n_orb))
        for i in range(self.n_orb):
            for j in range(i, self.n_orb):
                O[i,j] = np.dot(self.orbitals[i], self.orbitals[j])
                O[j,i] = O[i,j]
        return O
    
    def _cg_solve(self, O, b, n_iter=50, tol=1e-10):
        """Simple CG solver for O·x = b. For GPU port, replace O@p with implicit matvec."""
        x = np.zeros_like(b)
        r = b - O @ x
        p = r.copy()
        rsold = np.dot(r, r)
        n_iters = 0
        for _ in range(n_iter):
            n_iters += 1
            Op = O @ p
            denom = np.dot(p, Op)
            if abs(denom) < 1e-16:
                break
            alpha = rsold / denom
            x += alpha * p
            r -= alpha * Op
            rsnew = np.dot(r, r)
            if np.sqrt(rsnew) < tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        self._last_cg_iters = n_iters
        return x
    
    def projected_gradient_step(self, step_size=0.01):
        """
        Projected gradient step with exact constraint projection.
        Solves O·lambda = G via CG, applies windowed gradient + coefficient regularization.
        Replaces energy_step + orthogonalize with a single coupled step.
        """
        # 1. Energy gradients g_i = 2*H*phi_i (reuse existing H)
        grads = [2.0 * (self.H @ self.orbitals[i]) for i in range(self.n_orb)]
        
        # 2. Overlap matrix O_{ij}
        O = self.compute_overlap_matrix()
        
        # 3. RHS: G_{ik} = <g_i|phi_k>
        G = np.array([[np.dot(grads[i], self.orbitals[k]) for k in range(self.n_orb)]
                       for i in range(self.n_orb)])
        
        # 4. Solve O·lambda_i = G_i for each i via CG
        lambda_mat = np.zeros((self.n_orb, self.n_orb))
        max_cg_iters = 0
        for i in range(self.n_orb):
            lambda_mat[i] = self._cg_solve(O, G[i])
            max_cg_iters = max(max_cg_iters, self._last_cg_iters)
        self._last_max_cg_iters = max_cg_iters
        
        # 5. Projected gradient + window + regularization -> update c_i
        for i in range(self.n_orb):
            mi = self.masks[i]
            w = self.windows[i]
            pg = grads[i].copy()
            for j in range(self.n_orb):
                pg -= lambda_mat[i,j] * self.orbitals[j]
            grad_c = w * pg[mi] + self.lambda_reg * self.coefs[i]
            self.coefs[i] -= step_size * grad_c
        
        # 6. Reconstruct and normalize
        self.normalize_coefs()
        self.sync_orbitals()
        
        # 7. Constraint correction: Jacobi sweeps to reduce overlap
        #    (projected gradient ensures tangent direction, but finite step
        #     leaves the constraint manifold — this pulls us back)
        self.orthogonalize(n_iter=3, damping=0.5)
        self._sync_coefs_from_orbitals()
        self.normalize_coefs()
        self.sync_orbitals()

    def penalty_gradient_step(self, step_size=0.01, K=1.0, n_correct=0):
        """
        Simultaneous energy + orthogonality penalty step (no inner loop).
        Minimizes L = E + K * sum_{i<j} O_ij^2 in one shot.
        K controls the orthogonality/energy trade-off:
          - Low K: energy dominates, orbitals may overlap (bosonic collapse)
          - High K: orthogonality dominates, slower energy convergence
        This is the Car-Parrinello analogue: don't solve constraints exactly
        each step, just let the penalty force push toward orthogonality.
        n_correct: optional Jacobi correction sweeps after the penalty step
          (0 = pure simultaneous, >0 = hybrid: penalty + light correction)
        """
        # 1. Energy gradients g_i = 2*H*phi_i
        grads = [2.0 * (self.H @ self.orbitals[i]) for i in range(self.n_orb)]
        
        # 2. Overlap matrix O_{ij}
        O = self.compute_overlap_matrix()
        
        # 3. Combined gradient: energy + K * penalty + lambda_reg * ||c||²
        for i in range(self.n_orb):
            mi = self.masks[i]
            w = self.windows[i]
            pg = grads[i].copy()
            for j in range(self.n_orb):
                if i == j: continue
                pg += 2.0 * K * O[i,j] * self.orbitals[j]
            grad_c = w * pg[mi] + 2.0 * self.lambda_reg * self.coefs[i]
            self.coefs[i] -= step_size * grad_c
        
        # 4. Normalize ||phi||=1 by uniformly scaling c (not element-wise phi/w)
        self.sync_orbitals()
        for i in range(self.n_orb):
            nrm = np.linalg.norm(self.orbitals[i])
            if nrm > 1e-12:
                self.coefs[i] /= nrm
        self.sync_orbitals()
        
        # 5. Optional light constraint correction
        if n_correct > 0:
            self.orthogonalize(n_iter=n_correct, damping=0.5)
            self._sync_coefs_from_orbitals()  # regularized extraction
            self.sync_orbitals()
            # Normalize ||phi||=1 by uniformly scaling c
            for i in range(self.n_orb):
                nrm = np.linalg.norm(self.orbitals[i])
                if nrm > 1e-12:
                    self.coefs[i] /= nrm
            self.sync_orbitals()

    def _flatten_coefs(self):
        """Flatten all coefficient vectors into one 1D array."""
        return np.concatenate([self.coefs[i] for i in range(self.n_orb)])
    
    def _unflatten_coefs(self, x):
        """Unflatten 1D array back into per-orbital coefficient vectors."""
        idx = 0
        for i in range(self.n_orb):
            n = len(self.coefs[i])
            self.coefs[i] = x[idx:idx+n].copy()
            idx += n
    
    def _penalty_loss_and_grad(self, K):
        """Compute penalized loss L = E + K*sum O_ij^2 + lambda_reg*sum||c_i||^2
        and its gradient w.r.t. flattened coefficient vector.
        Returns (loss, grad_flat)."""
        self.sync_orbitals()
        
        # Energy
        E = 0.0
        grads = []
        for i in range(self.n_orb):
            Hphi = self.H @ self.orbitals[i]
            E += np.dot(self.orbitals[i], Hphi)
            grads.append(2.0 * Hphi)  # dE/dphi_i
        
        # Overlap matrix and penalty
        O = self.compute_overlap_matrix()
        penalty = 0.0
        for i in range(self.n_orb):
            for j in range(i+1, self.n_orb):
                penalty += O[i,j]**2
        E_total = E + K * penalty
        
        # Gradient: dL/dc_i = w_i * (dE/dphi_i + dPenalty/dphi_i)[mi] + 2*lambda_reg*c_i
        grad_flat = np.zeros(self._flatten_coefs().shape[0])
        idx = 0
        for i in range(self.n_orb):
            mi = self.masks[i]
            w = self.windows[i]
            pg = grads[i].copy()
            for j in range(self.n_orb):
                if i == j: continue
                pg += 2.0 * K * O[i,j] * self.orbitals[j]
            grad_c = w * pg[mi] + 2.0 * self.lambda_reg * self.coefs[i]
            n = len(grad_c)
            grad_flat[idx:idx+n] = grad_c
            idx += n
        
        # Add regularization to loss
        reg = sum(np.dot(self.coefs[i], self.coefs[i]) for i in range(self.n_orb))
        E_total += self.lambda_reg * reg
        
        return E_total, grad_flat
    
    def cg_minimize_step(self, K=10.0, n_correct=0):
        """Conjugate gradient step on penalized energy L = E + K*sum O_ij^2 + reg.
        Uses Polak-Ribière CG with backtracking line search.
        Much faster convergence than fixed-step gradient descent."""
        x = self._flatten_coefs()
        
        # Compute current loss and gradient
        f, g = self._penalty_loss_and_grad(K)
        
        # Initialize or update CG direction
        if not hasattr(self, '_cg_dir') or self._cg_dir.shape != x.shape:
            self._cg_dir = -g
            self._cg_prev_g = g.copy()
            self._cg_prev_f = f
        else:
            # Polak-Ribière coefficient
            denom = np.dot(self._cg_prev_g, self._cg_prev_g)
            if denom < 1e-16:
                beta = 0.0
            else:
                beta = np.dot(g, g - self._cg_prev_g) / denom
            beta = max(0.0, beta)  # PR+ variant (non-negative)
            
            # Reset if beta ~ 0 (steepest descent restart)
            self._cg_dir = -g + beta * self._cg_dir
            
            # Check if search direction is descent direction
            if np.dot(self._cg_dir, g) >= 0:
                self._cg_dir = -g
        
        # Backtracking line search (Armijo condition)
        alpha = 0.01  # initial step
        c1 = 1e-4     # Armijo constant
        dir_norm = np.dot(self._cg_dir, self._cg_dir)
        if dir_norm < 1e-20:
            return
        
        slope = np.dot(g, self._cg_dir)  # should be negative
        
        self._unflatten_coefs(x + alpha * self._cg_dir)
        f_new, _ = self._penalty_loss_and_grad(K)
        
        # Backtrack until Armijo condition is satisfied
        n_backtrack = 0
        while f_new > f + c1 * alpha * slope and n_backtrack < 20:
            alpha *= 0.5
            self._unflatten_coefs(x + alpha * self._cg_dir)
            f_new, _ = self._penalty_loss_and_grad(K)
            n_backtrack += 1
        
        # Accept step
        self._cg_prev_g = g.copy()
        self._cg_prev_f = f_new
        
        # Normalize ||phi||=1 by uniformly scaling c
        self.sync_orbitals()
        for i in range(self.n_orb):
            nrm = np.linalg.norm(self.orbitals[i])
            if nrm > 1e-12:
                self.coefs[i] /= nrm
        self.sync_orbitals()
        
        # Optional light constraint correction
        if n_correct > 0:
            self.orthogonalize(n_iter=n_correct, damping=0.5)
            self._sync_coefs_from_orbitals()
            self.sync_orbitals()
            for i in range(self.n_orb):
                nrm = np.linalg.norm(self.orbitals[i])
                if nrm > 1e-12:
                    self.coefs[i] /= nrm
            self.sync_orbitals()
    
    def recenter_supports(self, support_size=None):
        """Shift each support to center-of-mass of |phi|^2 and trim outside region."""
        if support_size is None:
            support_size = self.support_size
        half = support_size // 2
        for i in range(self.n_orb):
            phi = self.orbitals[i]
            rho = phi * phi
            norm_rho = rho.sum()
            if norm_rho < 1e-14:
                print(f"#DEBUG recenter skipped orb {i} (zero norm)")
                continue
            cog = (self.x * rho).sum() / norm_rho
            idx = int(np.clip(np.searchsorted(self.x, cog), 0, self.n - 1))
            new_start = max(0, idx - half)
            new_end = min(self.n, idx + half)
            new_mask = np.arange(new_start, new_end)
            old_center = self.masks[i][len(self.masks[i])//2] if len(self.masks[i]) else -1
            if old_center != idx:
                print(f"#DEBUG recenter orb {i}: center {old_center} -> {idx} (cog {cog:.3f})")
            
            # Build new window for new mask
            w_new = self.build_window(new_mask, idx) if self.use_window else np.ones(len(new_mask))
            
            # Extract coefficients from old orbital via regularized least-squares
            eps = self.w_clip**2
            new_c = phi[new_mask] * w_new / (w_new**2 + eps)
            
            self.coefs[i] = new_c
            self.windows[i] = w_new
            self.masks[i] = new_mask
            
            # Sync orbital to full grid
            self.orbitals[i] = np.zeros(self.n)
            self.orbitals[i][new_mask] = w_new * new_c
            # Normalize ||phi||=1 by uniformly scaling c
            nrm = np.linalg.norm(self.orbitals[i])
            if nrm > 1e-12:
                self.coefs[i] /= nrm
                self.orbitals[i] /= nrm

    def orthogonalize(self, n_iter=10, damping=0.5):
        """
        The 'Projective' Constraint Solver.
        Forces <phi_i | phi_j> = 0 and <phi_i | phi_i> = 1
        """
        for _ in range(n_iter):
            # We use Gauss-Seidel style (immediate update) for stability
            for i in range(self.n_orb):
                mi = self.masks[i]
                
                # 1. Orthogonalization Step (Push away from neighbors)
                # Correction: d_phi_i = - sum_{j!=i} <phi_i | phi_j> * phi_j
                correction = np.zeros(self.n)
                for j in range(self.n_orb):
                    if i == j: continue
                    
                    overlap = np.dot(self.orbitals[i], self.orbitals[j])
                    # Only project within our own support
                    correction[mi] += overlap * self.orbitals[j][mi]
                
                self.orbitals[i][mi] -= damping * correction[mi]
                
                # 2. Normalization Step (Project onto unit sphere)
                norm = np.linalg.norm(self.orbitals[i])
                if norm > 1e-9:
                    self.orbitals[i] /= norm
                    
    def energy_step(self, step_size=0.01):
        """Gradient descent on the energy functional: E = sum <phi|H|phi>"""
        for i in range(self.n_orb):
            mi = self.masks[i]
            # Gradient of <phi|H|phi> is 2 * H * phi
            grad = 2.0 * (self.H @ self.orbitals[i])
            
            # Update only within support
            self.orbitals[i][mi] -= step_size * grad[mi]

    def get_stats(self):
        """Calculate total energy and max overlap error."""
        total_e = 0
        max_overlap = 0
        for i in range(self.n_orb):
            total_e += np.dot(self.orbitals[i], self.H @ self.orbitals[i])
            for j in range(i + 1, self.n_orb):
                max_overlap = max(max_overlap, abs(np.dot(self.orbitals[i], self.orbitals[j])))
        return total_e, max_overlap

    def density_from_orbitals(self):
        """Return electron density rho(x) from current (normalized) orbitals."""
        psi = np.array(self.orbitals)
        rho = np.sum(psi * psi, axis=0)
        return rho

    def reference_ground_state(self, n_occ=None):
        """
        Solve the banded eigenproblem with SciPy eigh_tridiagonal.
        Returns eigenvalues, eigenvectors, and ground-state density.
        """
        n_occ = n_occ if n_occ is not None else self.n_orb
        main = self.main_diag + self.V
        evals, evecs = eigh_tridiagonal(main, self.off_diag, select="i", select_range=(0, n_occ - 1))
        rho = np.sum(evecs[:, :n_occ] ** 2, axis=1)
        return evals, evecs, rho

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="1D orbital minimization vs banded reference")
    ap.add_argument("--n_grid",       type=int,   default=100,   help="number of grid points")
    ap.add_argument("--l_max",        type=float, default=30.0,  help="box length")
    ap.add_argument("--n_orb",        type=int,   default=3,     help="number of occupied orbitals")
    ap.add_argument("--support_size", type=int,   default=20,    help="mask size for each orbital")
    ap.add_argument("--n_iter",       type=int,   default=1000,   help="OMM outer iterations")
    ap.add_argument("--step_size",    type=float, default=0.01,  help="energy gradient step")
    ap.add_argument("--ortho_iter",   type=int,   default=5,     help="orthogonalization iterations per step")
    ap.add_argument("--ortho_damp",   type=float, default=0.4,   help="orthogonalization damping")
    ap.add_argument("--report_every", type=int,   default=20,    help="print energy/overlap every k steps")
    ap.add_argument("--compare_ref",  type=int,   default=1,     help="solve reference banded eigenproblem and compare densities")
    ap.add_argument("--recenter_every", type=int, default=1,     help="recenter supports every k steps (0 to disable)")
    ap.add_argument("--mode",        type=str,   default="cg",   help="mode: 'jacobi', 'cg' (projected gradient), 'penalty' (simultaneous), 'cg_min' (CG energy minimizer)")
    ap.add_argument("--use_window",  type=int,   default=1,      help="use smooth window function (1) or hard cutoff (0)")
    ap.add_argument("--lambda_reg",  type=float, default=0.01,   help="coefficient regularization for windowed mode")
    ap.add_argument("--K",           type=float, default=1.0,    help="orthogonality penalty stiffness (penalty mode only)")
    ap.add_argument("--n_correct",   type=int,   default=0,      help="Jacobi correction sweeps in penalty mode (0=pure simultaneous)")
    ap.add_argument("--w_clip",      type=float, default=-1,      help="window clip threshold for coef sync (default: 0.05 for cg/jacobi, 0.2 for penalty)")
    ap.add_argument("--no_plot",      type=int,   default=0,     help="skip matplotlib plots")
    args = ap.parse_args()

    solver = OMM1DSolver(n_grid=args.n_grid, l_max=args.l_max)
    if args.w_clip < 0:
        solver.w_clip = 0.05
    else:
        solver.w_clip = args.w_clip
    solver.setup_orbitals(n_orb=args.n_orb, support_size=args.support_size,
                          use_window=bool(args.use_window), lambda_reg=args.lambda_reg)

    print(f"#{args.mode} window={bool(args.use_window)} lambda_reg={args.lambda_reg}" + (f" K={args.K}" if args.mode in ("penalty", "cg_min") else ""))
    if args.mode == "cg":
        print(f"{'Iter':>5} | {'Energy':>10} | {'Max Overlap':>12} | {'CG iters':>8} | {'Time/iter (ms)':>14}")
        print("-" * 62)
    else:
        print(f"{'Iter':>5} | {'Energy':>10} | {'Max Overlap':>12} | {'Time/iter (ms)':>14}")
        print("-" * 49)

    t_start = time.perf_counter()
    for i in range(args.n_iter):
        t0 = time.perf_counter()
        if args.mode == "cg":
            solver.projected_gradient_step(step_size=args.step_size)
        elif args.mode == "penalty":
            solver.penalty_gradient_step(step_size=args.step_size, K=args.K, n_correct=args.n_correct)
        elif args.mode == "cg_min":
            solver.cg_minimize_step(K=args.K, n_correct=args.n_correct)
        else:
            solver.energy_step(step_size=args.step_size)
            solver.orthogonalize(n_iter=args.ortho_iter, damping=args.ortho_damp)
        t1 = time.perf_counter()
        if args.recenter_every > 0 and (i % args.recenter_every == 0):
            solver.recenter_supports()
        if args.report_every > 0 and (i % args.report_every == 0):
            e, err = solver.get_stats()
            dt_ms = (t1 - t0) * 1000
            if args.mode == "cg":
                cg_it = getattr(solver, '_last_max_cg_iters', 0)
                print(f"{i:5d} | {e:10.5f} | {err:12.6e} | {cg_it:8d} | {dt_ms:14.3f}")
            else:
                print(f"{i:5d} | {e:10.5f} | {err:12.6e} | {dt_ms:14.3f}")
    t_total = time.perf_counter() - t_start

    rho_omm = solver.density_from_orbitals()

    if args.compare_ref:
        evals, evecs, rho_ref = solver.reference_ground_state(n_occ=args.n_orb)
        diff = rho_omm - rho_ref
        l2 = np.linalg.norm(diff)
        max_abs = np.max(np.abs(diff))
        total_charge_diff = np.sum(diff) * solver.dx
        print(f"#INFO ref_eigs_min={evals.min():.6f} ref_eigs_max={evals.max():.6f}")
        print(f"#INFO density L2={l2:.6e} max|diff|={max_abs:.6e} integral_diff={total_charge_diff:.6e}")
    print(f"#INFO total_time={t_total:.4f}s avg_per_iter={t_total/args.n_iter*1000:.3f}ms")

    if not args.no_plot:
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Orbital shapes
        ax[0].plot(solver.x, solver.V * 0.1, 'k--', alpha=0.3, label='Potential (scaled)')
        for i in range(solver.n_orb):
            ax[0].fill_between(solver.x, solver.orbitals[i], alpha=0.5, label=f'OMM Orbital {i}')
        if args.compare_ref:
            for i in range(args.n_orb):
                ax[0].plot(solver.x, evecs[:, i], lw=1.2, label=f'Ref Orbital {i}')
        ax[0].set_ylabel("ψ")
        ax[0].legend(ncol=2, fontsize=8)
        ax[0].set_title("Occupied orbitals (OMM filled; reference lines)")

        # Density comparison
        ax[1].plot(solver.x, rho_omm, 'k-', lw=2, label='Density OMM')
        if args.compare_ref:
            ax[1].plot(solver.x, rho_ref, 'r--', lw=1.5, label='Density reference')
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("ρ")
        ax[1].legend()
        ax[1].set_title("Electron density comparison")

        # DEBUG: visualize current supports as light spans
        colors = plt.cm.tab10.colors
        for i, mi in enumerate(solver.masks):
            if len(mi) == 0:
                continue
            x0 = solver.x[mi[0]]
            x1 = solver.x[mi[-1]]
            ax[0].axvspan(x0, x1, color=colors[i % len(colors)], alpha=0.1, lw=0)
            ax[0].axvline(x0, color=colors[i % len(colors)], ls=":", lw=0.8)
            ax[0].axvline(x1, color=colors[i % len(colors)], ls=":", lw=0.8)

        plt.tight_layout()
        plt.show()

