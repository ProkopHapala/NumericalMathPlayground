"""
Mechanical Green-function probing for vibration spectra of a spring-mass truss.


Problem setup
-------------
- Nodes are mass points in 3D (we keep z=0 for planar demos).
- Springs (edges) define stiffness matrix K (assembled from axial springs).
- Masses define diagonal mass matrix M.
- After removing fixed DOFs (Dirichlet), we study frequency response to external forces.

Dynamic stiffness
-----------------
For real driving frequency ω and small damping η>0 we use the shifted operator
    A(ω) = K - (ω + i η)^2 M
Poles of A^{-1} are near the real axis; small η lifts them slightly to keep the
factorization stable while preserving sharp modal peaks.

Probing strategy (dipole-driven)
--------------------------------
We care about how a homogeneous electric field E couples to point charges q_i:
  - The field exerts force f_i = q_i E (same direction for all nodes).
  - This is exactly a dipole probe; no need for deterministic combs.
  - For each ω we form the dipole RHS once (or per Cartesian direction),
    factor A(ω) once (Cholesky), and solve A(ω) U = f.
  - We scan ω near the real axis with small η to sample the vibration spectrum
    and resolve modes that couple strongly to the dipole operator.

Dipole response with point charges
----------------------------------
Given charges q_i and displacement u_i(ω) from the dipole force,
the induced dipole is
    Δp(ω) = Σ_i q_i u_i(ω)
which is the linear-response quantity multiplying the external field E via
interaction energy -E·Δp. Peaks of |Δp(ω)| highlight vibrational modes that
are IR-active under the chosen field direction.

Outputs
-------
mechanical_greens_probing returns:
  - omega: sampled frequencies
  - energy: proxy amplitude per ω (||U||^2 average)
  - dipole: complex dipole response per ω (vector of length 3)
  - n_probes: total probes used

Usage
-----
Call demo_run() for a triangular grid example; tune nx, ny, k_spring, mass,
and η to trade resolution vs speed.

Didactic notes / how to read results
------------------------------------
- Zero/near-zero modes: A free 3D body has 6 rigid modes (3 translations, 3 rotations) ⇒ ω≈0. Our planar springs also leave z-DOFs loose, so additional tiny eigenvalues (~1e-8) appear unless you add out-of-plane stiffness or run with dim=2.
- Spectrum plot: black/gray vertical lines mark eigenfrequencies; selected ones are highlighted (red). The blue curve is |Δp(ω)| from dipole forcing; peaks align with IR-active modes.
- Mode panels: each panel overlays the eigenvector (blue) and the forced response at that eigenfrequency (orange). Agreement indicates the dynamic solve matches the eigensolver at ω=ω_mode.
- Charge pattern: corner quadrupole (+,+,-,-) keeps net charge and dipole zero, so responses reflect quadrupolar coupling instead of rigid drift.
- Damping/stabilize: η shifts poles off the real axis to keep solves stable; stabilize adds a tiny diagonal to prevent singularity for free modes.


"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from Truss import (
    build_triangular_grid,
    grid_edges,
    assemble_stiffness_dense,
    mass_matrix,
    boundary_nodes,
    apply_dirichlet,
)
from TrussSolver import (
    dynamic_stiffness, cholesky_factor, cholesky_solve,
    solve_response, mechanical_greens_probing, expand_displacement,
    assemble_weighted_stiffness, classify_edge,
)
from TrussPlotting import (
    plot_truss_charge, plot_spectrum, plot_modes_with_response,
)

# DEBUG: mechanical probing inspired by GF.py (deterministic probing + Cholesky reuse)

# unlimited line lenght in numpy when printing
np.set_printoptions(linewidth=np.inf)

def build_test_truss( nx=6, ny=6, a=1.0, jitter=0.0, k_spring=1.0, mass_value=1.0, fixed_boundary="bottom", dim=3,):
    pos = build_triangular_grid(nx, ny, a=a, jitter=jitter)
    edges = grid_edges(nx, ny, include_diag=True)
    K = assemble_stiffness_dense(pos, edges, k_spring=k_spring, dim=dim)
    masses = np.full(nx * ny, mass_value)
    M = mass_matrix(masses, dim=dim)
    fixed = boundary_nodes(nx, ny, which=fixed_boundary)
    K_red, M_red, mask = apply_dirichlet(K, M, fixed, dim=dim)
    print(f"#DEBUG build_test_truss ndof_full={K.shape[0]} ndof_red={K_red.shape[0]}")
    return pos, edges, K_red, M_red, mask


def corner_quadrupole_charges(nx, ny, charge_val=1.0):
    """
    Neutral quadrupole: (+,+,-,-) on corners to kill net charge and dipole.
    Layout:
        (0,ny-1) : +q
        (nx-1,0) : +q
        (0,0)    : -q
        (nx-1,ny-1): -q
    """
    charges = np.zeros(nx * ny)
    idx = lambda ix, iy: iy * nx + ix
    charges[idx(0   ,    0)] = -charge_val
    charges[idx(0   , ny-1)] = charge_val
    charges[idx(nx-1, 0   )] = -charge_val
    charges[idx(nx-1, ny-1)] = charge_val
    return charges

def parse_indices(s):
    if s is None or s == "":
        return []
    return [int(x) for x in s.split(",") if x.strip() != ""]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dipole-driven vibration probing of a spring-mass grid.")
    p.add_argument("--nx", type=int, default=3, help="Grid size in x")
    p.add_argument("--ny", type=int, default=3, help="Grid size in y")
    p.add_argument("--a", type=float, default=1.0, help="Lattice spacing")
    p.add_argument("--jitter", type=float, default=0.02, help="Position jitter")
    p.add_argument("--k1", type=float, default=5.0, help="Spring stiffness along x")
    p.add_argument("--k2", type=float, default=5.0, help="Spring stiffness along y")
    p.add_argument("--kdiag", type=float, default=5.0, help="Spring stiffness on diagonals")
    p.add_argument("--mass", type=float, default=1.0, help="Mass per node")
    p.add_argument("--charge", type=float, default=1.0, help="Corner charge magnitude (quadrupole)")
    p.add_argument("--theta", type=float, default=90.0, help="Dipole angle in degrees (0=x, 90=y)")
    p.add_argument("--fmin", type=float, default=0.1, help="Min frequency")
    p.add_argument("--fmax", type=float, default=6.0, help="Max frequency")
    p.add_argument("--nfreq", type=int, default=1000, help="Number of frequency samples")
    p.add_argument("--eta", type=float, default=0.001, help="Damping shift")
    p.add_argument("--stabilize", type=float, default=1e-6, help="Diagonal stabilizer")
    p.add_argument("--eig_idx", type=str, default="15,16,17,18,19", help="Comma indices of eigenmodes to plot")
    p.add_argument("--verbosity", type=int, default=2, help="Verbosity level; >2 prints eigenvalues")
    args = p.parse_args()

    nx, ny = args.nx, args.ny
    pos = build_triangular_grid(nx, ny, a=args.a, jitter=args.jitter)
    edges = grid_edges(nx, ny, include_diag=True)
    k_edges = []
    for (i, j) in edges:
        cls = classify_edge(pos, i, j)
        if cls == "x":
            k_edges.append(args.k1)
        elif cls == "y":
            k_edges.append(args.k2)
        else:
            k_edges.append(args.kdiag)
    K = assemble_weighted_stiffness(pos, edges, k_edges, dim=3)
    masses = np.full(nx * ny, args.mass)
    M = mass_matrix(masses, dim=3)
    fixed = boundary_nodes(nx, ny, which="none")
    K_red, M_red, mask = apply_dirichlet(K, M, fixed, dim=3)

    omegas = np.linspace(args.fmin, args.fmax, args.nfreq)
    charges_full = corner_quadrupole_charges(nx, ny, charge_val=args.charge)
    node_mask = mask.reshape(pos.shape[0], 3)[:, 0]
    charges = charges_full[node_mask]
    theta = np.deg2rad(args.theta)
    direction_vec = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64)

    res = mechanical_greens_probing(K_red, M_red, omegas, eta=args.eta, direction_vec=direction_vec, charges=charges, stabilize=args.stabilize)
    omega_idx = min(len(omegas) - 1, max(0, args.nfreq // 4))
    disp_red = solve_response(K_red, M_red, omegas[omega_idx], eta=args.eta, charges=charges, direction_vec=direction_vec, dim=3, stabilize=args.stabilize)
    disp_full = expand_displacement(disp_red, mask, dim=3)
    plot_truss_charge(pos, edges, charges_full, disp=disp_full, scale=0.15, title=f"omega={omegas[omega_idx]:.2f} dipole response")
    # eigen-spectrum for comparison
    A = np.linalg.inv(M_red) @ K_red
    w, V = np.linalg.eigh(A)
    freq = np.sqrt(np.clip(w, 0.0, None))
    if args.verbosity > 1:
        print("#DEBUG eig freq:", freq)
    sel_idx = [i for i in parse_indices(args.eig_idx) if i >= 0 and i < len(freq)]
    resp_red_list = []
    eig_full_list = []
    resp_full_list = []
    for m in sel_idx:
        vec_red = V[:, m].reshape(-1, 3)
        eig_full_list.append(expand_displacement(vec_red, mask, dim=3))
        resp_red = solve_response(K_red, M_red, freq[m], eta=args.eta, charges=charges, direction_vec=direction_vec, dim=3, stabilize=args.stabilize)
        resp_full_list.append(expand_displacement(resp_red, mask, dim=3))
    plot_spectrum(omegas, res, eigfreq=freq, sel=sel_idx)
    plot_modes_with_response(pos, edges, charges_full, eig_full_list, resp_full_list, freq, sel_idx, scale=0.15)
    plt.show()
    print("#DEBUG spectra sample", res["energy"][:5])
