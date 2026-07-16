https://chatgpt.com/share/6a58903d-9f88-83eb-9bed-4ff8a4517197

---

# USER

I'm thinking about computational method for full atom forcefield using Multi-Grid method. Take for example planar molecule like PTCDA or pentacene (although other molecules are also possible, we should generalize it later). 
Now this molecule has strong interatomic degrees of freedom (bonds, angles) and soft collective modes like bending and twisting molecular plane. These soft DOFs can be described by torsions and digedrals but that is costly (typically UFF or Amber forcefield spend much more computation of torsions or dihedral then bonds and angles).
Another possibility is to explicitly precompute few soft modes of global hessian.

But I think about more geometric and intuitive approach. Let's consider two rigid body rotation frames (or more in future, bit for ptcda and pentacene start with 2) which we place near the ends of the molecule. Then we define some interpolation weights attributing the position of atoms to these to rigid control frames, one will primarily control rotation of one molecule end the other. Then we fit the torsions and bending constant only in terms of angles (resp. Normalized direction vectors or quaternions) defining these two rotation frames. This should reproduce the full hessian soft modes as close as possible.

There should be rigorous mathematical construction how to choose optimal atomic interpolation weights weights and rigid body alignment stiffness to how to best approximate the low energy flexibility of the molecule in these reduced representation (6+6=12DOF instead of 3N in full atomistic basis which is like 120DOFs in case of PTCDA)

Now the key is not the reduction of size but removing hard dogs from the dynamics which allows much faster simulation (large time step since dtmax=sqrt(m\k) where m and k are mass and stiffness of the hardest DOF. We take only the soft and heavy DOFs.

Now the question is if this give any advantage wrt taking directly the soft modes of Hessian?
Not sure but at least the memory footprint is lower. In our case we have shared weights each of the two rigid control frames has one set of weights which it use for all the 6 modes (3 translation and 3 rotation) so it means 2N weights in total while if we use directly the soft DOFs we have 12*N weights. 

Another question is if I can still use full atom dynamics (with the hard bonds and angles) to refine the coarse grained smoothened dynamics. I think I can there should be rigorous multigrid methodology for that.

---

# Chat GPT 5.6 sol hi

You are right: the previous answer identified the correct conceptual neighborhood but did not develop the method far enough to be implementable.

What you propose can be formulated as an **atomistically fitted nonlinear reduced force field built from overlapping rigid control frames**. Its strongest form combines:

[
\text{rigid-frame interpolation}
+
\text{Hessian/subspace fitting}
+
\text{atomistic energy condensation}
+
\text{coarse–fine correction}.
]

The closest established ingredients are rotation–translation-of-blocks normal-mode analysis, nonlinear rigid-block modes, geometrically exact (SE(3)) beam elements, component-mode synthesis, and energy-minimizing multigrid interpolation. I did not find a method that combines all of them in exactly your form: a few overlapping rigid frames, one shared scalar weight per atom, atomistically fitted frame stiffness, and optional atomistic refinement. RTB uses rigid blocks but normally assigns atoms discretely to blocks; NOLB gives finite nonlinear rotations of such blocks; your continuously weighted overlapping-frame map is closer to physically fitted skinning or a molecular beam element. ([PubMed][1])

---

# 1. Configuration and degrees of freedom

Let the molecule have (N) atoms, reference positions

[
\mathbf r_i^0,\qquad i=1,\ldots,N,
]

and masses (m_i).

For two control frames, let the reference poses be

[
G_a^0=(R_a^0,\mathbf c_a^0)\in SE(3),
\qquad a=1,2,
]

and current poses

[
G_a=(R_a,\mathbf c_a).
]

Each frame has six coordinates, so there are nominally twelve. But a common rigid transformation of both frames is the global molecular translation and rotation. Consequently,

[
n_{\mathrm{internal}}=12-6=6.
]

For (K) frames, the maximum number of internal coordinates is

[
6K-6=6(K-1).
]

The word “maximum” matters: poor or degenerate weights can reduce the actual rank.

For PTCDA or pentacene:

* two frames can represent one spatial wavelength of extension, shear, bending and twisting;
* three frames give twelve internal coordinates and can represent a central bow or two bending segments;
* four frames are likely necessary for saddle-like plate modes or independent deformation across both molecular length and width.

---

# 2. A frame interpolation that is exactly objective

Define the deformation transform associated with frame (a),

[
D_a=G_a(G_a^0)^{-1}.
]

Acting on a reference point,

[
D_a\mathbf r
============

\mathbf c_a
+
R_aR_a^{0T}
(\mathbf r-\mathbf c_a^0).
]

Define

[
A_a=R_aR_a^{0T}.
]

Then the candidate position of atom (i) predicted by frame (a) is

[
\mathbf p_{ia}
==============

\mathbf c_a+A_a(\mathbf r_i^0-\mathbf c_a^0).
]

## 2.1 Linear-blend frame interpolation

The cheapest map is

[
\boxed{
\mathbf r_i
===========

\sum_{a=1}^{K}w_{ia}\mathbf p_{ia}
}
]

with

[
w_{ia}\ge 0,
\qquad
\sum_a w_{ia}=1.
]

For two frames, only one independent number is needed:

[
w_{i1}=1-s_i,\qquad w_{i2}=s_i.
]

Thus

[
\boxed{
\mathbf r_i
===========

(1-s_i)\mathbf p_{i1}+s_i\mathbf p_{i2}
}
]

with (s_i\in[0,1]).

This satisfies the essential objectivity condition. If both frames undergo the same global rigid motion (S),

[
G_a=SG_a^0,
]

then

[
D_1=D_2=S
]

and therefore

[
\mathbf r_i=S\mathbf r_i^0.
]

No internal strain energy should be generated by a global translation or rotation.

## 2.2 Geometrically nonlinear (SE(3)) interpolation

Linear blending can artificially contract the molecule when the two frame rotations differ substantially. A more geometrically faithful map is

[
E=D_1^{-1}D_2,
\qquad
\boldsymbol\xi=\log(E)\in\mathfrak{se}(3),
]

and

[
\boxed{
D_i=D_1\exp(s_i\boldsymbol\xi),
\qquad
\mathbf r_i=D_i\mathbf r_i^0.
}
]

This interpolates the deformation transformation along an (SE(3)) exponential curve.

Important properties are:

[
s_i=0\Rightarrow D_i=D_1,
\qquad
s_i=1\Rightarrow D_i=D_2,
]

and if (D_1=D_2=S), then every (D_i=S).

This is closely related to the exponential interpolation used in geometrically exact (SE(3)) beam elements, where translation and rotation are coupled rather than interpolated independently. ([ScienceDirect][2])

For production, there are three choices:

1. **Linear blend:** cheapest forces and torques.
2. **Normalized dual-quaternion blend:** good finite-rotation behavior and moderate cost.
3. **Exact (SE(3)) exponential interpolation:** cleanest mathematically but more expensive.

Dual-quaternion skinning was developed precisely to blend rigid transformations without the collapse artifacts of ordinary linear blending. It is a computational geometry method rather than a physical force field, but it is directly useful here as a deformation map. ([users.cs.utah.edu][3])

---

# 3. Linearized frame basis

The Hessian fitting can be done entirely in the infinitesimal regime even if the final evaluation uses nonlinear rotations.

Let frame (a) have infinitesimal translation (\delta\mathbf t_a) and rotation (\delta\boldsymbol\theta_a). Around the reference geometry,

[
\delta\mathbf r_i
=================

\sum_a w_{ia}
\left[
\delta\mathbf t_a+
\delta\boldsymbol\theta_a
\times
(\mathbf r_i^0-\mathbf c_a^0)
\right].
]

Using the cross-product matrix,

[
[\mathbf x]_\times\mathbf y=\mathbf x\times\mathbf y,
]

this becomes

[
\delta\mathbf r_i
=================

\sum_a
w_{ia}
\begin{bmatrix}
I & -[\mathbf r_i^0-\mathbf c_a^0]_\times
\end{bmatrix}
\begin{pmatrix}
\delta\mathbf t_a\
\delta\boldsymbol\theta_a
\end{pmatrix}.
]

Stacking all atoms gives

[
\boxed{
\delta\mathbf r=B(\mathbf w),\delta\mathbf q,
}
]

where

[
B\in\mathbb R^{3N\times 6K}.
]

The (3\times6) block for atom (i), frame (a), is

[
B_{ia}
======

w_{ia}
\begin{bmatrix}
I&-[\mathbf r_i^0-\mathbf c_a^0]_\times
\end{bmatrix}.
]

For conditioning, scale rotation coordinates by a characteristic molecular length (\ell). Define

[
\widetilde{\boldsymbol\theta}_a=\ell\boldsymbol\theta_a
]

and use

[
B_{ia}
======

w_{ia}
\begin{bmatrix}
I&-\frac{1}{\ell}
[\mathbf r_i^0-\mathbf c_a^0]_\times
\end{bmatrix}.
]

Then translation and rotation columns have comparable numerical magnitudes.

---

# 4. Preparing the target Hessian modes

Let

[
H=\frac{\partial^2V}{\partial\mathbf r^2}
]

be the (3N\times3N) molecular Hessian and

[
M=\operatorname{diag}(m_1I_3,\ldots,m_NI_3).
]

Solve

[
H\mathbf u_k=\lambda_kM\mathbf u_k,
\qquad
\lambda_k=\omega_k^2,
]

with mass normalization

[
\mathbf u_j^TM\mathbf u_k=\delta_{jk}.
]

For an isolated molecule, remove the six global rigid modes. Construct their basis explicitly:

### Translations

[
\mathbf g_{T,\alpha,i}=\mathbf e_\alpha.
]

### Rotations about the molecular center of mass

[
\mathbf g_{R,\alpha,i}
======================

\mathbf e_\alpha\times
(\mathbf r_i^0-\mathbf r_{\mathrm{COM}}).
]

Mass-orthonormalize these six vectors to form

[
G\in\mathbb R^{3N\times6},
\qquad
G^TMG=I.
]

The (M)-orthogonal projector removing global motion is

[
P_0=I-GG^TM.
]

A useful practical warning: fit the **intramolecular Hessian**, not the Hessian of an adsorbed molecule plus substrate. Otherwise global molecular translations and rotations become mixed with substrate corrugation and vertical contact stiffness.

---

# 5. Rigorous objective for fitting the weights

The mass-orthogonal projector onto the frame space is

[
\Pi_B
=====

B(B^TMB)^{-1}B^TM.
]

A direct subspace-fitting loss is

[
\boxed{
{\cal L}_{\mathrm{mode}}(\mathbf w)
===================================

\sum_{k=1}^{n_{\mathrm{fit}}}
\alpha_k
\left|
(I-\Pi_B)\mathbf u_k
\right|_M^2.
}
]

Here

[
|\mathbf x|_M^2=\mathbf x^TM\mathbf x.
]

The weights (\alpha_k) express what “best” means:

[
\alpha_k=1
]

fits the low-mode subspace uniformly;

[
\alpha_k=\frac{1}{\lambda_k}
]

emphasizes equilibrium thermal displacement variance;

[
\alpha_k=\frac{1}{\lambda_k^2}
]

emphasizes static compliance under generic forces.

I would begin with

[
\alpha_k=\frac{1}{\lambda_k+\lambda_{\mathrm{floor}}},
]

where the floor prevents excessive domination by one exceptionally soft mode.

Do not fit only six target modes. Fit perhaps 10–20 low internal modes with decreasing weights. Otherwise six available frame modes can overfit six particular eigenvectors while behaving poorly immediately above them.

---

# 6. Smoothness and geometric regularization

Unrestricted per-atom (s_i) can become oscillatory: neighboring carbon atoms may be assigned very different weights merely to reproduce a particular eigenvector.

Define the molecular bond graph Laplacian (L_g):

[
(L_g)*{ii}=\sum*{j:(i,j)\in{\cal B}}\gamma_{ij},
]

[
(L_g)*{ij}=-\gamma*{ij}
\quad\text{for bonded }i,j.
]

A simple choice is (\gamma_{ij}=1). A length-scaled choice is

[
\gamma_{ij}=\frac{1}{|\mathbf r_i^0-\mathbf r_j^0|^2}.
]

Add

[
{\cal L}_{\mathrm{smooth}}
==========================

# \lambda_s\mathbf s^TL_g\mathbf s

\lambda_s
\sum_{(i,j)\in{\cal B}}
\gamma_{ij}(s_i-s_j)^2.
]

Also retain a weak prior toward a simple geometric interpolation:

[
{\cal L}_{\mathrm{prior}}
=========================

\lambda_0
\sum_i(s_i-s_i^{(0)})^2.
]

For an elongated molecule,

[
s_i^{(0)}
=========

\operatorname{clamp}
\left[
\frac{
(\mathbf r_i^0-\mathbf r_{\min})\cdot\mathbf e_{\mathrm{long}}
}{
L
},0,1
\right].
]

For symmetric pentacene or PTCDA, impose symmetry directly. If atom (\bar i) is related to (i) by inversion along the molecular long axis,

[
s_{\bar i}=1-s_i.
]

This both improves robustness and reduces the number of free parameters.

A stronger model is to assign one (s) per molecular cross-section or aromatic ring rather than per atom. Then interpolate these values along the molecule. This may be more transferable than a completely free atom-wise map.

---

# 7. Efficient alternating fitting algorithm

The projector objective is nonlinear in the weights, but there is a very simple alternating least-squares solution.

For target mode (k), let the fitted frame motion be

[
\mathbf q_k
===========

{
\mathbf v_{1k},\boldsymbol\omega_{1k},
\mathbf v_{2k},\boldsymbol\omega_{2k}
}.
]

The displacement predicted by frame (a) for atom (i) is

[
\mathbf p_{iak}
===============

\mathbf v_{ak}
+
\boldsymbol\omega_{ak}\times
(\mathbf r_i^0-\mathbf c_a^0).
]

For two frames,

[
\widehat{\mathbf u}_{ik}
========================

(1-s_i)\mathbf p_{i1k}
+s_i\mathbf p_{i2k}.
]

Define

[
\mathbf d_{ik}
==============

\mathbf p_{i2k}-\mathbf p_{i1k}.
]

Then

[
\widehat{\mathbf u}_{ik}
========================

\mathbf p_{i1k}+s_i\mathbf d_{ik}.
]

## Step A: fit frame motions for fixed weights

Construct (B(\mathbf s)). For every target mode,

[
\boxed{
\mathbf q_k
===========

(B^TMB+\epsilon I)^{-1}
B^TM\mathbf u_k.
}
]

This is only a (12\times12) solve.

Use pivoted QR, SVD, or an eigendecomposition rather than blindly forming ill-conditioned normal equations. With properly anchored weights, Cholesky is usually sufficient.

## Step B: fit weights for fixed frame motions

For each atom define

[
D_i
===

m_i
\sum_k
\alpha_k
|\mathbf d_{ik}|^2,
]

and

[
b_i
===

m_i
\sum_k
\alpha_k
\mathbf d_{ik}\cdot
(\mathbf u_{ik}-\mathbf p_{i1k}).
]

The weight update solves the convex quadratic problem

[
\boxed{
\min_{0\le s_i\le1}
\frac12\mathbf s^T
\left[
\operatorname{diag}(D_i)
+\lambda_sL_g
+\lambda_0I
\right]\mathbf s
----------------

\left[
\mathbf b+\lambda_0\mathbf s^{(0)}
\right]^T\mathbf s.
}
]

Without box constraints, this is one sparse SPD solve:

[
\left[
\operatorname{diag}(D_i)
+\lambda_sL_g
+\lambda_0I
\right]\mathbf s
================

\mathbf b+\lambda_0\mathbf s^{(0)}.
]

For (N\sim 30)–100, an active-set box-constrained solver is trivial:

1. solve the unconstrained system;
2. fix values below zero to zero;
3. fix values above one to one;
4. solve again for the remaining free variables;
5. repeat until no violations remain.

## Pseudocode

```text
function fit_two_frame_weights(r0, mass, H, bond_graph):

    U, lambda = lowest_internal_modes(H, mass)

    long_axis = principal_axis(r0, mass)
    s0 = normalized_coordinate_along_axis(r0, long_axis)
    s  = s0

    choose fixed frame centers c1, c2 near molecular ends
    build graph Laplacian Lg

    for outer_iteration = 1 .. max_iter:

        B = build_frame_basis(r0, c1, c2, s)

        A12 = transpose(B) * M * B
        regularize_and_factorize(A12)

        for k = 1 .. nfit:
            rhs  = transpose(B) * M * U[k]
            q[k] = solve(A12, rhs)

        for atom i:
            Di = 0
            bi = 0

            for mode k:
                p1 = q[k].v1
                     + cross(q[k].omega1, r0[i] - c1)

                p2 = q[k].v2
                     + cross(q[k].omega2, r0[i] - c2)

                d = p2 - p1

                Di += alpha[k] * mass[i] * dot(d, d)
                bi += alpha[k] * mass[i]
                      * dot(d, U[k][i] - p1)

            diagonal[i] = Di + lambda0
            rhs_s[i]    = bi + lambda0 * s0[i]

        Aweight = diag(diagonal) + lambda_smooth * Lg

        s_new = solve_box_constrained_QP(
                    Aweight,
                    rhs_s,
                    lower = 0,
                    upper = 1,
                    symmetry_constraints,
                    endpoint_anchors
                )

        if objective_change_small:
            break

        s = line_search_between(s, s_new)

    return s
```

The alternating algorithm monotonically decreases the fitting objective, but the joint problem is not globally convex. Use the long-axis initialization, perhaps a smoothstep initialization, and two or three randomized smooth perturbations as additional starts.

---

# 8. Frame-center choice is mostly a gauge

There is a useful simplification here.

For a single frame, shifting its reference center from (\mathbf c_a^0) to

[
\mathbf c_a^0+\Delta\mathbf c
]

changes

[
\boldsymbol\omega\times
(\mathbf r_i^0-\mathbf c_a^0-\Delta\mathbf c)
]

only by the constant translation

[
-\boldsymbol\omega\times\Delta\mathbf c.
]

Because translations are already present in the six frame coordinates, the linear subspace spanned by one frame does not depend on its chosen center.

Therefore, the Hessian-mode fit cannot uniquely determine the centers. Choose them for conditioning and nonlinear geometry:

[
\mathbf c_a^0
=============

\frac{\sum_i m_iw_{ia}\mathbf r_i^0}
{\sum_i m_iw_{ia}}.
]

The centers begin to matter for finite rotations and for the detailed nonlinear interpolation, but not for the infinitesimal rigid-motion subspace.

---

# 9. Measuring whether the two-frame model is adequate

Remove global rigid motion from the frame basis:

[
B_\perp=(I-GG^TM)B.
]

Mass-orthonormalize its nonzero columns:

[
Q_B^TMQ_B=I.
]

For two frames, (Q_B) should have rank six.

Let (U_s) contain the six target soft internal modes. Compute

[
S=Q_B^TMU_s.
]

The singular values

[
\sigma_k=\operatorname{svd}(S)
]

give the principal angles

[
\theta_k=\arccos\sigma_k.
]

Interpretation:

* (\sigma_k\approx1): the mode is represented accurately;
* (\sigma_k\ll1): an important part lies outside the frame tangent space;
* one or more very small (\sigma_k): two frames are structurally insufficient.

Also report individual projection errors:

[
e_k
===

\frac{
|(I-Q_BQ_B^TM)\mathbf u_k|_M
}{
|\mathbf u_k|_M
}.
]

For a convincing model, I would want the first several thermally important modes to have small errors, rather than only the first one or two.

---

# 10. Three ways to determine the reduced stiffness

Once the geometry or prolongation (P) is fixed, there are several legitimate reduced stiffness matrices. They answer subtly different questions.

Let

[
P\in\mathbb R^{3N\times6}
]

be the internal frame tangent basis after removing global rigid components.

Define

[
M_\eta=P^TMP.
]

## 10.1 Restricted or Galerkin stiffness

The simplest choice is

[
\boxed{
K_\eta^{\mathrm{Ritz}}
======================

P^THP.
}
]

The reduced normal modes satisfy

[
K_\eta^{\mathrm{Ritz}}\mathbf z_k
=================================

\widetilde\lambda_k
M_\eta\mathbf z_k.
]

This is fully consistent with restricting the atomistic harmonic energy to the frame manifold:

[
V(P\boldsymbol\eta)
===================

\frac12
\boldsymbol\eta^TP^THP\boldsymbol\eta.
]

Its disadvantage is that it usually makes the molecule too stiff: atoms are not allowed to relax in directions outside the frame space.

By Rayleigh–Ritz, exact normal modes are the optimal linear subspace for their own eigenvalues. A structured frame space can only approximate them.

## 10.2 Spectrally calibrated stiffness

Suppose the six target modes are

[
U_6=[\mathbf u_1,\ldots,\mathbf u_6]
]

with

[
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_6).
]

Map them into coarse coordinates:

[
A
=

M_\eta^{-1}P^TMU_6.
]

The columns of (A) are the least-squares coarse coordinates corresponding to the atomistic modes.

Mass-orthonormalize them:

[
S=A^TM_\eta A,
]

[
Z=AS^{-1/2}.
]

Then

[
Z^TM_\eta Z=I.
]

Define

[
\boxed{
K_\eta^{\mathrm{spec}}
======================

M_\eta Z\Lambda Z^TM_\eta.
}
]

This guarantees

[
K_\eta^{\mathrm{spec}}Z
=======================

M_\eta Z\Lambda.
]

Therefore, the reduced model has exactly the target six frequencies, provided (A) has full rank.

This is appropriate when matching the soft spectrum is more important than reproducing the atomistic energy on every arbitrary frame displacement.

A warning signal is a poorly conditioned (S). That means several atomistic modes collapse to nearly the same frame deformation and the six-dimensional frame representation is insufficient.

## 10.3 Relaxed or statically condensed stiffness

This is the most rigorous answer to “allow the hard atomic DOFs to refine the coarse deformation.”

Define a coarse extraction operator (C) satisfying

[
CP=I,
\qquad
CG=0.
]

A natural linear choice is

[
C=(P^TMP)^{-1}P^TM.
]

For a prescribed coarse deformation (\boldsymbol\eta), find the minimum-energy atomistic displacement:

[
\mathbf u_*(\boldsymbol\eta)
============================

\arg\min_{\mathbf u}
\frac12\mathbf u^TH\mathbf u
]

subject to

[
C\mathbf u=\boldsymbol\eta,
\qquad
G^TM\mathbf u=0.
]

This defines an energy-minimizing prolongation

[
\mathbf u_*=P_*\boldsymbol\eta.
]

On the internal subspace,

[
\boxed{
P_*
===

H_\perp^{-1}C^T
\left(
CH_\perp^{-1}C^T
\right)^{-1}.
}
]

The condensed stiffness is

[
\boxed{
K_\eta^{\mathrm{relax}}
=======================

# P_*^THP_*

\left(
CH_\perp^{-1}C^T
\right)^{-1}.
}
]

This is the harmonic Schur complement: the coarse deformation is fixed while all other atomistic coordinates relax.

It will generally be softer than

[
P^THP.
]

This construction is very close in spirit to atomistically derived coarse-grained finite elements and to energy-minimizing multigrid interpolation. CGMD derives coarse finite-element dynamics from the underlying atomistic model, while smoothed-aggregation multigrid constructs prolongation using low-energy or zero-energy modes and smooths the interpolation to reduce its energy. ([APS Link][4])

## Practical saddle-point solve

For each unit coarse coordinate (\mathbf e_\alpha), solve

[
\begin{pmatrix}
H&C^T&MG\
C&0&0\
G^TM&0&0
\end{pmatrix}
\begin{pmatrix}
\mathbf p_\alpha\
\boldsymbol\lambda_\alpha\
\boldsymbol\mu_\alpha
\end{pmatrix}
=============

\begin{pmatrix}
0\
\mathbf e_\alpha\
0
\end{pmatrix}.
]

The six vectors (\mathbf p_\alpha) form (P_*).

For a 40-atom molecule, this fitting operation is negligible.

---

# 11. Combining energy-minimizing interpolation with compact weights

The exact relaxed basis (P_*) has (18N) scalar entries for six internal coordinates, so it loses the memory advantage of shared scalar weights.

A good compromise is:

1. calculate the exact relaxed basis (P_*);
2. fit the structured frame basis (P(\mathbf s)) to it;
3. retain only the (s_i).

Use an objective such as

[
\min_{\mathbf s,T}
\left|
P(\mathbf s)T-P_*
\right|_M^2
+
\lambda_s\mathbf s^TL_g\mathbf s,
]

where

[
T\in\mathbb R^{6\times6}
]

allows an arbitrary change of internal coordinates.

This is again alternating least squares:

* fixed (\mathbf s): solve for (T);
* fixed (T): solve for the scalar weights.

This may be more robust than fitting individual Hessian eigenvectors because (P_*) already includes the relaxation of all omitted atomic coordinates.

---

# 12. Nonlinear frame energy

For finite rotations, define the relative deformation of the two frame deformation transforms:

[
E=D_1^{-1}D_2.
]

At equilibrium,

[
D_1=D_2=I,
\qquad
E=I.
]

Take its Lie-algebra logarithm:

[
\boldsymbol\xi
==============

# \log(E)

\begin{pmatrix}
\boldsymbol\rho\
\boldsymbol\phi
\end{pmatrix}.
]

Here:

* (\boldsymbol\rho) is the relative translational or screw displacement;
* (\boldsymbol\phi) is the relative rotation vector.

To avoid mixed units, introduce the frame separation (\ell) and define

[
\boldsymbol\eta
===============

S\boldsymbol\xi,
\qquad
S=
\operatorname{diag}(1,1,1,\ell,\ell,\ell).
]

All components of (\boldsymbol\eta) then have units of length.

The harmonic frame energy is

[
\boxed{
V_{\mathrm{frame}}
==================

\frac12\boldsymbol\eta^TK\boldsymbol\eta.
}
]

The complete symmetric (6\times6) (K) has 21 independent parameters.

Do not initially diagonalize it into “extension, shear, bend, twist springs.” The off-diagonal terms are physically meaningful. In an ordinary beam element, transverse displacement and endpoint rotation are strongly coupled. A diagonal stiffness would incorrectly treat them as independent motions.

Molecular symmetry may later reduce the allowed couplings.

---

# 13. Analytic frame wrenches

Let

[
\mathbf g_\eta=K\boldsymbol\eta,
\qquad
\mathbf g_\xi=S^T\mathbf g_\eta.
]

For body perturbations

[
D_a\rightarrow D_a\exp(\delta\boldsymbol\xi_a),
]

the relative transform changes as

[
E\rightarrow
\exp(-\delta\boldsymbol\xi_1)
E
\exp(\delta\boldsymbol\xi_2).
]

The logarithm differential gives

[
\delta\boldsymbol\xi
====================

-J_l^{-1}(\boldsymbol\xi)\delta\boldsymbol\xi_1
+
J_r^{-1}(\boldsymbol\xi)\delta\boldsymbol\xi_2,
]

where (J_l) and (J_r) are the left and right (SE(3)) Jacobians.

Therefore, the internal generalized wrenches are

[
\boxed{
W_1
===

J_l^{-T}(\boldsymbol\xi)\mathbf g_\xi,
}
]

[
\boxed{
W_2
===

-J_r^{-T}(\boldsymbol\xi)\mathbf g_\xi.
}
]

For small deformation,

[
J_l^{-1}\approx J_r^{-1}\approx I,
]

so

[
W_1\approx+\mathbf g_\xi,
\qquad
W_2\approx-\mathbf g_\xi.
]

For implementation:

* use small-angle series for (|\boldsymbol\phi|<10^{-4});
* use the closed-form (SO(3))/(SE(3)) Jacobians otherwise;
* validate analytic wrenches against central finite differences;
* do not use finite differences in the production force loop.

---

# 14. Efficient atom-position and force evaluation

For the linear-blend map,

[
\mathbf r_i
===========

\sum_a w_{ia}
\left[
\mathbf c_a+A_a(\mathbf r_i^0-\mathbf c_a^0)
\right].
]

Define

[
\mathbf y_{ia}
==============

A_a(\mathbf r_i^0-\mathbf c_a^0).
]

Then

[
\mathbf r_i
===========

\sum_a w_{ia}(\mathbf c_a+\mathbf y_{ia}).
]

Suppose an external or intermolecular atom force (\mathbf f_i) is evaluated at (\mathbf r_i). The exact Jacobian-transpose reduction is

[
\boxed{
\mathbf F_a
===========

\sum_iw_{ia}\mathbf f_i,
}
]

[
\boxed{
\boldsymbol\tau_a
=================

\sum_i
w_{ia}
\mathbf y_{ia}\times\mathbf f_i.
}
]

This is not an approximation: it follows directly from virtual work.

Because

[
\sum_a w_{ia}=1,
]

the total frame force is

[
\sum_a\mathbf F_a
=================

\sum_i\mathbf f_i.
]

Also,

[
\sum_a
\left[
\mathbf c_a\times\mathbf F_a+
\boldsymbol\tau_a
\right]
=======

\sum_i\mathbf r_i\times\mathbf f_i.
]

Thus linear and angular momentum are transferred correctly, provided force reduction uses the exact Jacobian transpose.

## Runtime pseudocode

```text
function evaluate_molecule(frame1, frame2, template):

    A1 = frame1.R * transpose(template.R01)
    A2 = frame2.R * transpose(template.R02)

    for atom i:

        y1 = A1 * (template.r0[i] - template.c01)
        y2 = A2 * (template.r0[i] - template.c02)

        p1 = frame1.c + y1
        p2 = frame2.c + y2

        s  = template.weight[i]

        atom_pos[i] = (1 - s) * p1 + s * p2

        cache y1, y2 if force reduction follows immediately

    atom_force = evaluate_external_and_nonbonded_forces(atom_pos)

    F1 = 0
    F2 = 0
    tau1 = 0
    tau2 = 0

    for atom i:

        s = template.weight[i]
        f = atom_force[i]

        F1 += (1 - s) * f
        F2 += s * f

        tau1 += (1 - s) * cross(y1[i], f)
        tau2 += s       * cross(y2[i], f)

    D1 = frame1.pose * inverse(template.reference_pose1)
    D2 = frame2.pose * inverse(template.reference_pose2)

    relative = inverse(D1) * D2
    xi       = log_SE3(relative)
    eta      = scale_translation_rotation(xi, template.length)

    E_internal = 0.5 * dot(eta, K * eta)

    gxi = transpose(scale_matrix) * K * eta

    W1_internal =  left_Jacobian_inverse_transpose(xi) * gxi
    W2_internal = -right_Jacobian_inverse_transpose(xi) * gxi

    add internal wrenches to F1, tau1, F2, tau2

    return E_internal, frame wrenches, atom positions
```

For many identical molecules, store only once:

* reference atom positions;
* weights;
* reference frame poses;
* the (6\times6) stiffness;
* atom types and interaction parameters.

Per molecule, store only the two current poses and velocities.

---

# 15. Computational cost relative to direct normal modes

For six internal normal modes,

[
\mathbf r_i
===========

\mathbf r_i^0+
\sum_{\mu=1}^{6}
a_\mu\mathbf u_{\mu i}.
]

Storage is

[
6\times3N=18N
]

floating-point numbers.

For the two-frame model, storage is:

* (N) independent scalar weights;
* two reference frame poses;
* 21 stiffness coefficients.

So the memory reduction relative to six arbitrary vector modes is roughly a factor of 18 in the mode representation.

However, the arithmetic comparison is less favorable to frames:

### Six normal modes

Approximately six vector multiply-adds per atom.

### Two-frame linear blend

Two quaternion rotations or two (3\times3) matrix-vector products per atom, followed by blending.

Thus direct normal-mode reconstruction may actually be faster for very small (r). The real advantages of frames are:

* nonlinear finite rotations;
* exact rigid covariance;
* interpretable bending/twisting geometry;
* natural hierarchical refinement;
* shared compact data for many identical molecules.

The memory advantage matters more if many molecular templates must coexist or if the data must pass through constrained GPU caches. For one repeated PTCDA template, both representations are already small.

---

# 16. Kinetic energy and the mass matrix

This is one of the main caveats.

If atoms are reconstructed from the frames, their velocities are

[
\dot{\mathbf r}=J(\mathbf q)\dot{\mathbf q}.
]

The exact atom-derived kinetic energy is

[
T
=

\frac12\dot{\mathbf q}^T
M_q(\mathbf q)
\dot{\mathbf q},
]

where

[
\boxed{
M_q(\mathbf q)
==============

J(\mathbf q)^TMJ(\mathbf q).
}
]

Because atoms are influenced by both frames, (M_q) contains cross-frame blocks. It is not generally the kinetic energy of two independent rigid bodies.

The exact equations are

[
M_q(\mathbf q)\ddot{\mathbf q}
+
\mathbf c(\mathbf q,\dot{\mathbf q})
+
\nabla_{\mathbf q}V
===================

\mathbf Q,
]

where (\mathbf c) contains Coriolis and geometric mass-matrix terms.

You have three implementation levels.

## 16.1 Constant projected mass

At equilibrium,

[
M_q^0=B^TMB.
]

Use it as a constant (12\times12) matrix.

This is accurate for small deformation and extremely cheap. It also guarantees that the harmonic frequencies are computed using the correct projected inertia.

## 16.2 Two independent pseudo-rigid-body masses

Assign atom mass fractions:

[
m_a=\sum_i m_iw_{ia},
]

[
\mathbf c_a^0
=============

\frac{\sum_i m_iw_{ia}\mathbf r_i^0}{m_a},
]

[
I_a
===

\sum_i
m_iw_{ia}
\left[
|\boldsymbol\rho_{ia}|^2I
-------------------------

\boldsymbol\rho_{ia}\boldsymbol\rho_{ia}^T
\right].
]

Then integrate the two frames as ordinary rigid bodies.

This is much simpler, positive definite, and exactly partitions the molecular mass because

[
\sum_am_a=\sum_im_i.
]

But it neglects cross-frame kinetic terms. If this option is used, fit (K) against the frequencies generated by this chosen kinetic model, not against (P^TMP).

## 16.3 Global rigid motion plus six internal coordinates

Use:

* molecular center of mass;
* one global molecular quaternion;
* six internal coordinates (\boldsymbol\eta).

Construct the atom map as

[
\mathbf r_i
===========

\mathbf R_{\mathrm{global}}
\widehat{\mathbf r}*i(\boldsymbol\eta)
+
\mathbf r*{\mathrm{COM}}.
]

Choose the internal tangent to satisfy Eckart orthogonality:

[
G^TMP=0.
]

Then global and internal motion decouple to first order.

This is mathematically the cleanest reduced dynamics, although coupling external atomic forces to two independently responding ends is somewhat less direct.

My recommendation for the first implementation is the constant projected mass matrix. Only implement the full configuration-dependent mass if large-amplitude inertial dynamics proves important.

---

# 17. Fitting beyond the harmonic Hessian

The Hessian only fixes the local quadratic behavior.

To fit finite bending and twisting:

1. choose a set of relative frame deformations (\boldsymbol\eta_s);
2. reconstruct an initial atomic geometry;
3. constrain the coarse frame deformation;
4. relax all remaining atomistic coordinates;
5. record relaxed energy and generalized force;
6. fit a nonlinear reduced potential.

Define

[
V_{\mathrm{relax}}(\boldsymbol\eta)
===================================

\min_{\mathbf y}
V_{\mathrm{AA}}
\bigl(
\Phi(\boldsymbol\eta,\mathbf y)
\bigr),
]

where (\mathbf y) contains fine internal corrections that do not change the coarse pose.

A polynomial model could be

[
V_{\mathrm{CG}}
===============

\frac12K_{\alpha\beta}\eta_\alpha\eta_\beta
+
\frac16C_{\alpha\beta\gamma}
\eta_\alpha\eta_\beta\eta_\gamma
+
\frac1{24}
D_{\alpha\beta\gamma\delta}
\eta_\alpha\eta_\beta\eta_\gamma\eta_\delta.
]

But a completely general six-dimensional quartic contains too many terms. Use molecular symmetry and only a few physically relevant nonlinear terms, for example:

[
V_{\mathrm{twist}}(\phi_x)
==========================

k_t[1-\cos(\phi_x)],
]

[
V_{\mathrm{bend},y}
===================

\frac12k_y\phi_y^2
+\frac14q_y\phi_y^4,
]

[
V_{\mathrm{bend},z}
===================

\frac12k_z\phi_z^2
+\frac14q_z\phi_z^4.
]

Retain the full quadratic (6\times6) coupling and add only selected cubic or quartic corrections.

## Energy-and-force fitting

Represent

[
V_\theta(\boldsymbol\eta)
=========================

\sum_p\theta_p\psi_p(\boldsymbol\eta).
]

Then fit

[
\min_\theta
\sum_s
w_E
\left[
V_\theta(\boldsymbol\eta_s)-E_s
\right]^2
+
w_F
\left|
\nabla_\eta V_\theta(\boldsymbol\eta_s)
-\mathbf g_s
\right|^2
+
\lambda|\theta|^2.
]

This remains linear least squares whenever the basis functions (\psi_p) are fixed.

---

# 18. Zero-temperature relaxed energy versus finite-temperature free energy

The relaxed energy

[
V_{\mathrm{relax}}(\boldsymbol\eta)
===================================

\min_{\mathbf y}V(\boldsymbol\eta,\mathbf y)
]

is appropriate for low-temperature mechanics and quasi-static molecular flexibility.

At finite temperature, the correct coarse potential is the potential of mean force:

[
A(\boldsymbol\eta)
==================

-k_BT
\ln
\int
d\mathbf y,
e^{-\beta V(\boldsymbol\eta,\mathbf y)}.
]

These are not the same. The free energy includes the entropy of eliminated bond, angle and local deformation modes.

Force matching and relative-entropy methods provide rigorous statistical-mechanical frameworks for fitting such coarse potentials from atomistic trajectories. 

The mapping itself can also be optimized from trajectory data rather than only from a Hessian. AutoGrain, for example, jointly optimizes coarse mappings, reconstruction and force regularization, although its bead assignments and decoder are different from your structured rigid-frame construction. ([Nature][5])

For your intended first application—mechanical deformation of planar adsorbed molecules—I would start with zero-temperature constrained relaxation and only later address finite-temperature free energies.

---

# 19. Can full atom dynamics refine the coarse trajectory?

Yes, but there are three physically different versions.

## 19.1 Coarse dynamics plus quasi-static fine relaxation

Write

[
\mathbf r=\Phi(\mathbf q)+Y\mathbf y,
]

where (\mathbf q) are frame coordinates and (Y\mathbf y) contains fine corrections.

At every coarse step,

[
\mathbf y_*(\mathbf q)
======================

\arg\min_{\mathbf y}
V[
\Phi(\mathbf q)+Y\mathbf y
].
]

Then integrate only

[
V_{\mathrm{eff}}(\mathbf q)
===========================

V[
\Phi(\mathbf q)+Y\mathbf y_*(\mathbf q)
].
]

The hard modes are minimized, not dynamically integrated, so they do not constrain the outer timestep.

At the fine optimum,

[
Y^T\nabla V=0.
]

By the envelope theorem,

[
\frac{dV_{\mathrm{eff}}}{d\mathbf q}
====================================

J_\Phi^T\nabla V.
]

Thus the generalized coarse force remains the Jacobian-transposed atom force.

A molecular method called Normal Mode Langevin follows a related philosophy: propagate slow modes while minimizing nearly instantaneous fast modes. ([PMC][6])

### Pseudocode

```text
for each coarse time step:

    integrate frame coordinates q using current coarse forces

    x = nonlinear_prolongation(q)

    for fine_relax_iteration = 1 .. n_relax:

        f = -gradient(V_atomistic, x)

        J = coarse_jacobian(q)

        coarse_projection =
            J * inverse(transpose(J) * M * J)
              * transpose(J) * M

        fine_displacement =
            (I - coarse_projection)
            * preconditioner_hard
            * inverse(M)
            * f

        x += damping * fine_displacement

        reimpose coarse frame coordinates exactly

    evaluate atomistic or external forces at relaxed x

    Fq = transpose(J) * f
```

The reimposition of the coarse coordinates can use weighted Kabsch fits for the two frames.

## 19.2 Fine-mode subcycling

Integrate frame variables with a large step (\Delta t), and fine atom variables with

[
\delta t=\frac{\Delta t}{n}.
]

This is a multiple-time-step method. It reduces cost but does not remove the hard frequencies. Stability resonances can occur when the outer step is too large.

Normal-mode multiple-time-step Langevin schemes have been developed for exactly this slow/fast decomposition. ([AIP Publishing][7])

## 19.3 Fully implicit atomistic correction

Retain all atomic coordinates and solve an implicit timestep:

[
\mathbf r_{n+1}
===============

\arg\min_{\mathbf r}
\left[
\frac{1}{2\Delta t^2}
|\mathbf r-\mathbf r_{\mathrm{pred}}|_M^2
+
V(\mathbf r)
\right].
]

Its approximate Hessian is

[
A=
\frac{M}{\Delta t^2}+H.
]

The frame space is then a coarse space for solving

[
A\delta\mathbf r=\mathbf b.
]

Use:

1. local bond/angle smoothing;
2. frame-level coarse correction;
3. local post-smoothing.

This is genuine multigrid rather than merely reduced dynamics.

```text
function two_level_solve(x, rhs):

    x = local_bond_angle_smoother(x, rhs, pre_steps)

    residual = rhs - A * x

    coarse_rhs = transpose(P) * residual
    coarse_A   = transpose(P) * A * P

    delta_q = solve_6x6_or_12x12(coarse_A, coarse_rhs)

    x += P * delta_q

    x = local_bond_angle_smoother(x, rhs, post_steps)

    return x
```

High-frequency modes are then numerically stable at a large timestep, but they are not temporally resolved. They will generally be damped or have incorrect phase. That is acceptable when they are regarded as rapid internal relaxation.

---

# 20. A true molecular multigrid hierarchy

A useful hierarchy could be

[
\text{two global frames}
\rightarrow
\text{three or four molecular frames}
\rightarrow
\text{one frame per aromatic ring/group}
\rightarrow
\text{all atoms}.
]

At every level (l), define a prolongation

[
P_l^{l+1}
]

and Galerkin matrices

[
M_l
===

P_l^TM_{l+1}P_l,
]

[
H_l
===

P_l^TH_{l+1}P_l.
]

A V-cycle for minimization or an implicit step is:

```text
function Vcycle(level, x, rhs):

    if level is finest:
        x = local_hard_mode_smoother(x, rhs)

    residual = rhs - A[level] * x
    coarse_rhs = transpose(P[level]) * residual

    if level is coarsest:
        correction_coarse = direct_solve(A[level-1], coarse_rhs)
    else:
        correction_coarse =
            Vcycle(level-1, zero, coarse_rhs)

    x += P[level] * correction_coarse

    x = local_hard_mode_smoother(x, rhs)

    return x
```

The key multigrid idea is complementary error removal:

* local bond/angle smoothers rapidly eliminate high-spatial-frequency atom errors;
* frame corrections eliminate coherent bending and twisting errors that local smoothing converges on very slowly.

Smoothed-aggregation AMG similarly starts from near-null modes—rigid-body modes in elasticity—and constructs or smooths interpolation operators that accurately represent low-energy motion. ([Springer Link][8])

---

# 21. Adaptive error indicator

During a coarse-only simulation, the atom forces may contain a component that cannot be represented by the frame manifold.

Let

[
J=\frac{\partial\mathbf r}{\partial\mathbf q}.
]

The mass-orthogonal displacement projector onto the coarse tangent is

[
\Pi_J
=====

J(J^TMJ)^{-1}J^TM.
]

A corresponding unresolved force indicator is

[
\mathbf f_\perp
===============

## \mathbf f

MJ(J^TMJ)^{-1}J^T\mathbf f.
]

A normalized measure is

[
\epsilon_{\mathrm{fine}}
========================

\frac{
|M^{-1/2}\mathbf f_\perp|
}{
|M^{-1/2}\mathbf f|+\varepsilon
}.
]

If (\epsilon_{\mathrm{fine}}) becomes large:

* perform fine atomistic relaxation;
* switch from two to three frames;
* activate one or more residual modes;
* reduce the timestep locally.

This would make the method adaptively multiresolution rather than committing to one fixed model everywhere.

---

# 22. Why frames can beat direct Hessian modes

For a fixed number of **linear** coordinates near equilibrium, the lowest Hessian modes are optimal. Frames cannot beat them there.

But the frame model is a nonlinear manifold:

[
\mathbf r=\Phi(G_1,G_2),
]

while the normal-mode model is normally

[
\mathbf r
=========

\mathbf r^0+
\sum_\mu a_\mu\mathbf u_\mu.
]

At finite amplitude, a linear mode extrapolation produces artificial bond stretching because it adds displacement vectors rather than rotating molecular regions.

NOLB was developed to address this issue by interpreting rigid-block normal modes as finite rotations and translations instead of linear Cartesian displacement extrapolations. ([American Chemical Society Publications][9])

A concise comparison is:

| Property                     |           Hessian modes |              Two rigid frames |
| ---------------------------- | ----------------------: | ----------------------------: |
| Harmonic optimality          |           Best possible |                   Approximate |
| Internal DOFs                |              Chosen (r) |                     Exactly 6 |
| Storage for 6 modes          |           (18N) scalars |                   (N) weights |
| Finite rotation              |     Poor unless updated |                       Natural |
| Interpretability             |          Mode-dependent |              Bend/shear/twist |
| Hierarchical refinement      | Possible but less local |                       Natural |
| Position evaluation          |           Often cheaper |                 Moderate cost |
| Transfer away from reference |              Often poor |            Potentially better |
| Fine atom correction         |                Possible | Natural geometric restriction |

The likely best practical method is hybrid:

[
\boxed{
\mathbf r
=========

\Phi_{\mathrm{frames}}(\mathbf q)
+
\sum_{\mu=1}^{n_r}
a_\mu\boldsymbol\psi_\mu.
}
]

Here the frames capture global nonlinear bending and twisting, while two or three residual modes capture local deformation the frame interpolation misses.

Component-mode synthesis uses a closely related idea: retain physically meaningful boundary or rigid coordinates and supplement them by selected internal modes. ([Massachusetts Institute of Technology][10])

---

# 23. Main caveats and failure modes

## Two frames may be too few

They provide only six internal coordinates and essentially one deformation segment. They cannot represent:

* two independent bending wavelengths;
* central bowing with nearly unchanged endpoint frames;
* saddle deformation across molecular width;
* local carbonyl or anhydride motion;
* ring breathing.

## Linear blending can reintroduce hard stiffness

If the interpolation stretches covalent bonds during nominal bending, then

[
V_{\mathrm{AA}}(\Phi(\mathbf q))
]

may still be very stiff along a frame coordinate. That would reduce the timestep advantage.

Use (SE(3)), dual-quaternion interpolation, additional frames, or atomistic relaxation to avoid this.

## The fitted Hessian is local

Weights fitted at one equilibrium geometry need not remain optimal under:

* strong bending;
* different charge state;
* adsorption;
* chemical substitution;
* strong external electric field.

Finite-amplitude validation is essential.

## Soft modes can be force-field dependent

Out-of-plane and torsional modes are precisely the modes for which empirical force fields can differ substantially. The reduced model cannot be more physically correct than its atomistic training Hessian.

## Kinetic energy is not automatically two rigid bodies

Overlapping weights generate cross-frame inertia. Ignoring this may distort frequencies even when the stiffness is fitted correctly.

## Coarse equilibrium dynamics is not exact atomistic dynamics

Eliminating fine variables generates entropic corrections, friction, noise, and generally memory. Mori–Zwanzig coarse dynamics leads naturally to generalized Langevin equations rather than simply Newton’s equations in fewer variables. ([AIP Publishing][11])

## Quaternion and logarithm singularities

Frame quaternion signs must be kept continuous. The logarithm becomes ill-conditioned near a relative rotation of (\pi). That should be far outside the intended molecular deformation range, but the code still needs a branch-safe implementation.

## Other forces can still limit the timestep

Even after eliminating hard covalent vibrations, a strongly repulsive atom–surface contact, stiff electrostatic confinement, or collision can produce a high generalized frequency. The timestep is limited by the stiffest mode remaining in the reduced manifold, not solely by intramolecular bonds.

---

# 24. Recommended implementation sequence

## Version 1: establish whether the representation works

1. Compute a gas-phase or isolated intramolecular Hessian.
2. Extract 12–20 lowest internal modes.
3. Fit two-frame weights using the alternating algorithm.
4. Measure principal angles and projection errors.
5. Compute:
   [
   M_\eta=P^TMP,
   \qquad
   K_\eta=P^THP.
   ]
6. Compare reduced frequencies and atom displacement patterns.
7. Repeat with three frames.

## Version 2: finite deformation

1. Keep the fitted weights.
2. Compare:

   * linear blend;
   * dual-quaternion blend;
   * (SE(3)) exponential interpolation.
3. Scan bending and twisting amplitudes.
4. Monitor:

   * bond-length distortion;
   * bond-angle distortion;
   * atomistic energy error;
   * residual force outside the frame tangent.

## Version 3: relaxed coarse potential

1. Prescribe coarse relative frame poses.
2. Constrained-relax all atomic coordinates.
3. Fit a (6\times6) harmonic stiffness plus selected quartic terms.
4. Validate energies and generalized forces on configurations omitted from fitting.

## Version 4: coarse–fine dynamics

1. Integrate frames at the large timestep.
2. Perform a few projected atomistic relaxation iterations.
3. Compare against full atom MD filtered to the same slow observables.
4. If necessary, add residual modes or one additional control frame.

---

# 25. Closest literature

The most directly relevant papers I found are:

1. **Tama, Gadea, Marques and Sanejouand, “Building-block approach for determining low-frequency normal modes of macromolecules,” Proteins 41, 1–7 (2000).**
   Rotation–translation-of-blocks projection of atomistic modes onto rigid-block DOFs. This is the closest linear harmonic predecessor. ([PubMed][1])

2. **Hoffmann and Grudinin, “NOLB: Nonlinear Rigid Block Normal-Mode Analysis Method,” J. Chem. Theory Comput. (2017), DOI 10.1021/acs.jctc.7b00197.**
   Converts RTB mode directions into nonlinear finite rigid-block motions. Closest to your finite-rotation motivation. ([American Chemical Society Publications][9])

3. **Essiz and Coalson, “Langevin dynamics of molecules with internal rigid fragments in the harmonic regime,” J. Chem. Phys. 127, 104109 (2007).**
   Derives reduced harmonic Langevin dynamics for molecules containing rigid internal fragments. ([AIP Publishing][12])

4. **Izaguirre, Sweet and Pande, “Multiscale Dynamics of Macromolecules Using Normal Mode Langevin,” Pacific Symposium on Biocomputing 15, 240–251 (2010).**
   Propagates slow modes and minimizes fast modes; very relevant to your proposed coarse-dynamics/fine-relaxation split. ([PMC][6])

5. **Rudd and Broughton, coarse-grained molecular dynamics and nonlinear finite-element formulations.**
   Atomistically derived coarse shape functions, mass and stiffness, finite-temperature coarse graining, and atomistic/coarse coupling. Closest rigorous precedent for your “molecular multigrid” interpretation. ([APS Link][4])

6. **Vaněk, Mandel and Brezina, “Algebraic multigrid by smoothed aggregation for second and fourth order elliptic problems,” Computing 56, 179–196 (1996).**
   Uses zero-energy modes and smoothed prolongation; the conceptual basis for beginning with rigid-frame modes and energy-smoothing the atom interpolation. ([Springer Link][8])

7. **Sonneville, Cardona and Brüls, “Geometrically exact beam finite element formulated on the special Euclidean group (SE(3)),” CMAME 268, 451–474 (2014), DOI 10.1016/j.cma.2013.10.008.**
   Gives rigorous exponential interpolation of coupled translation and rotation and Lie-group dynamics. ([ScienceDirect][2])

8. **Kavan and collaborators, dual-quaternion rigid-transformation blending.**
   Useful for implementing efficient nonlinear weighted frame interpolation without linear-blend collapse. ([users.cs.utah.edu][3])

9. **Noid et al., multiscale coarse-graining and force matching; Shell, relative-entropy coarse graining.**
   Relevant if the fitted frame potential should reproduce finite-temperature equilibrium statistics rather than only a Hessian or relaxed PES. 

My overall judgment is that the concept is promising, but the strongest scientific contribution would not be “two rigid frames instead of dihedrals” by itself. It would be the complete construction:

[
\boxed{
\text{structured overlapping-frame map}
\rightarrow
\text{optimal low-energy fitting}
\rightarrow
\text{nonlinear objective frame potential}
\rightarrow
\text{adaptive coarse–fine relaxation}
}
]

together with a demonstration that it reproduces the low Hessian subspace, retains accuracy at finite bending amplitudes, and permits a significantly larger stable timestep than the underlying all-atom force field.

[1]: https://pubmed.ncbi.nlm.nih.gov/10944387/?utm_source=chatgpt.com "Building-block approach for determining low-frequency ..."
[2]: https://www.sciencedirect.com/science/article/abs/pii/S0045782513002600 "Geometrically exact beam finite element formulated on the special Euclidean group SE(3) - ScienceDirect"
[3]: https://users.cs.utah.edu/~ladislav/kavan06dual/kavan06dual.pdf "https://users.cs.utah.edu/~ladislav/kavan06dual/kavan06dual.pdf"
[4]: https://link.aps.org/doi/10.1103/PhysRevB.72.144104?utm_source=chatgpt.com "Coarse-grained molecular dynamics: Nonlinear finite ..."
[5]: https://www.nature.com/articles/s41524-019-0261-5 "Coarse-graining auto-encoders for molecular dynamics | npj Computational Materials"
[6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4308582/ "https://pmc.ncbi.nlm.nih.gov/articles/PMC4308582/"
[7]: https://pubs.aip.org/aip/jcp/article/128/14/145101/847663/Normal-mode-partitioning-of-Langevin-dynamics-for "https://pubs.aip.org/aip/jcp/article/128/14/145101/847663/Normal-mode-partitioning-of-Langevin-dynamics-for"
[8]: https://link.springer.com/article/10.1007/BF02238511 "Algebraic multigrid by smoothed aggregation for second and fourth order elliptic problems | Computing | Springer Nature Link"
[9]: https://pubs.acs.org/doi/abs/10.1021/acs.jctc.7b00197?utm_source=chatgpt.com "NOLB: Nonlinear Rigid Block Normal-Mode Analysis Method"
[10]: https://web.mit.edu/kjb/www/Principal_Publications/Component_mode_synthesis_with_subspace_iterations_for_controlled_accuracy_of_frequency_and_mode_shape_solutions.pdf?utm_source=chatgpt.com "Component mode synthesis with subspace iterations for ..."
[11]: https://pubs.aip.org/aip/jcp/article/145/20/204117/281273/The-derivation-and-approximation-of-coarse-grained "https://pubs.aip.org/aip/jcp/article/145/20/204117/281273/The-derivation-and-approximation-of-coarse-grained"
[12]: https://pubs.aip.org/aip/jcp/article/127/10/104109/917865/Langevin-dynamics-of-molecules-with-internal-rigid?utm_source=chatgpt.com "Langevin dynamics of molecules with internal rigid fragments ..."

---

# USER

This is really great analysis!

I have miltiple ideas to expand it, and to make the methods even better.


1) multiple frames - for pentacene 2 is neigh (very narrow) for PTCDA 2 maybe 4 could be better (in each corner on =O oxygesn?), or for some riangular molecules 3 would be mroe suitable. Fr more flexible molecules e.g. 1,2,4,5-Tetraphenylbenzene it would make sense to put one rotationg frame on each fragment (central benzene and each phenyl that is 5 frames and 5*6=30DOFs in total), user will always try to balance accuracy and speed minimizing the number of DOFs.

Generalize your defivation for abritrary number of DOFs

1b) Start with geometrical/topological weights estimate - although optimal weights can be obtained from the vibration soft eigenmodes, reasonable estimate is often just geometrical.
i) radial distance function from the two centers with some decay (e.g. exponential) and renormalization so that w1+w2=1
ii) define linear directio between the position of the grid pivots and then use some weigts function e.g. linear or smoothstep to assing weights
this geometrical initial guess canalso regularize the weights because they are underdetermined (there is more weights/atoms than DOFs we want to represent. So I would start with this initial guess and use gradient-based optimization (FIRE,ADAM,BFGS) only for fine tuning. 


2) You wan te use some alternating least sqaure fitting to determine the weights and stiffnesses. That is perhaps good, but maybe what can be more general is to do non-linear optimization by gradiaent descent eventually interitally accelrated (like FIRE, ADAM, I have very good experience with FIRE) or eventually BFGS, it scale better for large number of coefs and non-linear problems. It will allow us to do also non-linear optimization for anharmonic deformation beyond hessian. Elaborate on this idea. 

3) non linear fitting - non linear effects. I Very much like the idea of using  exponential interpolation of rotation SE(3), we should think it through how exatly we will fit it (we need energy and force of many distorted geometries right? I only a bit worry about the performance as evaluating exponnetials is a bit costly sometimes.

3b) I think the exponential of rotation operator can be approximated by taylor exponsion quite efficiently, I often use taylor expansion of sin,cos for running rotation dynamcs for small dispacements. Also I often replace expontial exp(-r/rc) by polynominal (1-r/(n*rc))^n where rc is some cutoff, this polynominal possibly converge even faster than taylor (?). Perhps we can use something like that for faster evaluation here. 

---

# GPT 5.6 sol hi

Yes—these extensions make the method substantially more general. I would separate three concepts that were conflated in the two-frame version:

1. **Control frames** define a compact nonlinear geometry.
2. **Interpolation weights** define how atomic positions follow the frames.
3. **Dynamic coordinates** are the subset of frame motions that are soft enough to integrate explicitly.

The third point is important. With (K) frames there are (6K-6) internal frame coordinates, but it is not necessarily desirable to integrate all of them. Some correspond to stiff extension, shear, or bond-angle distortion and could again limit the timestep. The number of frames and the number of retained dynamical DOFs should therefore be independent design choices.

---

# 1. General construction for (K) frames

Let the molecule contain (N) atoms with reference positions

[
\mathbf r_i^0,\qquad i=1,\ldots,N.
]

Frame (a), (a=1,\ldots,K), has reference pose

[
G_a^0=(R_a^0,\mathbf c_a^0)\in SE(3)
]

and current pose

[
G_a=(R_a,\mathbf c_a).
]

For every atom–frame pair define the atom’s reference coordinate in the local frame,

[
\boldsymbol\rho_{ia}
====================

R_a^{0T}
(\mathbf r_i^0-\mathbf c_a^0).
]

The position predicted by frame (a) alone is

[
\mathbf p_{ia}
==============

\mathbf c_a+R_a\boldsymbol\rho_{ia}.
]

Let

[
w_{ia}\ge 0,
\qquad
\sum_{a=1}^K w_{ia}=1.
]

The cheapest interpolation is

[
\boxed{
\mathbf r_i
===========

\sum_{a=1}^K
w_{ia}
\left(
\mathbf c_a+R_a\boldsymbol\rho_{ia}
\right).
}
]

This is ordinary linear blending of rigid-frame predictions, but here the weights and frame stiffness are fitted from molecular mechanics rather than chosen artistically.

At the reference geometry,

[
\mathbf c_a=\mathbf c_a^0,\qquad R_a=R_a^0,
]

and each individual frame predicts

[
\mathbf c_a^0+R_a^0\boldsymbol\rho_{ia}
=======================================

\mathbf r_i^0.
]

Therefore the interpolated geometry is exactly the reference geometry independently of the weights.

The partition-of-unity property ensures that a common translation of all frames translates every atom correctly. A common spatial rotation also acts correctly provided all frames are transformed consistently.

---

## 1.1 Number of internal coordinates

The frames have (6K) coordinates. A common rigid transformation of all frames corresponds to the six global rigid motions of the whole molecule. Therefore the maximum internal rank is

[
\boxed{
d_{\mathrm{frame}}=6K-6.
}
]

Examples:

* pentacene with two frames: at most 6 internal frame coordinates;
* triangular molecule with three frames: at most 12;
* four-frame PTCDA: at most 18;
* central benzene plus four phenyl frames: at most 24 internal coordinates.

But those are only the available geometrical coordinates. The number of explicitly propagated soft coordinates may be smaller:

[
r_{\mathrm{soft}}\le 6K-6.
]

For tetraphenylbenzene, for example, the physically soft motions may be mostly the four phenyl torsions and a few collective bending modes. Relative translations across the central–phenyl bonds are stiff and should probably be constrained or eliminated rather than integrated explicitly.

---

# 2. General linearized basis

For a small displacement of frame (a), let

[
\delta\mathbf t_a
]

be its translation and

[
\delta\boldsymbol\theta_a
]

its infinitesimal spatial rotation.

Then

[
\delta\mathbf r_i
=================

\sum_a
w_{ia}
\left[
\delta\mathbf t_a
+
\delta\boldsymbol\theta_a
\times
(\mathbf r_i^0-\mathbf c_a^0)
\right].
]

Define

[
A_{ia}
======

\begin{bmatrix}
I&
-[\mathbf r_i^0-\mathbf c_a^0]_\times
\end{bmatrix},
]

where

[
[\mathbf x]_\times\mathbf y=\mathbf x\times\mathbf y.
]

The (3\times6) atom–frame block is

[
B_{ia}=w_{ia}A_{ia}.
]

Stacking all atoms and frames gives

[
\boxed{
\delta\mathbf r=B(W),\delta\mathbf q,
}
]

where

[
B(W)\in\mathbb R^{3N\times 6K}.
]

Given the atomic mass matrix (M) and Hessian (H),

[
M_f=B^TMB,
\qquad
K_f=B^THB.
]

After removing the six common rigid transformations, solve

[
K_f Z=M_f Z\Lambda.
]

The eigenvectors in (Z) identify the soft and stiff combinations of frame motions.

This suggests a very useful factorization:

[
\boxed{
\delta\mathbf r
===============

B(W),Z_s,\mathbf a,
}
]

where (Z_s) contains only (r_{\mathrm{soft}}) selected frame-space modes.

The stored representation is then:

* approximately (NK) interpolation weights, often sparse;
* (6K\times r_{\mathrm{soft}}) frame-space mode coefficients.

This remains dramatically smaller than storing

[
3N\times r_{\mathrm{soft}}
]

arbitrary atomistic eigenvectors.

More importantly, the finite-amplitude version can use frame rotations rather than linear atomic displacement modes.

---

# 3. Frame topology and internal energy

For arbitrary (K), it is useful to define a **frame graph**

[
{\cal G}_f=({\cal V}_f,{\cal E}_f).
]

An edge (e=(a,b)) connects frames whose relative motion should contribute to the internal energy.

Examples:

* pentacene chain: (1-2-3);
* triangular molecule: triangle of three frame edges;
* four-frame PTCDA: rectangular cycle, perhaps with one or two diagonals;
* tetraphenylbenzene: star graph with the central frame connected to each phenyl frame.

Define the deformation transformation of each frame,

[
D_a=G_a(G_a^0)^{-1}.
]

At equilibrium,

[
D_a=I.
]

For edge (e=(a,b)), define

[
E_{ab}=D_a^{-1}D_b
]

and its relative twist

[
\boldsymbol\xi_{ab}
===================

\log(E_{ab})
\in\mathbb R^6.
]

The simplest nonlinear frame-graph energy is

[
\boxed{
V_{\mathrm{frame}}
==================

\sum_{(a,b)\in{\cal E}*f}
\psi*{ab}(\boldsymbol\xi_{ab}).
}
]

In the harmonic approximation,

[
\psi_{ab}
=========

\frac12
\boldsymbol\xi_{ab}^T
K_{ab}
\boldsymbol\xi_{ab},
]

where each (K_{ab}) is a symmetric (6\times6) block.

This gives an (O(|{\cal E}_f|)) internal-force evaluation. For a sparse molecular frame graph,

[
|{\cal E}_f|=O(K).
]

A more general but less transferable alternative is a full internal stiffness matrix. Choose frame 1 as a root and define

[
\boldsymbol\eta
===============

\begin{pmatrix}
\log(D_1^{-1}D_2)\
\vdots\
\log(D_1^{-1}D_K)
\end{pmatrix}
\in\mathbb R^{6(K-1)}.
]

Then

[
V=
\frac12\boldsymbol\eta^T
K_{\mathrm{global}}
\boldsymbol\eta.
]

This can reproduce arbitrary harmonic coupling among frames but contains

[
\frac{6(K-1)[6(K-1)+1]}{2}
]

parameters. For (K=5), that is already 300 quadratic coefficients.

I would use:

* sparse edge potentials for fragment-like molecules;
* a full or moderately sparse matrix for a small fixed molecule such as PTCDA.

---

# 4. Do not propagate the stiff frame coordinates explicitly

Adding more frames does not automatically remove hard modes.

For example, a relative frame edge has six coordinates:

* axial extension;
* two transverse translations;
* three rotations.

For a phenyl–benzene connection, perhaps only rotation around the bond is genuinely soft. The remaining five directions can be very stiff.

If all six are propagated explicitly, the maximum timestep is still controlled by

[
\omega_{\max}^2
===============

\lambda_{\max}(M_f^{-1}K_f).
]

There are three ways to avoid that.

## 4.1 Retain only soft frame-space modes

Solve

[
K_f Z=M_fZ\Lambda
]

and retain only columns satisfying

[
\lambda_\mu<\lambda_{\mathrm{cut}}.
]

Then parameterize the internal frame motions by

[
\delta\mathbf q=Z_s\mathbf a.
]

This is normal-mode reduction, but in the small (6K-6)-dimensional frame space rather than the (3N)-dimensional atom space.

The memory cost of (Z_s) is negligible.

## 4.2 Use joints and constraints

For fragment molecules, define the allowed relative transformation directly.

For a hinge with axis (\mathbf h),

[
D_b
===

D_a
T_{\mathrm{bond}}
\exp
\begin{pmatrix}
0\
\phi\mathbf h
\end{pmatrix}
T_{\mathrm{bond}}^{-1}.
]

Then the edge has only one dynamical coordinate (\phi). Bond lengths and bond angles are fixed by construction.

For tetraphenylbenzene, a very efficient model might therefore contain:

* six global molecular DOFs;
* four phenyl torsions;
* perhaps two or four collective bending coordinates.

That is closer to 12–14 total DOFs than 30.

## 4.3 Minimize stiff frame coordinates

Split the internal coordinates into

[
\boldsymbol\eta=
\begin{pmatrix}
\boldsymbol\eta_s\
\boldsymbol\eta_h
\end{pmatrix}.
]

At every slow configuration define

[
\boldsymbol\eta_h^*(\boldsymbol\eta_s)
======================================

\arg\min_{\boldsymbol\eta_h}
V(\boldsymbol\eta_s,\boldsymbol\eta_h).
]

Only (\boldsymbol\eta_s) is propagated. This is static condensation inside the already reduced frame model.

---

# 5. Geometric initialization of the weights

I agree that the fitting should begin from a physically sensible geometry-dependent map. The Hessian optimization should be a fine correction, not an unconstrained search for arbitrary atom-wise coefficients.

There are several useful initializations.

---

## 5.1 Radial softmax weights

Let (d_{ia}) be the distance from atom (i) to frame center (a). Define

[
\phi_{ia}
=========

\exp
\left[
-\left(\frac{d_{ia}}{\sigma_a}\right)^p
\right]
]

and normalize:

[
\boxed{
w_{ia}^{(0)}
============

\frac{\phi_{ia}}
{\sum_b\phi_{ib}}.
}
]

Numerically, calculate this as a softmax:

[
\ell_{ia}
=========

-\left(\frac{d_{ia}}{\sigma_a}\right)^p,
]

[
w_{ia}^{(0)}
============

\frac{
\exp(\ell_{ia}-\ell_i^{\max})
}{
\sum_b\exp(\ell_{ib}-\ell_i^{\max})
}.
]

Common choices:

[
p=2
]

for Gaussian-like weights, or

[
p=1
]

for slower exponential decay.

An anisotropic distance is often better:

[
d_{ia}^2
========

(\mathbf r_i^0-\mathbf c_a^0)^T
A_a
(\mathbf r_i^0-\mathbf c_a^0),
]

where (A_a) can make the influence elongated along the molecular backbone.

For a flat molecule, vertical distance should usually receive little or no relevance because all atoms lie near one plane.

---

## 5.2 Inverse-distance or Shepard weights

Another option is

[
\phi_{ia}
=========

\left(
d_{ia}^2+\epsilon^2
\right)^{-p/2},
]

[
w_{ia}^{(0)}
============

\frac{\phi_{ia}}{\sum_b\phi_{ib}}.
]

These have longer tails than exponentials and can be useful when every frame should weakly influence the whole molecule.

---

## 5.3 One-dimensional interpolation along a molecular axis

For a long narrow molecule, find the principal molecular axis (\mathbf e).

For two frames at projected coordinates (x_1,x_2),

[
t_i
===

\operatorname{clamp}
\left(
\frac{
(\mathbf r_i^0-\mathbf c_1^0)\cdot\mathbf e
}{
x_2-x_1
},0,1
\right).
]

Choices include:

### Linear

[
w_{i2}=t_i,
\qquad
w_{i1}=1-t_i.
]

### Cubic smoothstep

[
w_{i2}
======

3t_i^2-2t_i^3.
]

### Quintic smoothstep

[
w_{i2}
======

6t_i^5-15t_i^4+10t_i^3.
]

Linear weights are actually natural for a constant-strain beam interpolation. Smoothstep is useful when atoms near each endpoint should move almost completely with the nearby frame.

For multiple frames along a chain, use one-dimensional finite-element or B-spline shape functions:

[
w_{ia}=N_a(t_i),
\qquad
\sum_aN_a(t)=1.
]

Piecewise linear hat functions give at most two active frames per atom. Cubic B-splines give at most four.

---

## 5.4 Barycentric coordinates for triangular molecules

For three frames in a molecular plane, project the atom and frame centers into 2D and calculate barycentric coordinates:

[
\mathbf r_i^\parallel
=====================

w_{i1}\mathbf c_1^\parallel+
w_{i2}\mathbf c_2^\parallel+
w_{i3}\mathbf c_3^\parallel,
]

[
w_{i1}+w_{i2}+w_{i3}=1.
]

Inside the frame triangle, these weights are nonnegative automatically.

For atoms outside the triangle, one can:

* clamp and renormalize;
* use mean-value coordinates;
* revert to radial softmax weights.

---

## 5.5 Bilinear coordinates for four-frame planar molecules

For a roughly rectangular PTCDA layout, introduce molecular-plane coordinates ((u_i,v_i)) and four corner-frame weights:

[
w_{i,00}=(1-u_i)(1-v_i),
]

[
w_{i,10}=u_i(1-v_i),
]

[
w_{i,01}=(1-u_i)v_i,
]

[
w_{i,11}=u_iv_i.
]

This gives a natural plate-like interpolation.

However, I would not necessarily place the control frames exactly on the four terminal oxygen atoms. The oxygen motion can contain local C=O stretching or bending that should not define a global frame. Better centers might be the four molecular quadrants or the two anhydride groups, with the terminal oxygens receiving high but not exclusive weights.

---

# 6. Molecular-topological weights

Pure Euclidean distance can fail when two fragments are spatially close but separated by molecular topology.

This is especially relevant for folded or branched molecules. An atom on one phenyl ring could be spatially near another ring while being covalently attached to the central benzene.

Define graph distances on the bond graph:

[
d_{ia}^{\mathrm{graph}}
=======================

\text{shortest-path distance from atom }i
\text{ to frame seed }a.
]

The edge cost can be:

[
c_{ij}=1,
]

or based on bond length,

[
c_{ij}=r_{ij},
]

or, more interestingly, inverse bond rigidity,

[
c_{ij}\propto \frac{1}{\sqrt{k_{ij}}}.
]

Strong rigid bonds then propagate frame influence readily, while a flexible single bond acts as a natural boundary.

Use

[
\phi_{ia}
=========

\exp
\left[
-\beta_g
(d_{ia}^{\mathrm{graph}})^p
-\beta_r
(d_{ia}^{\mathrm{Euclid}})^2
\right].
]

This combines molecular topology with geometry.

---

# 7. Harmonic and biharmonic weight initialization

A particularly relevant analogy comes from automatic skinning weights. Bounded biharmonic weights are generated by minimizing a smoothness energy subject to nonnegativity, partition of unity, and control-handle constraints. They were introduced for real-time geometric deformation, but their mathematical structure transfers almost directly to a molecular graph. ([igl.ethz.ch][1])

Let (L_g) be a weighted molecular graph Laplacian. For the complete weight matrix

[
W\in\mathbb R^{N\times K},
]

solve

[
\boxed{
\min_W
\frac12
\sum_a
|L_g\mathbf w_a|^2
+
\frac{\mu}{2}
|W-W^{(0)}|_F^2
}
]

subject to

[
W\mathbf 1_K=\mathbf 1_N,
]

[
w_{ia}\ge 0,
]

and optionally frame-anchor constraints such as

[
w_{h_a,a}=1.
]

The first term is a discrete biharmonic smoothness penalty:

[
|L_g\mathbf w_a|^2
==================

\mathbf w_a^TL_g^TL_g\mathbf w_a.
]

A simpler harmonic alternative is

[
\sum_a\mathbf w_a^TL_g\mathbf w_a
=================================

\sum_{(i,j)}
\gamma_{ij}
|W_i-W_j|^2.
]

I would use the biharmonic construction as the geometric initialization, then fit the low-energy atomistic modes with a weaker penalty keeping the weights near that initialization.

---

# 8. Sparse frame influence

For efficiency, there is no reason every atom must depend on every frame.

Define an active set

[
{\cal A}_i\subset{1,\ldots,K}
]

with perhaps two to four frames per atom.

Then

[
w_{ia}=0
\qquad
a\notin{\cal A}_i.
]

For a chain molecule, two neighboring frames are sufficient. For a plate, four surrounding frames may be appropriate. For fragment molecules, most atoms can have exactly one frame and only atoms near fragment junctions need blended weights.

Runtime storage per atom can be:

```text
uint8 frame_index[L]
float weight[L-1]
```

with the final weight reconstructed as

[
w_L=1-\sum_{\ell=1}^{L-1}w_\ell.
]

The interpolation and force reduction then cost

[
O(NL)
]

rather than (O(NK)), where typically (L=2)–4.

---

# 9. General harmonic fitting for arbitrary (K)

Let the target low-frequency atomistic modes be

[
U=
\begin{bmatrix}
\mathbf u_1&\cdots&\mathbf u_r
\end{bmatrix}.
]

For fixed weights, fit the frame-coordinate representation

[
C=
\begin{bmatrix}
\mathbf q_1&\cdots&\mathbf q_r
\end{bmatrix}
\in\mathbb R^{6K\times r}
]

from

[
U\approx B(W)C.
]

The weighted least-squares optimum is

[
\boxed{
C^*(W)
======

\left(B^TMB\right)^\dagger
B^TMU.
}
]

The pseudoinverse is needed because (B) contains the six global rigid-motion redundancies. In practice, either:

* fix one root frame;
* explicitly remove the common frame motion;
* solve using SVD or rank-revealing QR.

The fitting loss is

[
\boxed{
{\cal L}_{\mathrm{mode}}(W)
===========================

\sum_{k=1}^r
\alpha_k
\left|
\mathbf u_k-B(W)\mathbf q_k^*(W)
\right|_M^2.
}
]

Add geometric regularization:

[
{\cal L}(W)
===========

{\cal L}*{\mathrm{mode}}
+
\lambda_0|W-W^{(0)}|*F^2
+
\lambda_1
\sum*{(i,j)}
\gamma*{ij}|W_i-W_j|^2
+
\lambda_2
\sum_a|L_g\mathbf w_a|^2.
]

This is better interpreted as an ill-conditioned or non-identifiable fitting problem than as simply “more weights than equations.” Each atom contributes displacement information in multiple modes, so there may be many more scalar observations than coefficients. Nevertheless, different weight maps can span almost the same low-dimensional frame subspace, which creates broad flat directions in the objective. The geometric prior resolves this ambiguity.

---

# 10. Variable projection: combine exact least squares with nonlinear optimization

You suggested replacing alternating least squares by FIRE, Adam, BFGS, or another nonlinear optimizer. That is useful, but I would not discard the linear solve.

The best hybrid is **variable projection**:

1. optimize only the genuinely nonlinear variables—the weights;
2. for every weight configuration, solve the frame-mode coefficients exactly by least squares.

Thus

[
C=C^*(W)
]

is eliminated analytically, and the outer optimizer sees only

[
{\cal L}_{\mathrm{reduced}}(W)
==============================

{\cal L}(W,C^*(W)).
]

Because

[
\frac{\partial{\cal L}}{\partial C}=0
]

at the least-squares optimum, the envelope theorem gives

[
\frac{d{\cal L}_{\mathrm{reduced}}}{dW}
=======================================

\left.
\frac{\partial{\cal L}}{\partial W}
\right|_{C=C^*(W)}.
]

You do not need to differentiate through the linear solve to obtain the first derivative.

---

## 10.1 Explicit weight gradient

For mode (k), define the rigid displacement predicted by frame (a) at atom (i):

[
\mathbf p_{iak}
===============

A_{ia}\mathbf q_{ak}.
]

The reconstructed mode is

[
\widehat{\mathbf u}_{ik}
========================

\sum_a w_{ia}\mathbf p_{iak}.
]

Define the residual

[
\mathbf e_{ik}
==============

\widehat{\mathbf u}*{ik}-\mathbf u*{ik}.
]

Then the mode-fitting gradient is

[
\boxed{
\frac{\partial{\cal L}*{\mathrm{mode}}}
{\partial w*{ia}}
=================

2m_i
\sum_k
\alpha_k
\mathbf e_{ik}\cdot\mathbf p_{iak}.
}
]

Add the regularization gradients:

[
\frac{\partial}{\partial W}
\lambda_0|W-W^{(0)}|_F^2
========================

2\lambda_0(W-W^{(0)}),
]

[
\frac{\partial}{\partial W}
\lambda_1\operatorname{Tr}(W^TL_gW)
===================================

2\lambda_1L_gW,
]

[
\frac{\partial}{\partial W}
\lambda_2|L_gW|_F^2
===================

2\lambda_2L_g^TL_gW.
]

---

# 11. Enforcing valid weights

There are two good choices.

## 11.1 Softmax parameterization

Introduce unconstrained logits (z_{ia}):

[
w_{ia}
======

\frac{e^{z_{ia}}}
{\sum_be^{z_{ib}}}.
]

This automatically enforces

[
w_{ia}>0,\qquad \sum_aw_{ia}=1.
]

If

[
g_{ia}
======

\frac{\partial{\cal L}}{\partial w_{ia}},
]

then

[
\boxed{
\frac{\partial{\cal L}}{\partial z_{ia}}
========================================

w_{ia}
\left[
g_{ia}
------

\sum_bw_{ib}g_{ib}
\right].
}
]

A temperature can control weight sharpness:

[
w_{ia}
======

\operatorname{softmax}(z_{ia}/T).
]

Start with (T\sim1) and optionally decrease it.

Disadvantage: once a weight becomes extremely close to zero, its gradient can become very small.

## 11.2 Projected simplex optimization

Optimize (W) directly and after each step project each row onto

[
\Delta_K=
\left{
\mathbf w:
w_a\ge0,;
\sum_aw_a=1
\right}.
]

This permits exact zero weights and works naturally with FIRE or projected gradient descent.

For sparse active sets, project only the (L) active weights.

---

# 12. FIRE, Adam and L-BFGS-B

FIRE was introduced as a pseudo-dynamical structural minimizer and is particularly successful for smooth deterministic atomistic relaxation problems. ([APS Links][2]) It is therefore a reasonable optimizer for the deterministic full-batch weight fit.

L-BFGS-B is specifically designed for large smooth nonlinear problems with simple bounds and uses a limited-memory curvature approximation. ([ACM Digital Library][3])

My practical ranking would be:

### FIRE

Good when:

* the objective is deterministic;
* gradients are inexpensive;
* the initial guess is already reasonable;
* broad, curved valleys cause simple gradient descent to oscillate.

It is easy to combine with simplex projection.

### Adam

Good when:

* nonlinear training uses minibatches of many distorted geometries;
* different parameter groups have very different scales;
* gradients contain noise.

Less attractive for final high-accuracy deterministic convergence.

### L-BFGS or L-BFGS-B

Good for:

* final polishing;
* smooth full-batch fitting;
* a few hundred to a few thousand parameters;
* direct (0\le w_{ia}\le1) bounds.

For this problem, even (N=100) and (K=10) gives only about 1000 raw weights, so L-BFGS-B is not remotely too large.

### Recommended staged optimizer

1. Geometric or biharmonic initialization.
2. Variable-projection FIRE or Adam to move into a good basin.
3. L-BFGS-B for final convergence.
4. Recompute the exact least-squares frame coefficients after every outer iteration.

---

## 12.1 FIRE pseudocode for weight logits

```text
z = logits_from_initial_weights(W0)
v = 0

for iteration:

    W = row_softmax(z)

    B = build_frame_basis(W)

    C = solve_weighted_least_squares(B, M, U)

    loss, grad_W = mode_loss_and_weight_gradient(
                       W, B, C, U, M,
                       geometric_regularization
                   )

    grad_z = softmax_chain_rule(W, grad_W)
    force  = -precondition(grad_z)

    v += dt * force

    power = dot(v, force)

    if power > 0:
        v = (1-alpha) * v
            + alpha * norm(v) * force / norm(force)

        positive_count += 1

        if positive_count > N_delay:
            dt    = min(dt * f_inc, dt_max)
            alpha = alpha * f_alpha

    else:
        v = 0
        dt = dt * f_dec
        alpha = alpha_start
        positive_count = 0

    z += dt * v

    remove_per_atom_logit_gauge:
        z[i,:] -= mean(z[i,:])

    stop if gradient and loss change are small
```

The per-atom logit mean is a gauge because adding the same constant to every (z_{ia}) leaves the softmax unchanged.

Use separate preconditioning scales for:

* weight logits;
* stiffness parameters;
* nonlinear potential coefficients.

Otherwise the optimizer’s pseudo-dynamics may be dominated by whichever parameter group has the largest numerical gradient.

---

# 13. Fitting the reduced stiffness

Once the interpolation weights are fixed, the simplest harmonic stiffness is not an independent fit:

[
\boxed{
K_f=B^THB.
}
]

If only soft frame-space modes (Z_s) are propagated,

[
M_s=Z_s^TM_fZ_s,
]

[
K_s=Z_s^TK_fZ_s.
]

Choose mass normalization

[
Z_s^TM_fZ_s=I
]

so that

[
K_s=\operatorname{diag}(\omega_\mu^2)
]

at the harmonic level.

If you want an edge-sparse model rather than the dense Galerkin stiffness, fit the edge matrices (K_e).

For small frame displacements, let

[
\boldsymbol\xi_e=C_e\mathbf q,
]

where (C_e) is the linear frame-graph incidence and relative-motion operator.

Then

[
V=
\frac12
\sum_e
\mathbf q^TC_e^TK_eC_e\mathbf q.
]

Therefore

[
K_{\mathrm{model}}
==================

\sum_eC_e^TK_eC_e.
]

Fit

[
\min_{{K_e}}
\left|
K_f-
\sum_eC_e^TK_eC_e
\right|_F^2
+
\lambda_K\sum_e|K_e|_F^2.
]

This is linear least squares in the entries of (K_e), except for the requirement that the resulting energy be positive semidefinite.

A safe parameterization is

[
K_e=L_eL_e^T+\epsilon I.
]

That makes the fit nonlinear but guarantees stability.

Alternatively:

1. fit symmetric (K_e) linearly;
2. assemble the global internal stiffness;
3. diagonalize it;
4. clamp small negative eigenvalues;
5. refit or project back to the sparse representation.

---

# 14. Nonlinear interpolation for arbitrary (K)

There is no unique analogue of the two-frame formula

[
D(s)=D_1\exp[s\log(D_1^{-1}D_2)]
]

for more than two frames. Several alternatives are possible.

---

## 14.1 Linear blend of transformed positions

[
\mathbf r_i=\sum_aw_{ia}D_a\mathbf r_i^0.
]

Advantages:

* cheapest;
* symmetric in all frames;
* exact partition of unity;
* simplest analytic force reduction.

Disadvantage:

* different rotations can cause contraction and shearing.

This should be the baseline.

---

## 14.2 Dual-quaternion blending

Represent each deformation transform (D_a) by a unit dual quaternion

[
\widehat q_a=q_{ra}+\varepsilon q_{da}.
]

First align quaternion signs with a chosen reference:

[
\widehat q_a
\leftarrow
\operatorname{sign}
(q_{ra}\cdot q_{r,\mathrm{ref}})
\widehat q_a.
]

For atom (i),

[
\widehat b_i
============

\sum_a w_{ia}\widehat q_a.
]

Let

[
\widehat b_i=b_{ri}+\varepsilon b_{di}.
]

Normalize:

[
q_{ri}=\frac{b_{ri}}{|b_{ri}|},
]

[
\widetilde q_{di}
=================

\frac{b_{di}}{|b_{ri}|},
]

[
q_{di}
======

## \widetilde q_{di}

q_{ri}
(q_{ri}\cdot\widetilde q_{di}).
]

Then

[
\widehat q_i=q_{ri}+\varepsilon q_{di}
]

defines a valid rigid transformation for atom (i).

Dual-quaternion skinning was designed to blend rigid transformations efficiently while avoiding the characteristic collapse artifacts of linear blend skinning. ([users.cs.utah.edu][4])

For your problem its cost is:

* (L) weighted dual-quaternion accumulations per atom;
* one reciprocal square root and normalization per atom;
* one quaternion rotation.

This is probably the best general-purpose nonlinear interpolation for (K>2).

---

## 14.3 Logarithmic (SE(3)) blending

Choose an anchor transform (D_*), for example frame 1. Define

[
\boldsymbol\xi_a
================

\log(D_*^{-1}D_a).
]

For atom (i),

[
\boldsymbol\xi_i
================

\sum_aw_{ia}\boldsymbol\xi_a,
]

[
\boxed{
D_i=D_*\exp(\boldsymbol\xi_i).
}
]

This construction is objective under a common left transformation because

[
(SD_*)^{-1}(SD_a)=D_*^{-1}D_a.
]

Advantages:

* direct Lie-algebra interpolation;
* valid rigid transform for every atom;
* only (K) logarithms per molecule, not per atom.

Disadvantages:

* one exponential per atom;
* dependence on the chosen anchor;
* branch problems if relative rotations become large;
* not symmetric among frames.

A symmetric intrinsic mean or Karcher mean is mathematically possible, but it generally requires iterative nonlinear optimization and is less attractive for force-field evaluation. Proper invariant averages of rotations and their metric dependence are discussed in the rotation-averaging literature. ([SIAM][5])

---

# 15. Two-frame exact (SE(3)) interpolation

For two frames,

[
D_{12}=D_1^{-1}D_2,
]

[
\boldsymbol\xi=\log(D_{12})
===========================

\begin{pmatrix}
\boldsymbol\rho\
\boldsymbol\phi
\end{pmatrix}.
]

For atom (i) with weight (s_i),

[
D_i
===

D_1\exp(s_i\boldsymbol\xi),
]

[
\mathbf r_i=D_i\mathbf r_i^0.
]

This is closely related to exponential interpolation in geometrically exact (SE(3)) beam elements, which couple rotation and translation within the group rather than interpolating them as unrelated Euclidean variables. ([ScienceDirect][6])

The exponential of a twist can be evaluated as follows.

Let

[
\theta=|\boldsymbol\phi|,
\qquad
\Omega=[\boldsymbol\phi]_\times.
]

Then

[
R(s)
====

I+
\frac{\sin(s\theta)}{\theta}\Omega
+
\frac{1-\cos(s\theta)}{\theta^2}\Omega^2.
]

The translational part is

[
\boxed{
\mathbf t(s)
============

s\boldsymbol\rho
+
\frac{1-\cos(s\theta)}{\theta^2}
\Omega\boldsymbol\rho
+
\frac{s\theta-\sin(s\theta)}{\theta^3}
\Omega^2\boldsymbol\rho.
}
]

Thus

[
\exp(s\boldsymbol\xi)
=====================

\begin{pmatrix}
R(s)&\mathbf t(s)\
0&1
\end{pmatrix}.
]

---

# 16. Polynomial approximation without trigonometric functions

Your intuition is correct that the small-angle exponential can be evaluated very cheaply by polynomials. The important caveat is that a naïvely truncated matrix exponential does not preserve orthogonality exactly.

The safest approach is to construct an approximate quaternion and normalize it.

Let

[
\mathbf u=\frac12s\boldsymbol\phi,
\qquad
a=\mathbf u\cdot\mathbf u.
]

The exact quaternion is

[
q=
\left(
\cos\sqrt a,;
\frac{\sin\sqrt a}{\sqrt a}\mathbf u
\right).
]

Use the polynomials

[
c(a)
====

1-\frac a2+\frac{a^2}{24}-\frac{a^3}{720},
]

[
h(a)
====

1-\frac a6+\frac{a^2}{120}-\frac{a^3}{5040}.
]

Construct

[
q_{\mathrm{raw}}
================

\left(
c(a),;
h(a)\mathbf u
\right)
]

and normalize:

[
\boxed{
q=
\frac{q_{\mathrm{raw}}}
{|q_{\mathrm{raw}}|}.
}
]

This requires:

* dot products;
* fused multiply-add operations;
* one reciprocal square root;
* no square root before the polynomial;
* no sine or cosine.

The normalization guarantees that the resulting rotation is exactly orthogonal, up to floating-point error, even though the angle is approximate.

For (|s\phi|\lesssim1) radian, the sixth-order polynomial should be much more accurate than the underlying force-field approximation is likely to require. A lower fourth-order form may already suffice for molecular bending amplitudes below roughly (0.5) radian.

---

## 16.1 Polynomial (SE(3)) translation

The translational coefficients can also avoid trigonometric functions.

Let

[
x=\theta^2.
]

Then

[
\frac{1-\cos(s\theta)}{\theta^2}
================================

## \frac{s^2}{2}

\frac{s^4x}{24}
+
\frac{s^6x^2}{720}
-\cdots
]

and

[
\frac{s\theta-\sin(s\theta)}{\theta^3}
======================================

## \frac{s^3}{6}

\frac{s^5x}{120}
+
\frac{s^7x^2}{5040}
-\cdots.
]

Therefore

[
\mathbf t(s)
\approx
s\boldsymbol\rho
+
\left(
\frac{s^2}{2}
-\frac{s^4x}{24}
+\frac{s^6x^2}{720}
\right)
\Omega\boldsymbol\rho
]

[
+
\left(
\frac{s^3}{6}
-\frac{s^5x}{120}
+\frac{s^7x^2}{5040}
\right)
\Omega^2\boldsymbol\rho.
]

No division by (\theta) is needed, so this is also numerically well behaved near zero.

---

# 17. Your binomial approximation and rotation exponentials

For a scalar,

[
\left(1-\frac{x}{n}\right)^n
\rightarrow e^{-x}.
]

Taking the logarithm,

[
n\log\left(1-\frac{x}{n}\right)
===============================

-x-\frac{x^2}{2n}
-\frac{x^3}{3n^2}
-\cdots.
]

Therefore the leading exponent error is

[
O\left(\frac{x^2}{n}\right).
]

At fixed (x), this is algebraic convergence in (n). A Taylor polynomial has factorially decreasing truncation error on a bounded interval, so the binomial limit does not converge faster mathematically.

Nevertheless, your approximation can be attractive for radial potentials because:

* it uses repeated multiplication;
* it naturally reaches zero at (x=n);
* it can be converted into a compact-support polynomial.

For rotations, however, directly using

[
\left(I+\frac{\Omega}{n}\right)^n
]

is dangerous.

A skew-symmetric matrix has imaginary eigenvalues (i\theta). The corresponding factor has modulus

[
\left|1+\frac{i\theta}{n}\right|
================================

\sqrt{1+\frac{\theta^2}{n^2}}>1.
]

After (n) products, its modulus is

[
\left(
1+\frac{\theta^2}{n^2}
\right)^{n/2}>1.
]

Thus the transformation expands vectors instead of being a pure rotation.

---

# 18. Cayley approximation: the rotation-safe analogue

A much better repeated-factor approximation is the Cayley transform:

[
\boxed{
C(\Omega)
=========

\left(I-\frac12\Omega\right)^{-1}
\left(I+\frac12\Omega\right).
}
]

For skew-symmetric (\Omega),

[
C(\Omega)^TC(\Omega)=I.
]

Thus it is exactly orthogonal.

It approximates the exponential as

[
C(\Omega)
=========

\exp(\Omega)+O(|\Omega|^3).
]

A scaled repeated form is

[
\exp(\Omega)
\approx
\left[
C\left(\frac{\Omega}{n}\right)
\right]^n.
]

Its rotation-angle error decreases approximately as

[
O\left(\frac{\theta^3}{n^2}\right)
]

while every intermediate and final matrix remains orthogonal.

The equivalent quaternion construction is particularly simple:

[
q_{\mathrm{step}}
=================

\operatorname{normalize}
\left(
1,\frac{\boldsymbol\phi}{2n}
\right),
]

[
q\approx q_{\mathrm{step}}^n.
]

For (n=2,4,8), exponentiation by squaring makes this inexpensive. It is conceptually close to your repeated-polynomial idea, but normalization puts the factor on the rotation group.

In practice, I expect the normalized polynomial quaternion to be faster than repeated Cayley factors, but the Cayley form is an excellent reference implementation and stability check.

---

# 19. Reducing the number of exponentials

Even exact (SE(3)) interpolation may not be as expensive as it first appears because the expensive quantities are shared.

For two frames:

1. compute
   [
   \boldsymbol\xi=\log(D_1^{-1}D_2)
   ]
   once per molecule;
2. each atom only evaluates
   [
   \exp(s_i\boldsymbol\xi).
   ]

Further reductions are possible.

## Quantized or group-shared weights

Atoms in the same aromatic ring or molecular cross-section can share (s_i). Then calculate one interpolated transform per distinct weight and apply it to all atoms in that group.

For pentacene, one could use one weight per carbon cross-section or per ring rather than one independent weight per atom.

## Polynomial exponential

Use the quaternion and translation polynomials above.

## Hybrid interpolation

Use exact nonlinear interpolation only for the frame orientations, but linearly blend translation:

[
q_i=
\operatorname{normalize}
\left(
\sum_aw_{ia}q_a
\right),
]

[
\mathbf t_i=\sum_aw_{ia}\mathbf t_a.
]

This is not the full (SE(3)) exponential but may be accurate enough for mostly bending and twisting planar molecules.

## Linear blend for small deformation

If

[
\max_{a,b}
|\log(D_a^{-1}D_b)|
<
\epsilon,
]

use linear blending. Switch to nonlinear blending only above the threshold.

A smooth switching function would avoid force discontinuities.

---

# 20. Nonlinear fitting dataset

The harmonic Hessian determines only local behavior around equilibrium. Fitting finite-amplitude bending and twisting requires distorted geometries.

There are two main ways to generate them.

---

## 20.1 Prescribe frame deformation and relax atoms

This is the cleanest route.

For each sample (s):

1. choose frame coordinates (\mathbf q_s);
2. reconstruct an initial atom geometry using the current weights;
3. constrain the coarse frame coordinates;
4. relax all remaining atomistic DOFs;
5. record:

   * relaxed energy (E_s);
   * generalized restoring force (\mathbf Q_s);
   * relaxed atom geometry.

The effective relaxed energy is

[
E_{\mathrm{eff}}(\mathbf q)
===========================

\min_{\mathbf x:,C(\mathbf x)=\mathbf q}
V_{\mathrm{AA}}(\mathbf x).
]

If the constrained relaxation uses Lagrange multipliers,

[
{\cal L}
========

V_{\mathrm{AA}}(\mathbf x)
+
\boldsymbol\lambda^T
[C(\mathbf x)-\mathbf q],
]

then at the optimum the multipliers provide the generalized energy derivative, up to the sign convention:

[
\frac{dE_{\mathrm{eff}}}{d\mathbf q}
====================================

-\boldsymbol\lambda.
]

This is often more accurate than calculating finite differences of the relaxed energy.

---

## 20.2 Fit from atomistic snapshots

Alternatively, obtain configurations from:

* finite-temperature MD;
* mode sampling;
* random force perturbations;
* adsorbed-molecule simulations;
* constrained bending and twisting scans.

For each snapshot, infer the frame poses by minimizing

[
\min_{{G_a}}
\sum_i
m_i
\left|
\mathbf x_i-
\Phi_i(G_1,\ldots,G_K;W)
\right|^2.
]

For hard fragment assignments, each pose is obtained by weighted Kabsch alignment.

For overlapping nonlinear interpolation, the poses are coupled and require an iterative fit.

This route is more statistically representative, but the mapping (W) and the extracted coarse coordinates influence one another. Joint optimization becomes substantially more complicated.

---

# 21. Recommended staged nonlinear fitting

I would not optimize weights, frame poses, nonlinear interpolation, stiffness and anharmonic potential simultaneously from the beginning. The problem has too many compensating freedoms.

Use the following stages.

## Stage A: geometric weight construction

Compute

[
W^{(0)}
]

from radial, axial, graph-based, or biharmonic interpolation.

## Stage B: harmonic weight tuning

Optimize

[
W
]

against low-frequency Hessian modes using variable projection.

## Stage C: choose interpolation

Compare:

* linear blend;
* normalized dual quaternions;
* polynomial (SE(3)) exponential.

Freeze the interpolation choice.

## Stage D: generate relaxed nonlinear data

Using the fixed mapping, sample frame deformations and atomistically relax the fine coordinates.

## Stage E: fit nonlinear potential

Fit energy and generalized forces.

## Stage F: optional joint fine tuning

Allow small changes in (W) under a strong penalty

[
\lambda_0|W-W_{\mathrm{harmonic}}|^2.
]

This prevents the nonlinear energy model and the interpolation weights from arbitrarily compensating each other.

---

# 22. Nonlinear potential fitting

Let the reduced potential be

[
V_\theta(\mathbf q)
===================

\sum_p\theta_p\psi_p(\mathbf q).
]

The force is

[
\mathbf Q_\theta
================

-\nabla_{\mathbf q}V_\theta.
]

Fit

[
\boxed{
{\cal L}_{\mathrm{nonlinear}}
=============================

\sum_s
\left[
\frac{
(V_\theta(\mathbf q_s)-E_s)^2
}{
\sigma_E^2
}
+
\lambda_F
\frac{
\left|
\mathbf Q_\theta(\mathbf q_s)-\mathbf Q_s
\right|^2
}{
\sigma_Q^2
}
\right]
+
{\cal R}(\theta).
}
]

If the basis functions (\psi_p) are fixed, this remains linear least squares in (\theta_p), even when the potential is anharmonic.

For example,

[
\psi_p
\in
{
\eta_\alpha\eta_\beta,;
\eta_\alpha\eta_\beta\eta_\gamma,;
\eta_\alpha^4,;
1-\cos\eta_\alpha
}.
]

Energy and force fitting together is much better conditioned than fitting energies alone.

---

## 22.1 Symmetry-adapted terms

For a planar molecule symmetric under reflection through the molecular plane, an out-of-plane bending coordinate (b_z) changes sign under reflection. The energy must therefore satisfy

[
V(b_z)=V(-b_z)
]

unless the environment breaks the symmetry.

Allowed isolated-molecule terms include

[
b_z^2,\qquad b_z^4,
]

but not

[
b_z,\qquad b_z^3.
]

Similarly, inversion or mirror symmetries can eliminate many cubic and mixed coefficients.

This is important because a general quartic polynomial in 18 coordinates would be unmanageable, while a symmetry-adapted sparse expansion can remain compact.

---

# 23. Joint nonlinear optimization variables

A general joint parameter vector could contain:

[
\Theta=
{
Z_W,;
L_K,;
\theta_{\mathrm{anh}},;
\text{interpolation parameters}
}.
]

Here:

* (Z_W) are weight logits;
* (K=L_KL_K^T+\epsilon I) guarantees positive stiffness;
* (\theta_{\mathrm{anh}}) are nonlinear potential coefficients.

A total loss could be

[
{\cal L}
========

\lambda_{\mathrm{mode}}{\cal L}_{\mathrm{mode}}
+
\lambda_E{\cal L}_E
+
\lambda_F{\cal L}*F
+
\lambda_H{\cal L}*{H}
+
\lambda_W{\cal R}_W
+
\lambda_K{\cal R}_K.
]

Use automatic differentiation for the nonlinear stage unless there is a strong reason to derive all (SE(3)) Jacobians manually. The optimization is offline; production evaluation can still use hand-optimized analytic formulas.

---

# 24. Efficient runtime for (K) frames

For sparse linear blending:

```text
for each molecule:
    for frame a:
        load current position c[a]
        convert quaternion q[a] to rotation matrix R[a]

    zero frame forces and torques

    for atom i:
        x = 0

        for local influence l:
            a = frame_index[i][l]
            w = weight[i][l]

            y = R[a] * rho[i][l]
            x += w * (c[a] + y)

            cache y or recompute later

        atom_position[i] = x

    evaluate nonbonded atom forces

    for atom i:
        f = atom_force[i]

        for local influence l:
            a = frame_index[i][l]
            w = weight[i][l]
            y = cached_or_recomputed_rotated_reference[i][l]

            frame_force[a]  += w * f
            frame_torque[a] += w * cross(y, f)

    for frame edge e=(a,b):
        xi = log_SE3(inverse(D[a]) * D[b])
        energy += edge_potential[e](xi)
        compute equal/opposite edge wrenches
```

The atomic force reduction is exactly the Jacobian transpose of the linear-blend map:

[
\mathbf F_a
===========

\sum_iw_{ia}\mathbf f_i,
]

[
\boldsymbol\tau_a
=================

\sum_i
w_{ia}
(R_a\boldsymbol\rho_{ia})
\times\mathbf f_i.
]

On a GPU, avoiding atomics requires processing one molecule by one workgroup or one thread block, accumulating frame wrenches in local memory, then reducing within the group. Since (K) is small, this is a good fit.

---

# 25. Most important caveats

## The mapping and potential are not independently identifiable

A slightly different weight map can often be compensated by a different frame stiffness. Strong geometric priors and staged fitting are therefore essential.

## More frames can reintroduce hard frequencies

Frames should be viewed as geometric control nodes, not automatically as all dynamic coordinates.

## Local fragment frames may need nearly discontinuous weights

Across a rotatable single bond, smooth weights can incorrectly smear one fragment’s rotation into the other. The graph smoothness coefficient should be smaller across flexible junctions.

## Exact (SE(3)) interpolation is not always worth its runtime cost

The correct comparison should include:

* geometry error;
* bond-distortion error;
* energy error;
* force error;
* total simulation runtime.

Dual quaternions or normalized polynomial quaternion interpolation may provide a better compromise.

## The exponential approximation must remain on the rotation group

A raw Taylor matrix or binomial matrix approximation can generate scaling and shear. Quaternion normalization or Cayley transforms avoid this.

## Nonlinear reference data depend on the mapping

If weights are changed substantially, constrained-relaxation data generated with the old mapping are no longer exactly the same coarse potential. This is another reason to fit weights first and nonlinear energy second.

## Environment-dependent flexibility

Adsorption, charging, excitation, or chemical substitution may change the effective frame stiffness. It may be useful to keep the geometric weights fixed but fit several environment- or charge-state-dependent potentials.

---

# 26. Concrete model choices for your examples

## Pentacene

Start with:

* three frames along the long axis;
* piecewise-linear or cubic B-spline weights;
* at most two neighboring frame influences per atom;
* retain perhaps 4–8 soft internal frame combinations.

Two frames may reproduce global twist and first bending, but three frames are necessary for central bowing where both ends remain nearly unchanged.

## PTCDA

Compare:

1. two frames at the two anhydride ends;
2. four quadrant frames;
3. three frames along the long axis.

The four-frame model should be better for saddle and transverse plate-like bending. Use bilinear geometric weights or graph biharmonic weights as initialization.

## 1,2,4,5-tetraphenylbenzene

Use:

* one central benzene frame;
* one frame per phenyl;
* almost hard assignment of atoms to fragments;
* a narrow blending region around each inter-ring bond;
* hinge-like torsional coordinates for each phenyl;
* optional collective bending coordinates of the central ring.

Do not propagate all 24 internal frame coordinates unless they have been demonstrated to be soft.

---

# 27. Literature most relevant to these extensions

The rigid-block molecular precedent remains RTB normal-mode analysis, where translations and rotations of molecular blocks form a reduced low-frequency basis. NOLB extends this to nonlinear finite block motions. ([ACS Publications][7])

The (SE(3)) exponential interpolation and frame-graph interpretation are closely related to geometrically exact beam formulations on the special Euclidean group. ([ScienceDirect][6])

For automatic smooth nonnegative partition-of-unity weights, bounded biharmonic skinning is probably the most directly useful mathematical analogy to add to the literature review. ([igl.ethz.ch][1])

For efficient nonlinear rigid-transformation blending, dual-quaternion skinning is the closest computational precedent. ([users.cs.utah.edu][4])

For optimization, FIRE provides the atomistic pseudo-dynamical minimizer you already know, while L-BFGS-B supplies a rigorous bound-constrained quasi-Newton alternative. ([APS Links][2])

---

## Recommended overall formulation

The method I would now implement is:

[
\boxed{
\begin{aligned}
&\text{Choose }K\text{ frames and a sparse frame graph};\
&\text{construct topology-aware biharmonic initial weights};\
&\text{fine-tune weights against low Hessian modes by variable projection};\
&\text{identify and retain only soft frame-space coordinates};\
&\text{use linear blending as baseline and normalized polynomial}\
&\qquad\text{quaternion or dual-quaternion blending for finite rotations};\
&\text{generate constrained-relaxed nonlinear training configurations};\
&\text{fit a sparse symmetry-adapted energy to energies and forces};\
&\text{optionally relax unresolved atomistic coordinates during dynamics.}
\end{aligned}
}
]

The combination of **arbitrary sparse frame topology, physically optimized weights, elimination of stiff frame directions, and group-preserving polynomial rotation interpolation** is considerably stronger and more distinctive than the original two-frame concept.

[1]: https://igl.ethz.ch/projects/bbw/bounded-biharmonic-weights-siggraph-2011-jacobson-et-al.pdf?utm_source=chatgpt.com "Bounded Biharmonic Weights for Real-Time Deformation"
[2]: https://link.aps.org/doi/10.1103/PhysRevLett.97.170201?utm_source=chatgpt.com "Structural Relaxation Made Simple | Phys. Rev. Lett."
[3]: https://dl.acm.org/doi/10.1145/279232.279236?utm_source=chatgpt.com "Algorithm 778: L-BFGS-B: Fortran subroutines for large- ..."
[4]: https://users.cs.utah.edu/~ladislav/kavan07skinning/kavan07skinning.pdf?utm_source=chatgpt.com "Skinning with Dual Quaternions"
[5]: https://epubs.siam.org/doi/10.1137/S0895479801383877?utm_source=chatgpt.com "Means and Averaging in the Group of Rotations"
[6]: https://www.sciencedirect.com/science/article/abs/pii/S0045782513002600?utm_source=chatgpt.com "Geometrically exact beam finite element formulated on the ..."
[7]: https://pubs.acs.org/doi/abs/10.1021/acs.jctc.7b00197?utm_source=chatgpt.com "NOLB: Nonlinear Rigid Block Normal-Mode Analysis Method"
