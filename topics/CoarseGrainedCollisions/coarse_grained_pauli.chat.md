https://chatgpt.com/share/6a4f9048-012c-83ed-9e1b-ef8379032d27
https://chatgpt.com/share/6a4f9054-d984-83ed-b21f-cb83f64df97f
https://share.gemini.google/xEK6WGR5b66b
https://chat.deepseek.com/share/kfdh3fnlojoix921o3


# USER

Think about representation resp approximation of collisions in 2D and 3D. I have object made of spheres like molecule and want to coarsegrain it so I can compute approximative collision fast. For start think in 2D for simplicity (although later we want to generalize to 3D). 

Very common approximation is make convex hull, we consider this level 1 approximation (level0 is sphere or AABB bounding box)

Now consider some other options especially how fast or complicated is to compute pairwise collision between two objects with such representations. 

I was thinking about something like multipolar expansion (up to quadruple) where we do not compute electric field but shape of bounding surface (distance from center). Common way are ellipsoides - this is generalization.

But for exact ellipsoide-elipsoode collision one has  to do rather complex math with costly square root. I want to avoid that. To have fast math for arbitrarily rotated quadrupole expansion of shape, similar to Skater-Koster tables (the angular part), it can be approximative. Ideally if we can compute not just boolean (collide or not) but also contact forces which are basically gradient of signed distance field.

Note other limit is simply have some sum of radial potential from atoms. But that is what I want to approximate. If I have let's say benzene (12 atoms) to compute collision between two benzene that is 12*12 pairwise Interactions (distance field can be computed without square root if I use as radial functions only even power of r like (1-(r\rc)2)2. My coarsegrain to be competitive must be faster than that.

---

# Gemini

This is a brilliant approach. Trying to compute exact intersections between geometric ellipsoids (like the Perram-Wertheim method or Gay-Berne potentials) inevitably drags you into solving generalized eigenvalue problems or roots of polynomials. That completely kills performance in highly parallel compute environments, especially if you want to avoid branching and costly square root operations.

Your intuition to use a "shape multipole" expansion and generalize the angular mechanics of Slater-Koster tables is spot on. We can formalize this using **Cartesian Shape Tensors**. By applying a Taylor expansion to a simple radial polynomial, we can compute an overlap scalar field (and its force-yielding gradient) using nothing but fused multiply-adds (FMAs) and matrix-vector multiplications.

Here is how you can build a rigorously fast, square-root-free coarse-grained collision model in 2D or 3D.

### The Cartesian Shape Tensor Expansion

Instead of atomic points, let's treat a molecule as a continuous shape density field $\rho(\mathbf{r})$.

If we have a base radial interaction potential $\phi(R^2)$ that relies strictly on squared distance (like your example $\phi(R^2) = (1 - R^2/R_c^2)^2$), the total overlap energy between molecule A at the origin and molecule B at position $\mathbf{R}$ is the convolution:

$$E(\mathbf{R}) = \iint \rho_A(\mathbf{r}_A) \rho_B(\mathbf{r}_B) \phi(|\mathbf{R} + \mathbf{r}_B - \mathbf{r}_A|^2) \,d\mathbf{r}_A \,d\mathbf{r}_B$$

To avoid calculating all atomic pairs, we Taylor expand $\phi$ around the squared center-of-mass distance $R^2 = |\mathbf{R}|^2$. Let $\mathbf{s} = \mathbf{r}_B - \mathbf{r}_A$.

$$\phi(|\mathbf{R} + \mathbf{s}|^2) \approx \phi(R^2) + \nabla \phi(R^2) \cdot \mathbf{s} + \frac{1}{2} \mathbf{s}^T \mathbf{H}(R^2) \mathbf{s}$$

Because $\phi$ is strictly a function of $R^2$, its derivatives map directly to the displacement vector $\mathbf{R}$ without needing length normalizations:

* **Gradient:** $\nabla \phi = 2 \phi'(R^2) \mathbf{R}$
* **Hessian:** $\mathbf{H} = 4 \phi''(R^2) \mathbf{R}\mathbf{R}^T + 2 \phi'(R^2) \mathbf{I}$

### Integrating the Shape (The "Slater-Koster" Analog)

If we place the origin of our local coordinate system at the center of the shape for both molecules, their dipole moments (center of mass/geometry) integrate to zero: $\int \mathbf{r} \rho(\mathbf{r}) d\mathbf{r} = 0$.

When we integrate the quadratic term over the two density fields, the cross-terms vanish, and we are left strictly with the **Shape Quadrupole Moment (Covariance Matrix)** of each molecule:

$$\mathbf{Q} = \int \mathbf{r} \mathbf{r}^T \rho(\mathbf{r}) \,d\mathbf{r}$$

The final coarse-grained interaction energy collapses into a beautiful, computationally cheap form:

$$E(\mathbf{R}) = M_A M_B \phi(R^2) + 2 \phi''(R^2) \left[ \mathbf{R}^T (M_A \mathbf{Q}_B + M_B \mathbf{Q}_A) \mathbf{R} \right] + \phi'(R^2) \text{Tr}(M_A \mathbf{Q}_B + M_B \mathbf{Q}_A)$$

Where:

* $M$ is the Monopole (the total "volume" or sum of atomic weights in the coarse-grain).
* $\mathbf{Q}$ is the $2 \times 2$ (in 2D) or $3 \times 3$ (in 3D) shape matrix.
* $\phi'$ and $\phi''$ are just the derivatives of your polynomial with respect to $R^2$. For $\phi(w) = (1 - w/w_c)^2$, these are trivial constants and linear terms.

### Why This Crushes Pairwise $12 \times 12$ Interactions

Let's look at the math required for the interaction $E(\mathbf{R})$ of two rotated benzenes:

1. **Rotation:** When a molecule rotates by a matrix $\mathbf{U}$, you just rotate its shape tensor once per time step: $\mathbf{Q}_{rot} = \mathbf{U} \mathbf{Q} \mathbf{U}^T$.
2. **The Math:** To compute the anisotropic part of the collision, you evaluate $\mathbf{R}^T \mathbf{Q}_{total} \mathbf{R}$.
* In 3D, $\mathbf{Q}_{total} \mathbf{R}$ is a matrix-vector multiplication (9 multiply-adds).
* Dotting that with $\mathbf{R}^T$ is 3 multiply-adds.


3. **Total Cost:** You are computing a few scalar polynomials and one tensor contraction. This takes roughly **30-40 FLOPs** per molecule pair.

By contrast, computing the pairwise sum of 12x12 atoms requires 144 evaluations of distance squared ($144 \times 3$ FLOPs) plus 144 polynomial evaluations ($144 \times 3$ FLOPs). The shape tensor approach is well over **an order of magnitude faster** and inherently branchless, making it an absolute dream for SIMD or compute shader execution.

### Computing Contact Forces and Torques

You mentioned wanting the gradient of the signed distance field for contact forces. Because our energy $E(\mathbf{R})$ acts as a continuous pseudo-potential, the forces $\mathbf{F} = -\nabla_{\mathbf{R}} E$ drop right out via the chain rule.

Taking the derivative of $\mathbf{R}^T \mathbf{Q} \mathbf{R}$ with respect to $\mathbf{R}$ simply yields $2 \mathbf{Q} \mathbf{R}$.

$$\mathbf{F}_{contact} \propto \phi'''(R^2)(\dots)\mathbf{R} + \phi''(R^2)(\mathbf{Q}_{total}\mathbf{R})$$

You get a force vector that points normal to the overlapping iso-surfaces of your multipoles, naturally pushing flat shapes (like benzene discs) apart along their shortest overlapping axis. Torques are similarly fast, computed via the cross product of the rotated $\mathbf{Q}$ tensors with the displacement fields.

This formulation essentially builds a custom, localized Slater-Koster scheme, but instead of projecting $s, p, d$ orbital symmetries, you are projecting the Cartesian moments of a molecular geometry onto the displacement axis.

Do you want to sketch out the exact GLSL/WGSL implementation for extracting the forces and torques from this tensor contraction, or would you rather focus first on how to parameterize the $\mathbf{Q}$ matrices for specific geometries like your 12-atom benzene?

---

# Chat GPT 5.5

The expensive part of ellipse–ellipse collision is not really the square root. It is that the contact normal is unknown, so an exact algorithm must solve an optimization problem. Avoiding one `sqrt` while retaining that optimization will not buy much.

For your purpose I see **three especially useful representations**:

1. **Gaussian/quadrupolar overlap metric** — fastest smooth approximation.
2. **Support-function multipoles** — the mathematically correct version of your “multipolar bounding surface.”
3. **Rounded zonotope or rounded small polygon** — probably the best compromise between speed, accuracy, and robustness.

The first and third could easily beat (12\times12=144) atom-pair evaluations.

---

# 1. First observation: expand the support function, not the radial function

For a convex body (A), define its support function

[
h_A(\mathbf n)=\max_{\mathbf x\in A}\mathbf n\cdot \mathbf x ,
\qquad |\mathbf n|=1.
]

It gives the position of the tangent line or plane with outward normal (\mathbf n).

This is much more useful than the radial function (r(\theta)), because support functions obey

[
h_{A\oplus B}(\mathbf n)=h_A(\mathbf n)+h_B(\mathbf n),
]

where (\oplus) is the Minkowski sum. Collision detection between (A) and (B) is equivalent to testing whether the relative displacement lies inside (A\oplus(-B)). Support functions are therefore naturally additive in exactly the operation needed for collision detection. ([swt.informatik.uni-freiburg.de][1])

Let

[
\mathbf d=\mathbf c_B-\mathbf c_A.
]

Define

[
s(\mathbf n)
============

## \mathbf d\cdot\mathbf n

## h_A(\mathbf n)

h_B(-\mathbf n).
]

Then:

[
\max_{|\mathbf n|=1}s(\mathbf n)>0
]

means that a separating direction exists, so the objects do not intersect.

For separated convex objects,

[
D(A,B)=\max_{|\mathbf n|=1}s(\mathbf n).
]

Thus the whole collision problem becomes an optimization over direction. In 2D this is only a one-dimensional angular optimization.

That is the clean mathematical foundation for your multipolar idea.

---

# 2. Quadrupolar Gaussian-overlap metric

This is likely the **fastest useful approximation**.

Instead of approximating each solid surface and then performing exact geometric collision, approximate the entire relative-contact region by a quadratic form.

Represent each object by a symmetric positive-definite shape matrix

[
S_A,\qquad S_B.
]

After rotation,

[
A=R_A S_A R_A^T,
\qquad
B=R_B S_B R_B^T.
]

Construct a pair matrix

[
C=c_A A+c_B B.
]

Then define

[
q(\mathbf d)=\mathbf d^T C^{-1}\mathbf d.
]

The approximate collision test is simply

[
q<1.
]

This is essentially the geometric core behind Gaussian-overlap and Berne–Pechukas/Gay–Berne-type coarse-grained molecular models: orientation dependence is encoded in quadratic tensors, and the contact distance is obtained analytically rather than by solving exact ellipsoid contact. ([AIP Publishing][2])

## Explicit 2D version

Let

[
C=
\begin{pmatrix}
a & b\
b & c
\end{pmatrix},
\qquad
\Delta=ac-b^2.
]

Then

[
q=
\frac{
c,d_x^2-2b,d_xd_y+a,d_y^2
}{
\Delta
}.
]

This requires:

* three additions to combine the matrices,
* a (2\times2) determinant,
* one reciprocal,
* several multiply-add operations.

No square root is needed for the Boolean test.

A realistic implementation is perhaps **25–40 scalar arithmetic operations plus one reciprocal**, assuming the rotated matrices (A) and (B) are cached per object.

That is vastly cheaper than 144 atom-pair tests.

## Smooth force

Use a compact polynomial penalty

[
\delta=1-q,
]

[
U(q)=
\begin{cases}
k\delta^p, & q<1,\
0, & q\ge1.
\end{cases}
]

For (p\ge2), the force goes smoothly to zero at the cutoff.

Let

[
\mathbf y=C^{-1}\mathbf d.
]

Since

[
\nabla_{\mathbf d}q=2\mathbf y,
]

the repulsive force on (B) is

[
\boxed{
\mathbf F_B
===========

2kp(1-q)^{p-1}\mathbf y
}
]

for (q<1).

This is especially attractive: the contact force automatically points along the gradient of the anisotropic contact metric, rather than simply along the center–center direction.

## Torque

Because

[
q=\mathbf d^T C^{-1}\mathbf d,
]

the orientation derivative is

[
\frac{\partial q}{\partial\alpha_i}
===================================

-\mathbf y^T
\frac{\partial C}{\partial\alpha_i}
\mathbf y.
]

In 2D,

[
\frac{\partial A}{\partial\alpha}
=================================

\Omega A-A\Omega,
\qquad
\Omega=
\begin{pmatrix}
0&-1\
1&0
\end{pmatrix}.
]

Therefore

[
\tau_i
======

kp(1-q)^{p-1}
\frac{\partial q}{\partial\alpha_i}.
]

So forces and torques are branch-light and fully analytic.

## It is not a true signed distance

The value (1-q) is a dimensionless penetration measure. Its effective stiffness changes with the dimensions of the ellipsoid.

Near the surface, a first-order Euclidean distance estimate is

[
g
\approx
\frac{q-1}{|\nabla q|}
======================

\frac{q-1}{2|C^{-1}\mathbf d|}.
]

This needs one reciprocal square root, but only for pairs that actually reached the contact stage. One `rsqrt` per colliding object pair is negligible compared with 144 atomic interactions.

---

# 3. A conservative quadrupolar outer bound

The simple Gaussian form is normally fitted for accuracy, not guaranteed containment. But you can make a rigorous conservative ellipsoidal culling test.

Suppose objects are enclosed by ellipsoids with support functions

[
h_A(\mathbf n)=\sqrt{\mathbf n^T A\mathbf n},
\qquad
h_B(\mathbf n)=\sqrt{\mathbf n^T B\mathbf n}.
]

For any (\beta>0),

[
\left(
\sqrt{x}+\sqrt{y}
\right)^2
\le
\left(1+\frac1\beta\right)x
+
(1+\beta)y.
]

Therefore the Minkowski sum is enclosed by the ellipsoid

[
\boxed{
C(\beta)
========

\left(1+\frac1\beta\right)A
+
(1+\beta)B
}.
]

Hence

[
\mathbf d^T C(\beta)^{-1}\mathbf d>1
]

guarantees separation.

For similarly sized objects, use (\beta=1):

[
C=2(A+B).
]

For approximately homothetic objects with characteristic radii (r_A,r_B), a sensible choice is

[
\beta\approx \frac{r_A}{r_B}.
]

This gives a cheap, conservative intermediate test:

1. sphere rejection,
2. quadratic outer-ellipsoid rejection,
3. more accurate contact model only for ambiguous pairs.

Exact ellipsoid contact functions such as Perram–Wertheim are substantially more complicated because they optimize an auxiliary contact parameter; the analytical Gaussian-overlap approximations trade some accuracy for a fixed-cost formula. ([ScienceDirect][3])

---

# 4. Actual “shape multipoles”: Fourier support functions in 2D

For a smooth convex object, expand

[
h(\phi)
=======

a_0+
\sum_{m=1}^{L}
\left[
a_m\cos(m\phi)+b_m\sin(m\phi)
\right].
]

Interpretation:

* (m=0): isotropic radius,
* (m=1): translation of the chosen center,
* (m=2): quadrupolar or elliptic deformation,
* (m=3): triangular asymmetry,
* (m=4): square-like or rectangular deformation,
* (m=6): hexagonal shape, particularly relevant for benzene.

Rotation by angle (\alpha) merely rotates each coefficient pair by (m\alpha):

[
\begin{pmatrix}
a_m'\
b_m'
\end{pmatrix}
=============

\begin{pmatrix}
\cos m\alpha&-\sin m\alpha\
\sin m\alpha& \cos m\alpha
\end{pmatrix}
\begin{pmatrix}
a_m\
b_m
\end{pmatrix}.
]

This is almost exactly the Slater–Koster-like angular algebra you described.

For two objects,

[
H(\phi)
=======

h_A(\phi-\alpha_A)
+
h_B(\phi+\pi-\alpha_B),
]

and

[
s(\phi)
=======

d_x\cos\phi+d_y\sin\phi-H(\phi).
]

You need the maximum of (s(\phi)).

## Cheap optimization

The derivatives are explicit:

[
s'(\phi)
========

## \mathbf d\cdot\mathbf t

h_A'-
h_B',
]

[
s''(\phi)
=========

## -\mathbf d\cdot\mathbf n

h_A''-
h_B''.
]

where

[
\mathbf n=(\cos\phi,\sin\phi),
\qquad
\mathbf t=(-\sin\phi,\cos\phi).
]

A practical algorithm is:

1. Start from the center direction
   [
   \mathbf n_0=\frac{\mathbf d}{|\mathbf d|}.
   ]

2. Evaluate (s,s',s'').

3. Perform one or two clamped Newton updates
   [
   \Delta\phi=-\frac{s'}{s''}.
   ]

4. Use the final value (s(\phi_*)) as the approximate signed separation.

For low order (L=2,4), this might take roughly 50–150 arithmetic operations, depending on how many refinement iterations you use.

## Contact point and force

For a smooth planar support function, the boundary point whose normal is (\mathbf n) is

[
\boxed{
\mathbf x(\phi)
===============

h(\phi)\mathbf n
+
h'(\phi)\mathbf t
}.
]

This is very useful. Once the maximizing direction is found, you immediately obtain:

* contact normal,
* support point on (A),
* support point on (B),
* lever arms for torque.

The envelope theorem also means that, once (\phi_*) is optimized,

[
\nabla_{\mathbf d}s_*=\mathbf n_*.
]

Thus a penalty based on (s_*) naturally produces a force along the optimized separating normal.

---

# 5. Important limitation of a pure quadrupole

Not every Fourier series is a valid convex support function.

In 2D, convexity requires

[
h(\phi)+h''(\phi)\ge0.
]

For a pure quadrupole

[
h(\phi)=r_0+A\cos(2\phi-\phi_2),
]

we have

[
h+h''
=====

r_0-3A\cos(2\phi-\phi_2).
]

Therefore

[
r_0\ge3|A|.
]

The largest possible ratio of maximum to minimum support is then

[
\frac{r_0+|A|}{r_0-|A|}
\le2.
]

So a convex support function containing only (m=0) and (m=2) cannot represent a very elongated object. Higher harmonics or a piecewise-linear support representation are needed.

This is one reason ellipsoids look more expressive than a truncated polynomial support expansion: the square root in

[
h_{\rm ellipse}(\mathbf n)
==========================

\sqrt{\mathbf n^T A\mathbf n}
]

implicitly contains infinitely many Fourier harmonics.

For nearly circular molecules such as benzene viewed in its plane, however, (m=6) may be more important than a strong quadrupole.

---

# 6. Rounded zonotope: probably the best geometric representation

For a centrally symmetric molecule, I would seriously consider a **rounded zonotope**:

[
K
=

\left{
\sum_{k=1}^{K_g}\xi_k\mathbf g_k
;:;
-1\le\xi_k\le1
\right}
\oplus B(r).
]

Here:

* (\mathbf g_k) are a small number of generator vectors,
* (B(r)) is a disk or sphere of radius (r).

Its support function is

[
\boxed{
h(\mathbf n)
============

\sum_k|\mathbf n\cdot\mathbf g_k|
+
r|\mathbf n|
}.
]

For unit (\mathbf n),

[
h(\mathbf n)=r+\sum_k|\mathbf n\cdot\mathbf g_k|.
]

This needs only dot products, absolute values, and additions.

No matrices, no nonlinear shape functions, no trigonometry.

## Benzene in 2D

A centrally symmetric hexagon is a zonotope generated by three line segments. Thus a benzene-like collision body can be represented by

[
K_{\rm benzene}
===============

[-\mathbf g_1,\mathbf g_1]
\oplus
[-\mathbf g_2,\mathbf g_2]
\oplus
[-\mathbf g_3,\mathbf g_3]
\oplus B(r),
]

with generator directions separated by approximately (60^\circ).

That gives:

* three generator vectors,
* one rounding radius,
* six polygon sides,
* a much tighter fit than an ellipse,
* extremely cheap support queries.

In 3D, a flat aromatic molecule could use:

* three in-plane generators,
* one out-of-plane thickness generator,
* one spherical rounding radius.

So only four generators plus (r).

## Collision algorithm

Use GJK on the unrounded zonotope cores. GJK requires only support mappings and converts convex-body distance into a distance-to-origin problem in the Minkowski difference. ([graphics.stanford.edu][4])

The zonotope support point is

[
\boxed{
\operatorname{support}(\mathbf n)
=================================

\sum_k
\operatorname{sign}(\mathbf n\cdot\mathbf g_k)
\mathbf g_k
}.
]

After GJK returns the squared distance (d_{\rm core}^2) between the two cores, test

[
d_{\rm core}^2
<
(r_A+r_B)^2.
]

Thus:

* the GJK iterations require no square roots,
* the final Boolean test uses squared distance,
* only an accepted contact needs one `rsqrt` to obtain the force normal.

For (K_g=3) generators per body and perhaps 2–4 GJK iterations, this should be roughly on the order of **100–250 simple scalar operations**, depending on implementation.

That remains comfortably below a 144-pair atomistic interaction with polynomial potentials.

## Why zonotopes are especially attractive

They are the piecewise-linear analogue of your multipolar support expansion:

[
\cos(m\phi),\sin(m\phi)
\quad\longleftrightarrow\quad
|\mathbf n\cdot\mathbf g_k|.
]

The harmonic representation is smooth but requires angular maximization.

The zonotope representation is nonsmooth at a finite number of directions, but support maximization is trivial. Rounding by a disk or sphere restores smooth physical contact at edges and corners.

---

# 7. Small rounded polygon or polyhedron

A more general version is

[
K=P\oplus B(r),
]

where (P) is a polygon with perhaps 4–8 vertices.

Its support mapping is

[
\operatorname{support}_K(\mathbf n)
===================================

\operatorname{argmax}_{\mathbf v_j\in P}
\mathbf n\cdot\mathbf v_j
+
r\frac{\mathbf n}{|\mathbf n|}.
]

For Boolean collision, it is preferable to:

1. run GJK on the polygon cores,
2. compare the squared core distance to ((r_A+r_B)^2).

For equal-radius atom spheres, note that

[
\operatorname{conv}
\left(
\bigcup_i B(\mathbf x_i,r)
\right)
=======

\operatorname{conv}{\mathbf x_i}
\oplus B(r).
]

So the convex hull of equal-radius atomic disks or spheres is **exactly a rounded convex hull of the centers**.

For benzene, the core hull may have only six relevant outer vertices. GJK over six vertices is already far cheaper than 144 atomic pairs.

The original k-DOP idea is closely related: a k-DOP represents a convex object using a fixed small set of support directions. It is highly effective as a bounding volume, although arbitrary relative rotations make direct interval testing less trivial than for world-aligned boxes. ([ibr.cs.tu-bs.de][5])

---

# 8. Capsule or small primitive decomposition

For concave molecules, a single support function always loses concavity. Then I would use a very small union of convex primitives.

Examples:

* 2–4 capsules,
* 2–4 rounded boxes,
* 2–4 zonotopes,
* one central plate plus several protrusion spheres.

Capsule–capsule collision reduces to segment–segment distance:

[
d_{\rm seg}^2<(r_A+r_B)^2.
]

No square root is needed for Boolean collision.

For (M=2) capsules per molecule, only four primitive-pair tests are required. Even if each segment-distance calculation costs 30–50 operations, this is still much cheaper than 144 atom pairs.

This representation has an important advantage over all single-body quadrupolar models: it can capture actual molecular concavities and separate lobes.

---

# 9. What I would not recommend

## General superquadrics

An implicit function such as

[
F(x,y)=
\left|\frac{x}{a}\right|^p+
\left|\frac{y}{b}\right|^p-1
]

is cheap to evaluate at one point. But collision between two rotated superquadrics still requires finding a common closest point or common normal. Published superquadric contact methods generally solve nonlinear equations iteratively; the cheap implicit evaluation does not make the two-body problem cheap. ([web.ist.utl.pt][6])

## Generic radial multipoles

A radial expansion

[
r(\phi)=r_0+\sum_m r_m\cos(m\phi-\psi_m)
]

is good for rendering or ray intersection from the center, but Minkowski addition is not linear in (r). Pair collision therefore becomes an awkward two-angle optimization.

## Generic RBF implicit surfaces

A sum

[
F(\mathbf x)=\sum_i w_i f(|\mathbf x-\mathbf x_i|^2)
]

makes point-versus-object tests cheap, but object-versus-object contact still needs sampling or optimization. It does not eliminate the difficult part.

---

# 10. Rough cost comparison

These are approximate scalar-operation estimates, not hardware benchmarks.

| Representation                       | Typical 2D pair cost | Force quality          | Conservative possible?   |
| ------------------------------------ | -------------------: | ---------------------- | ------------------------ |
| Bounding circle                      |             5–10 ops | radial only            | yes                      |
| AABB                                 |             5–15 ops | poor under rotation    | yes                      |
| Quadratic Gaussian metric            |   25–50 + reciprocal | smooth                 | after inflation          |
| Quadrupolar support, centerline only |      30–60 + `rsqrt` | smooth but approximate | difficult                |
| Fourier support + 1–2 Newton steps   |               70–180 | very good              | yes with constrained fit |
| Rounded box SAT/GJK                  |               50–150 | piecewise smooth       | yes                      |
| 3-generator rounded zonotope         |              100–250 | good                   | yes                      |
| Six-vertex rounded hull              |              100–300 | good                   | yes                      |
| Two capsules per object              |              120–250 | good                   | yes                      |
| (12\times12) atom pairs              |    roughly 1000–3000 | reference              | exact for chosen atoms   |

The atom-pair estimate assumes approximately 7–20 arithmetic operations per pair once potential evaluation and force accumulation are included.

On SIMD hardware, 144 very regular atom-pair operations may perform better than the scalar count suggests. Nevertheless, the quadratic metric and small zonotope should still have a large margin.

---

# 11. My recommended hierarchy

For molecule-like rigid objects I would use:

## Level 0: bounding sphere

[
|\mathbf d|^2>(R_A+R_B)^2
\quad\Rightarrow\quad
\text{reject}.
]

## Level 0.5: quadrupolar pair metric

Use

[
q=\mathbf d^T C^{-1}\mathbf d.
]

This rejects most orientations missed by the sphere.

It is also sufficient by itself when you only need a smooth approximate repulsive force.

## Level 1-compressed: rounded zonotope

Use 3–6 generators and GJK, or a specialized SAT test in 2D.

This gives an accurate conservative contact body and meaningful normals/contact points.

## Level 2: small convex decomposition

Use 2–4 rounded zonotopes/capsules where the molecular shape has important concavities.

## Optional exact level

Evaluate the 144 atom pairs only when the coarse representation says that contact is close.

This hierarchy is likely better than trying to force everything into a single high-order analytic surface.

---

# 12. Best two concrete models to prototype

## Model A: branch-light soft quadrupolar collision

```cpp
// Rotated shape matrices A and B are cached per object.
Mat2 C = ca*A + cb*B;

float det = C.xx*C.yy - C.xy*C.xy;
float invDet = 1.0f/det;

Vec2 y{
    ( C.yy*d.x - C.xy*d.y)*invDet,
    (-C.xy*d.x + C.xx*d.y)*invDet
};

float q = dot(d,y);

if(q < 1.0f){
    float delta = 1.0f - q;
    float fac   = 2.0f*k*p*pow_int(delta,p-1);
    Vec2 force  = fac*y;
}
```

This is the fastest candidate and closest to a true quadrupolar coarse-graining.

## Model B: rounded three-generator zonotope

```cpp
Vec2 supportCore(
    Vec2 dir,
    const Vec2* generators,
    int n
){
    Vec2 p{0.0f,0.0f};

    for(int i=0; i<n; i++){
        float s = dot(dir,generators[i]);
        p += (s >= 0.0f) ? generators[i] : -generators[i];
    }
    return p;
}
```

Use this support mapping inside 2D GJK. Compare the resulting squared core distance with

```cpp
float R = radiusA + radiusB;
bool collision = dist2 < R*R;
```

For planar benzene, three generators plus a rounding radius should be a particularly natural test case.

---

# Bottom line

Your multipolar idea is sound, but there are two distinct implementations:

[
\boxed{
\text{quadrupolar Gaussian metric}
}
]

when you want the fastest smooth approximate force, and

[
\boxed{
\text{support-function expansion or rounded zonotope}
}
]

when you want a genuine approximation of geometric shape.

For benzene specifically, I expect a **three-generator rounded zonotope** to be more accurate than an ellipse and still much faster than 144 atomic interactions. A quadratic Gaussian contact metric would be even faster—probably only a few tens of operations—but it would represent a smooth orientation-dependent overlap rather than a faithful hexagonal surface.

[1]: https://swt.informatik.uni-freiburg.de/teaching/SS2012/hs-seminar/Resources/papers/Ghosh-Support-Functions?utm_source=chatgpt.com "Support Function Representation of Convex Bodies, Its ..."
[2]: https://pubs.aip.org/aip/jcp/article/56/8/4213/438765/Gaussian-Model-Potentials-for-Molecular?utm_source=chatgpt.com "Gaussian Model Potentials for Molecular Interactions"
[3]: https://www.sciencedirect.com/science/article/abs/pii/S0021999104003948?utm_source=chatgpt.com "II. Applications to ellipses and ellipsoids"
[4]: https://graphics.stanford.edu/courses/cs448b-00-winter/papers/gilbert.pdf?utm_source=chatgpt.com "A fast procedure for computing the distance between ..."
[5]: https://www.ibr.cs.tu-bs.de/courses/ws2122/ag/otherstuff/k-dops.pdf?utm_source=chatgpt.com "Efficient collision detection using bounding volume ..."
[6]: https://web.ist.utl.pt/daniel.s.lopes/papers/Contact%20detection%20between%20quadric%20and%20superquadric%20surfaces%20-%20Multibody%20Syst%20Dyn%202010.pdf?utm_source=chatgpt.com "Contact detection between quadric and superquadric ..."

---

# USER

I vary much like the idea of fourier because
1) it can be easily computed by complex multiplication,
2) it naturally represent shape of common cyclic molecules (benzene, pentacene, etc.

I have however several concenrst
1) in 2D how we can prepresent interacion of two such hexagonal objects? I can very well imagine how to compute interacion of point with such shape (basically evaluate signed distance of some potential with this spherical function), but if I have two of them arbitrarily rotated? Then I should use sin+cos which phase shift encode rotation, right? that is like that slater coster expansion, right?

2) Ok now what if we have full 3D, basically this is like two disks, each having own rotation frame a,b,c, consider a being the pole (flat side of the disk/ring) and a,b being basis of the pl;ane in which we have the hexagona  symmetry (in which we rotate our complex umber). the problem is however than in 3D space they can be arbitrarily rortated wrt each other - they do not have some orientation of pole c  (this is the same problem as with elipsoides), and it is not qite clear how to exper these arbitrary rotations without expanding it on absurly large space of spherical harmonics (slater coster tales for high multipoles). Is there a way how to preserve spares representation ( 6-fold symetri hexagon with only 2 basis function sin(6*phi), cos(6*phi) ) which make it so onvenient  as in 2D case ?

---

# GPT 5.5 hi

Yes. The useful distinction is:

* **Represent the shape in each object’s body-fixed frame.**
* **Rotate the query/contact direction into that frame.**
* Do **not** rotate the sparse harmonic into a global spherical-harmonic coefficient vector.

Then arbitrary 3D orientation costs a few dot products, while the sixfold shape still uses only one complex (m=6) coefficient.

There is, however, an important difference between 2D and 3D:

* In 2D, both objects share the same rotation axis, so their harmonics can be combined into a single complex coefficient.
* In 3D, arbitrarily tilted objects do not share one azimuthal angle, so they cannot generally be collapsed into one complex phase. But each object can still be evaluated sparsely in its own local frame.

---

# 1. Two Fourier-support shapes in 2D

Let the world-space candidate contact normal be

[
\mathbf n(\theta)=(\cos\theta,\sin\theta).
]

Represent it as the unit complex number

[
q=e^{i\theta}.
]

For object (A), let its orientation be

[
p_A=e^{i\alpha_A}.
]

The direction expressed in the local frame of (A) is simply

[
q_A=q,p_A^*=e^{i(\theta-\alpha_A)}.
]

So yes: rotation is encoded by complex phase multiplication exactly as you anticipated.

## General sparse Fourier support

Write the support function of object (A) as

[
h_A(q_A)
========

R_A+
\sum_m
\operatorname{Re}
\left[
c_{m,A}q_A^m
\right].
]

For a sixfold object,

[
h_A(q_A)
========

R_A+
\operatorname{Re}
\left[
c_{6,A}q_A^6
\right].
]

Since

[
q_A^6=q^6(p_A^*)^6,
]

the rotated coefficient in world coordinates is just

[
\tilde c_{6,A}
==============

c_{6,A}e^{-i6\alpha_A}.
]

This is precisely an irreducible representation of the 2D rotation group:

[
c_m \rightarrow c_m e^{-im\alpha}.
]

It is very similar in spirit to Slater–Koster angular algebra, although simpler because (\mathrm{SO}(2)) is commutative and each (m) is represented by one complex number.

---

# 2. Combine two arbitrarily rotated hexagonal shapes

Let

[
\mathbf d=\mathbf c_B-\mathbf c_A
]

be the center displacement.

For a separating normal (\mathbf n), the required center separation along that normal is

[
H(\theta)
=========

h_A(\mathbf n)
+
h_B(-\mathbf n).
]

For the (m=6) term,

[
(-q)^6=q^6,
]

so the opposite direction does not change the sixfold harmonic. Therefore

[
H(\theta)
=========

R_A+R_B+
\operatorname{Re}
\left[
C_6 q^6
\right],
]

with the combined coefficient

[
\boxed{
C_6
===

c_{6,A}e^{-i6\alpha_A}
+
c_{6,B}e^{-i6\alpha_B}
}.
]

This is very elegant: the two rotated hexagonal shapes combine by one complex addition.

The directional gap is

[
g(\theta)
=========

## \mathbf d\cdot\mathbf n(\theta)

H(\theta).
]

Using (d=d_x+i d_y),

[
\mathbf d\cdot\mathbf n
=======================

\operatorname{Re}(d q^*),
]

so

[
\boxed{
g(\theta)
=========

## \operatorname{Re}(d q^*)

## R_A-R_B

\operatorname{Re}(C_6 q^6)
}.
]

Then:

[
\max_\theta g(\theta)>0
]

means there is a separating axis, while

[
\max_\theta g(\theta)\le0
]

means overlap or contact.

Support functions are particularly suitable here because support functions add under Minkowski sums; this is the geometric operation behind two-body collision tests. ([swt.informatik.uni-freiburg.de][1])

---

# 3. Cheaply optimize the contact direction

The derivatives are analytic:

[
g'(\theta)
==========

\mathbf d\cdot\mathbf t
+
6\operatorname{Im}(C_6q^6),
]

where

[
\mathbf t=(-\sin\theta,\cos\theta),
]

and

[
g''(\theta)
===========

-\mathbf d\cdot\mathbf n
+
36\operatorname{Re}(C_6q^6).
]

Thus one Newton step is

[
\theta\leftarrow
\theta-\frac{g'}{g''}.
]

A practical fast approximation is:

1. Start with the center direction
   [
   q_0=\frac{d}{|d|}.
   ]

2. Compute (q_0^2,q_0^4,q_0^6) by three complex multiplications.

3. Perform one clamped Newton correction.

4. Optionally perform a second correction only when penetration is close to zero.

No trigonometric functions are needed after the initial normalization.

## Interesting cancellation

For two identical shapes rotated by (30^\circ),

[
e^{-i6(\alpha+\pi/6)}
=====================

-e^{-i6\alpha}.
]

Therefore their leading (m=6) coefficients cancel:

[
C_6=0.
]

At this truncation level, the Minkowski sum becomes circular. This is physically reasonable: two complementary hexagonal orientations can produce a considerably more isotropic exclusion shape.

---

# 4. Force and torque

Let

[
G=\max_\theta g(\theta)
]

and let (\theta_*) be the maximizing direction.

Define penetration

[
P=\max(0,-G).
]

For example,

[
U(P)=\frac{k}{2}P^2.
]

Because (\theta_*) is already optimized, the envelope theorem means that you do not need derivatives of (\theta_*) when differentiating (G).

The displacement derivative is simply

[
\frac{\partial G}{\partial \mathbf d}
=====================================

\mathbf n_*.
]

Therefore the repulsive force on (B) is

[
\boxed{
\mathbf F_B=kP,\mathbf n_*
}.
]

The orientation derivative follows from the harmonic phase. For object (A),

[
\frac{\partial h_A}{\partial\alpha_A}
=====================================

6\operatorname{Im}
\left[
c_{6,A}e^{i6(\theta_*-\alpha_A)}
\right].
]

So the torque is also extremely cheap.

This gives a proper contact normal and orientation-dependent torque, rather than only a center-to-center radial force.

---

# 5. Moving to 3D without a large spherical-harmonic vector

Take a body-fixed orthonormal frame

[
(\mathbf a,\mathbf b,\mathbf c),
]

where:

* (\mathbf a) is the molecular pole or disk normal,
* (\mathbf b,\mathbf c) span the molecular plane.

For a world-space candidate normal (\mathbf n), compute

[
w=\mathbf n\cdot\mathbf a,
]

[
u=\mathbf n\cdot\mathbf b,
\qquad
v=\mathbf n\cdot\mathbf c.
]

Define the complex in-plane projection

[
z=u+iv.
]

Its magnitude is

[
\rho=|z|=\sqrt{u^2+v^2}
=\sqrt{1-w^2}
]

for unit (\mathbf n).

The local in-plane azimuth is encoded by

[
e^{i\phi}=\frac{z}{\rho}.
]

Therefore the sixfold term is

[
e^{i6\phi}
==========

\left(\frac{z}{\rho}\right)^6.
]

So the entire arbitrary 3D rotation is handled by:

* three dot products,
* one normalization of (z),
* three complex multiplications to form (z^6).

No Euler angles, no `atan2`, and no global spherical-harmonic rotation matrix.

---

# 6. A natural support function for a disk-like molecule

Consider a planar convex shape (K), extruded to half-thickness (t):

[
\mathcal B
==========

K\oplus[-t\mathbf a,t\mathbf a].
]

Its support function is exactly

[
h_{\mathcal B}(\mathbf n)
=========================

h_K(\mathbf n_\parallel)
+
t|\mathbf n\cdot\mathbf a|,
]

where

[
\mathbf n_\parallel
===================

u\mathbf b+v\mathbf c.
]

Because support functions are homogeneous,

[
h_K(\mathbf n_\parallel)
========================

\rho H(\phi),
]

where (H(\phi)) is the 2D support function evaluated for a unit in-plane direction.

Thus

[
\boxed{
h(\mathbf n)
============

t|w|
+
\rho
\left[
R_0+
\operatorname{Re}
\left(
c_6e^{i6\phi}
\right)
\right]
}.
]

Equivalently,

[
\boxed{
h(\mathbf n)
============

t|w|
+
\rho R_0
+
\rho\operatorname{Re}
\left[
c_6
\left(\frac{z}{\rho}\right)^6
\right].
}
]

Add a spherical rounding radius (r_s) as

[
h_{\rm rounded}(\mathbf n)=h(\mathbf n)+r_s.
]

This represents a rounded, finite-thickness, sixfold disk.

---

# 7. Efficient implementation

```cpp
struct Frame3{
    Vec3 a;   // pole
    Vec3 b;   // in-plane axis
    Vec3 c;   // in-plane axis
};

struct Shape6{
    Frame3 frame;
    float radius;
    float halfThickness;

    // Complex sixfold coefficient
    float c6re;
    float c6im;

    float rounding;
};
```

Support-value evaluation:

```cpp
float support6(const Shape6& s, const Vec3& n){
    const float w = dot(n, s.frame.a);
    const float u = dot(n, s.frame.b);
    const float v = dot(n, s.frame.c);

    const float rho2 = u*u + v*v;

    // Pole direction: in-plane support vanishes.
    if(rho2 < 1e-12f){
        return s.halfThickness*fabsf(w) + s.rounding;
    }

    const float invRho = rsqrtf(rho2);
    const float rho    = rho2*invRho;

    // Unit complex in-plane direction q = (u + i*v)/rho
    float qr = u*invRho;
    float qi = v*invRho;

    // q^2
    float q2r = qr*qr - qi*qi;
    float q2i = 2.0f*qr*qi;

    // q^4
    float q4r = q2r*q2r - q2i*q2i;
    float q4i = 2.0f*q2r*q2i;

    // q^6 = q^4*q^2
    float q6r = q4r*q2r - q4i*q2i;
    float q6i = q4r*q2i + q4i*q2r;

    float harmonic =
        s.c6re*q6r
      - s.c6im*q6i;

    return s.halfThickness*fabsf(w)
         + rho*(s.radius + harmonic)
         + s.rounding;
}
```

The `rsqrt` is only used once. Relative to 144 atom-pair interactions, this is extremely cheap.

You can also use a hardware approximate `native_rsqrt` if the contact model is deliberately coarse-grained.

---

# 8. Two arbitrarily tilted 3D objects

For two objects,

[
G(\mathbf n)
============

## \mathbf d\cdot\mathbf n

## h_A(\mathbf n)

h_B(-\mathbf n).
]

The collision criterion is

[
\boxed{
\max_{|\mathbf n|=1}G(\mathbf n)\le0.
}
]

Each body evaluates the same sparse expression, but with its own local quantities:

[
z_A
===

\mathbf n\cdot\mathbf b_A
+
i\mathbf n\cdot\mathbf c_A,
]

[
z_B
===

## -\mathbf n\cdot\mathbf b_B

i\mathbf n\cdot\mathbf c_B.
]

Therefore arbitrary relative tilt causes no representation explosion.

## Pair-relative coordinates

You can reduce the per-iteration work further.

Transform the entire problem into frame (A):

[
\mathbf d_A=R_A^T\mathbf d,
]

[
R_{AB}=R_A^TR_B.
]

Search over (\mathbf n_A). Then

[
\mathbf n_B=-R_{AB}^T\mathbf n_A.
]

Thus during contact iteration:

* object (A) sees (\mathbf n_A) directly,
* object (B) needs one (3\times3) matrix-vector multiplication,
* both evaluate their sparse local (m=6) functions.

The relative rotation matrix can be cached while both rigid-body orientations remain unchanged.

---

# 9. Why a global spherical-harmonic expansion looks large

Your local polynomial

[
(u+iv)^6
]

is essentially the body-frame (m=6), (\ell=6) solid harmonic.

If you rotate it and insist on expressing it in a fixed global spherical-harmonic basis, it mixes into all

[
m=-6,\ldots,+6,
]

meaning 13 coefficients. This mixing is described by a Wigner (D^{(6)}) matrix. Spherical-harmonic vectors are steerable under rotations precisely through these Wigner matrices. ([arXiv][2])

But you do not need to perform this expansion.

Instead of

[
\text{rotate coefficients into world frame},
]

do

[
\boxed{
\text{rotate the query direction into body frame}.
}
]

Mathematically these are equivalent:

[
h_{\rm world}(\mathbf n)
========================

h_{\rm local}(R^T\mathbf n).
]

Computationally the second form preserves the original sparse representation.

That is the direct analogue of evaluating an oriented atomic orbital using local directional cosines rather than constructing a dense global high-rank tensor.

---

# 10. What is lost compared with 2D

In 2D, both objects share the same scalar angle (\theta). Consequently,

[
c_{6,A}e^{-i6\alpha_A}
+
c_{6,B}e^{-i6\alpha_B}
]

can be combined into one effective coefficient.

In arbitrary 3D orientation there is no common azimuth:

[
\phi_A
======

\arg z_A,
\qquad
\phi_B
======

\arg z_B.
]

Usually

[
\phi_A\ne\phi_B+\text{constant}
]

as the candidate normal (\mathbf n) changes.

Therefore there is generally no exact formula like

[
C_6=C_{6,A}+C_{6,B}
]

that eliminates the two local evaluations.

This is not mainly a spherical-harmonic problem. It is a geometrical fact: the two molecular planes are different planes.

Nevertheless, the evaluation remains sparse:

[
O(1)
]

per object per candidate normal.

The additional difficulty is only that the maximizing normal now lies on the two-dimensional sphere rather than on a one-dimensional circle.

---

# 11. Finding the contact normal in 3D

There are three useful accuracy levels.

## Level A: centerline normal

Use

[
\mathbf n_0=\frac{\mathbf d}{|\mathbf d|}.
]

Then compute

[
G_0
===

\mathbf d\cdot\mathbf n_0
-h_A(\mathbf n_0)
-h_B(-\mathbf n_0).
]

This is very fast and may already be sufficient for weak anisotropy.

## Level B: one or two tangent-plane corrections

Optimize (G(\mathbf n)) on the unit sphere.

For a smooth support function,

[
\nabla_{\mathbf n}h(\mathbf n)
]

is the support point of the body in direction (\mathbf n).

Therefore

[
\nabla_{\mathbf n}G
===================

## \mathbf d

\mathbf x_A(\mathbf n)
+
\mathbf x_B(-\mathbf n).
]

Project this gradient into the tangent plane:

[
\mathbf g_T
===========

\left(
I-\mathbf n\mathbf n^T
\right)
\nabla_{\mathbf n}G.
]

Then update

[
\mathbf n'
==========

\operatorname{normalize}
\left(
\mathbf n+\eta\mathbf g_T
\right).
]

One or two iterations may be enough for a coarse collision force.

## Level C: GJK

Provide a support-point mapping and use GJK. GJK works directly with support mappings and the Minkowski difference, avoiding explicit construction of the combined shape. ([graphics.stanford.edu][3])

For your smooth coarse-grained shapes, I suspect a fixed one- or two-step tangent optimization will be faster and more GPU-friendly than a fully converged GJK.

---

# 12. Support point from the Fourier function

For the planar support

[
H(\phi)
=======

R_0+\operatorname{Re}(c_6e^{i6\phi}),
]

its derivative is

[
H'(\phi)
========

-6\operatorname{Im}(c_6e^{i6\phi}).
]

The planar support point is

[
\mathbf x_\parallel
===================

H(\phi)\mathbf e_r
+
H'(\phi)\mathbf e_\phi.
]

With

[
\mathbf e_r
===========

\cos\phi,\mathbf b+\sin\phi,\mathbf c,
]

[
\mathbf e_\phi
==============

-\sin\phi,\mathbf b+\cos\phi,\mathbf c,
]

this becomes

[
\boxed{
\mathbf x_\parallel
===================

\left(H\cos\phi-H'\sin\phi\right)\mathbf b
+
\left(H\sin\phi+H'\cos\phi\right)\mathbf c.
}
]

For the extruded disk,

[
\mathbf x
=========

\mathbf x_\parallel
+
t,\operatorname{sign}(w)\mathbf a
+
r_s\mathbf n.
]

So you get:

* contact point,
* contact normal,
* force lever arm,
* torque,

all from the same complex harmonic evaluation.

---

# 13. Important limitation: one (m=6) support harmonic cannot be strongly hexagonal

There is a subtle but important problem.

For a smooth 2D function (H(\phi)) to be a valid convex support function, it must satisfy

[
H(\phi)+H''(\phi)\ge0.
]

For

[
H(\phi)=R_0+A_6\cos6\phi,
]

we obtain

[
H+H''
=====

R_0-35A_6\cos6\phi.
]

Therefore convexity requires

[
\boxed{
|A_6|\le\frac{R_0}{35}.
}
]

That is only about (2.9%) modulation.

So a single (m=6) support harmonic can describe:

* a nearly circular molecule with weak sixfold corrugation,

but not:

* a sharply faceted regular hexagon.

A true polygon has corners, and its Fourier description requires many harmonics (m=6,12,18,\ldots). Finite spectral approximations of convex bodies are possible, but convexity constraints must be respected. ([HAL][4])

This limitation applies specifically to expanding the **support function**. A radial function

[
r(\phi)=R_0+A_6\cos6\phi
]

can have much stronger modulation while remaining star-shaped, but radial functions do not add cleanly for two-body collision.

---

# 14. A useful hybrid for strongly hexagonal molecules

For a genuinely hexagonal coarse body, use an exact cheap hexagonal support as the base:

[
H_{\rm hex}(\mathbf n)
======================

\sum_{k=1}^{3}
|\mathbf n\cdot\mathbf g_k|.
]

The three in-plane generators are separated by (60^\circ).

Then add a small Fourier correction:

[
H(\phi)
=======

H_{\rm hex}(\phi)
+
\delta H_6(\phi)
+
\delta H_{12}(\phi).
]

This gives:

* a strongly hexagonal base with only three dot products and three absolute values,
* smooth chemically specific corrections from sparse Fourier modes,
* the same body-local 3D evaluation,
* a guaranteed convex base.

You can smooth the absolute value using

[
|x|_\epsilon=\sqrt{x^2+\epsilon^2}
]

or a polynomial approximation if differentiability is needed.

For benzene, I would probably use:

[
\boxed{
\text{rounded 3-generator hexagonal support}
+
\text{small }m=6\text{ correction}.
}
]

For pentacene, the dominant coarse mode is actually (m=2), because pentacene is strongly elongated rather than fivefold symmetric. A suitable expansion would be something like

[
H(\phi)
=======

R_0+
A_2\cos2\phi+
A_4\cos4\phi+
A_6\cos6\phi.
]

---

# Main conclusion

You can preserve the sparse two-component sixfold representation in 3D:

[
\boxed{
z_A
===

\mathbf n\cdot\mathbf b_A
+i\mathbf n\cdot\mathbf c_A,
\qquad
h_{6,A}\propto
\operatorname{Re}
\left[
c_{6,A}
\left(\frac{z_A}{|z_A|}\right)^6
\right].
}
]

Arbitrary rigid rotation is handled by rotating the **direction into the body frame**, not by rotating the harmonic coefficients into a global basis.

For two arbitrarily tilted objects, you cannot generally combine both shapes into one complex coefficient as in 2D, because they have different poles and different local azimuths. But evaluating both local sparse functions remains cheap, and the contact-normal search can be limited to one or two fixed tangent-plane iterations.

The likely practical design is:

[
\boxed{
h(\mathbf n)
============

t|\mathbf n\cdot\mathbf a|
+
\rho H_{\rm inplane}(\phi)
+
r_s
}
]

with (H_{\rm inplane}) represented by either:

* a few Fourier modes for smooth molecular shapes, or
* a three-generator hexagonal support plus small Fourier corrections for strongly faceted cyclic molecules.

[1]: https://swt.informatik.uni-freiburg.de/teaching/SS2012/hs-seminar/Resources/papers/Ghosh-Support-Functions?utm_source=chatgpt.com "Support Function Representation of Convex Bodies, Its ..."
[2]: https://arxiv.org/html/2409.07327v1?utm_source=chatgpt.com "Current Symmetry Group Equivariant Convolution ..."
[3]: https://graphics.stanford.edu/courses/cs448b-00-winter/papers/gilbert.pdf?utm_source=chatgpt.com "A fast procedure for computing the distance between ..."
[4]: https://hal.science/hal-03170318v1/file/supportf_applications.pdf?utm_source=chatgpt.com "Parametric Shape Optimization using the Support Function"

---

# USER

I'm thinking about slightly different but closely related formulation. Assume Pauli repulsion potential is proportional to overlap of electron density of the two molecules, and the density of molece is sum of atomic densities. 

I want some fast approximate way how to express this potential for two arbitrarily rotated molecules. Let's think in 2D first it is much easier. We employ the fact that the molecule is n-fold symmetric so we apply the symmetry operations so our density at any point can be expressed by splitting the space to n angular segments (like pizza pisces) and in each segment we evaluate only one site.
This transform can be evaluated very fast by n-power of unitary quaternion, right? There vil be some derivative discontinuities at the boundary angles (seam between two pizza pieces) but I don't care. But maybe the Fourier expansion is just as convenient and fast. By combining single n-fold angular functions and a radial function I can create n blobs symmetrically around center Wich is all I care about. Radially it can decay with some exponential or gaussian function. 

Now I care not about the density of one molecule but about overlap of two molecules.

In principle this can be just slater coster table with high enough degree but maybe there is some more efficient (faster) representation (formulation)

How can we expres efficient formula for overlap of the two molecules depending on any translation and rotation.

For translation I often use convolution trick using FFT but that is not really fast. But I can use it as reference for checking if the formula is correct. Can you derive these formulas for me?

---

# Chat GPT 5.5

## 1. The essential geometry

Let molecule (A) have center (\mathbf R_A) and orientation (\alpha), and molecule (B) have center (\mathbf R_B) and orientation (\beta). Define

[
\mathbf R=\mathbf R_B-\mathbf R_A
=R(\cos\phi,\sin\phi).
]

Because a simultaneous rotation of everything changes nothing, the interaction can depend only on three scalar variables:

[
R,\qquad
\delta_A=\phi-\alpha,\qquad
\delta_B=\phi-\beta.
]

This is important: at finite separation, the overlap cannot generally be a function only of (R) and the relative rotation (\beta-\alpha). Each molecule also has an orientation relative to the line joining the centers.

Assume

[
E_{\mathrm P}
=============

\kappa\int_{\mathbb R^2}
\rho_A(\mathbf x)\rho_B(\mathbf x),d^2x.
]

Putting (A) at the origin,

[
E(R,\alpha,\beta,\phi)
======================

\kappa\int
\rho_A^0!\left(\mathcal R_{-\alpha}\mathbf x\right)
\rho_B^0!\left(\mathcal R_{-\beta}(\mathbf x-\mathbf R)\right)
d^2x.
]

This is exactly a rotated cross-correlation.

---

# 2. First observation: atomic Gaussian sums already give an exact fast formula

Suppose the molecular density is a sum of isotropic atomic densities,

[
\rho_A(\mathbf x)
=================

\sum_a f_a(|\mathbf x-\mathbf a_a|),
\qquad
\rho_B(\mathbf x)
=================

\sum_b g_b(|\mathbf x-\mathbf b_b|).
]

Then linearity gives

[
E
=

\kappa\sum_{ab}
h_{ab}
\left(
\left|
\mathbf R+\mathcal R_\beta\mathbf b_b
-\mathcal R_\alpha\mathbf a_a
\right|
\right),
]

where the radial pair kernel is the convolution

[
h_{ab}(d)
=========

\int_{\mathbb R^2}
f_a(|\mathbf x|)
g_b(|\mathbf x-\mathbf d|)
d^2x.
]

So if the atomic densities are isotropic, arbitrary molecular rotation introduces no difficult integral. It only rotates the atomic center positions.

For normalized 2D Gaussian atomic densities

[
f_a(r)=q_a\frac{\zeta_a}{\pi}e^{-\zeta_a r^2},
]

the pair overlap is analytic:

[
\boxed{
h_{ab}(d)
=========

q_aq_b
\frac{\zeta_a\zeta_b}
{\pi(\zeta_a+\zeta_b)}
\exp\left[
-\frac{\zeta_a\zeta_b}{\zeta_a+\zeta_b}d^2
\right].
}
]

Therefore

[
\boxed{
E
=

\kappa\sum_{ab}
q_aq_b
\frac{\zeta_a\zeta_b}
{\pi(\zeta_a+\zeta_b)}
e^{-\mu_{ab}d_{ab}^2},
\qquad
\mu_{ab}=\frac{\zeta_a\zeta_b}{\zeta_a+\zeta_b}.
}
]

For a pair of six-site rings this means only (36) Gaussian evaluations. That is a useful exact reference, and for small molecules it may already be competitive with a more elaborate harmonic representation.

For Slater-like exponential densities, I would fit each radial density by a short Gaussian sum,

[
f_a(r)\simeq\sum_p c_{ap}e^{-\zeta_{ap}r^2},
]

because then all overlaps remain analytic.

---

# 3. The (N)-fold angular representation

In two dimensions, quaternions are unnecessary. Use the unit complex number

[
z=\frac{x+iy}{r}=e^{i\phi}.
]

In the body frame of a molecule oriented by (\alpha),

[
z_{\mathrm{body}}
=================

# e^{-i\alpha}z

e^{i(\phi-\alpha)}.
]

Then

[
z_{\mathrm{body}}^N
===================

e^{iN(\phi-\alpha)}.
]

This maps all (N) angular sectors onto the same phase.

There are two distinct constructions:

1. Compute (\arg(z^N)/N) and map explicitly into one pizza slice. This has a seam because `arg` jumps.
2. Make the density a smooth function of (z^N), or equivalently of
   (\cos N\phi) and (\sin N\phi). This has no seam.

The natural harmonic representation is

[
\boxed{
\rho(r,\phi)
============

\sum_{p=-P}^{P}
a_{pN}(r)e^{ipN(\phi-\alpha)}.
}
]

For a real mirror-symmetric molecule this becomes

[
\rho(r,\phi)
============

a_0(r)
+
2\sum_{p=1}^{P}
a_{pN}(r)\cos[pN(\phi-\alpha)].
]

Exact (N)-fold symmetry eliminates every angular harmonic that is not a multiple of (N). It does not imply that only the first harmonic (N) is sufficient; sharp individual blobs may require (2N,3N,\ldots).

---

## Exact example: a ring of Gaussian atoms

Take (N) identical Gaussian atoms at radius (a),

[
\mathbf a_j
===========

a\begin{pmatrix}
\cos(\theta_0+2\pi j/N)\
\sin(\theta_0+2\pi j/N)
\end{pmatrix}.
]

For one atomic Gaussian

[
g(\mathbf x)=q\frac{\zeta}{\pi}e^{-\zeta |\mathbf x|^2},
]

the complete ring density is

[
\rho(r,\phi)
============

q\frac{\zeta}{\pi}
e^{-\zeta(r^2+a^2)}
\sum_{j=0}^{N-1}
e^{2\zeta ar\cos(\phi-\theta_j)}.
]

Using

[
e^{s\cos\theta}
===============

\sum_{m=-\infty}^{\infty}
I_m(s)e^{im\theta},
]

and summing over the (N) roots of unity gives

[
\boxed{
\rho(r,\phi)
============

Nq\frac{\zeta}{\pi}
e^{-\zeta(r^2+a^2)}
\sum_{p=-\infty}^{\infty}
I_{pN}(2\zeta ar)
e^{ipN(\phi-\theta_0)}.
}
]

Equivalently,

[
\rho
====

Nq\frac{\zeta}{\pi}
e^{-\zeta(r^2+a^2)}
\left[
I_0(2\zeta ar)
+
2\sum_{p=1}^{\infty}
I_{pN}(2\zeta ar)
\cos[pN(\phi-\theta_0)]
\right].
]

So your proposed Fourier representation is not merely qualitative: for a ring of Gaussian atomic densities it is exact. The radial and angular dependences are coupled through (I_{pN}(2\zeta ar)).

---

# 4. Exact Fourier–Bessel overlap formula

Define the Fourier transform by

[
\widehat\rho(\mathbf k)
=======================

\int_{\mathbb R^2}
\rho(\mathbf x)e^{-i\mathbf k\cdot\mathbf x},d^2x.
]

In polar reciprocal coordinates (\mathbf k=(k,\psi)), expand each molecular density as

[
\widehat\rho_A(k,\psi)
======================

\sum_m A_m(k)e^{im\psi},
]

[
\widehat\rho_B(k,\psi)
======================

\sum_n B_n(k)e^{in\psi}.
]

Rotation acts diagonally:

[
\widehat\rho_A^{(\alpha)}(k,\psi)
=================================

\sum_m A_m(k)e^{im\psi}e^{-im\alpha}.
]

Translation by (\mathbf R) multiplies the Fourier transform by

[
e^{-i\mathbf k\cdot\mathbf R}.
]

This diagonal action of rotations and phase representation of translations is precisely why Fourier–Bessel bases are called steerable. 

Using Parseval,

[
E
=

\frac{\kappa}{(2\pi)^2}
\int
\widehat\rho_A^{(\alpha)}(\mathbf k)^*
\widehat\rho_B^{(\beta)}(\mathbf k)
e^{-i\mathbf k\cdot\mathbf R}
,d^2k.
]

Now use

[
\int_0^{2\pi}
e^{iq\psi}
e^{-ikR\cos(\psi-\phi)}
d\psi
=====

2\pi(-i)^qJ_q(kR)e^{iq\phi}.
]

This is the Bessel-function translation/addition structure underlying Graf-type addition theorems. ([dlmf.nist.gov][1])

Substitution gives

[
\boxed{
E(R,\delta_A,\delta_B)
======================

\sum_{m,n}
C_{mn}(R)
e^{-im\delta_A}
e^{in\delta_B},
}
]

where

[
\boxed{
C_{mn}(R)
=========

\frac{\kappa}{2\pi}
(-i)^{n-m}
\int_0^\infty
k,dk,
A_m(k)^*B_n(k)
J_{n-m}(kR).
}
]

This is the exact result.

It has a very Slater–Koster-like interpretation:

* (m) is an angular channel of molecule (A);
* (n) is an angular channel of molecule (B);
* (C_{mn}(R)) is a one-dimensional radial table;
* the angular factors are cheap complex powers.

Translation does not require an FFT at runtime. Its entire effect is hidden in the precomputed radial Bessel integral.

For (N_A)- and (N_B)-fold molecules,

[
m=pN_A,\qquad n=qN_B.
]

Thus

[
\boxed{
E
=

\sum_{p,q}
C_{pN_A,qN_B}(R)
e^{-ipN_A\delta_A}
e^{iqN_B\delta_B}.
}
]

Only symmetry-allowed channels exist.

---

# 5. Very compact first-harmonic formula

Suppose both molecules have the same (N)-fold symmetry and retain only

[
m,n\in{-N,0,+N}.
]

For mirror-symmetric nonchiral densities, the complete real interaction reduces to only five radial functions:

[
\boxed{
\begin{aligned}
E(R,\alpha,\beta,\phi)
\simeq {}&
V_0(R)
\
&+V_A(R)\cos[N(\phi-\alpha)]
\
&+V_B(R)\cos[N(\phi-\beta)]
\
&+V_{\mathrm{rel}}(R)\cos[N(\beta-\alpha)]
\
&+V_{\mathrm{sum}}(R)
\cos[N(2\phi-\alpha-\beta)].
\end{aligned}
}
]

The terms have clear meanings:

[
V_0:
\quad \text{isotropic overlap},
]

[
V_A,V_B:
\quad \text{orientation of each body relative to the line of centers},
]

[
V_{\mathrm{rel}}:
\quad \text{direct relative-orientation locking},
]

[
V_{\mathrm{sum}}:
\quad \text{coupled alignment of both bodies with the line of centers}.
]

For identical molecules,

[
V_A(R)=V_B(R).
]

This is probably the smallest useful representation of the interaction you are describing.

---

## Complex-multiplication implementation

Represent orientations by unit complex numbers

[
q_A=e^{i\alpha},
\qquad
q_B=e^{i\beta},
\qquad
p=\frac{R_x+iR_y}{R}=e^{i\phi}.
]

Precompute or update

[
a=q_A^N,\qquad b=q_B^N,
\qquad s=p^N.
]

Then

[
u_A=s,a^*=e^{iN(\phi-\alpha)},
]

[
u_B=s,b^*=e^{iN(\phi-\beta)}.
]

The two coupled quantities are

[
u_A^*u_B=e^{iN(\alpha-\beta)},
]

[
u_Au_B=e^{iN(2\phi-\alpha-\beta)}.
]

Therefore

[
\boxed{
E
=

V_0
+V_A\operatorname{Re}u_A
+V_B\operatorname{Re}u_B
+V_{\mathrm{rel}}\operatorname{Re}(u_A^*u_B)
+V_{\mathrm{sum}}\operatorname{Re}(u_Au_B).
}
]

C-like pseudocode:

```c
float R = sqrt(Rx*Rx + Ry*Ry);
float2 p = (float2)(Rx/R, Ry/R);  // exp(i phi)

// qAN = exp(i N alpha), qBN = exp(i N beta)
// These can be cached for rigid bodies.
float2 s  = complex_pow_N(p, N);

float2 uA = cmul(s, cconj(qAN));
float2 uB = cmul(s, cconj(qBN));

float E =
      V0(R)
    + VA(R)   * uA.x
    + VB(R)   * uB.x
    + Vrel(R) * cmul(cconj(uA), uB).x
    + Vsum(R) * cmul(uA, uB).x;
```

For (N=6),

```c
z2 = z*z;
z3 = z2*z;
z6 = z3*z3;
```

so the sixth power costs only three complex multiplications. There is no `atan2`, no sine or cosine evaluation, and no seam.

For a chiral molecule or a body axis not chosen along a mirror plane, the radial coefficients may be complex. The same implementation works using

[
E=\operatorname{Re}\sum_j V_j(R)z_j
]

instead of assuming purely real (V_j).

---

# 6. Behaviour near (R=0)

Because

[
J_q(x)\sim \frac{1}{|q|!}\left(\frac{x}{2}\right)^{|q|},
]

the radial coefficients obey

[
C_{mn}(R)\sim R^{|n-m|}.
]

Therefore:

* (C_{mm}(0)) can remain finite;
* (C_{mn}(0)=0) for (m\ne n);
* terms depending on the line direction vanish automatically when the center-to-center direction becomes undefined.

For the first-harmonic model,

[
V_A,V_B\sim R^N,
]

[
V_{\mathrm{sum}}\sim R^{2N},
]

while

[
V_0,;V_{\mathrm{rel}}
]

can remain finite at (R=0).

This is useful numerically. One can tabulate the regularized quantities

[
\frac{V_A(R)}{R^N},
\qquad
\frac{V_{\mathrm{sum}}(R)}{R^{2N}}
]

near the origin.

---

# 7. Computing the radial tables

## Route A: from Fourier–Bessel coefficients

Compute a reference density on a Cartesian grid, FFT it once, and interpolate the Fourier transform onto polar rings. Then

[
A_m(k_l)
\simeq
\frac{1}{N_\psi}
\sum_{j=0}^{N_\psi-1}
\widehat\rho_A(k_l,\psi_j)e^{-im\psi_j}.
]

Similarly for (B_n(k_l)).

For every required radial distance (R_s),

[
\boxed{
C_{mn}(R_s)
\simeq
\frac{\kappa}{2\pi}
(-i)^{n-m}
\sum_l
w_l k_l
A_m(k_l)^*B_n(k_l)
J_{n-m}(k_lR_s).
}
]

This precomputation may be relatively expensive, but it is done only once per pair of molecular types.

At runtime, use cubic Hermite or spline interpolation in (R). Derive both the energy and force from the same interpolating spline, rather than interpolating precomputed forces independently.

---

## Route B: fit the angular tables from brute-force overlaps

A simpler implementation is to calculate reference overlap values

[
E(R,\delta_A,\delta_B)
]

on a small uniform angular grid and perform a two-dimensional angular DFT:

[
C_{pq}(R)
=========

\frac{1}{N_\theta^2}
\sum_{j,l}
E(R,\delta_j,\delta_l)
e^{ipN\delta_j}
e^{-iqN\delta_l}.
]

Because the fundamental angular domain is only

[
0\leq \delta <\frac{2\pi}{N},
]

very few samples are needed. For (P=1), three or four samples per reduced angular period are theoretically enough, although using (8) or (16) is safer when constructing the reference.

The reference overlap itself can be computed either from the atomic Gaussian pair formula or from

[
E(\mathbf R)
============

\mathcal F^{-1}
\left[
\widehat\rho_A(\mathbf k)^*
\widehat\rho_B(\mathbf k)
\right](\mathbf R).
]

This is an excellent validation route.

---

# 8. Forces and torques are analytic

Write one term as

[
Z_{mn}
======

C_{mn}(R)
e^{-im\delta_A}
e^{in\delta_B}.
]

Then

[
E=\operatorname{Re}\sum_{mn}Z_{mn}.
]

The radial force is

[
\boxed{
F_R
===

-\operatorname{Re}
\sum_{mn}
C_{mn}'(R)
e^{-im\delta_A}
e^{in\delta_B}.
}
]

The tangential force is

[
\boxed{
F_\phi
======

\frac{1}{R}
\sum_{mn}
(n-m)\operatorname{Im}Z_{mn}.
}
]

The molecular torques are

[
\boxed{
\tau_A
======

\sum_{mn}
m,\operatorname{Im}Z_{mn},
}
]

[
\boxed{
\tau_B
======

-\sum_{mn}
n,\operatorname{Im}Z_{mn}.
}
]

These automatically satisfy angular-momentum conservation:

[
\tau_A+\tau_B+RF_\phi=0.
]

So the harmonic representation is not only fast for energies; it also gives consistent forces and torques without finite differences.

---

# 9. How many harmonics are needed?

For a hexagonal molecule, the allowed channels are

[
m=0,\pm6,\pm12,\pm18,\ldots
]

A practical hierarchy is:

[
P=0:
\quad \text{purely radial},
]

[
P=1:
\quad m=0,\pm6,
]

[
P=2:
\quad m=0,\pm6,\pm12.
]

The overlap is smoother than either original density because in reciprocal space it contains the product

[
A_m(k)^*B_n(k),
]

which suppresses high spatial frequencies. Therefore the interaction often requires fewer harmonics than are required to reconstruct the density itself.

For benzene-like shapes, I would start with (P=1), meaning the five radial functions above, and add the (m=\pm12) channels only if comparisons against the Gaussian-pair or FFT reference show significant error.

If many harmonics are retained, the matrix (C_{mn}(R)) can itself be compressed by a low-rank factorization. Low-rank factorizations of Fourier–Bessel translation kernels are an established acceleration strategy for rotated and translated correlations. ([arXiv][2])

---

# Recommended formulation

For your application I would use this pipeline:

[
\text{atomic or grid density}
\longrightarrow
\text{reference overlaps}
\longrightarrow
C_{pq}(R)
\longrightarrow
\text{complex-power runtime evaluation}.
]

For a sixfold symmetric pair, begin with

[
\boxed{
E=
V_0
+V_A\cos6\delta_A
+V_B\cos6\delta_B
+V_{\mathrm{rel}}\cos6(\beta-\alpha)
+V_{\mathrm{sum}}\cos6(2\phi-\alpha-\beta).
}
]

That is essentially a symmetry-adapted 2D Slater–Koster model, but it needs only five one-dimensional radial tables and a handful of complex multiplications per molecular pair.

[1]: https://dlmf.nist.gov/10.23 "DLMF: §10.23 Sums ‣ Bessel and Hankel Functions ‣ Chapter 10 Bessel Functions"
[2]: https://arxiv.org/abs/1905.12317 "[1905.12317] Factorization of the translation kernel for fast rigid image alignment"


---

# USER

## 1. The essential geometry

Let molecule (A) have center (\mathbf R_A) and orientation (\alpha), and molecule (B) have center (\mathbf R_B) and orientation (\beta). Define

[
\mathbf R=\mathbf R_B-\mathbf R_A
=R(\cos\phi,\sin\phi).
]

Because a simultaneous rotation of everything changes nothing, the interaction can depend only on three scalar variables:

[
R,\qquad
\delta_A=\phi-\alpha,\qquad
\delta_B=\phi-\beta.
]

This is important: at finite separation, the overlap cannot generally be a function only of (R) and the relative rotation (\beta-\alpha). Each molecule also has an orientation relative to the line joining the centers.

Assume

[
E_{\mathrm P}
=============

\kappa\int_{\mathbb R^2}
\rho_A(\mathbf x)\rho_B(\mathbf x),d^2x.
]

Putting (A) at the origin,

[
E(R,\alpha,\beta,\phi)
======================

\kappa\int
\rho_A^0!\left(\mathcal R_{-\alpha}\mathbf x\right)
\rho_B^0!\left(\mathcal R_{-\beta}(\mathbf x-\mathbf R)\right)
d^2x.
]

This is exactly a rotated cross-correlation.

---

# 2. First observation: atomic Gaussian sums already give an exact fast formula

Suppose the molecular density is a sum of isotropic atomic densities,

[
\rho_A(\mathbf x)
=================

\sum_a f_a(|\mathbf x-\mathbf a_a|),
\qquad
\rho_B(\mathbf x)
=================

\sum_b g_b(|\mathbf x-\mathbf b_b|).
]

Then linearity gives

[
E
=

\kappa\sum_{ab}
h_{ab}
\left(
\left|
\mathbf R+\mathcal R_\beta\mathbf b_b
-\mathcal R_\alpha\mathbf a_a
\right|
\right),
]

where the radial pair kernel is the convolution

[
h_{ab}(d)
=========

\int_{\mathbb R^2}
f_a(|\mathbf x|)
g_b(|\mathbf x-\mathbf d|)
d^2x.
]

So if the atomic densities are isotropic, arbitrary molecular rotation introduces no difficult integral. It only rotates the atomic center positions.

For normalized 2D Gaussian atomic densities

[
f_a(r)=q_a\frac{\zeta_a}{\pi}e^{-\zeta_a r^2},
]

the pair overlap is analytic:

[
\boxed{
h_{ab}(d)
=========

q_aq_b
\frac{\zeta_a\zeta_b}
{\pi(\zeta_a+\zeta_b)}
\exp\left[
-\frac{\zeta_a\zeta_b}{\zeta_a+\zeta_b}d^2
\right].
}
]

Therefore

[
\boxed{
E
=

\kappa\sum_{ab}
q_aq_b
\frac{\zeta_a\zeta_b}
{\pi(\zeta_a+\zeta_b)}
e^{-\mu_{ab}d_{ab}^2},
\qquad
\mu_{ab}=\frac{\zeta_a\zeta_b}{\zeta_a+\zeta_b}.
}
]

For a pair of six-site rings this means only (36) Gaussian evaluations. That is a useful exact reference, and for small molecules it may already be competitive with a more elaborate harmonic representation.

For Slater-like exponential densities, I would fit each radial density by a short Gaussian sum,

[
f_a(r)\simeq\sum_p c_{ap}e^{-\zeta_{ap}r^2},
]

because then all overlaps remain analytic.

---

# 3. The (N)-fold angular representation

In two dimensions, quaternions are unnecessary. Use the unit complex number

[
z=\frac{x+iy}{r}=e^{i\phi}.
]

In the body frame of a molecule oriented by (\alpha),

[
z_{\mathrm{body}}
=================

# e^{-i\alpha}z

e^{i(\phi-\alpha)}.
]

Then

[
z_{\mathrm{body}}^N
===================

e^{iN(\phi-\alpha)}.
]

This maps all (N) angular sectors onto the same phase.

There are two distinct constructions:

1. Compute (\arg(z^N)/N) and map explicitly into one pizza slice. This has a seam because `arg` jumps.
2. Make the density a smooth function of (z^N), or equivalently of
   (\cos N\phi) and (\sin N\phi). This has no seam.

The natural harmonic representation is

[
\boxed{
\rho(r,\phi)
============

\sum_{p=-P}^{P}
a_{pN}(r)e^{ipN(\phi-\alpha)}.
}
]

For a real mirror-symmetric molecule this becomes

[
\rho(r,\phi)
============

a_0(r)
+
2\sum_{p=1}^{P}
a_{pN}(r)\cos[pN(\phi-\alpha)].
]

Exact (N)-fold symmetry eliminates every angular harmonic that is not a multiple of (N). It does not imply that only the first harmonic (N) is sufficient; sharp individual blobs may require (2N,3N,\ldots).

---

## Exact example: a ring of Gaussian atoms

Take (N) identical Gaussian atoms at radius (a),

[
\mathbf a_j
===========

a\begin{pmatrix}
\cos(\theta_0+2\pi j/N)\
\sin(\theta_0+2\pi j/N)
\end{pmatrix}.
]

For one atomic Gaussian

[
g(\mathbf x)=q\frac{\zeta}{\pi}e^{-\zeta |\mathbf x|^2},
]

the complete ring density is

[
\rho(r,\phi)
============

q\frac{\zeta}{\pi}
e^{-\zeta(r^2+a^2)}
\sum_{j=0}^{N-1}
e^{2\zeta ar\cos(\phi-\theta_j)}.
]

Using

[
e^{s\cos\theta}
===============

\sum_{m=-\infty}^{\infty}
I_m(s)e^{im\theta},
]

and summing over the (N) roots of unity gives

[
\boxed{
\rho(r,\phi)
============

Nq\frac{\zeta}{\pi}
e^{-\zeta(r^2+a^2)}
\sum_{p=-\infty}^{\infty}
I_{pN}(2\zeta ar)
e^{ipN(\phi-\theta_0)}.
}
]

Equivalently,

[
\rho
====

Nq\frac{\zeta}{\pi}
e^{-\zeta(r^2+a^2)}
\left[
I_0(2\zeta ar)
+
2\sum_{p=1}^{\infty}
I_{pN}(2\zeta ar)
\cos[pN(\phi-\theta_0)]
\right].
]

So your proposed Fourier representation is not merely qualitative: for a ring of Gaussian atomic densities it is exact. The radial and angular dependences are coupled through (I_{pN}(2\zeta ar)).

---

# 4. Exact Fourier–Bessel overlap formula

Define the Fourier transform by

[
\widehat\rho(\mathbf k)
=======================

\int_{\mathbb R^2}
\rho(\mathbf x)e^{-i\mathbf k\cdot\mathbf x},d^2x.
]

In polar reciprocal coordinates (\mathbf k=(k,\psi)), expand each molecular density as

[
\widehat\rho_A(k,\psi)
======================

\sum_m A_m(k)e^{im\psi},
]

[
\widehat\rho_B(k,\psi)
======================

\sum_n B_n(k)e^{in\psi}.
]

Rotation acts diagonally:

[
\widehat\rho_A^{(\alpha)}(k,\psi)
=================================

\sum_m A_m(k)e^{im\psi}e^{-im\alpha}.
]

Translation by (\mathbf R) multiplies the Fourier transform by

[
e^{-i\mathbf k\cdot\mathbf R}.
]

This diagonal action of rotations and phase representation of translations is precisely why Fourier–Bessel bases are called steerable. 

Using Parseval,

[
E
=

\frac{\kappa}{(2\pi)^2}
\int
\widehat\rho_A^{(\alpha)}(\mathbf k)^*
\widehat\rho_B^{(\beta)}(\mathbf k)
e^{-i\mathbf k\cdot\mathbf R}
,d^2k.
]

Now use

[
\int_0^{2\pi}
e^{iq\psi}
e^{-ikR\cos(\psi-\phi)}
d\psi
=====

2\pi(-i)^qJ_q(kR)e^{iq\phi}.
]

This is the Bessel-function translation/addition structure underlying Graf-type addition theorems. ([dlmf.nist.gov][1])

Substitution gives

[
\boxed{
E(R,\delta_A,\delta_B)
======================

\sum_{m,n}
C_{mn}(R)
e^{-im\delta_A}
e^{in\delta_B},
}
]

where

[
\boxed{
C_{mn}(R)
=========

\frac{\kappa}{2\pi}
(-i)^{n-m}
\int_0^\infty
k,dk,
A_m(k)^*B_n(k)
J_{n-m}(kR).
}
]

This is the exact result.

It has a very Slater–Koster-like interpretation:

* (m) is an angular channel of molecule (A);
* (n) is an angular channel of molecule (B);
* (C_{mn}(R)) is a one-dimensional radial table;
* the angular factors are cheap complex powers.

Translation does not require an FFT at runtime. Its entire effect is hidden in the precomputed radial Bessel integral.

For (N_A)- and (N_B)-fold molecules,

[
m=pN_A,\qquad n=qN_B.
]

Thus

[
\boxed{
E
=

\sum_{p,q}
C_{pN_A,qN_B}(R)
e^{-ipN_A\delta_A}
e^{iqN_B\delta_B}.
}
]

Only symmetry-allowed channels exist.

---

# 5. Very compact first-harmonic formula

Suppose both molecules have the same (N)-fold symmetry and retain only

[
m,n\in{-N,0,+N}.
]

For mirror-symmetric nonchiral densities, the complete real interaction reduces to only five radial functions:

[
\boxed{
\begin{aligned}
E(R,\alpha,\beta,\phi)
\simeq {}&
V_0(R)
\
&+V_A(R)\cos[N(\phi-\alpha)]
\
&+V_B(R)\cos[N(\phi-\beta)]
\
&+V_{\mathrm{rel}}(R)\cos[N(\beta-\alpha)]
\
&+V_{\mathrm{sum}}(R)
\cos[N(2\phi-\alpha-\beta)].
\end{aligned}
}
]

The terms have clear meanings:

[
V_0:
\quad \text{isotropic overlap},
]

[
V_A,V_B:
\quad \text{orientation of each body relative to the line of centers},
]

[
V_{\mathrm{rel}}:
\quad \text{direct relative-orientation locking},
]

[
V_{\mathrm{sum}}:
\quad \text{coupled alignment of both bodies with the line of centers}.
]

For identical molecules,

[
V_A(R)=V_B(R).
]

This is probably the smallest useful representation of the interaction you are describing.

---

## Complex-multiplication implementation

Represent orientations by unit complex numbers

[
q_A=e^{i\alpha},
\qquad
q_B=e^{i\beta},
\qquad
p=\frac{R_x+iR_y}{R}=e^{i\phi}.
]

Precompute or update

[
a=q_A^N,\qquad b=q_B^N,
\qquad s=p^N.
]

Then

[
u_A=s,a^*=e^{iN(\phi-\alpha)},
]

[
u_B=s,b^*=e^{iN(\phi-\beta)}.
]

The two coupled quantities are

[
u_A^*u_B=e^{iN(\alpha-\beta)},
]

[
u_Au_B=e^{iN(2\phi-\alpha-\beta)}.
]

Therefore

[
\boxed{
E
=

V_0
+V_A\operatorname{Re}u_A
+V_B\operatorname{Re}u_B
+V_{\mathrm{rel}}\operatorname{Re}(u_A^*u_B)
+V_{\mathrm{sum}}\operatorname{Re}(u_Au_B).
}
]

C-like pseudocode:

```c
float R = sqrt(Rx*Rx + Ry*Ry);
float2 p = (float2)(Rx/R, Ry/R);  // exp(i phi)

// qAN = exp(i N alpha), qBN = exp(i N beta)
// These can be cached for rigid bodies.
float2 s  = complex_pow_N(p, N);

float2 uA = cmul(s, cconj(qAN));
float2 uB = cmul(s, cconj(qBN));

float E =
      V0(R)
    + VA(R)   * uA.x
    + VB(R)   * uB.x
    + Vrel(R) * cmul(cconj(uA), uB).x
    + Vsum(R) * cmul(uA, uB).x;
```

For (N=6),

```c
z2 = z*z;
z3 = z2*z;
z6 = z3*z3;
```

so the sixth power costs only three complex multiplications. There is no `atan2`, no sine or cosine evaluation, and no seam.

For a chiral molecule or a body axis not chosen along a mirror plane, the radial coefficients may be complex. The same implementation works using

[
E=\operatorname{Re}\sum_j V_j(R)z_j
]

instead of assuming purely real (V_j).

---

# 6. Behaviour near (R=0)

Because

[
J_q(x)\sim \frac{1}{|q|!}\left(\frac{x}{2}\right)^{|q|},
]

the radial coefficients obey

[
C_{mn}(R)\sim R^{|n-m|}.
]

Therefore:

* (C_{mm}(0)) can remain finite;
* (C_{mn}(0)=0) for (m\ne n);
* terms depending on the line direction vanish automatically when the center-to-center direction becomes undefined.

For the first-harmonic model,

[
V_A,V_B\sim R^N,
]

[
V_{\mathrm{sum}}\sim R^{2N},
]

while

[
V_0,;V_{\mathrm{rel}}
]

can remain finite at (R=0).

This is useful numerically. One can tabulate the regularized quantities

[
\frac{V_A(R)}{R^N},
\qquad
\frac{V_{\mathrm{sum}}(R)}{R^{2N}}
]

near the origin.

---

# 7. Computing the radial tables

## Route A: from Fourier–Bessel coefficients

Compute a reference density on a Cartesian grid, FFT it once, and interpolate the Fourier transform onto polar rings. Then

[
A_m(k_l)
\simeq
\frac{1}{N_\psi}
\sum_{j=0}^{N_\psi-1}
\widehat\rho_A(k_l,\psi_j)e^{-im\psi_j}.
]

Similarly for (B_n(k_l)).

For every required radial distance (R_s),

[
\boxed{
C_{mn}(R_s)
\simeq
\frac{\kappa}{2\pi}
(-i)^{n-m}
\sum_l
w_l k_l
A_m(k_l)^*B_n(k_l)
J_{n-m}(k_lR_s).
}
]

This precomputation may be relatively expensive, but it is done only once per pair of molecular types.

At runtime, use cubic Hermite or spline interpolation in (R). Derive both the energy and force from the same interpolating spline, rather than interpolating precomputed forces independently.

---

## Route B: fit the angular tables from brute-force overlaps

A simpler implementation is to calculate reference overlap values

[
E(R,\delta_A,\delta_B)
]

on a small uniform angular grid and perform a two-dimensional angular DFT:

[
C_{pq}(R)
=========

\frac{1}{N_\theta^2}
\sum_{j,l}
E(R,\delta_j,\delta_l)
e^{ipN\delta_j}
e^{-iqN\delta_l}.
]

Because the fundamental angular domain is only

[
0\leq \delta <\frac{2\pi}{N},
]

very few samples are needed. For (P=1), three or four samples per reduced angular period are theoretically enough, although using (8) or (16) is safer when constructing the reference.

The reference overlap itself can be computed either from the atomic Gaussian pair formula or from

[
E(\mathbf R)
============

\mathcal F^{-1}
\left[
\widehat\rho_A(\mathbf k)^*
\widehat\rho_B(\mathbf k)
\right](\mathbf R).
]

This is an excellent validation route.

---

# 8. Forces and torques are analytic

Write one term as

[
Z_{mn}
======

C_{mn}(R)
e^{-im\delta_A}
e^{in\delta_B}.
]

Then

[
E=\operatorname{Re}\sum_{mn}Z_{mn}.
]

The radial force is

[
\boxed{
F_R
===

-\operatorname{Re}
\sum_{mn}
C_{mn}'(R)
e^{-im\delta_A}
e^{in\delta_B}.
}
]

The tangential force is

[
\boxed{
F_\phi
======

\frac{1}{R}
\sum_{mn}
(n-m)\operatorname{Im}Z_{mn}.
}
]

The molecular torques are

[
\boxed{
\tau_A
======

\sum_{mn}
m,\operatorname{Im}Z_{mn},
}
]

[
\boxed{
\tau_B
======

-\sum_{mn}
n,\operatorname{Im}Z_{mn}.
}
]

These automatically satisfy angular-momentum conservation:

[
\tau_A+\tau_B+RF_\phi=0.
]

So the harmonic representation is not only fast for energies; it also gives consistent forces and torques without finite differences.

---

# 9. How many harmonics are needed?

For a hexagonal molecule, the allowed channels are

[
m=0,\pm6,\pm12,\pm18,\ldots
]

A practical hierarchy is:

[
P=0:
\quad \text{purely radial},
]

[
P=1:
\quad m=0,\pm6,
]

[
P=2:
\quad m=0,\pm6,\pm12.
]

The overlap is smoother than either original density because in reciprocal space it contains the product

[
A_m(k)^*B_n(k),
]

which suppresses high spatial frequencies. Therefore the interaction often requires fewer harmonics than are required to reconstruct the density itself.

For benzene-like shapes, I would start with (P=1), meaning the five radial functions above, and add the (m=\pm12) channels only if comparisons against the Gaussian-pair or FFT reference show significant error.

If many harmonics are retained, the matrix (C_{mn}(R)) can itself be compressed by a low-rank factorization. Low-rank factorizations of Fourier–Bessel translation kernels are an established acceleration strategy for rotated and translated correlations. ([arXiv][2])

---

# Recommended formulation

For your application I would use this pipeline:

[
\text{atomic or grid density}
\longrightarrow
\text{reference overlaps}
\longrightarrow
C_{pq}(R)
\longrightarrow
\text{complex-power runtime evaluation}.
]

For a sixfold symmetric pair, begin with

[
\boxed{
E=
V_0
+V_A\cos6\delta_A
+V_B\cos6\delta_B
+V_{\mathrm{rel}}\cos6(\beta-\alpha)
+V_{\mathrm{sum}}\cos6(2\phi-\alpha-\beta).
}
]

That is essentially a symmetry-adapted 2D Slater–Koster model, but it needs only five one-dimensional radial tables and a handful of complex multiplications per molecular pair.

[1]: https://dlmf.nist.gov/10.23 "DLMF: §10.23 Sums ‣ Bessel and Hankel Functions ‣ Chapter 10 Bessel Functions"
[2]: https://arxiv.org/abs/1905.12317 "[1905.12317] Factorization of the translation kernel for fast rigid image alignment"


---

# USER

Ok, and now how to generalize this to 3D for general case where the axis of symmetry (normal to our complex plane) is not colinear so we have 3 relevant vectors rij=ri-rj (shift between the two molecules) a d ni, nj which are unitary vectors of axis of the two molecules, or maybe better use quaternions qi,qj representing rotation of whole local reference frame of the two molecules 

---

# Chat GPT 5.5

Yes. In 3D the clean generalization is:

* use **quaternions** (q_i,q_j) to describe the complete molecular frames;
* use spherical harmonics rather than ordinary Fourier modes;
* use Wigner (D)-matrices as the 3D analogue of the complex phase (e^{im\alpha});
* absorb translation into radial tables containing spherical Bessel functions (j_L(kR)).

A crucial point is that (q^n) is **not** generally the 3D analogue of (e^{in\alpha}). Quaternion powers only work that way for repeated rotation about one fixed axis. For arbitrarily tilted molecular frames, the correct object is a selected column of the Wigner matrix (D^l_{m\mu}(q)).

---

# 1. Minimal pose variables

Let the quaternion (q_s), (s=i,j), rotate vectors from molecular body coordinates into world coordinates:

[
\mathbf v_{\mathrm{world}}
==========================

Q(q_s)\mathbf v_{\mathrm{body}}.
]

Let

[
\mathbf R=\mathbf r_j-\mathbf r_i.
]

The energy is invariant under an arbitrary simultaneous global rotation. Therefore the complete pair geometry can be reduced to

[
\boxed{
\mathbf u=Q(q_i)^T\mathbf R,
\qquad
q_{ij}=q_i^{-1}q_j.
}
]

Thus the most general rigid-body pair potential has the form

[
\boxed{
E(\mathbf R,q_i,q_j)
====================

F!\left(Q(q_i)^T\mathbf R,;q_i^{-1}q_j\right).
}
]

This is perhaps the most useful starting point computationally:

* (\mathbf u) is the translation expressed in the frame of molecule (i);
* (q_{ij}) is the orientation of molecule (j) relative to molecule (i).

There are six independent relative degrees of freedom: three translational and three rotational. No absolute world orientation is required.

Equivalently, write

[
R=|\mathbf R|,
\qquad
\widehat{\mathbf u}
===================

Q(q_i)^T\frac{\mathbf R}{R},
\qquad
q_{ij}=q_i^{-1}q_j.
]

Then

[
E=F(R,\widehat{\mathbf u},q_{ij}),
]

which means one radial variable and five angular variables.

---

# 2. When are only (\mathbf n_i,\mathbf n_j,\mathbf R) sufficient?

Let

[
\mathbf n_i=Q(q_i)\mathbf e_z,
\qquad
\mathbf n_j=Q(q_j)\mathbf e_z
]

be the molecular symmetry axes.

## Continuously axisymmetric molecules

If the densities are invariant under **all** rotations about their axes, then rotation around (\mathbf n_i) or (\mathbf n_j) does not matter. The quaternions contain redundant information, and the energy can depend only on

[
x=\widehat{\mathbf R}\cdot\mathbf n_i,
\qquad
y=\widehat{\mathbf R}\cdot\mathbf n_j,
\qquad
z=\mathbf n_i\cdot\mathbf n_j.
]

There is also the oriented triple product

[
\tau
====

\widehat{\mathbf R}\cdot
(\mathbf n_i\times\mathbf n_j),
]

but

[
\boxed{
\tau^2
======

1+2xyz-x^2-y^2-z^2.
}
]

For ordinary reflection-even scalar density overlap, the sign of (\tau) cannot matter, so

[
\boxed{
E=F(R,x,y,z).
}
]

## Discrete (C_n) symmetry

For a benzene-like object, the density is invariant only under

[
\chi\rightarrow\chi+\frac{2\pi}{n}
]

around its normal. The twist angle around the normal remains physically relevant modulo (2\pi/n).

Therefore (\mathbf n_i,\mathbf n_j,\mathbf R) are **not enough**. You need either:

* full quaternions (q_i,q_j), or
* each normal plus one tangent vector defining the in-plane orientation.

For example, define the complete local frame

[
\mathbf a_i=Q(q_i)\mathbf e_x,
\qquad
\mathbf b_i=Q(q_i)\mathbf e_y,
\qquad
\mathbf n_i=Q(q_i)\mathbf e_z.
]

For an (n)-fold planar molecule, the orientation of (\mathbf a_i,\mathbf b_i) matters modulo (2\pi/n).

---

# 3. Direct Gaussian formula in 3D

Before introducing harmonics, the direct atomic-density formula remains extremely useful.

Let

[
\rho_i(\mathbf x)
=================

\sum_a
q_a
\left(\frac{\zeta_a}{\pi}\right)^{3/2}
\exp\left[
-\zeta_a
\left|
\mathbf x-Q(q_i)\mathbf a_a-\mathbf r_i
\right|^2
\right].
]

Then

[
E=\kappa\int\rho_i(\mathbf x)\rho_j(\mathbf x),d^3x
]

becomes

[
\boxed{
E
=

\kappa
\sum_{ab}
q_aq_b
\left[
\frac{\zeta_a\zeta_b}
{\pi(\zeta_a+\zeta_b)}
\right]^{3/2}
\exp[-\mu_{ab}d_{ab}^2],
}
]

where

[
\mu_{ab}
========

\frac{\zeta_a\zeta_b}{\zeta_a+\zeta_b},
]

and

[
\boxed{
\mathbf d_{ab}
==============

\mathbf R
+
Q(q_j)\mathbf b_b
-----------------

Q(q_i)\mathbf a_a.
}
]

For two six-site rings this is only (36) pair evaluations. It is an excellent exact reference and may actually be faster than a general spherical-harmonic contraction unless the molecular densities contain many atoms or are reused extremely often.

---

# 4. Spherical-harmonic representation of one molecular density

In the molecular body frame, expand the Fourier-transformed density as

[
\boxed{
\widetilde\rho_s^0(k,\widehat{\mathbf k})
=========================================

\sum_{l=0}^{l_{\max}}
\sum_{\mu=-l}^{l}
A^s_{l\mu}(k)
Y_{l\mu}(\widehat{\mathbf k}).
}
]

Here:

* (l) is the total angular frequency;
* (\mu) is the azimuthal frequency around the molecular body (z)-axis;
* (A_{l\mu}(k)) is a radial function.

Under a rigid rotation (q_s), coefficients with the same (l) mix through a Wigner matrix:

[
\boxed{
A^s_{lm}(k;q_s)
===============

\sum_{\mu=-l}^{l}
D^l_{m\mu}(q_s)
A^s_{l\mu}(k).
}
]

The Wigner matrix is the three-dimensional analogue of the two-dimensional phase (e^{-im\alpha}): spherical harmonics of degree (l) form a (2l+1)-dimensional irreducible representation of rotations. ([arXiv][1])

I am defining the transformation by the equation above; this avoids ambiguities from the different conjugation and active/passive-rotation conventions used in libraries.

---

# 5. Point-group symmetry greatly reduces the coefficients

Suppose the body-frame (z)-axis is the (C_n) symmetry axis. Under a body-frame rotation by (2\pi/n),

[
Y_{l\mu}
\rightarrow
e^{i\mu 2\pi/n}Y_{l\mu}.
]

Invariance requires

[
e^{i\mu2\pi/n}=1,
]

so

[
\boxed{
\mu=pn,\qquad p\in\mathbb Z.
}
]

Consequently,

[
A_{l\mu}=0
\qquad\text{unless}\qquad
\mu=0,\pm n,\pm2n,\ldots
]

with (|\mu|\leq l).

This has an important consequence:

[
\boxed{
\text{The first twist-sensitive harmonic of a }C_n
\text{ object occurs at }l=n.
}
]

For a hexagonal ring:

[
\mu=0,\pm6,\pm12,\ldots
]

Therefore:

* all terms with (l<6) see the ring only as an axisymmetric disk;
* (l=6,\mu=\pm6) is the first mode that resolves the six individual lobes;
* (l=12,\mu=\pm12) sharpens the sixfold structure.

This is exactly analogous to the 2D modes (m=0,\pm6,\pm12,\ldots).

For a continuously axisymmetric body, only

[
\mu=0
]

survives.

---

# 6. Exact overlap in Fourier space

The density overlap is

[
E
=

\frac{\kappa}{(2\pi)^3}
\int
\widetilde\rho_i(\mathbf k)^*
\widetilde\rho_j(\mathbf k)
e^{-i\mathbf k\cdot\mathbf R}
,d^3k.
]

Expand the translation phase using spherical Bessel functions:

[
\boxed{
e^{-i\mathbf k\cdot\mathbf R}
=============================

4\pi
\sum_{LM}
(-i)^L
j_L(kR)
Y_{LM}^*(\widehat{\mathbf k})
Y_{LM}(\widehat{\mathbf R}).
}
]

This is the 3D counterpart of the Jacobi–Anger expansion used in 2D. ([arXiv][2])

Substituting all expansions gives

[
\begin{aligned}
E={}&
\operatorname{Re}
\sum_{\substack{
l_i\mu_i m_i\
l_j\mu_j m_j\
LM}}
D^{l_i}*{m_i\mu_i}(q_i)^*
D^{l_j}*{m_j\mu_j}(q_j)
Y_{LM}(\widehat{\mathbf R})
\
&\times
\mathcal G^{LM}_{l_im_i,l_jm_j}
,
K^{,l_i\mu_i,l_j\mu_j}_L(R),
\end{aligned}
]

where

[
\boxed{
K^{,l_i\mu_i,l_j\mu_j}_L(R)
===========================

\frac{4\pi\kappa}{(2\pi)^3}
(-i)^L
\int_0^\infty
k^2dk,
A^i_{l_i\mu_i}(k)^*
A^j_{l_j\mu_j}(k)
j_L(kR)
}
]

is a one-dimensional radial table, and

[
\boxed{
\mathcal G^{LM}_{l_im_i,l_jm_j}
===============================

\int
Y_{l_im_i}^*(\Omega)
Y_{l_jm_j}(\Omega)
Y_{LM}^*(\Omega)
,d\Omega
}
]

is a Gaunt coefficient.

Gaunt coefficients can be written in terms of Wigner (3j) symbols and obey strong triangle, magnetic-index, and parity selection rules, making this sum sparse. ([DLMF][3])

This is the exact 3D analogue of the 2D formula

[
E(R,\delta_i,\delta_j)
======================

\sum_{mn}
C_{mn}(R)e^{-im\delta_i}e^{in\delta_j}.
]

The correspondence is

[
e^{im\alpha}
\quad\longrightarrow\quad
D^l_{m\mu}(q),
]

[
J_{n-m}(kR)
\quad\longrightarrow\quad
j_L(kR)\times\text{Gaunt coefficient}.
]

---

# 7. Slater–Koster-like bond-frame formulation

The general world-frame formula is not the most efficient implementation.

Choose a bond-frame quaternion (q_R) such that

[
Q(q_R)\mathbf e_z=\widehat{\mathbf R}.
]

Transform both molecular frames into this bond frame:

[
\boxed{
\bar q_i=q_R^{-1}q_i,
\qquad
\bar q_j=q_R^{-1}q_j.
}
]

Now the translation is along the bond-frame (z)-axis:

[
\mathbf R_{\mathrm{bond}}=(0,0,R).
]

Because

[
Y_{LM}(\mathbf e_z)
===================

\sqrt{\frac{2L+1}{4\pi}}\delta_{M0},
]

only (M=0) remains. The azimuthal selection rule then requires the same laboratory magnetic index (m) on both molecules.

The interaction becomes

[
\boxed{
\begin{aligned}
E={}&
\operatorname{Re}
\sum_{\substack{
l_i,l_j,L\
\mu_i,\mu_j,m}}
K^{,l_i\mu_i,l_j\mu_j}*L(R)
\sqrt{\frac{2L+1}{4\pi}}
\
&\qquad\times
g^{,l_il_jL}*m
D^{l_i}*{m\mu_i}(\bar q_i)^*
D^{l_j}*{m\mu_j}(\bar q_j),
\end{aligned}
}
]

where

[
\boxed{
g^{,l_il_jL}_m
==============

\int
Y_{l_im}^*(\Omega)
Y_{l_jm}(\Omega)
Y_{L0}(\Omega),d\Omega.
}
]

Explicitly,

[
\boxed{
g^{,l_il_jL}_m
==============

(-1)^m
\sqrt{
\frac{
(2l_i+1)(2l_j+1)(2L+1)
}{4\pi}
}
\begin{pmatrix}
l_i&l_j&L\
0&0&0
\end{pmatrix}
\begin{pmatrix}
l_i&l_j&L\
-m&m&0
\end{pmatrix}.
}
]

This is very close to ordinary Slater–Koster theory:

* align the coordinate system with the inter-center vector;
* rotate each molecular orbital/density multipole into that frame;
* contract corresponding azimuthal channels;
* interpolate radial matrix elements in (R).

The arbitrary roll used when constructing (q_R) is a gauge choice. Rotating the bond-frame (x,y) axes changes both (\bar q_i) and (\bar q_j), but the common-(m) products cancel this phase.

---

# 8. Axisymmetric special case

For continuously axisymmetric densities,

[
\mu_i=\mu_j=0.
]

Then

[
\boxed{
E
=

\operatorname{Re}
\sum_{l_i,l_j,L,m}
V_{l_il_jL}(R)
g^{,l_il_jL}*m
D^{l_i}*{m0}(\bar q_i)^*
D^{l_j}_{m0}(\bar q_j).
}
]

But

[
D^l_{m0}(q)
\propto
Y_{lm}(\mathbf n),
]

so the Wigner matrices can be replaced by spherical harmonics evaluated on the two molecular axes.

Coordinate-independently,

[
\boxed{
E
=

\sum_{l_i,l_j,L}
V_{l_il_jL}(R)
\left[
Y_{l_i}(\mathbf n_i)
\otimes
Y_{l_j}(\mathbf n_j)
\right]_L
\cdot
Y_L(\widehat{\mathbf R}).
}
]

This is the systematic harmonic basis for an interaction

[
E(R,x,y,z).
]

For a low-order phenomenological approximation, you could fit

[
\begin{aligned}
E\simeq {}&
V_0(R)
+
V_i(R)P_2(x)
+
V_j(R)P_2(y)
\
&+
V_{ij}(R)P_2(z)
+
V_{xyz}(R)xyz
+
V_{xy}(R)P_2(x)P_2(y)
+\cdots.
\end{aligned}
]

For identical apolar disks, where (\mathbf n\equiv-\mathbf n), retain only combinations invariant under each independent sign flip

[
\mathbf n_i\rightarrow-\mathbf n_i,
\qquad
\mathbf n_j\rightarrow-\mathbf n_j.
]

---

# 9. Why quaternion power is not the right operation

In 2D,

[
z=e^{i\alpha}
\quad\Rightarrow\quad
z^n=e^{in\alpha}.
]

All rotations belong to the same commuting one-parameter group.

In 3D, a quaternion has the form

[
q=
\cos\frac{\theta}{2}
+
\mathbf u\sin\frac{\theta}{2}.
]

Then

[
q^n
===

\cos\frac{n\theta}{2}
+
\mathbf u\sin\frac{n\theta}{2}
]

means: repeat the **entire rotation about the same world-space axis** (n) times.

It does not mean:

> keep the molecular normal where it is, but multiply the internal twist angle around that normal by (n).

For the latter operation, decompose the body orientation conceptually into tilt and twist,

[
q=q_{\mathrm{tilt}}q_z(\chi).
]

A Wigner matrix satisfies

[
D^l_{m\mu}!\left(q,q_z(\chi)\right)
===================================

D^l_{m\mu}(q)e^{-i\mu\chi}.
]

Therefore the (\mu=n) column carries exactly the desired phase

[
e^{-in\chi}.
]

That is the proper 3D generalization of (z^n).

---

# 10. An intuitive tensor version

Instead of Wigner matrices, define the complex tangent vector

[
\mathbf e_i
===========

\mathbf a_i+i\mathbf b_i,
]

where

[
(\mathbf a_i,\mathbf b_i,\mathbf n_i)
]

is the local molecular frame.

Under a rotation by (\chi) around (\mathbf n_i),

[
\mathbf e_i\rightarrow e^{-i\chi}\mathbf e_i.
]

Therefore

[
\boxed{
\mathcal T_i^{(n)}
==================

\operatorname{STF}
\left(
\mathbf e_i^{\otimes n}
\right)
}
]

transforms as

[
\mathcal T_i^{(n)}
\rightarrow
e^{-in\chi}\mathcal T_i^{(n)}.
]

Here STF means symmetric traceless projection. This rank-(n) irreducible tensor is equivalent to an (l=n) spherical-harmonic vector with (2n+1) components.

For hexagonal symmetry,

[
\mathcal T_i^{(6)}
]

is the direct geometric analogue of the 2D complex number (e^{i6\alpha_i}).

One can construct scalar interaction terms such as contractions of

[
\mathcal T_i^{(6)},\qquad
\mathcal T_j^{(6)},\qquad
\widehat{\mathbf R}^{\otimes L}.
]

In practice, storing the (2l+1) spherical components is much cheaper than storing a full Cartesian rank-six tensor.

---

# 11. Practical representation I would use

For a general pair of (C_n)-symmetric molecules:

### Precomputation

Expand each body-frame density using only symmetry-allowed modes:

[
A_{l\mu}(k),
\qquad
\mu=0,\pm n,\pm2n,\ldots
]

and calculate radial tables

[
K^{,l_i\mu_i,l_j\mu_j}_L(R).
]

For a hexagonal ring, begin with

[
(l,\mu)=(0,0),\quad
(2,0),\quad
(4,0),\quad
(6,0),\quad
(6,\pm6).
]

The (\mu=0) terms describe the disk-like radial and axial shape. The (\mu=\pm6) terms introduce the actual hexagonal modulation.

### Runtime

1. Compute

   [
   \mathbf R,\quad R,\quad\widehat{\mathbf R}.
   ]

2. Construct any quaternion (q_R) that maps (\mathbf e_z) to (\widehat{\mathbf R}).

3. Compute

   [
   \bar q_i=q_R^{-1}q_i,
   \qquad
   \bar q_j=q_R^{-1}q_j.
   ]

4. Evaluate only the required Wigner columns

   [
   D^l_{m0},\qquad D^6_{m,\pm6},
   ]

   rather than complete dense (D^l) matrices.

5. Interpolate the radial tables and perform the sparse contractions.

For low (l), the required Wigner elements can be evaluated as explicit quaternion polynomials or by stable recurrence. There is no need to compute Euler angles or trigonometric functions. Wigner rotation coefficients are routinely computed recursively for spherical-harmonic rotation problems. ([arXiv][1])

---

# 12. Likely performance ranking

For two six-atom rings:

1. **Direct Gaussian atom-pair sum:** probably simplest and possibly fastest.
2. **Axisymmetric invariant model (F(R,x,y,z)):** fastest but does not resolve hexagonal twist.
3. **Axisymmetric model plus one (l=6,\mu=\pm6) channel:** probably the best compressed model.
4. **General spherical-harmonic/Wigner expansion:** systematically convergent but more machinery.

Thus I would not initially implement a large general (SO(3)) expansion. I would use:

[
\boxed{
E
=

E_{\mathrm{axis}}(R,x,y,z)
+
E_{C_6}(R,q_i,q_j,\widehat{\mathbf R}),
}
]

where (E_{\mathrm{axis}}) captures most of the repulsion and (E_{C_6}) is a relatively small correction constructed from the (\mu=\pm6) Wigner channels. This preserves the physical disk orientation and the sixfold atomic corrugation without requiring a high angular cutoff.

[1]: https://arxiv.org/abs/1403.7698?utm_source=chatgpt.com "Recursive computation of spherical harmonic rotation coefficients of large degree"
[2]: https://arxiv.org/pdf/2406.15385?utm_source=chatgpt.com "On a Generating Function for the Isotropic Basis ..."
[3]: https://dlmf.nist.gov/34.3?utm_source=chatgpt.com "34.3 Basic Properties: 3 ⁢ j Symbol"

---

# USER

Fuck. You forget the original motivation why we brother to do it at all which is speed. If you say that in the end pairwise sum over atoms of both molecules is the fastest you lost because you return where we started.

I already found faster representation if I make the piecewise symmetric representation of one molecule and I sample it by the second atom I get from O(n2) to O(n), but I was hoping that with some clever math there can be something faster like O(n). It can be approximative but it must preserve theesence (n-fold protrusion), is that clear?

---

# Chat GPT 5.5

Yes, clear. I lost the actual optimization target.

Let

* (N_i,N_j): numbers of atoms or protrusions being sampled;
* (\nu): symmetry order, such as (\nu=6) for benzene.

Your current construction reduces

[
O(N_iN_j)\longrightarrow O(N_j)
]

by evaluating an analytic (C_\nu)-symmetric field of molecule (i) at the atoms of molecule (j).

What we really want is to compress **both** molecules, giving a pair interaction whose runtime is independent of (N_i,N_j):

[
\boxed{O(1)\text{ with respect to atom count}}
]

or, more precisely, (O(\nu)) arithmetic for one retained (\nu)-fold harmonic. Since (\nu=6) is fixed, that is effectively constant cost.

There is a particularly convenient representation that does exactly this: a **Gaussian envelope multiplied by one complex angular polynomial**.

---

# 1. Encode the complete 3D orientation with one complex tangent vector

From quaternion (q_i), rotate the local frame:

[
\mathbf a_i=Q(q_i)\mathbf e_x,\qquad
\mathbf b_i=Q(q_i)\mathbf e_y,\qquad
\mathbf n_i=Q(q_i)\mathbf e_z.
]

Define the complex vector in the molecular plane

[
\boxed{\mathbf u_i=\mathbf a_i+i\mathbf b_i.}
]

It has the useful properties

[
\mathbf u_i\cdot\mathbf u_i=0,
\qquad
\mathbf u_i\cdot\mathbf u_i^*=2.
]

For a point in the molecular plane,

[
\mathbf x
=========

r(\cos\theta,\mathbf a_i+\sin\theta,\mathbf b_i),
]

we have

[
\mathbf u_i\cdot\mathbf x=re^{i\theta}.
]

Therefore

[
\boxed{
\operatorname{Re}\left[(\mathbf u_i\cdot\mathbf x)^\nu\right]
=============================================================

r^\nu\cos(\nu\theta).
}
]

This directly produces (\nu) protrusions.

No Euler angles, Wigner matrices, or quaternion powers are needed. The quaternion is used only to rotate (\mathbf a,\mathbf b,\mathbf n).

---

# 2. A compressed molecular-density primitive

Use an anisotropic Gaussian envelope

[
G_i(\mathbf x)
==============

\exp\left[
-\frac12
\mathbf x^T K_i\mathbf x
\right],
]

with

[
\boxed{
K_i
===

k_{\perp i}
\left(I-\mathbf n_i\mathbf n_i^T\right)
+
k_{\parallel i}
\mathbf n_i\mathbf n_i^T.
}
]

Here:

* (k_\perp) controls the radial size in the molecular plane;
* (k_\parallel) controls thickness normal to the plane.

Represent the molecular density approximately as

[
\boxed{
\rho_i(\mathbf x)
=================

G_i(\mathbf x)
\left[
c_{0i}
+
c_{\nu i}
\operatorname{Re}
\left(
\frac{\mathbf u_i\cdot\mathbf x}{\ell_i}
\right)^\nu
\right].
}
]

In the molecular plane, the corrugation amplitude is

[
r^\nu e^{-k_\perp r^2/2}.
]

Its radial maximum is at

[
\boxed{
r_0=\sqrt{\frac{\nu}{k_\perp}}.
}
]

Thus, to place six protrusions approximately at ring radius (r_0), choose

[
k_\perp=\frac{6}{r_0^2}.
]

This single primitive captures:

* molecular normal;
* arbitrary tilt;
* in-plane twist;
* (\nu)-fold protrusions;
* radial decay;
* finite out-of-plane thickness.

It is a very low-bandwidth steerable Gaussian representation: rotations act only on the complex orientation vector (\mathbf u_i). ([arXiv][1])

---

# 3. Overlap of two such primitives

Put molecule (i) at the origin and molecule (j) at

[
\mathbf R=\mathbf r_j-\mathbf r_i.
]

Their Gaussian factors are

[
G_i(\mathbf x)
==============

e^{-\frac12\mathbf x^TK_i\mathbf x},
]

[
G_j(\mathbf x-\mathbf R)
========================

e^{-\frac12(\mathbf x-\mathbf R)^T
K_j(\mathbf x-\mathbf R)}.
]

Define

[
K=K_i+K_j,
\qquad
\Sigma=K^{-1},
]

and

[
\boxed{
\mathbf m=\Sigma K_j\mathbf R.
}
]

The product of the two Gaussians is another Gaussian centered at (\mathbf m):

[
G_i(\mathbf x)G_j(\mathbf x-\mathbf R)
======================================

I_0(\mathbf R),
\mathcal N(\mathbf x;\mathbf m,\Sigma),
]

where the unnormalized Gaussian overlap is

[
\boxed{
I_0=
\frac{(2\pi)^{3/2}}{\sqrt{\det K}}
\exp\left[
-\frac12
\left(
\mathbf R^TK_j\mathbf R
-----------------------

(K_j\mathbf R)^T
\Sigma(K_j\mathbf R)
\right)
\right].
}
]

Computing this costs one symmetric (3\times3) inverse and a few matrix-vector products. It is independent of atom count.

---

# 4. The key complex Gaussian moment

Consider the general integral

[
M_{pq}(\mathbf u,\mathbf v)
===========================

\int
G_i(\mathbf x)
G_j(\mathbf x-\mathbf R)
(\mathbf u\cdot\mathbf x)^p
[\mathbf v\cdot(\mathbf x-\mathbf R)]^q
,d^3x.
]

Here (\mathbf u) and (\mathbf v) are complex vectors. Importantly, all dot products below are **bilinear**, not Hermitian:

[
\mathbf u^T\Sigma\mathbf v,
]

not (\mathbf u^\dagger\Sigma\mathbf v).

Define five complex scalars:

[
\boxed{
A=\mathbf u\cdot\mathbf m,
}
]

[
\boxed{
B=\mathbf v\cdot(\mathbf m-\mathbf R),
}
]

[
\boxed{
U=\mathbf u^T\Sigma\mathbf u,
\qquad
V=\mathbf v^T\Sigma\mathbf v,
\qquad
C=\mathbf u^T\Sigma\mathbf v.
}
]

Then

[
M_{pq}=I_0,H_{pq}(A,B,U,V,C).
]

The entire arbitrary 3D translation and orientation dependence is compressed into these five complex numbers.

---

# 5. Evaluate the moment in (O(p+q))

First calculate one-dimensional Gaussian moments

[
h_0=1,\qquad h_1=A,
]

[
\boxed{
h_{r+1}=A h_r+rU h_{r-1}.
}
]

Similarly,

[
g_0=1,\qquad g_1=B,
]

[
\boxed{
g_{r+1}=B g_r+rV g_{r-1}.
}
]

Then

[
\boxed{
H_{pq}
======

\sum_{k=0}^{\min(p,q)}
\binom pk\binom qk k!,
C^k
h_{p-k}g_{q-k}.
}
]

This formula follows directly by factoring the Gaussian generating function:

[
e^{As+\frac12Us^2}
e^{Bt+\frac12Vt^2}
e^{Cst}.
]

The important point is the complexity:

[
\boxed{
\text{cost}(H_{\nu\nu})=O(\nu),
}
]

not (O(\nu^2)), and certainly not (O(N_iN_j)).

For (\nu=6), this is merely:

* seven (h_r) values;
* seven (g_r) values;
* seven contraction terms.

That is only a few dozen complex multiply-adds.

---

# 6. Full real (C_\nu)-symmetric overlap

Write

[
z_i=\mathbf u_i\cdot\mathbf x,
\qquad
z_j=\mathbf u_j\cdot(\mathbf x-\mathbf R).
]

The density modulation is (\operatorname{Re}(z^\nu)). Therefore

[
\operatorname{Re}(z_i^\nu)
\operatorname{Re}(z_j^\nu)
==========================

\frac12\operatorname{Re}
\left[
z_i^\nu z_j^\nu
+
z_i^\nu (z_j^*)^\nu
\right].
]

We consequently need only two (\nu)-to-(\nu) moments:

[
H_{\nu\nu}^{++}
===============

H_{\nu\nu}
(\mathbf u_i,\mathbf u_j),
]

[
H_{\nu\nu}^{+-}
===============

H_{\nu\nu}
(\mathbf u_i,\mathbf u_j^*).
]

The complete overlap is

[
\boxed{
\begin{aligned}
E= \kappa I_0\Bigg[
&c_{0i}c_{0j}
\
&+
\frac{c_{\nu i}c_{0j}}{\ell_i^\nu}
\operatorname{Re}H_{\nu0}(\mathbf u_i)
\
&+
\frac{c_{0i}c_{\nu j}}{\ell_j^\nu}
\operatorname{Re}H_{0\nu}(\mathbf u_j)
\
&+
\frac{c_{\nu i}c_{\nu j}}
{2\ell_i^\nu\ell_j^\nu}
\operatorname{Re}
\left(
H_{\nu\nu}^{++}
+
H_{\nu\nu}^{+-}
\right)
\Bigg].
\end{aligned}
}
]

This is already the desired arbitrary-translation, arbitrary-rotation pair potential.

It directly consumes

[
\mathbf R,\quad q_i,\quad q_j.
]

---

# 7. Isotropic-envelope simplification

For the corrugation part, suppose

[
K_i=k_i I,\qquad K_j=k_j I.
]

Then

[
\Sigma=\frac{I}{k_i+k_j}.
]

Because

[
\mathbf u_i\cdot\mathbf u_i=0,
]

we get

[
U=V=0.
]

Therefore

[
h_r=A^r,\qquad g_r=B^r,
]

and

[
\boxed{
H_{pq}
======

\sum_{k=0}^{\min(p,q)}
\binom pk\binom qk k!,
C^kA^{p-k}B^{q-k}.
}
]

For equal harmonic orders (p=q=\nu), put

[
X=AB.
]

Then

[
\boxed{
H_{\nu\nu}
==========

\sum_{k=0}^{\nu}
\binom{\nu}{k}^2k!,
C^kX^{\nu-k}.
}
]

Equivalently,

[
\boxed{
H_{\nu\nu}
==========

\nu!C^\nu
L_\nu\left(-\frac{X}{C}\right),
}
]

where (L_\nu) is a Laguerre polynomial. For numerical evaluation near (C=0), the finite polynomial form is safer than the divided form. ([DLMF][2])

---

## Explicit hexagonal formula

For (\nu=6),

[
\boxed{
\begin{aligned}
H_{66}={}&
X^6
+36CX^5
+450C^2X^4
+2400C^3X^3
\
&+
5400C^4X^2
+4320C^5X
+720C^6.
\end{aligned}
}
]

That is a sixth-degree polynomial evaluated twice:

1. with (\mathbf v=\mathbf u_j);
2. with (\mathbf v=\mathbf u_j^*).

So the complete hexagonal interaction can be only a few tens of scalar operations.

---

# 8. What the complex contractions encode

The translation enters through

[
\mathbf u_i\cdot\mathbf R
=========================

\mathbf a_i\cdot\mathbf R
+
i,\mathbf b_i\cdot\mathbf R.
]

This is exactly the complex representation of the projection of (\mathbf R) into the plane of molecule (i).

Relative orientations enter through

[
\mathbf u_i\cdot\mathbf u_j
===========================

## \mathbf a_i\cdot\mathbf a_j

\mathbf b_i\cdot\mathbf b_j
+
i\left(
\mathbf a_i\cdot\mathbf b_j
+
\mathbf b_i\cdot\mathbf a_j
\right),
]

and

[
\mathbf u_i\cdot\mathbf u_j^*
=============================

\mathbf a_i\cdot\mathbf a_j
+
\mathbf b_i\cdot\mathbf b_j
+
i\left(
-\mathbf a_i\cdot\mathbf b_j
+
\mathbf b_i\cdot\mathbf a_j
\right).
]

Thus the formula knows about:

* tilt of each molecular plane relative to (\mathbf R);
* angle between the normals;
* relative in-plane twist;
* simultaneous alignment of both protrusion patterns.

All without explicitly extracting any angle.

---

# 9. Recommended fast practical model

I would not force a single primitive to reproduce the complete electron density. Instead split the potential into:

[
\boxed{
E=E_{\mathrm{smooth}}+E_{\mathrm{corrugation}}.
}
]

## Smooth body

Use one or two axisymmetric anisotropic Gaussians:

[
E_{\mathrm{smooth}}
===================

\sum_{\alpha\beta}
c_{\alpha i}c_{\beta j}
I_0(K_{\alpha i},K_{\beta j},\mathbf R).
]

This captures:

* disk shape;
* thickness;
* normal-normal orientation;
* broad Pauli overlap.

Cost: a few constant-size Gaussian-overlap calculations.

## (C_\nu) corrugation

Use one Gaussian-harmonic primitive:

[
E_{\mathrm{corrugation}}
\propto
\operatorname{Re}
\left[
H_{\nu\nu}^{++}
+
H_{\nu\nu}^{+-}
\right].
]

This adds the actual (\nu)-fold protrusions.

For sharper blobs, add just the next mode,

[
2\nu,
]

rather than introducing explicit atoms:

[
\rho
\simeq
G
\left[
c_0
+c_\nu r^\nu\cos\nu\theta
+c_{2\nu}r^{2\nu}\cos2\nu\theta
\right].
]

Usually one or two modes should be enough to preserve the essential geometry.

---

# 10. Complexity

For one radial primitive and one (\nu)-fold harmonic:

[
\boxed{
O(\nu)
}
]

per molecular pair.

For fixed (\nu=6):

[
\boxed{
O(1)
}
]

with respect to molecule size and atom count.

With (P) retained harmonics and (M) radial Gaussian primitives:

[
O(M_iM_jP_iP_j\nu),
]

but practically (M=1)–(3) and (P=1)–(2).

There is no:

* atom loop;
* angular grid;
* FFT;
* `atan2`;
* Wigner matrix;
* spherical-harmonic expansion at runtime.

The exact overlap of arbitrarily narrow explicit atomic blobs cannot generally be represented by a fixed number of coefficients. But once the requirement is “preserve the essential (\nu)-fold protrusion,” the single (r^\nu\cos\nu\theta) Gaussian harmonic is the minimal compressed representation, and its two-body overlap closes analytically.

[1]: https://arxiv.org/pdf/1807.02547?utm_source=chatgpt.com "3D Steerable CNNs: Learning Rotationally Equivariant ..."
[2]: https://dlmf.nist.gov/18.3?utm_source=chatgpt.com "18.3 Definitions ‣ Classical Orthogonal Polynomials ‣ ..."


---

# USER

You say "O(p^2)" with p being the retained models. I think there is the misundrstanding. I did not want general formula (like multipole expansion, spherical harmonics) to express any shape exactly. I wanted 1) tailored solution where we choose some nodes motivated by symmetry 2) we sacrifice substantial apporximation for speed. We edfinitely do now want complete basis, but highly biased/taylored basis containing e.g. just 1-2 modes but able to describe high symemtry (like 6-fold benzene).


Lets think about spehrical harmonics in angular (not caresian) basis. What if we choose just monopole (apporximating the molecule by sphere) plus one mode represeint the 6 protrusion of 6 atoms. Not sure which spherical harmonic it is, but lets say that it has like l=3 or 4 (d or f shell), and it has 6 nodes along phi (along circumference in x,y plane, where z is polar axis), we probably need two sin,cos to represent arbitrary phase shift (rotation in xy plane around z).

so we have 3x3=9 interactions instead of 6x6 interactions. The advantage is also that we have only one center (square root or exp), the rest are just polynominals (complex multiplication is polynominal as well).

elaborate on these idea, put them into rigrous math, concreate algorithm (pseoudocode) how to fficiently evaluate it. Care about numerical performance on real computer (CPU, GPU) not just abstract math.

---

# Chat GPT 5.5

Yes—now the target is precise:

[
\boxed{\text{one monopole channel}+\text{one }C_6\text{ channel}}
]

with a **fixed three-component descriptor**

[
{1,\cos 6\phi,\sin 6\phi}
]

per molecule. We do not construct a general spherical-harmonic basis. The overlap is a fixed (3\times3) contraction, independent of atom count, evaluated using one Gaussian exponential and polynomial arithmetic.

The useful trick is to combine:

1. a Gaussian radial envelope, because products and integrals of Gaussians close analytically;
2. the single solid harmonic ((x+iy)^6), because arbitrary 3D rotation can be encoded directly by the molecular frame;
3. a specialized degree-six Gaussian moment, avoiding Wigner matrices and general spherical harmonics.

---

# 1. Which spherical harmonic represents six protrusions?

A dependence with six periods around the azimuth is

[
\cos 6\phi,\qquad \sin 6\phi.
]

The magnetic index must therefore be

[
m=\pm6.
]

Because spherical harmonics require (|m|\le l), the smallest possible degree is

[
\boxed{l=6,\quad m=\pm6.}
]

So it is not a (d), (f), or (g) harmonic. The minimal linear harmonic is (Y_6^{\pm6}). For (m=l),

[
Y_6^6(\theta,\phi)\propto \sin^6\theta,e^{i6\phi}.
]

Multiplying by (r^6) produces the regular solid harmonic

[
\boxed{
r^6Y_6^6\propto (x+iy)^6.
}
]

This relation follows directly from the standard definition of spherical harmonics. ([DLMF][1])

Its real and imaginary parts are

[
C_6(x,y)=\Re[(x+iy)^6]
=r^6\sin^6\theta\cos6\phi,
]

[
S_6(x,y)=\Im[(x+iy)^6]
=r^6\sin^6\theta\sin6\phi.
]

Explicitly,

[
C_6=x^6-15x^4y^2+15x^2y^4-y^6,
]

[
S_6=6x^5y-20x^3y^3+6xy^5.
]

But these explicit polynomials should not be evaluated directly. Complex multiplication is shorter and numerically cleaner.

A cubic (m=3) feature can produce six **unsigned** lobes after squaring:

[
\left[\Re(x+iy)^3\right]^2
\propto r_\perp^6\left(1+\cos6\phi\right).
]

So your intuition about (l=3) was not entirely wrong. However, after squaring it decomposes into an axisymmetric component plus the (m=6) component. For a linear three-channel representation, (l=m=6) is the appropriate object.

---

# 2. Avoid rotating 13 spherical-harmonic coefficients

A general rotated (l=6) harmonic would normally mix all

[
m=-6,\ldots,+6,
]

giving 13 coefficients. We do **not** do that.

For molecule (i), obtain two perpendicular unit vectors from its quaternion:

[
\mathbf a_i=Q(q_i)\mathbf e_x,
\qquad
\mathbf b_i=Q(q_i)\mathbf e_y.
]

The normal is

[
\mathbf n_i=\mathbf a_i\times\mathbf b_i,
]

but the algorithm does not actually need it.

Define one complex 3-vector

[
\boxed{
\mathbf u_i=\mathbf a_i+i\mathbf b_i.
}
]

For a real world-space vector (\mathbf y),

[
z_i=\mathbf u_i\cdot\mathbf y
=============================

\mathbf a_i\cdot\mathbf y
+i,\mathbf b_i\cdot\mathbf y.
]

Therefore

[
\boxed{
z_i^6
=====

r_{\perp i}^6e^{i6\phi_i}.
}
]

The real part gives the sixfold pattern in the plane perpendicular to (\mathbf n_i):

[
\Re(z_i^6)
==========

r_{\perp i}^6\cos6\phi_i.
]

Thus an arbitrary 3D rotation is represented simply by rotating (\mathbf a_i,\mathbf b_i). We never construct spherical harmonics or Wigner matrices.

An important algebraic property is

[
\boxed{
\mathbf u_i\cdot\mathbf u_i=0.
}
]

Indeed,

[
(\mathbf a+i\mathbf b)\cdot(\mathbf a+i\mathbf b)
=================================================

1-1+2i,\mathbf a\cdot\mathbf b=0.
]

This identity is what makes the overlap formula unusually cheap.

---

# 3. Three-channel molecular density

Let

[
\mathbf y_i=\mathbf x-\mathbf r_i.
]

Use the deliberately approximate density

[
\boxed{
\rho_i(\mathbf x)
=================

A_i e^{-\alpha_i|\mathbf y_i|^2}
\left[
c_{0i}
+
\Re\left(
h_i
\left(
\frac{\mathbf u_i\cdot\mathbf y_i}{\ell_i}
\right)^6
\right)
\right].
}
]

Parameters:

* (A_i): overall density or Pauli amplitude;
* (\alpha_i): radial Gaussian width;
* (c_{0i}): spherical component;
* (h_i): complex amplitude and phase of the sixfold component;
* (\ell_i): length scale making the sixth-degree feature dimensionless.

When the quaternion frame is chosen so that (\mathbf a_i) points toward one protrusion, (h_i) can be real.

In a complex basis, the three channels are

[
\boxed{
\chi_i=
\left(
1,\ z_i^6,\ z_i^{*6}
\right).
}
]

The density coefficient vector is

[
\boxed{
\mathbf c_i=
\left(
c_{0i},\ \frac{h_i}{2},\ \frac{h_i^*}{2}
\right).
}
]

This is exactly equivalent to

[
{1,\cos6\phi,\sin6\phi}.
]

The quaternion carries the changing molecular orientation; (h_i) stores only a fixed internal phase offset, if one is needed.

---

# 4. Product of the two Gaussian envelopes

Let

[
\mathbf R=\mathbf r_j-\mathbf r_i.
]

Put (\mathbf y=\mathbf x-\mathbf r_i), so that

[
\mathbf x-\mathbf r_j=\mathbf y-\mathbf R.
]

The Gaussian product is

[
e^{-\alpha_i y^2}
e^{-\alpha_j|\mathbf y-\mathbf R|^2}
====================================

e^{-\mu R^2}
e^{-p|\mathbf y-\mathbf P|^2},
]

where

[
p=\alpha_i+\alpha_j,
]

[
\lambda_i=\frac{\alpha_i}{p},
\qquad
\lambda_j=\frac{\alpha_j}{p},
]

[
\mu=\frac{\alpha_i\alpha_j}{p},
]

[
\boxed{
\mathbf P=\lambda_j\mathbf R.
}
]

The monopole overlap is therefore

[
\boxed{
S_0(R)
======

A_iA_j
\left(\frac{\pi}{p}\right)^{3/2}
e^{-\mu R^2}.
}
]

This is the only exponential in the complete calculation.

There is no square root because everything uses (R^2).

All constants except (R^2) can be precomputed for every pair of molecule types.

---

# 5. The five complex invariants

Define

[
\sigma=\frac{1}{2p}.
]

The mean value of the complex coordinate belonging to molecule (i) is

[
\boxed{
A=
\lambda_j
\frac{\mathbf u_i\cdot\mathbf R}{\ell_i}.
}
]

For molecule (j), whose local vector is (\mathbf y-\mathbf R),

[
\boxed{
B=
-\lambda_i
\frac{\mathbf u_j\cdot\mathbf R}{\ell_j}.
}
]

Two different orientation contractions occur:

[
\boxed{
C_{++}
======

\frac{\sigma}{\ell_i\ell_j}
\mathbf u_i\cdot\mathbf u_j,
}
]

[
\boxed{
C_{+-}
======

\frac{\sigma}{\ell_i\ell_j}
\mathbf u_i\cdot\mathbf u_j^*.
}
]

These four complex numbers, together with (S_0(R)), contain the complete relative translation and orientation dependence.

In real vector form, define

[
d_{00}=\mathbf a_i\cdot\mathbf a_j,
\quad
d_{01}=\mathbf a_i\cdot\mathbf b_j,
]

[
d_{10}=\mathbf b_i\cdot\mathbf a_j,
\quad
d_{11}=\mathbf b_i\cdot\mathbf b_j.
]

Then

[
\boxed{
\mathbf u_i\cdot\mathbf u_j
===========================

(d_{00}-d_{11})
+i(d_{01}+d_{10}),
}
]

and

[
\boxed{
\mathbf u_i\cdot\mathbf u_j^*
=============================

(d_{00}+d_{11})
+i(-d_{01}+d_{10}).
}
]

So only four frame–frame dot products are required.

---

# 6. The specialized sixth-order overlap polynomial

Consider the Gaussian moment

[
F_6(A,B,C)
==========

\left\langle
(A+\xi)^6(B+\eta)^6
\right\rangle,
]

where

[
\langle \xi^2\rangle=0,
\qquad
\langle \eta^2\rangle=0,
\qquad
\langle\xi\eta\rangle=C.
]

The first two vanish precisely because

[
\mathbf u_i\cdot\mathbf u_i
===========================

\mathbf u_j\cdot\mathbf u_j=0.
]

Only cross-contractions remain, giving

[
\boxed{
F_6(A,B,C)
==========

\sum_{k=0}^{6}
\binom{6}{k}^2k!,
C^k(AB)^{6-k}.
}
]

Explicitly,

[
\begin{aligned}
F_6={}&
X^6
+36CX^5
+450C^2X^4
+2400C^3X^3
\
&+5400C^4X^2
+4320C^5X
+720C^6,
\end{aligned}
]

where

[
X=AB.
]

This is exact for the chosen three-channel Gaussian model.

---

## Faster recurrence

Do not evaluate all powers separately. Define

[
F_0=1,
\qquad
F_1=X+C.
]

Then

[
\boxed{
F_n=
\left[X+(2n-1)C\right]F_{n-1}
-----------------------------

(n-1)^2C^2F_{n-2}.
}
]

For (n=6):

[
F_2=(X+3C)F_1-C^2,
]

[
F_3=(X+5C)F_2-4C^2F_1,
]

[
F_4=(X+7C)F_3-9C^2F_2,
]

[
F_5=(X+9C)F_4-16C^2F_3,
]

[
\boxed{
F_6=(X+11C)F_5-25C^2F_4.
}
]

This requires:

* one multiplication for (X=AB);
* one for (C^2);
* ten complex multiplications in the recurrence.

The recurrence is division-free and behaves well when (A), (B), or (C) are close to zero. It is related to the standard Laguerre-polynomial recurrence, but the specialized form above is all the implementation needs. ([DLMF][2])

---

# 7. The complete (3\times3) interaction matrix

The overlap matrix between the channels

[
(1,z_i^6,z_i^{*6})
\quad\text{and}\quad
(1,z_j^6,z_j^{*6})
]

is

[
\boxed{
\frac{\mathbf T_{ij}}{S_0}
==========================

\begin{pmatrix}
1
&
B^6
&
B^{*6}
[2mm]
A^6
&
F_6(A,B,C_{++})
&
F_6(A,B^*,C_{+-})
[2mm]
A^{*6}
&
F_6(A^*,B,C_{+-}^*)
&
F_6(A^*,B^*,C_{++}^*)
\end{pmatrix}.
}
]

The interaction is simply

[
\boxed{
E_{ij}
======

\kappa,
\mathbf c_i^T
\mathbf T_{ij}
\mathbf c_j.
}
]

Although this is formally nine channel interactions, reality and complex conjugation mean that only the following need to be evaluated:

1. (S_0);
2. (A^6);
3. (B^6);
4. (F_6(A,B,C_{++}));
5. (F_6(A,B^*,C_{+-})).

Everything else is a conjugate.

---

# 8. Direct real-energy formula

For the density definition above,

[
\rho_i\sim c_{0i}+\Re(h_i z_i^6),
]

the final interaction is

[
\boxed{
\begin{aligned}
E_{ij}
======

\kappa S_0\Big[
&c_{0i}c_{0j}
\
&+
c_{0j}\Re(h_iA^6)
\
&+
c_{0i}\Re(h_jB^6)
\
&+
\frac12
\Re\left(
h_ih_jF_{++}
+
h_ih_j^*F_{+-}
\right)
\Big],
\end{aligned}
}
]

where

[
F_{++}=F_6(A,B,C_{++}),
]

[
F_{+-}=F_6(A,B^*,C_{+-}).
]

When (h_i,h_j) are real,

[
\boxed{
\begin{aligned}
E_{ij}
======

\kappa S_0\Big[
&c_{0i}c_{0j}
+h_ic_{0j}\Re(A^6)
+c_{0i}h_j\Re(B^6)
\
&+
\frac{h_ih_j}{2}
\Re(F_{++}+F_{+-})
\Big].
\end{aligned}
}
]

This is the central formula.

It preserves:

* the broad radial repulsion;
* six protrusions on each molecule;
* arbitrary tilting of either molecular plane;
* arbitrary in-plane phase;
* orientation of each molecule relative to (\mathbf R);
* relative phase of the two sixfold patterns.

And it has no dependence on the number of atoms.

---

# 9. Particularly simple identical-width case

When both molecules use the same Gaussian exponent,

[
\alpha_i=\alpha_j=\alpha,
]

then

[
p=2\alpha,
\qquad
\lambda_i=\lambda_j=\frac12,
]

[
\mu=\frac{\alpha}{2},
\qquad
\sigma=\frac{1}{4\alpha}.
]

Therefore

[
S_0
===

A_iA_j
\left(\frac{\pi}{2\alpha}\right)^{3/2}
e^{-\alpha R^2/2},
]

[
A=\frac{\mathbf u_i\cdot\mathbf R}{2\ell_i},
]

[
B=-\frac{\mathbf u_j\cdot\mathbf R}{2\ell_j},
]

[
C_{++}
======

\frac{\mathbf u_i\cdot\mathbf u_j}
{4\alpha\ell_i\ell_j},
]

[
C_{+-}
======

\frac{\mathbf u_i\cdot\mathbf u_j^*}
{4\alpha\ell_i\ell_j}.
]

For a fixed molecule type, every scalar prefactor can be compiled or stored as a constant.

---

# 10. OpenCL/C-like implementation

```c
typedef struct {
    float4 pos;

    // Cached world-space local x and y axes.
    // Use .xyz; .w can hold another parameter.
    float4 ax;
    float4 ay;

    float amp;
    float c0;

    // Complex sixfold coefficient.
    float2 h6;

    float invL;
} Sym6Body;

typedef struct {
    // Precomputed for a pair of molecule types.
    float prefactor;   // amp factors and (pi/p)^(3/2), if amplitudes fixed
    float mu;
    float lambda_i;
    float lambda_j;
    float sigma;
} Sym6Pair;
```

Complex helpers:

```c
inline float2 cadd(float2 a, float2 b) {
    return (float2)(a.x + b.x, a.y + b.y);
}

inline float2 csub(float2 a, float2 b) {
    return (float2)(a.x - b.x, a.y - b.y);
}

inline float2 cscale(float2 a, float s) {
    return (float2)(a.x*s, a.y*s);
}

inline float2 cconj(float2 a) {
    return (float2)(a.x, -a.y);
}

inline float2 cmul(float2 a, float2 b) {
    return (float2)(
        fma(a.x, b.x, -a.y*b.y),
        fma(a.x, b.y,  a.y*b.x)
    );
}

inline float creal_mul(float2 a, float2 b) {
    // real(a*b)
    return fma(a.x, b.x, -a.y*b.y);
}

inline float dot3f4(float4 a, float4 b) {
    return fma(a.x, b.x, fma(a.y, b.y, a.z*b.z));
}
```

Sixth power:

```c
inline float2 cpow6(float2 z) {
    const float2 z2 = cmul(z,  z);
    const float2 z3 = cmul(z2, z);
    return cmul(z3, z3);
}
```

Specialized (F_6):

```c
inline float2 moment66(float2 A, float2 B, float2 C) {
    const float2 X  = cmul(A, B);
    const float2 C2 = cmul(C, C);

    const float2 F0 = (float2)(1.0f, 0.0f);
    const float2 F1 = cadd(X, C);

    const float2 F2 = csub(
        cmul(cadd(X, cscale(C, 3.0f)), F1),
        C2
    );

    const float2 F3 = csub(
        cmul(cadd(X, cscale(C, 5.0f)), F2),
        cscale(cmul(C2, F1), 4.0f)
    );

    const float2 F4 = csub(
        cmul(cadd(X, cscale(C, 7.0f)), F3),
        cscale(cmul(C2, F2), 9.0f)
    );

    const float2 F5 = csub(
        cmul(cadd(X, cscale(C, 9.0f)), F4),
        cscale(cmul(C2, F3), 16.0f)
    );

    const float2 F6 = csub(
        cmul(cadd(X, cscale(C, 11.0f)), F5),
        cscale(cmul(C2, F4), 25.0f)
    );

    return F6;
}
```

Pair energy:

```c
inline float sym6_pair_energy(
    Sym6Body bi,
    Sym6Body bj,
    Sym6Pair pp
) {
    const float4 R = bj.pos - bi.pos;
    const float R2 = dot3f4(R, R);

    // Projection of R into each molecular complex plane.
    const float2 ziR = (float2)(
        dot3f4(bi.ax, R),
        dot3f4(bi.ay, R)
    );

    const float2 zjR = (float2)(
        dot3f4(bj.ax, R),
        dot3f4(bj.ay, R)
    );

    const float2 A = cscale(
        ziR,
        pp.lambda_j * bi.invL
    );

    const float2 B = cscale(
        zjR,
        -pp.lambda_i * bj.invL
    );

    // Relative orientation of the two in-plane frames.
    const float d00 = dot3f4(bi.ax, bj.ax);
    const float d01 = dot3f4(bi.ax, bj.ay);
    const float d10 = dot3f4(bi.ay, bj.ax);
    const float d11 = dot3f4(bi.ay, bj.ay);

    const float cscale_ij =
        pp.sigma * bi.invL * bj.invL;

    // ui . uj
    const float2 Cpp = cscale(
        (float2)(
            d00 - d11,
            d01 + d10
        ),
        cscale_ij
    );

    // ui . conj(uj)
    const float2 Cpm = cscale(
        (float2)(
            d00 + d11,
           -d01 + d10
        ),
        cscale_ij
    );

    const float2 A6 = cpow6(A);
    const float2 B6 = cpow6(B);

    const float2 Fpp = moment66(A, B, Cpp);
    const float2 Fpm = moment66(A, cconj(B), Cpm);

    float H = bi.c0 * bj.c0;

    H += bj.c0 * creal_mul(bi.h6, A6);
    H += bi.c0 * creal_mul(bj.h6, B6);

    const float2 hij_pp = cmul(bi.h6, bj.h6);
    const float2 hij_pm = cmul(bi.h6, cconj(bj.h6));

    H += 0.5f * (
          creal_mul(hij_pp, Fpp)
        + creal_mul(hij_pm, Fpm)
    );

    // One transcendental operation for the complete molecular pair.
    const float radial =
        pp.prefactor * native_exp(-pp.mu * R2);

    return radial * H;
}
```

For better accuracy, replace `native_exp` by `exp`. The rest of the calculation remains unchanged.

---

# 11. Actual computational cost

For one molecular pair:

### Transcendentals

[
\boxed{1\text{ exponential}}
]

and

[
\boxed{0\text{ square roots},\quad
0\text{ trigonometric functions},\quad
0\text{ atan2}.}
]

### Geometric operations

* four dot products involving (\mathbf R);
* four dot products between molecular frame vectors;
* one (R^2) dot product.

### Polynomial operations

* two complex sixth powers;
* two specialized (F_6) recurrences;
* several final complex contractions.

This is roughly a few hundred elementary floating-point operations plus one exponential. More importantly, it replaces:

* (36) atom-pair exponentials for two six-atom rings;
* or approximately (6) field evaluations in your (O(N)) sampling formulation;

by one exponential.

Its complexity is not really (O(p^2)) in the algorithmic sense because (p=3) is hard-wired:

[
\boxed{O(1)}
]

per molecular pair.

---

# 12. Performance details that matter

## Cache the molecular frame

Do not rotate (\mathbf e_x,\mathbf e_y) from the quaternion inside every pair interaction.

Once per molecule per simulation step, calculate

[
\mathbf a_i=Q(q_i)\mathbf e_x,
\qquad
\mathbf b_i=Q(q_i)\mathbf e_y.
]

Store them in aligned `float4` arrays.

The pair kernel should only load `ax` and `ay`.

## Precompute pair-type constants

For every pair of molecule types, precompute

[
\mu,\quad
\lambda_i,\quad
\lambda_j,\quad
\sigma,\quad
\left(\frac{\pi}{p}\right)^{3/2}.
]

This eliminates divisions, square roots, and general powers from the runtime kernel.

## Unroll degree six

Do not write a generic loop over (n). The sixfold recurrence should be explicitly unrolled as above. This gives the compiler a small fixed dependency graph and avoids indexable temporary arrays.

## Keep variables dimensionless

Use

[
A\sim\frac{\mathbf u_i\cdot\mathbf R}{\ell_i},
\qquad
C\sim\frac{\sigma}{\ell_i\ell_j}.
]

A sensible choice is

[
\ell_i\sim R_{\mathrm{ring},i}.
]

Then (A,B,C) remain around order unity near contact, avoiding overflow or loss of precision from sixth powers.

## Four-multiply complex multiplication

On modern CPUs and GPUs, the ordinary four-real-multiply complex product is generally appropriate because it maps well to fused multiply-add instructions and has a short dependency chain. The three-multiply trick saves one multiplication but introduces additional adds and dependencies; it should only be used after benchmarking on the target GPU.

## Scan with fixed orientations

For AFM-like scanning where (q_i,q_j) remain fixed while (\mathbf R) changes over a grid, precompute

[
C_{++},\qquad C_{+-}
]

once. Then the per-grid-point work contains only the projections of (\mathbf R), the polynomial moments, and one exponential.

---

# 13. Choosing the Gaussian width

The magnitude of the sixfold term in the molecular plane behaves as

[
r_\perp^6e^{-\alpha r^2}.
]

Its radial maximum satisfies

[
\frac{d}{dr}
\left(r^6e^{-\alpha r^2}\right)=0,
]

giving

[
\boxed{
r_{\max}=\sqrt{\frac{3}{\alpha}}.
}
]

Therefore, for a desired protrusion radius (R_0), a natural first choice is

[
\boxed{
\alpha=\frac{3}{R_0^2}.
}
]

Then use

[
\ell=R_0.
]

The remaining parameters (c_0) and (h_6) control the average repulsion and corrugation strength.

For benzene, orient (\mathbf a) toward one carbon atom. Then

[
\Re[(\mathbf u\cdot\mathbf y)^6]
]

has maxima along all six carbon directions automatically.

---

# 14. What this approximation captures—and what it does not

The model captures exactly the intended essence:

* one molecular center;
* a broad spherical envelope;
* six equatorial protrusions;
* arbitrary 3D tilt;
* arbitrary in-plane rotation;
* smooth translation dependence.

It does not attempt to reproduce six isolated atomic density peaks. The angular shape is

[
c_0+h_6r_\perp^6\cos6\phi,
]

so the six atoms appear as six smooth enhancements of a common envelope.

For a planar molecule, the largest missing feature is not additional sixfold detail but the difference between face-on and edge-on thickness. The smallest tailored extension would be one axisymmetric quadrupole,

[
Q_2(\mathbf y)
==============

3(\mathbf n\cdot\mathbf y)^2-y^2,
]

giving the four-channel descriptor

[
\boxed{
{1,Y_{20},\Re Y_{66},\Im Y_{66}}.
}
]

That remains a fixed tiny basis, not a general expansion. But the first implementation should be the three-channel model above: it directly tests whether one monopole plus one (C_6) mode provides sufficient accuracy for the intended speed advantage.

[1]: https://dlmf.nist.gov/14.30?utm_source=chatgpt.com "14.30 Spherical and Spheroidal Harmonics"
[2]: https://dlmf.nist.gov/18.9?utm_source=chatgpt.com "DLMF: §18.9 Recurrence Relations and Derivatives ‣ ..."


