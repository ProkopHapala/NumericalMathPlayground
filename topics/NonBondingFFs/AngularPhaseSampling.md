https://chatgpt.com/share/6a60ae77-4c60-83eb-95ec-5eac7c45dae5

---

# USER

I have often problem how to sample efficiently molecules around some other molecule when computing DFT reference, the radial spacing is easy, I can use some logaritmic soft-min like effective distance, but quation is sampling along the isolines of distance. I was thinking maybe some complex-valued functon analysis can help, I do not want to integrate lengh of isoline numerically (I can do that but it is last resolr, I hope there is some elagant math how to do that instead ), look on this code and description

---

# Chat GPT 5.6 sol hi

Your intuition about constructing a second “phase” coordinate is good, but there is an important limitation:

> **Complex analysis can give an elegant phase coordinate, but for a general molecular distance field it cannot give equal-arc-length sampling without solving an equivalent numerical problem.**

Fortunately, there is a much simpler geometric algorithm that avoids contour extraction and explicit integration of contour length: **particles constrained to the level set, redistributed only in the tangential direction**.

## 1. First correct the distance field

The present expression is

[
u(\mathbf x)=
-\frac1\beta
\log\sum_j
\exp[-\beta r_j-R_j].
]

This is not a soft minimum of (r_j-R_j). Algebraically, it is a soft minimum of

[
r_j+\frac{R_j}{\beta}.
]

Consequently:

* (R_j) has the wrong scaling.
* Larger radii suppress the influence of an atom instead of pushing its surface outward.
* (-\beta r_j) is dimensionless, while (-R_j) is not.
* Changing (\beta) changes the effective atomic radii.

This contradicts the stated interpretation of the zero isoline as a vdW surface. 

Use

[
d_j(\mathbf x)=|\mathbf x-\mathbf r_j|-R_j,
]

[
\boxed{
u_\beta(\mathbf x)
==================

-\frac1\beta\log
\sum_j e^{-\beta d_j(\mathbf x)}
}
]

and therefore change the code to

```python
exponents = -beta * (r - Rvdw[None, None, :])
```

instead of

```python
exponents = -beta * r - Rvdw[None, None, :]
```

The existing gradient formula then remains structurally correct, with the corrected weights. 

Define

[
p_j=\frac{e^{-\beta d_j}}{\sum_k e^{-\beta d_k}},
\qquad
\mathbf n_j=\frac{\mathbf x-\mathbf r_j}{r_j}.
]

Then

[
\nabla u_\beta=\sum_j p_j\mathbf n_j,
\qquad
|\nabla u_\beta|\le 1.
]

One additional correction: the unnormalized soft minimum does **not** approach the average as (\beta\to0):

[
u_\beta
=======

-\frac{\log N}{\beta}
+\frac1N\sum_j d_j+O(\beta).
]

It diverges because of the (-\log N/\beta) term. A log-mean-exp,

[
-\frac1\beta\log\left[\frac1N\sum_j e^{-\beta d_j}\right],
]

does approach the average, but introduces a different additive offset. For your application, I would keep the ordinary soft minimum and treat the physical contact shell as a calibrated level (u=c_0), rather than insisting that (u=0) have an exact meaning for every molecule.

## 2. Why the present angular sampling fails

The code starts points at uniform polar angles near the molecular center, and then follows normalized gradient trajectories. 

Gradient trajectories are not analogous to radial rays:

* Neighboring trajectories can converge strongly.
* Different initial angles can reach nearly the same part of a contour.
* Critical points with (\nabla u=0) can block or separate trajectories.
* The molecular center itself is frequently close to a symmetry-induced critical point.
* Uniform angular density at the center is not preserved by gradient flow.

There is also no reason to interpolate a precomputed grid with nearest-neighbor lookup. Since (u) and (\nabla u) are analytic sums over atoms, evaluating them directly at perhaps 20–100 sampling points is much cheaper and more accurate than computing the entire dense grid.

## 3. Can complex analysis provide the phase?

### A harmonic construction

Suppose you replace the soft-distance field by

[
U(z)=\sum_j q_j\log|z-z_j|.
]

This is the real part of

[
W(z)=\sum_j q_j\log(z-z_j)
=U(z)+iV(z),
]

with

[
V(z)=\sum_j q_j\arg(z-z_j).
]

Equivalently, for integer (q_j),

[
e^{W(z)}
========

# F(z)

C\prod_j(z-z_j)^{q_j}.
]

Then

[
U=\log|F|,\qquad V=\arg F.
]

The isolines are generalized polynomial lemniscates,

[
|F(z)|=\rho,
]

and (V) gives an exact phase coordinate. This is closely related to conformal maps between exteriors of compact sets, circles, and generalized lemniscatic domains. ([Springer Link][1])

This is mathematically beautiful. You can even find points by solving

[
F(z)=\rho e^{i\phi_k}.
]

But it has three problems for your application.

First, (U) is a logarithmic potential, not approximately the Euclidean distance from a union of vdW spheres.

Second, individual radii cannot be inserted simply as offsets. In

[
\sum_jq_j\log\frac{|z-z_j|}{R_j},
]

the (R_j) terms contribute only a global constant.

Third—and most importantly—uniform phase is not uniform arc length.

Along an isoline,

[
F(z)=\rho e^{i\phi}.
]

Differentiating,

[
F'(z)\frac{dz}{d\phi}=iF(z),
]

so

[
\boxed{
\frac{ds}{d\phi}
================

# \left|\frac{dz}{d\phi}\right|

\frac{\rho}{|F'(z)|}
}
]

which generally varies strongly around the contour.

Therefore equal increments in (\phi) give **harmonic-measure sampling**, not equal-length sampling. Arc length contains (|F'(z)|), which is non-holomorphic; residue calculus does not remove this modulus. You would still need to integrate or otherwise redistribute the points.

### Why your soft-min field has no harmonic conjugate

A phase (V) satisfying the Cauchy–Riemann equations exists only when (u) is harmonic:

[
\Delta u=0.
]

For your planar soft-min field,

[
\Delta u
========

## \sum_j\frac{p_j}{r_j}

\beta\left(1-|\nabla u|^2\right),
]

away from atom centers. This is not zero in general. Therefore there is no ordinary analytic function

[
f(z)=u(x,y)+iV(x,y)
]

whose real part is your molecular distance field.

## 4. A useful impossibility result

Suppose we seek a phase (\phi) whose coordinate lines are orthogonal to the (u)-isolines. In two dimensions this requires

[
\nabla\phi=a(\mathbf x),J\nabla u,
]

where

[
J(x,y)=(-y,x).
]

For this vector field to be a gradient, its curl must vanish:

[
\nabla\cdot(a\nabla u)=0.
]

Now require equal normalized arc length on every shell:

[
\frac{d\phi}{ds}
================

\frac{2\pi}{L(u)}=C(u).
]

Because the unit tangent is

[
\mathbf t=\frac{J\nabla u}{|\nabla u|},
]

we have

[
\frac{d\phi}{ds}
================

# \nabla\phi\cdot\mathbf t

a|\nabla u|.
]

Thus

[
a=\frac{C(u)}{|\nabla u|}.
]

Substituting into the integrability condition gives

[
0=
\nabla\cdot\left(C(u)\frac{\nabla u}{|\nabla u|}\right)
=======================================================

C'(u)|\nabla u|+C(u)\kappa,
]

where

[
\kappa=\nabla\cdot\frac{\nabla u}{|\nabla u|}
]

is the curvature of the isoline. Therefore

[
\boxed{
\frac{\kappa}{|\nabla u|}
=========================

-\frac{C'(u)}{C(u)}
}
]

would have to be constant along each isoline.

That is not true for a general molecular shape. Consequently:

> A globally orthogonal coordinate ((u,\phi)) that is simultaneously equal-arc-length on every isoline generally does not exist.

You may have either:

* a phase transported consistently between shells, or
* independently normalized arc length on every shell,

but generally not both exactly.

## 5. The practical elegant solution: tangential relaxation

You do not need the total length (L), contour extraction, or cumulative integration.

Let (\mathbf x_i), (i=0,\ldots,N-1), be cyclically ordered points near the level set

[
u(\mathbf x)=c.
]

### Projection onto the isoline

Newton projection in the normal direction is

[
\boxed{
\mathbf x\leftarrow
\mathbf x-
\frac{u(\mathbf x)-c}
{|\nabla u(\mathbf x)|^2}
\nabla u(\mathbf x)
}
]

Repeated two or three times, this places a point accurately on the isoline. For an exact signed-distance field, one iteration would be enough.

### Equalize the parameterization tangentially

Define

[
\mathbf n_i=
\frac{\nabla u(\mathbf x_i)}
{|\nabla u(\mathbf x_i)|},
]

and the discrete second difference

[
\mathbf q_i=
\mathbf x_{i-1}-2\mathbf x_i+\mathbf x_{i+1}.
]

Remove its normal component:

[
\mathbf q_i^\parallel
=====================

\mathbf q_i-
\mathbf n_i
(\mathbf n_i\cdot\mathbf q_i).
]

Update

[
\mathbf x_i\leftarrow
\mathbf x_i+\eta\mathbf q_i^\parallel
]

and project back onto (u=c).

Why this works becomes clear in the continuum. For a parameterized curve (\mathbf x(\alpha)),

[
\mathbf x_{\alpha\alpha}
========================

\kappa|\mathbf x_\alpha|^2\mathbf n
+
\partial_\alpha|\mathbf x_\alpha|,\mathbf t.
]

Its tangential component is

[
\left(\mathbf x_{\alpha\alpha}\right)_\parallel
===============================================

\partial_\alpha|\mathbf x_\alpha|,\mathbf t.
]

Therefore tangential diffusion drives

[
\partial_\alpha|\mathbf x_\alpha|\to0,
]

which means constant-speed—and hence approximately equal-arc-length—parameterization.

This is essentially the implicit-surface “floater particle” idea: constrain particles to a level set and let tangential forces distribute them evenly. It has long been used for uniform sampling of implicit geometry and generalizes directly to 3D surfaces. ([Robotics Institute Publications Archive][2])

A minimal implementation is:

```python
import numpy as np


def eval_rmol(xy, apos, rvdw, beta, z_height=0.0):
    """Evaluate corrected soft molecular distance and xy gradient.

    xy:    (M, 2)
    apos:  (N, 3)
    rvdw:  (N,)
    """
    xy = np.asarray(xy, dtype=float)
    apos = np.asarray(apos, dtype=float)
    rvdw = np.asarray(rvdw, dtype=float)

    dxy = xy[:, None, :] - apos[None, :, :2]       # (M,N,2)
    dz = z_height - apos[None, :, 2]               # (1,N)
    rr = np.sqrt(np.sum(dxy*dxy, axis=2) + dz*dz)  # (M,N)
    rr_safe = np.maximum(rr, 1.0e-12)

    # Correct geometric offset: d_j = r_j - R_j
    exponents = -beta * (rr - rvdw[None, :])

    amax = np.max(exponents, axis=1, keepdims=True)
    weights = np.exp(exponents - amax)
    zsum = np.sum(weights, axis=1, keepdims=True)
    probs = weights / zsum

    u = -(amax[:, 0] + np.log(zsum[:, 0])) / beta

    atom_grad = dxy / rr_safe[:, :, None]
    grad = np.sum(probs[:, :, None] * atom_grad, axis=1)

    return u, grad


def project_to_level(
    points,
    level,
    eval_field,
    n_iter=4,
    max_step=0.5,
):
    """Normal Newton projection onto eval_field(x)[0] == level."""
    x = np.asarray(points, dtype=float).copy()

    for _ in range(n_iter):
        u, grad = eval_field(x)
        gg = np.sum(grad*grad, axis=1)

        dx = -((u - level) / np.maximum(gg, 1.0e-14))[:, None] * grad

        # Prevent a point near a critical point from making a huge jump.
        dxnorm = np.linalg.norm(dx, axis=1)
        scale = np.minimum(
            1.0,
            max_step / np.maximum(dxnorm, 1.0e-14),
        )
        x += dx * scale[:, None]

    return x


def equalize_level_points(
    points,
    level,
    eval_field,
    n_iter=200,
    eta=0.2,
    tolerance=1.0e-3,
):
    """Equal-chord/equal-arclength relaxation on a closed 2D level set.

    Input points must already be in cyclic order.
    """
    x = project_to_level(points, level, eval_field)

    for _ in range(n_iter):
        _, grad = eval_field(x)
        gnorm = np.linalg.norm(grad, axis=1)
        normal = grad / np.maximum(gnorm[:, None], 1.0e-14)

        lap = (
            np.roll(x, +1, axis=0)
            - 2.0*x
            + np.roll(x, -1, axis=0)
        )

        # Only tangential redistribution; do not deform the contour.
        lap_tangent = lap - normal * np.sum(
            normal * lap, axis=1, keepdims=True
        )

        x += eta * lap_tangent
        x = project_to_level(x, level, eval_field, n_iter=2)

        gaps = np.linalg.norm(np.roll(x, -1, axis=0) - x, axis=1)
        relative_spread = np.std(gaps) / max(np.mean(gaps), 1.0e-14)

        if relative_spread < tolerance:
            break

    return x
```

For your molecule:

```python
eval_field = lambda xy: eval_rmol(
    xy,
    mol.apos,
    get_vdw_radii(mol),
    beta,
)

points = equalize_level_points(
    initial_points,
    r_sample,
    eval_field,
)
```

The initial points can still come from angular rays. They only need to have the correct cyclic ordering; the relaxation removes the bad angular density.

## 6. Transport corresponding points between radial shells

To preserve an angular correspondence between neighboring shells, transport each point using

[
\boxed{
\frac{d\mathbf x}{dc}
=====================

\frac{\nabla u}{|\nabla u|^2}
}
]

because

[
\frac{du}{dc}
=============

\nabla u\cdot\frac{d\mathbf x}{dc}=1.
]

For a step (\Delta c),

[
\mathbf x(c+\Delta c)
\approx
\mathbf x(c)
+
\Delta c
\frac{\nabla u}{|\nabla u|^2},
]

followed by Newton projection and a few tangential relaxation iterations.

This gives a very natural workflow:

1. Construct one well-distributed outer shell.
2. Transport it to the next radial level.
3. Project to the exact level.
4. Relax tangentially.
5. Repeat inward.

That produces a coherent grid

[
(c_m,\phi_i)\longmapsto\mathbf x_{mi}
]

without defining (\phi) analytically.

Topology changes still require special handling. A regular level set cannot self-intersect; self-intersection, splitting, or merging can occur only at critical levels where (\nabla u=0). The concern in the description is therefore better stated as “multiple components and topology changes,” not generic self-intersection. 

## 7. Equal arc length may not be the optimal metric

Because you orient the probe along the surface normal, two nearby points differ in both position and orientation.

Along a planar contour,

[
\frac{d\theta_{\rm normal}}{ds}=\kappa(s).
]

A geometrically meaningful configuration-space metric is therefore

[
d\sigma^2
=========

\frac{ds^2}{\ell_{\rm pos}^2}
+
\frac{d\theta_{\rm normal}^2}{\ell_{\rm angle}^2}.
]

Using (d\theta=\kappa ds),

[
\boxed{
\frac{d\sigma}{ds}
==================

\sqrt{
\frac1{\ell_{\rm pos}^2}
+
\frac{\kappa(s)^2}{\ell_{\rm angle}^2}
}
}
]

Thus points should be denser where the normal rotates rapidly. This is likely more efficient for your DFT reference set than pure equal arc length.

For example:

* (\ell_{\rm pos}=0.5) Å means you tolerate roughly 0.5 Å translational changes.
* (\ell_{\rm angle}=15^\circ) means you tolerate roughly (15^\circ) changes in probe orientation.

The target spacing becomes

[
\Delta s(s)
\propto
\left[
\ell_{\rm pos}^{-2}
+
\kappa(s)^2\ell_{\rm angle}^{-2}
\right]^{-1/2}.
]

You can implement this by making the equilibrium spring length position-dependent.

The Hessian of the soft-min field is available analytically:

[
H
=

\sum_j p_j
\frac{I-\mathbf n_j\mathbf n_j^T}{r_j}
--------------------------------------

\beta
\left[
\sum_jp_j\mathbf n_j\mathbf n_j^T
---------------------------------

\mathbf g\mathbf g^T
\right],
]

where (\mathbf g=\nabla u). In 2D,

[
\kappa
======

\frac{\mathbf t^TH\mathbf t}{|\mathbf g|}.
]

So curvature-based sampling need not involve finite differences.

## 8. Two smaller code problems

Your nearest-grid-point interpolation produces staircase errors and makes the final position dependent on grid spacing. Direct analytic evaluation plus Newton projection is both simpler and more accurate. 

There is also a rotation bug in the exactly antiparallel case. The code selects a perpendicular axis and sets `rot_axis_norm = 1.0`; later it uses

```python
sin_angle = rot_axis_norm
```

so a (180^\circ) rotation incorrectly gets (\sin\theta=1), whereas it must be zero. 

In that branch, use

```python
cos_angle = -1.0
sin_angle = 0.0
```

after selecting the perpendicular axis.

## Recommendation

I would not replace your distance field with a complex logarithmic potential. The complex construction is mathematically elegant but changes the meaning of radial distance and still does not provide equal arc length.

The strongest practical formulation is:

[
\boxed{
\text{soft molecular distance}
+
\text{Newton level-set projection}
+
\text{tangential particle relaxation}
}
]

with gradient-flow transport between radial levels. It avoids contour meshing, avoids computing total contour length, works with concavities, and generalizes naturally to 3D by replacing the 1D neighbor relaxation with tangent-plane particle repulsion. Constrained particles with local repulsion are a well-established approach for uniform implicit-surface sampling; stochastic constrained motion is another mathematically clean alternative, though likely unnecessary for your deterministic DFT grid. ([Robotics Institute Publications Archive][2])

[1]: https://link.springer.com/article/10.1007/s40315-022-00462-4?utm_source=chatgpt.com "Walsh’s Conformal Map onto Lemniscatic Domains for Polynomial Pre-images I | Computational Methods and Function Theory | Springer Nature Link"
[2]: https://publications.ri.cmu.edu/using-particles-to-sample-and-control-implicit-surfaces?utm_source=chatgpt.com "Using particles to sample and control implicit surfaces - Robotics Institute Carnegie Mellon University"

---

# USER

OK, this tangential relaxation is probably good, but I'm still thinking about this analytical phase. 

if we have n-atom anclosed by our controu-line it is something like topological winding number (chren topological invariant right?). Tot total phase change when going around that circumference is n*2*pi right?  I see that on the HSV pots as the Hue go around the rainbow n-times where n is number of atoms f the molecule. Then we map it on checkberboard. I see clearly that if we just put the atoms in the vertexes of the checkerboard it would be sampled very nicely.

Or at least it is good gobal initial gues. Then we can do one ort two iterations of local refinement (like average distance from nearest points on the same controur and re-project distance to correct effective distance form molecule computed using that exp-averaging)

please elaborate on this idea especially how to distibrute the angular (phi) samples accoring to the checkerboard

 Look on the code wech we already implemented based on your ideas but olso ilustrating the checkerboard

 ---

 # Chat GPT 5.6 sol hi

 Yes—your topological intuition is essentially correct, and it leads to a useful construction. The most promising approach is not to replace your molecular distance by the complex potential, but to use:

[
\boxed{
\text{soft molecular distance }u(\mathbf r)
\quad+\quad
\text{complex topological phase }\Phi(\mathbf r)
}
]

as a mixed coordinate system. Then the sampling points are the vertices of a checkerboard in ((u,\Phi))-space.

## 1. The total phase change is (2\pi n)

Define

[
F(z)=\prod_{j=1}^{N}(z-z_j)^{q_j},
]

where (z=x+iy), (z_j=x_j+iy_j), and (q_j) are integer multiplicities. Usually (q_j=1).

The phase is

[
\Phi(z)=\arg F(z)
=\sum_j q_j\arg(z-z_j).
]

For a closed contour (C),

[
\boxed{
\frac{1}{2\pi}\oint_C d\Phi
===========================

\frac{1}{2\pi i}
\oint_C\frac{F'(z)}{F(z)},dz
============================

\sum_{z_j\text{ inside }C}q_j
}
]

Therefore, if the contour encloses (n) atoms and every atom has (q_j=1),

[
\Delta\Phi=2\pi n.
]

So yes: when traversing the contour once, HSV hue goes through the complete rainbow (n) times.

### Is this a Chern invariant?

It is more precisely:

* the winding number,
* the degree of the map (C\simeq S^1\rightarrow U(1)\simeq S^1),
* the argument principle,
* or the total vortex charge.

It is closely related to the mathematics underlying Chern numbers, but it is not itself normally called a Chern number. A Chern number is usually an integral of curvature over a two-dimensional closed manifold. Here you have a one-dimensional phase winding:

[
\pi_1(U(1))=\mathbb Z.
]

You can think of it as the one-dimensional topological ancestor of the Chern construction.

Each atom is a phase vortex carrying charge (q_j). At the atom,

[
F(z_j)=0,
]

and the phase winds by (2\pi q_j) around it.

## 2. What the checkerboard really represents

Introduce the complex logarithm locally:

[
W(z)=\log F(z)=U(z)+i\Phi(z),
]

where

[
U(z)=\log|F(z)|
=\sum_j q_j\log|z-z_j|.
]

In the (W=U+i\Phi) plane, draw an ordinary rectangular grid:

[
U=U_m,
\qquad
\Phi=\Phi_l.
]

Mapping that grid backward through (F) produces the distorted checkerboard in the physical (x,y) plane.

Its vertices satisfy

[
U(z)=U_m,
\qquad
\Phi(z)=\Phi_l,
]

or equivalently

[
\boxed{
F(z)=\exp(U_m+i\Phi_l).
}
]

For a polynomial of degree

[
Q=\sum_jq_j,
]

each checkerboard vertex in the (F)-plane generally has (Q) preimages in the molecular plane. Those are the (Q) repeated copies of each hue around the molecule.

The atoms themselves are not regular checkerboard vertices. They are singular branching points where

[
U\rightarrow-\infty
]

and all phase directions converge. They behave more like topological defects or vortices in the checkerboard.

## 3. How many angular samples does the checkerboard generate?

Suppose you divide one hue cycle (0\leq\Phi<2\pi) into (K) phase intervals:

[
\Phi_l=\Phi_0+\frac{2\pi l}{K},
\qquad l=0,\ldots,K-1.
]

Because the phase winds (Q) times around the complete contour, every phase value occurs (Q) times.

Therefore the total number of points on the contour is

[
\boxed{M=KQ.}
]

This is important for interpreting your current visualization. The code uses

```python
n_phase_bands = 16
```

for the wrapped HSV phase. 

For HCOOH with all five atoms assigned (q_j=1),

[
Q=5,
]

so the enclosing contour contains

[
M=16\times5=80
]

phase sectors, not 16.

That is likely why the visual checkerboard looks much denser than the requested `--n-points 24`.

### Natural sample counts

For a strict checkerboard, choose

[
M=KQ.
]

For example, with all five HCOOH atoms:

* (K=4\rightarrow M=20)
* (K=5\rightarrow M=25)
* (K=6\rightarrow M=30)
* (K=8\rightarrow M=40)

Alternatively, use only three heavy atoms as phase centers:

[
q_{\rm C}=q_{\rm O_1}=q_{\rm O_2}=1,
\qquad
q_{\rm H}=0.
]

Then (Q=3), and

[
K=8\rightarrow M=24.
]

That is particularly convenient for your current default of 24 points.

This is not merely numerical convenience. The (q_j) define which molecular features receive angular topological weight. You could use:

* all atoms;
* heavy atoms only;
* chemically active atoms;
* selected lone-pair or bond-direction sites;
* ring centers or functional-group centers;
* integer multiplicity (q_j>1) for especially important sites.

The phase centers need not literally be atomic nuclei. They can be **sampling landmarks**.

## 4. The best checkerboard for your purpose is ((u_{\rm mol},\Phi))

You do not need to use (U=\log|F|) as the radial coordinate.

Use your physically meaningful soft distance

[
u(\mathbf r)
============

-\frac1\beta
\log\sum_j
e^{-\beta(|\mathbf r-\mathbf r_j|-R_j)}
]

as the radial coordinate, and use

[
\Phi(\mathbf r)
===============

\sum_jq_j\arg(z-z_j)
]

as the angular coordinate.

Then the checkerboard vertices are solutions of

[
\boxed{
u(\mathbf r)=c_m,
\qquad
\Phi(\mathbf r)=\Psi_k\pmod{2\pi}.
}
]

Here:

* (c_m) is one of your physical distance shells;
* (\Psi_k) is an angular phase level.

This is better than first sampling a (|F|=\rho) lemniscate and then projecting it onto the molecular distance contour. You solve for the intersection of the desired molecular contour and the desired phase line directly.

The coordinates are no longer holomorphic because (u\neq U), but the angular coordinate retains its exact topological winding.

## 5. Sampling an arbitrary number (M)

A wrapped hue contains only (\Phi\bmod2\pi), but the phase accumulated around the complete molecular contour is

[
2\pi Q.
]

Define an unwrapped phase coordinate (\Psi) along the contour, with

[
\Psi\in[\Psi_0,\Psi_0+2\pi Q).
]

For exactly (M) samples, choose

[
\boxed{
\Psi_k
======

\Psi_0+\frac{2\pi Q,k}{M},
\qquad
k=0,\ldots,M-1.
}
]

This works even when (M) is not divisible by (Q).

For (Q=5), (M=24),

[
\Delta\Psi=\frac{10\pi}{24}=\frac{5\pi}{12}.
]

The wrapped hues repeat, but continuation along the contour keeps track of which phase branch each point belongs to.

This is analogous to walking through the checkerboard continuously. After the wrapped phase advances by (2\pi), you do not return to the same spatial point: you arrive at the corresponding point on the next branch. Only after (Q) such phase cycles do you return to the starting point.

That branch permutation is the geometric manifestation of the winding number.

## 6. Local differential geometry of the phase

You do not need to evaluate the potentially huge product (F). Define

[
S(z)=\frac{F'(z)}{F(z)}
=\sum_j\frac{q_j}{z-z_j}.
]

Write

[
S=A+iB.
]

Then

[
\nabla U=(A,-B),
]

and

[
\boxed{
\nabla\Phi=(B,A).
}
]

For one atom at the origin,

[
S=\frac1z=\frac{x-iy}{x^2+y^2},
]

and therefore

[
\nabla\Phi
==========

\frac{1}{x^2+y^2}(-y,x),
]

as expected for the angular coordinate.

A stable implementation is:

```python
def eval_complex_phase(xy, apos, charges=None):
    """Unwrapped local phase value and analytic phase gradient.

    xy:      (M,2)
    apos:    (N,2) or (N,3)
    charges: integer q_j; default all one
    """
    xy = np.asarray(xy, dtype=float)
    apos2 = np.asarray(apos, dtype=float)[:, :2]

    if charges is None:
        charges = np.ones(len(apos2), dtype=float)
    else:
        charges = np.asarray(charges, dtype=float)

    dxy = xy[:, None, :] - apos2[None, :, :]
    dx = dxy[:, :, 0]
    dy = dxy[:, :, 1]

    # Sum of local angles. It may contain branch jumps, but wrapped
    # phase differences remain well defined.
    phi_raw = np.sum(
        charges[None, :] * np.arctan2(dy, dx),
        axis=1,
    )

    dz = dx + 1j*dy
    dz = np.where(np.abs(dz) > 1e-14, dz, 1e-14 + 0j)

    S = np.sum(charges[None, :] / dz, axis=1)

    # ∇Phi = (Im(F'/F), Re(F'/F))
    grad_phi = np.column_stack((S.imag, S.real))

    return phi_raw, grad_phi
```

## 7. Trace equal-phase points around the actual contour

Let

[
\mathbf g=\nabla u,
\qquad
\mathbf n=\frac{\mathbf g}{|\mathbf g|},
\qquad
\mathbf t=J\mathbf n=(-n_y,n_x).
]

Along the contour (u=c),

[
\frac{d\Phi}{ds}
================

\nabla\Phi\cdot\mathbf t.
]

Therefore a desired phase step (\Delta\Psi) corresponds locally to

[
\boxed{
\Delta s
\approx
\frac{\Delta\Psi}
{\nabla\Phi\cdot\mathbf t}.
}
]

This gives a predictor:

[
\mathbf x_{k+1}^{\rm pred}
==========================

\mathbf x_k
+
\frac{\Delta\Psi}
{\nabla\Phi\cdot\mathbf t}
\mathbf t.
]

Then correct the predicted point by solving the two equations

[
u(\mathbf x)=c,
\qquad
\Phi(\mathbf x)=\Psi_{k+1}\pmod{2\pi}.
]

The Newton Jacobian is simply

[
J=
\begin{pmatrix}
\nabla u^T\
\nabla\Phi^T
\end{pmatrix}.
]

A compatible implementation is:

```python
def wrap_phase(phi):
    return np.arctan2(np.sin(phi), np.cos(phi))


def correct_level_and_phase(
    x,
    level,
    target_phase,
    eval_field,
    apos,
    charges=None,
    n_iter=4,
    max_step=0.5,
):
    """Solve u(x)=level and Phi(x)=target_phase modulo 2π."""
    x = np.asarray(x, dtype=float).copy()

    for _ in range(n_iter):
        u, grad_u = eval_field(x[None, :])
        phi, grad_phi = eval_complex_phase(
            x[None, :],
            apos,
            charges,
        )

        residual = np.array([
            u[0] - level,
            wrap_phase(phi[0] - target_phase),
        ])

        jacobian = np.vstack([
            grad_u[0],
            grad_phi[0],
        ])

        det = np.linalg.det(jacobian)
        if abs(det) < 1e-10:
            break

        dx = np.linalg.solve(jacobian, -residual)

        dxnorm = np.linalg.norm(dx)
        if dxnorm > max_step:
            dx *= max_step / dxnorm

        x += dx

    return x


def sample_level_by_complex_phase(
    seed,
    level,
    n_points,
    eval_field,
    apos,
    charges=None,
):
    """Trace one complete u=level contour using uniform unwrapped phase."""
    apos = np.asarray(apos, dtype=float)

    if charges is None:
        charges = np.ones(len(apos), dtype=int)
    else:
        charges = np.asarray(charges, dtype=int)

    winding = int(np.sum(charges))
    if winding <= 0:
        raise ValueError("Total phase winding must be positive")

    # Put seed exactly on the requested molecular level.
    x0 = project_to_level(
        np.asarray(seed)[None, :],
        level,
        eval_field,
        n_iter=8,
    )[0]

    phi0, _ = eval_complex_phase(
        x0[None, :],
        apos,
        charges,
    )
    phi0 = phi0[0]

    dphi = 2.0*np.pi*winding/n_points

    points = np.empty((n_points, 2), dtype=float)
    points[0] = x0

    for k in range(1, n_points):
        x = points[k - 1]

        _, grad_u = eval_field(x[None, :])
        _, grad_phi = eval_complex_phase(
            x[None, :],
            apos,
            charges,
        )

        normal = grad_u[0]
        normal /= max(np.linalg.norm(normal), 1e-14)

        tangent = np.array([-normal[1], normal[0]])
        phase_rate = np.dot(grad_phi[0], tangent)

        # Choose the orientation in which phase increases.
        if phase_rate < 0.0:
            tangent = -tangent
            phase_rate = -phase_rate

        if phase_rate < 1e-8:
            # Degenerate phase coordinate: use a small geometric predictor.
            predictor = x + 0.1*tangent
        else:
            ds = dphi/phase_rate
            ds = np.clip(ds, -1.0, 1.0)
            predictor = x + ds*tangent

        target_phase = phi0 + k*dphi

        points[k] = correct_level_and_phase(
            predictor,
            level,
            target_phase,
            eval_field,
            apos,
            charges,
            n_iter=4,
        )

    return points
```

This is the direct realization of the checkerboard idea. It never computes the total contour length.

## 8. Why equal phase may already give good geometric spacing

Equal phase does not mean equal arc length. Its local spacing is

[
\Delta s_k
\approx
\frac{\Delta\Psi}
{\left|\nabla\Phi\cdot\mathbf t\right|}.
]

Therefore the phase construction automatically puts more points where

[
\left|\nabla\Phi\cdot\mathbf t\right|
]

is large.

Near an atom or a protruding part of the molecule, the angular field changes rapidly, so the checkerboard becomes denser. In sheltered or distant regions, the phase changes more slowly, so points spread farther apart.

This may actually be more useful than equal arc length for DFT sampling. It is a kind of analytically generated adaptive density based on how strongly the atoms are “seen” from the contour.

On a true polynomial lemniscate (U=\log|F|=\mathrm{const}), the Cauchy–Riemann equations give

[
\nabla\Phi=J\nabla U,
]

so

[
\frac{d\Phi}{ds}=|\nabla U|.
]

The phase is strictly monotonic around a regular connected lemniscate. For your mixed contour (u_{\rm mol}=\mathrm{const}), strict monotonicity is not mathematically guaranteed, but it should usually work well on outer connected molecular shells.

## 9. Then apply one or two genuinely local spacing corrections

Your proposed hybrid is very sensible:

1. Generate globally ordered points by equal topological phase.
2. Correct only the strongest local gap inequalities.
3. Reproject onto (u=c).

You no longer need the global cumulative-length algorithm currently implemented in `equalize_level_points`. That routine explicitly computes every segment length, the total contour length, cumulative length, and target positions along the polygon. 

A truly local update is simpler.

Let

[
\ell_i^+=|\mathbf x_{i+1}-\mathbf x_i|,
\qquad
\ell_i^-=|\mathbf x_i-\mathbf x_{i-1}|.
]

Move point (i) tangentially by

[
\boxed{
\delta s_i
==========

\frac{\eta}{2}
\left(\ell_i^+-\ell_i^-\right).
}
]

If the forward gap is larger than the backward gap, the point moves forward, enlarging the backward gap and shrinking the forward gap.

```python
def local_gap_relaxation(
    points,
    level,
    eval_field,
    n_iter=2,
    eta=0.4,
):
    """Purely local equal-gap correction, followed by level projection."""
    x = np.asarray(points, dtype=float).copy()

    for _ in range(n_iter):
        _, grad = eval_field(x)

        normal = grad / np.maximum(
            np.linalg.norm(grad, axis=1)[:, None],
            1e-14,
        )
        tangent = np.column_stack((-normal[:, 1], normal[:, 0]))

        x_next = np.roll(x, -1, axis=0)
        x_prev = np.roll(x, +1, axis=0)

        gap_next = np.linalg.norm(x_next - x, axis=1)
        gap_prev = np.linalg.norm(x - x_prev, axis=1)

        ds = 0.5*eta*(gap_next - gap_prev)

        # Avoid reordering due to an excessive correction.
        max_ds = 0.25*np.minimum(gap_next, gap_prev)
        ds = np.clip(ds, -max_ds, max_ds)

        x += ds[:, None]*tangent

        x = project_to_level(
            x,
            level,
            eval_field,
            n_iter=3,
        )

    return x
```

This matches your idea much more closely:

* global topology and ordering from complex phase;
* local metric correction from neighboring distances;
* exact physical radial placement from the soft distance.

## 10. The current “equal-phase” code is not yet equal-phase

The function `initial_points_on_lemniscate()` computes

```python
phi = 2 * np.pi * i / n_points
```

but never uses `phi`. It instead uses the same index to construct an ordinary geometric ray from the molecular center and stops when (|F|) crosses the target. 

So the present method labeled “Complex harmonic phase” is actually:

[
\text{uniform geometric polar angle}
+
\text{projection onto }|F|=\rho,
]

not equal complex phase.

Likewise, the later Newton refinement corrects only (\log|F|), not the phase. 

There is also a sign error in the gradient of (\log|F|). Since

[
S=\sum_j\frac{q_j}{z-z_j},
]

the correct gradient is

[
\nabla\log|F|
=============

(\operatorname{Re}S,-\operatorname{Im}S),
]

not

[
(\operatorname{Re}S,\operatorname{Im}S).
]

The code currently adds `imag(1/dz)` directly to the (y)-component. 

## 11. Improve the plotted checkerboard

The current radial checker bands are generated from globally normalized (U):

```python
U_norm = (U - U.min()) / (U.max() - U.min())
radial_idx = floor(U_norm * n_radial_bands)
```

Therefore their positions depend on the plotting margin and do not correspond to your physical sampling shells. 

For the hybrid checkerboard, use:

* hue from (\Phi\bmod2\pi);
* phase band index from wrapped (\Phi);
* radial band index from the actual `r_mol` shell levels.

Conceptually:

```python
H = np.mod(phi_grid, 2*np.pi) / (2*np.pi)

phase_idx = np.floor(H * n_phase_bands).astype(int)

radial_idx = np.digitize(
    r_mol,
    ISOLINE_LEVELS,
)

checker = (phase_idx + radial_idx) & 1
```

Then the plotted checkerboard is precisely the coordinate system you intend to sample:

[
(r_{\rm mol},\Phi).
]

Its vertices correspond to your DFT configurations.

## 12. Multi-shell sampling becomes especially elegant

Choose radial levels (c_m) and common unwrapped phases (\Psi_k):

[
u(\mathbf x_{mk})=c_m,
\qquad
\Phi(\mathbf x_{mk})=\Psi_k.
]

The resulting grid has:

* rows: constant molecular distance;
* columns: constant topological phase;
* cells: the physical checkerboard;
* vertices: DFT sampling positions.

Construct the outer shell first by phase continuation. For every inner shell, use the point from the previous shell with the same (\Psi_k) as the initial guess, then solve the same two-equation Newton system.

That maintains a coherent angular identity between shells. A point with index (k) always belongs to the same phase column, rather than being arbitrarily shifted by independent equal-arc-length relaxation.

## Recommended construction

For your H-bond reference sampling, I would use:

[
\boxed{
u_{\rm mol}=c_m,\qquad
\Psi_k=\Psi_0+\frac{2\pi Qk}{M}
}
]

with integer phase charges (q_j) assigned to chemically meaningful landmarks.

Then:

1. Trace the outer contour using phase continuation.
2. Solve ((u,\Phi)=(c_m,\Psi_k)) for every shell.
3. Perform one or two local neighbor-gap corrections.
4. Reproject onto (u=c_m).
5. Orient the probe using (\nabla u).

This preserves the beautiful global/topological structure you see in the HSV checkerboard while correcting its local geometric distortion with only a tiny amount of relaxation.
