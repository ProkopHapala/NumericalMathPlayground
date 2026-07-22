# USER

I want to develop pairwise potential which is cheap short-range, and depending on parameters can transition between multiple use cases smoothly. 

What I want is to have something like Morse or Lenard jones for normal atoms which have binding minimum at some R0 and and I can choose E0 for binging energy. But the functional form should be something simple like polynominal or low degree recimprocal function (fast evaluation matters)

Generlly it can have form like 

F_tot(r) = c1*F_repulsive(r) + c2*F_atractive(r)
where both   F_repulsive(r) and F_atractive(r) have finite cutoff rc

or 
F_tot(r) = fcut(r,rc)*(  c1*F_repulsive(r) + c2*F_atractive(r) )
simplest is 
F_tot(r) = fcut(r,rc)*(  c1*F_repulsive(r) - c2 )
where the atractive potential is bacially c2*fcut(r,rc) and the repulsive is c1*fcut(r,rc)*F_repulsive(r)

What is for me very importnt to have not only the functions but most importantly nice fast mixing rules, that it the best feature of Morse and Lenard-Jones that we have R0ij = Ri + Rj and E0ij = sqrt(Eii,Ejj), I usuually precaclulate and store Ei=sqrt(Eii) and  Ej=sqrt(Ejj) so I do not need sqrt evaluation on the fly when evaluating the forcefield (performance is critical) 

The functions I'm thinking of using

2) polynominal apporximation of exponential
exp(-b*r) ~ lim (1-b*r/n)^n
to efficiently apporximate it with finite curof rc I want to use n=2^m
f(r,m) (1-b*r/(2^m))^(2^m), where rc = 2^m/b
which can be effiicnely obtained as u=1-(b/(2^m))*r; u=u*u; u=u*u ...

 3) polynominal apporximation of gaussin
exp(-b*x^2) = (1-(r/rc)^2)^n
... similarly using n=2^m for efficient evaluation
u=1-(r/rc)^2; u=u*u; u=u*u ...
 

```python
def calculate_energy_and_force(r, rc=1.0):
    # Initialize arrays with zeros
    energy = np.zeros_like(r)
    force = np.zeros_like(r)

    # Create a mask for r < rc
    mask = r < rc
    r_ = r[mask]

    # Calculate energy for r < rc: (1 - (r/rc)^2)^2
    u2 = (r_ / rc)**2
    energy[mask] = (1 - u2)**2

    # Calculate force: F(r) = -d/dr [ (1 - (r/rc)^2)^2 ]
    # F(r) = -2 * (1 - (r/rc)^2) * (-2r / rc^2)
    # F(r) = (4r / rc^2) * (1 - (r/rc)^2)
    force[mask] = (4 * r_ / rc**2) * (1 - u2)

    return energy, force

```

What I also fing quite usefull is Lorenz function as it is like 1/r^2 but with tuneable vertex width.
Lorenz(r,w)=(1/(w*w + r*r)
 But it is long range so I like to multiply it by cutoff function polynominal fcp22(r,rc) = (1-(r/rc)^2)^2

fL22(r,rc,w) = fcp22(r,rc)*Lorenz(r,w)

to be super fast it is also good to have function of only |r|^2 = dot(rij,rij) which does not need square root. Gauss approximation is one such function. Lorentz 1/(w^2+r^2) is anither such function.

Now to dapnden the peak (this I need more for electro pairs where the atom should fall into zero distance form the center - epair and atom shoud doc to be collocated so I want to have smooth parabolic vrtex for r=0 rather than spike). Lorenz is one option. But I can aloso apply some cheap reciplocal modifier to choose it bend down starting from ystart and saturate (assymptitically aproach but never overreach ymax) I call it  soft_clamp(y,y1, y2), I apply it like this
f_clamped(r,ystart,ymax) =  soft_clamp( f(r),ystart, ymax)


Softclamp function
```python
def soft_clamp(y, dy, y1, y2):
    """
    Applies a soft clamp to y, smoothly transitioning values above y1 towards y2.
    Also computes the derivative dy accordingly using the chain rule.

    Parameters:
    - y: np.ndarray, input values to clamp
    - dy: np.ndarray, derivatives of y with respect to some variable x
    - y1: float, lower threshold for clamping
    - y2: float, upper threshold for clamping

    Returns:
    - y_new: np.ndarray, clamped y values
    - dy_new: np.ndarray, updated derivatives
    """
    y_new  = y.copy()
    dy_new = dy.copy()
    mask   = y_new > y1
    y12    = y2 - y1
    invdy  = 1.0 / y12
    z = (y_new[mask] - y1) * invdy
    y_new[mask]   = y1 + y12 * (1 - 1 / (1 + z))
    dy_new[mask] *= 1.0 / (1.0 + z)**2
    return y_new, dy_new```
```

---

### Chat GPT 5.6 sol hi

## Core idea: make the potential quadratic in a cheap “overlap” variable

Morse and Lennard–Jones have the same algebraic structure:

[
V(r)=D\left[y(r)^2-2y(r)\right],
]

with different choices of (y(r)):

[
y_{\rm Morse}=e^{-a(r-R_0)},
\qquad
y_{\rm LJ}=\left(\frac{R_0}{r}\right)^6.
]

In both cases:

[
y(R_0)=1,\qquad V(R_0)=-D,\qquad \frac{dV}{dr}(R_0)=0.
]

Therefore, rather than approximating the whole Morse potential, I suggest replacing only its overlap-like variable (y(r)) by your compact polynomial.

# 1. Compact polynomial Morse potential

Define

[
s=r^2,\qquad
x=\frac{s}{r_c^2},\qquad
u=1-x,
]

and, for (r<r_c),

[
z(r)=u^q
========

\left(1-\frac{r^2}{r_c^2}\right)^q,
]

with (z=0) for (r\ge r_c).

Then use

[
\boxed{
V(r)=C_R z(r)^2-C_A z(r)
}
]

where (C_R) and (C_A) are the repulsive and attractive coefficients.

This has exactly the structure you wanted:

[
F_{\rm repulsive}=z^2,\qquad
F_{\rm attractive}=z.
]

Both have finite support, both depend only on (r^2), and if (q=2^m), they are evaluated by repeated squaring.

The stationary point in terms of (z) is simply

[
z_*=\frac{C_A}{2C_R},
]

and its depth is

[
V_{\min}=-\frac{C_A^2}{4C_R}.
]

Thus the shape changes very smoothly as you vary (C_R) and (C_A):

* (C_R>0,\ C_A=0): purely repulsive;
* (C_R>0,\ C_A>0): shell minimum at (R_0>0);
* (C_R=0,\ C_A>0): purely attractive, with its minimum at (r=0);
* both zero: no interaction.

That is a particularly clean unification of normal atoms, soft particles, and atom–electron-pair docking.

# 2. Exact parametrization by (R_0) and (E_0)

For a normal atomic pair, define

[
z_0=z(R_0)
==========

\left(1-\frac{R_0^2}{r_c^2}\right)^q.
]

Choose

[
\boxed{
C_R=\frac{E_0}{z_0^2},
\qquad
C_A=\frac{2E_0}{z_0}.
}
]

Then, exactly,

[
V(R_0)=-E_0,
\qquad
V'(R_0)=0.
]

Equivalently, define the normalized overlap

[
y(r)=\frac{z(r)}{z_0}
=====================

\left[
\frac{1-r^2/r_c^2}
{1-R_0^2/r_c^2}
\right]^q,
]

and write

[
\boxed{
V(r)=E_0\left[y(r)^2-2y(r)\right].
}
]

This is literally a compact-support polynomial Morse potential.

There is no need to multiply it by another cutoff function. The cutoff is already built into (y(r)), so the minimum is not shifted by the derivative of an external cutoff.

## Force without square root

Because (V=V(s)), where (s=\mathbf r\cdot\mathbf r),

[
\mathbf F=-\nabla_{\mathbf r}V
=-2\frac{dV}{ds}\mathbf r.
]

For the unnormalized form,

[
z=u^q,
\qquad
V=C_Rz^2-C_Az,
]

the force is

[
\boxed{
\mathbf F
=========

2q,\frac{u^{q-1}}{r_c^2}
\left(2C_Rz-C_A\right)\mathbf r.
}
]

For (r<R_0), the coefficient is positive and the force is repulsive. For (r>R_0), it is negative and attractive.

No (r), (\sqrt{s}), reciprocal square root, exponential, or general `pow()` is needed.

# 3. Particularly efficient (q=4) implementation

For (q=4),

[
z=u^4,\qquad z^2=u^8.
]

The cutoff is (C^3): energy, force, Hessian, and third derivative all approach zero continuously at (r_c).

```cpp
struct PairPotential {
    float rc2;
    float invRc2;
    float cRep;   // C_R
    float cAttr;  // C_A
};

inline float evalCompactMorse(
    const Vec3f& dr,
    const PairPotential& par,
    Vec3f& force
){
    const float r2 = dot(dr, dr);

    if (r2 >= par.rc2) {
        force = Vec3f{0.0f, 0.0f, 0.0f};
        return 0.0f;
    }

    const float x  = r2 * par.invRc2;
    const float u  = 1.0f - x;

    const float u2 = u * u;
    const float u3 = u2 * u;
    const float z  = u2 * u2;           // u^4

    const float energy =
        z * (par.cRep * z - par.cAttr);

    // q = 4:
    // F = 2*q*invRc2*u^(q-1)*(2*cRep*z-cAttr)*dr
    const float f =
        8.0f * par.invRc2 * u3
        * (2.0f * par.cRep * z - par.cAttr);

    force = dr * f;
    return energy;
}
```

The branch must happen before repeated squaring. Otherwise, because (q) is even, (u<0) beyond the cutoff would become positive again.

For (q=8), use

```cpp
u2 = u*u;
u4 = u2*u2;
z  = u4*u4;      // u^8
u7 = u4*u2*u;
```

and the force prefactor becomes (16/r_c^2).

# 4. Very clean mixing rules

Let every normal atom type store

[
R_i,\qquad e_i=\sqrt{E_{ii}}.
]

Use

[
\boxed{
R_{0,ij}=R_i+R_j,
\qquad
E_{0,ij}=e_i e_j.
}
]

This is the same additive-size/geometric-energy logic used by common Lennard–Jones mixing schemes. It is computationally convenient but should still be regarded as a transferable approximation rather than an exact physical law; cross-pair corrections are sometimes needed. ([ACS Publications][1])

Now choose a global cutoff ratio

[
r_{c,ij}=\lambda R_{0,ij}.
]

Then

[
u_0=1-\frac{1}{\lambda^2}
]

and

[
z_0=u_0^q
]

are global constants. Therefore,

[
C_{R,ij}
========

K_R e_i e_j,
\qquad
C_{A,ij}
========

K_A e_i e_j,
]

where

[
K_R=\frac{1}{z_0^2},
\qquad
K_A=\frac{2}{z_0}.
]

So the only pair-energy operation is

[
E_{ij}=e_i e_j.
]

No square root is evaluated during force-field evaluation.

For maximum runtime speed, I would precompute a small type-pair table containing

[
r_c^2,\quad r_c^{-2},\quad C_R,\quad C_A.
]

The mixing rules remain valuable because they regularize the parametrization and generate the table from (O(N_{\rm type})) atomic parameters rather than fitting (O(N_{\rm type}^2)) unrelated pairs.

## Independent repulsive and attractive mixing

You can make the two channels independently factorized:

[
C_{R,ij}=K_R,p_i p_j,
\qquad
C_{A,ij}=K_A,a_i a_j.
]

Here (p_i) and (a_i) are atomic repulsive and attractive amplitudes.

This immediately gives useful transitions:

* (a_i=0): the type has no attractive channel;
* (p_i=0): interactions involving that channel can be centrally attractive;
* scaling (a_i) continuously switches adhesion or bonding on and off;
* scaling both controls the total energy scale.

If rank-one products are too restrictive, a cheap extension is

[
C_{R,ij}
========

\sum_{k=1}^{K}p_i^{(k)}p_j^{(k)},
\qquad
C_{A,ij}
========

\sum_{k=1}^{K}a_i^{(k)}a_j^{(k)}.
]

Even (K=2) gives much richer chemistry while retaining (O(K)) evaluation instead of an unrestricted pair matrix.

---

# 5. Relation to the Morse range parameter

Near (R_0), the compact overlap behaves locally like an exponential.

For

[
y(r)=
\left[
\frac{1-r^2/r_c^2}
{1-R_0^2/r_c^2}
\right]^q,
]

the effective logarithmic slope at the minimum is

[
a_{\rm eff}
===========

# -\left.\frac{d\ln y}{dr}\right|_{R_0}

\frac{2qR_0}{r_c^2-R_0^2}.
]

With (r_c=\lambda R_0),

[
\boxed{
a_{\rm eff}
===========

\frac{2q}{R_0(\lambda^2-1)}.
}
]

The harmonic curvature is therefore

[
\boxed{
k
=

# V''(R_0)

# 2E_0a_{\rm eff}^2

\frac{8E_0q^2}
{R_0^2(\lambda^2-1)^2}.
}
]

Thus you can choose (q) and (\lambda) to match a Morse potential locally:

[
\boxed{
\lambda
=======

\sqrt{1+\frac{2q}{aR_0}}.
}
]

So the new potential can have the same (R_0), (E_0), and local stiffness as Morse, while having exact finite support and using only (r^2) and multiplication. The original Morse form was introduced as an anharmonic diatomic potential with explicit equilibrium and dissociation scales. ([Physical Review Journals][2])

# 6. Choosing (q) and (r_c)

The energy at complete overlap is

[
\frac{V(0)}{E_0}
================

## \frac{1}{z_0^2}

\frac{2}{z_0},
\qquad
z_0=
\left(1-\lambda^{-2}\right)^q.
]

For the center to be repulsive,

[
z_0<\frac12.
]

This gives approximately

[
\lambda_{\max}\approx
\begin{cases}
1.85,&q=2,\
2.51,&q=4,\
3.47,&q=8.
\end{cases}
]

Some representative barriers are:

| (q) | (\lambda) | (V(0)/E_0) |
| --: | --------: | ---------: |
|   4 |       2.0 |       3.67 |
|   8 |       2.0 |       79.8 |
|   4 |       1.7 |       19.0 |
|   8 |       2.2 |       27.8 |

Therefore:

* (q=4) is a good soft, very cheap default;
* (q=8) gives a much harder core while still needing only one additional squaring;
* very large (q) can create unnecessarily stiff dynamics and large single-precision dynamic ranges.

I would begin with (q=4) for bonded or relatively soft interactions and (q=8) for ordinary excluded-volume atom pairs.

# 7. Central docking at (r=0)

For an electron pair or another site that should attract an atom to the exact center, use the same kernel with

[
C_R=0,\qquad C_A=D.
]

Then

[
\boxed{
V_{\rm dock}(r)
===============

-D\left(1-\frac{r^2}{r_c^2}\right)^q.
}
]

Near the center,

[
V_{\rm dock}(r)
===============

-D
+
\frac{qD}{r_c^2}r^2
+O(r^4),
]

so it has a smooth harmonic minimum with

[
\boxed{
k_{\rm center}=\frac{2qD}{r_c^2}.
}
]

This is cheaper than the Lorentz function and already gives exactly the smooth parabolic vertex you want.

To control the center curvature independently of (D) and (r_c), use

[
V=u^q(c_0+c_1x),
\qquad x=r^2/r_c^2,
]

with

[
c_0=-D,
\qquad
c_1=\frac12kr_c^2-qD.
]

This remains polynomial and adds only one FMA.

# 8. Where the Lorentz form is still useful

Your Lorentz function should preferably be normalized:

[
L(r)=\frac{w^2}{w^2+r^2}
========================

\frac{1}{1+r^2/w^2}.
]

Then (L(0)=1), so the energy amplitude is independent of the width (w).

A compact Lorentz overlap can be

[
z_L(r)
======

\frac{w^2}{w^2+r^2}
\left(1-\frac{r^2}{r_c^2}\right)^q.
]

You can put it into exactly the same quadratic wrapper:

[
V=C_R z_L^2-C_Az_L.
]

This gives an independent inner width (w) and outer support (r_c), at the cost of one reciprocal. It is worthwhile when the purely polynomial (qD/r_c^2) relation is too restrictive.

For most central docking interactions, however,

[
-Du^q
]

is probably sufficient and faster.

# 9. Your exponential and Gaussian approximations

Your formulas are mathematically sound, with two qualifications.

For

[
\left(1-\frac{br}{n}\right)^n,
\qquad r_c=\frac{n}{b},
]

the logarithm is

[
n\ln\left(1-\frac{r}{r_c}\right)
================================

-\frac{nr}{r_c}
-\frac{nr^2}{2r_c^2}
-\cdots.
]

So it resembles an exponential near the origin, but near the cutoff it should be regarded as a compact polynomial basis rather than a quantitatively accurate exponential approximation.

The (r^2)-based version

[
\left(1-\frac{r^2}{r_c^2}\right)^q
]

is preferable for your application because it:

* avoids `sqrt`;
* has a smooth parabolic expansion around (r=0);
* gives force directly as a scalar multiplying (\mathbf r);
* works naturally as the overlap (z(r)) in the quadratic potential.

Your Python force for (q=2),

[
V=(1-r^2/r_c^2)^2,
]

is correct:

[
\mathbf F
=========

\frac{4}{r_c^2}
\left(1-\frac{r^2}{r_c^2}\right)\mathbf r.
]

In implementation, return this scalar force coefficient and multiply by the displacement vector, rather than computing a scalar radial force and then normalizing (\mathbf r).

# 10. Soft clamp assessment

Your soft clamp can be simplified to

[
S(y)
====

y_1+
\Delta
\frac{y-y_1}{\Delta+y-y_1},
\qquad
\Delta=y_2-y_1,
]

for (y>y_1).

Its derivative is indeed

[
S'(y)
=====

\left[
\frac{\Delta}
{\Delta+y-y_1}
\right]^2,
]

so your chain-rule implementation is correct.

However:

1. It is (C^1), but not (C^2), at (y=y_1).
2. As (y\rightarrow\infty), the derivative approaches zero.
3. If applied to an atomic repulsive core, the force becomes weaker at extreme overlap.

That last property is dangerous for excluded volume: once an integration error pushes atoms deeply into the saturated region, the repulsive force no longer grows and they may pass through each other.

I would use soft clamp for:

* bounded docking interactions;
* avoiding numerical overflow;
* deliberately penetrable soft particles;

but not as the sole hard-core protection for ordinary atoms.

A compact polynomial core with sufficiently large (q), or a rare fallback rational core activated only at very small (r), is safer.

## Recommended final family

The most useful minimal kernel is

[
\boxed{
z(r)=
\begin{cases}
\left(1-r^2/r_c^2\right)^q,&r<r_c,\
0,&r\ge r_c,
\end{cases}
}
]

[
\boxed{
V(r)=C_Rz(r)^2-C_Az(r).
}
]

It gives:

[
\boxed{
\mathbf F
=========

2q,r_c^{-2}
\left(1-r^2/r_c^2\right)^{q-1}
\left(2C_Rz-C_A\right)\mathbf r.
}
]

For standard atoms:

[
R_{0,ij}=R_i+R_j,\qquad
r_{c,ij}=\lambda R_{0,ij},\qquad
E_{ij}=e_i e_j,
]

[
C_{R,ij}=\frac{E_{ij}}{z_0^2},
\qquad
C_{A,ij}=\frac{2E_{ij}}{z_0},
\qquad
z_0=(1-\lambda^{-2})^q.
]

For central docking:

[
C_R=0,\qquad C_A=D.
]

For pure repulsion:

[
C_R=H,\qquad C_A=0.
]

This gives one branch, one (r^2), repeated squaring, no square root, no exponential, no on-the-fly energy square root, exact finite support, and a continuous parameter-space transition among the main interaction types you described.

[1]: https://pubs.acs.org/doi/10.1021/acs.jctc.2c01170?utm_source=chatgpt.com "Influence of the Lennard-Jones Combination Rules on the Simulated Properties of Organic Liquids at Optimal Force-Field Parametrization | Journal of Chemical Theory and Computation"
[2]: https://journals.aps.org/pr/abstract/10.1103/PhysRev.34.57?utm_source=chatgpt.com "Diatomic Molecules According to the Wave Mechanics. II. Vibrational Levels | Phys. Rev."
