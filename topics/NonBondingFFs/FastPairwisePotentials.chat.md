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

---

# USER

I found quite good formula how to aproximate Morse which I can use also for other thing (epairs, they have finite cutff, look like gaussain approx), the only problem is than I would prefer to use mixing rules and analytical expression rather than fitting

can we find analytically these fitting coefficients (or at las good aporximation) using mixing rules

Rij=Ri+Rj
Eij = sqrt( Eii * Ejj )

think about it and write it in the file @fit_radial.py 
even if you do not know the answer at least formulate the question

for formulas
fc*((r-r_node)*(a*fc³ + c))
and
fc^2*((r-r_node)*(a*fc^2 + c))

with fc=smoothstep, or (1-(r/rc)^2)^2

good thing is that at least we can compute r_node analystically, right?

see attached python code with detailed comments

---

# Chat GPT 5.6 sol hi

I edited the two requested compact-Morse forms already present in your script—`morse2c` and `morse2d`—and replaced the approximate coefficient argument with exact analytical constructions.  The code retains your standard mixed parameters (R_{ij}=R_i+R_j) and (E_{ij}=\sqrt{E_{ii}E_{jj}}). 

[Download the updated `fit_radial.py`](sandbox:/mnt/data/fit_radial.py)

## Main analytical result

Write both models as

[
V(r)=f(r)^p(r-r_n)\left[a f(r)^q+c\right],
]

with

[
(p,q)=
\begin{cases}
(1,3),&\texttt{morse2d},\
(2,2),&\texttt{morse2c},
\end{cases}
\qquad n=p+q=4.
]

From Morse,

[
\Delta=\frac{\ln 2}{\beta},
\qquad
r_n=R_0-\Delta,
]

so the zero crossing is exact.

At (r=R_0), define

[
f_0=f(R_0),\qquad
h_0=\frac{f'(R_0)}{f_0},
\qquad
B=-\frac{E_0}{\Delta},
\qquad
s=-\frac{1}{\Delta h_0}.
]

Imposing

[
V(R_0)=-E_0,
\qquad
V'(R_0)=0
]

gives

[
\boxed{
a=
\frac{B(s-p)}
{qf_0^{p+q}}
}
]

and

[
\boxed{
c=
\frac{B(p+q-s)}
{qf_0^p}.
}
]

Thus the pair coefficients are generated directly from

[
R_0=R_i+R_j,
\qquad
E_0=e_i e_j,
\qquad
e_i=\sqrt{E_{ii}},
]

without fitting. The coefficients scale linearly with (E_0); their shape dependence is only a deterministic function of (R_0,\beta,r_c).

## More elegant `pure-tail` solution

There is an important issue with a fixed, distant (r_c): the exact-minimum solution often produces (c>0). Since near the cutoff

[
V(r)\sim c(r-r_n)f(r)^p,
]

this means the potential approaches zero from above and creates a small outer repulsive bump.

The clean boundary is (c=0). Then both of your formulas collapse to the same especially cheap potential:

[
V(r)=a(r-r_n)f(r)^4.
]

The cutoff can be chosen analytically so that this one-term potential still satisfies

[
V(R_0)=-E_0,
\qquad
V'(R_0)=0.
]

For

[
f(r)=\left(1-\frac{r^2}{r_c^2}\right)^2,
]

the result is particularly simple:

[
\boxed{
r_c^2=R_0^2+16\Delta R_0
}
]

and

[
\boxed{
a=-\frac{E_0}{\Delta f(R_0)^4},
\qquad c=0.
}
]

This version has several advantages:

* exact Morse zero crossing;
* exact equilibrium distance (R_0);
* exact depth (-E_0);
* no outer repulsive bump;
* purely attractive approach to the cutoff;
* only one energy coefficient;
* (r_c^2) can be generated directly without a square root;
* evaluation uses only (r^2), multiplication, and repeated squaring.

For (n=4),

[
f(R_0)
======

\left(
\frac{16\Delta}{R_0+16\Delta}
\right)^2,
]

so even (a) may be evaluated without first calculating (r_c):

[
\boxed{
a=
-\frac{E_0}{\Delta}
\left(
\frac{R_0+16\Delta}{16\Delta}
\right)^8.
}
]

This is probably the nicest analytical mixing rule in the file.

## Added code

The updated script now contains:

* `analytical_coeffs(...)` for fixed-(r_c) exact depth and minimum;
* `mix_morse_pair(...)` applying (R_i+R_j) and (e_i e_j);
* `critical_cutoff_for_pure_tail(...)`;
* `mix_morse_pair_pure_tail(...)`;
* both `smoothstep` and `fc22` cutoff choices;
* diagnostics for zero, depth, equilibrium derivative, curvature, and tail sign;
* comparison with the original weighted least-squares coefficients.

Example commands:

```bash
python fit_radial.py \
    --basis morse2d \
    --morse-cutoff fc22 \
    --analytical \
    --analytical-mode minimum
```

and the more elegant pair-cutoff construction:

```bash
python fit_radial.py \
    --basis morse2d \
    --morse-cutoff fc22 \
    --analytical \
    --analytical-mode pure-tail
```

The same works with `--basis morse2c`.

I syntax-checked the file and exercised all combinations of `morse2c`/`morse2d`, smoothstep/`fc22`, and the three analytical modes. The exact-minimum and pure-tail residuals were at roughly (10^{-17})–(10^{-18}). I used representative stub atom types because the repository’s actual `ElementTypes.dat` and `AtomTypes.dat` were not included with the uploaded standalone script.

---

# USER

OK, the pure-tail is very nice, the minimum is not usefull at all as ti create bump.

But I'm interested to know - if I would like to make this with lower power like fc^2 rather than fc^4 it is also possible?

this means that I can then interpolate smoothely between these two. Why?

Because pure-tail solution using only fc^4 is a bit too steep, what if I want something which aproximate better tha tail of Morse. lower power of fc is better for that but it make worse aroud minimum, so I was thinking to do analytical solution for both and then make some linear combination in between.

---

# GPT 5.6 sol hi

Yes—(f^2) has an equally clean pure-tail solution, and there is an especially elegant **single-cutoff interpolation between (f^2) and (f^4)**. I added both to the script containing your original `morse2c`/`morse2d` forms. 

[Download the updated `fit_radial.py`](sandbox:/mnt/data/fit_radial.py)

## 1. Pure (f^n) solution for arbitrary power

Let

[
\Delta=\frac{\ln 2}{\beta},
\qquad
r_{\rm node}=R_0-\Delta,
]

and

[
V_n(r)=a_n(r-r_{\rm node})f(r)^n.
]

For

[
f(r)=\left(1-\frac{r^2}{r_c^2}\right)^2,
]

requiring

[
V_n(R_0)=-E_0,
\qquad
V_n'(R_0)=0
]

gives, for any (n>0),

[
\boxed{
r_c^2=R_0^2+4n\Delta R_0
}
]

and

[
\boxed{
a_n=-\frac{E_0}{\Delta f(R_0)^n}.
}
]

Therefore, for (f^2),

[
\boxed{
r_{c,2}^2=R_0^2+8\Delta R_0,
\qquad
a_2=-\frac{E_0}{\Delta f_0^2}.
}
]

For (f^4),

[
\boxed{
r_{c,4}^2=R_0^2+16\Delta R_0,
\qquad
a_4=-\frac{E_0}{\Delta f_0^4}.
}
]

The standard mixing remains

[
R_0=R_i+R_j,
\qquad
E_0=e_i e_j,
\qquad
e_i=\sqrt{E_{ii}}.
]

So no pair fitting is needed.

## 2. Why (f^2) gives a longer tail

Near the cutoff, the base function behaves as

[
f(r)\sim (r_c-r)^2.
]

Consequently,

[
f^2\sim (r_c-r)^4,
\qquad
f^4\sim (r_c-r)^8.
]

Thus (f^2) approaches zero substantially more slowly and better imitates the extended attractive tail of Morse.

There is a tradeoff: because exact stationarity determines (r_c), the (f^2) cutoff is shorter than the (f^4) cutoff. It also produces a sharper minimum. For the pure (f^n) family,

[
\boxed{
V_n''(R_0)
==========

E_0\left[
\frac{1+\frac{1}{2n}}{\Delta^2}
+
\frac{1}{\Delta R_0}
\right].
}
]

Therefore (n=2) is stiffer around the minimum than (n=4), even though its final decay toward the cutoff is slower.

That explains exactly the behavior you observed: lower power improves the tail but worsens the minimum region.

## 3. Common-cutoff interpolation between (f^2) and (f^4)

The useful interpolating family is

[
\boxed{
V_\lambda(r)
============

-\frac{E_0}{\Delta}(r-r_{\rm node})
\left[
(1-\lambda)
\left(\frac{f(r)}{f_0}\right)^2
+
\lambda
\left(\frac{f(r)}{f_0}\right)^4
\right],
}
]

where

[
0\le\lambda\le1,
\qquad
f_0=f(R_0).
]

At (R_0), the bracket is always one, so

[
V_\lambda(R_0)=-E_0.
]

Its effective logarithmic exponent at the minimum is

[
n_{\rm eff}
===========

# 2(1-\lambda)+4\lambda

2+2\lambda.
]

Therefore, choose

[
\boxed{
r_c^2
=====

# R_0^2+4n_{\rm eff}\Delta R_0

R_0^2+8(1+\lambda)\Delta R_0.
}
]

Then

[
V_\lambda'(R_0)=0
]

exactly for every (\lambda).

Written in your preferred polynomial form,

[
\boxed{
V_\lambda(r)
============

(r-r_{\rm node})
\left[
c_2 f(r)^2+a_4f(r)^4
\right],
}
]

with

[
\boxed{
c_2=
-\frac{E_0}{\Delta}
\frac{1-\lambda}{f_0^2},
\qquad
a_4=
-\frac{E_0}{\Delta}
\frac{\lambda}{f_0^4}.
}
]

Both coefficients are non-positive. Therefore:

* the tail always approaches zero from below;
* there is no repulsive bump;
* there is no additional zero;
* (\lambda=0) is exactly the pure (f^2) solution;
* (\lambda=1) is exactly the pure (f^4) solution.

This is better than literally combining two separately constructed potentials,

[
(1-\lambda)V_2(r;r_{c,2})+\lambda V_4(r;r_{c,4}),
]

because the latter requires two different cutoffs and two cutoff evaluations. The new construction uses one (r_c), one (f), then

[
f_2=f^2,\qquad f_4=f_2^2,
]

followed by one linear combination.

## Added usage

Pure (f^2):

```bash
python fit_radial.py \
    --basis morse2c \
    --morse-cutoff fc22 \
    --analytical \
    --analytical-mode pure-tail \
    --pure-tail-power 2
```

Pure (f^4):

```bash
python fit_radial.py \
    --basis morse2c \
    --morse-cutoff fc22 \
    --analytical \
    --analytical-mode pure-tail \
    --pure-tail-power 4
```

Interpolated solution, for example (35%) (f^4) and (65%) (f^2):

```bash
python fit_radial.py \
    --basis morse2c \
    --morse-cutoff fc22 \
    --analytical \
    --analytical-mode tail-blend \
    --tail-blend 0.35
```

The pair-construction function is:

```python
params = mix_morse_pair_tail_blend(
    Ri, ei,
    Rj, ej,
    beta,
    blend=0.35,
    cutoff_kind="fc22",
    power_lo=2,
    power_hi=4,
)
```

I tested (\lambda=0,0.25,0.5,0.75,1) for both `fc22` and smoothstep cutoffs. The node, well depth, and equilibrium-force residuals were at approximately (10^{-16}) numerical precision.

---

# USER

this is good but even with high power ti does not converge to morse tail,
I think this is becasue we use apporximation of Gaussaian rather than approximation of exp(-b*r)

we should try aproximate morese with 
exp(-b*r)~= e(r,b,n)=(1-a*r/n)^n=(1-r/rc)^n

morse  = e*e-2*e 

this should converge mur faster 

the only question is - our epair potential should be smooth near zero not have spike, if we wnat to use same potential for atom-atom morese (spike at zero), and for atom-epair (blunt blob) we should smoothen it, we scan use smooth-clamp for it but is seems as overkill. Maybe there is better way?

Why I insist of using the same formula for atom-atom and atom-epair when they should have completely different shape ?
Because I want to avoid branch divergence in GPU, if one thread is processing epair dummy atom, other thread is processing real atom, they should process exactly same instruction just with different numerical parameters. That is the main motivation of this excercise

---

# GPT 5.6 sol hi

Yes. The correct way to recover the Morse tail is to approximate its exponential variable directly, rather than multiplying a linear function by Gaussian-like cutoffs. Your previous `morse2c/morse2d` forms are still retained for comparison. 

[Download the updated `fit_radial.py`](sandbox:/mnt/data/updated/fit_radial.py)

## 1. Compact exponential Morse

Define

[
u(r)=\max\left(0,,
1-\frac{\beta}{n}(r-R_0)
\right),
\qquad
y(r)=u(r)^n.
]

Then

[
\boxed{
V(r)=E_0\left[y(r)^2-2y(r)\right].
}
]

As (n\to\infty),

[
\left(1-\frac{\beta(r-R_0)}{n}\right)^n
\longrightarrow
e^{-\beta(r-R_0)},
]

so this converges directly to Morse:

[
V(r)\longrightarrow
E_0
\left[
e^{-2\beta(r-R_0)}
------------------

2e^{-\beta(r-R_0)}
\right].
]

The expansion is

[
\ln y
=====

-\beta x
-\frac{\beta^2x^2}{2n}
-\frac{\beta^3x^3}{3n^2}
-\cdots,
\qquad x=r-R_0.
]

This is qualitatively different from

[
\left(1-\frac{r^2}{r_c^2}\right)^n,
]

which converges toward a Gaussian-like radial dependence rather than an exponential.

The compact cutoff is

[
\boxed{
r_c=R_0+\frac{n}{\beta}.
}
]

At the equilibrium distance, independently of (n),

[
y(R_0)=1,
\qquad
V(R_0)=-E_0,
\qquad
V'(R_0)=0,
]

and even the curvature is exactly Morse-like:

[
\boxed{
V''(R_0)=2E_0\beta^2.
}
]

The inner zero is analytical:

[
\boxed{
r_{\rm node}
============

R_0+\frac{n}{\beta}
\left(1-2^{1/n}\right).
}
]

It converges to

[
R_0-\frac{\ln2}{\beta}.
]

## 2. One formula for atoms and electron pairs

The useful generalization is

[
\boxed{
V(r)
====

E_0y(r)
\left[
\alpha y(r)-(1+\alpha)
\right].
}
]

Equivalently,

[
V=A y^2-B y,
\qquad
A=E_0\alpha,
\qquad
B=E_0(1+\alpha).
]

This guarantees

[
V(y=1)=-E_0
]

for every (\alpha).

For an ordinary atom pair,

[
\alpha=1,
]

and therefore

[
V=E_0(y^2-2y),
]

which is compact Morse.

For an atom–electron-pair interaction,

[
0\le\alpha<1.
]

In particular,

[
\alpha=0
]

gives the very simple purely attractive overlap

[
V=-E_0y.
]

This eliminates the repulsive spike without changing the instruction sequence.

## 3. Blunting the origin without `soft_clamp`

Use a softened radial coordinate:

[
\boxed{
\rho(r,w)=\sqrt{r^2+w^2}-w.
}
]

For numerical stability near (r=0), evaluate it as

[
\boxed{
\rho(r,w)
=========

\frac{r^2}
{\sqrt{r^2+w^2}+w}.
}
]

Then replace (r) by (\rho) in the compact exponential:

[
u=
\max\left(
0,,
1-\frac{\beta}{n}(\rho-R_0)
\right),
\qquad
y=u^n.
]

This mapping has exactly the desired limits.

For a real atom, set

[
w=0.
]

Then

[
\rho(r,0)=r,
]

so the normal compact Morse core is recovered exactly.

For an electron-pair site, use (w>0). Near the origin,

[
\rho(r,w)
=========

\frac{r^2}{2w}
+O(r^4),
]

so the interaction becomes a smooth isotropic blob.

For (R_0=0),

[
y
=

1-\frac{\beta r^2}{2w}
+O(r^4).
]

The generalized potential then behaves as

[
\boxed{
V(r)
====

-E_0
+
\frac{E_0(1-\alpha)\beta}{2w}r^2
+O(r^4).
}
]

Thus the central harmonic curvature is

[
\boxed{
k_0=
\frac{E_0(1-\alpha)\beta}{w}.
}
]

You can therefore choose the blunting width analytically:

[
\boxed{
w=
\frac{E_0(1-\alpha)\beta}{k_0}.
}
]

This also clarifies why both modifications are needed:

* (w>0) makes the radial coordinate smooth at the origin.
* (\alpha<1) produces a parabolic minimum.
* With (\alpha=1), the central minimum becomes quartic because the Morse polynomial itself is stationary at (y=1).

## 4. Branch-free mixing rules

I added this per-type parameterization:

[
(R_i,e_i,g_i,w_i).
]

Here:

* (R_i) is the atomic radius;
* (e_i=\sqrt{E_{ii}});
* (g_i=1) for a real atom;
* (g_i=0) for an electron-pair dummy site;
* (w_i=0) for a real atom;
* (w_i>0) for an electron pair.

The pair rules are

[
\boxed{
g_{ij}=g_i g_j,
}
]

[
\boxed{
R_{0,ij}=g_{ij}(R_i+R_j),
}
]

[
\boxed{
E_{0,ij}=e_i e_j,
}
]

[
\boxed{
\alpha_{ij}=g_{ij},
}
]

[
\boxed{
w_{ij}=w_i+w_j.
}
]

Therefore:

### Real atom–real atom

[
g_i=g_j=1
]

gives

[
R_0=R_i+R_j,\qquad
\alpha=1,\qquad
w=0,
]

which is compact Morse.

### Real atom–electron pair

[
g_i=1,\qquad g_j=0
]

gives

[
R_0=0,\qquad
\alpha=0,\qquad
w=w_j,
]

which is a smooth central attractive well.

There is no runtime pair-type branch. These parameters can be mixed once when constructing a type-pair table.

For an undesired electron-pair–electron-pair interaction, its energy channel can simply be set to zero in the pair table.

## 5. Physical cutoff with soft radius

The cutoff in transformed radius is

[
\rho_c=R_0+\frac{n}{\beta}.
]

Since

[
\rho_c=\sqrt{r_c^2+w^2}-w,
]

the physical squared cutoff is especially simple:

[
\boxed{
r_c^2=\rho_c(\rho_c+2w).
}
]

This is convenient for neighbor search because `rc2` can be stored directly without calculating a square root.

## 6. GPU evaluation

For a displacement vector (\mathbf r):

```cpp
float r2 = dot(dr, dr);

float rw  = sqrt(r2 + w*w);
float rho = r2 / max(rw + w, eps);

float u = max(
    0.0f,
    1.0f - (beta / float(n)) * (rho - R0)
);

float y = pow_integer_by_squaring(u, n);

float E = E0 * y * (alpha*y - (1.0f + alpha));
```

The vector force is

[
\boxed{
\mathbf F
=========

E_0\beta
\left[
2\alpha y-(1+\alpha)
\right]
u^{n-1}
\frac{\mathbf r}{\sqrt{r^2+w^2}}.
}
]

For (n=8):

```cpp
float u2 = u*u;
float u4 = u2*u2;
float y  = u4*u4;       // u^8
float u7 = u4*u2*u;     // u^7
```

All lanes execute the same instructions. `max()` normally becomes a scalar maximum/select instruction, not divergent control flow.

## Recommended power

The exact Morse energy at the compact cutoff is approximately

[
\frac{V_{\rm Morse}(r_c)}{E_0}
\approx -2e^{-n}.
]

Therefore:

[
n=4:\quad -2e^{-4}\approx -0.0366,
]

[
n=8:\quad -2e^{-8}\approx -6.71\times10^{-4},
]

[
n=16:\quad -2e^{-16}\approx -2.25\times10^{-7}.
]

Thus (n=8) looks like the best default:

* only three squarings for (u^8);
* exact depth, minimum and curvature;
* a very small truncation error at the cutoff;
* much better Morse-tail behavior than the Gaussian-like family.

The updated script adds:

```bash
python fit_radial.py \
    --compact-exp-demo \
    --exp-powers 2,4,8 \
    --soft-radius sqrt \
    --epair-width 0.6 \
    --epair-alpha 0.0
```

It also includes the optional cheaper mapping

```bash
--soft-radius rational
```

using

[
\rho=\frac{r^2}{r+w}.
]

That alternative saves some arithmetic but is only finitely smooth at the Cartesian origin. The `sqrt` form is the safer default for force and Hessian continuity.

I syntax-checked the modified file and numerically verified the equilibrium depth, zero force, Morse curvature, analytical zero crossing, vector-force expression, and electron-pair central curvature for both soft-radius variants.
