# USER

I would like to define simple energy functional for optimizing occupancy of pi-bonds in pi-system (graphene of poly aromatic hydrocarbon), it should be motivated by quautm mechanics, but solved purely clascally. What is key that to fine groun state of the system I should not need diagonalize the system and find eigenstate ( O(N^3)), but simply minimize the energy (linear scaling if I have just local interactions, eventually electrosatatics can be done by FFT of Fast multipole method).  

So the starting point/assumtion:

1) every atom is connected to 3 neighbors (unless on the dege)
2) each atom contribute 0,1,2 electrons ( carbon contribute 1, nitrogen can contribute 1 or 2 depending if it is -NH- with electron pair or =N- which is like carbon ), they have also different on-site energie
3) these electron delocalize by hopping which is basically laplace operator (i.e. rapid changes in electron density cost energy)
4) bonds localize by pierls distortion like in SSH model, increased number of electrons in the bonds makes the bond shorter which also lowers the energy of the bond, colloquially the electron dig a hole  and fall into it. 
5) the lattice of sigma bonds want all bonds to have same length (act agains pierls distortion) byt some elastic hook law.
6) The occupancy of the bond is given by Mulliken bond charge q_ij = (S_ij c_i c_j)
7) The occupancy q_ii=|c_i|^2 of each site can be anything but there is Hubard U (penalty for more than 1 electron on one site) and electrostatics (maybe we can have separate spin-densities?)
8) we have coulomb matrix for occupancy between the charge states. 

Please try to formulate this model more rigorously with equations and psudocode how to solve it. I do not want any matrix algebra just gradient descent, something like QEq (charge equalibration) but with added delocalization due to hopping and localizedin by pierls distortion. I want to solve for bond charges rather than for wavefunction (nor moleculer orbitlas), the wavefunction coefs c_i c_j are just axuliary variables to find bond charge q_ij and site  occupancy q_ii. (possibly spin polarized)

---

# Chat GPT 5.5

Below is a formulation I think matches your intention quite well: **a QEq-like variational model, but with sparse local density-matrix/bond-order variables instead of molecular orbitals**. The central variables are not eigenstates (c_{n i}), but local occupancies and bond orders:

[
n_{i\sigma} \equiv P^\sigma_{ii}, \qquad p_{ij\sigma} \equiv P^\sigma_{ij}
]

where (P^\sigma) is a **spin-resolved one-particle density matrix**, but we never diagonalize it. We only store its diagonal and near-neighbor off-diagonal elements.

The model is quantum-mechanically motivated because (P_{ij}) is exactly the object from which Mulliken charges, bond orders, hopping energy, Hubbard energy, and Peierls coupling are normally computed. But computationally it becomes a **classical field minimization problem on a graph**.

# 1. Graph and variables

Let the (\pi)-system be a graph

[
G=(V,E)
]

where each atom (i\in V) has up to three neighbors (j\in N(i)). Each site has one (p_z)-like orbital. For each spin (\sigma=\uparrow,\downarrow), define

[
0 \le n_{i\sigma} \le 1
]

and for each bond (ij),

[
p_{ij\sigma}=p_{ji\sigma}.
]

The total (\pi)-electron number is fixed:

[
\sum_i \left(n_{i\uparrow}+n_{i\downarrow}\right)=N_\pi.
]

For neutral carbon,

[
N_{\pi,i}^{0}=1.
]

For heteroatoms, you can set different reference occupancies and onsite energies:

[
N_{\pi,i}^{0} =
\begin{cases}
1 & \text{C, pyridine-like } =\mathrm{N}-, \
2 & \text{pyrrole-like } -\mathrm{NH}-, \
0,1,2 & \text{depending on chemistry / boundary condition}.
\end{cases}
]

The physical site charge is

[
Q_i = Z_i^\pi - n_i,
\qquad
n_i=n_{i\uparrow}+n_{i\downarrow},
]

where (Z_i^\pi) is the positive reference charge associated with the (\pi)-electron system. For carbon, (Z_i^\pi=1).

The bond order is

[
B_{ij}=p_{ij\uparrow}+p_{ij\downarrow}.
]

If you want a Mulliken-like nonorthogonal form, then with overlap (S_{ij}),

[
q_{ij}^{\mathrm{bond}} = 2 S_{ij} B_{ij}.
]

In the simplest orthogonal Hückel-like version, you can just take (B_{ij}) itself as the bond order.

# 2. Why bond orders need constraints

This is the crucial point.

You cannot make (p_{ij}) arbitrary. Quantum mechanically, (P) must obey approximately

[
0 \le P^\sigma \le 1.
]

For a zero-temperature closed-shell Slater determinant,

[
(P^\sigma)^2 = P^\sigma.
]

This is the hidden Pauli constraint. If you ignore it, the minimization will put maximal bond order on every bond and produce an unphysical state where every bond is simultaneously “double”.

The local version of idempotency gives a very useful constraint:

[
(P^2)*{ii}=P*{ii}.
]

Expanding this locally,

[
n_{i\sigma}^2+\sum_{j\in N(i)} p_{ij\sigma}^2 \approx n_{i\sigma}.
]

Therefore

[
\sum_{j\in N(i)} p_{ij\sigma}^2 \approx n_{i\sigma}(1-n_{i\sigma}).
]

This is beautiful because it gives a local Pauli competition rule:

[
\boxed{
\sum_{j\in N(i)} p_{ij\sigma}^2
\lesssim
n_{i\sigma}(1-n_{i\sigma})
}
]

So if a spin channel is nearly empty or nearly full on a site, it cannot carry much covalent bond order. Maximum delocalization occurs near

[
n_{i\sigma}=1/2.
]

For neutral graphene, closed shell,

[
n_{i\uparrow}=n_{i\downarrow}=1/2,
]

so each spin channel can delocalize.

This is the density-matrix analogue of the fact that covalent bonding needs partially occupied local orbitals in each spin channel.

# 3. Energy functional

I would write the total energy as

[
E =
E_{\mathrm{onsite}}
+
E_{\mathrm{Hub}}
+
E_{\mathrm{Coul}}
+
E_{\mathrm{hop}}
+
E_{\mathrm{Peierls}}
+
E_{\mathrm{Pauli}}
+
E_{\mathrm{chem}}.
]

Each term has a clear physical meaning.

## 3.1 Onsite QEq-like term

For different atom types, use electronegativity and hardness:

[
E_{\mathrm{onsite}}
===================

\sum_i
\left[
\chi_i (n_i-N_i^0)
+
\frac{\eta_i}{2}(n_i-N_i^0)^2
\right].
]

Here (\chi_i) controls whether a site wants to accept or donate (\pi)-charge. Nitrogen, oxygen, edge groups, protonated sites, etc. enter mainly through (\chi_i,\eta_i,N_i^0).

## 3.2 Hubbard term

For spin polarization, use

[
E_{\mathrm{Hub}}
================

\sum_i U_i n_{i\uparrow}n_{i\downarrow}.
]

This penalizes double occupation of the same (p_z) orbital.

Equivalent form:

[
n_{i\uparrow}n_{i\downarrow}
============================

\frac{n_i^2-m_i^2}{4},
]

where

[
m_i=n_{i\uparrow}-n_{i\downarrow}.
]

So

[
E_{\mathrm{Hub}}
================

\sum_i
\frac{U_i}{4}
\left(n_i^2-m_i^2\right).
]

This shows why Hubbard (U) favors spin polarization: for fixed (n_i), increasing (|m_i|) lowers the double-occupation penalty.

You may also add a small spin stiffness if needed:

[
E_{\mathrm{spin;reg}}
=====================

\sum_i
\frac{A_i}{2}m_i^2
]

if the model otherwise overpolarizes.

## 3.3 Coulomb interaction

Let

[
Q_i=Z_i^\pi-n_i.
]

Then

[
E_{\mathrm{Coul}}
=================

\frac{1}{2}
\sum_{ij}
Q_i J_{ij} Q_j.
]

Here (J_{ij}) can be Ohno-like, screened Coulomb, image-charge corrected substrate Coulomb, FFT/FMM, etc.

A useful regularized form is

[
J_{ij}
======

\frac{1}{\sqrt{R_{ij}^2+a_i a_j}}
\frac{e^2}{4\pi\varepsilon_0\varepsilon_{\mathrm{eff}}},
]

or in your usual units,

[
J_{ij}
======

\frac{14.3996}{\varepsilon_{\mathrm{eff}} \sqrt{R_{ij}^2+a_i a_j}}
\quad \mathrm{eV}.
]

For (i=j), this should be consistent with onsite hardness/Hubbard terms, so avoid double-counting.

## 3.4 Hopping / resonance energy

In ordinary tight-binding,

[
E_{\mathrm{band}}
=================

\operatorname{Tr}(PH).
]

For nearest-neighbor hopping,

[
H_{ij}=-t_{ij}.
]

Therefore the local hopping contribution is

[
E_{\mathrm{hop}}
================

-2
\sum_{\langle ij\rangle,\sigma}
t_{ij} p_{ij\sigma}.
]

The factor 2 is convention-dependent. You can absorb it into (t). The important part is:

[
\boxed{
E_{\mathrm{hop}} \sim -t_{ij} B_{ij}
}
]

so positive bond order lowers energy.

This term favors delocalization and aromatic resonance.

## 3.5 Peierls / SSH coupling

Let the bond-length distortion be

[
u_{ij}=r_{ij}-r_{0,ij}.
]

The sigma skeleton pays elastic energy

[
E_{\sigma}
==========

\sum_{\langle ij\rangle}
\frac{K_{ij}}{2}u_{ij}^2.
]

The hopping depends on bond length:

[
t_{ij}(u_{ij})
==============

t_{0,ij} - \alpha_{ij}u_{ij},
]

or more realistically,

[
t_{ij}(u_{ij})
==============

t_{0,ij}\exp(-\beta_{ij}u_{ij}).
]

For the linear SSH form,

[
E_{ij}
======

## \frac{K_{ij}}{2}u_{ij}^2

2
\sum_\sigma
\left(t_{0,ij}-\alpha_{ij}u_{ij}\right)p_{ij\sigma}.
]

Using

[
B_{ij}=p_{ij\uparrow}+p_{ij\downarrow},
]

this becomes

[
E_{ij}
======

## \frac{K_{ij}}{2}u_{ij}^2

2t_{0,ij}B_{ij}
+
2\alpha_{ij}u_{ij}B_{ij}.
]

Depending on sign convention, shorter bond should increase (t). If (u>0) means longer bond, then (\alpha>0), and the energy is minimized by negative (u), i.e. shortening.

Minimize with respect to (u_{ij}):

[
\frac{\partial E_{ij}}{\partial u_{ij}}
=======================================

K_{ij}u_{ij}
+
2\alpha_{ij}B_{ij}=0.
]

Therefore

[
\boxed{
u_{ij}^{*}
==========

-\frac{2\alpha_{ij}}{K_{ij}}B_{ij}
}
]

so larger bond order shortens the bond.

Substituting back,

[
E_{ij}^{\mathrm{eff}}
=====================

## -2t_{0,ij}B_{ij}

\frac{2\alpha_{ij}^2}{K_{ij}}B_{ij}^2.
]

So Peierls coupling generates an attractive nonlinear bond-order term:

[
\boxed{
E_{\mathrm{Peierls}}^{\mathrm{eff}}
===================================

-\sum_{\langle ij\rangle}
\gamma_{ij} B_{ij}^2
}
]

with

[
\gamma_{ij}=\frac{2\alpha_{ij}^2}{K_{ij}}.
]

This is exactly the “electron digs a hole and falls into it” mechanism. The hopping term wants delocalized bonding everywhere; the Peierls term wants to concentrate bond order into alternating shorter bonds. Their competition gives aromatic versus Kekulé behavior.

# 4. Pauli / representability penalty

A simple local penalty is

[
E_{\mathrm{Pauli,diag}}
=======================

\frac{\mu}{2}
\sum_{i,\sigma}
\left[
n_{i\sigma}^2
+
\sum_{j\in N(i)}p_{ij\sigma}^2
------------------------------

n_{i\sigma}
\right]^2.
]

Define

[
R_{i\sigma}
===========

n_{i\sigma}^2
+
\sum_{j\in N(i)}p_{ij\sigma}^2
------------------------------

n_{i\sigma}.
]

Then

[
E_{\mathrm{Pauli,diag}}
=======================

\frac{\mu}{2}
\sum_{i,\sigma}R_{i\sigma}^2.
]

This term alone already gives strong local competition between bonds.

But you can also include off-diagonal idempotency. For nearest-neighbor (i,j),

[
(P^2)_{ij}
==========

\sum_k P_{ik}P_{kj}.
]

Keeping only local terms,

[
(P^2)*{ij}
\approx
n*{i\sigma}p_{ij\sigma}
+
p_{ij\sigma}n_{j\sigma}
+
\sum_{k\in N(i)\cap N(j)}
p_{ik\sigma}p_{kj\sigma}.
]

Then impose

[
(P^2)*{ij}\approx P*{ij}.
]

So define

[
R_{ij\sigma}
============

(n_{i\sigma}+n_{j\sigma}-1)p_{ij\sigma}
+
\sum_{k\in N(i)\cap N(j)}
p_{ik\sigma}p_{kj\sigma}.
]

and add

[
E_{\mathrm{Pauli,off}}
======================

\frac{\mu_{\mathrm{off}}}{2}
\sum_{\langle ij\rangle,\sigma}
R_{ij\sigma}^2.
]

This is more expensive but still local. It couples bond orders around rings and improves aromaticity/Kekulé competition.

# 5. Minimal useful functional

A compact working model could be:

[
\boxed{
\begin{aligned}
E
=&
\sum_i
\left[
\chi_i(n_i-N_i^0)
+
\frac{\eta_i}{2}(n_i-N_i^0)^2
+
U_i n_{i\uparrow}n_{i\downarrow}
\right]
\
&+
\frac{1}{2}
\sum_{ij}Q_iJ_{ij}Q_j
\
&-
2\sum_{\langle ij\rangle,\sigma}
t_{0,ij}p_{ij\sigma}
\
&-
\sum_{\langle ij\rangle}
\gamma_{ij}
\left(p_{ij\uparrow}+p_{ij\downarrow}\right)^2
\
&+
\frac{\mu}{2}
\sum_{i,\sigma}
\left[
n_{i\sigma}^2
+
\sum_{j\in N(i)}p_{ij\sigma}^2
------------------------------

n_{i\sigma}
\right]^2.
\end{aligned}
}
]

This is already a pretty good “classical (\pi)-bond order” functional.

The terms mean:

[
\begin{array}{lll}
\chi_i,\eta_i &:& \text{QEq-like electronegativity and hardness},\
U_i &:& \text{Hubbard spin physics},\
J_{ij} &:& \text{long-range Coulomb},\
-tp &:& \text{quantum resonance / hopping stabilization},\
-\gamma B^2 &:& \text{Peierls bond localization},\
\mu R^2 &:& \text{Pauli / density-matrix representability}.
\end{array}
]

No diagonalization is required.

Everything except Coulomb is local. Coulomb can be FFT/FMM/multigrid.

# 6. Gradients

For the local Pauli residual,

[
R_{i\sigma}
===========

n_{i\sigma}^2
+
\sum_jp_{ij\sigma}^2
--------------------

n_{i\sigma},
]

the derivatives are simple:

[
\frac{\partial R_{i\sigma}}{\partial n_{i\sigma}}
=================================================

2n_{i\sigma}-1,
]

[
\frac{\partial R_{i\sigma}}{\partial p_{ij\sigma}}
==================================================

2p_{ij\sigma}.
]

Therefore the Pauli force on (n_{i\sigma}) is

[
\frac{\partial E_{\mathrm{Pauli}}}{\partial n_{i\sigma}}
========================================================

\mu R_{i\sigma}(2n_{i\sigma}-1).
]

The Pauli force on a bond variable (p_{ij\sigma}) receives contributions from both ends:

[
\frac{\partial E_{\mathrm{Pauli}}}{\partial p_{ij\sigma}}
=========================================================

2\mu p_{ij\sigma}
\left(
R_{i\sigma}
+
R_{j\sigma}
\right).
]

For the hopping and Peierls terms,

[
E_{ij}
======

## -2t_{ij}p_{ij\sigma}

\gamma_{ij}B_{ij}^2,
]

with

[
B_{ij}=p_{ij\uparrow}+p_{ij\downarrow},
]

we get

[
\frac{\partial E_{ij}}{\partial p_{ij\sigma}}
=============================================

## -2t_{ij}

2\gamma_{ij}B_{ij}.
]

So the raw electronic force wants to increase all (p_{ij\sigma}), while the Pauli constraint prevents every bond from becoming maximally occupied.

For the charge variables,

[
\frac{\partial E}{\partial n_{i\sigma}}
=======================================

\chi_i
+
\eta_i(n_i-N_i^0)
+
U_i n_{i\bar\sigma}
-------------------

\sum_j J_{ij}Q_j
+
\mu R_{i\sigma}(2n_{i\sigma}-1)
+
\ldots
]

because

[
Q_i=Z_i^\pi-n_i,
]

so

[
\frac{\partial E_{\mathrm{Coul}}}{\partial n_i}
===============================================

-\sum_j J_{ij}Q_j.
]

# 7. Electron-number constraint

You need to conserve

[
\sum_{i\sigma} n_{i\sigma}=N_\pi.
]

In gradient descent, update using a Lagrange multiplier (\lambda):

[
\dot n_{i\sigma}
================

*

\left(
\frac{\partial E}{\partial n_{i\sigma}}
---------------------------------------

\lambda
\right).
]

Choose (\lambda) so that

[
\sum_{i\sigma}\dot n_{i\sigma}=0.
]

Thus

[
\lambda
=======

\frac{1}{2N_{\mathrm{atoms}}}
\sum_{i\sigma}
\frac{\partial E}{\partial n_{i\sigma}}.
]

This is exactly analogous to charge equilibration: equalize chemical potential while preserving total charge.

# 8. Projection / clipping

After each step, enforce

[
0 \le n_{i\sigma}\le 1.
]

For bond orders, enforce the local Pauli bound

[
\sum_{j\in N(i)}p_{ij\sigma}^2
\le
n_{i\sigma}(1-n_{i\sigma}).
]

A simple projection is:

[
s_{i\sigma}
===========

\sqrt{
\frac{n_{i\sigma}(1-n_{i\sigma})}
{\sum_j p_{ij\sigma}^2+\epsilon}
}.
]

If (s_{i\sigma}<1), rescale all bonds adjacent to (i):

[
p_{ij\sigma}\leftarrow s_{i\sigma}p_{ij\sigma}.
]

Because each bond touches two sites, do this iteratively or symmetrically:

[
p_{ij\sigma}
\leftarrow
\min(s_{i\sigma},s_{j\sigma},1)p_{ij\sigma}.
]

This projection is cheap and local.

# 9. Pseudocode

Using arrays:

```cpp
// Natoms, Nbonds
// edge[e] = {i,j}
// n[2][Natoms]       spin occupancies
// p[2][Nbonds]       spin bond orders
// chi[i], eta[i], U[i], Zpi[i], N0[i]
// t0[e], gamma[e]
// neigh list: atom -> incident bond ids

for(iter=0; iter<maxIter; iter++){
    // --- site charge ---
    for(i=0; i<Natoms; i++){
        double ni = n[0][i] + n[1][i];
        Q[i] = Zpi[i] - ni;
    }

    // --- Coulomb potential phi_i = sum_j J_ij Q_j ---
    computeCoulombPotential(Q, phi); // direct / FFT / FMM

    // --- Pauli residuals R[i][s] ---
    for(s=0; s<2; s++){
        for(i=0; i<Natoms; i++){
            double r = n[s][i]*n[s][i] - n[s][i];
            for(each bond e touching i){
                double pe = p[s][e];
                r += pe*pe;
            }
            R[s][i] = r;
        }
    }

    // --- initialize gradients ---
    clear(gn);
    clear(gp);

    // --- gradients wrt n ---
    for(i=0; i<Natoms; i++){
        double ni = n[0][i] + n[1][i];
        for(s=0; s<2; s++){
            int so = 1-s;
            double g = 0.0;
            g += chi[i];
            g += eta[i]*(ni - N0[i]);
            g += U[i]*n[so][i];
            g += -phi[i]; // d/dn_i of Coulomb
            g += mu*R[s][i]*(2.0*n[s][i] - 1.0);
            gn[s][i] = g;
        }
    }

    // --- gradients wrt bond order p ---
    for(e=0; e<Nbonds; e++){
        int i = edge[e].i;
        int j = edge[e].j;

        double B = p[0][e] + p[1][e];

        for(s=0; s<2; s++){
            double g = 0.0;

            // hopping/resonance
            g += -2.0*t0[e];

            // Peierls effective attraction
            g += -2.0*gamma[e]*B;

            // local Pauli competition from both endpoints
            g += 2.0*mu*p[s][e]*(R[s][i] + R[s][j]);

            gp[s][e] = g;
        }
    }

    // --- conserve total electron number using lambda ---
    double lambda = 0.0;
    int ndof_n = 2*Natoms;
    for(s=0; s<2; s++){
        for(i=0; i<Natoms; i++){
            lambda += gn[s][i];
        }
    }
    lambda /= ndof_n;

    // --- gradient step for n ---
    for(s=0; s<2; s++){
        for(i=0; i<Natoms; i++){
            n[s][i] += -dt_n*(gn[s][i] - lambda);
        }
    }

    // --- gradient step for p ---
    for(s=0; s<2; s++){
        for(e=0; e<Nbonds; e++){
            p[s][e] += -dt_p*gp[s][e];
        }
    }

    // --- clamp n ---
    for(s=0; s<2; s++){
        for(i=0; i<Natoms; i++){
            n[s][i] = clamp(n[s][i], 0.0, 1.0);
        }
    }

    // --- renormalize total electron number after clipping ---
    renormalizeElectronNumber(n, Npi);

    // --- project bond orders to local Pauli bound ---
    for(project_iter=0; project_iter<3; project_iter++){
        for(s=0; s<2; s++){
            for(i=0; i<Natoms; i++){
                double max2 = n[s][i]*(1.0 - n[s][i]);
                double sum2 = 0.0;

                for(each bond e touching i){
                    sum2 += p[s][e]*p[s][e];
                }

                if(sum2 > max2){
                    double scale = sqrt(max2/(sum2 + 1e-12));
                    for(each bond e touching i){
                        p[s][e] *= scale;
                    }
                }
            }
        }
    }

    // --- optional convergence check ---
    double maxForce = maxAbs(gn,gp);
    if(maxForce < Ftol) break;
}
```

This is linear scaling except for Coulomb.

For carbon networks, (z\le 3), so all local operations are (O(N)).

# 10. More physical version with explicit bond lengths

Instead of integrating out the Peierls distortion, you can keep (u_{ij}) as a variable:

[
E_{\mathrm{bond}}
=================

\sum_{\langle ij\rangle}
\left[
\frac{K_{ij}}{2}u_{ij}^2
------------------------

2t_{ij}(u_{ij})B_{ij}
\right].
]

For linear SSH,

[
t_{ij}=t_{0,ij}-\alpha_{ij}u_{ij}.
]

Then the force on (u_{ij}) is

[
\frac{\partial E}{\partial u_{ij}}
==================================

K_{ij}u_{ij}
+
2\alpha_{ij}B_{ij}.
]

Gradient step:

[
u_{ij}
\leftarrow
u_{ij}
------

\Delta t_u
\left(
K_{ij}u_{ij}
+
2\alpha_{ij}B_{ij}
\right).
]

Or solve it instantly:

[
u_{ij}
======

-\frac{2\alpha_{ij}}{K_{ij}}B_{ij}.
]

For most purposes I would eliminate (u_{ij}) analytically and use the effective term

[
-\gamma_{ij}B_{ij}^2.
]

That makes the electronic minimization simpler and faster.

# 11. Relation to Hückel / SSH / density matrix

If you used the full density matrix (P), imposed

[
P^2=P,
\qquad
\operatorname{Tr}P=N,
]

and minimized

[
E=\operatorname{Tr}(PH),
]

you would recover the normal occupied molecular orbitals. Diagonalization is just one way of doing this.

Here you do something different:

You store only

[
P_{ii},\quad P_{ij}\text{ for bonds}
]

and enforce idempotency approximately and locally. This turns quantum mechanics into a local constrained field theory.

So the hierarchy is:

[
\text{MO diagonalization}
\quad\rightarrow\quad
\text{density matrix minimization}
\quad\rightarrow\quad
\text{sparse local density-matrix functional}
\quad\rightarrow\quad
\text{classical bond-order field model}.
]

Your desired model lives in the last two levels.

# 12. What this model should capture

It should capture qualitatively:

[
\begin{array}{ll}
\text{aromatic delocalization} & \text{via } -tB,\
\text{Kekulé localization} & \text{via } -\gamma B^2,\
\text{heteroatom charge transfer} & \text{via } \chi_i,\eta_i,N_i^0,\
\text{spin-polarized edge states} & \text{via } U n_\uparrow n_\downarrow,\
\text{Coulomb frustration} & \text{via } Q_iJ_{ij}Q_j,\
\text{Pauli bond competition} & \text{via } P^2\approx P,\
\text{local chemical boundary conditions} & \text{via } t,\chi,U,N^0 \text{ changes}.
\end{array}
]

It will not exactly reproduce the full spectrum. But that is not your goal. Your goal is local energetics, bond pattern, charge redistribution, hydrogenation response, and Peierls/Kekulé texture.

# 13. A still simpler “first implementation” model

For debugging, I would start even simpler:

[
E =
-2t\sum_{\langle ij\rangle,\sigma}p_{ij\sigma}
----------------------------------------------

\gamma\sum_{\langle ij\rangle}
(p_{ij\uparrow}+p_{ij\downarrow})^2
+
\frac{\mu}{2}
\sum_{i,\sigma}
\left[
n_{i\sigma}^2
+
\sum_jp_{ij\sigma}^2
--------------------

n_{i\sigma}
\right]^2
+
\sum_i U n_{i\uparrow}n_{i\downarrow}.
]

Set neutral graphene:

[
n_{i\uparrow}=n_{i\downarrow}=1/2.
]

At first, freeze (n) and optimize only (p). Then the model becomes a pure bond-order field theory:

[
E[p]
====

## -2t\sum_{\langle ij\rangle}B_{ij}

\gamma\sum_{\langle ij\rangle}B_{ij}^2
+
\frac{\mu}{2}
\sum_{i,\sigma}
\left[
\sum_jp_{ij\sigma}^2-\frac{1}{4}
\right]^2.
]

This already gives competition between distributing bond order over three bonds versus localizing it into one or two stronger bonds.

After this works, unfreeze (n_i), add heteroatoms, then Coulomb, then spin.

# 14. Main conceptual warning

The model needs a term that enforces something like

[
P^2\approx P.
]

Without this, bond charges are too free. Then Peierls attraction will collapse the solution into unphysical overbonding.

So the minimal physically meaningful structure is not merely

[
E[n,B],
]

but rather

[
E[n,p] + \text{local representability constraint}.
]

That is the key ingredient that makes this more than just an empirical valence-bond force field.

The most useful final form is therefore:

[
\boxed{
\begin{aligned}
E[n,p]
=&
E_{\mathrm{QEq}}[n]
+
E_{\mathrm{Hub}}[n_\uparrow,n_\downarrow]
+
E_{\mathrm{Coul}}[n]
\
&-
2\sum_{\langle ij\rangle,\sigma}
t_{ij}p_{ij\sigma}
------------------

\sum_{\langle ij\rangle}
\gamma_{ij}
\left(p_{ij\uparrow}+p_{ij\downarrow}\right)^2
\
&+
\frac{\mu}{2}
\sum_{i,\sigma}
\left[
n_{i\sigma}^2
+
\sum_jp_{ij\sigma}^2
--------------------

n_{i\sigma}
\right]^2.
\end{aligned}
}
]

This is probably the best starting point for your intended “classical but quantum-motivated” (\pi)-bond occupancy optimizer.

---

# USER

On one hand I agree that adding idempotendy to preserve pauli exclusion principle is usefull, perhaps exacly by the constran 0 < pij < 1 or somthing like that.

But I do not agree that without it system will collapse to add double bond to all bonds, because we have oter contrains preventing it, and we should explicitly consider these in our minimizer

1) number of electrons - our bond occupancy is simply moving electrons around, but we have fixed amout of electrons and the total charge must sum up to this total number.

2) lattice distortion and position of atoms. We should explicitly consider displacment of atoms from its ideal position as dynamical variable (DOF) and if one bond get shorter (localization of electron in that bond) the neighbor bonds probably get longer. Or should we let the whole molecule relax (like moving absolute atomic coordinates in free space with some forcefield for sigma bonds? that is also possible but we will do this only in next step)

3) hubard U - is we put double bonds on all bonds attached to an atom, its occupancy would be higher than 1 which will be penalized by hubard U. 

check if our model represent these physical principles eventually introduce physical terms for this. 

---

# Chat GPT 5.5

Yes — your objections are physically correct. The previous formulation was too “density-matrix purist”: it treated (p_{ij}) as an almost free off-diagonal density variable, and then idempotency became the main thing preventing overbonding. But if we instead formulate the model in terms of **electron allocation between sites and bonds**, then the conservation laws and elastic compatibility already do much of the job.

The better model should distinguish three related quantities:

[
n_i \quad \text{site/lone/localized occupancy}
]

[
b_{ij} \quad \text{bond occupancy / covalent electron-pair density}
]

[
r_{ij} \quad \text{bond length / sigma-lattice distortion}
]

Then the physical constraint is not merely (0<p_{ij}<1), but rather a **local electron-counting rule**:

[
\boxed{
n_i + \frac{1}{2}\sum_{j\in N(i)} b_{ij} = N_i^\pi
}
]

or, if charge is allowed to redistribute,

[
\boxed{
n_i + \frac{1}{2}\sum_{j\in N(i)} b_{ij} = N_i^\pi + \delta q_i
}
]

where (N_i^\pi) is the nominal number of (\pi)-electrons supplied by atom (i), and (\delta q_i) is charge transfer.

This is much closer to chemistry than treating (p_{ij}) as an unconstrained density-matrix element.

For neutral carbon,

[
N_i^\pi=1.
]

If carbon forms one strong (\pi)-bond to a neighbor, then roughly

[
b_{ij}\sim 1
]

and that bond consumes about half an electron from each endpoint:

[
\frac{1}{2}b_{ij}\sim 0.5.
]

So if a carbon tried to make three full (\pi)-bonds,

[
\frac{1}{2}(b_{i1}+b_{i2}+b_{i3}) = \frac{3}{2},
]

which already exceeds its available one (\pi)-electron before even counting (n_i). This is the valence/electron-counting constraint you are pointing to.

So, yes: **electron conservation plus local bond-electron bookkeeping already prevents every bond from becoming a double bond**.

---

# 1. Better choice of variables

I would define the primary classical variables as:

[
b_{ij\sigma}\ge 0
]

bond electron density in spin channel (\sigma), and

[
n_{i\sigma}\ge 0
]

localized/site electron density in spin channel (\sigma).

Total bond order:

[
B_{ij}=b_{ij\uparrow}+b_{ij\downarrow}.
]

Total site occupancy:

[
n_i=n_{i\uparrow}+n_{i\downarrow}.
]

Total (\pi)-electron population associated with atom (i):

[
\rho_i =
n_i+\frac{1}{2}\sum_{j\in N(i)}B_{ij}.
]

This is the important quantity. It is the number of (\pi)-electrons “owned” by atom (i), including half of each adjacent bond.

Then the site charge is

[
Q_i=Z_i^\pi-\rho_i.
]

So instead of saying

[
Q_i=Z_i^\pi-n_i,
]

we say

[
\boxed{
Q_i=Z_i^\pi-
\left(
n_i+\frac{1}{2}\sum_jB_{ij}
\right)
}
]

This is much more chemically meaningful.

It means that putting electrons into a bond automatically changes the charge balance of the two atoms involved.

---

# 2. Global electron-number constraint

The total number of electrons is

[
N_{\mathrm{el}}
===============

\sum_i n_i+\sum_{\langle ij\rangle}B_{ij}.
]

Why no factor (1/2) in the second term? Because each bond variable (B_{ij}) is the total number of electrons in that bond. Locally each atom owns half, but globally the bond electrons are counted once.

So impose

[
\boxed{
\sum_i n_i+\sum_{\langle ij\rangle}B_{ij}
=========================================

N_\pi^{\mathrm{tot}}
}
]

where

[
N_\pi^{\mathrm{tot}}=\sum_i Z_i^\pi
]

for a neutral system.

This directly implements your point 1.

---

# 3. Bounds on bond occupancy

For a two-center (\pi)-bond, a natural bound is

[
0\le b_{ij\sigma}\le 1.
]

Therefore

[
0\le B_{ij}\le 2.
]

But in aromatic carbon systems, the (\pi)-bond order is usually between 0 and 1 in the valence-bond sense. If (B_{ij}) is interpreted as **extra (\pi)-bond electron-pair occupancy**, then use

[
0\le B_{ij}\le 1.
]

This is a matter of convention.

I would choose:

[
\boxed{
0\le B_{ij}\le 1
}
]

where

[
B_{ij}=1
]

means a full local (\pi)-bond / double-bond-like occupancy, and

[
B_{ij}=1/3 \text{ to } 2/3
]

means aromatic delocalized bond order.

Then the electron counting becomes:

[
\rho_i=n_i+\sum_j w_{ij}B_{ij}.
]

The value of (w) depends on convention.

If (B_{ij}=1) means one full (\pi)-bond made from two electrons total, then each atom contributes one electron to it only if the bond is full. Then

[
w_{ij}=1
]

not (1/2). In that convention,

[
\rho_i=n_i+\sum_jB_{ij}.
]

For benzene, each carbon has two adjacent bonds with something like (B\approx 1/2), giving

[
\rho_i \approx \frac{1}{2}+\frac{1}{2}=1
]

if (n_i\approx 0). That works.

So I suggest this convention:

[
\boxed{
\rho_i = n_i+\sum_{j\in N(i)}B_{ij}
}
]

where (B_{ij}) is the fraction of one electron from atom (i) committed into the (ij) (\pi)-bond. The bond is symmetric, so the other atom also contributes (B_{ij}). A full (\pi)-bond has

[
B_{ij}=1
]

and contains two (\pi)-electrons total.

Then the total number of (\pi)-electrons is

[
\boxed{
N_{\mathrm{el}}
===============

\sum_i n_i + \sum_{\langle ij\rangle}2B_{ij}
}
]

because each bond contains (2B_{ij}) electrons.

For a neutral carbon system,

[
\sum_i \rho_i
=============

\sum_i
\left(
n_i+\sum_jB_{ij}
\right)
=======

N_C.
]

Since every bond is counted twice in the local sum,

[
\sum_i n_i+2\sum_{\langle ij\rangle}B_{ij}=N_C.
]

That is internally consistent.

This is probably the best convention for your purpose.

---

# 4. Energy functional in electron-allocation variables

Now write

[
E =
E_{\mathrm{atom}}
+
E_{\mathrm{bond}}
+
E_{\mathrm{charge}}
+
E_{\mathrm{Hubbard}}
+
E_{\mathrm{elastic}}
+
E_{\mathrm{compat}}.
]

The local population at atom (i) is

[
\rho_i = n_i+\sum_jB_{ij}.
]

The charge is

[
Q_i=Z_i^\pi-\rho_i.
]

The total-electron constraint is automatically equivalent to

[
\sum_i \rho_i = \sum_i Z_i^\pi
]

for neutral systems, or

[
\sum_i \rho_i = \sum_i Z_i^\pi - Q_{\mathrm{tot}}
]

for charged systems.

---

## 4.1 Atomic onsite/QEq term

[
E_{\mathrm{atom}}
=================

\sum_i
\left[
\chi_i(\rho_i-Z_i^\pi)
+
\frac{\eta_i}{2}(\rho_i-Z_i^\pi)^2
\right].
]

This penalizes excessive charge transfer.

It also makes nitrogen, oxygen, boron, charged edges, protonated sites, etc. easy to include.

---

## 4.2 Coulomb term

[
E_{\mathrm{Coul}}
=================

\frac{1}{2}
\sum_{ij}
Q_iJ_{ij}Q_j.
]

with

[
Q_i=Z_i^\pi-\rho_i.
]

Important: now a strong bond changes the charges of both endpoint atoms because it consumes local (\pi)-electron density from them.

---

## 4.3 Bond resonance / hopping energy

A bond with occupancy (B_{ij}) gains covalent resonance energy.

Simplest form:

[
E_{\mathrm{bond,elec}}
======================

-\sum_{\langle ij\rangle}
\varepsilon_{ij}(r_{ij}) B_{ij}.
]

Here (\varepsilon_{ij}) is the energy gained per unit bond order. It is related to hopping (t_{ij}).

For a Hückel two-site bond, the bonding-antibonding splitting is roughly (2t), so (\varepsilon_{ij}) should be of order (2t), but you can absorb conventions into the parameter.

A better saturating form is:

[
E_{\mathrm{bond,elec}}
======================

-\sum_{\langle ij\rangle}
\varepsilon_{ij}(r_{ij}) B_{ij}
+
\sum_{\langle ij\rangle}
\frac{A_{ij}}{2}B_{ij}^2.
]

The positive (A B^2) prevents too much concentration if needed. But if electron counting and Hubbard are strong enough, you may not need it.

---

## 4.4 Hubbard term

The Hubbard penalty should act on local population (\rho_i), not only on (n_i), because bond electrons still live partly on the atom.

For spin variables,

[
\rho_{i\sigma}=n_{i\sigma}+\sum_jB_{ij\sigma}.
]

Then

[
\boxed{
E_U=
\sum_i U_i \rho_{i\uparrow}\rho_{i\downarrow}
}
]

This is better than applying (U) only to localized site electrons.

It means: if one atom participates in too many bonds, both spin populations on it become large, and (U) penalizes that.

For closed-shell non-spin-polarized approximation,

[
\rho_{i\uparrow}=\rho_{i\downarrow}=\frac{\rho_i}{2},
]

so

[
E_U
===

\sum_i U_i\frac{\rho_i^2}{4}.
]

But if you include spin,

[
\rho_i=\rho_{i\uparrow}+\rho_{i\downarrow},
\qquad
m_i=\rho_{i\uparrow}-\rho_{i\downarrow},
]

then

[
E_U=
\sum_i
\frac{U_i}{4}
\left(
\rho_i^2-m_i^2
\right).
]

So spin polarization lowers Hubbard cost, which can generate graphene zigzag-edge magnetism qualitatively.

---

# 5. Explicit Peierls/lattice distortion

Here I fully agree with your point 2: if a bond gets shorter, neighboring bonds cannot all independently get shorter if atoms move in real space. The compatibility of atomic coordinates is a real physical constraint.

There are two levels.

---

## Level A: independent bond distortions

Use bond variables (u_{ij}):

[
r_{ij}=r_{0,ij}+u_{ij}.
]

Energy:

[
E_{\mathrm{elast}}
==================

\sum_{\langle ij\rangle}
\frac{K_{ij}}{2}u_{ij}^2.
]

Electronic bond energy:

[
E_{\mathrm{bond}}
=================

-\sum_{\langle ij\rangle}
\varepsilon_{ij}(u_{ij})B_{ij}.
]

With

[
\varepsilon_{ij}(u)=\varepsilon_{0,ij}-g_{ij}u,
]

where (u>0) means longer bond, so longer bond reduces bonding energy.

Then

[
E_{ij}
======

## \frac{K_{ij}}{2}u_{ij}^2

(\varepsilon_{0,ij}-g_{ij}u_{ij})B_{ij}.
]

Derivative:

[
\frac{\partial E}{\partial u_{ij}}
==================================

K_{ij}u_{ij}+g_{ij}B_{ij}.
]

Minimum:

[
\boxed{
u_{ij}^*
========

-\frac{g_{ij}}{K_{ij}}B_{ij}
}
]

So strong bond order shortens the bond.

But with independent (u_{ij}), all bonds could shorten simultaneously. This captures local Peierls but not geometric compatibility.

---

## Level B: atom-coordinate lattice

Better: make atomic positions real variables

[
\mathbf R_i=\mathbf R_i^0+\mathbf u_i.
]

Bond length:

[
r_{ij}=|\mathbf R_j-\mathbf R_i|.
]

Then the sigma skeleton energy is something like

[
E_\sigma
========

\sum_{\langle ij\rangle}
\frac{K_r}{2}(r_{ij}-r_0)^2
+
\sum_{\angle jik}
\frac{K_\theta}{2}(\theta_{jik}-\theta_0)^2
+
E_{\mathrm{oop}}.
]

For (sp^2) carbon,

[
\theta_0=120^\circ.
]

The (\pi)-bond energy is

[
E_{\pi\text{-bond}}
===================

-\sum_{\langle ij\rangle}
\varepsilon(r_{ij})B_{ij}.
]

Then if one bond shortens, the atom positions shift, and this automatically stretches/compresses neighboring bonds and bends angles.

This is much more physical than independent (u_{ij}). It gives the compatibility constraint for free.

The force on atom (i) from one (\pi)-bond term is

[
\mathbf F_i^{\pi,ij}
====================

-\frac{\partial}{\partial \mathbf R_i}
\left[-\varepsilon(r_{ij})B_{ij}\right]
=======================================

B_{ij}
\frac{d\varepsilon}{dr}
\frac{\mathbf R_i-\mathbf R_j}{r_{ij}}.
]

Since

[
\frac{d\varepsilon}{dr}<0,
]

larger (B_{ij}) pulls atoms (i,j) together.

This is the cleanest Peierls formulation:

[
\boxed{
E[\mathbf R,B,n]
================

## E_\sigma[\mathbf R]

\sum_{\langle ij\rangle}\varepsilon(r_{ij})B_{ij}
+
E_{\mathrm{charge/Hubbard}}[\rho]
}
]

with

[
\rho_i=n_i+\sum_jB_{ij}.
]

No matrix algebra.

---

# 6. Valence/electron-allocation constraint

The most important missing term is not idempotency; it is the **local valence/electron budget**.

For each atom:

[
\rho_i = n_i+\sum_j B_{ij}.
]

There are two possible ways to impose this.

## Hard constraint

For fixed local valence:

[
\boxed{
n_i+\sum_jB_{ij}=Z_i^\pi
}
]

This is appropriate if you do not want charge transfer.

Then

[
n_i = Z_i^\pi-\sum_jB_{ij}.
]

You can eliminate (n_i) entirely.

For carbon:

[
n_i=1-\sum_jB_{ij}.
]

Then if carbon tries to put (B=1) on three bonds,

[
n_i=1-3=-2,
]

impossible. You simply reject or project it.

This is the strongest and simplest anti-overbonding mechanism.

## Soft constraint

Allow charge transfer, but penalize deviations:

[
E_{\mathrm{val}}
================

\sum_i
\frac{k_i}{2}
\left(
n_i+\sum_jB_{ij}-Z_i^\pi
\right)^2.
]

This is basically QEq hardness.

Then charge transfer is possible, but expensive.

This is better for heteroatoms and charged/protonated systems.

---

# 7. Revised minimal functional

I would now write the model as:

[
\boxed{
\begin{aligned}
E
=&
E_\sigma[\mathbf R]
-------------------

\sum_{\langle ij\rangle}
\varepsilon_{ij}(r_{ij})B_{ij}
\
&+
\sum_i
\left[
\chi_i(\rho_i-Z_i^\pi)
+
\frac{\eta_i}{2}(\rho_i-Z_i^\pi)^2
\right]
\
&+
\frac{1}{2}
\sum_{ij}
Q_iJ_{ij}Q_j
\
&+
\sum_i
U_i\rho_{i\uparrow}\rho_{i\downarrow}
\
&+
E_{\mathrm{bond;saturation}}
+
E_{\mathrm{optional;Pauli}}.
\end{aligned}
}
]

where

[
\rho_{i\sigma}=n_{i\sigma}+\sum_jB_{ij\sigma},
]

[
\rho_i=\rho_{i\uparrow}+\rho_{i\downarrow},
]

[
Q_i=Z_i^\pi-\rho_i.
]

The essential constraints are:

[
0\le B_{ij\sigma}\le 1,
]

[
0\le \rho_{i\sigma}\le 1
]

for a single (p_z) orbital per spin,

and

[
\sum_i\rho_i=N_\pi^{\mathrm{tot}}.
]

Actually, for carbon with one (p_z) orbital, the Pauli principle at the site level is simply:

[
0\le \rho_{i\uparrow}\le 1,
\qquad
0\le \rho_{i\downarrow}\le 1.
]

That is a very transparent version of Pauli.

---

# 8. What remains of idempotency?

I would now demote idempotency from “central required ingredient” to “optional quantum correction.”

The hierarchy becomes:

1. **Electron conservation** prevents infinite bonding globally.
2. **Local valence constraint** prevents one atom from using the same electron in several bonds.
3. **Hubbard (U)** penalizes putting too much opposite-spin density on one atom.
4. **Atomic-coordinate elasticity** prevents all bonds around a site from shortening independently.
5. **Bond saturation / Pauli bounds** prevent (B_{ij}) exceeding chemically meaningful limits.
6. **Idempotency-like penalties** improve quantum representability but are not the first line of defense.

So instead of

[
P^2=P
]

as a main constraint, use simpler local Pauli bounds:

[
0\le \rho_{i\sigma}\le 1,
]

[
0\le B_{ij\sigma}\le B_{\max},
]

and optionally:

[
\sum_jB_{ij\sigma}\le \rho_{i\sigma}.
]

This last one is very important.

If (B_{ij\sigma}) means the amount of spin-(\sigma) electron from atom (i) participating in bond (ij), then:

[
\boxed{
n_{i\sigma}+\sum_jB_{ij\sigma}=\rho_{i\sigma}
}
]

and

[
0\le\rho_{i\sigma}\le 1.
]

This is a local Pauli/valence constraint without density-matrix idempotency.

---

# 9. Recommended variable convention

To avoid ambiguity, I suggest the following interpretation:

[
x_{ij\sigma}^{(i)}
]

means: amount of spin-(\sigma) electron donated by atom (i) into bond (ij).

For a symmetric covalent bond,

[
x_{ij\sigma}^{(i)} \approx x_{ij\sigma}^{(j)}.
]

Define bond spin occupancy as the geometric mean:

[
B_{ij\sigma}
============

\sqrt{
x_{ij\sigma}^{(i)}
x_{ij\sigma}^{(j)}
}.
]

or simpler, if you enforce symmetry,

[
x_{ij\sigma}^{(i)}=x_{ij\sigma}^{(j)}=B_{ij\sigma}.
]

Then local electron budget is

[
n_{i\sigma}+\sum_jx_{ij\sigma}^{(i)}=\rho_{i\sigma}.
]

For a single orbital,

[
0\le\rho_{i\sigma}\le1.
]

This formulation makes it impossible for one electron to be reused in three bonds.

A bond energy should depend on both endpoints:

[
E_{ij}^{\mathrm{bond}}
======================

-\varepsilon_{ij}(r_{ij})
\sum_\sigma
2\sqrt{
x_{ij\sigma}^{(i)}
x_{ij\sigma}^{(j)}
}.
]

Why geometric mean? Because a covalent bond needs electron amplitude from both atoms. If one side contributes zero, the bond vanishes.

This is very close to tight-binding intuition:

[
\text{bond stabilization} \sim -t, c_i c_j.
]

But now (x_i\sim c_i^2), so

[
c_i c_j\sim \sqrt{x_i x_j}.
]

Therefore this term is quite quantum-motivated while using only positive classical populations.

This may be the cleanest bridge between your original Mulliken idea (S_{ij}c_i c_j) and a purely classical occupancy model.

---

# 10. Very good final form

Use directed half-bond variables:

[
x_{i\to j,\sigma}\ge0.
]

Local spin population:

[
\rho_{i\sigma}=n_{i\sigma}+\sum_{j\in N(i)}x_{i\to j,\sigma}.
]

Pauli bound:

[
0\le\rho_{i\sigma}\le1.
]

Charge:

[
Q_i=Z_i^\pi-\sum_\sigma\rho_{i\sigma}.
]

Bond order:

[
B_{ij}
======

\sum_\sigma
2\sqrt{x_{i\to j,\sigma}x_{j\to i,\sigma}}.
]

Then the energy:

[
\boxed{
\begin{aligned}
E
=&
E_\sigma[\mathbf R]
-------------------

\sum_{\langle ij\rangle}
\varepsilon_{ij}(r_{ij})
\sum_\sigma
2\sqrt{x_{i\to j,\sigma}x_{j\to i,\sigma}}
\
&+
\sum_i
\left[
\chi_i(\rho_i-Z_i^\pi)
+
\frac{\eta_i}{2}(\rho_i-Z_i^\pi)^2
\right]
\
&+
\frac{1}{2}
\sum_{ij}Q_iJ_{ij}Q_j
+
\sum_iU_i\rho_{i\uparrow}\rho_{i\downarrow}.
\end{aligned}
}
]

This is excellent because:

[
\begin{array}{ll}
\text{fixed electron number} & \text{via } \sum_i\rho_i=N_\pi,\
\text{local Pauli} & \text{via } 0\le\rho_{i\sigma}\le1,\
\text{valence competition} & \text{via } n_i+\sum_jx_{i\to j}=\rho_i,\
\text{bond needs two atoms} & \text{via } \sqrt{x_{i\to j}x_{j\to i}},\
\text{Peierls shortening} & \text{via } \varepsilon(r_{ij}),\
\text{sigma resistance} & \text{via } E_\sigma[\mathbf R],\
\text{Hubbard magnetism} & \text{via } U\rho_\uparrow\rho_\downarrow,\
\text{electrostatics} & \text{via } QJQ.
\end{array}
]

No diagonalization, no matrix algebra, and still very quantum-like.

---

# 11. Gradients for half-bond model

Let

[
E_{ij\sigma}^{\mathrm{bond}}
============================

-2\varepsilon_{ij}\sqrt{x_{i\to j,\sigma}x_{j\to i,\sigma}}.
]

Then

[
\frac{\partial E}{\partial x_{i\to j,\sigma}}
=============================================

-\varepsilon_{ij}
\sqrt{
\frac{x_{j\to i,\sigma}}
{x_{i\to j,\sigma}+\epsilon}
}.
]

This has the nice property that if one side contributes much less than the other, the force pulls it up strongly. It symmetrizes the covalent bond.

The charge/QEq derivative contributes through

[
\rho_{i\sigma}=n_{i\sigma}+\sum_jx_{i\to j,\sigma}.
]

Define local chemical potential

[
\mu_{i\sigma}
=============

\frac{\partial E_{\mathrm{charge+Hub}}}{\partial \rho_{i\sigma}}.
]

Then

[
\frac{\partial E}{\partial x_{i\to j,\sigma}}
=============================================

-\varepsilon_{ij}
\sqrt{
\frac{x_{j\to i,\sigma}}
{x_{i\to j,\sigma}+\epsilon}
}
+
\mu_{i\sigma}.
]

where

[
\mu_{i\sigma}
=============

\chi_i
+
\eta_i(\rho_i-Z_i^\pi)
-\phi_i
+
U_i\rho_{i\bar\sigma}.
]

Here

[
\phi_i=\sum_jJ_{ij}Q_j.
]

The derivative with respect to local nonbonding density (n_{i\sigma}) is simply

[
\frac{\partial E}{\partial n_{i\sigma}}=\mu_{i\sigma}.
]

So the minimization equalizes the benefit of placing an electron in a bond versus leaving it localized on a site.

This is exactly QEq logic, but enriched with covalent bond formation.

---

# 12. Pseudocode for the improved model

```cpp
// Variables:
// n[i][s]             localized pi electron on atom i, spin s
// x[e][side][s]       half-bond electron donation
//                      side=0 means i->j, side=1 means j->i
// R[i]                atom coordinates
//
// rho[i][s] = n[i][s] + sum half-bonds leaving atom i
// Q[i]      = Zpi[i] - rho[i][up] - rho[i][dn]

for(iter=0; iter<maxIter; iter++){

    // --- compute local spin populations ---
    for(i=0; i<N; i++){
        rho[i][0] = n[i][0];
        rho[i][1] = n[i][1];
    }

    for(e=0; e<Nb; e++){
        int i = bond[e].i;
        int j = bond[e].j;
        for(s=0; s<2; s++){
            rho[i][s] += x[e][0][s]; // i -> j
            rho[j][s] += x[e][1][s]; // j -> i
        }
    }

    // --- charges ---
    for(i=0; i<N; i++){
        double rhoi = rho[i][0] + rho[i][1];
        Q[i] = Zpi[i] - rhoi;
    }

    computeCoulombPotential(Q, phi); // phi_i = sum_j J_ij Q_j

    // --- local chemical potentials ---
    for(i=0; i<N; i++){
        double rhoi = rho[i][0] + rho[i][1];

        for(s=0; s<2; s++){
            int so = 1-s;

            mu_loc[i][s]  = chi[i];
            mu_loc[i][s] += eta[i]*(rhoi - Zpi[i]);
            mu_loc[i][s] += -phi[i];
            mu_loc[i][s] += U[i]*rho[i][so];
        }
    }

    // --- gradients for localized electrons ---
    for(i=0; i<N; i++){
        for(s=0; s<2; s++){
            gn[i][s] = mu_loc[i][s];
        }
    }

    // --- gradients for half-bond variables and atomic forces ---
    clear(gx);
    clear(F);

    for(e=0; e<Nb; e++){
        int i = bond[e].i;
        int j = bond[e].j;

        Vec3 d = R[j] - R[i];
        double r = norm(d);
        Vec3 h = d / r;

        double eps  = eps0[e] * exp(-beta[e]*(r-r0[e]));
        double deps = -beta[e]*eps;

        for(s=0; s<2; s++){
            double xi = x[e][0][s];
            double xj = x[e][1][s];

            double sq = sqrt((xi+small)*(xj+small));

            // E_bond = -2 eps sqrt(xi*xj)

            gx[e][0][s] += mu_loc[i][s] - eps*sqrt((xj+small)/(xi+small));
            gx[e][1][s] += mu_loc[j][s] - eps*sqrt((xi+small)/(xj+small));

            // force from bond electronic energy
            // dE/dr = -2 deps sqrt(xi*xj)
            double dEdr = -2.0*deps*sq;

            Vec3 fij = -dEdr*h;
            F[i] += fij;
            F[j] -= fij;
        }
    }

    // --- add sigma skeleton forces ---
    addSigmaBondForces(R, F);       // bond stretch
    addSigmaAngleForces(R, F);      // sp2 angle stiffness
    addOutOfPlaneForces(R, F);      // optional

    // --- electron-number constraint ---
    // Apply global lambda to all electron variables n and x.
    double gsum = 0.0;
    int ndof = 0;

    for(i=0; i<N; i++){
        for(s=0; s<2; s++){
            gsum += gn[i][s];
            ndof++;
        }
    }

    for(e=0; e<Nb; e++){
        for(side=0; side<2; side++){
            for(s=0; s<2; s++){
                gsum += gx[e][side][s];
                ndof++;
            }
        }
    }

    double lambda = gsum / ndof;

    // --- gradient descent update ---
    for(i=0; i<N; i++){
        for(s=0; s<2; s++){
            n[i][s] -= dt_e*(gn[i][s] - lambda);
        }
    }

    for(e=0; e<Nb; e++){
        for(side=0; side<2; side++){
            for(s=0; s<2; s++){
                x[e][side][s] -= dt_e*(gx[e][side][s] - lambda);
            }
        }
    }

    for(i=0; i<N; i++){
        R[i] += dt_R * F[i];
    }

    // --- positivity projection ---
    for(i=0; i<N; i++){
        for(s=0; s<2; s++){
            n[i][s] = max(0.0, n[i][s]);
        }
    }

    for(e=0; e<Nb; e++){
        for(side=0; side<2; side++){
            for(s=0; s<2; s++){
                x[e][side][s] = max(0.0, x[e][side][s]);
            }
        }
    }

    // --- local Pauli projection: rho_i_sigma <= 1 ---
    projectLocalSpinOccupancy(n, x, maxOcc=1.0);

    // --- global electron normalization ---
    renormalizeTotalElectrons(n, x, Npi_tot);
}
```

---

# 13. What changed relative to the previous answer

The previous density-matrix model said:

[
p_{ij}
]

is an off-diagonal density-matrix element, so Pauli/idempotency is central.

The improved electron-allocation model says:

[
x_{i\to j,\sigma}
]

is an actual fraction of electron density from atom (i) committed to bond (ij). Then overbonding is prevented primarily by:

[
\boxed{
n_{i\sigma}+\sum_jx_{i\to j,\sigma}\le 1
}
]

This is the direct local version of Pauli and valence conservation.

Then the quantum hopping amplitude is mimicked by

[
\boxed{
E_{ij}\sim -2\varepsilon(r_{ij})\sqrt{x_{i\to j}x_{j\to i}}
}
]

instead of a linear (-tp_{ij}).

That square root is very important: it keeps the (c_i c_j) nature of hopping while using classical positive occupancies.

---

# 14. Does this still describe aromaticity?

Yes, at least qualitatively.

For benzene-like carbon, each atom has one (\pi)-electron. It can distribute that electron among its two neighbors:

[
x_{i\to i-1}+x_{i\to i+1}+n_i=1.
]

If the geometry is fixed and all bonds equal, a delocalized aromatic solution has approximately equal sharing:

[
x_{i\to i-1}\approx x_{i\to i+1}.
]

If Peierls coupling is strong, one bond shortens, its (\varepsilon(r)) increases, and the electrons preferentially flow into alternating bonds:

[
B_{12},B_{34},B_{56} \uparrow,
\qquad
B_{23},B_{45},B_{61} \downarrow.
]

The sigma skeleton resists this distortion.

So the competition is:

[
\text{hopping/resonance wants delocalization}
]

versus

[
\text{Peierls electron-lattice coupling wants bond alternation}
]

versus

[
\text{sigma elasticity resists bond alternation}.
]

That is exactly the SSH/Kekulé physics you want, but expressed without orbitals or diagonalization.

---

# 15. Final recommendation

I would not start from (p_{ij}=S_{ij}c_ic_j) as the primary variable.

I would start from **half-bond electron allocations**:

[
x_{i\to j,\sigma}.
]

Then define the effective quantum bond order as

[
B_{ij}
======

\sum_\sigma
2\sqrt{x_{i\to j,\sigma}x_{j\to i,\sigma}}.
]

This gives you:

[
\text{classical variables}
\quad+\quad
\text{quantum-like hopping}
\quad+\quad
\text{local electron conservation}
\quad+\quad
\text{Hubbard spin physics}
\quad+\quad
\text{Peierls geometry}.
]

The compact final model is:

[
\boxed{
\begin{aligned}
E
=&
E_\sigma[\mathbf R]
-------------------

\sum_{\langle ij\rangle,\sigma}
2\varepsilon_{ij}(r_{ij})
\sqrt{x_{i\to j,\sigma}x_{j\to i,\sigma}}
\
&+
\sum_i
\left[
\chi_i(\rho_i-Z_i^\pi)
+
\frac{\eta_i}{2}(\rho_i-Z_i^\pi)^2
+
U_i\rho_{i\uparrow}\rho_{i\downarrow}
\right]
\
&+
\frac{1}{2}
\sum_{ij}
Q_iJ_{ij}Q_j,
\end{aligned}
}
]

with

[
\rho_{i\sigma}
==============

n_{i\sigma}
+
\sum_jx_{i\to j,\sigma},
]

[
Q_i=Z_i^\pi-\rho_i,
]

[
0\le \rho_{i\sigma}\le1,
]

[
\sum_i\rho_i=N_\pi^{\mathrm{tot}}.
]

This is, I think, the more natural version of your idea. It is not a density-matrix minimizer anymore. It is a **classical valence-bond/QEq/SSH hybrid**, with covalent stabilization modeled by the geometric mean of half-bond electron donations.

---

# USER

this is very nice, I like it, but we should try to define it in term of local updates of each atom. We should clearly define what are our indepndnet degrees of freedom (dynamical variables) which we optimize and keep persistent in memory (thry carry state of the system) and what are just axuliary quntities intermediate in computation which we discard each gradient descent iteration.

then define local rules - how we evolve these DOFs, which means primarlily compute variational derivative (generalized force) acting on each DOF (grandinet of energy with respect to that DOF).

once we have this we can rewrite the algorithm in highly parallel manner where each DOF is evaluated by single thread (gather rather then scatter operation) like inc parralezied classicalforcefield. I want to implement it ultimately as OpenCL kernel. 

therefore there should not be loops over DOFs (over all atoms in the sytem) as this is global_id of the parralel OpenCL kernel. But we should iterate for given atom over its neighbors. Also identify points where we need global reduction or other oprations which break this framework as this is bottleneck for GPU execution (synchronization points).

---

# ChatGPT 5.5

Yes. For OpenCL/GPU implementation we should formulate the model in an **atom-local state representation**, where each atom owns its own electron budget and its outgoing half-bond variables. Then every kernel can be written mostly as:

[
\text{one work item} \leftrightarrow \text{one atom}
]

or

[
\text{one work item} \leftrightarrow \text{one half-bond}.
]

The key is to avoid “scatter” updates like “bond thread adds force to atom (i) and atom (j).” Instead, each atom gathers information from its own neighbors and computes its own gradient/force.

# 1. Persistent dynamical variables

For each atom (i), with neighbors (j\in N(i)), define local spin-resolved allocation variables

[
y_{i0\sigma}
]

for localized/nonbonding (\pi)-electron density on atom (i), and

[
y_{ik\sigma}
]

for the amount of spin-(\sigma) electron from atom (i) donated into the half-bond pointing from (i) to its (k)-th neighbor.

Here (k=1,\dots,z_i), where (z_i\le 3) for graphene-like systems.

So for each atom and spin we store a small local vector:

[
\mathbf y_{i\sigma}
===================

\left(
y_{i0\sigma},
y_{i1\sigma},
y_{i2\sigma},
y_{i3\sigma}
\right).
]

Interpretation:

[
y_{i0\sigma} = \text{localized site electron},
]

[
y_{ik\sigma} = \text{electron from atom } i \text{ committed to bond } i-j_k.
]

These are persistent DOFs.

For geometry, store atomic positions:

[
\mathbf R_i.
]

So the persistent state is:

[
\boxed{
\mathbf R_i,\quad y_{ia\sigma}
}
]

where

[
a=0,1,2,3.
]

Optionally also store velocities/momenta if using damped dynamics instead of pure gradient descent:

[
\mathbf v_i,\quad v_{ia\sigma}.
]

But the minimal state is just (\mathbf R_i) and (y_{ia\sigma}).

# 2. Derived auxiliary quantities

At each iteration we compute and discard:

[
\rho_{i\sigma}=\sum_{a=0}^{z_i}y_{ia\sigma},
]

[
\rho_i=\rho_{i\uparrow}+\rho_{i\downarrow},
]

[
Q_i=Z_i^\pi-\rho_i,
]

[
\phi_i=\sum_jJ_{ij}Q_j,
]

[
\mu_{i\sigma}
=============

\frac{\partial E_{\mathrm{local}}}{\partial \rho_{i\sigma}},
]

[
r_{ij}=|\mathbf R_j-\mathbf R_i|,
]

[
\varepsilon_{ij}=\varepsilon_{ij}(r_{ij}).
]

These are auxiliary buffers or temporary registers. They do not carry state except between kernels within one iteration.

So:

[
\boxed{
\text{persistent: } \mathbf R_i, y_{ia\sigma}
}
]

[
\boxed{
\text{auxiliary: } \rho_i,Q_i,\phi_i,\mu_i,\varepsilon_{ij},\text{ gradients}
}
]

# 3. Local electron budget

For each atom and spin,

[
\rho_{i\sigma}
==============

y_{i0\sigma}
+
\sum_{k=1}^{z_i}y_{ik\sigma}.
]

Single (p_z) orbital Pauli bound:

[
0\le \rho_{i\sigma}\le 1.
]

Total local (\pi)-population:

[
\rho_i=\rho_{i\uparrow}+\rho_{i\downarrow}.
]

Charge:

[
Q_i=Z_i^\pi-\rho_i.
]

For neutral carbon,

[
Z_i^\pi=1.
]

For pyridine-like nitrogen,

[
Z_i^\pi\approx 1
]

in the active (\pi)-system, with different (\chi_i,\eta_i).

For pyrrole-like (-\mathrm{NH}-),

[
Z_i^\pi\approx 2.
]

This local electron bookkeeping prevents the “one electron used in three bonds” problem.

# 4. Bond order from two half-bonds

For a bond (i-j), atom (i) contributes (y_{i\to j,\sigma}), and atom (j) contributes (y_{j\to i,\sigma}).

Define spin-resolved bond order:

[
B_{ij\sigma}
============

2\sqrt{
y_{i\to j,\sigma}
y_{j\to i,\sigma}
}.
]

Total bond order:

[
B_{ij}
======

\sum_\sigma B_{ij\sigma}.
]

This keeps the quantum-like (c_i c_j) structure because if

[
y_i \sim c_i^2,
]

then

[
\sqrt{y_i y_j}\sim c_i c_j.
]

The factor 2 is a convention. You can absorb it into (\varepsilon_{ij}). I keep it because a symmetric full bond with

[
y_{i\to j,\sigma}=y_{j\to i,\sigma}
]

gets a simple form.

# 5. Energy functional

Use

[
E
=

E_{\mathrm{bond}}
+
E_{\mathrm{charge}}
+
E_U
+
E_{\sigma}.
]

The covalent (\pi)-bond energy is

[
E_{\mathrm{bond}}
=================

-\sum_{\langle ij\rangle,\sigma}
2\varepsilon_{ij}(r_{ij})
\sqrt{
y_{i\to j,\sigma}
y_{j\to i,\sigma}
}.
]

The QEq-like local term:

[
E_{\mathrm{charge,local}}
=========================

\sum_i
\left[
\chi_i(\rho_i-Z_i^\pi)
+
\frac{\eta_i}{2}(\rho_i-Z_i^\pi)^2
\right].
]

The long-range Coulomb term:

[
E_{\mathrm{Coul}}
=================

\frac12
\sum_{ij}
Q_iJ_{ij}Q_j.
]

The Hubbard term:

[
E_U=
\sum_i
U_i\rho_{i\uparrow}\rho_{i\downarrow}.
]

The sigma skeleton:

[
E_\sigma
========

\sum_{\langle ij\rangle}
\frac{K_r}{2}(r_{ij}-r_{0,ij})^2
+
\sum_{\angle jik}
\frac{K_\theta}{2}(\theta_{jik}-\theta_0)^2
+
E_{\mathrm{oop}}.
]

The electronic bond strength depends on geometry:

[
\varepsilon_{ij}(r)
===================

\varepsilon_{0,ij}\exp[-\beta_{ij}(r-r_{0,ij})].
]

Then shorter bonds have larger (\varepsilon), so electrons prefer shorter bonds, and stronger electronic bond order pulls atoms together.

# 6. Variational derivative with respect to local electronic DOFs

For atom (i), define local chemical potential

[
\mu_{i\sigma}
=============

\frac{\partial}{\partial \rho_{i\sigma}}
\left[
E_{\mathrm{charge,local}}
+
E_{\mathrm{Coul}}
+
E_U
\right].
]

We get

[
\boxed{
\mu_{i\sigma}
=============

\chi_i
+
\eta_i(\rho_i-Z_i^\pi)
----------------------

\phi_i
+
U_i\rho_{i\bar\sigma}
}
]

where

[
\phi_i=\sum_jJ_{ij}Q_j.
]

The minus sign comes from

[
Q_i=Z_i^\pi-\rho_i.
]

Now the derivative of the localized variable (y_{i0\sigma}) is simply

[
\boxed{
\frac{\partial E}{\partial y_{i0\sigma}}
========================================

\mu_{i\sigma}.
}
]

For an outgoing half-bond variable (y_{i\to j,\sigma}),

[
E_{ij\sigma}
============

-2\varepsilon_{ij}
\sqrt{
y_{i\to j,\sigma}
y_{j\to i,\sigma}
}.
]

Therefore

[
\frac{\partial E_{ij\sigma}}
{\partial y_{i\to j,\sigma}}
============================

-\varepsilon_{ij}
\sqrt{
\frac{y_{j\to i,\sigma}}
{y_{i\to j,\sigma}+\epsilon}
}.
]

So the total derivative is

[
\boxed{
\frac{\partial E}{\partial y_{i\to j,\sigma}}
=============================================

## \mu_{i\sigma}

\varepsilon_{ij}
\sqrt{
\frac{y_{j\to i,\sigma}}
{y_{i\to j,\sigma}+\epsilon}
}.
}
]

This is a perfect gather rule: atom (i) only needs its own (\mu_i), its own half-bond variable, the neighbor’s opposite half-bond variable, and the distance (r_{ij}).

No scatter.

# 7. Local update rule for electronic variables

Gradient descent:

[
y_{ia\sigma}
\leftarrow
y_{ia\sigma}
------------

\Delta t_y
\left(
\frac{\partial E}{\partial y_{ia\sigma}}
-\lambda
\right).
]

Here (\lambda) is a global electron-number Lagrange multiplier. If total electron number must be exactly conserved,

[
\sum_{i,a,\sigma}y_{ia\sigma}=N_\pi,
]

then

[
\lambda=
\frac{1}{N_{\mathrm{dof}}}
\sum_{i,a,\sigma}
\frac{\partial E}{\partial y_{ia\sigma}}.
]

This requires a global reduction.

But there are alternatives.

# 8. Avoiding global reduction: local simplex update

If you want fully local GPU dynamics, make each atom conserve its own electron number approximately or exactly.

For each atom and spin, impose

[
\sum_a y_{ia\sigma}=\rho_{i\sigma}^{\mathrm{target}}.
]

For neutral non-spin-polarized carbon,

[
\rho_{i\uparrow}^{\mathrm{target}}
==================================

# \rho_{i\downarrow}^{\mathrm{target}}

\frac12.
]

Then update only redistribution among the local channels:

[
a=0,\dots,z_i.
]

Use a local Lagrange multiplier per atom and spin:

[
\lambda_{i\sigma}
=================

\frac{1}{z_i+1}
\sum_{a=0}^{z_i}
g_{ia\sigma},
]

where

[
g_{ia\sigma}=\frac{\partial E}{\partial y_{ia\sigma}}.
]

Then

[
\boxed{
y_{ia\sigma}
\leftarrow
y_{ia\sigma}
------------

\Delta t_y
\left(
g_{ia\sigma}-\lambda_{i\sigma}
\right).
}
]

This conserves

[
\sum_a y_{ia\sigma}
]

locally, per atom, per spin.

That means no global reduction.

This is probably the best first implementation for graphene/PAHs if you want speed and locality.

Then later, to allow charge transfer, introduce a slowly evolving (\rho_i) field with QEq/Coulomb, which does require either global neutrality correction or local charge-conserving diffusion.

# 9. Local Pauli projection

After updating (y_{ia\sigma}), enforce:

[
y_{ia\sigma}\ge0,
]

and

[
\sum_a y_{ia\sigma}\le1.
]

For fixed local electron number, project onto simplex:

[
\sum_a y_{ia\sigma}=\rho_{i\sigma}^{\mathrm{target}},
\qquad
y_{ia\sigma}\ge0.
]

Because (z_i\le3), each atom has at most four channels per spin:

[
y_{i0\sigma},y_{i1\sigma},y_{i2\sigma},y_{i3\sigma}.
]

Projection is tiny and can be done inside one atom-thread.

For a simple approximate projection:

1. clamp negatives to zero,
2. compute sum,
3. rescale.

[
y_{ia\sigma}\leftarrow
\max(y_{ia\sigma},0),
]

[
S_{i\sigma}=\sum_a y_{ia\sigma},
]

[
y_{ia\sigma}\leftarrow
y_{ia\sigma}
\frac{\rho_{i\sigma}^{\mathrm{target}}}{S_{i\sigma}+\epsilon}.
]

This preserves local population and positivity.

For four variables, exact simplex projection is also cheap, but clamp-rescale is probably enough for the first version.

# 10. Geometry force on atom (i)

The atom position (\mathbf R_i) is also a persistent DOF.

Its force has sigma and pi contributions:

[
\mathbf F_i
===========

# -\frac{\partial E}{\partial \mathbf R_i}

\mathbf F_i^\sigma+\mathbf F_i^\pi.
]

For the electronic bond term:

[
E_{ij\sigma}
============

-2\varepsilon(r_{ij})
\sqrt{
y_{i\to j,\sigma}
y_{j\to i,\sigma}
}.
]

Let

[
s_{ij\sigma}
============

\sqrt{
y_{i\to j,\sigma}
y_{j\to i,\sigma}
}.
]

Then

[
E_{ij}
======

-2\varepsilon(r_{ij})
\sum_\sigma s_{ij\sigma}.
]

Derivative:

[
\frac{dE_{ij}}{dr}
==================

-2\varepsilon'(r)
\sum_\sigma s_{ij\sigma}.
]

If

[
\varepsilon(r)=\varepsilon_0 e^{-\beta(r-r_0)},
]

then

[
\varepsilon'(r)=-\beta\varepsilon(r).
]

Therefore

[
\frac{dE_{ij}}{dr}
==================

2\beta\varepsilon(r)
\sum_\sigma s_{ij\sigma}.
]

Force on atom (i):

[
\boxed{
\mathbf F_i^{\pi,ij}
====================

-\frac{dE_{ij}}{dr}
\frac{\mathbf R_i-\mathbf R_j}{r}
}
]

Equivalently, using

[
\mathbf h_{ij}=\frac{\mathbf R_j-\mathbf R_i}{r},
]

[
\boxed{
\mathbf F_i^{\pi,ij}
====================

2\varepsilon'(r)
\left(
\sum_\sigma s_{ij\sigma}
\right)
\mathbf h_{ij}.
}
]

Since (\varepsilon'(r)<0), this pulls (i) toward (j).

In implementation, an atom thread computes:

[
\mathbf F_i^\pi
===============

\sum_{j\in N(i)}
\mathbf F_i^{\pi,ij}
]

by gathering neighbor positions and opposite half-bond variables.

No scatter.

---

# 11. Sigma skeleton force

For bond stretching,

[
E_{\sigma,r}
============

\sum_{\langle ij\rangle}
\frac{K_{ij}}{2}
(r_{ij}-r_{0,ij})^2.
]

Atom (i) gathers from neighbors:

[
\mathbf F_i^{\sigma,r}
======================

\sum_{j\in N(i)}
K_{ij}(r_{ij}-r_{0,ij})
\frac{\mathbf R_j-\mathbf R_i}{r_{ij}}.
]

For angle stiffness, each atom also loops over neighbor pairs:

[
(j,k)\in N(i)\times N(i).
]

For (sp^2),

[
\theta_0=120^\circ.
]

Angle energy:

[
E_{\theta}
==========

\sum_{j<k\in N(i)}
\frac{K_\theta}{2}
(\cos\theta_{jik}-\cos\theta_0)^2.
]

Using (\cos\theta) avoids expensive arccos.

[
c=
\frac{
(\mathbf R_j-\mathbf R_i)\cdot(\mathbf R_k-\mathbf R_i)
}{
r_{ij}r_{ik}
}.
]

Then

[
E_\theta=
\frac{K_\theta}{2}(c-c_0)^2.
]

This is still local. The force is more algebra, but still gather-local because atom (i) knows positions of (j,k).

For a first electronic test, I would freeze (\mathbf R_i), or use only bond stretching. Add angle forces later.

# 12. Atom-local kernel structure

Use fixed-size neighbor arrays:

```c
int neigh[N][3];     // neighbor atom index, -1 if absent
int rev[N][3];       // reverse slot: if j=neigh[i][k], then neigh[j][rev[i][k]]=i
float r0[N][3];
float eps0[N][3];
float beta[N][3];
float Kr[N][3];
```

Persistent electronic state:

```c
float y[N][2][4];
// y[i][s][0] = localized
// y[i][s][1+k] = half-bond from atom i to neigh[i][k]
```

Persistent geometry:

```c
float4 R[N];
```

Auxiliary:

```c
float rho[N][2];
float Q[N];
float phi[N];
float mu[N][2];
float4 F[N];
```

# 13. Kernel 1: compute (\rho_i) and (Q_i)

One thread per atom.

For atom (i):

[
\rho_{i\sigma}=\sum_a y_{ia\sigma}.
]

[
Q_i=Z_i^\pi-\rho_i.
]

Pseudocode:

```c
__kernel void eval_rho_Q(
    __global const float* y,
    __global const float* Zpi,
    __global float2* rho,
    __global float* Q
){
    int i = get_global_id(0);

    float ru = 0.0f;
    float rd = 0.0f;

    for(int a=0; a<4; a++){
        ru += Y(i,0,a);
        rd += Y(i,1,a);
    }

    rho[i] = (float2)(ru,rd);
    Q[i] = Zpi[i] - (ru+rd);
}
```

Purely local.

# 14. Kernel 2: Coulomb potential

Compute

[
\phi_i=\sum_jJ_{ij}Q_j.
]

This is the first nonlocal bottleneck.

Options:

### Direct Coulomb

One thread per atom (i), loop over all (j):

[
O(N^2)
]

but gather-only and simple. For PAHs up to maybe (10^4), this may still be okay depending on how many iterations.

### FFT

If charges live on a grid:

[
O(N_g\log N_g)
]

but requires charge spreading and interpolation.

### FMM / treecode

Approximate

[
O(N)
\quad\text{or}\quad
O(N\log N).
]

But more complex.

### Local/screened approximation

For early development:

[
J_{ij}=0
]

beyond a cutoff, or use only onsite hardness (\eta_i). Then the whole solver is local.

This is probably best for initial testing.

# 15. Kernel 3: compute (\mu_{i\sigma})

One thread per atom.

[
\mu_{i\sigma}
=============

\chi_i
+
\eta_i(\rho_i-Z_i^\pi)
----------------------

\phi_i
+
U_i\rho_{i\bar\sigma}.
]

Pseudocode:

```c
__kernel void eval_mu(
    __global const float2* rho,
    __global const float* phi,
    __global const float* Zpi,
    __global const float* chi,
    __global const float* eta,
    __global const float* U,
    __global float2* mu
){
    int i = get_global_id(0);

    float ru = rho[i].x;
    float rd = rho[i].y;
    float r  = ru + rd;

    float base = chi[i] + eta[i]*(r - Zpi[i]) - phi[i];

    mu[i].x = base + U[i]*rd;
    mu[i].y = base + U[i]*ru;
}
```

Purely local.

---

# 16. Kernel 4: electronic update, atom-local gather

One thread per atom.

For each spin and local channel (a):

### Localized channel

[
g_{i0\sigma}=\mu_{i\sigma}.
]

### Bond channel

For neighbor (j=neigh[i][k]):

[
g_{i,k+1,\sigma}
================

## \mu_{i\sigma}

\varepsilon_{ij}(r_{ij})
\sqrt{
\frac{
y_{j,rev(i,k)+1,\sigma}
}{
y_{i,k+1,\sigma}+\epsilon
}
}.
]

This is gather: read neighbor’s opposite half-bond.

Then use local electron-conserving update:

[
\lambda_{i\sigma}
=================

\frac{1}{z_i+1}\sum_{a=0}^{z_i}g_{ia\sigma},
]

[
y_{ia\sigma}
\leftarrow
y_{ia\sigma}
------------

dt
\left(
g_{ia\sigma}-\lambda_{i\sigma}
\right).
]

Then local projection:

[
y_{ia\sigma}\ge0,
\qquad
\sum_a y_{ia\sigma}=\rho_{i\sigma}^{target}.
]

Pseudocode:

```c
__kernel void update_y_local(
    __global float* y,
    __global const float4* R,
    __global const int* neigh,
    __global const int* rev,
    __global const float2* mu,
    __global const float* eps0,
    __global const float* beta,
    __global const float* r0,
    __global const float2* rhoTarget,
    float dt
){
    int i = get_global_id(0);

    float g[2][4];
    float yn[2][4];

    for(int s=0; s<2; s++){
        g[s][0] = (s==0 ? mu[i].x : mu[i].y);

        for(int k=0; k<3; k++){
            int j = NEIGH(i,k);

            if(j < 0){
                g[s][1+k] = 0.0f;
                continue;
            }

            int rk = REV(i,k);

            float4 d4 = R[j] - R[i];
            float r = sqrt(d4.x*d4.x + d4.y*d4.y + d4.z*d4.z);

            float eps = EPS0(i,k) * exp(-BETA(i,k)*(r - R0(i,k)));

            float yi = Y(i,s,1+k);
            float yj = Y(j,s,1+rk);

            float mui = (s==0 ? mu[i].x : mu[i].y);

            g[s][1+k] = mui - eps*sqrt((yj+1e-12f)/(yi+1e-12f));
        }

        // local lambda over active channels
        float gsum = g[s][0];
        int nchan = 1;

        for(int k=0; k<3; k++){
            if(NEIGH(i,k) >= 0){
                gsum += g[s][1+k];
                nchan++;
            }
        }

        float lam = gsum / (float)nchan;

        // gradient step
        yn[s][0] = Y(i,s,0) - dt*(g[s][0] - lam);

        for(int k=0; k<3; k++){
            if(NEIGH(i,k) >= 0){
                yn[s][1+k] = Y(i,s,1+k) - dt*(g[s][1+k] - lam);
            }else{
                yn[s][1+k] = 0.0f;
            }
        }

        // positivity + simplex projection
        float sum = 0.0f;

        for(int a=0; a<4; a++){
            yn[s][a] = fmax(yn[s][a], 0.0f);
            sum += yn[s][a];
        }

        float target = (s==0 ? rhoTarget[i].x : rhoTarget[i].y);
        float scale = target / (sum + 1e-12f);

        for(int a=0; a<4; a++){
            Y(i,s,a) = yn[s][a] * scale;
        }
    }
}
```

This kernel has no global reductions. It locally redistributes each atom’s spin electron budget among lone density and bonds.

This is the most GPU-friendly version.

# 17. Kernel 5: geometry force, gather style

One thread per atom.

Compute

[
\mathbf F_i =
\sum_{j\in N(i)}
\left(
\mathbf F_{ij}^{\sigma}
+
\mathbf F_{ij}^{\pi}
\right)
+
\text{angle forces}.
]

For the (\pi)-bond force:

[
s_{ij}
======

\sum_\sigma
\sqrt{
y_{i\to j,\sigma}
y_{j\to i,\sigma}
}.
]

[
E_{ij}=-2\varepsilon(r)s_{ij}.
]

[
\frac{dE_{ij}}{dr}
==================

-2\varepsilon'(r)s_{ij}.
]

With exponential (\varepsilon),

[
\varepsilon'(r)=-\beta\varepsilon.
]

So

[
\frac{dE}{dr}
=============

2\beta\varepsilon s_{ij}.
]

Force on (i):

[
\mathbf F_i^{\pi,ij}
====================

\frac{dE}{dr}
\frac{\mathbf R_j-\mathbf R_i}{r}.
]

This pulls (i) toward (j).

Sigma stretch:

[
E_{\sigma,ij}
=============

\frac{K_r}{2}(r-r_0)^2.
]

[
\mathbf F_i^{\sigma,ij}
=======================

K_r(r-r_0)
\frac{\mathbf R_j-\mathbf R_i}{r}.
]

But note sign: if (r>r_0), this pulls together, so the above sign is correct.

Combined pair force gathered by atom (i):

[
\boxed{
\mathbf F_i^{ij}
================

\left[
K_r(r-r_0)
----------

2\varepsilon'(r)s_{ij}
\right]
\frac{\mathbf R_j-\mathbf R_i}{r}.
}
]

Since (\varepsilon'<0),

[
-2\varepsilon'(r)s_{ij}>0,
]

so the electronic term also pulls atoms together.

Pseudocode:

```c
__kernel void update_R_gather(
    __global float4* R,
    __global const float* y,
    __global const int* neigh,
    __global const int* rev,
    __global const float* eps0,
    __global const float* beta,
    __global const float* r0,
    __global const float* Kr,
    float dtR,
    float damp
){
    int i = get_global_id(0);

    float3 Fi = (float3)(0.0f,0.0f,0.0f);
    float4 Ri4 = R[i];
    float3 Ri = (float3)(Ri4.x,Ri4.y,Ri4.z);

    for(int k=0; k<3; k++){
        int j = NEIGH(i,k);
        if(j < 0) continue;

        int rk = REV(i,k);

        float4 Rj4 = R[j];
        float3 Rj = (float3)(Rj4.x,Rj4.y,Rj4.z);

        float3 d = Rj - Ri;
        float r = sqrt(dot(d,d) + 1e-20f);
        float3 h = d / r;

        float eps = EPS0(i,k) * exp(-BETA(i,k)*(r - R0(i,k)));
        float deps = -BETA(i,k)*eps;

        float sij = 0.0f;
        for(int s=0; s<2; s++){
            float yi = Y(i,s,1+k);
            float yj = Y(j,s,1+rk);
            sij += sqrt(fmax(yi*yj,0.0f));
        }

        float fpair = KR(i,k)*(r - R0(i,k)) - 2.0f*deps*sij;

        Fi += fpair*h;
    }

    // optionally add angle force here by looping over neighbor pairs

    Ri += dtR*Fi;

    R[i] = (float4)(Ri.x,Ri.y,Ri.z,0.0f);
}
```

In practice use FIRE/velocity damping rather than raw position gradient descent.

# 18. Important subtlety: double-counting pair forces

If each atom gathers its own force from neighbors, the pair (ij) is computed twice: once in atom (i)’s thread and once in atom (j)’s thread.

That is okay if both compute only their own force.

There is no energy accumulation here. No scatter. The pair force is symmetric automatically if both read the same state.

If you also accumulate total energy, then you must avoid double counting by using only (i<j), or multiply pair energy by (1/2) in atom-gather energy kernels.

# 19. Global operations / bottlenecks

The clean local framework is broken only by a few operations.

## 19.1 Coulomb potential

[
\phi_i=\sum_jJ_{ij}Q_j
]

is nonlocal.

This requires one of:

[
O(N^2)\text{ direct gather},
]

FFT,

FMM/treecode,

or cutoff screening.

For first implementation, I would use either no Coulomb or direct (O(N^2)) for small systems. Because the local electronic model is already interesting without Coulomb.

## 19.2 Global electron-number conservation

If you allow (\rho_i) to change freely, you need:

[
\sum_i\rho_i=N_\pi.
]

That requires global reduction for (\lambda), or periodic renormalization.

Avoid this initially by using local fixed targets:

[
\sum_a y_{ia\sigma}=\rho_{i\sigma}^{target}.
]

Then no global reduction.

Later, introduce charge transfer with an additional local variable (\rho_i) and global neutralization.

## 19.3 Convergence criterion

Computing

[
\max_i |F_i|
]

or total energy requires reduction.

For GPU production, you do not need to do this every iteration. Check every 20–100 steps.

## 19.4 Adaptive timestep / FIRE

FIRE requires dot products such as

[
\mathbf F\cdot\mathbf v
]

and norms. These are reductions.

You can avoid them initially with fixed damped gradient descent. Or do block-local FIRE approximately.

## 19.5 Exact simplex projection with sorting

Exact projection of four variables is local and does not need global sync. Sorting four values is cheap.

So this is not a global bottleneck.

# 20. Two solver modes

I would separate two modes.

## Mode A: strictly local neutral-valence solver

Persistent variables:

[
\mathbf R_i,
\quad
y_{ia\sigma}.
]

Constraints:

[
\sum_a y_{ia\sigma}=\rho_{i\sigma}^{target}.
]

For neutral closed-shell carbon:

[
\rho_{i\uparrow}^{target}=
\rho_{i\downarrow}^{target}
===========================

\frac12.
]

For spin-polarized open shell, allow target spin populations to differ or update them slowly.

No global electron reduction. No Coulomb required.

This mode captures:

[
\text{bond order alternation},
]

[
\text{Peierls distortion},
]

[
\text{heteroatom local preferences through }\varepsilon,\chi,\rho^{target},
]

[
\text{edge effects},
]

but not long-range charge transfer.

This is the best first OpenCL implementation.

## Mode B: QEq charge-transfer solver

Persistent variables:

[
\mathbf R_i,
\quad
y_{ia\sigma},
\quad
\rho_{i\sigma}^{target}
]

or directly unconstrained (y).

Allow

[
\rho_i
]

to vary.

Need:

[
\sum_i\rho_i=N_\pi.
]

Need Coulomb (\phi_i).

This captures heteroatom charge redistribution, protonation, charged states, electrostatics, substrate screening.

But it introduces global operations.

For GPU, do this as outer loop:

1. several local bond-order relaxation steps at fixed (\rho_i),
2. one QEq/Coulomb update of (\rho_i),
3. repeat.

This separates fast local covalency from slower long-range charge equilibration.

# 21. Atom-local equations in final compact form

For each atom (i), spin (\sigma), channel (a):

[
a=0 \quad \text{localized},
]

[
a=k>0 \quad \text{bond to neighbor } j_k.
]

Persistent DOF:

[
y_{ia\sigma}.
]

Local population:

[
\rho_{i\sigma}=\sum_a y_{ia\sigma}.
]

Chemical potential:

[
\mu_{i\sigma}
=============

\chi_i+\eta_i(\rho_i-Z_i^\pi)-\phi_i
+
U_i\rho_{i\bar\sigma}.
]

Gradient:

[
g_{i0\sigma}=\mu_{i\sigma},
]

[
\boxed{
g_{ik\sigma}
============

## \mu_{i\sigma}

\varepsilon_{ij}(r_{ij})
\sqrt{
\frac{
y_{j,\mathrm{rev}(i,k),\sigma}
}{
y_{ik\sigma}+\epsilon
}
}
}
]

where (k>0).

Local constrained update:

[
\lambda_{i\sigma}
=================

\frac{1}{z_i+1}
\sum_{a=0}^{z_i}
g_{ia\sigma},
]

[
y_{ia\sigma}
\leftarrow
y_{ia\sigma}
------------

dt
\left(
g_{ia\sigma}-\lambda_{i\sigma}
\right).
]

Projection:

[
y_{ia\sigma}\ge0,
\qquad
\sum_a y_{ia\sigma}=\rho_{i\sigma}^{target}.
]

Geometry force:

[
\boxed{
\mathbf F_i
===========

\sum_{j\in N(i)}
\left[
K_{ij}(r_{ij}-r_{0,ij})
-----------------------

2\varepsilon'*{ij}(r*{ij})
\sum_\sigma
\sqrt{
y_{i\to j,\sigma}
y_{j\to i,\sigma}
}
\right]
\frac{\mathbf R_j-\mathbf R_i}{r_{ij}}
+
\mathbf F_i^{angle}
+
\mathbf F_i^{oop}.
}
]

This is almost exactly the structure of a classical force field.

The only difference is that every atom also has internal electronic allocation variables (y_{ia\sigma}), and these are relaxed by local gradient descent.

# 22. Recommended kernel schedule

For strictly local mode:

```text
repeat:
    kernel_update_y_local          // gather neighbor y and R, update y
    kernel_update_R_gather         // gather neighbor y and R, update positions
    occasionally:
        kernel_eval_energy         // optional reduction
```

For QEq/Coulomb mode:

```text
repeat:
    kernel_eval_rho_Q              // local
    kernel_eval_phi                // nonlocal Coulomb bottleneck
    kernel_eval_mu                 // local
    kernel_update_y_global_or_local// local except optional lambda
    kernel_update_R_gather         // local
    occasionally:
        reduction for energy / max force
```

For best GPU performance, do several electronic substeps per geometry step:

```text
repeat geometry iteration:
    for m = 1..M:
        update_y_local
    update_R_gather
```

because electronic variables should relax faster than atoms.

# 23. Practical advice for first prototype

I would start with the most local, robust model:

[
\rho_{i\uparrow}^{target}
=========================

# \rho_{i\downarrow}^{target}

\frac{Z_i^\pi}{2}
]

for closed shell.

Freeze charges:

[
\phi_i=0,
\qquad
\chi_i=0,
\qquad
\eta_i=0.
]

Use only:

[
E=
E_\sigma[\mathbf R]
-------------------

\sum_{\langle ij\rangle,\sigma}
2\varepsilon(r_{ij})
\sqrt{
y_{i\to j,\sigma}y_{j\to i,\sigma}
}.
]

with local simplex constraints.

This should already generate Peierls/Kekulé patterns if electron-lattice coupling is strong enough.

Then add in order:

1. different (\varepsilon_{ij}) and (Z_i^\pi) for heteroatoms,
2. geometry relaxation,
3. Hubbard spin,
4. QEq onsite hardness,
5. long-range Coulomb,
6. global charge transfer.

The main design win is that the core relaxation rule is fully local:

[
\boxed{
\text{atom reads own state + neighbor states, computes own gradients, writes only own state.}
}
]

That maps naturally to OpenCL and avoids atomics.

---

# USER

plese read @KekuleQM.md and implement the optimized OpenCL kernels for this in 

KekuleQM.cl file, put physical explanation and justification direcly in the code what we are doing and why, describe principles, math and physics arther then technicalieits (variable types etc.) also dscribe perfroamcne consideration (loval memory use, workgroup size, synchronization points, bottlenecks, brache divergence etc.)

Note - I would frame it similarly to classicar forcefield where we have local "bonding" terms and global "non-bonding terms", the electrostatics is non-local O(n^2) (if we do not use fastmultipole methods or FFT) so I would keep it separate.


So my roguth idea is to have 3 kernels per ietaration

1) gather local bonding forces on each atom
2) gather nonlocal electrosatics forces on each atom non-local electrosatics
3) update DOFs by force

but maybe you realize that it is very advantagenous to have global memory buffer for bond charge densities as intermediate, and I'm fine with that. 

I think this is key architectural consideration, think about this fist, and then forumate routh plan (pseudocode) and only then write the kernels

---

# GPT 5.5 devin

Created KekuleQM.cl with the requested 3-kernel architecture:

KekuleQM_gatherLocalBonding: KekuleQM.cl:106-181
Atom-local gather kernel.
Computes rho, Q, local chemical potential, half-bond gradients, sigma stretch force, and π Peierls force.
Writes optional directed bondData buffer for diagnostics/intermediate bond densities.
KekuleQM_gatherCoulombDirect: KekuleQM.cl:183-232
Separate nonlocal electrostatics kernel.
Direct tiled O(N^2) gather using local memory.
Keeps Coulomb isolated so it can later be replaced by FFT/FMM/treecode/cutoff.
KekuleQM_updateDOFs: KekuleQM.cl:234-302
Updates electronic DOFs y[i,spin,channel].
Supports local simplex projection for fixed local valence/spin population.
Updates positions from local + optional Coulomb forces.
Design choice
I followed the half-bond allocation model from KekuleQM.md:2190-2272 and the atom-local OpenCL plan from KekuleQM.md:3522-4359.

The persistent state is:

Geometry: R[i]
Electronic allocations: y[i,spin,channel]
The auxiliary per-iteration buffers are:

Local: rhoQ, gyLocal, FLocal, bondData
Electrostatic: phi, FCoul
Validation
OpenCL syntax compiler unavailable: clang is not installed.
Fallback static check passed:
File exists.
303 lines.
3 kernels found.
Delimiters balanced.
Notes
I also saved a project memory for the implemented KekuleQM.cl kernel architecture so this context is available in future work.

Task complete.

---

# USER

---

# GLM 5.2

Now I have a clear picture of the codebase. No `AtomicSystem` class or `.xyz` files exist yet, and the `OpenCLBase` module is missing. I'll create a self-contained harness.

The Coulomb kernel's workgroup size (128) exceeds the number of atoms (31). I need to round up the global size to a multiple of the workgroup size.

Harness runs successfully on the RTX 3090. The build warnings are harmless (just NVIDIA's inlining notices).
Created `@/home/prokop/git/NumericalMathPlayground/topics/LinearScalingQM/KekuleQM/KekuleQM_ocl.py` — a complete pyOpenCL harness for the KekuleQM kernels.

## What it does

1. **Molecule loading**: [read_xyz()](cci:1://file:///home/prokop/git/NumericalMathPlayground/topics/LinearScalingQM/KekuleQM/KekuleQM_ocl.py:30:0-41:66) parses standard `.xyz` files; [generate_graphene_flake()](cci:1://file:///home/prokop/git/NumericalMathPlayground/topics/LinearScalingQM/KekuleQM/KekuleQM_ocl.py:44:0-60:22) builds a honeycomb flake for testing
2. **Neighbor lists**: [build_neighbor_lists()](cci:1://file:///home/prokop/git/NumericalMathPlayground/topics/LinearScalingQM/KekuleQM/KekuleQM_ocl.py:67:0-105:21) creates the fixed-size `neigh[i][k]` and `rev[i][k]` arrays with reverse-slot mapping that the kernels need
3. **Parameter setup**: [default_params()](cci:1://file:///home/prokop/git/NumericalMathPlayground/topics/LinearScalingQM/KekuleQM/KekuleQM_ocl.py:129:0-157:41) builds per-slot `eps0`, `beta`, `r0`, `Kr` and per-atom `(chi, eta, U, Zpi)` arrays, with element-specific defaults for C/N/O
4. **GPU solver**: [KekuleQM_OCL](cci:2://file:///home/prokop/git/NumericalMathPlayground/topics/LinearScalingQM/KekuleQM/KekuleQM_ocl.py:164:0-388:22) class handles buffer allocation, kernel execution (3-kernel iteration), and state download
5. **Plotting**: [plot_molecule()](cci:1://file:///home/prokop/git/NumericalMathPlayground/topics/LinearScalingQM/KekuleQM/KekuleQM_ocl.py:395:0-454:13) draws bonds with width ∝ bond order and color encoding (red=double, green=aromatic, blue=single), with bond length labels; [plot_convergence()](cci:1://file:///home/prokop/git/NumericalMathPlayground/topics/LinearScalingQM/KekuleQM/KekuleQM_ocl.py:457:0-488:50) shows bond order evolution over iterations

## Test result

```
System: 31 atoms, 36 bonds
Final bond orders: min=0.353 max=1.547 mean=0.833
Final bond lengths: min=1.391 max=1.456 mean=1.420
```

The solver successfully relaxes the π-electron allocations on GPU and produces physically reasonable bond orders for a graphene flake.

## Usage

```python
from KekuleQM_ocl import *

# Load from .xyz
enames, positions = read_xyz("molecule.xyz")

# Or generate a flake
enames, positions = generate_graphene_flake(radius=5.0)

# Build neighbors and parameters
neigh, rev = build_neighbor_lists(positions, cutoff=1.8)
eps0, beta, r0, Kr, atomParams = default_params(len(positions), neigh, enames)

# Solve
solver = KekuleQM_OCL()
solver.setup(positions, neigh, rev, eps0, beta, r0, Kr, atomParams)
solver.relax(nsteps=500, dtY=0.05, verbose=True)

# Get results
bonds, B = solver.compute_bond_orders(neigh, rev)
lengths = solver.compute_bond_lengths(bonds)
R = solver.download_R()

# Plot
plot_molecule(R, bonds, B, bond_lengths=lengths, enames=enames)
plt.show()
```