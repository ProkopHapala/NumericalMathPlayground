# Kekulé Fluid: Topological Defect Dynamics in Graphene via Schrödinger-Smoke Analogy

## A Didactical Monograph

---

## Table of Contents

1. [Motivation and Introduction](#1-motivation-and-introduction)
2. [Physical Background](#2-physical-background)
3. [The Schrödinger Smoke Analogy](#3-the-schrödinger-smoke-analogy)
4. [The Complex Kekulé Order Parameter](#4-the-complex-kekulé-order-parameter)
5. [Defects as Topological Objects](#5-defects-as-topological-objects)
6. [Nodal Lines and Vortices in 2D](#6-nodal-lines-and-vortices-in-2d)
7. [The Ginzburg–Landau Free Energy](#7-the-ginzburglandau-free-energy)
8. [The Pauli–Idempotency Analogy](#8-the-pauliidempotency-analogy)
9. [Model A: Kekulé Fluid Solver — Complete Specification](#9-model-a-kekulé-fluid-solver--complete-specification)
10. [Model B: Dirac / Tight-Binding Solver — Complete Specification](#10-model-b-dirac--tight-binding-solver--complete-specification)
11. [Comparison of the Two Models](#11-comparison-of-the-two-models)
12. [Aromaticity and Antiaromaticity](#12-aromaticity-and-antiaromaticity)
13. [Speculative Physical Interpretations](#13-speculative-physical-interpretations)
14. [Code Review and Physics Audit](#14-code-review-and-physics-audit)
15. [Suggested Roadmap](#15-suggested-roadmap)
16. [References](#16-references)

---

## 1. Motivation and Introduction

### 1.1 The Core Idea

The **Schrödinger smoke** method [1] represents an incompressible fluid by a $\mathbb{C}^2$-valued wavefunction evolving under a Schrödinger-type equation. Vortex filaments — the topological singularities of the flow — emerge naturally as zero-density regions where both real components of the wavefunction vanish simultaneously. In 2D, these become point vortices: clockwise and anticlockwise, like cutting a 3D vortex filament in half.

This monograph explores a physically precise analogy: **can the π-electron system of graphene be described by a similar compact complex field whose topological defects correspond to chemical irregularities** (radicals, protonated edges, vacancies, charge injections)?

The answer is yes, with an important correction: the vortex is **not** a vortex of physical electron density. It is a vortex of the **Kekulé order parameter** — the complex bond-order field that encodes which bonds are double-like and which are single-like. This shifts the analogy from "hydrodynamics of electron density" to "hydrodynamics of dimer/bond-order topology," which is both more physically correct and more chemically transparent.

### 1.2 Why This Matters

Traditional π-electron calculations require diagonalizing the Hamiltonian — an $O(N^3)$ operation that becomes prohibitive for large PAHs (polycyclic aromatic hydrocarbons) or graphene flakes. The Kekulé fluid approach instead evolves a **local bond-order field** with $O(N)$ cost per timestep, using GPU-accelerated local updates. The topological defects (vortices, domain walls, nodal crossings) emerge from the dynamics without any spectral computation.

This enables:
- Fast exploration of defect configurations across many PAH geometries
- Real-time visualization of aromatic bond-order rearrangement
- Qualitative prediction of where Dirac quasiparticles will localize
- A two-layer architecture: fast topological screening + occasional electronic validation

### 1.3 The Analogy Chain

$$
\text{Schrödinger smoke vortex filament (3D)}
\;\Downarrow\;
\text{2D phase vortex / point defect}
\;\Downarrow\;
\text{Kekulé mass vortex in graphene}
\;\Downarrow\;
\text{zero mode / radical / vacancy / topological charge defect}
$$

At each level, the topology is carried by a complex field whose phase winds around zeros of its amplitude. The physics changes, but the mathematical structure of topological defects is preserved.

---

## 2. Physical Background

### 2.1 Graphene and the π System

Graphene is a 2D honeycomb lattice of carbon atoms. Each carbon contributes one $p_z$ orbital to the π system. The low-energy electronic structure is described by a tight-binding Hamiltonian:

$$
H = -\sum_{\langle ij \rangle} t_{ij}\, c_i^\dagger c_j + \sum_i V_i\, c_i^\dagger c_i
$$

where $t_{ij}$ is the nearest-neighbor hopping integral and $V_i$ is an onsite potential. For uniform graphene with $t_{ij} = t_0$, the band structure has Dirac cones at the $K$ and $K'$ points of the Brillouin zone, giving massless Dirac quasiparticles with Fermi velocity $v_F \approx 10^6\,\text{m/s}$.

### 2.2 Kekulé Bond Order and Peierls Distortion

In a Kekulé-distorted state, the three bonds around each carbon become inequivalent: one is shorter (double-bond-like) and two are longer (single-bond-like). This is the Peierls instability of the honeycomb lattice. The bond order on bond $j$ is:

$$
B_j = \langle c_A^\dagger(\mathbf{r})\, c_B(\mathbf{r} + \boldsymbol{\delta}_j) \rangle
$$

where $\boldsymbol{\delta}_j$ ($j = 1,2,3$) are the three nearest-neighbor vectors from an A-sublattice atom to its B-sublattice neighbors:

$$
\boldsymbol{\delta}_1 = a(0,\, 1), \quad
\boldsymbol{\delta}_2 = a\!\left(\tfrac{\sqrt{3}}{2},\, -\tfrac{1}{2}\right), \quad
\boldsymbol{\delta}_3 = a\!\left(-\tfrac{\sqrt{3}}{2},\, -\tfrac{1}{2}\right)
$$

with $a$ the C–C bond length. The bond order $B_j$ measures the quantum-mechanical hopping coherence — how much the π electrons prefer that particular bond.

### 2.3 The Four-Component Dirac Spinor

At low energy, the π electrons near the Dirac points are described by a four-component spinor:

$$
\Psi = \begin{pmatrix} \psi_{A,K} \\ \psi_{B,K} \\ \psi_{A,K'} \\ \psi_{B,K'} \end{pmatrix}
$$

where $A/B$ labels the sublattice and $K/K'$ labels the valley. The continuum Hamiltonian is:

$$
H = v_F\!\left(\tau_z \sigma_x p_x + \sigma_y p_y\right) + \text{Re}\,\Delta_K(\mathbf{r})\, M_1 + \text{Im}\,\Delta_K(\mathbf{r})\, M_2 + V(\mathbf{r})\, I_4
$$

Here $\sigma$ acts on sublattice, $\tau$ on valley, and $M_1, M_2$ are anticommuting Kekulé mass matrices (explicit forms given in §10). The complex field $\Delta_K(\mathbf{r}) = |\Delta_K| e^{i\phi}$ is the **Kekulé mass**: its real and imaginary parts gap the Dirac spectrum in different ways, and its phase $\phi$ selects which of the three Kekulé bond patterns is realized.

### 2.4 The $\mathbb{C}^2 \cong \mathbb{R}^4 \cong \mathbb{H}$ Correspondence

A two-component complex wavefunction $\Psi = (z_1, z_2)^T \in \mathbb{C}^2$ is equivalent as a real vector space to $\mathbb{R}^4$ and also to a single quaternion $q = z_1 + j\, z_2$. However, the natural gauge operation in $\mathbb{C}^2$ is multiplication by a common phase $e^{i\theta}$, which is only a subgroup of full unit-quaternion rotations. The quaternion algebra is richer and noncommutative, containing extra "spin texture" degrees of freedom.

In 3D, $\mathbb{C}^2$ provides enough degrees of freedom to encode complex vorticity robustly (this is why Schrödinger smoke uses it). In 2D, a **single** complex scalar $\psi = u + iv$ is already sufficient: the two equations $u(x,y) = 0$ and $v(x,y) = 0$ define two curves whose intersections are isolated points — the vortex cores.

---

## 3. The Schrödinger Smoke Analogy

### 3.1 Schrödinger Smoke

In the Schrödinger smoke framework [1], an incompressible fluid is represented by a $\mathbb{C}^2$-valued wavefunction evolving under a Schrödinger equation with Hamiltonian/Landau–Lifshitz structure. The Madelung transformation gives:

$$
\psi(\mathbf{r}, t) = \sqrt{\rho(\mathbf{r}, t)}\, e^{i\theta(\mathbf{r}, t)}
$$

where $\rho$ is the fluid density and $\nabla\theta$ gives the fluid velocity. Vortex filaments occur where $\rho \to 0$ and the phase $\theta$ winds by $2\pi n$.

### 3.2 The Kekulé Analogue

For graphene, the analogous object is not a two-component smoke spinor but the **complex Kekulé order parameter**:

$$
\Delta_K(\mathbf{r}) = |\Delta_K(\mathbf{r})|\, e^{i\phi(\mathbf{r})}
$$

A Kekulé vortex is:

$$
\Delta_K(\mathbf{r}) = \Delta_0\, f(r)\, e^{in\theta}
$$

with winding number $n = \frac{1}{2\pi} \oint \nabla\phi \cdot d\mathbf{l}$.

This is extremely close to the vortex logic of Schrödinger smoke — except the vortex is in the **mass/bond texture**, not in a superfluid density. The key distinction:

$$
|\Psi|^2 \neq \text{total π-electron density}
$$

but

$$
|\Delta_K|^2 \sim \text{strength of local Kekulé/dimer order}
$$

The actual electron density is a filled-Fermi-sea quantity: $\rho(\mathbf{r}) = \sum_{n \in \text{occ}} |\psi_n(\mathbf{r})|^2 - \rho_0(\mathbf{r})$, or more generally a density matrix $\rho_{ij} = \langle c_i^\dagger c_j \rangle$. Propagating a single "smoke-like" wavefunction models a quasiparticle or collective amplitude, not the full electronic ground state.

### 3.3 Blaschke Products as Vortex Initializers

A finite Blaschke product [2] provides a compact way to create a complex phase field with prescribed zeros:

$$
B(z) = \prod_k \frac{a_k - z}{1 - \bar{a}_k\, z}
$$

Each zero $a_k$ contributes a phase winding. For a 2D fluid shader, this initializes many vortices at once. For the Kekulé model, the analogous initializer is:

$$
\Delta_K(z) = \Delta_0 \prod_k \left(\frac{a_k - z}{1 - \bar{a}_k\, z}\right)^{s_k}
$$

where $s_k = +1$ is a vortex and $s_k = -1$ is an antivortex. On a discrete honeycomb graph, this is replaced by direct phase assignment at atom positions (see §9.2).

### 3.4 Connection to "Simple and Fast Fluids"

The "Simple and Fast Fluids" method [6] by Guay, Colin, and Egli represents a 2D fluid by a 4-component grid state $(v_x, v_y, \rho, \omega)$ where $\omega$ is vorticity. The solver performs: advection → pressure projection (incompressibility) → viscosity → optional vorticity confinement. In the Chimera's Breath shader implementation, vorticity confinement injects energy back into curls to counter numerical dissipation.

The Kekulé fluid solver follows the same architectural pattern:
- **Advection** → complex order parameter propagation on the graph
- **Pressure projection** → valence/dimer constraint projection (the "Pauli-like" step)
- **Vorticity confinement** → Z₃ anisotropy locking (keeps defects sharp)

---

## 4. The Complex Kekulé Order Parameter

### 4.1 Definition from Bond Orders

For each atom $i$, define the local complex Kekulé order as the threefold Fourier component of the three bond orders around it:

$$
\Delta_i = \sum_{b \in N(i)} (x_b - \bar{x}_i)\, \omega^{d_b}, \qquad \omega = e^{2\pi i/3}
$$

where:
- $b$ runs over the (up to three) bonds touching atom $i$
- $d_b \in \{0, 1, 2\}$ is the honeycomb bond direction
- $\bar{x}_i = \frac{1}{\deg_i} \sum_{b \in N(i)} x_b$ is the local average bond order
- $x_b \in [0, 1]$ is the normalized dimer/bond occupation

This is not an abstract field — it is literally the threefold Fourier component of actual bond orders around an atom.

### 4.2 Physical Interpretation

- $A_i = |\Delta_i|$ — **local Kekulé/dimerization strength**: how strongly the bonds alternate
- $\phi_i = \arg \Delta_i$ — **local orientation of bond alternation**: which of the three Kekulé patterns
- $A_i \to 0$ — the bond alternation vanishes: a defect core, domain wall, or frustrated region
- $\phi_i$ winding by $\pm 2\pi$ around a loop — a Kekulé vortex or antivortex

### 4.3 Inverse Relation: From Phase to Bonds

Given $\Delta_i = A_i\, e^{i\phi_i}$, the bond orders are reconstructed as:

$$
B_j(\mathbf{r}) = B_0 + A(\mathbf{r}) \cos\!\left(\phi(\mathbf{r}) + \frac{2\pi j}{3}\right)
$$

So:
- $A = |\Delta_K|$ says how strong the bond alternation is
- $\phi = \arg \Delta_K$ says which Kekulé pattern
- A vortex is a point where $A \to 0$ and $\phi$ winds by $2\pi$

### 4.4 The Pipeline

$$
x_{ij} \;\longrightarrow\; \Delta_i \;\longrightarrow\; A_i, \phi_i \;\longrightarrow\; \text{vortices / domain walls / aromatic frustration}
$$

No hexagon variable is required. The primary degrees of freedom are atoms and bonds. Rings (hexagons) are used only as **diagnostic loops** for measuring winding number, aromatic frustration, and circulation — they do not carry primary dynamical variables.

---

## 5. Defects as Topological Objects

### 5.1 The Dimer/Vacancy Picture

In an RVB/Kekulé/dimer picture, a perfect Kekulé structure is a close-packed dimer covering. A radical, vacancy, injected carrier, or hydrogenated carbon is a **monomer defect**: one site cannot participate normally in the dimer covering.

This defect forces frustration into the surrounding bond pattern. In 2D dimer/height-field language, monomers are topological defects or vortex-like charges of the coarse-grained height field [3].

$$
\text{dimer covering} + \text{unpaired site} \;\Longrightarrow\; \text{topological defect}
$$

### 5.2 Chemical Defect Types

| Chemical defect | Model representation | Physical effect |
|---|---|---|
| Radical / unpaired electron | `defect = 1`, `targetVal = 0` | π site removed, $|\Delta| \to 0$ |
| Protonated edge (=NH⁺−) | `defect = 0.5–1.0`, `pinStrength = 1.0` | Partial π removal + phase pinning |
| Vacancy / sp³ carbon | `defect = 1`, remove from graph | Site entirely gone |
| Hole / electron injection | `q_i` charge variable | Local charge perturbation |
| Edge functional group | `pinPhase = θ_dir[d]` | Boundary condition on Kekulé phase |

### 5.3 The Hou–Chamon–Mudry Result

Hou, Chamon, and Mudry [4] showed that graphenelike Dirac systems with a vortex in the Kekulé mass can bind **zero modes** and **fractionalized charge**. The defect is explicitly a twist in the phase of the Dirac mass. Later work [5] connects the fractional charge to bulk topology, in close analogy with SSH domain-wall charge in 1D.

A 2024 Nature Communications experiment [7] reported Kekulé vortices around hydrogen adatoms in graphene. The H adatom effectively removes a $p_z$ site from the π system, and the observed Kekulé bond texture winds by $2\pi$ around it. In spinless descriptions, the vortex carries $Q = -e/2$; in spinful descriptions, it gives neutral spin-$1/2$ at half filling and charged spinless states upon doping.

### 5.4 Vortex–Antivortex Conservation

In a closed finite system or periodic simulation cell, total winding must cancel:

$$
\sum_a n_a = 0
$$

unless the boundary carries compensating winding. This is analogous to the statement that killing a dimer at one site must be compensated somewhere else. In a finite PAH, compensation may appear at another radical, an edge, a sublattice imbalance, a charged reservoir, or a domain wall ending at the boundary.

### 5.5 The Z₃ Complication

The honeycomb lattice has three equivalent Kekulé patterns. The Kekulé phase is therefore not a perfect continuous $U(1)$ angle — at lattice scale it is closer to a **$\mathbb{Z}_3$** order parameter. A vortex is often a junction of three Kekulé domains/domain walls meeting at a core, rather than a perfectly smooth XY vortex. The continuum Dirac model treats this as approximately $U(1)$, which is elegant, but the atomistic model remembers the $\mathbb{Z}_3$ locking.

---

## 6. Nodal Lines and Vortices in 2D

### 6.1 Two Nodal Networks

A complex scalar field $\Delta(x,y) = u(x,y) + i\, v(x,y)$ has two real components. Each defines a nodal line:

- $u(x,y) = 0$ — the $\text{Re}\,\Delta = 0$ network
- $v(x,y) = 0$ — the $\text{Im}\,\Delta = 0$ network

Their **intersections** are the vortex cores: points where both components vanish and the phase winds.

### 6.2 Local Structure of a Vortex

A $+1$ vortex looks locally like:

$$
\Delta(x,y) = x + iy
$$

Then $\text{Re}\,\Delta = 0$ is the vertical line $x = 0$, and $\text{Im}\,\Delta = 0$ is the horizontal line $y = 0$. Their crossing is the vortex.

An antivortex looks like $\Delta(x,y) = x - iy$: the same two nodal lines cross, but the phase winding has opposite handedness.

### 6.3 Domain Walls vs. Vortices

Two distinct objects:

**A real dimer domain wall** — one scalar sign change: $\text{Re}\,\Delta = 0$. It connects two monomer/radical/edge defects. This is like an SSH soliton line. It has no chirality by itself.

**A complex Kekulé vortex** — requires both: $\text{Re}\,\Delta = 0$ **and** $\text{Im}\,\Delta = 0$. The bond alternation amplitude vanishes ($|\Delta| = 0$) and the phase winds by $\pm 2\pi$. In a $\mathbb{Z}_3$-locked system, this often appears as three domain walls meeting at a core.

### 6.4 The "Second Nodal Line" Problem

If one defect pair generates a domain wall (one nodal line), where does the second nodal line come from?

Several possibilities:
1. **Naturally from the boundary**: the molecular edge imposes boundary conditions on the Kekulé phase. A protonated edge, zigzag edge, or armchair edge can pin the local dimer orientation, providing the second constraint.
2. **From a second defect pair**: if two independent defect/domain-wall constraints cross, the intersection is a vortex core.
3. **From the $\mathbb{Z}_3$ structure**: three domain walls meeting at a point naturally provide the crossing.

### 6.5 Detection on the Lattice

For each bond $b = (i,j)$, check sign changes:

```
Re(z_i) * Re(z_j) < 0  →  Re nodal line crosses this bond
Im(z_i) * Im(z_j) < 0  →  Im nodal line crosses this bond
```

A vortex core is near a ring where **both** nodal networks cross **and** the ring winding $|W_R| > 0.5$.

If only one nodal line exists, it is a real dimer domain wall, not a complex vortex. The winding test is more robust than the nodal-crossing test alone.

---

## 7. The Ginzburg–Landau Free Energy

### 7.1 The Energy Functional

Let $\Delta = A\, e^{i\phi}$. The local free energy is:

$$
V(A, \phi) = \frac{a}{2}\, A^2 + \frac{b}{4}\, A^4 - \lambda\, A^3 \cos(3\phi)
$$

Each term has a direct physical interpretation:

### 7.2 Term-by-Term Interpretation

**$\frac{a}{2}\, A^2$ — cost or tendency to dimerize:**
- If $a > 0$: bond alternation costs energy; the undistorted aromatic state ($A = 0$) is preferred.
- If $a < 0$: the system wants spontaneous Kekulé/Peierls distortion ($A > 0$).
- Physically: electronic Peierls/Jahn–Teller energy gain vs. elastic cost of distorting the σ-bond backbone.

**$\frac{b}{4}\, A^4$ — saturation:**
- If $a < 0$, the quadratic alone would make $A$ grow without bound. The quartic stabilizes it: $A_{\text{eq}} \sim \sqrt{-a/b}$.
- Physically: nonlinear elasticity; bond order cannot grow infinitely (steric hindrance, electronic repulsion).

**$-\lambda\, A^3 \cos(3\phi)$ — three Kekulé choices ($\mathbb{Z}_3$ locking):**
- Graphene has three equivalent Kekulé bond patterns, so the energy must be invariant under $\phi \to \phi + 2\pi/3$.
- The lowest phase-locking term allowed by this symmetry is $\cos(3\phi)$.
- It selects $\phi = 0,\; 2\pi/3,\; 4\pi/3$ — the three bond directions.
- Physically: the lattice says "you may not choose arbitrary continuous dimer orientation; only three bond patterns are truly equivalent."

### 7.3 The Full Free Energy on the Graph

$$
F[\Delta] = \sum_{\langle i,j \rangle} \frac{\kappa}{2}\, |\Delta_i - \Delta_j|^2 + \sum_i \left[ \frac{a}{2}\, A_i^2 + \frac{b}{4}\, A_i^4 - \lambda\, A_i^3 \cos(3\phi_i) \right] + F_{\text{defect}}
$$

- $\kappa$: spatial stiffness (penalizes rapid phase changes — the graph Laplacian term)
- $F_{\text{defect}}$: defect pinning and core suppression energies

### 7.4 Defect Energy

$$
F_{\text{defect}} = \sum_i \left[ g_i\, |\Delta_i - \Delta_i^{\text{preferred}}|^2 + h_i\, A_i^2 \right]
$$

- $g_i$ pins the nearby Kekulé phase to a preferred orientation
- $h_i > 0$ suppresses $A_i$, creating a vortex core
- $h_i < 0$ enhances local bond alternation
- The sign/phase of $\Delta_i^{\text{preferred}}$ depends on which local valence pattern the chemical group favors

---

## 8. The Pauli–Idempotency Analogy

### 8.1 Fluid Projection vs. Quantum Projection

In the "Simple and Fast Fluids" shader, after advecting the velocity field, a **pressure projection** enforces incompressibility:

$$
\mathbf{u} \to \mathbf{u} - \nabla p, \qquad \nabla \cdot \mathbf{u} = 0
$$

In quantum mechanics, the density matrix must satisfy **idempotency** (Pauli exclusion):

$$
D^2 = D, \qquad n_\alpha \in \{0, 1\}
$$

Both are projection steps onto a constraint manifold:

| Fluid mechanics | Quantum mechanics |
|---|---|
| $\nabla \cdot \mathbf{u} = 0$ | $D^2 = D$, $0 \leq D \leq 1$ |
| Pressure projection | Chemical potential / Pauli repulsion |
| Prevents mass pile-up | Prevents double occupancy |

### 8.2 Local Approximation

Full idempotency is spectral/nonlocal: $D^2_{ij} = \sum_k D_{ik} D_{kj} = D_{ij}$. However, for the Kekulé fluid model, a **local approximate Pauli projection** serves as the useful "game physics" analogue.

Instead of projecting density, we project the local bond field into a chemically allowed dimer/bond-order manifold:

$$
0 \leq x_{ij} \leq 1, \qquad \sum_{j \in N(i)} x_{ij} = \text{targetVal}_i
$$

For normal carbon: $\sum_j x_{ij} = 1$ (one π dimer on average).
For a radical/protonated/sp³ site: $\sum_j x_{ij} < 1$ (dimer deficiency).

This is a **local valence constraint**, not full density-matrix idempotency. It is much closer to the existing bond-order solver and is chemically transparent: it says "each carbon participates in at most one π dimer."

### 8.3 The Valence Projection as Chemical Incompressibility

The analogy is precise:

- **Fluid**: pressure projection prevents compression beyond $\rho = \text{const}$
- **Chemistry**: valence projection prevents over-bonding beyond $\sum x_{ij} = 1$

Just as the fluid shader prevents mass from piling up in one pixel, the valence projection prevents more than one electron from occupying a single π dimer. A radical/monomer defect becomes $\sum_j x_{ij} = 0$ or $< 1$ — a "hole" in the dimer covering.

---

## 9. Model A: Kekulé Fluid Solver — Complete Specification

### 9.1 Design Decisions

The following choices are frozen for the v0 implementation:

1. **Dynamics**: Complex Ginzburg–Landau equation with both dissipative and conservative terms: $\dot{z} = -(\eta + i\Omega)\, g$, where $g = \delta F / \delta z^*$. No advection, no fluid velocity, no graph pressure projection in v0.
2. **Integrator**: RK4 (not Euler), because the $-i\Omega g$ part is oscillatory.
3. **Variables on atoms and bonds**: atoms carry $z_i$ (complex), $q_i$, `targetVal`, `defect`; bonds carry $x_{ij} \in [0,1]`. Rings are diagnostic only.
4. **Projection**: local Jacobi-style valence projection, not full density-matrix idempotency.

### 9.2 Data Structures

```cpp
struct Atom {
    Vec2  pos;
    int   sub;          // +1 for A, -1 for B
    int   bonds[3];     // incident bond ids, -1 if absent
    float targetVal;    // desired sum of x around atom (1.0 for normal C)
    float defect;       // 0 normal, 1 fully π-removed
    float pinStrength;  // 0 no pin, 1 hard phase pin
    float pinPhase;     // preferred Kekulé phase
    complex z;          // local complex Kekulé order
};

struct Bond {
    int   iA;      // A-sublattice atom
    int   iB;      // B-sublattice atom
    int   dir;     // 0, 1, 2 (honeycomb bond direction)
    float x;       // normalized dimer/bond order, 0..1
    float xRaw;    // temporary unprojected bond order
};
```

### 9.3 Global Conventions

```cpp
aCC   = 1.0;     // C-C bond length unit
hbar  = 1.0;
dtA   = 0.02;    // Model A timestep

thetaDir[0] = 0;
thetaDir[1] = 2*PI/3;
thetaDir[2] = 4*PI/3;
omega[d]    = exp(i * thetaDir[d]);
```

Every bond is oriented from sublattice A to sublattice B.

### 9.4 Base Bond Order

For each atom $i$, define its active degree: `deg[i] = number of existing incident bonds`.

For each bond $b = (i,j)$, the neutral local average:

$$
x^0_b = \frac{1}{2}\left(\frac{\texttt{targetVal}_i}{\deg_i} + \frac{\texttt{targetVal}_j}{\deg_j}\right)
$$

For interior graphene ($\deg = 3$): $x^0 = 1/3$. For edge carbon ($\deg = 2$): $x^0 = 1/2$. This automatically distinguishes interior from finite PAH edges.

### 9.5 Step 1: Reconstruct $z_i$ from Bonds (`bondsToZ`)

For every atom $i$:

$$
\bar{x}_i = \frac{1}{\deg_i} \sum_{b \in N(i)} x_b
$$

$$
z_i = \sum_{b \in N(i)} (x_b - \bar{x}_i)\, e^{i\,\theta_{\text{dir}(b)}}
$$

```cpp
void bondsToZ() {
    for atom i:
        float xavg = 0; int nb = 0;
        for b in atoms[i].bonds:
            if (b < 0) continue;
            xavg += bonds[b].x; nb++;
        if (nb == 0) { atoms[i].z = 0; continue; }
        xavg /= nb;
        complex z = 0;
        for b in atoms[i].bonds:
            if (b < 0) continue;
            z += (bonds[b].x - xavg) * omega[bonds[b].dir];
        atoms[i].z = z;
}
```

For a perfect Kekulé pattern ($x = 1$ on one bond, $x = 0$ on two others, $\bar{x} = 1/3$): $|z| = 1$. This is the natural normalization.

### 9.6 Step 2: Apply Defect and Boundary Pinning (`applyPinsToZ`)

For a defect core:

$$
z_i \leftarrow (1 - \texttt{defect}_i)\, z_i
$$

For a pinned boundary atom:

$$
z_i \leftarrow (1 - p_i)\, z_i + p_i\, A_{\text{pin}}\, e^{i\,\phi_{\text{pin}}}
$$

```cpp
void applyPinsToZ() {
    for atom i:
        atoms[i].z *= (1.0 - atoms[i].defect);
        float p = atoms[i].pinStrength;
        if (p > 0) {
            complex zpin = polar(A_pin, atoms[i].pinPhase);
            atoms[i].z = (1 - p) * atoms[i].z + p * zpin;
        }
}
```

**Important physical note**: A protonated atom should NOT be both suppressed ($\texttt{defect}=1$, $z \to 0$) and hard-pinned to finite amplitude ($\texttt{pinStrength}=1$, $z \to A_{\text{pin}}\, e^{i\phi}$). These are contradictory. The correct model is either:
- Pin the **neighboring** atoms' phase, not the removed atom, or
- Use phase-only pinning that does not impose amplitude, or
- Remove the protonated atom from the π graph entirely and pin adjacent boundary bonds.

### 9.7 Step 3: Evolve $z_i$ by Complex Ginzburg–Landau (`evolveZ_RK4`)

The equation of motion:

$$
\frac{dz_i}{dt} = -\eta\, g_i - i\,\Omega\, g_i
$$

where the gradient is:

$$
g_i = \kappa \sum_{j \in N(i)} (z_i - z_j) + r\, z_i + u\, |z_i|^2\, z_i - \lambda\, z_i^{*2} + k_{\text{pin}}\, p_i\, (z_i - z_i^{\text{pin}}) + k_{\text{core}}\, \texttt{defect}_i\, z_i
$$

**Default parameters:**

| Parameter | Value | Meaning |
|---|---|---|
| $\kappa$ | 0.20 | Spatial stiffness (penalizes phase variation) |
| $r$ | $-1.0$ | Wants finite Kekulé amplitude ($a < 0$) |
| $u$ | $1.0$ | Amplitude saturation ($b > 0$) |
| $\lambda$ | 0.15 | $\mathbb{Z}_3$ Kekulé locking |
| $\eta$ | 0.08 | Dissipative relaxation rate |
| $\Omega$ | 1.0 | Conservative Schrödinger-like phase rotation |
| $k_{\text{pin}}$ | 2.0 | Pinning stiffness |
| $k_{\text{core}}$ | 5.0 | Defect core suppression |
| $A_{\text{pin}}$ | 1.0 | Pinned amplitude |
| dtA | 0.02 | Timestep |

**Term interpretation:**
- $\kappa \sum (z_i - z_j)$: graph Laplacian — penalizes rapid spatial variation of Kekulé order
- $r\, z_i$ ($r < 0$): drives spontaneous Kekulé distortion
- $u\, |z_i|^2\, z_i$ ($u > 0$): prevents infinite amplitude
- $-\lambda\, z_i^{*2}$: $\mathbb{Z}_3$ locking — selects $\phi = 0, 2\pi/3, 4\pi/3$
- $k_{\text{pin}}\, p_i\, (z_i - z_i^{\text{pin}})$: soft pinning toward preferred phase
- $k_{\text{core}}\, \texttt{defect}_i\, z_i$: defects suppress order amplitude

Use **RK4** (not Euler), because the $-i\Omega g$ part is oscillatory:

```cpp
void evolveZ_RK4() {
    z0 = copy of current z array  // device-to-device copy

    k1[i] = rhs_z(i, z0)
    k2[i] = rhs_z(i, z0 + 0.5 * dtA * k1)
    k3[i] = rhs_z(i, z0 + 0.5 * dtA * k2)
    k4[i] = rhs_z(i, z0 + dtA * k3)

    for atom i:
        atoms[i].z = z0[i] + dtA * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6
}
```

**Implementation note**: The $z_0$ save must be a **device-to-device copy** (`cl.enqueue_copy(queue, b_z_temp, b_z_real)`), NOT a GPU→CPU→GPU round-trip through host NumPy arrays.

### 9.8 Step 4: Convert $z_i$ Back to Bond Orders (`zToRawBonds`)

For bond $b = (i,j)$:

$$
z_b = \frac{1}{2}(z_i + z_j), \quad A_b = \min(|z_b|, 1), \quad \phi_b = \arg z_b
$$

$$
a_b = 0.8 \cdot \min(x^0_b,\; 1 - x^0_b)
$$

$$
x^{\text{raw}}_b = x^0_b + a_b\, A_b\, \cos(\phi_b - \theta_{\text{dir}(b)})
$$

Clamp: $x^{\text{raw}}_b \in [0, 1]$.

```cpp
void zToRawBonds() {
    for bond b:
        complex zb = 0.5 * (atoms[bonds[b].iA].z + atoms[bonds[b].iB].z);
        float A = min(abs(zb), 1.0f);
        float phi = atan2(imag(zb), real(zb));
        float x0 = bondBase[b];
        float amp = 0.8 * min(x0, 1.0f - x0);
        bonds[b].xRaw = clamp(x0 + amp * A * cos(phi - thetaDir[bonds[b].dir]), 0.0f, 1.0f);
}
```

Then set: `bonds[b].x = bonds[b].xRaw;`

**Note on amplitude consistency**: The factor 0.8 caps the alternation at 80% of the maximum possible. For $x_0 \approx 1/3$, this gives $a_b \approx 0.267$, so even with $A = 1$ the bond pattern is roughly $(0.6, 0.2, 0.2)$, not a perfect Kekulé $(1, 0, 0)$. Reconstructing $z$ from that gives $|z|$ much less than 1. The Ginzburg–Landau dynamics tries to relax toward $|z| \sim 1$, but the bond mapping cannot sustain that scale after projection. This explains observed $|z|$ values around 0.2–0.5. A calibration test (initialize uniform phase, run `zToRawBonds → project → bondsToZ`, report fixed-point $|z|$) is recommended.

### 9.9 Step 5: Valence Projection (`projectBondOrders`)

Enforce:

$$
\sum_{b \in N(i)} x_b = \texttt{targetVal}_i
$$

Use 12 Jacobi-like projection iterations with red-black sublattice ordering (race-free on GPU):

```cpp
nProj = 12;

void projectBondOrders() {
    for (int it = 0; it < nProj; it++) {
        // Pass 1: A-sublattice atoms
        for atom i where sub[i] == +1:
            float s = 0; int nb = 0;
            for b in atoms[i].bonds:
                if (b < 0) continue;
                s += bonds[b].x; nb++;
            if (nb == 0) continue;
            float corr = (atoms[i].targetVal - s) / float(nb);
            for b in atoms[i].bonds:
                if (b < 0) continue;
                bonds[b].x = clamp(bonds[b].x + 0.5f * corr, 0.0f, 1.0f);

        // Pass 2: B-sublattice atoms
        for atom i where sub[i] == -1:
            // same logic
    }
}
```

The factor 0.5 accounts for bond sharing between two atoms. With red-black ordering, only one endpoint is active per pass, so this is an under-relaxed projection. A relaxation parameter $\omega_{\text{proj}}$ could be exposed (0.5 = current, 1.0 = exact active-sublattice correction).

### 9.10 Complete Timestep

```cpp
void stepModelA() {
    bondsToZ();           // x^n → z^n
    applyPinsToZ();       // impose chemical defects / boundary conditions
    evolveZ_RK4();        // z^n → z^{n+1}
    zToRawBonds();        // z^{n+1} → raw bond orders
    for bond b:
        bonds[b].x = bonds[b].xRaw;
    projectBondOrders();  // enforce local valence constraints
    bondsToZ();           // optional: reconstruct for diagnostics
}
```

### 9.11 Initialization

**Uniform Kekulé state:**

```cpp
for atom i:
    atoms[i].z = polar(1.0, phi0);  // phi0 = 0
zToRawBonds();
for bond b: bonds[b].x = bonds[b].xRaw;
projectBondOrders();
bondsToZ();
```

**Vortex/antivortex pair:**

For defects $k$ with positions $\mathbf{R}_k$ and winding $s_k = \pm 1$:

$$
\phi_i = \sum_k s_k\, \text{atan2}(y_i - Y_k,\; x_i - X_k)
$$

$$
A_i = \prod_k \tanh\!\left(\frac{|\mathbf{r}_i - \mathbf{R}_k|}{r_{\text{core}}}\right), \qquad r_{\text{core}} = 2.0\, a_{\text{CC}}
$$

$$
z_i = A_i\, e^{i\phi_i}
$$

**Edge protonation (=NH⁺−):**

```cpp
atoms[p].defect    = 0.5;   // partial suppression
atoms[p].targetVal = 0.5;
atoms[p].pinStrength = 1.0;
atoms[p].pinPhase    = thetaDir[d_preferred];
```

### 9.12 Diagnostics

**Local π bond-density:**

$$
\rho_i^{\text{bond}} = \sum_{b \in N(i)} x_b, \qquad \delta\rho_i^{\text{bond}} = \rho_i^{\text{bond}} - \texttt{targetVal}_i
$$

**Peierls distortion:**

$$
u_b = -\alpha_x\, (x_b - x^0_b), \qquad \texttt{bondLength}_b = a_{\text{CC}} + u_b
$$

Large $x_b$ → shorter bond. Use $\alpha_x = 0.05\, a_{\text{CC}}$.

**Nodal lines:** Check sign changes of $\text{Re}(z)$ and $\text{Im}(z)$ across bonds.

**Ring winding:**

$$
W_R = \frac{1}{2\pi} \sum_{i \in R} \text{wrap}\!\left[\arg(z_{i+1}) - \arg(z_i)\right]
$$

- $|W_R| < 0.25$: no vortex
- $W_R > 0.5$: vortex
- $W_R < -0.5$: antivortex

---

## 10. Model B: Dirac / Tight-Binding Solver — Complete Specification

### 10.1 Two Versions

There are two implementations of Model B in the codebase:

1. **Lattice tight-binding** (current, `DiracLattice_ocl.py`): One complex amplitude $\psi_i$ per carbon atom on the honeycomb graph. The Dirac cones emerge automatically from the graph topology. No separate grid needed.

2. **Continuum 4-component Dirac** (legacy, `Dirac4_ocl.py`): A 4-component spinor $\Psi = (\psi_{A,K}, \psi_{B,K}, \psi_{A,K'}, \psi_{B,K'})^T$ on a Cartesian grid, with explicit Dirac matrices. This is the long-wavelength limit of the lattice model.

### 10.2 Lattice Tight-Binding (Current)

One complex amplitude per π orbital:

$$
i\hbar\, \frac{d\psi_i}{dt} = \epsilon_i\, \psi_i - \sum_{j \in N(i)} t_{ij}\, \psi_j
$$

**Building $t_{ij}$ from bond order:**

$$
t_{ij} = t_0 + \gamma\, (x_{ij} - x_0)
$$

**Defects / onsite potentials:**

$$
\epsilon_i = \epsilon_0 + V_{\text{defect}_i} + V_{\text{charge}_i}
$$

**Important**: For protonated/sp³ carbon, $V_{\text{def}}$ must be much larger than the bandwidth ($\sim 6t_0$). Using $V_{\text{def}} = 2$ with $t_0 = 1$ leaves the defect state inside the band. Recommended: $V_{\text{def}} = 20$ or higher, or physically remove the atom and all its π bonds from the Hamiltonian.

**RK4 propagation:**

```cpp
for each timestep:
    // 1. Build t_ij from bond-order/distortion field
    for bond b: t_b = t0 + gamma * (x_b - x0);

    // 2. Apply defects
    for atom i: eps_i = eps0 + V_defect_i;

    // 3. Tight-binding propagation (RK4)
    for atom i:
        Hpsi_i = eps_i * psi_i;
        for bond b around i:
            j = otherAtom(b, i);
            Hpsi_i += -t_b * psi_j;
    // ... RK4 stages ...

    // 4. Renormalize
    normalize(psi);

    // 5. Bond coherence/current diagnostics
    for bond b = (i,j):
        C_b = real(conj(psi_i) * psi_j);  // bond coherence
        J_b = imag(conj(psi_i) * psi_j);  // current

    // 6. Optional: feed C_b back into bond order
    for bond b: x_b += eta_elec * (C_b - C0_b);

    // 7. Project bond orders chemically
    projectBondOrders();
```

**Dense diagonalization** (for spectrum, LDOS):

The `build_hamiltonian()` method constructs the full dense matrix incorporating Kekulé-modulated hopping and onsite defect potentials. `diagonalize()` performs the eigenvalue decomposition. `compute_ldos()` calculates the local density of states with Gaussian broadening.

### 10.3 Continuum 4-Component Dirac (Legacy)

**Basis:**

$$
\Psi = \begin{pmatrix} \psi_{A,K} \\ \psi_{B,K} \\ \psi_{A,K'} \\ \psi_{B,K'} \end{pmatrix}
$$

**Equation:**

$$
i\, \frac{\partial \Psi}{\partial t} = \left[-i\, v_F\, (\alpha_x \partial_x + \alpha_y \partial_y) + \Delta_R\, M_1 + \Delta_I\, M_2 + V\, I_4\right] \Psi
$$

**Explicit matrices** (verified correct):

$$
\alpha_x = \tau_z \otimes \sigma_x = \begin{pmatrix} 0&1&0&0 \\ 1&0&0&0 \\ 0&0&0&-1 \\ 0&0&-1&0 \end{pmatrix}
$$

$$
\alpha_y = \tau_0 \otimes \sigma_y = \begin{pmatrix} 0&-i&0&0 \\ i&0&0&0 \\ 0&0&0&-i \\ 0&0&i&0 \end{pmatrix}
$$

$$
M_1 = \tau_x \otimes \sigma_x = \begin{pmatrix} 0&0&0&1 \\ 0&0&1&0 \\ 0&1&0&0 \\ 1&0&0&0 \end{pmatrix}
$$

$$
M_2 = \tau_y \otimes \sigma_x = \begin{pmatrix} 0&0&0&-i \\ 0&0&-i&0 \\ 0&i&0&0 \\ i&0&0&0 \end{pmatrix}
$$

These matrices anticommute with the kinetic Dirac matrices and with each other, so $(\Delta_R, \Delta_I)$ act as the two components of a complex Kekulé mass.

**Grid:** Cartesian, $N_x \times N_y$, $dx = dy = 1.0$. Central differences for spatial derivatives. Periodic boundaries for v0.

**Stability:** $dt_B \leq 0.2\, dx / v_F$.

**Dirac mass from vortices:**

$$
\Delta(x,y) = \Delta_0 \prod_k \tanh\!\left(\frac{|\mathbf{r} - \mathbf{R}_k|}{r_{\text{core}}}\right) e^{i \sum_k s_k \text{atan2}(y - Y_k, x - X_k)}
$$

with $\Delta_0 = 0.3$, $r_{\text{core}} = 2.0$.

**Dirac mass from Model A:** Interpolate $z_i$ from atoms to grid using Gaussian weighting:

$$
\Delta(\mathbf{r}) = \frac{\sum_i z_i\, \exp\!\left[-\frac{|\mathbf{r} - \mathbf{r}_i|^2}{2\sigma_I^2}\right]}{\sum_i \exp\!\left[-\frac{|\mathbf{r} - \mathbf{r}_i|^2}{2\sigma_I^2}\right]}, \qquad \sigma_I = 0.75\, a_{\text{CC}}
$$

**Diagnostics:**

$$
\rho = \Psi^\dagger \Psi, \qquad S_K = \Psi^\dagger M_1 \Psi + i\, \Psi^\dagger M_2 \Psi
$$

Compare: $\arg(S_K)$ vs. $\arg(\Delta)$, and $\rho$ peaks vs. vortex cores of $\Delta$.

### 10.4 Scope of Model B in v0

Model B is a **single-quasiparticle Dirac propagation** in a prescribed Kekulé texture. It does **not**:
- Compute many-electron bond order (would require density matrix or occupied-state sums)
- Include Pauli projection / idempotency
- Self-consistently update $\Delta$ from $\Psi$
- Compute true ground-state bond order

This is enough for the key question: *does the Dirac quasiparticle localize/respond at the same topological defects produced by the dimer-smoke model?*

---

## 11. Comparison of the Two Models

### 11.1 The Causal Chain

$$
\text{chemical defects / edge pins}
\;\rightarrow\;
x_{ij} \text{ (bond orders, Model A)}
\;\rightarrow\;
z_i \text{ (complex Kekulé order, Model A)}
\;\rightarrow\;
\Delta(x,y) \text{ (interpolated to grid)}
\;\rightarrow\;
\Psi(x,y,t) \text{ (Dirac quasiparticle, Model B)}
\;\rightarrow\;
\rho(x,y,t),\; S_K(x,y,t)
$$

### 11.2 What Should Agree

If the models are topologically equivalent, they should agree on:
- Vortex/antivortex positions
- Domain-wall connectivity
- Qualitative defect propagation
- Whether a defect is pinned or mobile
- Whether two defects annihilate or remain separated
- Nodal line topology: $\text{Re}(z) = 0$ and $\text{Im}(z) = 0$ networks

### 11.3 What Will Not Automatically Agree

- Exact charge density
- Zero-mode occupancy
- Fractional charge ($Q = 0, \pm e/2, \pm e$)
- Spin state ($S = 1/2$)
- Exact tunneling spectrum / LDOS
- Quantitative energetics

These are genuinely fermionic/electronic observables that require the Dirac/tight-binding spectrum, not just classical dimer hydrodynamics. The Hou–Chamon–Mudry zero modes [4] are tied to the Dirac spectrum, not only to classical bond-order topology.

### 11.4 Comparison Protocol

Use the same atom/bond graph and same defects. For each frame:

```
// From Model A
DeltaA_i = localKekuleFromBondOrders(xA)

// From Model B
C_b = real(conj(psi_i) * psi_j)  // bond coherence
DeltaB_i = localKekuleFromBondCoherence(C_b)

compare:
    abs(DeltaA_i), arg(DeltaA_i)  vs  abs(DeltaB_i), arg(DeltaB_i)
    vortex winding on rings
    Re(Delta) = 0 nodal network
    Im(Delta) = 0 nodal network
    bond-order pattern x_ij
    charge/depletion q_i
```

### 11.5 Correspondence Table

| Model A (Kekulé fluid) | Model B (Dirac) |
|---|---|
| Vortex cores ($\|z\| \to 0$) | $\rho$ localization |
| $\arg(z)$ | $\arg(S_K)$ |
| Nodal crossing | Density/mass core |
| Domain wall | SSH-like soliton |

---

## 12. Aromaticity and Antiaromaticity

### 12.1 The Dimer-Field Criterion

Aromaticity can be partially described in the dimer-field language:

**Aromatic** $\approx$ phase/order closes smoothly around ring ($W_R = 0$, $A_{\min}$ large, frustration small)

**Antiaromatic** $\approx$ phase/order cannot close without a node/domain wall/defect ($W_R \neq 0$ or $A_{\min}$ collapses or frustration high)

### 12.2 Frustration Score

For each ring $R$, compute for each of the three Kekulé phases $\Phi_m = 2\pi m / 3$ ($m = 0, 1, 2$):

$$
E_m(R) = \sum_{b \in R} \left(x_b - x^{(m)}_b\right)^2
$$

Ring frustration: $F_R = \min_m E_m(R)$.

Also compute: $A_{\min,R} = \min_{i \in R} |z_i|$.

### 12.3 Classification

```
if |W_R| < 0.25 and F_R small and A_min large:
    aromatic-like / Kekulé-compatible

if F_R large or A_min collapses or |W_R| > 0.5:
    frustrated / antiaromatic-like / defect-threaded
```

### 12.4 Connection to Hückel's Rule

For $(4n+2)$ rings: the π phase can close with filled-shell stability — no nodal line cuts through the ring.

For $(4n)$ rings: the boundary condition tends to create degeneracy/frustration; the system escapes by bond alternation, distortion, spin polarization, or symmetry breaking. In the dimer-field language: it wants to create a domain wall, node, or Peierls deformation.

This is **not** the whole of aromaticity — magnetic ring currents, orbital occupancy, and Hückel filling are quantum spectral effects. But the dimer-field picture captures the **real-space valence-bond/frustration side** of aromaticity.

---

## 13. Speculative Physical Interpretations

The preceding sections treat the complex Kekulé order parameter $z_i$ and the two-component wavefunction as somewhat abstract mathematical objects. This section explores two concrete physical pictures that give them a more tangible meaning.

**A note on what is established vs. speculative:** The individual physical ingredients — Kekulé/SSH dimerization, electron-phonon coupling, lattice displacement fields with divergence/curl/saddle topology, density matrices, Pauli idempotency, Mulliken charges — are all well-established and uncontroversial. What is **speculative** is the specific proposal to combine them into a simplified single-wavefunction effective model that avoids full diagonalization, and the claim that this captures the essential topological physics. We flag speculative claims explicitly throughout.

### 13.1 Picture 1: Lattice Displacement Field (Static Phonons)

#### The Idea

**[Established]** The Kekulé/Peierls dimerization is a static lattice distortion — a frozen optical phonon mode. It arises from electron-phonon coupling: the electronic energy gain from opening a gap at the Fermi level outweighs the elastic cost of distorting the bonds. This is the standard Peierls instability of the honeycomb lattice.

**[Established]** The two-component field $(u_x, u_y)$ — the static lattice displacement of each carbon atom from its equilibrium position — is a real, physically measurable quantity. It can have divergence, convergence, shear, and rotation, just like any 2D vector field. These are standard vector calculus operations on the elastic displacement field.

**[Speculative]** The proposal to identify the complex Kekulé order parameter $z_i = A_i e^{i\phi_i}$ with the **envelope** of this displacement field — and to use the Ginzburg–Landau equation to evolve this envelope — is a simplification. The full electron-phonon coupling problem involves both the displacement field (bosonic, single-component, easy to describe) and the electronic response (fermionic, requiring diagonalization or density-matrix methods). The Ginzburg–Landau model integrates out the electrons into an effective free energy for the displacement envelope. This is a legitimate coarse-graining, but the precise form of the effective potential and the validity of the envelope approximation near vortex cores remain open questions.

In graphene, the relevant optical phonon at the $K$ point has atoms on opposite sublattices moving in opposite directions. The displacement field therefore alternates sign at the lattice scale: A-sublattice atoms move one way, B-sublattice atoms move the other. This **short-wavelength** spatial alternation **is the ground state** — it is the regular Kekulé bond alternation, not a defect. It is static (zero frequency in time), but varies rapidly in space (short wavelength $\lambda \sim a_{\text{CC}}$).

#### Nodal Lines from Displacement

Consider two neighboring atoms $A$ and $B$:

- If $A$ moves right ($u_x > 0$) and $B$ moves left ($u_x < 0$), they approach each other: the bond shortens (double-bond-like). A nodal plane of $u_x$ lies between them.
- If $A$ moves left and $B$ moves right, they separate: the bond lengthens (single-bond-like). Again a nodal plane of $u_x$.
- The same logic applies to $u_y$.

The two nodal networks — $u_x = 0$ and $u_y = 0$ — are curves in 2D where the respective displacement component changes sign. Their intersections are points where both components vanish: **displacement vortices**.

#### Vector Calculus Interpretation

The intersection patterns of the two nodal networks map onto standard vector calculus operations:

- **Divergence** ($\nabla \cdot \mathbf{u} > 0$): both $u_x$ and $u_y$ point away from a central point — the lattice expands locally. This is a dilatation center.
- **Convergence** ($\nabla \cdot \mathbf{u} < 0$): both components point inward — the lattice contracts. This is a compression center.
- **Shear / squashing**: $u_x$ converges while $u_y$ diverges (or vice versa) — the lattice is squeezed in one direction and stretched in the other. This corresponds to a uniaxial strain vortex.
- **Rotation** ($\nabla \times \mathbf{u} \neq 0$): the displacement field circulates — left side moves up, right side moves down, top moves left, bottom moves right. This is a rotational vortex in the displacement field.

These are the same topological classifications as in fluid mechanics — divergence, curl, and saddle — applied to the elastic displacement field instead of velocity.

#### The Crucial Distinction: Ground State vs. Defect

**[Established]** The regular sublattice-alternating displacement pattern (A moves in, B moves out, alternating at every bond) is **not** a defect. It is the Kekulé/SSH ground state — the short-wavelength optical phonon texture. It is static in time ($\omega = 0$) but rapidly varying in space ($\lambda \sim a_{\text{CC}}$). The nodal lines of this pattern are dense and regular, reflecting the lattice-scale alternation.

**[Speculative]** A **defect** is proposed to be an irregularity in this regular alternation — a **long-wavelength** modulation of the short-wavelength pattern. Just as a phonon envelope function modulates a carrier wave, the Kekulé vortex is a slow (in space) twist in the phase of the short-wavelength optical-phonon alternation. The vortex core is where the envelope amplitude goes to zero: the regular alternation pattern breaks down. This envelope picture is a useful simplification but is not rigorously derived from the microscopic electron-phonon Hamiltonian.

In this picture:

$$
\mathbf{u}(\mathbf{r}) = \underbrace{A(\mathbf{r})}_{\text{envelope}} \cdot \underbrace{e^{i\phi(\mathbf{r})}}_{\text{slow phase}} \cdot \underbrace{\mathbf{e}_{\text{sublattice}}}_{\text{short-wavelength alternation}}
$$

The complex order parameter $z_i = A_i\, e^{i\phi_i}$ is the **envelope** of the optical phonon displacement, not the displacement itself. The nodal lines of $\text{Re}(z)$ and $\text{Im}(z)$ are nodal lines of the **envelope**, not of the raw displacement. This is why they are smooth and slowly varying, not lattice-scale.

#### Connection to the Solver

In Model A, the Ginzburg–Landau equation evolves the envelope $z_i$. The lattice displacement itself is reconstructed from $z_i$ via the bond-order relation:

$$
u_b = -\alpha_x\, (x_b - x^0_b), \qquad \texttt{bondLength}_b = a_{\text{CC}} + \nu_b$$

The displacement field $\mathbf{u}_i$ can be recovered (up to a sublattice sign) from the bond-length changes. The divergence and curl of $\mathbf{u}$ then provide physical diagnostics:

- $\nabla \cdot \mathbf{u}$: local compression/dilation — corresponds to charge accumulation/depletion
- $\nabla \times \mathbf{u}$: local rotation — corresponds to a Kekulé vortex in the envelope

This picture makes the Schrödinger-smoke analogy very concrete: the "fluid" is the elastic displacement field of the lattice, and the "vortex" is a topological defect in the phonon envelope.

#### Wavelength vs. Frequency: A Clarification

It is important to distinguish **spatial wavelength** from **temporal frequency** when describing the Kekulé distortion:

- The Kekulé bond alternation has a **short wavelength** in space ($\lambda \sim a_{\text{CC}}$, alternating every bond) but **zero frequency** in time — it is a static distortion, a frozen mode. It does not oscillate; it is the equilibrium ground state.
- The Kekulé **vortex** (defect) has a **long wavelength** in space (the envelope varies over many lattice spacings, $r_{\text{core}} \gg a_{\text{CC}}$) and is also static (or slowly evolving under the Ginzburg–Landau relaxation).

In the harmonic approximation, a phonon's temporal frequency $\omega$ is related to its energy $E = \hbar\omega$, and its spatial wavelength $\lambda$ is related to its wavevector $k = 2\pi/\lambda$ via the dispersion relation $\omega(k)$. For the Kekulé optical phonon at the $K$ point, $k = |K| \sim 1/a_{\text{CC}}$ (short wavelength), and the mode is at the bottom of the optical branch — but it is **frozen** (static), so $\omega = 0$ in the sense of dynamical evolution.

However, the Ginzburg–Landau potential for the Kekulé order parameter is **anharmonic** — it is a double-well (or rather, a three-well $\mathbb{Z}_3$) potential of the form $V(A, \phi) = \frac{r}{2}A^2 + \frac{u}{4}A^4 - \lambda A^3 \cos(3\phi)$ with $r < 0$. This is not a quadratic harmonic well; it has discrete minima at finite $A$ and specific $\phi$ values. The relationship $E = \hbar\omega$ holds only for small oscillations around a harmonic minimum (the curvature of the well gives the oscillation frequency). For the static ground state itself, the relevant quantity is the potential energy at the minimum, not a frequency. The anharmonic nature means:

- The system can **bifurcate** (choose one of three equivalent Kekulé patterns) — this is a spontaneous symmetry breaking, not a harmonic oscillation.
- Small fluctuations around the minimum are approximately harmonic (giving a phonon frequency), but the static distortion itself is a displacement of the equilibrium position, not an oscillation.
- The vortex is a **topological soliton** in this anharmonic potential — it cannot be described as a superposition of harmonic phonon modes.

So when we say "short-wavelength texture," we mean the spatial pattern alternates at the lattice scale. When we say "static" or "zero-frequency," we mean it does not evolve in time — it is the equilibrium configuration. The two concepts are independent: a pattern can be short-wavelength and static (Kekulé ground state), long-wavelength and static (vortex envelope), short-wavelength and dynamic (optical phonon propagation), or long-wavelength and dynamic (acoustic phonon, or vortex dynamics under Ginzburg–Landau evolution).

#### Limitations

- This is a **classical** picture: it describes lattice distortion, not electron density. The connection to electronic zero modes and fractional charge requires the Dirac/tight-binding model.
- The optical phonon at $K$ involves out-of-plane as well as in-plane motion; the 2D $(u_x, u_y)$ simplification captures only the in-plane component.
- The envelope approximation is valid only when the vortex core radius $r_{\text{core}} \gg a_{\text{CC}}$; at lattice scale the distinction between envelope and short-wavelength carrier pattern blurs.

---

### 13.2 Picture 2: Single-Wavefunction Density Matrix

#### The Idea

**[Established]** The full many-electron density matrix $\rho_{ij} = \langle c_i^\dagger c_j \rangle$ is the rigorous object that encodes all bonding information. In a tight-binding basis, $\rho_{ii}$ gives the Mulliken site occupancy (charge on atom $i$) and $\rho_{ij}$ gives the bond charge (Mulliken bond order between atoms $i$ and $j$). Computing $\rho$ rigorously requires diagonalizing the Hamiltonian and summing over all occupied molecular orbitals (or Kohn-Sham states) — an $O(N^3)$ operation.

**[Speculative]** Instead of this expensive computation, consider a **single complex wavefunction** $|\psi\rangle = \sum_i c_i |i\rangle$ and construct the density matrix from it:

$$
\rho_{ij} = c_i^*\, c_j$$

This is a rank-1 approximation: $\rho = |\psi\rangle\langle\psi|$. It is not the true ground-state density matrix (which is a sum over all occupied states), but the speculative proposal is that it captures the **bonding pattern topology** in a compact way — enough to reproduce vortex positions and domain-wall structure, if not exact spectroscopy.

**[Established]** The density matrix must satisfy idempotency ($\rho^2 = \rho$) as a consequence of Pauli exclusion. In density-matrix functional theories (DFTB, McWeeny purification, LNV), this idempotency constraint replaces diagonalization: instead of solving for orthogonal molecular orbitals, one enforces $\rho^2 = \rho$ via Lagrange multipliers or iterative purification.

**[Speculative]** In a nearest-neighbor tight-binding model, the density matrix is **super-sparse** (only onsite $\rho_{ii}$ and nearest-neighbor $\rho_{ij}$ elements are nonzero). The idempotency constraint $\rho^2 = \rho$ then becomes **local**: the condition that $\sum_k \rho_{ik}\rho_{kj} = \rho_{ij}$ involves only a few terms (the neighbors of $i$ and $j$). This raises the possibility of enforcing Pauli exclusion with only $O(N)$ local operations — the central speculative question explored below.

#### Real-Valued Amplitudes: The $\pm 1$ Picture

If we restrict to real-valued $c_i \in \{+1, -1\}$, the bond order between sites $i$ and $j$ is:

$$
\rho_{ij} = c_i\, c_j = \begin{cases} +1 & \text{if } c_i, c_j \text{ same sign (bonding)} \\ -1 & \text{if } c_i, c_j \text{ opposite sign (antibonding)} \end{cases}
$$

The onsite charge is $\rho_{ii} = c_i^2 = 1$ for all sites.

Now impose the **Pauli constraint**: each atom can participate in **at most one bonding pair** (one neighbor with same sign). The other two neighbors must be antibonding (opposite sign). This immediately forces a **wave-like alternation** of $+$ and $-$ phases across the lattice — exactly the Kekulé pattern.

On the honeycomb lattice, this constraint is almost satisfiable: assign $+1$ to all A-sublattice atoms and $-1$ to all B-sublattice atoms. Then every A–B bond is antibonding ($\rho_{ij} = -1$). This is the uniform state — no Kekulé alternation. To get Kekulé order, we need a more refined assignment where **one** of the three bonds around each atom is bonding (same sign) and two are antibonding. This requires a sublattice-breaking pattern, which is exactly the three Kekulé patterns.

#### Complex-Valued Amplitudes

If we allow complex $c_i \in \mathbb{C}$, the bond order becomes:

$$
\rho_{ij} = c_i^*\, c_j = |c_i|\,|c_j|\, e^{i(\phi_j - \phi_i)}$$

The phase difference $\Delta\phi_{ij} = \phi_j - \phi_i$ now controls the bond character continuously:

- $\Delta\phi = 0$: fully bonding ($\rho_{ij} > 0$, real)
- $\Delta\phi = \pi$: fully antibonding ($\rho_{ij} < 0$, real)
- $\Delta\phi = \pm\pi/2$: neutral ($\rho_{ij}$ purely imaginary — current-carrying bond)
- Intermediate $\Delta\phi$: partial bond with complex character

The complex phase $\phi_i$ at each site is the analogue of the Kekulé phase. A vortex in $\phi_i$ (phase winding by $2\pi$ around a loop) creates a point where the bond character rotates through all values — a Kekulé vortex.

#### The Pauli Constraint as a Projection

The key question: **can we enforce algorithmically that only one neighbor can be bonding (same phase) and the other two must be antibonding (opposite phase)?**

In the real-valued $\pm 1$ picture, this is a combinatorial constraint — it is the **dimer covering problem**: each atom is covered by exactly one dimer (bonding pair). Enforcing this during relaxation is NP-hard in general, but on bipartite planar graphs it is tractable.

In the complex-valued picture, the constraint softens to:

$$
\sum_{j \in N(i)} |\rho_{ij}|^2 \leq 1 \qquad \text{(local idempotency)}
$$

or equivalently, the local density matrix block $D_i$ (the $4 \times 4$ matrix of onsite and nearest-neighbor elements around atom $i$) should satisfy $D_i^2 \approx D_i$ (McWeeny purification).

This can be enforced via **Lagrange multipliers** in the variational energy:

$$
F[\psi] = \langle\psi|H|\psi\rangle + \sum_i \lambda_i \left(\sum_{j \in N(i)} |c_i^* c_j|^2 - 1\right)$$

or more practically, via a **projection step** after each update:

1. Evolve $c_i$ under the tight-binding Schrödinger equation
2. Compute local bond orders $\rho_{ij} = c_i^* c_j$
3. For each atom, check if $\sum_j |\rho_{ij}|^2 > 1$ (over-bonding)
4. If so, rescale the bond orders: $\rho_{ij} \to \rho_{ij} / \sqrt{\sum_j |\rho_{ij}|^2}$
5. Reconstruct $c_i$ from the projected bond orders

This is the **local Pauli projection** — the direct analogue of the pressure projection in fluid mechanics, applied to the quantum bonding constraint.

#### Connection to the Valence Projection in Model A

The valence projection in Model A (§9.9) enforces:

$$
\sum_{j \in N(i)} x_{ij} = \texttt{targetVal}_i$$

where $x_{ij} \in [0, 1]$ is the normalized bond order. In the single-wavefunction picture, $x_{ij} \sim |\rho_{ij}|^2 = |c_i|^2 |c_j|^2$. The valence constraint becomes:

$$
\sum_{j \in N(i)} |c_i|^2 |c_j|^2 = \texttt{targetVal}_i$$

If $|c_i|^2 = 1$ for all $i$ (normalized amplitudes), this reduces to $\sum_j |c_j|^2 = \texttt{targetVal}_i$, which is a local normalization constraint on the neighbors. This is a weaker condition than full idempotency but captures the essential physics: **each atom can contribute at most one unit of bond order to its neighborhood**.

#### What the Single-Wavefunction Model Can and Cannot Capture

**Can capture:**
- Bond-order alternation pattern (Kekulé structure)
- Phase topology (vortices, domain walls, nodal lines)
- Qualitative charge depletion at defect cores
- Current-carrying bonds (imaginary part of $\rho_{ij}$)

**Cannot capture:**
- Filled-shell stability (requires sum over all occupied MOs)
- Hückel $4n+2$ vs. $4n$ aromaticity (requires spectral information)
- Fractional charge at vortex cores (requires Dirac spectrum)
- Spin polarization (requires spin degree of freedom)
- True ground-state energy (rank-1 approximation overestimates kinetic energy)

The single-wavefunction model is best understood as a **mean-field approximation** to the bond-order field — it captures the topology but not the spectroscopy. It is the natural bridge between the purely phenomenological Model A (which has no wavefunction at all) and the full tight-binding Model B (which requires diagonalization).

#### Open Question: Complex Phases and the $\mathbb{Z}_3$ Locking

With complex $c_i$, the phase $\phi_i$ is a continuous $U(1)$ variable. But the honeycomb lattice only has three equivalent Kekulé patterns, so the physical energy should be invariant under $\phi \to \phi + 2\pi/3$. This $\mathbb{Z}_3$ locking must be imposed externally (via the $\lambda\, A^3 \cos(3\phi)$ term in the Ginzburg–Landau functional) — it does not emerge automatically from the single-wavefunction construction.

An interesting question is whether the $\mathbb{Z}_3$ locking can be derived from the lattice geometry itself, rather than imposed phenomenologically. In the real-valued $\pm 1$ picture, the three Kekulé patterns correspond to three distinct dimer coverings, and the lattice geometry naturally selects them. In the complex picture, this connection is less direct — the continuous phase allows intermediate patterns that are not true dimer coverings. The $\mathbb{Z}_3$ anisotropy term can be seen as the lattice's way of pulling the continuous phase back toward the three discrete dimer coverings.

---

### 13.3 Key Speculative Questions

The two physical pictures above raise two central questions that go beyond established theory and are proposed here as directions for further research.

#### Question 1: Can the Multi-Component Wavefunction Be Rephrased in Terms of Two Physical Fields?

**The two fields in question:**

1. **Lattice displacement field** $(u_x, u_y)$ — bosonic, single-component per atom, physically measurable, describes the static Peierls/Kekulé distortion. This field is the direct output of the electron-phonon coupling: the electrons lower their energy by distorting the lattice, and the distortion feeds back into the hopping integrals $t_{ij} = t_0 + \gamma\,(x_{ij} - x_0)$.

2. **Electron density field** — fermionic, encoded in the density matrix $\rho_{ij}$. The diagonal $\rho_{ii} = q_i$ gives Mulliken site charges; the off-diagonal $\rho_{ij} = q_{ij}$ gives Mulliken bond orders. Rigorously, this requires solving for single-particle molecular orbitals or Kohn-Sham states and summing their densities — an $O(N^3)$ diagonalization.

**The speculative proposal:** Can we avoid the diagonalization by "integrating out" the electronic degrees of freedom into an effective single-component wavefunction $|\psi\rangle = \sum_i c_i |i\rangle$ that generates the density matrix as $\rho_{ij} = c_i^* c_j$? This would give the density field physical meaning (Mulliken charges and bond orders) without solving the full electronic structure problem.

**The electron-phonon coupling as bridge:**

The two fields are coupled through the electron-phonon interaction:

$$
F[\mathbf{u}, \rho] = E_{\text{elastic}}[\mathbf{u}] + E_{\text{electronic}}[\rho, \mathbf{u}] + E_{\text{constraint}}[\rho]
$$

where:
- $E_{\text{elastic}}[\mathbf{u}] = \frac{K}{2}\sum_i |\mathbf{u}_i|^2$ — harmonic elastic cost of displacement
- $E_{\text{electronic}}[\rho, \mathbf{u}] = \text{Tr}(\rho\, H[\mathbf{u}])$ — electronic energy in the distorted lattice
- $E_{\text{constraint}}[\rho]$ — Pauli/idempotency constraint on the density matrix

The displacement field $\mathbf{u}$ modifies the hopping: $t_{ij}(\mathbf{u}) = t_0 + \gamma\,(u_i - u_j) \cdot \hat{e}_{ij}$. The density matrix $\rho$ responds to the modified Hamiltonian. In the full problem, one solves self-consistently: given $\mathbf{u}$, diagonalize $H[\mathbf{u}]$ to get $\rho$; given $\rho$, compute forces on $\mathbf{u}$ and relax.

**[Speculative]** The proposal is to replace this self-consistent loop with a single effective wavefunction $\psi$ that captures both fields simultaneously:

- The amplitude $|c_i|$ encodes the local site occupancy $q_i = |c_i|^2$
- The phase difference $\Delta\phi_{ij} = \phi_j - \phi_i$ encodes the bond order $q_{ij} = c_i^* c_j$
- The displacement field $\mathbf{u}$ is reconstructed from the bond orders via the Peierls relation $u_b \propto (x_b - x_0)$
- The complex order parameter $z_i = A_i e^{i\phi_i}$ is the envelope of the displacement pattern, which is also the threefold Fourier component of the bond orders

In this view, the multi-component wavefunction (whether the Madelung/Schrödinger-smoke $\psi = \sqrt{\rho}\, e^{i\theta}$ or the Dirac 4-spinor $\Psi$) is not a fundamental field but a **convenient parametrization** of the two physical fields $(\mathbf{u}, \rho)$ collapsed into a single complex object. The Schrödinger-smoke $\psi$ and the Dirac $\Psi$ are different choices of this parametrization, optimized for different regimes (continuum vs. lattice, low-energy vs. full-band).

**Why complex coefficients require time evolution:**

**[Established]** For a static ground state with real Hamiltonian, the molecular orbitals can always be chosen real, and the density matrix is real. Complex coefficients $c_i \in \mathbb{C}$ are only needed when:
- There is a magnetic field (breaks time-reversal symmetry)
- There are currents (nonzero imaginary part of $\rho_{ij}$)
- There is time evolution (the Schrödinger equation generates complex phases)
- There are degenerate states that mix with relative phases (as in the $K/K'$ valley mixing)

So for the static Kekulé ground state, real $c_i \in \{+1, -1\}$ suffice. Complex $c_i$ become meaningful when we want to describe **time evolution** of the wavefunction — e.g., a Dirac quasiparticle propagating through a Kekulé texture, or the Ginzburg–Landau relaxation dynamics of the order parameter. The complex phase $\phi_i$ in the static Kekulé order parameter $z_i = A_i e^{i\phi_i}$ is not a quantum-mechanical phase of a wavefunction — it is a **classical phase** selecting which of the three Kekulé patterns is realized. This is a subtle but important distinction: the $\phi$ in $z$ is a classical order-parameter angle, not a quantum phase.

**What remains open:**

- Whether the rank-1 approximation $\rho = |\psi\rangle\langle\psi|$ captures enough of the true density matrix to reproduce topological defect positions correctly
- Whether the displacement field reconstructed from bond orders via the Peierls relation is accurate enough to give meaningful divergence/curl diagnostics
- Whether the Schrödinger-smoke and Dirac parametrizations are truly equivalent descriptions of the same underlying $(\mathbf{u}, \rho)$ fields, or whether they capture different physics

---

#### Question 2: Can We Enforce Pauli Exclusion via Local Idempotency Without Diagonalization?

**The problem:** The rigorous way to enforce Pauli exclusion is to compute the density matrix from occupied molecular orbitals: $\rho = \sum_{n \in \text{occ}} |\psi_n\rangle\langle\psi_n|$. This requires diagonalization — $O(N^3)$ for dense methods. We want to avoid this.

**[Established]** In density-matrix functional theory, diagonalization is replaced by the **idempotency constraint** $\rho^2 = \rho$. This is a nonlinear constraint that can be enforced via:
- **McWeeny purification**: $\rho \to 3\rho^2 - 2\rho^3$ (iteratively drives $\rho$ toward idempotency)
- **Lagrange multipliers**: add $\sum_{ij} \Lambda_{ij}(\rho^2 - \rho)_{ij}$ to the energy functional
- **Penalty methods**: add $\mu\, \|\rho^2 - \rho\|^2$ to the energy

These methods are well-established in the density-matrix literature (LNV method, DFTB, etc.).

**[Speculative]** The key simplification in our case is that the tight-binding density matrix is **super-sparse**: only onsite ($\rho_{ii}$) and nearest-neighbor ($\rho_{ij}$) elements are nonzero. The idempotency condition $\rho^2 = \rho$ then becomes:

$$
\rho_{ij} = \sum_k \rho_{ik}\, \rho_{kj}$$

For nearest-neighbor $i,j$, the sum over $k$ involves only the common neighbors of $i$ and $j$ — typically 1–2 atoms on the honeycomb lattice. For onsite ($i=j$), it involves the 3 neighbors:

$$
\rho_{ii} = \rho_{ii}^2 + \sum_{k \in N(i)} |\rho_{ik}|^2$$

This is a **local** constraint — it involves only the density matrix elements within one bond length of atom $i$.

**Important clarification — what locality means here:**

We are **not** claiming that the density matrix is short-ranged. Graphene is a semimetal — it is close to a metal, and the states near the Fermi level have **long-range correlations**. These long-range correlations are not a nuisance to be truncated; they are **the entire point** of the model. The topological defects, the vortex textures, the domain walls, the cellular-automaton-like long-range order — these are the main output and motivation of the Kekulé fluid approach.

The locality claim is about **how the long-range order is generated**, not about whether it exists. The proposal is that long-range correlations in $\rho_{ij}$ **emerge spontaneously from purely local rules** — just as long-range spin correlations in the Ising model emerge from nearest-neighbor interactions, or just as long-range fluid vortices emerge from local Navier–Stokes dynamics. We never need to insert long-range order a priori; it builds up from the local idempotency constraint applied at every site.

This is fundamentally different from the standard electronic structure approach, which is wasteful in a specific sense: one starts with $N$ localized basis states (tight-binding orbitals, $N$ numbers), diagonalizes the Hamiltonian to obtain $N$ **delocalized** molecular orbitals (a coefficient matrix $C$ of $N^2$ numbers), then localizes them back into bonds/pockets via local interactions (electron-phonon coupling, Hubbard $U$, etc.) — or worse, forms Slater determinants and configuration-interaction matrices. The information travels: local → global (diagonalization) → local (localization). The Kekulé fluid approach proposes to skip the global detour entirely: enforce idempotency locally, and let the long-range order self-organize.

The local idempotency constraint prevents the density matrix from collapsing to a **bosonic** state (where all $c_i$ are identical and $\rho_{ij} = 1$ everywhere — no Pauli exclusion, no structure). By enforcing $\sum_j |\rho_{ij}|^2 \leq q_i(1-q_i)$ at each site, we force the wavefunction to develop the alternating bonding/antibonding pattern — and this local frustration, propagated across the lattice, generates the long-range Kekulé texture with its vortices and domain walls. The long-range order is an **emergent consequence** of local Pauli exclusion, not an input.

**The proposed local idempotency constraints:**

1. **Site occupancy**: $0 \leq q_i = \rho_{ii} \leq 1$ (at most one electron per site, spinless)
2. **Bond order sum rule**: $\sum_{j \in N(i)} |\rho_{ij}|^2 \leq q_i (1 - q_i)$ (from the onsite idempotency condition above)
3. **Bond order normalization**: $|\rho_{ij}| \leq \sqrt{q_i\, q_j}$ (Cauchy-Schwarz for rank-1 $\rho$)

In the single-wavefunction picture ($\rho_{ij} = c_i^* c_j$), these become:

1. $0 \leq |c_i|^2 \leq 1$ — amplitude bounded
2. $\sum_{j \in N(i)} |c_j|^2 \leq 1 - |c_i|^2$ — neighbors' occupancy limited by remaining capacity
3. $|c_i^* c_j| \leq |c_i|\, |c_j|$ — automatically satisfied

Constraint 2 is the key: it says that the total bond order around atom $i$ cannot exceed the available "slot" $1 - q_i$. This is the **local Pauli exclusion** — the direct analogue of the valence projection in Model A.

**Algorithmic proposal:**

```
1. Evolve c_i under effective Schrödinger / GL equation (O(N) per step)
2. Compute local bond orders: q_ij = c_i* c_j  (O(N), local)
3. For each atom, check: sum_j |q_ij|^2 <= q_i * (1 - q_i)
4. If violated, rescale: c_j -> c_j * sqrt(q_i*(1-q_i) / sum_j |q_ij|^2) for j in N(i)
5. Also clamp: |c_i|^2 -> min(|c_i|^2, 1)  (site occupancy)
6. Iterate steps 3-5 a few times (Jacobi-like, O(N) per iteration)
```

This is $O(N)$ per timestep with only local communication — exactly the same structure as the valence projection in Model A (§9.9), but now interpreted as a **local idempotency projection** rather than a heuristic valence constraint.

**What is established vs. speculative here:**

| Aspect | Status |
|---|---|
| Density matrix idempotency $\rho^2 = \rho$ encodes Pauli exclusion | **Established** |
| McWeeny purification and LNV methods enforce idempotency without diagonalization | **Established** |
| Kohn's nearsightedness: $\rho_{ij}$ decays exponentially in gapped systems | **Established** (but does NOT apply to graphene near Fermi level — see clarification above) |
| In NN tight-binding, idempotency reduces to local constraints on $\rho_{ii}$, $\rho_{ij}$ | **Established** (mathematically follows from sparsity of the TB basis) |
| Long-range correlations emerge from local idempotency rules (like Ising model) | **Speculative** (the central hypothesis: long-range order is output, not input) |
| A single wavefunction $\|\psi\rangle$ can approximate the density matrix as $\rho = \|\psi\rangle\langle\psi|$ | **Speculative** (rank-1 approximation; true $\rho$ is rank-$N_{\text{occ}}$) |
| Local Jacobi projection of bond orders converges to idempotent $\rho$ | **Speculative** (convergence not proven; may need under-relaxation) |
| This $O(N)$ projection captures the same topology as full diagonalization | **Speculative** (the central hypothesis of this work) |

**The connection to the valence projection in Model A:**

The valence projection in Model A (§9.9) enforces $\sum_j x_{ij} = \texttt{targetVal}_i$ with $x_{ij} \in [0,1]$. In the density-matrix picture, $x_{ij} \sim |\rho_{ij}|^2 = |c_i|^2 |c_j|^2$. The valence constraint becomes the idempotency constraint:

$$
\sum_{j \in N(i)} |c_i|^2 |c_j|^2 \leq q_i (1 - q_i)$$

So the valence projection in Model A is **already** a (heuristic, under-relaxed) local idempotency projection — it just wasn't derived from the density-matrix formalism. Recognizing this connection suggests:

- The projection could be improved by using the exact idempotency constraint $\sum_j |\rho_{ij}|^2 \leq q_i(1-q_i)$ instead of the simpler $\sum_j x_{ij} = \texttt{targetVal}$
- The site occupancy $q_i = |c_i|^2$ should be a dynamical variable, not fixed at 1
- The McWeeny purification $\rho \to 3\rho^2 - 2\rho^3$ could be applied locally to the sparse density matrix blocks

**Why this might work and why it might not:**

*Why it might work:*
- The local idempotency constraint is a **generative rule**: it does not assume short-range $\rho$, it produces long-range order as emergent output — analogous to how the Ising model generates long-range spin correlations from nearest-neighbor coupling
- The topological information (vortex positions, domain walls) is encoded in the **phase** of $\rho_{ij}$, which is determined by local phase differences $\Delta\phi_{ij}$ — a local quantity that propagates across the lattice
- The $O(N)$ cost enables real-time exploration of many defect configurations, which is the whole point of the approach
- The standard approach (diagonalize $\to$ delocalize $\to$ re-localize) is informationally wasteful: it transforms $N$ local numbers into $N^2$ global coefficients and back. The local approach keeps everything local throughout.
- The idempotency constraint prevents collapse to a trivial bosonic state ($c_i = \text{const}$) and forces the alternating pattern to self-organize

*Why it might not work:*
- The rank-1 approximation $\rho = |\psi\rangle\langle\psi|$ cannot capture multi-reference effects (e.g., open-shell diradicals where two configurations contribute equally)
- Graphene is a semimetal, not an insulator — the density matrix does **not** decay exponentially, so the local idempotency constraint may not be sufficient to reproduce the correct long-range correlations without additional nonlocal terms
- The local Jacobi projection may not converge, or may converge to a different fixed point than the true idempotent density matrix — convergence theory for local projections of rank-1 density matrices is not established
- Fractional charge and zero modes are spectral properties — they may not be captured by any local approximation, since they depend on the global topology of the Hamiltonian spectrum
- The emergence of long-range order from local rules is well-established for classical systems (Ising, Navier–Stokes) but **not** for quantum density matrices — the quantum case may require nonlocal information that cannot be generated by local rules alone

---

### 13.4 Synthesis: Two Pictures, One Order Parameter

Both pictures converge on the same mathematical object: a complex field on the honeycomb lattice whose amplitude measures dimerization strength and whose phase selects the Kekulé pattern.

| | Picture 1: Lattice displacement | Picture 2: Single-wavefunction DM |
|---|---|---|
| **Physical object** | Static optical phonon envelope | Rank-1 density matrix $\rho = \|\psi\rangle\langle\psi\|$ |
| **Two components** | $(u_x, u_y)$ displacement | $(\text{Re}\, c_i, \text{Im}\, c_i)$ amplitude |
| **Nodal lines** | Zero-displacement curves | Zero-amplitude curves |
| **Vortex** | Envelope phase winding | Wavefunction phase winding |
| **Ground state** | Regular phonon alternation | Regular $\pm$ sign alternation |
| **Defect** | Envelope irregularity | Phase frustration / over-bonding |
| **Projection** | Elastic constraint | Pauli / idempotency constraint |
| **$\mathbb{Z}_3$ locking** | From lattice symmetry | From dimer covering combinatorics |

The first picture grounds the **elastic/phonon** side: the Kekulé pattern is a frozen optical phonon, and the vortex is a topological defect in its envelope. The second picture grounds the **electronic/bonding** side: the Kekulé pattern is a wave-like sign alternation, and the Pauli constraint enforces the dimer covering. Together, they give the abstract complex order parameter $z_i$ two complementary physical interpretations.

The two speculative questions in §13.3 ask whether these pictures can be made rigorous enough to replace full electronic structure calculations: (1) whether the multi-component wavefunction is a valid reparametrization of the two physical fields $(\mathbf{u}, \rho)$, and (2) whether local idempotency projection can enforce Pauli exclusion in $O(N)$ time. Both questions are open and represent the central hypotheses of this work.

---

## 14. Code Review and Physics Audit

### 14.1 Critical Bugs

**1. OpenCL struct packing mismatch** (`ModelA.py:56-63`, `DiracLattice_ocl.py:41-46`)

The `Params` struct in OpenCL contains `int` fields (`nProj`, `natom`, `nbond`), but Python packs all values as `np.float32`. The integer fields receive float bit patterns, not integer values. Currently does not break because these fields are unused in kernels, but it is a latent bug.

*Fix*: Remove unused `int` fields from both structs, or use a structured NumPy dtype.

**2. Model A RK4 does GPU→CPU→GPU round-trip** (`ModelA.py:187-193`)

`evolveZ_RK4()` copies $z$ into host NumPy arrays and back to device temp buffers every step. Should be direct device-to-device copy:

```python
cl.enqueue_copy(queue, b_z_temp_r, b_z_real)
cl.enqueue_copy(queue, b_z_temp_i, b_z_imag)
```

The Dirac solver already does this correctly.

**3. `plot_phase_hsv` crashes when `show_nodal=False`** (`visualize.py:370`)

`re_segs` and `im_segs` are never defined if `show_nodal=False`, causing `NameError`.

*Fix*: Initialize `re_segs = []` and `im_segs = []` before the `if show_nodal:` block.

### 14.2 Physics Inconsistencies

**1. Proton defect is both suppressed and hard-pinned** (CRITICAL)

`applyPinsToZ` first suppresses $z$ by $(1 - \texttt{defect})$, then hard-pins it toward $A_{\text{pin}}\, e^{i\phi}$. For `defect=1`, `pinStrength=1`: $z \to 0$ then immediately $z \to A_{\text{pin}}\, e^{i\phi}$. This contradicts the physical meaning that a protonated atom has no π orbital.

*Fix*: Pin the **neighboring** atoms' phase, not the removed atom. Or use phase-only pinning without amplitude imposition. Or remove the protonated atom from the π graph entirely.

**2. Defect onsite potential $V_{\text{def}} = 2.0$ is too small** (HIGH)

With $t_0 = 1$, graphene bandwidth is $\sim 6t_0 = 6$. So $V_{\text{def}} = 2$ is inside the band — the defect state hybridizes rather than being removed. The code comments claiming $V_{\text{def}} > \text{bandwidth}$ are wrong.

*Fix*: Use $V_{\text{def}} = 20$ or higher, or physically remove the atom from the Hamiltonian (set hoppings to/from defect site to zero).

**3. $z$ amplitude scale inconsistency** (HIGH)

Ginzburg–Landau parameters ($r = -1$, $u = 1$) want $|z| \approx 1$, but `zToRawBonds` with `amp = 0.8 * min(x0, 1-x0)` cannot generate bond patterns that reconstruct to $|z| \approx 1$ after projection. Observed $|z|$ values are 0.2–0.5.

*Fix*: Calibrate the amplitude. Either adjust $r, u, A_{\text{pin}}$ to match the reconstructed scale, or increase the bond modulation amplitude. Add a calibration test.

**4. Pinned atoms drift during RK4** (MEDIUM)

`applyPinsToZ` is applied before `evolveZ_RK4` but not re-applied after. The soft pinning in `rhsZ` pulls $z$ toward the pin, but after RK4 combination the final $z$ may have drifted.

*Fix*: Re-apply `applyPinsToZ` after `evolveZ_RK4`, or at least after the final RK4 combination.

### 14.3 Architecture Issues

**1. Separate OpenCL contexts** (MEDIUM)

`ModelA` auto-selects NVIDIA; `DiracLatticeSolver` calls `cl.create_some_context()`. They should share a context and queue to avoid context-switching overhead and allow direct buffer sharing.

**2. Defect scaling API is inconsistent** (MEDIUM)

`DiracLattice.cl` says the defect array is already scaled by $V_{\text{def}}$, but dense diagonalization multiplies by $V_{\text{def}} = 2$ internally. The OpenCL time-propagation path and NumPy diagonalization path can use different defect scalings.

**3. `compute_local_density_of_states` is a stub** (LOW)

Dead code — just `pass`. Should be implemented or removed.

### 14.4 Visualization Issues

**1. Voronoi interpolation looks blocky** (MEDIUM)

HSV phase background uses nearest-atom assignment, creating blocky Voronoi patches. Gaussian-weighted interpolation or barycentric interpolation would be smoother and more physically meaningful.

**2. No color wheel / hue legend** (MEDIUM)

The HSV rainbow phase plot has no legend explaining which color corresponds to which phase angle. A small color wheel inset would make it self-explanatory.

**3. "Nodal lines" are component-zero lines, not physical domain walls** (HIGH)

`Re(z) = 0` and `Im(z) = 0` lines depend on global phase convention. A `Re(z) = 0` line can move if all phases are shifted by a constant. Better physical overlays:
- $|z| = \text{threshold}$ contour (low-amplitude band)
- Phase-sector boundaries between nearest $\mathbb{Z}_3$ minima
- $\cos(3\phi)$ or $\arg(z) \bmod 2\pi/3$
- Domain-wall energy density: $\sum |z_i - z_j|^2$

**4. Nodal line detection misses zero-amplitude regions** (MEDIUM)

Sign-change detection across bonds only finds where the real/imaginary component flips sign. It does not show regions where $|z|$ is small but nonzero — the broad domain-wall "valley."

### 14.5 Didactic Improvements

1. **Color wheel legend** on HSV phase plots
2. **Discrete $\mathbb{Z}_3$ label** showing which of the 3 Kekulé patterns each region belongs to
3. **Animate the relaxation** to show nodal lines forming from uniform initial state
4. **Kekulé bond pattern overlay**: draw thick vs thin lines for double/single bonds on top of HSV phase
5. **Dirac mass gap $|\Delta(\mathbf{r})|$** as a separate panel
6. **Controlled comparison plots**: clean PAH vs. same-pin defects vs. mismatched-pin defects vs. Kekulé distortion only
7. **Defect connecting line**: dashed line between defect positions to show expected nodal line direction

---

## 15. Suggested Roadmap

### 15.1 Immediate Fixes

1. Fix struct packing: remove unused `int` fields or use structured dtype
2. Fix Model A RK4 host copies: device-to-device copy
3. Fix defect model: do not hard-pin finite $z$ on a deleted/protonated atom
4. Fix Dirac defect strength: $V_{\text{def}} \gg$ bandwidth, or remove site from Hamiltonian
5. Fix context selection: shared NVIDIA context helper
6. Fix `NameError` in `plot_phase_hsv` when `show_nodal=False`

### 15.2 Physics Cleanup

1. Calibrate $z$ amplitude: make Ginzburg–Landau $|z|$ scale consistent with reconstructed bond-order $|z|$
2. Separate phase pins from proton cores: pin neighbors or bonds, not the removed atom
3. Add energy and residual diagnostics: $F_{\text{GL}}[z]$, valence residual, projection residual, domain-wall length
4. Expose projection relaxation parameter $\omega_{\text{proj}}$

### 15.3 Visualization Improvements

1. Replace "nodal line" wording: distinguish $\text{Re}(z) = 0$, $\text{Im}(z) = 0$, low-$|z|$, and $\mathbb{Z}_3$ domain wall
2. Add smooth interpolation + molecule mask
3. Add phase wheel and $\mathbb{Z}_3$ sector legend
4. Add comparison panels for same-pin vs. mismatched-pin defects
5. Add Kekulé bond pattern overlay (thick/thin bonds)

### 15.4 Testing

1. PAH topology: atom/ring/degree counts for `n_shells = 0..4`
2. Mirror symmetry: every atom has mirror partners after `build_pah()`
3. Projection convergence: max valence residual after `projectBondOrders()`
4. No-race regression: compare GPU projection to CPU serial projection
5. Pinned defect consistency: test whether defect $|z|$ matches intended model
6. Parameter sweeps: $\lambda$, $\kappa$, $k_{\text{core}}$, `pin_strength`, $A_{\text{pin}}$, `nProj`

### 15.5 The Two-Layer Architecture

The recommended long-term design is:

$$
\boxed{\text{Layer 1: fast dimer smoke solver (Model A)}}
$$

$$
\boxed{\text{Layer 2: occasional tight-binding/Dirac validation (Model B)}}
$$

Explore many edge functionalizations, radical pairs, protonated sites, or PAH geometries very cheaply with Model A, then use the electronic solver only on interesting cases. The hydrodynamic model moves the vortex; the electronic model checks whether the vortex core carries fractional charge, zero-mode occupancy, or spin.

---

## 16. References

1. **Schrödinger's smoke** — Caltech AUTHORS: https://authors.library.caltech.edu/records/829zt-g0315

2. **Blaschke product** — Wikipedia: https://en.wikipedia.org/wiki/Blaschke_product

3. **Height field theory for 2D dimer models** — arXiv: https://arxiv.org/html/2606.17154v1

4. **Electron fractionalization in 2D graphenelike structures** (Hou, Chamon, Mudry) — arXiv: https://arxiv.org/abs/cond-mat/0609740

5. **Fractional charge bound to a vortex in 2D topological crystalline insulators** — arXiv: https://arxiv.org/abs/1903.02737

6. **Simple and Fast Fluids** (Guay, Colin, Egli) — Inria HAL: https://inria.hal.science/inria-00596050/document

7. **Observation of Kekulé vortices around hydrogen adatoms in graphene** — Nature Communications: https://www.nature.com/articles/s41467-024-47267-8

---

*This document is a rewritten, systematic synthesis of a multi-turn conversation between the author and several AI assistants (ChatGPT 5.5, Gemini 3.1 Pro, GLM 5.2). The original conversation explored the analogy between Schrödinger smoke fluid simulation and Kekulé topological defects in graphene, progressively refining the physics into a concrete implementable specification. This monograph extracts the key ideas in didactical order: motivation → physics → mathematics → algorithm → implementation → review.*
