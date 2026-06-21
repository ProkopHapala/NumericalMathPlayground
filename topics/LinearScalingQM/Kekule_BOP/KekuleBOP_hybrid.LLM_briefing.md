# Kekulé-BOP hybrid: my research notes and questions

I am building a fast, linear-scaling, GPU-friendly solver for bond orders in conjugated π-electron systems (graphene, nanoribbons, doped graphene, quinone / phenazine / pyrazine redox aromatics). I want to avoid any `O(N^3)` diagonalization — not Hückel, not full CI. This note summarizes the method I have, the insights that surprised me, and what I still want to understand.

---

## 1. The current method

### Energy functional (currently implemented)

I currently minimize a simple energy, not a free energy:

\[
E[p] = E_{\text{val}}[p] + E_{\text{aro}}[p] + E_{\text{loc}}[p]
\]

with

- **valence penalty** `E_val = ½ Σ_i (Σ_j p_ij − n_i)^2` — enforces the constraint,
- **aromatic term** `E_aro = c2 Σ_{ij} (p_ij − ½)^2` — favors equal bond sharing in aromatic rings,
- **localization snap** `E_loc = c3 Σ_{ij} (p_ij − round(p_ij))^2` — favors integer single/double bonds.

The constraint `Σ_j p_ij = n_i` and the bounds `0 ≤ p_ij ≤ 1` are enforced separately. There is no explicit temperature and no entropy term in the current code.

### Variables

- A graph of sp2 atoms and bonds (e.g. hexagonal lattice, graphene, PAH). One π orbital per atom.
- On each edge `(i,j)` a classical bond order `p_ij ∈ [0,1]`.
- On each atom `i` a local π-electron count `n_i` (`1` for neutral carbon, variable for defects or charge).

\[
\sum_{j \sim i} p_{ij} = n_i .
\]

This says the total bond order around atom `i` equals the number of π electrons that belong to that atom.

### Pseudocode

```
initialize p_ij on the graph edges
repeat:
    compute valence error v_i = Σ_j p_ij - n_i
    compute energy gradient from E_val, E_aro, E_loc
    update p_ij by gradient descent
    project p_ij back into [0,1]
    update n_i from charging model if needed
until convergence
```

### The quantum side: bond-order potential (BOP)

Instead of diagonalizing the Hamiltonian, I compute the density-matrix element `ρ_ij` for a bond by a local expansion of the Fermi operator:

- **Chebyshev expansion** of the step function `θ(ε_F − H)`.
- **Lanczos / continued fraction** (Haydock recursion) for the local Green's function.

Both are `O(N)` for a fixed recursion depth and give `ρ_ij` for one bond without eigenstates. The result is compared to the classical `p_ij` and to Wiberg / Mayer bond orders.

### Relation to Pauling / Hückel bond order

- **Pauling bond order**: count the number of Kekulé structures in which a given bond is double, divided by the total number of structures. My `p_ij` is the continuous version of this count.
- **Hückel bond order**: from the off-diagonal density matrix `ρ_ij` after diagonalizing the Hückel Hamiltonian. My quantum BOP side computes `ρ_ij` without diagonalization.
- The classical optimizer gives a fractional Pauling-like average; the BOP gives a quantum `ρ_ij`. I want to combine them.

### Status: implemented vs planned

| Component | Status |
|---|---|
| Classical Kekulé optimizer with valence, aromatic, and localization terms | Implemented |
| Quantum BOP (Chebyshev / Lanczos) for local `ρ_ij` | Implemented |
| 1D orbital minimization solver (OMM) | Implemented |
| Resonance entropy `S_i` in the energy functional | Considered / not yet implemented |
| Variable `n_i` via Hubbard U / QEq self-consistency | Considered / not yet implemented |
| Kasteleyn dimer counting for entropy benchmark | Considered |
| GPU local-update Monte Carlo over dimer coverings | Planned |
| RVB amplitude reweighting of coverings | Planned |

---

## 2. Possible extension: a free-energy formulation

A natural next step would be to add a resonance entropy term and write

\[
F[p,n] = U[p,n] - T \, S_{\text{res}}[p],
\]

with `S_i = n_i − Σ_j p_ij^2`. This would make the classical optimizer look like the mean-field limit of the Quantum Dimer Model. **This is a proposal for discussion, not something already implemented.** Open questions: is `S_i` the right entropy functional? What is the physical meaning of `T`? How do I calibrate it against Kasteleyn or RVB?

---

## 3. Insights that changed my understanding

### 3.1 Resonance structures in VB are the off-diagonal kinetic energy

This is the main insight I missed in standard quantum chemistry. The Hückel / Hubbard kinetic term

\[
H_t = -t \sum_{\langle i,j \rangle} (c_i^\dagger c_j + \text{h.c.})
\]

moves an electron from one atom to a neighbor. When it acts on a valence-bond (Kekulé) state, it does **not** stay inside the same structure; it turns one Kekulé structure into another. The kinetic energy lowering of the ground state is therefore an **off-diagonal resonance integral between Kekulé structures**, not a sum of isolated dimer orbital energies. A single Kekulé structure has high kinetic energy; the real energy lowering comes from the superposition of many structures.

**Question:** Can you make this more rigorous — what is the exact matrix element between two VB states and how does it map to the Hückel / Hubbard `t`?

### 3.2 The valence sum rule is a linearized Pauli / idempotency constraint

For a single Slater determinant, the one-particle density matrix `ρ` is idempotent and its row sum satisfies `Σ_j ρ_ij = n_i`. The classical constraint `Σ_j p_ij = n_i` is a linearization of this condition when `p_ij ≈ ρ_ij`. It is an electron-counting rule, not a current law. The caveat is that idempotency is quadratic (`ρ^2 = ρ`) while the sum rule is linear, so the classical model is a relaxation.

**Question:** Under what exact conditions is the linearized sum rule a valid approximation to idempotency, and how should I handle `n_i ≠ 1`?

### 3.3 Resonance entropy is a classical proxy for kinetic energy

The local entropy

\[
S_i = n_i - \sum_{j \sim i} p_{ij}^2
\]

is zero for a single Kekulé structure (integer bonds) and positive when bonds are fractional. In benzene, where all bonds are `0.5`, it measures the local number of equally weighted Kekulé structures. I think of it as a cheap classical substitute for the quantum resonance / kinetic energy that makes the system aromatic.

**Question:** Can you derive this entropy from the distribution of Kekulé structures counted by Kasteleyn, or from RVB entanglement entropy? Is the formula `S_i = n_i − Σ_j p_ij^2` the right one, or should it be replaced?

### 3.4 SSH: the localization snap is a bond-length double well

The **Su–Schrieffer–Heeger (SSH)** model couples electron hopping to lattice dimerization. In a long chain, the system wants to distort so that single and double bonds alternate. The `E_loc` term in my functional is the classical analog: a double well that favors integer bonds. In aromatic rings, the resonance entropy fights this term and keeps bonds equal.

**Question:** Is this competition exactly the SSH / Peierls transition, or only an analogy? Can I use the SSH electron-phonon coupling to set `c3` in my functional?

### 3.5 Hubbard U: variable `n_i` and charging

The **Hubbard model** adds an on-site energy `U n_i↑ n_i↓` that penalizes double occupancy. For me, this means `n_i` can deviate from 1: a positively charged site has fewer π electrons, a negatively charged site has more. Combined with QEq, the model can self-consistently determine `n_i` from electrostatics.

### 3.6 Kasteleyn: exact counting of Kekulé structures by a determinant

For a planar graph, the number of perfect matchings (Kekulé structures) is

\[
N_{\text{Kekulé}} = \sqrt{|\det(K)|}
\]

where `K` is a skew-symmetric signed adjacency matrix. With edge weights `w_ij`, the Pfaffian gives the partition function `Z`. Local bond probabilities are `p_ij = K_ij (K^{-1})_ji`. This is a **determinant of a sparse matrix**, not a Hamiltonian diagonalization. It gives me an exact benchmark for the classical entropy and the bond probabilities in small planar molecules.

**Question:** How does this O(N^3) scaling relate to normal diagonalization of a Hamiltonian? Is it just a coincidence, or does the Pfaffian of the bond adjacency matrix encode some information about the Hamiltonian spectrum?

### 3.7 G0 / Landauer-Büttiker channel-counting analogy

In mesoscopic transport, the **conductance quantum `G0 = 2e^2/h`** corresponds to one fully transmitting orbital channel. Because of the Pauli exclusion principle, each orbital can carry at most one electron per spin. The valence sum rule `Σ_j p_ij = n_i` can be viewed as a channel-counting rule on the bond graph: the bond orders measure how many of the atom’s available channels are occupied. I use this as an intuitive picture, but I want to know whether it can be made rigorous.

**Question:** Can the channel-counting analogy be turned into a rigorous counting argument for the sum rule, or is it only a heuristic?

---

## 4. The quantum correction: dimer coverings and the Quantum Dimer Model

### 4.1 Dimer covering

A **Kekulé structure** is a **perfect matching** of the graph: each atom is incident to exactly one occupied bond. A set of all such matchings is the Valence Bond (VB) basis.

### 4.2 Resonating Valence Bond (RVB) state

The exact ground state is a superposition

\[
|\Psi\rangle = \sum_{C} a_C \, |C\rangle
\]

where `|C⟩` is one matching and `a_C` is its amplitude. The classical optimizer averages probabilities, not amplitudes.

### 4.3 Quantum Dimer Model (QDM)

The QDM is a Hamiltonian that acts directly on dimer coverings, without molecular orbitals:

\[
H = \sum_e V_e \, n_e \; - \; t \sum_{\text{plaq}} \big( |\text{after}\rangle\langle\text{before}| + \text{h.c.} \big) .
\]

- `V_e` is a classical energy assigned to an occupied bond.
- A **plaquette** is a small face of the lattice, e.g. a square or a hexagon.
- A plaquette is **flippable** if two opposite edges of that face are occupied by dimers. The **flip** replaces those two parallel dimers by the other two parallel edges of the same face, keeping every atom singly matched.

**Question:** Can the QDM be derived from the Hubbard or PPP Hamiltonian in a controlled way, and how do I choose `t` and `V_e` without diagonalizing the Hamiltonian?

### 4.4 Kinetic energy without molecular orbitals

Because the kinetic operator connects two coverings by moving one bond locally, the local kinetic energy contribution in a sampled covering `C` can be estimated as

\[
E_{\text{kin}}(C) \approx -t \, N_{\text{flip}}(C)
\]

where `N_{\text{flip}}(C)` is the number of flippable plaquettes in the current covering. This is why the QDM does not need MO diagonalization.

**Question:** Is `-t N_flip` a valid expectation value, or does it need to be corrected by the non-orthogonality of VB states and by the overlap between coverings?

### 4.5 Proposed GPU algorithm

I want to build a local-update Monte Carlo sampler over dimer coverings:

1. Start from the current optimized `p_ij` and compute bond energies `V_ij`.
2. Assign Boltzmann weights `w_ij = exp(-β V_ij)`.
3. For small planar molecules, use Kasteleyn to get exact probabilities.
4. For large or non-planar systems, use local-update MC.
5. Each GPU thread proposes a local flip, computes the local energy change, and accepts/rejects by Metropolis.

Pseudocode for one GPU thread:

```
pick a random plaquette
if the two opposite edges are occupied by dimers:
    flip them to the complementary pair
    ΔE = sum of local bond-energy changes
    accept with probability min(1, exp(-β ΔE))
```

This is the same parallel structure as the GPU Ising / Coulomb samplers I already have.

**Question:** For hexagonal graphene, what is the complete set of allowed local moves that preserve the hard-core dimer constraint, and how do I avoid conflicts when running many GPU threads in parallel?

---

## 5. What I still want to understand and investigate

1. Can `S_i = n_i − Σ_j p_ij^2` be derived from the number of Kekulé structures counted by Kasteleyn, or from RVB entanglement entropy?
2. How do I choose the QDM parameters `t` and `V_e` without diagonalizing the Hückel or Hubbard Hamiltonian?
3. How do I allow `n_i ≠ 1` in a physically consistent way: is `n_i` an occupancy, a Hilbert-space capacity, or both, and how do Hubbard `U` and QEq enter?
4. What is the cleanest way to map local dimer flips to a GPU-efficient QMC / Ising-like sampler, especially for hexagonal (graphene) lattices?
5. How does my classical energy minimizer relate to Kasteleyn, iterative b-matching, and min-cost-flow algorithms? Is there a more principled way to solve it?

---

## 6. What I would like from you

- Give me rigorous, literature-backed definitions for the quantities and equations above.
- Show the connections between my classical model and the established theories (VB, RVB, QDM, Kasteleyn, Hubbard, SSH, conductance quantization).
- Provide concrete references (DOI, arXiv, or standard book chapters).
- Answer my questions in section 5.
- Propose a practical, GPU-friendly dimer-covering MC algorithm that uses the current optimized `p_ij` as a seed and computes local energy changes without diagonalization.
- Point me to recent work (reviews, preprints, or key papers) that combine classical Kekulé models, bond-order potentials, quantum dimer models, or GPU dimer Monte Carlo.
