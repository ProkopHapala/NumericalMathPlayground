# Kekulé–BOP Hybrid Theory

A working document that connects the local quantum **Bond-Order Potential (BOP)** solver with the global **classical Kekulé bond-order optimizer**, and places both on a rigorous quantum-statistical footing.

---

## 1. The two computational pieces in this project

### 1.1 Quantum BOP / Fermi-Operator Expansion

The BOP code starts from a π-only tight-binding Hamiltonian

```@/home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleOrderN_Gemini_BOP_2D_v2.py:39-59
t0 = -2.7        # Base hopping (eV)
r0 = 1.42        # Equilibrium distance (Angstrom)
beta = 3.0       # Exponential decay factor (1/Angstrom)
...
if dist < 1.8:
    t = t0 * np.exp(-beta * (dist - r0))
    H[i, j] = H[j, i] = t
```

and estimates the zero-temperature one-particle density matrix

\[
\hat\rho = \Theta(\mu\hat I - \hat H)
\]

without full diagonalization. Two expansion techniques are used:

- **Chebyshev / Fermi-Operator Expansion (FOE)**

  ```@/home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleOrderN_Gemini_BOP_2D_v2.py:91-122
  def get_bond_density_cheb(H_scaled, start_site, target_site, M):
      ...
      v_next = H_scaled @ v_curr
      current_density += c[1] * v_next[target_site] * 2.0
      for k in range(2, M):
          v_next_next = 2.0 * (H_scaled @ v_next) - v_curr
          if k % 2 != 0:
              current_density += c[k] * v_next_next[target_site] * 2.0
  ```

- **Lanczos / continued fraction (Haydock recursion)**

  ```@/home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleOrderN_Gemini_BOP_2D_v2.py:169-209
  def get_bond_density_CF(H, i, j, M, eta=0.2, ...):
      ...
      rho_plus = -(1.0 / np.pi) * np.imag(G_plus) * dE
      ...
      return 2.0 * 0.5 * (rho_plus - rho_minus)
  ```

Output: an approximation to the off-diagonal density-matrix element `ρ_ij` for a **non-interacting** or weakly interacting π system.

- **Captures**: local Pauli/interference physics, Fermi-Dirac filling, Peierls distortion, short-range bond-density oscillations.
- **Misses**: strong electron correlations, global topological/valence constraints, long-range propagation of defects (the expansion is truncated after `M` scatterings).

### 1.2 Classical Kekulé bond-order optimizer (`KekulePure`)

`KekuleBondSum.py` optimizes a continuous variable `p_ij ∈ [0,1]` on each graph edge. The core constraint is a weighted atom valence:

```@/home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:104-140
def project_valence(self):
    val = np.zeros(self.natom, dtype=float)
    np.add.at(val, self.i0, self.bo)
    np.add.at(val, self.i1, self.bo)
    return val
```

The energy contains three phenomenological terms:

```@/home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:142-165
# aromatic parabola centered at 0.5
d_aro = bo - 0.5
f_bond = -2.0 * self.Karo * d_aro
# localization snap parabolas around 0 / 0.5 / 1
target = self._localization_target(bo)
```

- **Valence penalty**: `Σ_i (Σ_j p_ij - n_i)^2`
- **Aromatic term**: parabola centered at `p = 0.5`
- **Localization snap**: pushes bonds toward 0, 0.5, or 1

- **Captures**: global valence conservation, resonance averaging, topological-defect propagation, large-scale bond-order waves.
- **Misses**: local quantum interference, true density matrix, charge polarization.

---

## 2. Quantum meaning of the classical bond order

### 2.1 One-particle density matrix, pure vs mixed

For a single Slater determinant `|D⟩`, the one-particle density matrix

\[
\rho_{ij} = \langle D | c_i^\dagger c_j | D \rangle
\]

is a **projector**:

\[
\rho^2 = \rho .
\]

A CI / resonance wavefunction `|Ψ⟩ = Σ_m C_m |K_m⟩` has

\[
\rho_{ij} = \sum_m |C_m|^2 \rho_{ij}^{(m)} + \text{cross terms}.
\]

If the Kekulé determinants are orthogonal and the phases are effectively random, the cross terms vanish and `ρ` is the **average of projectors** — a **mixed-state density matrix**. This is what the aromatic `p = 0.5` pattern represents.

### 2.2 The Kekulé sum rule as a linear relaxation of idempotency

Exact idempotency implies, for each atom `i`,

\[
\sum_j |\rho_{ij}|^2 = \rho_{ii} = n_i .
\]

The classical Kekulé rule is a **linear** relaxation:

\[
\sum_j p_{ij} = n_i .
\]

For a **binary** Kekulé structure `p_{ij} ∈ {0,1}` the two are equivalent because `p^2 = p`. For an **aromatic average** they differ:

\[
\sum_j p_{ij} = 1,
\qquad
\sum_j p_{ij}^2 = 2 \times (0.5)^2 = 0.5 \neq 1.
\]

So the aromatic 0.5 pattern is **not** a single idempotent density matrix; it is an **ensemble average** of binary Kekulé determinants.

### 2.3 Atoms vs bonds

In benzene:

- Each atom has **two** bonds.
- Aromatic average: each bond has `p = 0.5`.
- Atom sum: `2 × 0.5 = 1`.

The **atom** sum stays at `n_i`; only the **bonds** become fractional. The `KekulePure` valence term enforces the atom sum, while the aromatic/localization terms control the bond values.

### 2.4 Valence-bond / RVB / dimer-model interpretation

In the **strong-coupling (large Hubbard U)** limit, the relevant Hilbert space is spanned by **singlet bonds** (dimers) between singly occupied p_z sites. Each Kekulé structure is a **dimer covering** of the graph. The classical variable `p_ij` is the **probability / amplitude of a singlet bond on edge (i,j)**.

The quantum **resonating valence bond (RVB)** / **quantum dimer model** is the rigorous Hamiltonian version. The Kekulé sum rule

\[
\sum_j p_{ij} = 1
\]

is the **hard-core dimer constraint**: each singly occupied atom must belong to exactly one dimer.

### 2.5 Pauli exclusion, Fermi-Dirac, and the bound `p_ij ≤ 1`

The Pauli principle says at most one electron of each spin can occupy a given p_z orbital. In the large-U limit, double occupancy is suppressed, so the local Hilbert space reduces to empty / singly occupied. The bound

\[
0 \le p_{ij} \le 1
\]

is the classical projection of this capacity: a given π bond can be at most a full double bond.

---

## 3. Currents, Kirchhoff, and probability flow

### 3.1 Quantum probability current per orbital

For each molecular orbital `α`, the bond current

\[
J_{ij}^{(\alpha)} \propto \operatorname{Im}\left( \psi_i^{(\alpha)*} H_{ij} \psi_j^{(\alpha)} \right)
\]

satisfies Kirchhoff’s law at every node:

\[
\sum_{j\sim i} J_{ij}^{(\alpha)} = 0 .
\]

This is probability conservation, not charge conservation. In a real, time-reversal-symmetric system the eigenfunctions can be chosen real, so every stationary orbital carries **zero net current**. Clockwise and counter-clockwise ring currents cancel.

### 3.2 Why the Kekulé sum rule is not a Kirchhoff law

The Kekulé constraint is defined on an **undirected** graph:

```@/home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:96-110
self._A = np.zeros((self.natom, self.natom), dtype=float)
...
np.add.at(self._A, (self.i0, ib), 1.0)
np.add.at(self._A, (self.i1, ib), 1.0)
```

Both endpoints receive `+1`. A true Kirchhoff law requires an **oriented** incidence matrix with `+1 / -1`. The Kekulé rule is therefore a **weighted-degree / resource-allocation** constraint, not a flow-conservation law.

### 3.3 Probability-channel analogy

The useful intuition is a **probability channel**, not an electrical current:

- Atom `i` has `n_i` electrons.
- Each electron chooses one neighboring bond.
- The probability of choosing bond `(i,j)` is `p_{ij}/n_i`.
- The Kekulé sum rule is the normalization of this probability distribution.

So the “conductance channel” picture is valid as an **ensemble-statistical** statement: each singly occupied orbital contributes to at most one bond channel, and the total probability over all channels is 1.

---

## 4. Topology, SSH model, and Peierls distortion

### 4.1 SSH / bond-order wave

The Su–Schrieffer–Heeger (SSH) model shows that a one-dimensional π chain is unstable to dimerization. A topological defect is a **soliton** that separates two dimerization phases. The bond order alternates between `≈1` and `≈0`.

`KekulePure` with its localization snap term is a **discrete, graph-based analogue of the SSH order parameter**: it pushes the system into a bond-order wave (BOW) and naturally creates soliton-like defects at sp³ sites or at heteroatom substitutions.

### 4.2 Topological defects in graphene nanoribbons

Hydrogenating one carbon removes its p_z orbital. In the graph model this is a site with `n_i = 0` (or removed). The global constraint `A p = n` then forces a reorganization of double bonds across the whole flake. This is why the classical solver can propagate topological effects over long distances, while a low-order BOP expansion only sees a few scattering steps.

---

## 5. Entropy, free energy, and aromaticity

### 5.1 Resonance entropy

For a binary Kekulé microstate `p_{ij}^{(m)} ∈ {0,1}`, we have `p^2 = p`. Averaging over the ensemble gives:

\[
\left\langle \sum_j (p_{ij}^{(m)})^2 \right\rangle
=
\left\langle \sum_j p_{ij}^{(m)} \right\rangle
= n_i .
\]

For the **averaged** variables, however:

\[
\sum_j p_{ij}^2 \le n_i .
\]

The difference

\[
S_i = n_i - \sum_j p_{ij}^2
\]

is a **local resonance-entropy / mixedness measure**. It is zero for a single Kekulé structure and maximal for a uniform aromatic pattern.

Example: benzene aromatic state

- `n_i = 1`
- `p_ij = 0.5` on two bonds
- `Σ_j p_ij^2 = 2 × 0.25 = 0.5`
- `S_i = 0.5`

### 5.2 Free-energy functional

A rigorous effective model can be written as a constrained free-energy minimization:

Variables: `p_ij ∈ [0,1]`, `n_i ∈ [0,2]`.

Constraints:

\[
\sum_j p_{ij} = n_i,
\qquad
\sum_i n_i = N_\pi,
\qquad
p_{ij} \le 1.
\]

Energy:

\[
U[p,n] = \sum_{(ij)} \left[ -t_{ij}\, p_{ij} + V_{\text{loc}}(p_{ij}) \right]
+ \frac{\kappa}{2}\sum_i \left(\sum_j p_{ij} - n_i\right)^2
+ E_{\text{QEq}}(n)
+ E_U(n) .
\]

- `t_ij`: hopping / BOP prior.
- `V_loc(p)`: localization snap.
- `E_QEq`: electrostatic charge energy.
- `E_U(n) = (U/2) Σ_i n_i(n_i - 1)`: Hubbard penalty for double occupancy.

Resonance entropy:

\[
S_{\text{res}}[p] = \sum_i \left( n_i - \sum_j p_{ij}^2 \right)
\]

Free energy:

\[
F[p,n] = U[p,n] - T\, S_{\text{res}}[p] .
\]

- **Low T / strong localization**: `p_ij` snaps to 0/1, `S → 0` → SSH-like dimerized state.
- **High T / weak localization**: `p_ij` equalize, `S` maximal → aromatic state.

### 5.2.1 Where the entropy and free energy come from

The classical starting point is the **dimer-covering ensemble**. A covering `C` is a set of edges such that every atom is incident to exactly one chosen edge (for `n_i=1`). Its energy is

\[
E[C] = \sum_{(ij) \in C} V_{ij}
\]

where `V_ij` is the energy of a double bond on that edge. The partition function is

\[
Z = \sum_{C} e^{-\beta E[C]} .
\]

The average bond occupation is

\[
p_{ij} = \langle \mathbf{1}_{(ij)\in C} \rangle
= -\frac{1}{\beta}\frac{\partial \ln Z}{\partial V_{ij}},
\]

and the entropy is

\[
S = -\frac{\partial F}{\partial T},
\qquad
F = -\frac{1}{\beta}\ln Z .
\]

This is exact for a **classical** dimer model. Our solver does not enumerate coverings. Instead it replaces the exact combinatorial sum by a **variational mean-field** distribution over bonds, subject to the hard-core constraint `Σ_j p_ij = n_i`. The free-energy functional is then

\[
F[p] = U[p] - T S_{\text{MF}}[p]
\]

where `S_MF` is the entropy of the variational distribution. The cheapest choice is the **linear / Tsallis-2 entropy**

\[
S_{\text{res}}[p] = \sum_i \left( n_i - \sum_j p_{ij}^2 \right)
\]

which is the leading-order approximation to the exact Shannon entropy of the local bond distribution and is exact in the binary limit `p ∈ {0,1}`.

The **quantum dimer model** adds kinetic terms that flip pairs of dimers around plaquettes:

\[
H_{\text{QDM}} = \sum_{(ij)} V_{ij} n_{ij}
+ \sum_{\text{plaq}} t \left( | \text{flip} \rangle \langle \text{flip} | + \text{h.c.} \right).
\]

For small `t` (or high temperature) the kinetic term simply increases the number of accessible coverings and lowers the free energy of uniform aromatic patterns. The `KekulePure` aromatic penalty centered at `p=0.5` and the resonance-entropy term are both proxies for this kinetic / resonance energy.

Therefore our free-energy functional can be read as:

- `U[p]` = bond potential + electrostatics + Hubbard repulsion (the `V_e` of QDM plus QEq/U).
- `S_res[p]` = mean-field resonance entropy of the dimer ensemble.
- `T` = effective temperature that controls the competition between localization (bond energy) and delocalization (resonance).

At `T=0` we recover a single Kekulé structure; at high `T` we recover the aromatic average.

---

## 6. Green’s functions, NEGF, and Landauer–Büttiker conductance

### 6.1 Local density matrix from the Green’s function

The exact one-particle density matrix can be written through the lesser Green’s function:

\[
\rho_{ij} = -\frac{1}{\pi} \int_{-\infty}^{\mu} \operatorname{Im}\, G_{ij}^<(E)\, dE .
\]

The BOP Lanczos/continued-fraction method is essentially a cheap local approximation to `G_{ij}`:

```@/home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleOrderN_Gemini_BOP_2D_v2.py:169-209
rho_plus = -(1.0 / np.pi) * np.imag(G_plus) * dE
...
return 2.0 * 0.5 * (rho_plus - rho_minus)
```

### 6.2 Landauer–Büttiker channel analogy

In a two-terminal nanoconductor, the conductance is

\[
G = \frac{2e^2}{h} \sum_\alpha T_\alpha .
\]

Each transverse mode carries at most one electron per spin. The Pauli principle therefore gives a **unit quantum conductance per channel**. The Kekulé bond-order sum rule can be viewed as a **classical projection** of this channel idea: each singly occupied p_z orbital at atom `i` contributes to exactly one bond channel, so the total weight of all channels leaving the atom is 1.

This is a formal analogy, not a literal conductance statement. It becomes literal only if a finite bias or magnetic field is added and the density matrix becomes complex.

---

## 7. Established theories and related methods

| Method | What it shares with this project |
|---|---|
| **Pauling / Hückel bond order** | Classical π-bond order concept; Hückel off-diagonal density matrix gives bond order `2 ρ_ij`. |
| **Wiberg / Mayer bond order** | Quantum bond-order indices derived from the density matrix; used to validate classical bond orders. |
| **Pariser–Parr–Pople (PPP)** | π-only Hamiltonian with Coulomb interaction; gives physical parameters for our energy terms. |
| **Valence Bond (VB) theory** | Explicit superposition of Kekulé structures; rigorous basis for the classical resonance picture. |
| **Resonating Valence Bond (RVB)** | Quantum superposition of dimer coverings; the quantum version of aromatic resonance. |
| **Quantum dimer model** | Hamiltonian acting on dimer coverings; provides the parent model for our constraints. |
| **SSH / Peierls model** | Bond-order wave and solitons in polyacetylene. |
| **Graph b-matching / Kasteleyn dimer counting** | Combinatorial methods for counting Kekulé structures; the classical entropy of the ensemble. |

### 7.1 Pauling / Hückel bond order

#### What it is

- **Pauling bond order**: a classical resonance average over Kekulé structures. For each bond, count how many of the `K` Kekulé structures have a double bond there, and divide by `K`. For benzene there are 2 such structures, so every bond is double in exactly one of them and the Pauling bond order is `0.5`.
- **Hückel bond order**: a quantum MO result. For a closed-shell determinant in an orthonormal basis,

\[
P_{ij} = 2 \sum_{k \in \text{occ}} c_{ik} c_{jk}
\]

(spinless convention). For benzene the Hückel bond order is `2/3`, not `0.5`.

#### Assumptions

- **Pauling**: only covalent Kekulé structures, equal weights, no explicit Hamiltonian.
- **Hückel**: non-interacting π electrons, one electron per atom, orthonormal basis.

#### Core principles

Both assign a single number to each bond that interpolates between a single and a double bond. Pauling is a *counting* rule; Hückel is a *density-matrix* rule.

#### Derivation / formula

- Pauling:

\[
p^{\text{Pauling}}_{ij} = \frac{1}{K} \sum_{m=1}^{K} p^{(m)}_{ij},
\qquad p^{(m)}_{ij} \in \{0,1\} .
\]

- Hückel:

\[
p^{\text{Hückel}}_{ij} = 2 \rho_{ij} .
\]

#### Cost / scaling

- **Pauling**: enumeration of Kekulé structures. For planar molecules, Kasteleyn’s determinant gives the number in `O(N^3)`; for arbitrary graphs, counting is #P-hard. The number of structures grows exponentially with system size.
- **Hückel**: full diagonalization `O(N^3)` dense, or `O(N)` sparse for 1D/2D grids.

#### Connection to our solver

- `KekulePure` with `allow_aromatic=True` computes the Pauling bond order when the aromatic term weights all Kekulé structures equally.
- The valence constraint `Σ_j p_ij = 1` is exactly the Kekulé counting rule.
- The BOP computes the Hückel bond order (or a local approximation). The classical solver should reproduce Pauling, not Hückel.
- **Kasteleyn vs. iterative minimization**: Kasteleyn does *not* diagonalize a Hamiltonian. It computes a Pfaffian / determinant of a **signed adjacency matrix** and gives the exact number (or weighted partition function) of Kekulé matchings on a planar graph. To obtain local bond probabilities one also needs `K^{-1}` or ratios of determinants, so the whole procedure is `O(N^3)`. It is a **global counting** method, not a linear solve of our optimization problem. For non-planar graphs or variable `n_i` (b-matching), Kasteleyn does not apply directly, and our iterative solver is the practical alternative.
- **Cheap improvement**: fit the localization-snap parameters to reproduce known Pauling bond orders for a small set of training molecules.

---

### 7.2 Wiberg / Mayer bond order

#### What it is

Quantum bond-order indices computed from the one-particle density matrix. They are widely used in computational chemistry to quantify covalent bond strength.

#### Assumptions

- **Wiberg**: density matrix in an orthogonal basis (Löwdin-orthogonalized atomic orbitals).
- **Mayer**: non-orthogonal basis; uses the AO overlap matrix `S`.

#### Core principles

Covalent bond strength between two atoms is captured by the off-diagonal density shared between them.

#### Derivation / formula

- Wiberg index:

\[
W_{AB} = \sum_{\mu \in A} \sum_{\nu \in B} \big(P^\perp_{\mu\nu}\big)^2 .
\]

- Mayer bond order:

\[
B_{AB} = \sum_{\mu \in A} \sum_{\nu \in B}
\big[(PS)_{\mu\nu}(PS)_{\nu\mu} + \text{spin terms}\big] .
\]

#### Cost / scaling

- `O(N^2)` for dense matrices, `O(N)` for localized orbitals. The cost is negligible once a density matrix is available.

#### Connection to our solver

- Wiberg / Mayer gives the quantum target that our classical `p_ij` should approximate.
- The local linear entropy

\[
S_i = n_i - \sum_j p_{ij}^2
\]

is a Wiberg-like sum of squared bond orders: it measures how much the bond order is spread over neighbors.
- **Cheap improvement**: compute Wiberg / Mayer indices from a reference DFT/HF calculation for small molecules and fit the `Karo` / `Kloc` parameters in `KekulePure`.

---

### 7.3 Pariser–Parr–Pople (PPP)

#### What it is

A semiempirical π-electron Hamiltonian that includes one-electron hopping and electron-electron interactions.

#### Assumptions

- π-only active space.
- Zero differential overlap (only one p_z orbital per atom).
- Semiempirical Coulomb integrals `γ_ij` and on-site repulsion `U`.

#### Core principles

The physics is governed by the competition between delocalization (hopping `t`) and Coulomb repulsion (`U`, `γ_ij`).

#### Derivation / formula

\[
H_{\text{PPP}} =
\sum_{ij} t_{ij} \big(c_i^\dagger c_j + \text{h.c.}\big)
+ U \sum_i n_{i\uparrow} n_{i\downarrow}
+ \frac{1}{2} \sum_{ij} \gamma_{ij} (n_i - 1)(n_j - 1) .
\]

#### Cost / scaling

- Self-consistent mean-field: `O(N^3)` dense, `O(N)` sparse.
- Full configuration interaction: exponential in active-space size.
- Modern selected CI / DMRG: a few hundred π electrons.

#### Purpose

Reproduces electronic spectra, bond-length alternation, charge distributions, and excitation energies in conjugated molecules.

#### Connection to our solver

- PPP provides physically motivated parameters for our free-energy functional:
  - `t_ij` → hopping bond energy.
  - `U` → Hubbard penalty `E_U(n)`.
  - `γ_ij` → QEq electrostatic kernel.
- The PPP ground state can be written in a VB basis; the dominant structures are Kekulé structures. Our classical solver is a mean-field approximation to PPP in the VB/dimer basis.
- **Cheap improvement**: fit `V_loc(p)` to PPP energies of small training molecules (e.g., benzene vs. cyclobutadiene) instead of using generic parabolas.

---

### 7.4 Valence Bond (VB) theory

#### What it is

A wavefunction method that expands the exact many-electron state in a basis of covalent and ionic structures. Each structure is a particular pairing of electrons into bonds and charges.

#### Assumptions

- Structures are non-orthogonal.
- Matrix elements are evaluated by the Pauling rules or spin algebra.

#### Core principles

The ground state is a **resonance superposition** of many structures. The coefficients are obtained from the secular equations.

#### Derivation / formula

\[
| \Psi \rangle = \sum_m C_m | K_m \rangle,
\qquad
H C = E S C
\]

where `H` and `S` are the Hamiltonian and overlap matrices in the VB basis. The bond order is the expectation of a dimer number operator:

\[
p_{ij} = \sum_{m,n} C_m^* C_n \langle K_m | \hat{n}_{ij} | K_n \rangle .
\]

#### Cost / scaling

- Number of structures grows combinatorially with `N`; non-orthogonality adds overhead.
- Modern selected VB or spin-coupled methods keep it manageable for a few tens of atoms.
- Full VB is not a linear-scaling method.

#### Purpose

The rigorous quantum justification of Kekulé resonance and bond-order concepts.

#### Connection to our solver

- The constraint `Σ_j p_ij = n_i` is the VB normalization of the bond distribution.
- The `KekulePure` energy `E_val + E_aro + E_loc` is a parametrized approximation to the VB energy surface.
- Aromatic `p = 0.5` is the VB ensemble average.
- **Cheap improvement**: run a small selected-VB calculation for a representative molecule to calibrate the relative weights of the aromatic and localization terms.

---

### 7.5 Resonating Valence Bond (RVB)

#### What it is

A quantum superposition of all possible singlet dimer coverings of a lattice.

#### Assumptions

- Half-filled, strongly correlated, singlet pairing; double occupancy is suppressed.
- The wavefunction is a sum over dimer coverings with amplitudes.

#### Core principles

The ground state is not a single Kekulé structure but a coherent superposition of many. The amplitudes are chosen to minimize the Heisenberg / t-J energy.

#### Derivation / formula

\[
| \text{RVB} \rangle = \sum_{C} \psi(C) | C \rangle,
\]

with each covering written as a product of singlet bonds:

\[
| C \rangle = \prod_{(ij) \in C}
\frac{1}{\sqrt{2}}
\big( | \uparrow_i \downarrow_j \rangle - | \downarrow_i \uparrow_j \rangle \big) .
\]

#### Cost / scaling

- Exact diagonalization of the covering space is exponential.
- Quantum Monte Carlo or tensor-network methods can handle large lattices with polynomial scaling, but are still much more expensive than our classical solver.

#### Purpose

Describe spin liquids, high-temperature superconductivity, and strongly correlated aromatic systems.

#### Connection to our solver

- Our classical Kekulé ensemble is the **incoherent / classical** limit of an RVB state: we average probabilities rather than amplitudes.
- The valence sum rule is the RVB hard-core constraint.
- The resonance entropy is the classical counterpart of RVB entanglement entropy.
- **Small-object / GPU framing**: each dimer covering is a list of local singlet pairs. In a QMC sampler, every GPU thread can propose a local move (e.g., flip a pair of dimers on a plaquette) and compute the acceptance probability from the local energy change. The covering itself is a compact integer/bond list, so memory per configuration is tiny. This is the natural direction for adding quantum corrections to the classical solver without ever building large matrices or molecular orbitals.
- **Cheap improvement**: start from the `KekulePure` classical minimum, then run a short local-update QMC around it to sample RVB-style quantum fluctuations. The classical seed supplies the energy landscape; the QMC supplies the resonance weights.

---

### 7.6 Quantum dimer model

#### What it is

A Hamiltonian that acts directly on the Hilbert space of dimer coverings.

#### Assumptions

- Only the dimer (singlet-bond) sector is retained.
- Dynamics is generated by local dimer flips.

#### Core principles

- **Potential term**: `V_e n_e` assigns an energy to a dimer on edge `e`.
- **Kinetic term**: a local dimer flip. On a square plaquette, if two parallel edges are occupied by dimers, the Hamiltonian can replace them by the complementary pair of parallel edges.

#### What the “flip” actually is

It is **not** a spin flip, a phase sign change, or a bond-direction swap. It is a local rewrite of the dimer covering. On a plaquette with vertices `i, j, l, k`:

```
Before:           After:
  i —— j           i    j
                   |    |
  k —— l           k    l

Dimer edges:      Dimer edges:
(i,j) and (k,l)  (i,k) and (j,l)
```

The two parallel dimers on one pair of opposite edges are replaced by the two parallel dimers on the other pair. All four vertices remain singly matched; the hard-core constraint is preserved.

#### Mapping to an Ising-like update

For each plaquette, define a binary variable `σ_p = +1` for one of the two possible dimer configurations and `σ_p = -1` for the other. A dimer flip is exactly a local flip of `σ_p`. The QDM is therefore an Ising-type model on the plaquettes of the dual lattice, with the hard-core constraint enforced by the dimer structure. This is the same structure as the Coulomb/Ising models you have already implemented efficiently on GPU.

#### Derivation / formula

\[
H = \sum_e V_e n_e
+ \sum_{\text{plaq}} t
\big( | \text{flip} \rangle \langle \text{flip} | + \text{h.c.} \big) .
\]

#### Kinetic energy without molecular orbitals

A single dimer covering is **not** an eigenstate of the full Hamiltonian; it is a product of localized singlet bonds. Its energy is therefore not a sum of isolated dimer orbital energies. In VB/QDM, the kinetic energy is represented **off-diagonal in the dimer basis**.

- The Hubbard/Hückel kinetic term `H_t = -t \sum_{\langle i,j \rangle} (c_i^\dagger c_j + \text{h.c.})` moves an electron from one site to a neighbor.
- Acting on a valence-bond state, this move changes which bonds are occupied: it turns one covering `|C\rangle` into a different covering `|C'\rangle`.
- The kinetic energy lowering is the **resonance energy** between these two coverings, i.e. the matrix element

\[
  \langle C' | H_t | C \rangle \approx -t .
\]

In the QDM, this is exactly the plaquette flip term. The Hamiltonian is a sparse local operator on the graph of dimer coverings: it connects a configuration only to its immediate neighbors in covering space.

For a QMC sampler, the local energy of a covering `C` is

\[
E(C) = \sum_{e \in C} V_e - t \, N_{\text{flip}}(C) ,
\]

where `N_{\text{flip}}(C)` is the number of flippable plaquettes in the current covering. The kinetic contribution is therefore a **local count of allowed moves**, not a sum of MO energies. The molecular-orbital delocalization is recovered by sampling many coverings and their superposition, not by diagonalizing a Hamiltonian in the AO basis.

This is the same mechanism in VB theory: the Hamiltonian matrix is built in the non-orthogonal VB basis, and only local bond rearrangements give non-zero matrix elements.

#### Cost / scaling

- Exact diagonalization in the VB/dimer basis is still exponential in `N`.
- Quantum Monte Carlo uses the local kinetic estimator above and remains polynomial for sign-problem-free cases.
- Classical limit (`t → 0`): reduces to enumerating coverings with edge energies `V_e`.

#### Purpose

Model RVB states, topological order, and finite-temperature dimer liquids; provide the parent Hamiltonian for the Kekulé bond-order pattern.

#### Connection to our solver

- The `KekulePure` valence constraint is exactly the dimer hard-core constraint.
- The localization snap `V_loc(p)` is the dimer potential `V_e`.
- The aromatic term / resonance entropy approximates the kinetic flip energy.
- The free-energy functional in section 5 is the **mean-field / classical limit** of the QDM.

---

### 7.7 Graph b-matching / Kasteleyn dimer counting

#### What it is

Combinatorial methods to count the number of ways to cover a graph with dimers.

#### Assumptions

- **Perfect matching**: every vertex is incident to exactly one dimer.
- **b-matching**: vertex `i` is incident to `b_i` dimers (generalizes the perfect-matching case).

#### Core principles

A **Pfaffian** is a sum over perfect matchings of a graph. For a skew-symmetric matrix `K`, the Pfaffian is defined as

\[
\operatorname{Pf}(K) = \sum_{M} \operatorname{sgn}(M) \prod_{(ij) \in M} K_{ij}
\]

where the sum runs over all perfect matchings `M`. **Kasteleyn’s theorem** says that for a planar graph, one can assign signs to the edges such that every matching has the same sign. Then the number of perfect matchings is

\[
N_{\text{Kekulé}} = \operatorname{Pf}(K) = \sqrt{ | \det(K) | } .
\]

For a **weighted** dimer model, the edge weight `w_{ij}` is put into `K_{ij}` and the partition function is

\[
Z = \operatorname{Pf}(K) .
\]

#### Numerical algorithm / pseudocode

1. **Build the Kasteleyn matrix**.
   - Create an `N × N` skew-symmetric matrix, one row/column per vertex.
   - For each edge `(i,j)` with weight `w_{ij}`, assign a sign `σ_{ij} ∈ {+1, -1}` according to a Kasteleyn orientation of the planar graph, and set
     \[
     K_{ij} = σ_{ij} w_{ij}, \qquad K_{ji} = -σ_{ij} w_{ij}.
     \]
   - For an unweighted count, set `w_{ij} = 1`.

2. **Compute the determinant**.
   - `N_Kekulé = sqrt(|det(K)|)`.
   - For a weighted partition function: `Z = sqrt(det(K))`.

3. **Compute local dimer probabilities**.
   - Solve `K x = e_j` for each column `j`, or compute the full inverse `K^{-1}`.
   - The probability that edge `(i,j)` is occupied is
     \[
     p_{ij} = K_{ij} (K^{-1})_{ji}.
     \]
   - This is the same information as our optimized `p_ij`, but exact for the planar weighted dimer model.

```python
def kasteleyn_count(graph, weights=None):
    N = len(graph.nodes)
    K = np.zeros((N, N), dtype=float)
    for (i, j), w in graph.edges(data='weight', default=1.0):
        s = kasteleyn_sign(i, j, graph)   # planar edge orientation
        K[i, j] =  s * w
        K[j, i] = -s * w
    return np.sqrt(abs(np.linalg.det(K)))

def kasteleyn_probabilities(graph, weights=None):
    K = build_kasteleyn(graph, weights)   # same as above
    Kinv = np.linalg.inv(K)               # use sparse solve in production
    p = {}
    for (i, j) in graph.edges:
        p[(i, j)] = K[i, j] * Kinv[j, i]
    return p
```

#### Sparse / fast determinant algorithms

- The Kasteleyn matrix is as sparse as the adjacency matrix. For a planar graph, a nested-dissection ordering gives an `O(N^{3/2})` or `O(N \log N)` sparse determinant.
- Standard sparse LU or LDL^T factorizations can be used; the matrix is skew-symmetric, so a specialized LDL^T saves half the work.
- For very large planar graphs, fast multipole / H-matrix techniques can push the determinant down to near-linear.
- For arbitrary non-planar graphs, exact counting is #P-hard; use Monte Carlo or belief propagation instead.

#### From counting to entropy

For the uniform classical dimer model, the entropy is simply

\[
S = \ln N_{\text{Kekulé}} .
\]

For the weighted model, the free energy is

\[
F = -\frac{1}{\beta} \ln Z = -\frac{1}{\beta} \ln \operatorname{Pf}(K) .
\]

The entropy is then

\[
S = -\frac{\partial F}{\partial T}
= \beta \sum_e V_e p_e - \ln Z .
\]

This gives the exact classical resonance entropy for a given set of edge weights.

#### How to assign Boltzmann weights without diagonalizing a Hamiltonian

The edge weights do not need a full Hückel or CI diagonalization. They can come directly from the **classical energy model** we already have:

\[
w_{ij} = \exp[-\beta V_{ij}], \qquad
V_{ij} = -t_{ij} p_{ij} + V_{\text{loc}}(p_{ij}) + \text{QEq/U terms}.
\]

In other words, `KekulePure` supplies the bond energies, and Kasteleyn tells us how many ways those energies can be satisfied and what the resulting thermal bond probabilities are. The quantum Hamiltonian is replaced by an effective classical energy function on dimer coverings. For quantum corrections beyond this classical energy, use the RVB/QMC sampler described in 7.5 and 7.6.

#### b-matching for variable `n_i`

- Perfect matching is `b_i = 1` for every vertex.
- For general `b_i`, the problem is a **b-matching**: each vertex `i` must be incident to exactly `b_i` chosen edges.
- On planar graphs, b-matchings can be handled by constructing a larger graph with `b_i` copies of each vertex, or by a generalized Pfaffian method.
- On general graphs, use local Monte Carlo updates that preserve the vertex degree (degree-conserving bond swaps).

#### Cost / scaling

- **Planar Kasteleyn**: `O(N^3)` for the determinant; `O(N^2)` with fast algorithms; near-linear for very large planar graphs with H-matrices.
- **b-matching on planar graphs**: similar to perfect matching, with a constant-factor overhead from vertex replication.
- **Arbitrary non-planar graphs**: #P-hard in general; approximate methods (cavity / Bethe–Peierls, Monte Carlo) are needed.

#### Purpose

- Exact resonance entropy for small molecules.
- Statistical mechanics of dimer coverings (free energy, correlations, local probabilities).
- Benchmark for the classical KekulePure solver.

#### Connection to our solver

- The valence constraint `Σ_j p_ij = n_i` is the **fractional** b-matching condition.
- The exact entropy of the Kekulé ensemble is the logarithm of the number of b-matchings.
- Kasteleyn is the exact benchmark for small planar molecules; our iterative solver is the fast approximation for large or non-planar systems.
- **Cheap improvement**: for small training molecules, compute `ln N_Kekulé` with Kasteleyn and fit the coefficient of the linear entropy `S_res`. For large systems, use the Bethe–Peierls / cavity approximation to the b-matching entropy, which scales as `O(N)`.

The classical Kekulé bond-order idea is very old (Pauling, 1930s), but it has a modern echo in the study of **topological states in graphene**, **Kekulé bond-order distortions**, and **Kekulé edge reconstructions**, where the bond pattern can open gaps, create edge states, and stabilize non-trivial phases.

---

## 7.8 Clarifying the core concepts

### 7.8.1 What is a Pfaffian, and why is the Kasteleyn matrix signed?

The **Pfaffian** of a skew-symmetric matrix `K` is a sum over all ways to pair up the rows/columns, i.e. over all perfect matchings of the graph:

\[
\operatorname{Pf}(K) = \sum_M \operatorname{sgn}(M) \prod_{(ij) \in M} K_{ij}.
\]

Each matching comes with a sign `sgn(M)` that depends on the order in which the pairs are written. If we simply put the adjacency matrix into `K`, different matchings can have different signs and cancel each other. **Kasteleyn’s theorem** says: for a planar graph, we can assign signs to the edges so that every matching has the same sign. Then the Pfaffian is exactly the number of matchings:

\[
N_{\text{Kekulé}} = \operatorname{Pf}(K) = \sqrt{|\det(K)|}.
\]

The sign assignment is purely combinatorial: it comes from orienting the edges of the planar graph so that every face has an odd number of clockwise-pointing arrows. It has **nothing to do with current conservation** or Kirchhoff’s laws. Kirchhoff’s matrix-tree theorem counts spanning trees using the **Laplacian** matrix; Kasteleyn counts perfect matchings using a **skew-symmetric signed adjacency** matrix.

### 7.8.2 What is a plaquette, and what is a flip?

A **plaquette** is just a small face of the lattice. For a square lattice it is a square; for graphene it is a hexagon.

In a **dimer covering** (a perfect matching), every vertex is touched by exactly one dimer. Consider a plaquette whose four boundary edges form a square. If the two **opposite** edges of the plaquette are occupied by dimers, the other two opposite edges are empty. A **flip** is the local move that swaps the occupied pair with the empty pair:

```
Before:           After:
  i —— j           i    j
  |    |           |    |
  k —— l           k —— l

Dimer edges:      Dimer edges:
(i,j) and (k,l)  (i,k) and (j,l)
```

The two parallel dimers on one pair of opposite edges are replaced by the two parallel dimers on the other pair. Every vertex still has exactly one dimer; the hard-core constraint is preserved.

### 7.8.3 Is a dimer covering one dimer or a pattern of many dimers?

A **dimer covering** is a **global pattern** of many dimers that covers the whole graph. It is not a single dimer, and it is not a sum of independent dimers. The constraint is that every vertex is incident to exactly one chosen edge.

Because of this constraint, the allowed states are **global patterns** (Kekulé structures). For a large lattice, the number of such patterns grows exponentially. We do not enumerate them independently. Instead, we use local moves (flips) to jump from one pattern to a nearby pattern, and we sample the patterns statistically.

### 7.8.4 Can the Monte Carlo compute only local energy differences?

Yes. In a local-update MC, the total energy is a sum of local bond energies. When a flip changes only a few bonds, the energy difference is

\[
\Delta E = E_{\text{new}} - E_{\text{old}} = \sum_{\text{changed bonds}} V_{\text{new}} - \sum_{\text{changed bonds}} V_{\text{old}}.
\]

All unchanged bonds contribute the same amount to both `E_new` and `E_old` and cancel. For a plaquette flip, only the four bonds of the plaquette (and perhaps their immediate neighbors) are involved. This is the key reason the method is cheap and GPU-friendly: each thread can compute a small local update without touching the rest of the system.

---

## 8. Open questions

### 8.1 Theoretical gaps

1. **What is the best entropy functional?**
   - Linear entropy `S_i = n_i - Σ p_ij^2` is simple.
   - Shannon entropy of the bond distribution is more standard.
   - The exact combinatorial entropy of dimer coverings (b-matchings) is hard to compute for large graphs.

2. **How does the BOP prior enter the free energy?**
   - Should `t_ij` be the raw Hamiltonian hopping, or a screened value derived from the local density matrix?
   - How many Chebyshev / Lanczos steps are needed before the BOP prior is accurate enough?

3. **How to handle variable `n_i` rigorously?**
   - Is `n_i` the **π-electron count**, the **spin-orbital occupancy**, or both?
   - How to enforce the Pauli bound `p_ij ≤ 1` and the Hubbard penalty consistently?

4. **How to include the σ framework?**
   - The current model is π-only. The σ skeleton determines geometry and provides the hopping graph.
   - A hybrid model may need to couple σ relaxation to the π bond order.

5. **What about antiaromatic / open-shell / radical systems?**
   - The classical Kekulé ansatz assumes a closed-shell singlet with `n_i = 1`.
   - Radicaloids, antiaromatic rings, and polyradical graphene edges require a spin generalization.

6. **Can topological invariants be read from `p_ij`?**
   - SSH winding number, Zak phase, and graphene edge states depend on the bond-order pattern.
   - It is not yet clear how to extract these quantities from the classical optimizer alone.

### 8.2 Computational / practical gaps

7. **How to make the self-consistent loop stable?**
   - BOP → KekulePure → QEq → BOP loops may oscillate.
   - Need robust mixing / penalty-strength scheduling.

8. **How to validate?**
   - Compare against exact diagonalization of the PPP/Hubbard model for small molecules.
   - Compare against DFT for medium-sized systems.
   - Benchmark for graphene nanoribbons and topological defects.

9. **How to scale to large systems?**
   - The BOP part is already linear-scaling.
   - The KekulePure optimizer is currently local-gradient-based; may need graph-Newton / multiscale methods for large flakes.

---

## 9. Applications

### 9.1 Quinone / aromatic transitions

In oxidative aromatization of quinone, pyrazine, or phenazine, the process

\[
\ce{C=O <-> C-OH}
\]

or

\[
\ce{N= <-> -NH-}
\]

changes the local π count and creates a topological defect in the π graph. The classical Kekule solver can propagate the defect non-locally: a missing π orbital at one site forces a reorganization of double bonds up to a second defect (or the edge). This gives the well-known rule that quinones accept/release two hydrogens at a time, with a bond-order soliton between the two redox sites.

### 9.2 Graphene nanoribbons and doped sheets

- Nitrogen or oxygen edge dopants change local `n_i` and `t_ij`.
- Edge reconstruction (e.g. Klein, armchair, zigzag) changes the bond graph.
- A fast Kekulé/BOP hybrid could predict edge states, charge localization, and aromatic domains in large graphene nanoribbons without a full DFT diagonalization.

---

## 10. Proposed hybrid algorithm

A practical way to combine the two pieces:

1. **Initialize** `n_i` from atom types or a QEq guess.
2. **Build** `H(n)` with charge-dependent on-site energies `H_ii = ε_i^0 + U(n_i - n_i^0)`.
3. **BOP step**: run a few Chebyshev / Lanczos steps per bond to obtain a local prior `p_ij^BOP = f(ρ_ij^BOP)`.
4. **KekulePure step**: minimize

   \[
   E_{\text{Kekule}} = E_{\text{val}} + E_{\text{aro}} + E_{\text{loc}}
   + \lambda \sum_{(ij)} \big(p_{ij} - p_{ij}^{\text{BOP}}\big)^2
   + E_{\text{QEq}}(n) + E_U(n)
   \]

   subject to `Σ_j p_ij = n_i`, `0 ≤ p_ij ≤ 1`, `Σ_i n_i = N_π`.
5. **Update** `n_i` from the new bond distribution (or from a fresh BOP pass).
6. **Iterate** 2–5 until self-consistency.

Add the resonance-entropy term `S_res[p]` if you want to control the localization/aromatization balance explicitly.

---

## 11. Summary mindmap

- **Quantum side** (BOP)
  - Hamiltonian → density matrix `ρ`
  - `ρ` is a projector for a single determinant
  - Off-diagonal `ρ_ij` = bond density
  - Local probes → linear-scaling approximation
  - Misses global correlations and strong coupling

- **Classical side** (`KekulePure`)
  - Bond graph → variables `p_ij ∈ [0,1]`
  - Constraint `Σ_j p_ij = n_i`
  - Energy terms enforce valence, aromaticity, localization
  - Captures global/topological effects and resonance
  - Misses quantum interference

- **Rigorous connection**
  - Kekulé sum rule = linear relaxation of idempotency `Σ_j |ρ_ij|^2 = n_i`
  - Aromatic 0.5 = mixed-state average of binary Kekulé determinants
  - Strong-coupling limit = RVB / quantum dimer model
  - Variable `n_i` = generalized b-matching + QEq + Hubbard U

- **Physical analogies**
  - Probability channels, not electrical currents
  - SSH = bond-order wave; KekulePure = graph-level SSH solver
  - Entropy = deviation from idempotency
  - Landauer–Büttiker = unit channel capacity from Pauli exclusion

- **Open goals**
  - Choose the best entropy functional
  - Make BOP ↔ KekulePure self-consistency robust
  - Validate on small exact systems, then scale to graphene nanoribbons
  - Apply to redox-active aromatics (quinone, phenazine, pyrazine)

---

## 12. The intended numerical path: classical seed + local GPU statistics

The focus of this project is **not** to diagonalize large matrices (not even Hückel `O(N^3)`), and certainly not to do full CI. The goal is to keep the solver fast, local, and GPU-friendly, while putting the classical bond-order model on a more rigorous physical footing.

### 12.1 No large diagonalizations

- **Kasteleyn** is not a diagonalization of a Hamiltonian; it is a determinant of a sparse signed adjacency matrix. It gives exact classical weights for planar graphs, but it is still `O(N^3)` and restricted to fixed, perfect matchings on planar graphs.
- **Our iterative solver** (`KekulePure`) is the practical fallback: it handles non-planar graphs, variable `n_i`, arbitrary energy terms, and scales linearly with local updates.
- **QMC / RVB** does not build MOs or density matrices; it samples compact dimer coverings and evaluates local energy changes.

### 12.2 Classical seed

1. Run `KekulePure` to get an approximate bond pattern `p_ij` and effective bond energies `V_ij`.
2. From these, assign Boltzmann weights `w_ij = exp(-β V_ij)` to each edge.
3. This gives a classical dimer model whose partition function and probabilities can be estimated by Kasteleyn (small/planar) or by local Monte Carlo (large/non-planar).

### 12.3 Local GPU update strategy

The covering is a list of occupied bonds. Each GPU thread handles a small local object:

- **Plaquette flip** (QDM): four bonds around a plaquette are replaced by the complementary four bonds.
- **Bond swap** (b-matching): rotate a pair of adjacent dimers among three atoms.
- **RVB amplitude update**: recompute the local singlet product on the changed bonds.

Because the moves are local, the energy change and acceptance probability can be computed from the few bonds involved. No global diagonalization is required. This is the same structure as the Coulomb/Ising GPU solvers you already have.

### 12.4 What the quantum correction provides

- It reweights the classical Kekulé structures according to their quantum (RVB) amplitudes, not just their classical energy.
- It can move bond order away from the single classical minimum toward a more uniform, aromatic distribution.
- It provides the kinetic/resonance energy that the classical entropy term approximates.

### 12.5 Work plan

1. Use `KekulePure` to get a fast classical baseline.
2. For small molecules, use Kasteleyn to calibrate the entropy term `S_res`.
3. For large systems, run a GPU local-update QMC sampler around the classical seed to add quantum fluctuations.
4. Validate the final bond orders against Wiberg/Mayer indices from a small reference calculation.

This keeps the method cheap, local, and GPU-friendly while connecting it to the rigorous RVB/QDM/QMC framework.

---

*Document status: living draft. The framework is conceptually rigorous; the remaining work is to make the numerical hybrid stable, to choose the right entropy functional, and to validate against exact and DFT results.*
