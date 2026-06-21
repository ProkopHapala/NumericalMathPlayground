# USER

look into @KekuleBondSum.py @README.md and the other files in this folder 
/home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP
also the chat here @KekuleOrderN.chat.md 

can you layout the principles of the two methods?

then I have major question how these two methods (one quantum the other semiclasical) are related.

Most importantly what the classical kekule bond order obtained by summing the bonds to given atom means in quantum terms. How it is related to Pauli exclusion principle (orthogonalility, fermi-direc statistics) and idenpotency of density matrix. can we say that we somehow estimate density matrix? Can we relate out 1 electron per pi-orbital ot hubberd U and singly/doubly occupied sites 

---

# Kimi 2.6

# The two methods in [Kekule_BOP](cci:9://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP:0:0-0:0)

## 1. Quantum Bond-Order Potential / Fermi-Operator Expansion

The quantum route (files `KekuleOrderN_Gemini*.py`, `KekuleOrderN_Gemini_BOP*.py`) starts from a Hückel / tight-binding Hamiltonian for the π system

```@/home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleOrderN_Gemini_BOP_2D_v2.py:39-59
t0 = -2.7        # Base hopping (eV)
r0 = 1.42        # Equilibrium distance (Angstrom)
beta = 3.0       # Exponential decay factor (1/Angstrom)
...
if dist < 1.8:
    t = t0 * np.exp(-beta * (dist - r0))
    H[i, j] = H[j, i] = t
```

and approximates the zero-temperature one-particle density matrix

\[
\hat{\rho}=\Theta(\mu\hat I-\hat H)
\]

without diagonalizing the full matrix. The two implementations are:

- **Chebyshev / Fermi-Operator Expansion (FOE)**: expand the step function in Chebyshev polynomials and apply the three-term recurrence to local vectors:

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

- **Lanczos / continued fraction (Haydock recursion)**: build a small tridiagonal Krylov subspace for a bond-centered state \(|+\rangle=(|i\rangle+|j\rangle)/\sqrt2\) and \(|-\rangle=(|i\rangle-|j\rangle)/\sqrt2\), then use the algebraic identity

\[
\rho_{ij}=\frac{\rho_+-\rho_-}{2}
\]

```@/home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleOrderN_Gemini_BOP_2D_v2.py:169-209
def get_bond_density_CF(H, i, j, M, eta=0.2, ...):
    ...
    rho_plus = -(1.0 / np.pi) * np.imag(G_plus) * dE
    ...
    return 2.0 * 0.5 * (rho_plus - rho_minus)
```

Both methods give the π bond density \(\rho_{ij}\) as an off-diagonal element of the one-particle density matrix, and the bond energies are computed as \(\mathrm{Tr}[H\rho]\):

```@/home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleOrderN_Gemini1.py:97-99
E_pi += 2.0 * 2.0 * H[i, i+1] * rho_i_i1
```

The README calls this the “Bond-Order Potential” idea:

```@/home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/README.md:22-32
Instead of computing all eigenstates, BOP propagates a local Chebyshev
probe from a single bond (i,j) to estimate the bond density rho_ij.
```

## 2. Semiclassical Kekulé bond-order optimizer

[KekuleBondSum.py](cci:7://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:0:0-0:0) ([KekulePure](cci:2://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:26:0-417:17)) does **not** build a Hamiltonian. It optimizes a continuous π bond-order variable \(p_{ij}\in[0,1]\) on each bond with a phenomenological energy:

1. **Atom valence constraint**: each sp² carbon must have total π bond order equal to 1:

```@/home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:104-140
def project_valence(self):
    val = np.zeros(self.natom, dtype=float)
    np.add.at(val, self.i0, self.bo)
    np.add.at(val, self.i1, self.bo)
    return val
...
E_val = self.Kval * np.sum(d_val * d_val)
```

2. **Aromatic stabilization**: a parabola centered at \(p=0.5\).

3. **Localization snap**: piecewise parabolas pushing bonds toward 0, 0.5 or 1.

4. **Hard bounds** \([0,1]\).

```@/home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:142-165
# aromatic parabola centered at 0.5
d_aro = bo - 0.5
f_bond = -2.0 * self.Karo * d_aro
# localization snap parabolas around 0 / 0.5 / 1
target = self._localization_target(bo)
```

The result is a discrete Kekulé pattern (or an aromatic 0.5 pattern if `allow_aromatic=True`).

# How the two methods are related

| | Quantum BOP / FOE | [KekulePure](cci:2://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:26:0-417:17) |
|---|---|---|
| **Input** | Hückel Hamiltonian \(H_{ij}(r_{ij})\), on-site energies | Geometry + bond graph; no Hamiltonian |
| **Output** | Off-diagonal density-matrix element \(\rho_{ij}\) | Classical π bond order \(p_{ij}\) |
| **Kekulé pattern** | Emerges from interference / boundary reflections / Peierls energy minimum | Imposed by valence + localization constraints |
| **Electron count** | Fermi-Dirac filling of MOs; \(\langle n_i\rangle=1\) enforced by \(\mu=0\) | Hard constraint \(\sum_{j}p_{ij}=1\) per sp² atom |
| **Pauli / idempotency** | Exact: \(\rho^2=\rho\) at zero T | Not enforced; only a linear sum rule |

So the two methods are **different approximations to the same object**: the π bond order per bond. The quantum one is derived from the electronic structure; the classical one is a valence-bond / electron-bookkeeping model.

# What the classical bond-order sum means in quantum terms

The [KekulePure](cci:2://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:26:0-417:17) constraint

\[
\sum_{j} p_{ij}=1\quad\text{(per sp² atom)}
\]

is the **real-space, linearized analogue** of the diagonal occupancy of the p\(_z\) orbital:

\[
\rho_{ii}=1
\]

In a non-interacting Hückel picture, the carbon π system is half-filled and each atom carries exactly one π electron on average. The quantum BOP enforces this through the Fermi level:

```@/home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleOrderN_Gemini_BOP_2D_v2.py:64-68
for lam in range(N):
    if evals[lam] < 0.0: # Half-filled (mu = 0)
        rho_exact += 2.0 * np.outer(evecs[:, lam], evecs[:, lam])
```

However, the **true** quantum relation is not a linear sum, but the **idempotency** of the zero-temperature density matrix:

\[
\rho^2=\rho
\quad\Longrightarrow\quad
\sum_j |\rho_{ij}|^2=\rho_{ii}
\]

For a real symmetric density matrix with \(\rho_{ii}=1\),

\[
\sum_j \rho_{ij}^2 = 1
\]

The classical rule \(\sum_j p_{ij}=1\) is a **linearization** of this. It is exact for a single Kekulé structure where one bond per atom is a double bond (\(p=1\)) and the others are single (\(p=0\)):

\[
\sum_j p_{ij}=1,\qquad \sum_j p_{ij}^2=1
\]

But for an aromatic 0.5 pattern, the linear sum is still 1 while the quadratic sum is only

\[
\sum_j p_{ij}^2 = 2\times(0.5)^2 = 0.5
\]

So the classical 0.5 aromatic solution is **not** a single idempotent density matrix. It is the **average** of two (or more) Kekulé projectors. That is why [KekulePure](cci:2://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:26:0-417:17) has two modes:

- `allow_aromatic=False` → snaps to a single Kekulé structure (idempotent-like).
- `allow_aromatic=True` → allows 0.5 bonds (resonance average, non-idempotent).

# Pauli exclusion, Fermi-Dirac and idempotency

- **Pauli principle / orthogonality**: in the quantum BOP, the MOs are orthogonal, and the density matrix built from them is a projector. The eigenvalues of \(\rho\) are 0 or 1. No state can be fractionally occupied.
- **Fermi-Dirac**: at \(T=0\), the filling is a step function. For a half-filled carbon chain/graphene flake, \(\mu=0\), and the lower half of the spectrum is filled. This gives exactly one electron per atom on average.
- **Idempotency**: \(\rho^2=\rho\) is the density-matrix version of the Pauli principle. The quantum BOP approximates this directly.

The [KekulePure](cci:2://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:26:0-417:17) constraint \(\sum_j p_{ij}=1\) captures the **average occupancy** but not the projector property. Therefore it cannot, by itself, represent the coherent quantum superposition that produces aromatic 0.5 bond orders; it can only represent them as a statistical average.

# Does [KekulePure](cci:2://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:26:0-417:17) estimate the density matrix?

**No — it estimates a classical bond-order variable.**

The quantum BOP **does** estimate the density matrix:

\[
\rho_{ij}\approx\sum_k c_k\,\langle j|T_k(\tilde H)|i\rangle
\]

or, equivalently, via the local Green’s function. The output is a one-particle density matrix for non-interacting electrons.

[KekulePure](cci:2://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:26:0-417:17) outputs a set \(\{p_{ij}\}\) that can be **mapped** to \(\rho_{ij}\) only in the binary Kekulé limit. The aromatic 0.5 solution is not a single quantum density matrix; it is a resonance average.

# Relation to Hubbard \(U\) and singly/doubly occupied sites

In the Hückel / BOP picture, the electron-electron interaction is absent (\(U=0\)). The statement “one electron per π orbital” means

\[
\langle n_i\rangle = \langle n_{i\uparrow}+n_{i\downarrow}\rangle = 1
\]

This is an **average**. The actual wavefunction contains local configurations with both singly and doubly occupied p\(_z\) orbitals.

[KekulePure](cci:2://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:26:0-417:17) is closer to the **strongly correlated / large-\(U\)** limit:

- Each atom is constrained to contribute exactly one π electron.
- Bonds snap to 0 or 1, as if electrons are singly occupied and form spin-singlet bonds between neighboring singly occupied sites.
- The 0.5 aromatic bond is the resonance average of such singly-occupied valence-bond structures.

A finite Hubbard \(U\) interpolates between these limits: small \(U\) gives the delocalized Hückel density matrix computed by BOP; large \(U\) gives a Mott insulator with singly occupied sites and resonating valence bonds, more like the output of [KekulePure](cci:2://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:26:0-417:17).

# Bottom line

- **Quantum BOP** is a real-space approximation to the **idempotent one-particle density matrix** of a non-interacting π system. It captures interference, boundary Friedel oscillations, and Peierls distortion through the Fermi-Dirac filling of orthogonal molecular orbitals.
- **[KekulePure](cci:2://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:26:0-417:17)** is a **semiclassical valence-bond optimizer**. Its constraint \(\sum_j p_{ij}=1\) is a real-space electron-counting rule that mimics the diagonal occupancy \(\rho_{ii}=1\), but it replaces the true quantum idempotency \(\sum_j \rho_{ij}^2=\rho_{ii}\) by a linear sum. It is most accurate for a single Kekulé structure; its aromatic 0.5 solution corresponds to a **mixture** of Kekulé projectors, not a single quantum density matrix.

---

# USER

yes I know they are not the same that is the point. I was thinkig if they may be two limits (two aspects) of same problem, and maybe they can e combined to obtain better theory. My view:
1) The BOP with local probes and Chebyshev or Lanczos to obtain local approximation of Huckel orbitals capture properly the pauli principle locally, but it may miss long-range correlations.
2) The classical bond order capture global conservation laws and the structures often look globally OK (e.g. topologucal transtion, Su–Schrieffer–Heeger model) it produce alternating pi-orbitals dimerization etc. (with proper parameters). For example if I indroduce topological defect (like one missing pi-orbitals due to hudrogenation of some atom) this effect non-locally propagate because it enforce localization of bonds on distant sites. This is property which is missing in BOP (computed only to low order- that is few scatterings)

So I was thinking we can combine the two to obtian proper estimate of bensitymatrix fullfilling both local and global constains. 

But for that we need to clarify exact relations between the properties at hand. What is quenmtum mechanical property related to our classical kekule bond order of atom and bond? Can it be related to elements of density matrix? Can we relate it to Hubard U (0,1,2; ocupation spin [00][p0][0d][pd]). What about electro field? Can we relax thr constain that bond orders must sum up to exactly 1 of we consider polarization of the pi-system (e.g. using classical charge equalibraion like QEq), so we can increase occupancy to certain site to lets say 1.5 instead of 1 but that would mean partical charge, and we would pay classical electrostatic penalty, or 0.8 would mean particla negative charge of the site. This is related to resonnat structures in pi-systems, for example phenol delocalize charge from O- to ortho and para position,  Or oxidative aromatization of phenzine/pyrzine ofthen create 4-bonded nitrongs delocalized in ppi-system like instead of hain=vng just -N= and -NH- we can have =NH(+)-

I would like to indroduce these aspects into the model by combination of my Classical global solver, and quantum local BOP solver. Any ideas about this? How to put this on rigorous grund?

---

# Kimi 2.6

Your intuition is exactly right. The two pieces are not “wrong” and “right”; they are **two complementary limits of the same correlated π‑electron problem**. A useful hybrid is absolutely possible.

## 1. What the classical Kekulé bond order is in quantum language

The classical [KekulePure](cci:2://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:26:0-417:17) optimizer works with a **bond graph** and an **atom valence law**:

\[
\sum_{j\sim i} p_{ij}=n_i
\]

This is encoded in the incidence matrix `A` in [KekuleBondSum.py](cci:7://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:0:0-0:0):

```@/home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:96-110
self._A = np.zeros((self.natom, self.nbond), dtype=float)
...
np.add.at(self._A, (self.i0, ib), 1.0)
np.add.at(self._A, (self.i1, ib), 1.0)
```

### Weak coupling (U=0): one-particle density matrix

For a noninteracting Hückel system the exact object is the **one-particle density matrix**

\[
\rho_{ij}=\langle c_i^\dagger c_j\rangle
\]

and the bond order is a linear functional of it. In the BOP files this is literally what is computed:

```@/home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleOrderN_Gemini_BOP_2D_v2.py:91-122
def get_bond_density_cheb(...):
    ...
    current_density += c[1] * v_next[target_site] * 2.0
    ...
```

For a single determinant of localized π bonds, the Kekulé variable maps directly as

\[
p_{ij}\;\approx\;2\rho_{ij}^{\text{spinless}} \;=\;\rho_{ij}^{\text{spinful}}
\]

and the linear sum rule

\[
\sum_j p_{ij}=1
\]

is the **classical, linearized version** of the diagonal occupancy condition

\[
\rho_{ii}=1 .
\]

The exact quantum relation is not linear, but **idempotent**:

\[
\rho^2=\rho \quad\Rightarrow\quad \sum_j |\rho_{ij}|^2=\rho_{ii}.
\]

So [KekulePure](cci:2://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:26:0-417:17) replaces the quadratic projector constraint by a linear Kirchhoff-like flow conservation. That is why it is so good at propagating global/topological constraints.

### Strong coupling (U→∞): singlet bond weight / dimer covering

When you introduce a Hubbard U, the relevant object is no longer the one-particle density matrix. The low-energy Hilbert space is spanned by local states:

| state | notation | n_i | double occupancy |
|---|---|---|---|
| empty | `\|`0⟩` | 0 | 0 |
| singly ↑ | `\|↑⟩` | 1 | 0 |
| singly ↓ | `\|↓⟩` | 1 | 0 |
| doubly | `\|↑↓⟩` | 2 | 1 |

For large U, the `\|↑↓⟩` state is penalized. The ground state lives in the **singly occupied** sector and can be written as a superposition of **singlet bonds** (valence bonds). The Kekulé weight `p_{ij}` is exactly the **probability / amplitude of a singlet bond between i and j**. This is the **resonating valence bond (RVB)** / dimer picture.

In that limit the linear sum rule

\[
\sum_j p_{ij}=1
\]

is exact: every singly occupied atom must participate in exactly one singlet bond. The [KekulePure](cci:2://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:26:0-417:17) localization snap to `p=0` or `p=1` is precisely the dimer-covering constraint.

So:

- **BOP ≈ weak-coupling, one-particle density matrix** (U=0)
- **KekulePure ≈ strong-coupling, singlet-bond / dimer model** (U→∞)
- A real molecule is somewhere in between

## 2. Hubbard U and the two limits

The BOP files compute the noninteracting density matrix

\[
\hat\rho=\Theta(\mu-\hat H)
\]

which is a projector. This is the **Pauli principle** in density-matrix form: eigenvalues are 0 or 1. The local probes capture this **locally** but, because they are truncated at low order, they do not propagate the global “no-double-occupancy” constraint that a Hubbard U imposes.

[KekulePure](cci:2://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:26:0-417:17) does the opposite: it imposes global conservation and singlet bonding, but it contains no local quantum interference. It can capture a **topological defect** (e.g. an sp³ atom with `n_i=0`) because removing one atom from the graph changes the global constraint `A p = n` and forces a reorganization of double bonds far away.

## 3. Electric field, polarization, and variable charge

Yes, the rigid rule

\[
\sum_j p_{ij}=1
\]

can be relaxed. Replace it with a **variable occupancy** constraint

\[
\sum_j p_{ij}=n_i
\]

where `n_i` is the local π charge on atom i. Then add a classical electrostatic/QEq energy:

\[
E_{\text{es}}(\{n_i\}) = \sum_i \chi_i n_i
+ \frac12\sum_{ij} J_{ij}(n_i-n_i^0)(n_j-n_j^0)
+ \sum_i e\,\mathbf E\cdot\mathbf r_i\,n_i
\]

with a global charge constraint

\[
\sum_i n_i = N_\pi .
\]

The quantum BOP enters because the on-site energies in the Hamiltonian must depend on the charges:

\[
H_{ii} = \varepsilon_i^0 + U(n_i-n_i^0) + e\mathbf E\cdot\mathbf r_i .
\]

This is the same self-consistent charge (SCC) loop used in DFTB. In your setting:

- **Phenolate**: O has `n_O≈2`, the excess charge is distributed to ortho/para carbons (`n_i≈1-δ`).
- **Protonated pyridine / N-oxides**: N can be positively charged (`n_N<1`) or neutral.
- **External field**: shifts `H_{ii}`, polarizes the density matrix, and the QEq term balances the electrostatic cost.

The [KekuleBondSum.py](cci:7://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:0:0-0:0) class already accepts `n_pi` as an array, so extending it to variable `n_i` is natural:

```@/home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:69-75
if n_pi is None:
    ...
    n_pi = np.where(pi_atoms, 1.0, 0.0)
self.n_pi = np.asarray(n_pi, dtype=float)
```

## 4. A rigorous combined framework

The most rigorous way to view this is as a **variational problem on the one-particle density matrix** plus a **local Hubbard correlation** term. The exact ground-state (for a given Hamiltonian) is

\[
\min_{\rho} \mathrm{Tr}[H\rho] + E_U[\rho]
\quad\text{s.t.}\quad
\rho^2=\rho,\ \rho^\dagger=\rho,\ \mathrm{Tr}\rho=N_\pi,
\]

where `E_U` is the interaction energy. In a mean-field / Hubbard-I approximation, `E_U` penalizes double occupancy.

The BOP and [KekulePure](cci:2://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:26:0-417:17) are two different approximations to this problem:

- BOP approximates the **projector** `\rho^2=\rho` by a truncated Fermi operator expansion.
- [KekulePure](cci:2://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:26:0-417:17) approximates the same projector by a **linear flow constraint** on bond variables.

A hybrid can be written as an **augmented Lagrangian / penalty functional**:

\[
\mathcal F[\rho,\{p_{ij}\},\{n_i\}]
=
\mathrm{Tr}[H(n)\rho]
+ E_{\text{es}}(n)
+ E_U(n)
+ \frac{\lambda}{2}\sum_{(ij)}\big(p_{ij}-f(\rho_{ij})\big)^2
+ \frac{\kappa}{2}\sum_i\big(\sum_j p_{ij}-n_i\big)^2
+ \frac{\mu}{2}\|\rho^2-\rho\|^2
\]

Here:

- `p_{ij}` are the Kekulé bond-order variables.
- `f(\rho_{ij})` maps the local quantum density to a bond order (e.g. `f=ρ` for spinful density).
- The λ-term forces the Kekulé variables to agree with the BOP locally.
- The κ-term enforces the global valence conservation.
- The μ-term restores idempotency / Pauli principle.

This is rigorous as a **variational approximation**; the quality is controlled by `M` (BOP truncation) and the penalty strengths.

### Practical hybrid algorithm

A simple, force-field-like implementation:

1. **Initialize** charges `n_i` from atom types or QEq.
2. **Build** `H(n)` with charge-dependent on-site energies.
3. **BOP step**: compute a local low-order density matrix `\rho^{\text{BOP}}` and extract priors `p_{ij}^{\text{BOP}}=f(\rho_{ij}^{\text{BOP}})`.
4. **KekulePure step**: minimize

   \[
   E_{\text{Kekule}} = E_{\text{val}} + E_{\text{aro}} + E_{\text{loc}}
   + \lambda\sum(p_{ij}-p_{ij}^{\text{BOP}})^2
   + E_{\text{es}}(n)
   + E_U(n)
   \]

   subject to `A p = n`, `0≤p≤1`, `∑ n_i = N_π`.
5. **Update** charges `n_i = \rho_{ii}` from the BOP (or from the Kekule solution).
6. **Iterate** 2–5 until self-consistency.

This gives you:

- Local quantum interference from BOP.
- Global topological / valence conservation from [KekulePure](cci:2://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:26:0-417:17).
- Charge polarization and heteroatom effects from QEq + U.
- The ability to handle missing π orbitals by setting `n_i=0`.

### Topological defects

A hydrogenated atom is simply a site with `n_i=0` removed from the graph. In [KekulePure](cci:2://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:26:0-417:17), this is automatically enforced by the incidence matrix:

```@/home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:312-340
def solve_constrained(self, target=None, ...):
    ...
    err = self._A @ self.bo - self.n_pi
```

The constraint `A p = n` then propagates the defect globally: the remaining atoms must satisfy their valence with fewer neighbors, producing the long-range Kekulé reorganization that a low-order BOP misses. You can use the BOP only to provide the **local quantum shape** around each bond, while [KekulePure](cci:2://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:26:0-417:17) supplies the **global compatibility**.

### A note on OMM

The [OMM_1D_grid.py](cci:7://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/OMM/OMM_1D_grid.py:0:0-0:0) you are editing also solves the local quantum problem, but in a more “global” way: it minimizes localized orbitals while enforcing orthogonality. In the same hybrid spirit, OMM could replace the BOP as the local quantum solver, and [KekulePure](cci:2://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:26:0-417:17) could be added as a **penalty on the OMM density matrix** to enforce the global valence / dimer constraints. The conceptual framework is the same.

## Bottom line

- **BOP** is the weak-coupling, one-particle density-matrix limit.
- **KekulePure** is the strong-coupling, singlet-bond / dimer-covering limit.
- The classical bond-order sum `\sum_j p_{ij}=n_i` is the linearized version of the quantum conservation law, exact in the large-U RVB limit.
- You can relax `n_i` and add QEq + Hubbard U to describe polarization, heteroatoms, and charged resonance structures.
- The rigorous combined model is a **variational density-matrix functional** with local quantum priors and global valence/electrostatic constraints; the practical algorithm is a **self-consistent BOP → KekulePure → charge update loop**.

That gives you the best of both: local Pauli/interference physics from the BOP, and global topological/defect propagation from the classical Kekulé solver.

---

# USER

OK, I need to clarify few things here:
You speak about kirchhoff's law and flow contrains. But I do not see how this flow constrain (closed current loops right?) is related to my constiant that number of bonds sum-up to 1 (resp npi,  npi_i = sum_j bo_ij ). I would see it clealy if these would be oriented bonds - incomming and outgoing charge, but these are not oriented bonds. Maybe we can somehow relate it to assumption that we have no magnetic field so we have time-reversal symmetry and pi orbitasl ring currents in clockwise and anticlockwise direection are the same ( like we can crete linear cobination of ring currect which are complex conjugates lie +k and -k bloch wave in solid state physics, and assume they are the same), but still I do not see how this give the rule all bonds sum to 1?
Maybe if we consider conductivity quantum G0 that every oribital can carry max 1 electron at a time, so we have unit quantum conductance per orbital. Our single pi-orbitals can be member of different channles (different ring currents, or different orbitals - these are all just different pojections/rotations) but in total sum of its contribution to all those channels cannot be more than 1. That is the idenpotency? Sorry I may talk to confusing way, I need to make this clear and rigorous. Make order in this brainstorm.

---

# Kimi 2.6

You are right to push back on the wording. The Kekulé constraint is **not** a Kirchhoff current law and it is **not** about flow loops, ring currents, or Bloch waves. Those are separate ideas. Let me clean up the brainstorm.

## 1. What the constraint actually is

In [KekuleBondSum.py](cci:7://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:0:0-0:0) the incidence matrix is built like this:

```@/home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:96-110
self._A = np.zeros((self.natom, self.natom), dtype=float)
...
np.add.at(self._A, (self.i0, ib), 1.0)
np.add.at(self._A, (self.i1, ib), 1.0)
```

Both endpoints of a bond get **+1**. This is the incidence matrix of an **undirected** graph, not an oriented one. Therefore the equation

\[
A\,p = n
\]

means

\[
\sum_{j\sim i} p_{ij} = n_i
\]

i.e. the **sum of the weights of all bonds touching atom i equals the local π-electron count n_i**. This is a **weighted-degree / resource-allocation** constraint, not a current-conservation law.

A true Kirchhoff current law would use an **oriented** incidence matrix with entries `+1` and `-1`, and it would enforce zero net flow into each node:

\[
\sum_{\text{outgoing}} f - \sum_{\text{incoming}} f = 0 .
\]

That is a completely different statement. The Kekulé rule does not involve directions, loops, or circulation.

## 2. Where the sum rule comes from

The rule is a **classical electron-counting / valence-bond rule**, not a consequence of time-reversal symmetry or ring currents.

For a neutral sp² carbon in a π system, there is one π electron. In a perfect Kekulé structure that electron is shared in one π bond (double bond) with a neighbor. So each carbon has:

- one bond with `p = 1`
- the other bonds with `p = 0`
- total `Σ_j p_{ij} = 1`

This is the **hard-core dimer-covering constraint**: each atom must be “covered” by exactly one π bond. In graph theory, the integer version is a **perfect matching** (or more generally a **b-matching** when `n_i` can differ from 1). The [KekulePure](cci:2://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:26:0-417:17) optimizer relaxes this to fractional `p ∈ [0,1]`, making it a **fractional matching**.

The aromatic `p = 0.5` solution is then the **average** over many such dimer coverings. For each individual covering the linear sum rule holds; in the averaged picture the same rule still holds per atom:

\[
\left\langle \sum_{j\sim i} p_{ij} \right\rangle = 1
\]

but an individual bond is no longer 0 or 1.

## 3. Relation to idempotency

The exact quantum relation is not linear. For the zero-temperature one-particle density matrix:

\[
\rho^2 = \rho
\quad\Longrightarrow\quad
\sum_j |\rho_{ij}|^2 = \rho_{ii} = n_i .
\]

This is the **idempotency** condition. The Kekulé sum rule

\[
\sum_j p_{ij} = n_i
\]

is a **linear relaxation** of this. They are equivalent only in the **binary limit** `p_{ij} ∈ {0,1}`, because then

\[
p_{ij}^2 = p_{ij}
\]

and the linear sum equals the quadratic sum. For a single Kekulé structure, the classical constraint is exactly the Boolean version of idempotency.

For an aromatic 0.5 bond, this is no longer true:

\[
\sum_j p_{ij} = 1,
\qquad
\sum_j p_{ij}^2 = 2 \times (0.5)^2 = 0.5 \neq 1 .
\]

So the 0.5 aromatic state is **not** a single idempotent density matrix. It is an **ensemble average** of binary idempotent projectors. The linear Kekulé rule captures the average electron count, but not the projector property.

## 4. Ring currents, Bloch waves, and time reversal

These are not the source of the sum rule.

- **Time-reversal symmetry** makes the Hamiltonian real and the density matrix real and symmetric. This is why `p_{ij}` is undirected and real.
- **Standing waves** are superpositions of `+k` and `-k` Bloch states. They produce real, symmetric bond densities, not currents.
- **Ring currents** only appear if you break time-reversal symmetry, e.g. with a magnetic field, and then the density matrix becomes complex and the bond order has an imaginary part related to the current.

The Kekulé sum rule is about **electron counting**, not about interference, currents, or Bloch phases.

## 5. The “conductance channel” analogy

The idea that “each orbital can carry at most one electron, so the sum of its contributions to all channels cannot exceed 1” is related to the **Pauli exclusion principle**, not to idempotency directly.

In the local spin-orbital picture, each p_z orbital can be:

| state | occupancy | double occupancy |
|---|---|---|
| empty | 0 | 0 |
| singly ↑ or ↓ | 1 | 0 |
| doubly ↑↓ | 2 | 1 |

The Pauli principle says you cannot put more than one electron of the same spin into the same orbital. The Kekulé rule `Σ_j p_{ij} = 1` is a **classical projection** of this for the singly-occupied case: one electron at site i must be paired with exactly one neighbor.

This is not a quantum conductance statement. It is a **valence-bond / dimer counting** statement.

## Summary

| idea | correct interpretation | relation to Kekulé sum |
|---|---|---|
| `A p = n` in [KekuleBondSum.py](cci:7://file:///home/prokophapala/git/NumericalMathPlayground/topics/LinearScalingQM/Kekule_BOP/KekuleBondSum.py:0:0-0:0) | weighted graph degree / fractional matching | this is the actual constraint |
| Kirchhoff current law | oriented incidence, zero divergence | **not** what the code does |
| time-reversal symmetry | makes `ρ` real and symmetric | explains why bonds are undirected, not why they sum to 1 |
| Bloch `+k/-k` standing waves | real bond densities, no current | unrelated to the sum rule |
| ring currents | require broken time-reversal / magnetic field | unrelated to static Kekulé rule |
| idempotency `ρ² = ρ` | exact projector relation `Σ_j ρ_{ij}² = n_i` | Kekulé sum is its **linear, Boolean approximation** |
| Pauli / conductance channel | one electron per spin orbital | classical motivation for `Σ_j p_{ij} = 1` |

So the rigorous statement is:

> The Kekulé bond-order sum rule is a **linear, classical resource-allocation constraint** on an undirected graph. It is the Boolean/strong-coupling relaxation of the quantum idempotency relation `Σ_j |ρ_{ij}|² = n_i`. It is exact for a single Kekulé dimer covering, and it becomes an ensemble average when aromatic `p = 0.5` bonds are allowed. It has no connection to Kirchhoff currents, ring currents, or Bloch-wave channels unless you add an external magnetic field and switch to a complex, current-carrying description.

---

# USER

Fuck: stop spliting hairs and start searching connections! Yes I want to put it to rigorous ground and we need define things clearly. But most importntly we need o see what is the connection beteen ouir classical proeprteis and the quentum one. 

1) I very like the idea that our classical bond order is ensamble average of many single-determinant solutions resp. like configuration interaction (CI) methods. But density matrix can be experessed also for those (pure state ves mixed state) right? 
2) But what is the actuall reason (quantum mechanical justification) why it make sense to conserve bond order sume for each atom =1 ? Is that somethig like statistical normalization from the slater determinants in the mix ? like normalization of total probability that electron is somewhere?
3) you say for aromaic system we get 0.5, that is not true. We get 0.5 for bonds, but not 0.5 for antom. Atoms are still at 1. We should distinguis that. 
4) I think the conservation of current flow and Kirchofs law is worth exploring, and try to see how it can be related to our kekule structures and conservation of bond order. In absence of magnetic field we can conserder that our clockwise and anticlockwise curent are same, so we always have bonds which carry same current to and from the atom. Nevertheless, we should assume that kirchofs law is tru for each molecular orbital separately not only in the sum so we should consider that if we have some orbital which carry .8 to atom by one bond, it must carry -.8 out by the other bonds. And at the same time sum of all these orbitals together give total curent 1 over that atom. I do not want to solve for all these current channels (all the kekule structures or slater determinants) as it would be as costly as full diagonaliation of hamiltonina (or even worse, like CI), what I want is to use this insight to define proper effective model which fix physically menaingfull ocnstrain on this ensamel average withoiut actually solving for the microstates. It would be also great if we can estimate entropy of the system, because I think the delicalization (aromatization) is kinda like entropic effect. And so we have competition between enthalpy or internal energy given by constrain (e.g. boundary condtion enforcing dimarization/localization like SSH-model) versus the entropic contribution promoting resonant structures. Can this be quentiofied within this framework?
5) also you mentioned that the boolen constain 0,1 is linear while idepmotency is quadratinc, and it works only for 0,1 but not at 0.5^2=0.25 right?
So how it should be properly? Especially if we thingn that we can allow npi != 1 ( like 1.5 for negatively charged atom, -0.8 for positively charged atom)
6) I'm not sure if that idea to allow npi != 1 is physically correct. Because we should justify what we mean by npi. 
a) is this number of electrons on given site? In that case it can be anything between [0.0 to 2.0] (two electron with different spin), and it is completely fine, and it depend on electrostatics (hubard U? and QEq)
b) or is this number of orbitals (aka room where the electrons live, room in hilbert space), in that case the sum rule bond order sum to 1 is expression of pauli principle and flow conservation. 
We should distingus this, and somehow make sense of the fact that we jhave only single value npi which express both.

Instead of pinpotinty litte inaccuracies in what I say try to find connection, and formulate it beter, how it can be put on solid ground and used for something.

---

