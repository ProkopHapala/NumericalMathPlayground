# USER

I would like to finally understand superconductivity and with my molecular chemistry background the only way is plot it in real space. In superconductor theory you typically show correlated motion of electron pair and hole right? Now they are both from superconductive gap edges (you have kinda flat band there, and spatially localized bunch of electron spin pair and hole is moving with some group velocity.

In localized picture in 1D it can be visible as two localized orbitals with n and n+1 nodes oscillating with different Eigenfrequenzy. And together with them moves polaron (?) or maybe that polaron is static?

Basically I want to plot 1D tight binding real space picture how looks superconductive Transport (moving Wavepocket of electron density) as linear combination of molecular orbitals coupled by some electron phonon coupling. Wich should represent established theory of superconductivity.

---

# ChatGPT 5.5

Yes — but the key correction is: **superconducting transport is usually not best visualized as a moving lump of electron density.** A supercurrent is mostly a **phase-gradient flow of the Cooper-pair condensate**, while the moving “electron–hole pair” object is a **Bogoliubov quasiparticle wavepacket**. That quasiparticle has both electron-like and hole-like amplitudes, but it is an *excitation above* the superconducting ground state, not the condensate current itself.

The clean real-space picture to plot is therefore:

[
\Psi(t)=
\begin{pmatrix}
u_i(t)\
v_i(t)
\end{pmatrix},
]

where (u_i) is the electron amplitude on site (i), and (v_i) is the hole amplitude. This is the **Bogoliubov–de Gennes** picture.

---

## 1. Start from normal 1D tight binding

Take a 1D chain:

[
H_0 =
-t\sum_{i,\sigma}
\left(
c^\dagger_{i+1,\sigma} c_{i,\sigma}
+
c^\dagger_{i,\sigma} c_{i+1,\sigma}
\right)
-\mu\sum_{i,\sigma} c^\dagger_{i,\sigma}c_{i,\sigma}.
]

The normal eigenstates are molecular-orbital-like standing waves in a finite chain, or Bloch waves in an infinite chain.

For a finite open chain, you indeed get orbitals with (n), (n+1), (n+2) nodes. A localized electron wavepacket can be made by superposing many of them. Two adjacent orbitals alone give a beating pattern, but a proper moving packet needs a narrow band of orbitals.

---

## 2. Add superconductivity: real-space BdG Hamiltonian

For a simple (s)-wave BCS superconductor, add local pairing:

[
H_\Delta =
\sum_i
\left(
\Delta_i c^\dagger_{i,\uparrow}c^\dagger_{i,\downarrow}
+
\Delta_i^* c_{i,\downarrow}c_{i,\uparrow}
\right).
]

In Nambu form,

[
\Phi_i =
\begin{pmatrix}
c_{i,\uparrow}\
c^\dagger_{i,\downarrow}
\end{pmatrix},
]

the BdG Hamiltonian is

[
H_\text{BdG}
============

\begin{pmatrix}
H_0 & \Delta\
\Delta^* & -H_0^*
\end{pmatrix}.
]

For a uniform chain with real constant (\Delta),

[
H_\text{BdG}
============

\begin{pmatrix}
H_0 & \Delta I\
\Delta I & -H_0
\end{pmatrix}.
]

Each quasiparticle eigenstate is not just an electron orbital. It has the form

[
\gamma^\dagger_n
================

\sum_i
u_{n,i} c^\dagger_{i,\uparrow}
+
v_{n,i} c_{i,\downarrow}.
]

So in real space you should plot both:

[
|u_i(t)|^2
]

and

[
|v_i(t)|^2.
]

The quasiparticle charge density is roughly

[
\rho_i(t)
=========

|u_i(t)|^2 - |v_i(t)|^2.
]

Near the superconducting gap edge, (u) and (v) are similar in magnitude, so the excitation may carry very little net charge even though it has strong electron and hole components.

---

## 3. What “moves” in a superconducting wavepacket?

For a translationally invariant chain, the BdG spectrum is

[
E_k =
\sqrt{\xi_k^2 + |\Delta|^2},
]

with

[
\xi_k = -2t\cos k - \mu.
]

The group velocity is

[
v_g(k)
======

# \frac{dE_k}{dk}

\frac{\xi_k}{E_k}
\frac{d\xi_k}{dk}.
]

This is important.

At the Fermi level, (\xi_k=0), so

[
E_k = \Delta,
]

and

[
v_g = 0.
]

So the **exact gap-edge excitation is actually slow / flat**. A wavepacket slightly above the gap moves, but its velocity is reduced by the factor

[
\frac{\xi_k}{E_k}.
]

This is why the superconducting excitation is not just a normal electron packet. It is a mixed electron-hole packet.

---

## 4. Electron–hole oscillation: the thing you probably want to see

For a fixed (k), the BdG Hamiltonian is a two-level system:

[
H_k =
\xi_k \tau_z + \Delta \tau_x.
]

If you inject a pure electron-like state, it evolves as a mixture of electron and hole:

[
\begin{pmatrix}
u_k(t)\
v_k(t)
\end{pmatrix}
=============

e^{-iH_k t/\hbar}
\begin{pmatrix}
1\
0
\end{pmatrix}.
]

At (\xi_k=0), this becomes a clean oscillation:

[
u_k(t) = \cos(\Delta t/\hbar),
]

[
v_k(t) = -i\sin(\Delta t/\hbar).
]

So an injected electron periodically turns into a hole-like excitation and back. This is closely related to **Andreev physics**.

In real space, a packet will do both:

1. move with BdG group velocity,
2. internally oscillate between electron and hole character.

That is probably the most visual and intuitive “superconducting transport” movie to make.

---

## 5. Where is the phonon or polaron?

In standard BCS theory, you do **not** usually draw a static polaron moving together with the pair.

The phonons produce an effective retarded attraction:

[
V_\text{eff}(\omega)
\sim
-\frac{g^2 \omega_D^2}{\omega^2+\omega_D^2},
]

which favors pairing of time-reversed states:

[
(k,\uparrow),(-k,\downarrow).
]

After mean-field reduction, the phonon field is hidden inside the pairing gap (\Delta). The superconducting condensate is described by the anomalous expectation value

[
F_i =
\langle c_{i,\downarrow}c_{i,\uparrow}\rangle,
]

and

[
\Delta_i = -g F_i.
]

So in a minimal established BCS/BdG plot, the phonon does not appear as a moving lattice deformation. The order parameter (\Delta_i) is usually static.

A real-space lattice deformation moving with the charge is more like a **polaron** or **bipolaron** picture. That is relevant in strong electron-phonon coupling systems, Peierls chains, CDW systems, Holstein models, etc., but it is not the simplest conventional BCS picture.

So I would separate two visual models:

### Conventional BCS/BdG picture

Plot:

[
|u_i(t)|^2,\quad |v_i(t)|^2,\quad |u_i(t)|^2-|v_i(t)|^2,\quad F_i.
]

Here (\Delta) is imposed or self-consistent but usually static.

### Strong electron-phonon / polaronic picture

Add a lattice displacement (Q_i) and let it couple to density:

[
H_\text{e-ph}
=============

g\sum_i Q_i n_i
+
\frac{K}{2}\sum_i Q_i^2.
]

Then solve self-consistently:

[
Q_i \sim -\frac{g}{K}\langle n_i\rangle.
]

That gives a local lattice deformation. But that is no longer pure textbook weak-coupling BCS. It is more like Holstein superconductivity / bipolaron physics.

---

## 6. Minimal Python model to plot real-space BdG wavepacket

This is the cleanest starting point. It creates a 1D chain, builds a BdG Hamiltonian, injects a Gaussian electron wavepacket, and evolves it in time.

```python
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
N = 300          # number of sites
t = 1.0          # hopping
mu = 0.0         # chemical potential
Delta = 0.15     # superconducting gap
x0 = N//3        # initial packet center
sigma = 18.0     # packet width
k0 = np.pi/2 + 0.35   # central wavevector, slightly above Fermi point
dt = 0.5
nsteps = 240

# -----------------------------
# Normal tight-binding Hamiltonian
# -----------------------------
H0 = np.zeros((N, N), dtype=complex)

for i in range(N):
    H0[i, i] = -mu
    if i + 1 < N:
        H0[i, i+1] = -t
        H0[i+1, i] = -t

# -----------------------------
# BdG Hamiltonian
#
# [ H0      Delta I ]
# [ Delta I  -H0*   ]
# -----------------------------
BdG = np.zeros((2*N, 2*N), dtype=complex)

BdG[:N, :N] = H0
BdG[N:, N:] = -H0.conjugate()
BdG[:N, N:] = Delta * np.eye(N)
BdG[N:, :N] = Delta * np.eye(N)

# -----------------------------
# Initial electron-like wavepacket
# u_i = Gaussian * exp(i k0 x)
# v_i = 0
# -----------------------------
x = np.arange(N)

u0 = np.exp(-(x - x0)**2 / (2*sigma**2)) * np.exp(1j*k0*x)
u0 /= np.sqrt(np.sum(np.abs(u0)**2))

v0 = np.zeros(N, dtype=complex)

psi0 = np.concatenate([u0, v0])

# -----------------------------
# Diagonalize BdG once
# -----------------------------
E, V = np.linalg.eigh(BdG)

coeff = V.conjugate().T @ psi0

# -----------------------------
# Time evolution
# psi(t) = V exp(-iEt) V† psi(0)
# hbar = 1 units
# -----------------------------
frames = []

for istep in range(nsteps):
    time = istep * dt
    phase = np.exp(-1j * E * time)
    psi = V @ (phase * coeff)

    u = psi[:N]
    v = psi[N:]

    ne = np.abs(u)**2
    nh = np.abs(v)**2
    charge = ne - nh

    frames.append((time, ne, nh, charge))

# -----------------------------
# Plot selected snapshots
# -----------------------------
for time, ne, nh, charge in frames[::40]:
    plt.figure(figsize=(10, 4))
    plt.plot(x, ne, label=r"electron component $|u_i|^2$")
    plt.plot(x, nh, label=r"hole component $|v_i|^2$")
    plt.plot(x, charge, label=r"net charge $|u_i|^2-|v_i|^2$", linestyle="--")
    plt.title(f"BdG wavepacket, t = {time:.1f}")
    plt.xlabel("site i")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.show()
```

This will show the electron packet partially converting into a hole packet while propagating.

---

## 7. What you should expect to see

If (\Delta=0), you get an ordinary tight-binding electron wavepacket.

If (\Delta>0), the packet splits into electron and hole character. The electron and hole densities are not independent particles. They are the two Nambu components of one Bogoliubov excitation.

The net charge density,

[
|u_i|^2-|v_i|^2,
]

can become much smaller than either component separately.

That is a very superconducting effect: the excitation is partly electron, partly missing electron.

---

## 8. To visualize Cooper pairs themselves

The wavepacket above visualizes a **quasiparticle excitation**. To visualize the **Cooper pair condensate**, plot the anomalous density matrix:

[
F_{ij}
======

\langle c_{i,\downarrow} c_{j,\uparrow} \rangle.
]

In uniform BCS,

[
F_{ij}
======

\frac{1}{N}
\sum_k
u_k v_k^*
e^{ik(i-j)}.
]

For (s)-wave pairing,

[
u_k v_k
=======

\frac{\Delta}{2E_k}.
]

So:

[
F(r)
====

\frac{1}{N}
\sum_k
\frac{\Delta}{2\sqrt{\xi_k^2+\Delta^2}}
e^{ikr}.
]

This is the real-space Cooper-pair wavefunction. In weak-coupling BCS, it is very extended:

[
\xi_\text{coh}
\sim
\frac{\hbar v_F}{\Delta}.
]

So unlike a chemical bond, a Cooper pair is usually huge in real space, often many lattice constants or more.

This is the object to plot if you want the “paired electron correlation cloud”.

---

## 9. Minimal code for Cooper-pair correlation in real space

```python
import numpy as np
import matplotlib.pyplot as plt

N = 4096
t = 1.0
mu = 0.0
Delta = 0.05

k = 2*np.pi*np.arange(N)/N - np.pi
xi = -2*t*np.cos(k) - mu
E = np.sqrt(xi**2 + Delta**2)

# anomalous amplitude in k-space
Fk = Delta / (2*E)

# real-space pair function F(r)
Fr = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Fk)))
r = np.arange(-N//2, N//2)

plt.figure(figsize=(10, 4))
plt.plot(r, np.real(Fr))
plt.xlim(-300, 300)
plt.xlabel("relative separation r")
plt.ylabel("pair amplitude F(r)")
plt.title("Real-space Cooper-pair correlation")
plt.tight_layout()
plt.show()
```

This is maybe the most chemically meaningful plot: it shows that the Cooper pair is not a local two-electron bond. It is a long-range correlated pair spread over many sites.

---

## 10. The chemically intuitive summary

In a molecule, you often think:

[
\text{electron occupies orbital } \phi_n.
]

In a superconductor, the ground state is closer to:

[
\prod_k
\left(
u_k + v_k c^\dagger_{k\uparrow}c^\dagger_{-k\downarrow}
\right)
|0\rangle.
]

So each time-reversed orbital pair is in a coherent superposition:

[
\text{empty pair}
\quad+\quad
\text{occupied spin pair}.
]

That is the essential superconducting object.

A Bogoliubov quasiparticle is then not just “an electron”. It is:

[
\text{electron added}
\quad+\quad
\text{electron removed}.
]

That is why the real-space BdG plot should show two fields:

[
u_i(t), v_i(t).
]

And the condensate should be shown through:

[
F_{ij}=\langle c_{i\downarrow}c_{j\uparrow}\rangle.
]

For your molecular intuition, I would make three animations:

1. **Normal electron packet**: (\Delta=0).
2. **BdG quasiparticle packet**: (\Delta>0), plot (|u|^2), (|v|^2), and charge.
3. **Cooper-pair correlation**: plot (F(i-j)), showing the spatial size of the pair.

That gives the most honest real-space bridge between tight-binding molecular orbitals and established superconductivity.

---

# Gemini

Your intuition about the electron-hole mix and the oscillating nodes is a beautiful way to visualize **Bogoliubov quasiparticles** (the elementary excitations of a superconductor). At the superconductive gap edge, an excitation is exactly a 50/50 linear combination of an electron-like molecular orbital (MO) and a hole-like MO.

However, mapping this directly to **superconductive transport** requires sorting out a few common points of confusion regarding the ground state versus its excitations, and the difference between weak-coupling BCS pairs and real-space polarons.

---

## 1. Ground State Transport vs. Quasiparticle Wavepackets

If you plot a steady DC supercurrent in real space, you will not see a localized wavepacket of electron density rushing down the 1D chain. The electron density $\rho(x)$ remains completely **uniform**.

* **The Condensate (Transport):** In the BCS ground state, all Cooper pairs occupy the exact same macroscopic quantum state. To get a supercurrent, you don't excite particles to the gap edge; instead, you give the *entire ground state* a collective momentum boost $2q$. In real space, this means the superconducting order parameter acquires a phase gradient: $\Delta(x) = \Delta_0 e^{i 2q x}$. Every single electron MO in the filled Fermi sea gets a phase factor $e^{i q x}$. Charge moves, but density stays flat.
* **The Gap Edge (Excitations):** The localized wavepacket you described—built from states at the gap edges—is a **Bogoliubov quasiparticle**. If you create a wavepacket of these, you are visualizing an *excited state* moving through the superconductor. In a two-fluid model, this corresponds to the "normal fluid" component, which actually carries entropy and experiences dissipation when it scatters.

## 2. Where Does the Polaron Fit In?

Whether the lattice distortion (polaron) is static or moves depends entirely on the coupling regime:

* **Weak-Coupling (Standard BCS):** The electron-phonon coupling is dynamic and retarded. An electron passes by, momentarily pulls the ions inward, and leaves behind a localized region of excess positive charge. A fraction of a microsecond later, a second electron is attracted to that positive wake. There is **no static or co-moving physical polaron cloud** bound to the pair; the pairing is mediated by *virtual* phonons.
* **Strong-Coupling (Bipolaron Picture):** If the electron-phonon coupling is extremely strong (like in the Holstein model), the lattice deforms deeply around the electrons. Two electrons can trap themselves inside a shared local lattice distortion, forming a **bipolaron**. In this real-space picture, if the pair moves, the polaron deformation *must* drag along with them. This severely increases their effective mass, making the wavepacket move with a very slow group velocity.

---

## 3. The Real-Space Tight-Binding Picture (BdG)

To plot this, the standard tool is the **Bogoliubov-de Gennes (BdG)** Hamiltonian. For a 1D tight-binding chain of $N$ sites, instead of a standard Hückel/tight-binding matrix, you double the basis to account for electrons and holes (Nambu space).

The Hamiltonian matrix looks like this:

$$H_{BdG} = \begin{pmatrix} \hat{\mathcal{H}}_e & \hat{\Delta} \\ \hat{\Delta}^\dagger & -\hat{\mathcal{H}}_h \end{pmatrix}$$

Where:

* $\hat{\mathcal{H}}_e$ is your standard 1D tight-binding matrix: diagonal elements are $-\mu$ (chemical potential), and off-diagonals are $-t$ (hopping).
* $-\hat{\mathcal{H}}_h$ is the hole block (the time-reverse of the electron block).
* $\hat{\Delta}$ is a diagonal matrix where $\Delta_{jj}$ represents the local superconducting pairing amplitude on site $j$.

When you diagonalize this matrix, each eigenstate is a spinor $\Psi_n = \begin{pmatrix} u_n(x) \\ v_n(x) \end{pmatrix}$, where $u_n(x)$ is the electron MO component and $v_n(x)$ is the hole MO component.

---

## Interactive 1D BdG Explorer

The widget below simulates a 1D periodic tight-binding chain using the Bogoliubov-de Gennes framework. You can dial in the pairing gap ($\Delta$), shift the Fermi level ($\mu$), and add a supercurrent phase gradient ($q$).

By clicking on the energy states near the gap edge, you can see exactly how the electron ($u$) and hole ($v$) amplitudes construct the molecular orbitals in real space. Notice how at the gap edge, the electron and hole densities perfectly track each other, differing only by their spatial phase or nodal structures.

---

## How to build your moving wavepacket

If you want to construct the exact moving wavepacket you envisioned in your own scripts (e.g., in Python or Rust), here is the recipe to superimpose these MOs:

1. **Select a range of states** right at the positive gap edge ($E_k \approx \Delta$).
2. For each state $k$ in that window, extract its electron amplitude $u_k(x)$ and hole amplitude $v_k(x)$.
3. **Construct a Gaussian envelope** in $k$-space centered around some carrier momentum $k_0$:

$$c_k = e^{-(k-k_0)^2 / (2\sigma^2)}$$


4. **Sum them up** to form the real-space time-dependent wavepacket:

$$\Psi_e(x, t) = \sum_k c_k u_k(x) e^{-i E_k t / \hbar}$$


$$\Psi_h(x, t) = \sum_k c_k v_k(x) e^{-i E_k t / \hbar}$$



Because $E_k = \sqrt{\epsilon_k^2 + \Delta^2}$, the dispersion relation is incredibly flat near the gap edge ($dE/dk \to 0$ as $\epsilon_k \to 0$). As a result, your quasiparticle wavepacket will have a massive effective mass and will propagate highly slowly, showcasing the tight spatial correlation of the electron-hole mix.