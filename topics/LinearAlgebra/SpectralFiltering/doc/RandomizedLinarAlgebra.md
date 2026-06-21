https://gemini.google.com/share/dd24c3f67fd8

---

# USER

I heard about SVD use in machine learning for LoRA etc and that it is using some stochastic algorithm of random projection to approximate SVD as full SVD of such large matrix is costly.

But SVD is something like diagonalization (related not the same)

So I was thinking if similar method can be used for diagonalization (finding eigenvectors) in chemistry and material science for large systems.

And maybe it is related to probe methods used to estimate density of states by solving linear response to random probes witch is much faster for large sparse matrix than full inversion or diagonalization

---

# Gemini

You have hit on a brilliant connection. The short answer is **yes, absolutely**—the exact same core philosophy behind Randomized SVD is used in quantum chemistry and materials science to avoid the brutal $O(N^3)$ cost of full diagonalization.

In fact, the relationship between these random projection methods and the "random probe" methods for Density of States (DOS) is incredibly tight. They are both branches of the same mathematical tree: **Randomized Numerical Linear Algebra (RandNLA)**.

Here is how these concepts connect, where they diverge, and how they are used in electronic structure calculations.

## 1. Randomized SVD vs. Randomized Diagonalization

You are entirely right that SVD ($A = U \Sigma V^T$) and Eigendecomposition ($A = X \Lambda X^{-1}$) are close cousins.

* For a symmetric/Hermitian matrix $H$ (like a Hamiltonian), the SVD and the eigendecomposition are practically identical ($U=V=X$, and the singular values are the absolute values of the eigenvalues).

Because of this, the **Randomized Range Finder** algorithm used in LoRA can be adapted directly to find eigenvectors.

### How it works (The Randomized Subspace Iteration)

If you want the lowest (or highest) $k$ eigenvalues of a massive Hamiltonian $H$:

1. **Random Projection:** You multiply $H$ by a thin, random matrix $\Omega$ of size $N \times (k + p)$, where $p$ is a small oversampling parameter (like 5 or 10). This forms a "sample" matrix $Y = H \Omega$.
2. **Orthonormalization:** You find an orthonormal basis $Q$ of $Y$ (usually via QR decomposition). $Q$ now tightly captures the subspace where the most important physics happens.
3. **Projection:** You project the massive Hamiltonian into this tiny subspace: $T = Q^T H Q$ (where $T$ is only a small $(k+p) \times (k+p)$ matrix).
4. **Standard Diagonalization:** You diagonalize $T$ cheaply using traditional methods to get the eigenvalues and eigenvectors, then rotate them back to the full space using $Q$.

**Why it's a win for chemistry:** If $H$ is large and sparse (or can be applied fast via FFT/linear scaling), you never store or fully diagonalize $H$. You just do a few matrix-vector multiplications.

## 2. The Connection to "Random Probe" Methods (Stochastic DFT)

Your intuition about "probe methods" estimating the Density of States (DOS) is spot on. This is highly related to **Stochastic Density Functional Theory (Stochastic DFT)** and the **Kernel Polynomial Method (KPM)**.

However, there is a subtle but beautiful algorithmic shift here depending on what you actually *need* from the system:

### Case A: You need explicit Wavefunctions (Randomized Subspace)

If you need the actual molecular orbitals (eigenvectors) for further analysis, you use the **Randomized Subspace Iteration** described in Section 1. It acts like a smart filter, using random vectors to isolate the occupied subspace.

### Case B: You only need Global Properties like DOS (Stochastic Probes)

If you *don't* care about individual eigenvectors, but you want the Density of States, $\rho(E) = \sum_i \delta(E - \epsilon_i)$, or the total energy, explicit diagonalization is overkill.

Instead of finding the subspace, you use **Hutchinson’s Trace Estimator**. The trace of an operator can be approximated by averaging its expectation value over random "probe" vectors $\eta$ (often filled with random $\pm 1$ elements):

$$\text{Tr}(A) \approx \frac{1}{M} \sum_{m=1}^M \eta_m^T A \eta_m$$

To get the DOS, you combine this with a spectral expansion (like Chebyshev polynomials in KPM).

1. You inject a random probe vector $\eta$.
2. You propagate it through the Hamiltonian repeatedly ($H\eta$, $H^2\eta$, $H^3\eta$). This is a linear response/time-evolution style calculation.
3. By taking the scalar products of these propagated vectors, you get a stochastic estimate of the Green's function or the spectral density.

Because it relies on random vectors sampling the entire Hilbert space simultaneously, the errors vanish as $1/\sqrt{M}$ (where $M$ is the number of random probes), completely independent of the system size $N$. For massive systems, this changes the scaling from $O(N^3)$ to **$O(N)$ linear scaling**.

## Summary of the Landscape

| Task | Machine Learning (LoRA) | Quantum Chemistry (Orbitals) | Materials Science (DOS/Spectral) |
| --- | --- | --- | --- |
| **Target** | Low-rank weight updates | Subspace of occupied eigenvectors | Continuous spectrum / Green's function |
| **Method** | Randomized SVD | Randomized Subspace Iteration / Lanczos | Stochastic DFT / Kernel Polynomial Method |
| **Role of Randomness** | Projects matrix to a lower-dimensional subspace. | Projects Hamiltonian to the physically relevant low-energy subspace. | Uses random noise cancellation (trace tricks) to sample the whole spectrum at once. |

So, your mental bridge is completely accurate. The transition from using random projections to compress LLM layers to using random probes to bypass $O(N^3)$ diagonalization in materials science is a manifestation of the same underlying math: when a matrix is too big to look at directly, look at how it scatters random noise.

---

# USER

Ok if I understand there are two options either use spectral filter using chebyshev polynomial (pure matrix vector multiplication) or linear solver at z_e right?)

What are the pros and cons?

To me it seems that inversion is kinda just different spectral filter (with 1\(Delta E) shape)

If I want just estimate density of states around some energy or in some energy interval perhaps not so accurately.

Maybe the chebyshev can be used as estimate to initialize the eigensolver to converge faster?

For example if I want to estimate vibration spectrum of large nanokrystal with thousands atoms

---

# Gemini

You have grasped the core concepts perfectly. Yes, inversion is ultimately just a different—and mathematically sharper—spectral filter. While a Chebyshev polynomial approximates a step or box function with a high-degree polynomial curve, the matrix inverse $(zI - H)^{-1}$ provides a natural Lorentzian filter ($1/\Delta E$) centered exactly at the complex energy $z$.

Your intuition about combining these approaches is spot-on. Let's break down the pros, cons, and how this applies directly to your goal of calculating the vibrational spectrum of a massive nanocrystal.

## 1. Chebyshev Filter vs. Rational Filter (Inversion)

Here is how the two approaches stack up when you need to isolate an interior energy window:

| Feature | Chebyshev Polynomial Filter ($f(H)\Omega$) | Rational Filter / FEAST ($(zI - H)^{-1}\Omega$) |
| --- | --- | --- |
| **Mathematical Shape** | Polynomial approximation of a window. | Exact Lorentzian or sharp box (via contour integration). |
| **Primary Operation** | **SpMV / SpMM:** Pure Matrix-Matrix multiplication ($H \cdot H \cdot H \dots \Omega$). | **Linear Solver:** Solving $(z_e I - H)Y = \Omega$. |
| **Scaling & Memory** | Extremely low memory footprint. Predictable, deterministic loop counts. | Higher memory footprint (especially if using direct solvers like MUMPS/PARDISO). |
| **Filter Sharpness** | Requires a **very high degree** ($M \approx 100$ to $1000$) if the window is narrow or the total spectrum is wide. | **Incredibly sharp** with just a few quadrature points ($z_e \approx 8$ to $16$). |
| **Parallelization** | Synchronous sequential multiplications (but highly parallelizable per step on GPUs via BLAS-3). | Perfectly asynchronous. Each $z_e$ point can be solved on a completely separate GPU/node. |

## 2. Estimating DOS via Chebyshev (The Kernel Polynomial Method)

If your goal is to **estimate the Density of States (vibrational spectrum) without needing the explicit eigenvectors**, you shouldn't use an eigensolver at all.

Instead, you can use the **Kernel Polynomial Method (KPM)**. This is a brilliant marriage of Chebyshev polynomials and the random probe concepts you mentioned earlier:

1. **Scale the Spectrum:** Rescale your Dynamical Matrix (the Hessian) so its eigenvalues lie strictly within $[-1, 1]$.
2. **Inject Random Noise:** Generate a few random probe vectors $\eta$.
3. **Generate Chebyshev Moments:** Compute the moments $c_m = \eta^T T_m(H) \eta$ using the Chebyshev recurrence relation:

$$T_m(H) = 2H T_{m-1}(H) - T_{m-2}(H)$$


4. **Reconstruct the Spectrum:** Multiply these moments by a smoothing kernel (to prevent Gibbs oscillations) and sum them up to get the continuous DOS curve.

For a nanocrystal with thousands of atoms, KPM will give you the global vibrational spectrum incredibly fast, scaling strictly as $O(N)$ with system size.

## 3. Chebyshev as a Preconditioner / Initializer

Your idea of using the Chebyshev filter as an initializer to speed up a more accurate eigensolver is a highly respected strategy in numerical linear algebra. It is known as **Polynomial Preconditioning** or **Filtered Subspace Initialization**.

If you *do* need explicit, highly accurate localized vibrational modes (eigenvectors) in a specific window, doing a full FEAST or shifted Lanczos calculation from scratch with a completely random guess can waste a lot of inner-loop iterations.

Instead, you can:

1. Pass a random matrix $\Omega$ through a low-to-medium degree Chebyshev band-pass filter. This cheaply dampens the extreme high and low frequencies that you don't care about.
2. Use the resulting filtered matrix $Y = f(H)\Omega$ as the **starting subspace** for a shifted Lanczos, Davidson, or FEAST solver.

Because the starting vectors are already heavily enriched with the frequencies in your target window, the inner linear solvers will converge in a fraction of the time.

## 4. Application: Vibrational Spectrum of a Nanocrystal

Calculating the vibrational spectrum (phonons) of a nanocrystal with thousands of atoms means dealing with a massive **Hessian matrix** (dynamical matrix). Fortunately, the physics of vibrations plays beautifully into these randomized methods:

### The Real-World Advantage

Vibrational interactions are inherently **short-ranged** (mostly near-neighbor bonds). This means your massive Hessian matrix is **extremely sparse**.

* **For KPM / Chebyshev:** Sparse matrix-vector multiplication is lightning-fast. You can push the polynomial degree very high on a GPU to resolve fine details in the vibrational acoustic and optical bands without worrying about memory limits.
* **For Inversion/FEAST:** While direct solvers might struggle with memory fill-in for 3D nanocrystals, iterative solvers (like Conjugate Gradient) perform incredibly well here because the Hessian is symmetric positive-definite.

If you just want to see how the overall spectrum changes (e.g., how surface relaxation alters the vibrational DOS compared to the bulk), **KPM with random probes** is your best bet. If you need to isolate specific edge-localized or surface-vibrational modes, **Chebyshev-filtered subspace iteration** will hand you those specific eigenvectors with minimal fuss.