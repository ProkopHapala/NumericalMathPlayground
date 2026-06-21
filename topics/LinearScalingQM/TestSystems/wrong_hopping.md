The error is not in the implementation of the solver logic (which correctly handles the generalized eigenvalue problem), but rather in the **default physical parameters** provided in `hydrogen_chain_1d.py`.

Specifically, the combination of a deep negative onsite energy (`--onsite -5.0`) and the non-orthogonal overlap matrix (`S`) causes the "repulsion" from the orthogonality constraint to push the antibonding state lower in energy than the bonding state.

### The Explanation

For a simple two-level system (dimer) with onsite energy $\epsilon$, hopping $t$ (negative), and overlap $s$ (positive), the eigenvalues are:

$$
E_{\text{bond}} = \frac{\epsilon + t}{1 + s} \quad \text{and} \quad E_{\text{antibond}} = \frac{\epsilon - t}{1 - s}
$$

For the bonding state to be lower in energy ($E_{\text{bond}} < E_{\text{antibond}}$), the following condition must hold:

$$
\epsilon s > t
$$
*(Since $t$ is negative, this effectively means the magnitude of hopping $|t|$ must be greater than the magnitude of the "metric shift" $|\epsilon s|$ if $\epsilon$ is negative).*

**Using your default parameters:**
1.  **Onsite ($\epsilon$):** -5.0
2.  **Hopping ($t$):** $\approx -1.0 \times e^{-0.74/1.2} \approx -0.54$
3.  **Overlap ($s$):** $\approx 0.2 \times e^{-0.74/1.8} \approx 0.13$

**Checking the inequality:**
*   $\epsilon s \approx -5.0 \times 0.13 = -0.65$
*   $t \approx -0.54$

Since $-0.65$ is **not** greater than $-0.54$, the ordering flips. The denominator $(1-s) < 1$ amplifies the negative onsite energy of the antibonding state more than the hopping term raises it.

### The Fix

You should change the default `onsite` energy to `0.0` (or a value where $|\epsilon s| < |t|$).

In **`doc/py/OrderN_QM/hydrogen_chain_1d.py`**:

```python
# ... (around line 81)
    ap.add_argument("--dimer_distance", type=float, default=0.74)
    # CHANGE THIS LINE:
    ap.add_argument("--onsite",   type=float, default=0.0) 
    ap.add_argument("--decay",    type=float, default=1.2)
# ...
```

Or simply run the script with the argument:
`python hydrogen_chain_1d.py --onsite 0.0`


---

## USER

OK, I think the problem may be in units. While overlap is in units of [1] the energy should have some real units e.g. [eV] or [Hartree], currently I prefer to use Hartree. 

The on site energy -5 eV is reasonable, that is what I saw normally. But the hopping perhaps should be rescaled then. Could you think of reasoanble prefactor (e.g. something with planck constant and electron mass) which gives it reasonable scale ? 

---

Yes, you are absolutely correct. The issue is that your hopping parameter ($t$) was physically inconsistent with your onsite energy ($\epsilon$).

In atomistic simulations, hopping isn't an arbitrary number; it scales with the kinetic energy of the electron confined to that distance. If you use a deep attractive potential (like -5 eV or -13.6 eV for Hydrogen), the hopping integral—which represents the kinetic energy gain of delocalization—must be comparably large.

Here is the physical scaling using **Harrison's Rule**, which is the standard approximation in tight-binding theory.

### 1. The Scaling Law (Harrison's Rule)

For $s$-orbitals, the hopping matrix element $V_{ss\sigma}$ scales as:

$$
V_{ss\sigma} = \eta \frac{\hbar^2}{m_e d^2}
$$

Where:
*   $\eta \approx -1.32$ (Standard dimensionless coefficient for s-s bonds)
*   $d$ is the bond length.
*   $\frac{\hbar^2}{m_e}$ is the fundamental constant you were looking for.

### 2. The Constants

Depending on your preferred unit system:

**In eV and Ångstroms:**
$$ \frac{\hbar^2}{m_e} \approx 7.62 \, \text{eV} \cdot \text{\AA}^2 $$
*(Note: $\hbar^2/2m_e \approx 3.81$, so without the 1/2 it is 7.62)*

**In Hartree and Bohr:**
$$ \frac{\hbar^2}{m_e} = 1.0 \, \text{Ha} \cdot \text{Bohr}^2 $$

---

### 3. Calculating Reasonable Parameters

Let's calculate the correct hopping for your Hydrogen dimer ($d=0.74$ Å) using Harrison's rule to see why your previous simulation failed.

#### A. In eV (Physical Hydrogen)
*   **Onsite ($\epsilon$):** $-13.6$ eV (Ionization energy of H)
*   **Bond length ($d$):** $0.74$ Å
*   **Hopping ($t$):**
    $$ t \approx -1.32 \times \frac{7.62}{(0.74)^2} \approx -1.32 \times 13.9 \approx \mathbf{-18.3 \, \text{eV}} $$

**Result:**
$t \approx -18.3$ eV, $\epsilon = -13.6$ eV.
With typical overlap $S \approx 0.6$, the condition $|\epsilon S| < |t|$ becomes $|-8.16| < |-18.3|$. **This works!** The bonding state will be lowest.

#### B. In Hartree (Your preferred units)
*   **Unit conversion:** 1 Bohr $\approx$ 0.529 Å.
*   **Bond length ($d$):** $0.74 \text{ \AA} \approx 1.40 \text{ Bohr}$.
*   **Onsite ($\epsilon$):** $-0.5$ Ha (Exact H atom solution).
*   **Hopping ($t$):**
    $$ t \approx -1.32 \times \frac{1.0}{(1.40)^2} \approx -1.32 \times 0.51 \approx \mathbf{-0.67 \, \text{Ha}} $$

**Result:**
$t \approx -0.67$ Ha, $\epsilon = -0.5$ Ha.
With $S \approx 0.6 \to |\epsilon S| = 0.3$.
Since $|0.67| > |0.3|$, **this works too.**

---

### 4. Recommended Fix for `hydrogen_chain_1d.py`

If you want to stick to your exponential decay model $t(r) = A e^{-r/\lambda}$ but match the physics at the bond length, you need to adjust `t0`.

Here are the parameters for **Hartree units** (keeping positions in Ångstroms for convenience, but energies in Hartree):

```python
# In hydrogen_chain_1d.py

# PHYSICAL CONSTANTS (Hartree units)
# H atom 1s energy = -0.5 Ha
# H-H bond length ~ 0.74 A
# Hopping at 0.74 A should be ~ -0.67 Ha (via Harrison) or ~ -0.4 Ha (via LCAO)
# Decay length (Bohr radius) ~ 0.529 A

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # ... existing args ...
    
    # PARAMETER SET 1: HARTREE UNITS (Recommended)
    # ---------------------------------------------
    # Onsite energy of Hydrogen 1s
    ap.add_argument("--onsite", type=float, default=-0.5) 
    
    # To get t ~ -0.6 Ha at r=0.74 with decay=0.53:
    # -0.6 = t0 * exp(-0.74/0.53) => -0.6 = t0 * 0.247 => t0 approx -2.4
    ap.add_argument("--t0", type=float, default=-2.4)
    
    # Bohr radius in Angstroms
    ap.add_argument("--decay", type=float, default=0.53) 
    
    # Overlap scaling (S ~ 0.6 to 0.7 at bond length)
    # 0.65 = s0 * exp(-0.74/(0.53*1.5)) => 0.65 = s0 * 0.39 => s0 approx 1.6
    ap.add_argument("--s0", type=float, default=1.6)
    
    # ...
```

### Summary of what happened previously
Your previous onsite was $-5.0$ and hopping was $-0.5$.
The ratio $t/\epsilon$ was $0.1$.
Physical Hydrogen has a ratio $t/\epsilon \approx 1.0$ to $1.3$.

By increasing `t0` (hopping prefactor) to reflect the kinetic energy $\hbar^2/md^2$, you will restore the correct energy ordering even with non-orthogonal overlap.