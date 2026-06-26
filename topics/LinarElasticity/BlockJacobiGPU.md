# USER

I'm thinking how to solve game engine distance constrains (system of mass points connected by distance constains - basically springs) efficiely on GPU using something like projective dynamics, positions based dynamics or vertex block descent (not so important now which methods to use).
Basically it is truss simulation.

Currently I do two things
1) On CPU I have precomputed Cholesky L*D*L^T factorization, which is very fast precise but serial
2) on GPU I do Jacobi solver accelerated by heavy-ball momentum. I tried Cholesky, but heavy ball was in fact working better. I cannot do  Cholesky  on GPU efficiently as it is serial.

But I was thinking cannot I use Cholesky somehow on GPU in blocks ? Like overlaping patches of the truss each 
One think which bay work is to do cholesky locally within workgroup. and then apply it on overlapping patches. Or use technique of schur complement. 

Other thing is that possibly I can do Conjugated gradient on GPU, but the bottleneck is need for global reduction ideally if I can compute all the necessary reductions just once, so I can do everything by max 2 kernell calls.

---

# Chat GPT 5.5

Yes — **local Cholesky on overlapping GPU patches is a very sensible idea**, but I would not think of it as “GPU Cholesky”. I would think of it as a **local direct preconditioner / smoother** embedded inside a globally parallel iterative method.

For truss / distance constraints, this is often better than trying to reproduce a CPU-style sparse factorization.

The rough hierarchy is:

[
\text{Jacobi / heavy ball}
\quad < \quad
\text{Chebyshev / damped Richardson with good preconditioner}
\quad < \quad
\text{PCG with patch preconditioner}
\quad < \quad
\text{Schur complement domain decomposition}
\quad < \quad
\text{global sparse Cholesky}
]

For GPU game physics I would probably aim near the middle: **overlapping local Cholesky + Chebyshev/Richardson**, and only use CG if you really need high accuracy.

---

## 1. The linear system in projective dynamics / implicit springs

For many PBD/PD/truss formulations, each time step reduces to something like

[
A x = b
]

where

[
A = \frac{M}{\Delta t^2} + K
]

and (K) is a stiffness/Laplacian-like matrix from distance constraints.

For scalar springs:

[
E_{ij} = \frac{k_{ij}}{2}
\left|\mathbf{x}_i-\mathbf{x}*j - \mathbf{d}*{ij}\right|^2
]

or in projective dynamics:

[
E_{ij}
======

\frac{k_{ij}}{2}
\left|
(\mathbf{x}_i-\mathbf{x}*j) - \mathbf{p}*{ij}
\right|^2
]

The matrix contribution is essentially graph Laplacian-like:

[
K_{ii} += k_{ij} I_3,\qquad
K_{ij} -= k_{ij} I_3
]

with possible anisotropic (3\times 3) blocks if you linearize true distance springs.

The important part: **the global matrix is sparse, symmetric positive definite if the inertial/mass term is present**.

Without the mass term, free rigid-body modes make the matrix singular. With

[
\frac{M}{\Delta t^2}
]

it becomes well-conditioned enough for iterative methods.

This is exactly why your heavy-ball Jacobi works better than expected: the time-step mass term pins the system and kills the exact zero modes.

---

## 2. Why global Cholesky is bad on GPU

Sparse Cholesky has three unpleasant GPU properties:

1. **Symbolic dependencies are serial-ish.**
   Elimination introduces fill-in, and the ordering matters.

2. **Triangular solves are sequential along dependency chains.**
   Even if factorization is precomputed, applying (L^{-1}) and (L^{-T}) has poor parallelism.

3. **Irregular memory access dominates.**
   For a truss graph, the sparsity pattern is usually not a nice dense block.

So the CPU result

[
A = LDL^T
]

is excellent if you solve one big system serially, but on GPU it becomes a dependency graph problem, not a FLOP problem.

However, **small Cholesky inside a workgroup is completely different**. A (16)-, (32)-, (64)-vertex patch can be solved very efficiently using shared/local memory.

That gives the natural strategy:

[
\text{global sparse problem}
\quad \rightarrow \quad
\text{many overlapping small dense/sparse local problems}
]

---

## 3. Overlapping local Cholesky = additive Schwarz preconditioner

Suppose you divide your truss into patches (P_a). Each patch contains, for example, 16–64 vertices, possibly with overlap.

For each patch you extract a local matrix:

[
A_a = R_a A R_a^T
]

where (R_a) selects the DOFs belonging to patch (a).

Then precompute or build a local Cholesky:

[
A_a = L_a D_a L_a^T
]

Now given a residual

[
r = b - A x
]

you solve many independent local systems:

[
A_a \delta x_a = R_a r
]

and combine the patch corrections:

[
\delta x
========

\sum_a R_a^T W_a \delta x_a
]

where (W_a) are blending weights, ideally forming a partition of unity:

[
\sum_{a\ni i} W_{a,i} = 1
]

Then update:

[
x \leftarrow x + \omega \delta x
]

This is **additive overlapping Schwarz**.

In practice, this is exactly the kind of thing GPUs like:

```text
for each patch in parallel:
    gather residual of local vertices
    solve local Cholesky in shared memory
    scatter/blend local correction
```

This is not exact, but it is a very strong relaxation step. It damps the modes that Jacobi damps slowly: local bending/stretching modes inside the patch.

The remaining error is mostly long-wavelength global deformation. That can be handled by outer iteration.

---

## 4. Why overlap matters

If patches do not overlap, block-Jacobi sees artificial boundaries. Modes crossing patch boundaries converge slowly.

With overlap, each local solve can “see through” the boundary a little. This makes the local solve behave more like a real physical relaxation.

A simple patch layout might be:

```text
core vertices:      vertices owned by this patch
halo vertices:      neighboring vertices included for local solve
write-back:         only core vertices, or weighted all vertices
```

Two common choices:

### Option A: solve core + halo, write only core

This avoids atomics.

```text
patch contains core + halo
solve local correction
write correction only to core vertices
```

This is closer to multiplicative Schwarz but still parallel if each vertex has one owner.

Pros:

* no atomics
* simple write pattern
* deterministic

Cons:

* boundary information is less symmetrically blended

### Option B: solve overlapping patches, weighted scatter

Each patch writes to all vertices it contains:

[
\delta x_i += w_{a,i}\delta x_{a,i}
]

Then you normalize.

Pros:

* smoother
* usually better convergence

Cons:

* needs atomics or two-pass gather

For GPU I would prefer a **gather formulation** if possible:

```text
for each vertex i:
    collect corrections from patches containing i
    weighted average
```

But this requires storing patch corrections. If memory is not too bad, it is cleaner than atomics.

---

## 5. Use local Cholesky as preconditioner, not as the solver itself

The global iteration would be:

[
r_n = b - A x_n
]

[
z_n = P^{-1} r_n
]

[
x_{n+1} = x_n + \omega z_n
]

where (P^{-1}) is the overlapping patch-Cholesky operator.

This is preconditioned Richardson iteration.

Better:

[
x_{n+1}
=======

x_n
+
\alpha_n z_n
+
\beta_n (x_n - x_{n-1})
]

This is your heavy-ball idea, but with **patch-Cholesky-preconditioned residuals** instead of Jacobi residuals.

Even better: use **Chebyshev acceleration**.

Chebyshev iteration has a major GPU advantage over CG:

[
\textbf{no global dot products}
]

It only needs approximate spectral bounds:

[
\lambda_{\min},\lambda_{\max}
]

of the preconditioned system:

[
P^{-1}A
]

For game physics, approximate bounds are usually enough. You can tune them empirically or estimate by a few power iterations offline.

So the practical solver becomes:

```text
repeat 5-20 times:
    r = b - A*x
    z = ApplyPatchCholeskyPreconditioner(r)
    x = ChebyshevUpdate(x, z, previous_state)
```

This may beat CG in wall-clock time even if it needs more iterations, because it avoids global reductions.

---

## 6. CG on GPU: what is the real bottleneck?

Standard preconditioned conjugate gradient has this structure:

[
r = b - A x
]

[
z = P^{-1}r
]

[
\rho = r^T z
]

[
p = z + \beta p
]

[
q = A p
]

[
\sigma = p^T q
]

[
\alpha = \rho/\sigma
]

[
x \leftarrow x + \alpha p
]

[
r \leftarrow r - \alpha q
]

So each iteration needs at least these reductions:

[
r^Tz
]

[
p^TAp
]

Naively this means two global reductions, hence multiple kernel launches or global barriers.

You can fuse the reductions inside one kernel only up to workgroup level. Across the whole grid, OpenCL has no global synchronization inside one kernel unless you use special cooperative mechanisms, persistent kernels, or vendor-specific extensions.

So for a single large truss, standard CG usually needs something like:

```text
kernel 1: compute q=A*p and partial dot products
kernel 2: reduce dot products, update alpha/beta
kernel 3: update x,r,p
```

With careful fusion, maybe two kernels per iteration.

There are pipelined CG variants that reduce synchronization, but they are more complex and often less numerically robust. In single precision, especially for stiff trusses, I would be cautious.

For game physics, I would not make CG the first choice unless you need accurate solves.

---

## 7. Can all CG reductions be done “just once”?

Per CG iteration, no, not really.

CG coefficients depend recursively on fresh dot products:

[
\alpha_n = \frac{r_n^T z_n}{p_n^T A p_n}
]

[
\beta_n = \frac{r_{n+1}^T z_{n+1}}{r_n^T z_n}
]

These quantities are not known ahead of time. They depend on the current residual.

You can reduce the number of global synchronizations per iteration, but you cannot compute all reductions for many future iterations at once unless you switch to a different Krylov method, such as (s)-step CG or communication-avoiding CG. Those methods build a block Krylov basis:

[
{r, Ar, A^2r,\dots,A^sr}
]

and orthogonalize less frequently.

But for game physics and single precision, this is probably overkill and may become fragile.

So the best practical answer is:

> If global reductions are expensive, avoid CG and use Chebyshev / heavy-ball / polynomial iteration with a strong local preconditioner.

That fits your current observation: heavy-ball Jacobi already works well because it is synchronization-free.

---

## 8. Schur complement version

Schur complement is the more mathematically exact domain decomposition version.

Split DOFs into patch interiors (I) and interface/boundary DOFs (B):

[
\begin{bmatrix}
A_{II} & A_{IB} \
A_{BI} & A_{BB}
\end{bmatrix}
\begin{bmatrix}
x_I \
x_B
\end{bmatrix}
=============

\begin{bmatrix}
b_I \
b_B
\end{bmatrix}
]

Eliminate interiors:

[
x_I = A_{II}^{-1}(b_I - A_{IB}x_B)
]

Insert into boundary equation:

[
\left(
A_{BB} - A_{BI}A_{II}^{-1}A_{IB}
\right)x_B
==========

b_B - A_{BI}A_{II}^{-1}b_I
]

The Schur complement is

[
S =
A_{BB} - A_{BI}A_{II}^{-1}A_{IB}
]

The nice part:

* (A_{II}^{-1}) is local per patch.
* You can apply it with local Cholesky.
* The global solve is only over boundary/interface DOFs.

The bad part:

* (S) is denser than the original matrix.
* Assembling (S) explicitly can be expensive.
* Solving the interface problem still requires global iteration.

Matrix-free Schur application is possible:

```text
given boundary vector xB:
    for each patch:
        tmpI = A_IB * xB
        yI   = solve(A_II, tmpI)
        contribution_to_boundary = A_BI * yI
    yB = A_BB*xB - contribution_to_boundary
```

This is attractive if patches have many interior DOFs and relatively few boundary DOFs.

But for a thin truss graph, especially with many edges crossing patch boundaries, the interface may be large. Then Schur complement may not reduce enough.

So I would use Schur complement if your structure has natural substructures:

```text
beams, plates, clusters, rigid-ish modules, cloth tiles, lattice chunks
```

I would not use it for arbitrary small graph patches unless you really need accuracy.

---

## 9. Vertex block descent / local star solve

Another very interesting middle ground is **vertex-star block descent**.

Instead of solving arbitrary patches, solve the local block associated with one vertex and its incident edges.

For vertex (i), minimize local quadratic energy with respect to (\mathbf{x}_i), keeping neighbors fixed:

[
\left(
\frac{m_i}{\Delta t^2}I
+
\sum_{j\in N(i)} k_{ij}I
\right)\mathbf{x}_i
===================

\frac{m_i}{\Delta t^2}\mathbf{y}*i
+
\sum*{j\in N(i)} k_{ij}(\mathbf{x}*j + \mathbf{p}*{ij})
]

For isotropic PD springs, this is only a (3\times 3) or even scalar denominator solve per vertex.

That is Jacobi / Gauss-Seidel-like.

But you can generalize:

* one vertex block: cheap, Jacobi-like
* edge block: solve two vertices
* small graph patch: solve 8–64 vertices
* overlapping Cholesky patch: strongest local correction

This gives a continuum between your current Jacobi method and full local factorization.

---

## 10. Multigrid-like interpretation

The hard modes for Jacobi are usually long-wavelength modes.

Local Cholesky patches remove short/mid-wavelength error very well, but global bending or stretching modes still converge slowly.

So another path is:

```text
fine truss solver: local patch Cholesky / Jacobi / Chebyshev
coarse truss solver: reduced graph, fewer vertices
prolongation: interpolate coarse correction to fine graph
```

This becomes algebraic multigrid-ish.

For game physics, even a crude geometric/coarsened hierarchy can help a lot:

```text
original graph
↓ cluster vertices
coarse graph
↓ cluster again
very coarse graph
```

At the coarsest level, CPU Cholesky or one workgroup dense solve may be cheap.

This may be better than Schur complement if the truss has large global modes.

---

## 11. Practical recommendation

I would test four solvers in this order.

### Solver A: current Jacobi + heavy ball

Keep it as baseline.

```text
r = b - A*x
dx = D^-1*r
v = beta*v + omega*dx
x += v
```

This is cheap and often visually good.

---

### Solver B: Jacobi + Chebyshev

Replace manually tuned heavy-ball with Chebyshev semi-iteration.

Conceptually:

[
x_{n+1}=x_n+\alpha_n D^{-1}(b-Ax_n)+\beta_n(x_n-x_{n-1})
]

but (\alpha_n,\beta_n) follow a Chebyshev schedule based on spectral bounds.

This keeps the same memory pattern as Jacobi but may converge more predictably.

No reductions.

---

### Solver C: overlapping patch-Cholesky + Richardson/heavy-ball/Chebyshev

This is probably the sweet spot.

```text
for iteration:
    compute residual r = b - A*x
    z = ApplyOverlappingPatchCholesky(r)
    x = accelerated_update(x,z)
```

Patch application:

```text
for patch p in parallel:
    gather r_p
    solve A_p z_p = r_p using local LDLT
    write weighted z_p
```

This attacks local constraint stiffness much better than Jacobi.

No global reductions.

---

### Solver D: patch-Cholesky-preconditioned CG

Use only if you need accurate convergence.

```text
z = P^-1 r
CG with P
```

This likely gives the fewest iterations, but each iteration needs global reductions.

Useful for offline, editor mode, or high-quality simulation, less attractive for real-time.

---

## 12. Kernel organization for Solver C

A realistic GPU frame might look like this:

```text
kernel 1:
    project constraints
    assemble b or constraint RHS terms

kernel 2:
    compute residual r = b - A*x

kernel 3:
    apply patch Cholesky preconditioner
    produce z

kernel 4:
    update x using heavy-ball/Chebyshev
```

You can fuse some of these.

For example, if (A) is matrix-free graph Laplacian:

```text
kernel residual:
    for each vertex i:
        Ax_i = mass_i*x_i/dt2
        for each neighbor j:
            Ax_i += k_ij*(x_i - x_j)
        r_i = b_i - Ax_i
```

Patch solve kernel:

```text
kernel patch_solve:
    p = global_id
    load patch vertex indices
    load local residuals into shared memory
    solve local LDLT
    store local correction
```

Then update:

```text
kernel update:
    for each vertex i:
        gather/blend patch corrections touching i
        v_i = beta*v_i + omega*z_i
        x_i += v_i
```

If you use atomics, patch solve can directly accumulate `z_i`, but I would prefer the two-pass gather if the patch count per vertex is bounded and known.

---

## 13. Precomputing local Cholesky

If your topology, masses, stiffnesses, and timestep are fixed, then

[
A = \frac{M}{\Delta t^2}+K
]

is fixed.

Therefore every patch matrix (A_p) is fixed.

You can precompute:

```text
patch vertex indices
patch local matrix
patch LDLT factors
patch blending weights
vertex -> incident patches
```

Then each frame only applies the factors.

If stiffness or timestep changes, you either rebuild factors or use a small set of factor banks.

For projective dynamics, this is especially nice because the left-hand matrix is usually constant. Only (b) changes due to projections.

---

## 14. What patch size?

A good first experiment:

```text
patch size: 16 vertices
DOFs:       48
```

or

```text
patch size: 32 vertices
DOFs:       96
```

A dense (48\times48) Cholesky is already nontrivial but feasible inside a workgroup if reused many times. However, if the local graph is sparse, you may not want a dense solve.

For trusses, I would first use sparse local LDLT with fixed pattern, or dense only for small patches.

Reasonable settings:

```text
8 vertices   -> very cheap, weak
16 vertices  -> good starting point
32 vertices  -> stronger, more shared memory/register pressure
64 vertices  -> maybe too heavy unless sparse/local structure is nice
```

A nice patch is not just any 16 vertices. It should be graph-local:

```text
seed vertex
1-ring or 2-ring neighbors
possibly clipped to max size
```

Overlap naturally comes from neighboring seeds.

---

## 15. Boundary treatment inside patches

This is subtle.

If patch contains vertices (P), but those vertices have edges going outside the patch, you have choices.

### Dirichlet local solve

Treat outside vertices as fixed. Their effect appears in the residual but they are not solved.

This is the standard local subspace correction.

The local matrix includes only patch DOFs:

[
A_p = R_p A R_p^T
]

This is simple and stable.

### Neumann-like local solve

Ignore outside connections in the local matrix.

This makes the patch more singular/soft and can overcorrect.

Usually worse unless carefully damped.

### Include halo but write core

This is often best:

```text
local solve includes core + halo
write only core correction
```

The halo gives better local physics, but ownership stays clean.

---

## 16. Important warning: local solves can over-stiffen or over-relax

If many overlapping patches all apply full corrections, you can overshoot.

That is why weights and damping matter.

The global update should be damped:

[
x \leftarrow x + \omega P^{-1}r
]

with

[
0 < \omega \leq 1
]

For heavy-ball:

[
v \leftarrow \beta v + \omega P^{-1}r
]

[
x \leftarrow x + v
]

Start conservatively:

```text
omega = 0.2 .. 0.8
beta  = 0.3 .. 0.8
```

If using Chebyshev, tune spectral bounds of (P^{-1}A), not of (A).

---

## 17. Why this may beat CG visually

CG minimizes the global quadratic energy efficiently, but it spends time maintaining global conjugacy. That requires dot products and synchronization.

For real-time physics, you often care more about:

```text
remove local constraint error quickly
avoid jitter
avoid global barriers
produce plausible deformation
```

Patch Cholesky does exactly that. It is a powerful **local physics smoother**.

The remaining global low-frequency error is often visually acceptable, or can be handled by a few coarse-grid corrections.

So I would not ask:

> How do I make Cholesky parallel?

I would ask:

> How do I use many tiny Cholesky solves to make a better GPU smoother/preconditioner?

That is the right framing.

---

## 18. My preferred design

For your case, I would implement this:

[
A x = b
]

with matrix-free (A), and solver:

```text
for k = 0..nIter:
    r = b - A*x
    z = OverlappingPatchLDLT(r)
    x = ChebyshevOrHeavyBallUpdate(x,z)
```

Patch preconditioner:

```text
for each patch:
    gather residuals
    solve local A_p z_p = r_p
    store z_p
for each vertex:
    weighted gather of z_p from incident patches
```

Use:

```text
Jacobi-heavy-ball as baseline
PatchLDLT-heavy-ball as next step
PatchLDLT-Chebyshev as final non-reduction solver
PatchLDLT-PCG only for accuracy testing
```

This gives you a clean comparison:

```text
same A, same b
different preconditioners / accelerators
measure residual decay and visual stiffness
```

---

## 19. Compact pseudocode

```cpp
// fixed/precomputed:
Patch patches[nPatches];        // vertex ids, local factors, weights
VertexPatches vp[nVerts];       // incident patch ids per vertex
Graph graph;                    // neighbor list for A*x

for(frame){
    project_constraints_make_rhs(x_pred, b);

    for(int it=0; it<nIter; it++){
        // kernel 1: residual
        for_each_vertex(i){
            Vec3 Ax = mass[i]*inv_dt2*x[i];
            for(e in graph.edges_of(i)){
                int j = e.j;
                Ax += e.k*(x[i]-x[j]);
            }
            r[i] = b[i] - Ax;
        }

        // kernel 2: local patch solves
        for_each_patch(p){
            gather rp from r[patch.vertex_ids];
            zp = solve_LDLT(p.factor, rp);
            store patch_corr[p] = zp;
        }

        // kernel 3: gather/blend/update
        for_each_vertex(i){
            Vec3 z = Vec3Zero;
            double wsum = 0;
            for(pp in vp[i]){
                int p = pp.patch_id;
                int loc = pp.local_index;
                double w = pp.weight;
                z += w * patch_corr[p][loc];
                wsum += w;
            }
            z /= wsum;
            v[i] = beta*v[i] + omega*z;
            x[i] += v[i];
        }
    }
}
```

For Chebyshev, replace the last update by a Chebyshev recurrence. The rest stays the same.

---

## 20. Final recommendation

Your best path is probably:

[
\boxed{
\text{overlapping patch Cholesky as preconditioner}
+
\text{Chebyshev or heavy-ball iteration}
}
]

not GPU global Cholesky and probably not standard CG as the main real-time solver.

Schur complement is worth considering only if your truss has natural large subdomains with small interfaces. Otherwise it may be more complexity than benefit.

The most promising architecture is:

```text
local exactness inside patches
global approximate iteration outside patches
no global reductions
fixed number of iterations per frame
fully predictable GPU workload
```

That matches game-engine constraints much better than exact linear algebra.

---

# USER

We should think in realistic terms. For decent simulation it is enough to do just few Jacobi constrain steps per time-step (5-15), so we cannot fit in much more complicated things (like several CG or Cholesky steps) into that time budget.

From your discussion I perticularily like two ideas

1) multigrid - I think multigrid would provide major improvement to solving problematic long wavelenghs. The problem is that I do not know how ti implement it for general mesh - like my truss is not like cloth (regular grid) it is system of girders froming spacecraft, something like crane or power line girders, it is a bit like fractal structure. if I colapse the griders how I ensure the coarse-grained structure have same mechanical properties (e.g. bending stiffness) as the original high resolution one? This seems to me non-trivial problem. 

2) overlapping patches but with Jacobi rather than Cholesky. The more I think about it the more I have feeling Cholesky will not give to much on GPU considering the complexity and higher memory footprint (fill-in problem). The shared local memory is most precious resource on GPU.  Jacobi is naturally parallel and maximally sparse. What slow down Jacobi is global memory access and kernel calls (I have to call kernel e.g. 5x-15x every time step). Since global memory access is more costly than claculation it would be good to amortize it by utilizing data preloaded in local memory. This can be perhaps done if I consider these overlaping patches. I preload the patch to local memory, do few jacobi steps within one kernel call, store the updates in axuliary buffer, and then next kernel call assemble them into updated position vector (to avoid rise-condition and need for ping-pong buffers), in this assembly step I can choose how to combine patches together. overlaping vertexes can be just averaged perhaps? 

---

# Chat GPT 5.5

Yes. Under the constraint “roughly the cost of 5–15 Jacobi/PBD iterations”, I would also abandon local Cholesky as the primary idea. It is elegant, but it spends too much complexity and local memory on accuracy you may not need.

The more realistic target is:

[
\boxed{
\text{block/patch Jacobi with several inner local-memory iterations per kernel call}
}
]

and, later,

[
\boxed{
\text{very cheap graph multigrid/coarse correction for long waves}
}
]

These two attack the two different weaknesses of plain Jacobi:

[
\text{plain Jacobi}
\begin{cases}
\text{bad at local stiffness because each step uses stale neighbors} \
\text{bad at long wavelengths because information diffuses slowly}
\end{cases}
]

Patch-Jacobi improves the first problem by doing several local relaxations while data is already in shared/local memory. Multigrid improves the second problem by letting global deformations move on a coarse graph.

---

# 1. Patch-Jacobi is probably the best near-term idea

Your proposed algorithm is basically:

```text
global positions x
↓
load overlapping patch into local memory
↓
do several Jacobi / PBD / constraint-relaxation steps inside patch
↓
write patch result to auxiliary buffer
↓
assemble overlapping patch outputs into global x
```

This is very reasonable.

The key point is that the expensive operation in plain Jacobi is not the arithmetic. It is repeatedly reading neighboring vertices and constraints from global memory, and launching kernels repeatedly.

If a vertex/constraint neighborhood is loaded once and reused for, say, 4–8 local iterations, the arithmetic intensity improves a lot.

So instead of:

```text
15 global Jacobi sweeps
= 15 kernel launches
= 15 global reads of edges and vertices
= 15 global writes
```

you do something like:

```text
3 outer patch passes
each patch pass does 5 local Jacobi substeps
= 3 expensive global synchronizations
= 15 effective local iterations
```

The result is not identical to 15 global Jacobi sweeps, because patch boundaries are stale during the inner iterations, but for truss-like mechanics it may be better per bandwidth.

---

# 2. What exactly is solved inside the patch?

Suppose your constraints are distance constraints between vertices (i,j):

[
C_{ij}(\mathbf{x}) = |\mathbf{x}_i-\mathbf{x}*j| - L*{ij}
]

A PBD-style correction is roughly:

[
\Delta \mathbf{x}_i
===================

-\frac{w_i}{w_i+w_j}
C_{ij}
\frac{\mathbf{x}_i-\mathbf{x}_j}{|\mathbf{x}_i-\mathbf{x}_j|}
]

[
\Delta \mathbf{x}_j
===================

+\frac{w_j}{w_i+w_j}
C_{ij}
\frac{\mathbf{x}_i-\mathbf{x}_j}{|\mathbf{x}_i-\mathbf{x}_j|}
]

For Jacobi, you compute all corrections from old positions and then apply accumulated corrections.

Inside one patch, you can do the same, but using local arrays:

```cpp
x_loc[local_vertex_id]
dx_loc[local_vertex_id]
```

Then inner loop:

```cpp
for(int inner=0; inner<nInner; inner++){
    dx_loc = 0
    for each local constraint e=(i,j):
        compute PBD correction from x_loc[i], x_loc[j]
        accumulate into dx_loc[i], dx_loc[j]
    for each local vertex i:
        x_loc[i] += omega * dx_loc[i] / weight_sum[i]
}
```

This is simple, sparse, and uses mostly local memory.

If you use projective dynamics instead of nonlinear PBD, the local iteration is even simpler because each edge contributes a linear Laplacian-like correction once the projection direction is known.

---

# 3. Boundary vertices: the most important design choice

A patch should probably contain:

```text
core vertices: owned by this patch
halo vertices: neighboring vertices needed to evaluate constraints
```

The local solve can update both core and halo internally, but the write-back policy matters.

There are three natural choices.

---

## Option A: update only core vertices

Patch contains core + halo, but writes only core.

```text
load core + halo
relax all local vertices
write only core vertices
```

This avoids atomics and avoids overlap averaging.

The halo gives the patch a better local environment, but global ownership remains clean.

This is probably the first version I would implement.

Advantages:

```text
no race conditions
no atomics
no auxiliary patch-result averaging
one global output x_new per vertex
deterministic
```

Disadvantage:

```text
overlap is used only as context, not as multiple competing predictions
```

But for a game solver this may be completely fine.

---

## Option B: each patch writes prediction for all its vertices, then average

This is closer to what you proposed.

Each patch outputs:

[
\tilde{\mathbf{x}}_i^{(p)}
]

for every vertex (i) in patch (p). Then assembly does:

[
\mathbf{x}_i^{new}
==================

\frac{
\sum_{p\ni i} w_{p i}\tilde{\mathbf{x}}*i^{(p)}
}{
\sum*{p\ni i} w_{p i}
}
]

This is a nice method. It is like an overlapping additive Schwarz method, but with Jacobi/PBD inner iterations instead of exact local solves.

The weights should not necessarily be uniform. Better is to make core vertices high weight and halo vertices low weight.

For example:

[
w_{p i} =
\begin{cases}
1 & i \in \text{core of patch }p \
0.25 & i \in \text{first halo} \
0.0 \text{ or } 0.1 & i \in \text{outer halo}
\end{cases}
]

This avoids the situation where a vertex near the boundary is pulled equally by several stale local worlds.

Advantages:

```text
smooth blending
uses overlap fully
can be more stable than hard ownership
```

Disadvantages:

```text
needs auxiliary storage patch_corr[patch][local_vertex]
needs assembly kernel
more memory traffic
```

This is still realistic if patch size and overlap count are small.

---

## Option C: patches atomically accumulate corrections

Patch kernel directly does:

```cpp
atomic_add(x_accum[i], w * x_patch[i])
atomic_add(w_accum[i], w)
```

Then normalize.

I would avoid this initially. Atomic floating point adds are often nondeterministic, and on older GPUs they can be slow. It may still work, but it makes debugging harder.

So I would prefer:

```text
patch kernel writes patch-local results to auxiliary buffer
assembly kernel gathers/averages them per vertex
```

Gather is usually cleaner than scatter.

---

# 4. Patch-Jacobi as “communication avoiding Jacobi”

A useful mental model:

Plain Jacobi communicates with global memory every iteration:

[
x^{n+1} = J(x^n)
]

Patch-Jacobi communicates globally only every (m) inner iterations:

[
x^{n+m} \approx J_{\text{patch}}^m(x^n)
]

Inside the patch, information propagates quickly among local vertices. Across patch boundaries, information propagates only once per outer pass.

So this method improves high-frequency/local convergence, but long-wavelength global modes remain slow. That is exactly where multigrid enters.

---

# 5. Patch size and inner iteration count

You want enough reuse to amortize global reads, but not so much that boundary staleness dominates.

A good starting point:

```text
patch core:       16–32 vertices
halo:             1-ring neighbors
local vertices:   maybe 32–96 total
inner iterations: 3–8
outer passes:     2–4
```

Total effective constraint relaxation:

[
n_{\text{eff}} = n_{\text{outer}} n_{\text{inner}}
]

For example:

```text
3 outer passes × 5 inner iterations = 15 local iterations
```

But boundary communication happens only 3 times.

This may behave like:

```text
better than 5 global Jacobi sweeps
not always as globally coherent as 15 sweeps
much cheaper than 15 full global sweeps
```

For truss structures, that tradeoff may be favorable because many constraints are local along beams/girders.

---

# 6. Avoiding race conditions without ping-pong explosion

A clean buffer layout:

```cpp
float4 x_old[nVerts];
float4 x_new[nVerts];

float4 patch_x[nPatches * maxPatchVerts];  // auxiliary patch predictions
```

Kernel 1: patch relaxation

```text
input:  x_old
output: patch_x
```

Kernel 2: assembly

```text
input:  patch_x
output: x_new
```

Then swap:

```cpp
swap(x_old, x_new)
```

So per outer pass:

```text
patch_relax_kernel
assemble_kernel
```

If you do 3 outer passes, this is 6 kernels. That may already replace 15 Jacobi kernels.

If assembly is cheap and memory-coalesced, this can be a win.

You can reduce storage by outputting only core vertices:

```cpp
patch_core_x[nPatches * coreSize]
```

Then assembly may even be trivial if each global vertex belongs to exactly one core.

---

# 7. The core-only version is probably the first implementation

I would first implement this variant:

```text
patches overlap by halo
each global vertex belongs to exactly one patch core
each patch writes only core vertices
```

Then each outer pass is:

```text
patch kernel:
    load core + halo from x_old
    do nInner local Jacobi/PBD steps
    write relaxed core to x_new
swap x_old/x_new
```

This needs no assembly kernel, no atomics, and no patch result averaging.

In pseudocode:

```cpp
kernel patch_relax(
    const float4* x_old,
    float4* x_new,
    const int* patchVerts,
    const int* patchIsCore,
    const Edge* patchEdges
){
    int p = get_group_id(0);

    // load local vertices
    for(local thread loop)
        xloc[a] = x_old[patchVerts[p*MAXV+a]];

    barrier();

    for(int it=0; it<nInner; it++){
        dxloc[a] = 0;
        wsum[a]  = 0;

        barrier();

        for(local edge e assigned to thread){
            int ia = e.local_i;
            int ib = e.local_j;
            // compute constraint correction from xloc[ia], xloc[ib]
            // accumulate local dx
        }

        barrier();

        for(local vertex a assigned to thread){
            xloc[a] += omega * dxloc[a] / max(wsum[a], eps);
        }

        barrier();
    }

    // write only core
    for(local vertex a assigned to thread){
        if(patchIsCore[p*MAXV+a]){
            int i = patchVerts[p*MAXV+a];
            x_new[i] = xloc[a];
        }
    }
}
```

The catch: accumulating `dxloc` from multiple local edges into local vertices may need local atomics or a different schedule.

Better for local memory is often **vertex-gather Jacobi**:

```cpp
for each local vertex a:
    dx = 0
    for each incident local edge e:
        compute correction contribution to vertex a
    x_next[a] = x[a] + omega*dx/wsum
```

This avoids edge scatter.

That matches your general preference for gather kernels.

---

# 8. Local-memory gather formulation

For each local vertex (a), store a local adjacency list:

```cpp
localNeighStart[a]
localNeighCount[a]
localNeighIds[]
localEdgeParams[]
```

Then each local vertex thread does:

```cpp
Vec3 xi = xloc[a];
Vec3 dx = 0;
float wsum = 0;

for each neighbor b of a:
    Vec3 xj = xloc[b];
    compute constraint correction contribution to a
    dx += ...
    wsum += ...

xnext[a] = xi + omega * dx/wsum;
```

Then swap local buffers:

```cpp
xloc <-> xnext
```

This is very GPU-friendly:

```text
no local atomics
no global atomics
regular per-vertex ownership
Jacobi semantics are clean
```

You may duplicate edge work because both endpoints compute their side of the constraint. That is fine. Arithmetic is cheap. Avoiding atomics is probably worth it.

---

# 9. Patch construction for arbitrary truss

For your spacecraft/crane/powerline-girder example, I would build patches graph-theoretically, not geometrically.

One simple method:

1. Choose patch seeds.
2. Grow each patch by graph distance.
3. Define core by Voronoi assignment to nearest seed.
4. Add 1-ring or 2-ring halo.

So:

```text
core = vertices closest to seed p
halo = neighbors of core up to radius h
patch = core ∪ halo
```

This works on arbitrary graph topology.

For long slender girders, you may want elongated patches along beam direction, not spherical graph balls. But graph balls are good first.

Patch quality criteria:

```text
each core vertex belongs to exactly one patch
patch size bounded
all constraints touching core vertices are included
halo includes at least all immediate neighbors of core
```

The minimum required halo is:

[
\text{halo} = N(\text{core})
]

so every constraint incident to a core vertex can be evaluated.

---

# 10. Multigrid for arbitrary truss: do not manually preserve bending stiffness; use Galerkin coarsening

Your concern is correct: if you simply collapse girders naively, you may destroy mechanical properties.

But there is a standard way to avoid manually guessing coarse spring constants:

[
\boxed{
A_c = R A_f P
}
]

or, for symmetric systems,

[
\boxed{
A_c = P^T A_f P
}
]

where:

* (A_f) is the fine stiffness/system matrix,
* (P) maps coarse displacements to fine displacements,
* (R=P^T) is the restriction operator,
* (A_c) is the coarse matrix.

This is the Galerkin coarse operator.

The important point is:

> You do not invent coarse springs directly. You define how coarse motion interpolates to fine vertices, and the coarse stiffness follows from energy preservation.

Because:

[
E_f(x_f) = \frac12 x_f^T A_f x_f
]

and if

[
x_f = P x_c
]

then

[
E_c(x_c)
========

\frac12 x_c^T P^T A_f P x_c
]

so

[
A_c = P^T A_f P
]

This automatically preserves the energy of all deformation modes representable by the coarse basis (P).

That is the key idea.

---

# 11. What should (P) be for a truss?

For scalar graph Laplacian problems, (P) can just be interpolation weights from coarse nodes to fine nodes.

For mechanical structures, each vertex has vector displacement:

[
\mathbf{x}_i \in \mathbb{R}^3
]

The simplest (P) says:

[
\mathbf{x}_i
============

\sum_{I \in \text{coarse neighbors}}
w_{iI} \mathbf{X}_I
]

where (I) indexes coarse vertices.

This preserves translations automatically if:

[
\sum_I w_{iI}=1
]

But pure displacement interpolation may not preserve local rotations/bending well.

For truss made of axial springs, that may be acceptable, because the fine model itself only has axial springs. Bending stiffness emerges from geometry, triangulation, and cross-bracing. If the coarse interpolation captures the main geometric modes, Galerkin (P^TAP) will produce the correct effective stiffness for those modes.

For actual beam/girder bending, a better coarse basis includes rotations.

Each coarse node has:

[
\mathbf{u}_I \in \mathbb{R}^3,
\qquad
\boldsymbol{\theta}_I \in \mathbb{R}^3
]

and a fine vertex at rest position (\mathbf{r}_i) attached to coarse node (I) moves as:

[
\mathbf{u}_i
============

\mathbf{u}_I
+
\boldsymbol{\theta}_I
\times
(\mathbf{r}_i-\mathbf{R}_I)
]

This is a rigid-body interpolation. For clusters/substructures, this is much better because it preserves local rigid motion.

Then (P) maps coarse translations and rotations to fine vertex displacements.

This is more complex, but powerful.

---

# 12. Two levels of coarse model

For real-time, I would not implement full algebraic multigrid first. I would implement a crude two-level correction.

Fine level:

```text
original vertices and constraints
patch-Jacobi smoother
```

Coarse level:

```text
clusters of vertices
one coarse node per cluster
coarse springs / coarse matrix built by Galerkin or fitted stiffness
few Jacobi steps or CPU solve
```

Algorithm:

```text
1. Fine pre-smoothing:
   do 1–2 patch-Jacobi passes

2. Compute residual:
   r_f = b_f - A_f x_f

3. Restrict residual:
   r_c = P^T r_f

4. Solve approximately on coarse graph:
   A_c e_c = r_c
   using few Jacobi / small CPU solve / one GPU kernel

5. Prolongate correction:
   x_f += P e_c

6. Fine post-smoothing:
   do 1–2 patch-Jacobi passes
```

This is a V-cycle in minimal form.

For game physics, even one coarse correction per time step may greatly improve long-wavelength behavior.

---

# 13. How to build coarse clusters for spacecraft-like truss

Several practical options:

## Graph matching coarsening

Pair strongly connected vertices.

Define edge strength:

[
s_{ij} = k_{ij}
]

or, better,

[
s_{ij} = \frac{k_{ij}}{m_i+m_j}
]

Then greedily match vertices connected by strong edges. Each pair becomes a coarse node. Repeat to build hierarchy.

This is simple and generic.

## Geometric clustering

Use rest positions. Partition space into cells or use k-means-like clustering. Good for large structures.

But for fractal truss/girders, pure spatial clustering may group vertices that are close in space but not mechanically close.

## Beam-aware clustering

If you know girders/struts as higher-level objects, coarsen along them.

For example:

```text
fine triangles/struts inside one girder segment
→ one coarse beam element
```

This gives best mechanical meaning but requires semantic structure.

## Hybrid

Use graph connectivity but bias by spatial proximity and stiffness.

For a first implementation:

[
\text{edge score}_{ij}
======================

k_{ij}
\exp\left(-\frac{|\mathbf{r}_i-\mathbf{r}_j|^2}{R^2}\right)
]

Then greedily aggregate strong edges.

---

# 14. How to preserve bending stiffness in coarse model

If the original structure is a pure pin-jointed truss, bending stiffness is not a property of individual beams. It arises from the geometry of triangles/tetrahedra/cross-bracing.

So preserving bending stiffness means preserving the energy of low-frequency deformation modes.

The Galerkin formula does that automatically for modes in the interpolation space:

[
A_c = P^T A_f P
]

If your coarse basis only allows cluster translations, it preserves translational deformation modes. If your basis includes cluster rotations, it also preserves bending-like modes much better.

So there are two routes:

### Cheap route

One coarse node per cluster, translation only.

```text
coarse DOF per cluster: 3
```

Good for:

```text
mass-spring networks
cloth-like systems
rough global correction
```

May be too soft/stiff in bending for girder structures.

### Better route

One coarse rigid cluster per aggregate.

```text
coarse DOF per cluster: 6
translation + small rotation
```

Fine displacement:

[
\mathbf{u}_i
============

\mathbf{U}_C
+
\boldsymbol{\omega}_C
\times
(\mathbf{r}_i-\mathbf{R}_C)
]

This lets a whole girder cell rotate nearly rigidly. Then coarse coupling between clusters captures bending and torsion much better.

This is very close in spirit to substructuring / component mode synthesis, but you can keep it cheap.

---

# 15. Coarse correction with rigid-cluster DOFs

For cluster (C), define coarse DOF:

[
q_C =
\begin{bmatrix}
\mathbf{U}_C \
\boldsymbol{\omega}_C
\end{bmatrix}
\in \mathbb{R}^6
]

For fine vertex (i\in C):

[
\mathbf{u}_i
============

\mathbf{U}_C
+
\boldsymbol{\omega}_C \times \mathbf{d}_i
]

where:

[
\mathbf{d}_i = \mathbf{r}_i - \mathbf{R}_C
]

This gives a local interpolation block:

[
\mathbf{u}_i
============

\begin{bmatrix}
I & -[\mathbf{d}*i]*\times
\end{bmatrix}
q_C
]

So (P) is made from blocks:

[
P_{iC}
======

\begin{bmatrix}
I & -[\mathbf{d}*i]*\times
\end{bmatrix}
]

Then:

[
A_c = P^T A_f P
]

This automatically produces effective coarse stiffnesses including bending/torsional response caused by the fine truss geometry.

This is probably the physically correct answer to your worry.

The expensive part is building (A_c), but you can do it offline or during topology setup. Runtime just applies the coarse correction.

---

# 16. But do not start with full multigrid

I would not implement multigrid first. It is more conceptual risk.

The first practical experiment should be:

[
\boxed{
\text{patch-Jacobi with core+halo and local-memory inner iterations}
}
]

Then compare against ordinary global Jacobi for the same time budget.

Measure:

```text
constraint violation after frame
energy drift
visual stiffness
kernel time
global memory traffic
```

Only after that, add a cheap coarse correction.

---

# 17. A realistic solver stack

Given your budget, I would aim for something like:

```text
per time step:

1. predict positions from velocity/external forces

2. patch-Jacobi pass
   nInner = 4

3. optional coarse correction
   one or two coarse Jacobi passes

4. patch-Jacobi pass
   nInner = 4

5. update velocities from position difference
```

This gives the shape:

[
\text{smoother} \rightarrow \text{coarse correction} \rightarrow \text{smoother}
]

which is exactly multigrid logic, but very cheap.

---

# 18. Patch-Jacobi versus global Jacobi: expected behavior

For local high-frequency errors, patch-Jacobi should win strongly.

Example: a stiff cluster/beam segment internally oscillates or violates distances. Local inner iterations fix it without rereading global memory.

For global bending of a long crane arm, patch-Jacobi still propagates slowly across patch boundaries. That is where coarse correction helps.

So the division of labor is:

```text
patch-Jacobi:
    local stiffness, local constraints, short wavelengths

coarse correction:
    global bending, global stretch, low-frequency modes
```

That is the right architecture.

---

# 19. Averaging overlapping vertices: yes, but weighted averaging is better

Uniform averaging:

[
x_i^{new}
=========

\frac{1}{N_i}
\sum_{p\ni i}
\tilde{x}_{i}^{(p)}
]

is a good starting point.

But I would quickly move to weighted averaging:

[
x_i^{new}
=========

\frac{
\sum_p w_{pi}\tilde{x}*i^{(p)}
}{
\sum_p w*{pi}
}
]

where (w_{pi}) depends on how central vertex (i) is in patch (p).

For example:

```text
core vertex:       w = 1.0
halo distance 1:   w = 0.25
halo distance 2:   w = 0.05
```

or use smooth graph-distance weights:

[
w(d)=\exp(-\alpha d^2)
]

This avoids noisy seams.

But again, the simplest robust version is:

```text
write only core
no averaging
```

I would test both.

---

# 20. The most promising concrete design

I would implement two variants and benchmark.

## Variant 1: core-write patch Jacobi

```text
patch = core + 1-ring halo
inner iterations = 4–8
write only core vertices
```

Runtime:

```text
for outerPass:
    patchRelaxCoreWrite(x_old, x_new)
    swap(x_old, x_new)
```

This is the simplest, probably fastest, least memory hungry.

## Variant 2: overlapping prediction + gather averaging

```text
patch = core + halo
inner iterations = 4–8
write all patch predictions to patch buffer
assemble by weighted averaging
```

Runtime:

```text
for outerPass:
    patchRelaxWritePredictions(x_old, patch_x)
    assembleWeighted(patch_x, x_new)
    swap(x_old, x_new)
```

This is smoother and more symmetric, but uses more memory bandwidth.

---

# 21. Important stability issue

If you do many inner iterations with stale boundary conditions, the patch interior may become too consistent with an outdated boundary. When assembled, neighboring patches may disagree.

Therefore:

```text
too many inner iterations can become worse
```

I would expect an optimal inner count around:

```text
3–8
```

not 20 or 50.

This is similar to domain decomposition: local solves are useful, but without frequent boundary exchange they overfit local boundary assumptions.

So tune:

[
n_{\text{inner}}
]

against:

[
n_{\text{outer}}
]

For fixed cost, compare:

```text
1 × 12
2 × 6
3 × 4
4 × 3
```

I suspect `3 × 4` or `2 × 6` may be better than `1 × 12`.

---

# 22. My final recommendation

The practical design I would pursue is:

[
\boxed{
\text{graph patches with core+halo}
}
]

[
\boxed{
\text{local-memory vertex-gather Jacobi/PBD inner sweeps}
}
]

[
\boxed{
\text{core-only write first; weighted-overlap assembly second}
}
]

[
\boxed{
\text{optional two-level coarse correction using } A_c=P^T A_f P
}
]

For multigrid, the key is not to hand-design equivalent coarse springs. Instead define coarse deformation modes (P), then compute:

[
A_c = P^T A_f P
]

For truss/girder structures, the best coarse (P) probably uses **rigid cluster DOFs**:

[
\mathbf{u}_i
============

\mathbf{U}_C
+
\boldsymbol{\omega}_C\times(\mathbf{r}_i-\mathbf{R}_C)
]

because that preserves bending/torsion-like behavior much better than translation-only cluster nodes.

But as the first real-time GPU improvement, I would absolutely start with patch-Jacobi. It keeps the spirit of your current solver:

```text
sparse
parallel
few kernels
no reductions
no global factorization
no large fill-in
mostly local memory reuse
```

That is much closer to a game-engine solver than CG or Cholesky.

---

# USER

No, I think I want the averaging of local patches. You say that the weights for averaging should be non-uniform. I agree but rather than constant weights I was thinking we can avrage proepry by decomposition of the constrains on sub-patches.

What I mean is that the stiffnes matrix of projection dynamics have summ of neigbor masses sum_i K_ij on diagonal. This is the natural weighting. If we solve the truss without those additional external bonds, we effectively lower the stiffnes. I think the optimal weight for averaging should be derived from this - that together when combined displacement form one patch and the other patch we recover rughly the action of the global jacobi step.  This should have rigorous analytical solution, not guess random weight coefficient.


Ad multi grid - perhaps I can do it within one kernel. My patch local jacobii kernel can compute the trasform (projection of fine-truss stiffness onto coarse frame. 
 
how many coarse points should my coarse-grained matrix have. I mean f I have blocks e.g. 64 vertexes in my workgroup for the block-jacobi, the coarse representation should not have more than e.g. 8 (like cube) otherwise it is not woth the complexity if we do not reduce it enough.

I still do not see clearly how this Garlekin works, I mean you write the matrix ewuations it is to abstract, I want to see geometry and physics behind it, what the new coarse grained DOFs represent, where are located (are these even points or rather like vibration mdes, some elipsoide or elasticity tensor?), what physically represnt the transformation matrix R and P in this coarsegraining.

---

# Chat GPT 5.5

Yes — your intuition is right. The averaging weights should not be arbitrary “fade to boundary” coefficients. They should come from a **partition of the global stiffness/Jacobi denominator**. Then the sum of patch contributions reconstructs, as closely as possible, the global Jacobi update.

Let me start from the cleanest formulation.

---

# 1. Global Jacobi for a truss

For a linearized projective-dynamics/truss step, write the system as

[
A x = b
]

with

[
A = M/\Delta t^2 + K
]

where (K) is the spring/truss stiffness matrix.

For scalar notation first, Jacobi is:

[
x_i^{new}
=========

x_i
+
\omega
\frac{r_i}{A_{ii}}
]

where

[
r_i = b_i - \sum_j A_{ij}x_j
]

For vector vertices, (x_i\in\mathbb{R}^3), the diagonal block is often approximately scalar:

[
A_{ii}
======

\left(
m_i/\Delta t^2
+
\sum_{j\in N(i)} k_{ij}
\right) I_3
]

or, for directional springs, a (3\times 3) block:

[
A_{ii}
======

m_i/\Delta t^2 I_3
+
\sum_{j\in N(i)}
k_{ij},\mathbf{n}*{ij}\mathbf{n}*{ij}^T
]

The essential point is:

[
A_{ii}
======

\text{mass term}
+
\text{sum of stiffness contributions touching vertex }i
]

So the natural Jacobi denominator is not a geometric weight. It is the **local mechanical stiffness seen by the vertex**.

---

# 2. Patch-local Jacobi misses outside stiffness

Suppose patch (p) contains vertex (i), but only some of its incident constraints. Define the local patch matrix contribution:

[
A^{(p)}
]

such that

[
A \approx \sum_p A^{(p)}
]

If every constraint is assigned to one or more patches with appropriate splitting weights, then this decomposition can be exact:

[
A = \sum_p A^{(p)}
]

Now for vertex (i), the patch-local Jacobi denominator is

[
D_i^{(p)} = A_{ii}^{(p)}
]

while the global denominator is

[
D_i = A_{ii} = \sum_p A_{ii}^{(p)}
]

If patch (p) only sees internal constraints, then

[
D_i^{(p)} < D_i
]

and a local Jacobi step inside that patch would over-move (i), because it divides by too small a stiffness.

So you are right: if we solve the truss inside a patch without external bonds, we effectively lower the stiffness. Averaging must compensate for that.

---

# 3. Deriving patch recombination from Jacobi

The global Jacobi correction is:

[
\Delta x_i
==========

D_i^{-1} r_i
]

Now decompose the residual into patch residuals:

[
r_i = \sum_p r_i^{(p)}
]

where

[
r^{(p)} = b^{(p)} - A^{(p)}x
]

If we want the patch corrections to recombine into the global Jacobi correction, then we want:

[
\Delta x_i
==========

D_i^{-1}
\sum_p r_i^{(p)}
]

A patch computes something like:

[
\Delta x_i^{(p)}
================

\left(D_i^{(p)}\right)^{-1}
r_i^{(p)}
]

To recombine these into the global Jacobi correction, multiply each patch correction by:

[
W_i^{(p)}
=========

D_i^{-1}D_i^{(p)}
]

because then:

[
\sum_p
W_i^{(p)}
\Delta x_i^{(p)}
================

\sum_p
D_i^{-1}D_i^{(p)}
\left(D_i^{(p)}\right)^{-1}
r_i^{(p)}
]

# [

D_i^{-1}
\sum_p r_i^{(p)}
]

# [

D_i^{-1}r_i
]

So the rigorous scalar weight is:

[
\boxed{
W_i^{(p)}
=========

\frac{D_i^{(p)}}{D_i}
}
]

For vector/block diagonal stiffness:

[
\boxed{
W_i^{(p)}
=========

D_i^{-1}D_i^{(p)}
}
]

where (D_i) and (D_i^{(p)}) are (3\times 3) SPD matrices.

This is exactly the analytical solution you were looking for.

The physical meaning is simple:

> A patch should contribute to vertex (i) in proportion to how much of the total mechanical stiffness around (i) is represented inside that patch.

---

# 4. This is better than averaging positions

Strictly speaking, the mechanically correct quantity to average is not the final patch position, but the **patch correction** or **patch force/preconditioned residual**.

Instead of:

[
x_i^{new}
=========

\frac{\sum_p w_i^{(p)}\tilde{x}_i^{(p)}}{\sum_p w_i^{(p)}}
]

I would write:

[
x_i^{new}
=========

x_i
+
\sum_p
W_i^{(p)}
\Delta x_i^{(p)}
]

where

[
\Delta x_i^{(p)}
================

\tilde{x}_i^{(p)} - x_i
]

and

[
W_i^{(p)}
=========

D_i^{-1}D_i^{(p)}
]

This matters because different patches start from the same global (x_i). What they produce is a suggested displacement. These displacements should be combined according to stiffness share.

So in code, the assembly is:

```cpp
dx_i = 0;
for each patch p containing vertex i:
    dx_i += W_ip * (x_patch[p][loc_i] - x_old[i]);
x_new[i] = x_old[i] + omega * dx_i;
```

If (D_i) is scalar, this is very cheap:

```cpp
w = D_patch_ip / D_global_i;
dx += w * (x_patch - x_old);
```

---

# 5. What exactly is (D_i^{(p)})?

For isotropic PD springs, each edge ((i,j)) contributes approximately (k_{ij} I_3) to both endpoint diagonals.

Then:

[
D_i =
m_i/\Delta t^2
+
\sum_{j\in N(i)} k_{ij}
]

Now define edge-to-patch ownership coefficients:

[
\alpha_{e}^{(p)}
]

such that

[
\sum_{p\ni e} \alpha_e^{(p)} = 1
]

Then patch (p)'s contribution to the diagonal of vertex (i) is:

[
D_i^{(p)}
=========

\alpha_m^{(p)} m_i/\Delta t^2
+
\sum_{e=(i,j)\in p}
\alpha_e^{(p)} k_e
]

The mass term also has to be partitioned:

[
\sum_{p\ni i} \alpha_{m,i}^{(p)} = 1
]

Usually:

[
\alpha_{m,i}^{(p)}
==================

\frac{D_{i,\text{springs}}^{(p)}}{D_{i,\text{springs}}}
]

or just uniformly among patches containing (i). But the first option is more mechanically consistent.

Then:

[
D_i = \sum_p D_i^{(p)}
]

and the weights satisfy:

[
\sum_p W_i^{(p)} = 1
]

in scalar case.

So this is not heuristic averaging. It is a **partition of the Jacobi preconditioner**.

---

# 6. Important distinction: patch decomposition of constraints, not only vertices

This is probably the core conceptual point.

If a vertex belongs to several patches, that alone is not enough. You also need to know how each **constraint/edge** contributes to patches.

For each edge (e=(i,j)), there are several options.

### Edge assigned to one patch

[
\alpha_e^{(p)} = 1
]

for one owner patch, zero for others.

Then the patch decomposition is exact but non-overlapping at the constraint level.

This is simple, but some patches may not feel boundary constraints symmetrically.

### Edge duplicated across patches but split

If edge (e) appears in multiple patches, split it:

[
\alpha_e^{(p)} =
\frac{1}{N_e}
]

where (N_e) is the number of patches containing that edge.

Then:

[
\sum_p \alpha_e^{(p)} = 1
]

This prevents duplicated stiffness.

### Edge duplicated with stiffness-aware shape functions

More generally:

[
\alpha_e^{(p)}
==============

\frac{\phi_p(i)\phi_p(j)}
{\sum_q \phi_q(i)\phi_q(j)}
]

where (\phi_p(i)) is a patch membership function.

This is elegant. The patch has a smooth partition of unity over vertices, and edge contribution follows from endpoint membership.

If vertex (i) is central in patch (p), (\phi_p(i)) is large. If it is only halo, (\phi_p(i)) is small.

Then the stiffness split is automatic and exact after normalization.

But again, this is not arbitrary weighting: it is a partition of the energy.

---

# 7. Energy-based patch decomposition

The original truss energy is:

[
E(x)
====

\frac12
\sum_e
k_e
C_e(x)^2
+
\frac12
\sum_i
m_i/\Delta t^2
|x_i-y_i|^2
]

Patch energy can be defined as:

[
E^{(p)}(x)
==========

\frac12
\sum_e
\alpha_e^{(p)} k_e C_e(x)^2
+
\frac12
\sum_i
\alpha_{m,i}^{(p)}
m_i/\Delta t^2
|x_i-y_i|^2
]

with

[
\sum_p \alpha_e^{(p)} = 1
]

and

[
\sum_p \alpha_{m,i}^{(p)} = 1
]

Then:

[
E(x) = \sum_p E^{(p)}(x)
]

This is the rigorous way to decompose constraints into sub-patches.

The patch-local Jacobi uses the diagonal of the local Hessian of (E^{(p)}). The recombination weights come from the same decomposition.

So your intuition can be formalized as:

[
\boxed{
\text{split the energy, not merely the vertices}
}
]

---

# 8. But there is one catch with multiple inner iterations

The derivation above is exact for **one Jacobi step**.

If each patch does several inner Jacobi steps, then the patch output is no longer a linear one-step residual contribution. It is approximately:

[
\Delta x^{(p)}
\approx
B_p r^{(p)}
]

where (B_p) is a local approximate inverse, not just (D_p^{-1}).

Then the exact recombination weight is no longer simply:

[
D^{-1}D_p
]

However, (D^{-1}D_p) remains the correct first-order/stiffness-consistent scaling.

A better interpretation is:

[
x^{new}
=======

x
+
\omega
\sum_p
R_p^T
B_p
R_p
r
]

where (B_p) is your local patch smoother. If the patch solve uses only the patch energy (E^{(p)}), then the patch correction already has the right stiffness scale. You should still blend corrections according to the share of vertex diagonal represented in the patch.

So for multiple inner steps, use:

[
\boxed{
W_i^{(p)} = D_i^{-1}D_i^{(p)}
}
]

as a stable physically motivated recombination, but tune global damping (\omega).

---

# 9. For PBD nonlinear constraints

For nonlinear distance constraints, the stiffness matrix is local and configuration-dependent.

The same logic still works if you use the diagonal approximation actually used by the constraint solver.

For a distance constraint (e=(i,j)), the PBD denominator is:

[
w_i + w_j
]

where (w_i=1/m_i). In XPBD, stiffness/compliance also enters.

But for assembling patch predictions, it is usually better to use the **linearized effective stiffness**:

[
D_i =
m_i/\Delta t^2
+
\sum_{e\ni i} k_e
]

or directional:

[
D_i =
m_i/\Delta t^2I
+
\sum_{e\ni i} k_e n_e n_e^T
]

This gives stable weighting independent of instantaneous constraint violation.

---

# 10. Now multigrid: what do coarse DOFs physically represent?

Your discomfort with abstract (P^TAP) is justified. Let’s make it geometric.

A coarse DOF is not necessarily “a point”. It is a **deformation pattern of a group of fine vertices**.

Think of a patch/block of 64 fine vertices. We choose, for example, 8 coarse handles. Each coarse handle controls a smooth displacement field inside the block.

The prolongation matrix (P) answers this question:

> If I move the coarse handles by (X_C), how do all fine vertices move?

So (P) is not mysterious. It is just interpolation.

---

# 11. Example: 8 coarse points inside a patch

Suppose a patch has 64 vertices with rest positions (\mathbf{r}_i).

Put 8 coarse nodes around the patch, for example like cube corners of the patch bounding box:

[
\mathbf{R}_A,\quad A=1,\dots,8
]

Each coarse node has displacement:

[
\mathbf{U}_A \in \mathbb{R}^3
]

Each fine vertex displacement is interpolated:

[
\mathbf{u}_i
============

\sum_{A=1}^{8}
\phi_A(\mathbf{r}_i)\mathbf{U}_A
]

where (\phi_A) are shape functions.

For a regular box, (\phi_A) can be trilinear interpolation weights. For an irregular patch, use distance/RBF/geodesic weights.

This equation is the physical meaning of (P).

[
\boxed{
P_{iA} = \phi_A(\mathbf{r}_i) I_3
}
]

So the coarse variables are literal control-point displacements, like an embedded deformation cage.

They are not necessarily real mass points of the truss. They are deformation handles.

---

# 12. What does (R) physically represent?

Restriction (R) maps fine residual forces to coarse forces.

If

[
\mathbf{u}_i
============

\sum_A \phi_A(i)\mathbf{U}_A
]

then virtual work must match:

[
\sum_i \mathbf{f}_i\cdot \delta \mathbf{u}_i
============================================

\sum_A \mathbf{F}_A\cdot \delta \mathbf{U}_A
]

Substitute:

[
\delta \mathbf{u}_i
===================

\sum_A \phi_A(i)\delta \mathbf{U}_A
]

Then:

[
\sum_i \mathbf{f}_i\cdot
\sum_A \phi_A(i)\delta\mathbf{U}_A
==================================

\sum_A
\left(
\sum_i \phi_A(i)\mathbf{f}_i
\right)
\cdot
\delta\mathbf{U}_A
]

Therefore:

[
\boxed{
\mathbf{F}_A
============

\sum_i
\phi_A(i)\mathbf{f}_i
}
]

This is restriction:

[
\boxed{
R = P^T
}
]

So (R) is not abstract either. It is the operation:

> Collect fine residual forces onto coarse handles using the same shape functions.

This preserves work/energy consistency.

---

# 13. What does (A_c=P^TAP) physically mean?

The fine elastic energy is:

[
E_f(\mathbf{u})
===============

\frac12
\mathbf{u}^T A_f \mathbf{u}
]

But on the coarse level we only allow fine displacements of the form:

[
\mathbf{u}=P\mathbf{U}
]

So the energy becomes:

[
E_c(\mathbf{U})
===============

\frac12
(P\mathbf{U})^T A_f (P\mathbf{U})
]

# [

\frac12
\mathbf{U}^T
(P^TA_fP)
\mathbf{U}
]

Therefore:

[
A_c=P^TA_fP
]

Physically:

> Move the coarse handles. Interpolate that motion to the fine truss. Measure how much fine spring energy this causes. The matrix that gives that energy is the coarse stiffness matrix.

So coarse stiffness is not guessed. It is measured by asking: “If this coarse handle moves, how much do the original fine springs stretch?”

This automatically includes bending stiffness, torsion, axial stiffness, weird truss geometry, fractal structure, everything — as long as the deformation can be represented by your coarse handles.

---

# 14. Edge-level construction of coarse stiffness

You do not need to build the full fine matrix explicitly.

For each fine spring/constraint (e=(i,j)), rest direction (\mathbf{n}_{ij}), stiffness (k_e), the linearized spring energy is:

[
E_e
===

\frac12 k_e
\left[
\mathbf{n}_{ij}\cdot(\mathbf{u}_i-\mathbf{u}_j)
\right]^2
]

Now substitute coarse interpolation:

[
\mathbf{u}_i
============

\sum_A \phi_A(i)\mathbf{U}_A
]

[
\mathbf{u}_j
============

\sum_A \phi_A(j)\mathbf{U}_A
]

Then:

[
\mathbf{u}_i-\mathbf{u}_j
=========================

\sum_A
\left[
\phi_A(i)-\phi_A(j)
\right]\mathbf{U}_A
]

So this fine spring contributes to coarse stiffness between coarse handles (A,B):

[
K_{AB}
+=
k_e
\left[\phi_A(i)-\phi_A(j)\right]
\left[\phi_B(i)-\phi_B(j)\right]
\mathbf{n}*{ij}\mathbf{n}*{ij}^T
]

This is the geometric meaning of Galerkin.

Each fine edge says:

> If coarse handle (A) and (B) move, how much does this fine spring stretch?

That gives the coarse stiffness.

For isotropic PD Laplacian springs, replace (\mathbf{n}\mathbf{n}^T) by (I_3):

[
K_{AB}
+=
k_e
\left[\phi_A(i)-\phi_A(j)\right]
\left[\phi_B(i)-\phi_B(j)\right]
I_3
]

This is very implementable.

---

# 15. Coarse mass matrix

The inertial term also coarsens naturally.

Fine inertial energy:

[
E_m
===

\frac12
\sum_i
m_i/\Delta t^2
|\mathbf{u}_i-\mathbf{y}_i|^2
]

The coarse mass contribution is:

[
M_{AB}^{c}
==========

\sum_i
m_i
\phi_A(i)\phi_B(i)
]

Often you lump it:

[
M_A^c
=====

\sum_i m_i \phi_A(i)
]

or

[
M_A^c
=====

\sum_i m_i \phi_A(i)^2
]

depending on whether you want exact Galerkin mass or diagonal/lumped mass.

For game physics, lumped coarse mass is fine.

---

# 16. Are coarse DOFs points, modes, or tensors?

They can be any of these. It depends on the chosen basis (P).

### Coarse points / handles

DOF:

[
\mathbf{U}_A
]

Meaning: displacement of a coarse control point.

Fine interpolation:

[
\mathbf{u}_i=\sum_A\phi_A(i)\mathbf{U}_A
]

Best first implementation.

### Rigid cluster modes

DOF per cluster:

[
(\mathbf{U}_C,\boldsymbol{\omega}_C)
]

Meaning: translation and rotation of a cluster.

Fine interpolation:

[
\mathbf{u}_i
============

\mathbf{U}_C+\boldsymbol{\omega}_C\times(\mathbf{r}_i-\mathbf{R}_C)
]

Good for beams/girders because it captures bending/torsion better.

### Vibration modes

DOF:

[
q_\alpha
]

Fine interpolation:

[
\mathbf{u}*i=\sum*\alpha q_\alpha \mathbf{v}_{\alpha i}
]

Meaning: amplitude of selected low-frequency deformation modes.

Most accurate but expensive/offline.

### Elasticity tensor / homogenized element

DOF: affine deformation gradient

[
F
]

Fine interpolation:

[
\mathbf{u}_i = \mathbf{U} + B(\mathbf{r}_i-\mathbf{R})
]

where (B) is a (3\times3) small deformation matrix.

This is like treating the patch as a small continuum element.

Good for dense lattice blocks; maybe less ideal for open trusses unless cluster is sufficiently connected.

---

# 17. For your 64-vertex patch, what coarse representation?

You said: if patch has about 64 vertices, coarse representation should be around 8 points, otherwise it does not reduce enough.

That is a good instinct.

For a patch with a roughly 3D spatial extent, 8 coarse handles are natural:

```text
cube-corner handles
3 DOF each
8 × 3 = 24 coarse DOF
```

Fine patch:

```text
64 vertices × 3 = 192 fine DOF
```

So the reduction is:

[
192 \rightarrow 24
]

which is meaningful.

But if the patch is a long girder segment, 8 cube corners may be wasteful. Better might be:

```text
2 or 3 cross-section frames along the girder
each frame has translation + rotation
```

For example:

```text
2 rigid frames × 6 DOF = 12 coarse DOF
```

or

```text
3 rigid frames × 6 DOF = 18 coarse DOF
```

For long beams/trusses, rotational DOFs are very valuable.

---

# 18. Can this be done inside one kernel?

Partially yes, but with a caveat.

Inside a patch kernel, you can compute a **local coarse matrix**:

[
A_c^{(p)} = P_p^T A_p P_p
]

where (A_p) is the patch stiffness.

This is local and can be accumulated in local memory.

For 8 coarse points:

[
n_c = 8 \times 3 = 24
]

The matrix is:

[
24\times24
]

That is 576 floats. Not crazy.

If symmetric, 300 floats. Fine.

But then what do you do with it?

Options:

### Local coarse solve inside the patch

You solve the patch’s coarse problem locally and use it as a better patch correction.

This is not global multigrid. It is a **patch coarse-mode smoother**.

That may still be very useful.

Each patch correction includes not only vertex-wise Jacobi but also low-dimensional coherent modes inside the patch.

### Global coarse solve

To solve long wavelengths across the whole spacecraft, coarse nodes must be shared between patches, or assembled into a global coarse graph. That needs global communication and probably another kernel pass.

So:

[
\text{local coarse inside patch}
\neq
\text{global multigrid}
]

Local coarse modes improve intra-patch behavior. True multigrid requires communication between patches at the coarse level.

---

# 19. A very attractive hybrid: patch-local coarse modes

This may be exactly what fits your budget.

Inside each patch:

1. Load fine vertices.
2. Do a few fine Jacobi iterations.
3. Compute residual.
4. Restrict residual to 8 coarse handles:

[
r_c = P^T r_f
]

5. Solve tiny coarse system:

[
A_c \delta U = r_c
]

6. Prolongate:

[
\delta x_f = P\delta U
]

7. Maybe do another fine Jacobi iteration.
8. Write patch prediction.
9. Global assembly averages patch predictions with stiffness partition weights.

This stays inside one patch kernel.

It does not solve global long-wavelength modes across many patches, but it does solve the slowest modes **inside each patch**, such as coherent bending of the local truss chunk.

This can make patch-Jacobi much stronger without full Cholesky.

The local coarse solve is tiny:

```text
8 handles × 3 DOF = 24 DOF
```

A dense (24\times24) solve in local memory may be acceptable if done once per patch per outer pass. But if even that is too much, do Jacobi on the coarse matrix too.

---

# 20. But true long-wavelength multigrid needs shared coarse nodes

If you want global crane-arm bending to relax quickly, coarse DOFs must connect across patches.

That means coarse handles should not be private to each fine patch. They should form a lower-resolution global graph.

For example:

```text
fine vertices:         100000
patches:               2000
coarse nodes:           5000 or 10000
coarse constraints:     derived from fine stiffness
```

Then a coarse solver step propagates information over longer distances.

But that is more infrastructure.

A compromise:

```text
one global coarse node per patch core
or one per cluster of fine vertices
```

Then the patch kernel contributes to coarse residuals/stiffness, and a separate coarse Jacobi kernel solves the coarse graph.

This is a true two-level method.

---

# 21. Physical picture of Galerkin with a simple spring chain

Take a fine chain:

```text
o--o--o--o--o
0  1  2  3  4
```

Coarse nodes at fine vertices 0, 2, 4:

```text
O-----O-----O
0     2     4
```

Fine vertex 1 is interpolated:

[
u_1 = \frac12 U_0 + \frac12 U_2
]

Fine vertex 3:

[
u_3 = \frac12 U_2 + \frac12 U_4
]

Now ask: if coarse node (U_0) moves and (U_2,U_4) stay fixed, how much do the original fine springs stretch?

That gives the coarse stiffness between (U_0) and (U_2).

For two equal springs in series, the equivalent stiffness becomes:

[
k_{\text{coarse}} = \frac{k}{2}
]

You do not insert this formula manually. It falls out of:

[
A_c = P^TAP
]

So Galerkin is just automatic equivalent-spring computation for arbitrary networks and arbitrary interpolation.

---

# 22. Important warning: Galerkin with bad (P) gives bad physics

The coarse operator is only as good as the coarse deformation basis.

If (P) cannot represent bending, then (A_c) cannot preserve bending.

For a girder, translation-only coarse points may represent axial stretching and rough bending, but not torsion well.

Rigid-frame coarse DOFs are better.

For a spacecraft truss, I would consider:

```text
coarse nodes = structural junctions or cluster centers
coarse DOFs = translation only initially
later: add rotational DOFs for elongated girder clusters
```

The main practical question is not (A_c=P^TAP). That part is clean.

The practical question is:

[
\boxed{
\text{what low-dimensional deformation fields should }P\text{ contain?}
}
]

---

# 23. Direct answer to your patch averaging proposal

Yes, the optimal averaging can be derived from constraint decomposition.

For each patch (p), define a local energy (E^{(p)}) such that:

[
E = \sum_p E^{(p)}
]

Then define:

[
D_i^{(p)} =
\frac{\partial^2 E^{(p)}}{\partial x_i^2}
]

and

[
D_i =
\sum_p D_i^{(p)}
]

Then assemble patch corrections as:

[
\boxed{
x_i^{new}
=========

x_i
+
\omega
\sum_{p\ni i}
D_i^{-1}D_i^{(p)}
\left(
\tilde{x}_i^{(p)} - x_i
\right)
}
]

Scalar isotropic version:

[
\boxed{
x_i^{new}
=========

x_i
+
\omega
\sum_{p\ni i}
\frac{D_i^{(p)}}{D_i}
\left(
\tilde{x}_i^{(p)} - x_i
\right)
}
]

with:

[
D_i^{(p)}
=========

\alpha_{m,i}^{(p)}m_i/\Delta t^2
+
\sum_{e\ni i}
\alpha_e^{(p)}k_e
]

and:

[
D_i
===

m_i/\Delta t^2
+
\sum_{e\ni i}k_e
]

This gives you the rigorous stiffness-weighted patch average.

---

# 24. Direct answer to “what are coarse DOFs?”

For your problem, I would think of them in this order:

### First implementation

Coarse DOFs are **control-point displacements**.

```text
8 coarse handles per 64-vertex patch
each handle has Vec3 displacement
fine vertex displacement = weighted interpolation of handle displacements
```

They are like an embedded deformation cage.

### Better for girders

Coarse DOFs are **rigid-frame motions**.

```text
each coarse frame has translation Vec3 + small rotation Vec3
fine vertex displacement = frame translation + rotation cross offset
```

This captures bending/torsion with fewer DOFs.

### Most accurate but expensive

Coarse DOFs are **selected vibration/deformation modes**.

```text
q0 = first bending mode amplitude
q1 = second bending mode amplitude
q2 = torsion mode amplitude
...
```

This is elegant but needs modal preprocessing and is less flexible for topology changes.

---

# 25. The concrete algorithm I would try

For patch averaging:

```cpp
// Patch kernel
for each patch p:
    load x_old for patch vertices
    do nInner local Jacobi/PBD steps using patch-split stiffness alpha_e_p*k_e
    write x_patch[p][a]

// Assembly kernel
for each global vertex i:
    Vec3 dx = 0;
    Mat3 Dinv = inverse(D_global[i]); // or scalar reciprocal
    for each incident patch p:
        int a = local_index(i,p);
        Vec3 dxi = x_patch[p][a] - x_old[i];
        dx += Dinv * D_patch[p][a] * dxi;
    x_new[i] = x_old[i] + omega * dx;
```

For scalar diagonal:

```cpp
dx += (D_patch[p][a] * invD_global[i]) * dxi;
```

This is probably the right rigorous assembly.

For coarse patch modes later:

```cpp
inside patch:
    after some local Jacobi:
        compute fine residual r_f
        r_c = P^T r_f
        solve A_c du_c = r_c
        x_f += P du_c
```

This is still patch-local and may fit into one kernel if (n_c\leq 24).

---

The main conceptual correction is this:

[
\textbf{Do not average patches by geometric membership.}
]

Average their **corrections** by their share of the **local Jacobi denominator / stiffness diagonal**.

That makes the patch method reduce to ordinary global Jacobi in the one-inner-step limit, while still allowing you to do several useful local-memory iterations inside each patch.


