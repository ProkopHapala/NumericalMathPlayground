# Agent Guidelines for NumericalMathPlayground

## Core Philosophy: Playground for Didactic Demos

This repo explores numerical math ideas through small, clear, reusable demos — primarily in **Python** and **JavaScript (+WebGL shaders)**, occasionally **pyOpenCL**. Prioritize clarity and correctness over polishing a single monolithic application.

1. **Debuggability:** Transparent, inspectable code. Traceability beats UX; never hide bugs with silent fallbacks.
2. **Simplicity:** Direct logic. Avoid unnecessary branching, micro-abstractions, and defensive boilerplate.
3. **Modularity:** Related demos belong in one script with CLI options, not scattered across dozens of single-purpose files.

## Rule 1 — Fail Loudly

- **No silent fallbacks.** Unexpected states must crash with a full stack trace. Avoid broad `try/except` that mask root causes.
- **Fix the root cause.** Do not apply "quick-fixes" that hide underlying problems.

## Rule 2 — Surgical Edits & Simplicity

- **Minimum Intervention.** Touch only what the task requires. No unrelated formatting, cleanup, or aesthetic edits.
- **No Guessing.** Stop and ask if requirements, behavior, or architecture are ambiguous.
- **Report Immediately.** Write ambiguities, unexpected errors, or risky decisions in the chat. No silent workarounds.
- **Checkpoint & Preserve.** After significant changes, summarize what changed, what was verified, and what remains. Comment out old or experimental code with `# TODO` or `# DEBUG` instead of deleting it.

## Rule 3 — Three-Layer Architecture (STRICT)

Every topic must split code into three layers. 

1. **Core module** (e.g. `Truss.py`, `TrussSolver.py`) — math, algorithms, data structures. Pure functions, no I/O, no plotting.
2. **Plotting module** (e.g. `TrussPlotting.py`) — all visualization functions. Shared and reusable across scripts.
3. **Scripts** (e.g. `demo_BlockJacobiTruss.py`) — thin CLI wrapper. Parses command-line arguments, calls core + plotting, prints results. **No complex functions defined here.** If a helper is needed by more than one script, it goes in a module.

**DO NOT** write complex single-purpose code/functions in test script, try to generalize and refactor into core or plotting modules. 

- **Reusability First.** Before writing new code, check existing modules for functions you can reuse or generalize.
- **Generalization Over Duplication.** If an existing function almost fits, prefer generalizing it with a new parameter over copy-pasting.
- **Consolidate.** If you find yourself making `demo_a.py`, `demo_b.py`, `demo_c.py` for variants of the same idea, merge them into `demo.py --mode a|b|c`.

## Rule 4 — Validate & Test

- **Sanity Checks.** Assert invariants and check for `NaN`, `inf`, or unexpected zeros in intermediate results.
- **Test on Completion.** Run demos after modifications. Do not claim something works without verifying it.
- **Physical & Analytical Parity.** Define how you will verify correctness before coding: compare against analytical solutions, conservation laws, symmetry checks, or reference implementations.
- **Foreground Execution.** Run tests synchronously with full output visible. Never pipe through `tail`/`head` to hide output.

## Rule 5 — Performance Guidelines

- **Vectorized Python.** Use NumPy for number crunching. Avoid Python loops in hot paths.
- **GPU / Shaders.** Offload heavy parallel work to WebGL shaders (JS) or pyOpenCL when genuinely needed, but do not over-engineer. Keep the Python orchestration minimal.
- **Memory.** Prefer flat, contiguous arrays. Preallocate and reuse buffers. Be explicit about dtypes and shapes.

## Rule 6 — Concise Style

- **No Micro-Abstractions.** Do not create one-line wrappers. If it is simple, inline it.
- **Clean Interfaces.** Use default named arguments to avoid long call strings. Group related state into structs/dicts or classes.
- **Compact Layout.** Prefer compact code; avoid excessive blank lines. Assume infinite line width; do not wrap expressions in ways that hurt readability.
- **Naming & Comments.** Use short, clear names for math symbols (`E_tot`, `T_ij`, `m_i`). Comments explain the non-obvious: assumptions, intent, mental models, and the theory or derivation behind the code. Supplement with equations where helpful. Good code is concise and self-explanatory; do not waste space stating the obvious or comment every line like an elementary-school exercise. Place inline comments behind the line.

## Rule 7 — Output Location

- **All generated outputs** (plots, .xyz files, logs, debug artifacts) go to `debug/` — never to source directories.
- `debug/` is gitignored. This keeps the repo clean while making results inspectable.
- Scripts should default to `debug/` for any output path.
