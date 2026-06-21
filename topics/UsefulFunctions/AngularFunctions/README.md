# AngularFunctions

Efficient angular potential functions for atomistic simulations.

## Files

| File | Role |
|------|------|
| `AngularFunctions.md` | Design discussion: expressing angular potentials for sp1, sp2, sp3 geometries using complex numbers (2D) and quaternions (3D) instead of storing port vectors |

## Concept

Standard angular potentials (like in CHARMM, AMBER) store direction vectors
("ports") for each preferred neighbor direction and sum cosine angles. This
is memory-intensive and requires many trigonometric calls.

The idea explored here: use symmetry to compute angular potentials more cheaply.
- **sp2 (trigonal)**: raise a complex number (unit vector in XY plane) to the
  3rd power — the 3-fold symmetry is encoded in z^3, avoiding 3 separate port vectors
- **sp3 (tetrahedral)**: use unitary quaternions — a polynomial/power of a
  quaternion may encode the 4 tetrahedral poles without storing 4 vectors

This is related to the concept of "multipole" or "symmetry-adapted" angular
potentials used in some modern force fields.
