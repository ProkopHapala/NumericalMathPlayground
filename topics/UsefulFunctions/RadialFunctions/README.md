# RadialFunctions

Compact-support radial basis functions for atomistic simulations — smooth,
C2-continuous at the cutoff, with tunable peak width.

## Files

| File | Role |
|------|------|
| `plot_radial.py` | Interactive matplotlib explorer for radial basis functions: visualize shape, cutoff, derivatives, and compare candidates |
| `RadialFunctions.md` | Design discussion: candidate radial functions with finite support, C2 continuity at cutoff, and avoidance of costly sqrt() |

## Requirements

The radial functions should:
- Have **finite support** — exactly zero beyond cutoff R_c
- Be **C2 continuous** at the cutoff (zero value, zero first and second derivative)
- Have a **blunt maximum** at zero (not a cusp)
- Have **tunable width** of the peak around zero
- Avoid **sqrt()** where possible — prefer dependence on r^2 over |r|

Two categories:
- **r-dependent**: require sqrt(|r|), more flexible shape
- **r^2-dependent**: only use r^2 = x^2 + y^2 + z^2, cheaper to evaluate

## Usage

```bash
cd topics/UsefulFunctions/RadialFunctions
python plot_radial.py
```
