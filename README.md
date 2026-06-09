# NumericalMathPlayground

Testing and deriving different things from numerical math mostly for computational chemistry, quantum mechanics and forcefield development.

## Contents

### topics/
Research documentation and utilities for mathematical functions used in computational chemistry.

- **AngularFunctions/** - Mathematical derivations for angular potentials (sp1, sp2, sp3 hybridizations). Includes AI conversations about complex number tricks, quaternion approaches, spherical harmonics, and topological methods for tetrahedral symmetry.
- **RadialFunctions/** - C² continuous radial functions optimized for GPU acceleration. Includes polynomial families, Lorentzian alternatives, and force derivative optimizations.
  - `plot_radial.py` - Interactive matplotlib tool with sliders to visualize and tune radial function parameters.

### web/
Interactive WebGL-based visualization tools for exploring angular functions.

- **angular-plotter/** - 2D angular function plotter. Visualizes functions on planes with optional radial envelopes, multiple colormaps, and preset angular functions (sp1, sp2, sp3, octahedral, etc.).
- **angular-plotter-3d/** - Advanced 3D visualizer with multiple render modes:
  - 2D slice through 3D space
  - Volume min/max projection
  - Isosurface rendering
  - Filament visualization (zero-crossing lines)
  - Chiral filament (colors tetrahedral vs anti-tetrahedral)
  - Zero planes (real/imaginary surfaces)
  - Quaternion trackball camera control
