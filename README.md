# Stellar Pulsations

Numerical study of the normal modes of oscillation of compact stars, in both
the **Newtonian** and **General Relativistic (GR)** frameworks. This
repository contains the full computational code developed for the
undergraduate thesis:

> **Θεωρητική και Υπολογιστική Μελέτη των Κανονικών Τρόπων Ταλάντωσης
> Συμπαγών Αστέρων**
> *(Theoretical and Computational Study of the Normal Modes of Oscillation
> of Compact Stars)*
> Evangelos Peppas, National and Kapodistrian University of Athens
> Supervisor: Theocharis Apostolatos — September 2026

## Overview

The project builds hydrostatic equilibrium models of compact stars and then
solves the associated Sturm-Liouville eigenvalue problem for small adiabatic
perturbations around them, using a **shooting method**, independently
cross-checked with the variational **Rayleigh quotient**. It covers:

- **Newtonian polytropic stars** — the Lane-Emden equation, and the
  Newtonian adiabatic pulsation equations (radial *p-/g-modes* and
  non-radial modes in the Cowling approximation).
- **Relativistic stellar structure** — the Tolman-Oppenheimer-Volkoff (TOV)
  equations, solved for both a phenomenological polytrope and a realistic
  equation of state (degenerate free-neutron Fermi gas), yielding
  Mass-Radius curves and the theoretical maximum-mass limit.
- **Relativistic radial oscillations** — the linearized pulsation equations
  for a relativistic background, used to extract the fundamental (f-mode)
  eigenfrequency and to study the onset of dynamical instability as the
  central density approaches its critical (maximum-mass) value.
- A symbolic **derivation of the field equations** (Christoffel symbols,
  Ricci/Einstein tensors) for the static spherically symmetric metric,
  using `sympy` and `einsteinpy`, as a check of the equations used
  elsewhere in the project.

## Repository structure

```
Stellar-Pulsations/
├── solvers/
│   ├── background.py     # Lane-Emden solver, TOV solver, equations of state
│   ├── shooting.py        # Newtonian shooting method (radial & non-radial,
│   │                       #   Cowling approximation) + Rayleigh quotient
│   └── shooting_gr.py      # Relativistic radial pulsation shooting method
├── lane-emden.ipynb        # Newtonian polytrope: background + eigenmodes
├── tov.ipynb                # TOV equilibrium models, Mass-Radius curves,
│                            #   GR f-mode via shooting_gr
├── radial_oscillations.ipynb # f-mode frequency & stability vs. central
│                            #   density for the realistic neutron EoS
├── gr.ipynb                 # Symbolic derivation of the field equations
├── newtonian.npz             # Cached dimensionless Newtonian background,
│                            #   used for comparison inside tov.ipynb
└── README.md
```

## Installation

```bash
git clone https://github.com/Vagge31/Stellar-Pulsations.git
cd Stellar-Pulsations
pip install numpy scipy matplotlib pandas sympy einsteinpy jupyter
```

`einsteinpy` is only required for `gr.ipynb` (the symbolic tensor
calculations); the numerical notebooks only depend on `numpy`, `scipy`,
`matplotlib` and `pandas`.

## Usage

The notebooks are meant to be run with the `solvers/` package importable
from the repository root (e.g. via Jupyter Lab/Notebook started in the
top-level folder).

Recommended order:

1. **`lane-emden.ipynb`** — solves the Lane-Emden equation for a chosen
   polytropic index, builds the dimensionless Newtonian background, and
   computes the p-, f- and g-mode eigenfrequencies via the shooting method,
   cross-checked with the Rayleigh quotient.
2. **`tov.ipynb`** — integrates the TOV equations for a polytropic and a
   realistic equation of state, compares the relativistic background to the
   Newtonian one (loaded from `newtonian.npz`), builds the Mass-Radius
   curves, and computes the relativistic fundamental mode frequency.
3. **`radial_oscillations.ipynb`** — scans the realistic-EoS models across
   central densities up to the critical (maximum-mass) value, tracking how
   the f-mode frequency drops to zero and turns into an exponential
   instability beyond it.
4. **`gr.ipynb`** — standalone; symbolically re-derives the curvature and
   field-equation tensors used to justify the TOV equations.

## Key results

- Newtonian polytrope (n = 3): fundamental (f-mode) and first few p-/g-mode
  eigenfrequencies, reproduced independently via shooting and Rayleigh
  quotient.
- Relativistic polytropic model (Γ = 2, K = 100 km²): Mass-Radius curve with
  a maximum mass of **M ≈ 1.109 M<sub>☉</sub>**.
- Realistic degenerate free-neutron-gas equation of state: theoretical
  maximum-mass limit of **M ≈ 0.71 M<sub>☉</sub>**, consistent with the
  original Oppenheimer-Volkoff (1939) result, at a critical central density
  of ρ<sub>crit</sub> ≈ 4.34 × 10<sup>15</sup> g/cm³.
- Fundamental-mode frequency ν₀ tracked across central densities for all
  three models, vanishing (and turning into a growing instability) exactly
  at the maximum-mass point — the onset of dynamical instability.

## References

The full theoretical background, derivations and discussion of these
results are presented in the accompanying thesis. Key references used
throughout the code:

- J. R. Oppenheimer and G. M. Volkoff, *On Massive Neutron Cores*,
  Phys. Rev. 55, 374 (1939).
- K. D. Kokkotas and J. Ruoff, *Radial oscillations of relativistic stars*,
  A&A 366, 565 (2001).
- J. Christensen-Dalsgaard, *Lecture Notes on Stellar Oscillations*, 5th ed.
  (2003).

## License

This project is licensed under the [MIT License](LICENSE).

## Author

Evangelos Peppas — [github.com/Vagge31](https://github.com/Vagge31)
