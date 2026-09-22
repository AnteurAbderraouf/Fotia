# PSI-117

## Project

**Project Type:** Ground Investigation
**Proposal Title:** Effect of External Thermo-Convective Perturbation on Cool Flame Dynamics: A Multidimensional Multi-Physics CFD Analysis
**Investigation Start Date:** 02-15-2017
**Investigation End Date:** 02-14-2022
**Sponsoring Agency:** National Aeronautics and Space Administration (NASA), PSI grant
**NASA Center:** Glenn Research Center (GRC)

## Objectives

The objective of this ground investigation was to use flight data obtained from PSI through NASA's Flame Extinguishment Experiment (FLEX) to quantify how externally imposed thermal and convective perturbations regulated the dynamics and ultimate fate of already established droplet cool flames, and to explain run-to-run variability by isolating the roles of deliberate heating and unintentional drift-induced convection. Focusing on both small and large droplets, the work mapped stability windows for ignition, sustained oscillation, transitions to hot ignition and warm-flame behavior, and extinction under microgravity. By leveraging and extending the FLEX data in the PSI, this study produced reusable models and sensitivity analyses that coupled low-temperature chemistry with transport and radiative effects.

## Approach

OpenFOAM was used to develop a 3-D CFD model of n-alkane droplet cool flames in microgravity, coupling variable-density Navier–Stokes flow with low-temperature combustion chemistry, coupled heat and mass transfer, and radiation. Model fidelity was established against ISS FLEX measurements, after which controlled simulations examined three forcing modes (i) convection alone to mimic droplet drift, (ii) imposed heat flux, and (iii) combined convection plus heating applied to both large (1–4 mm) and sub-millimeter (~0.5 mm) droplets.

To accelerate progress, the study began with simplified chemistry and transitioned to reduced mechanisms for parameter sweeps over forcing amplitude and frequency, ambient composition (including ozone-assisted cases), and pressure. From each case, predictive observables were extracted (ignition delay, oscillation frequency and amplitude, stand-off and extinction diameters), and were used to construct stability maps that delineated conditions for cool-flame onset, persistence, and quenching. These maps were reconciled with FLEX observations and were used to guide next-generation test matrices for both large- and small-droplet experiments.

## Hypothesis

External thermo-convective perturbations (imposed temperature gradients, non-uniform heating, oscillatory/shear inflow) were expected to govern cool-flame ignition, oscillation, transition to hot ignition, and extinction by shifting residence time and scalar dissipation. In microgravity, where buoyancy was suppressed, this forcing was anticipated to dominate transport. A multidimensional, multi-physics CFD model was used to map perturbation-dependent stability boundaries and predict observables for comparison with FLEX data.

## Research Impacts/Earth Benefits

**Space Benefits:** Potential to improve spacecraft fire-safety modeling by resolving cool-flame/extinction behavior in microgravity; leverages FLEX to define safer operating envelopes/materials.

**Earth Benefits:** Advances low-temperature combustion understanding relevant to engine knock, efficiency, and emissions; delivers validated CFD tools transferrable to practical combustors.

## Contact(s)

Tanvir Farouk, Mike Hicks