---
title: 'RegularizedOptimization.jl: A Julia framework for regularization-based nonlinear optimization'
tags:
  - Julia
  - nonsmooth optimization
  - nonconvex optimization
  - regularization methods
  - trust-region methods
authors:
  - name: Youssef Diouane
    orcid: 0000-0002-6609-7330
    affiliation: 1
  - name: Maxence Gollier^[corresponding author]
    orcid: 0000-0002-8017-7687
    affiliation: 1
  - name: Mohamed Laghdaf Habiboullah^[corresponding author]
    orcid: 0000-0003-3385-9379
    affiliation: 1
  - name: Dominique Orban
    orcid: 0000-0002-8017-7687
    affiliation: 1
affiliations:
 - name: GERAD and Department of Mathematics and Industrial Engineering, Polytechnique Montréal, QC, Canada
   index: 1
date: 1 September 2025
bibliography: paper.bib
header-includes: |
  \usepackage{booktabs}
  \usepackage{fontspec}
  \setmonofont[Path = ./, Scale=0.68]{JuliaMono-Regular.ttf}
---

# Summary

[RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) is a Julia package that implements a family of regularization and trust-region type algorithms for solving unconstrained or composite nonsmooth optimization problems of the form

$$
\min_{x \in \mathbb{R}^n} f(x) + h(x),
$$

where $f$ is typically smooth (possibly nonconvex) and $h$ is convex but possibly nonsmooth.
The library provides a modular and extensible framework for experimenting with regularization-based methods such as:

- **Trust-region methods (TR, TRDH)**,
- **Quadratic regularization methods (R2, R2N)**,
- **Levenbergh-Marquadt methods (LM, LMTR)**.

These methods rely solely on gradient and Hessian(-vector) information and can handle cases where Hessian approximations are unbounded, making the package particularly suited for large-scale, ill-conditioned, or nonsmooth problems.

# Statement of need

## Unified framework for regularization methods

RegularizedOptimization.jl provides a consistent API to formulate optimization problems and apply a range of regularization methods.
It allows researchers to:

- Test and compare different regularization algorithms within a common environment.
- Switch between exact Hessians, quasi-Newton updates, and diagonal Hessian approximation via [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl).
- Incorporate nonsmooth terms $h$ via proximal mappings.

The package is particularly motivated by recent advances in the complexity analysis of regularization and trust-region methods.

## Compatibility with JuliaSmoothOptimizers ecosystem

RegularizedOptimization.jl integrates seamlessly with other [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers) packages:

- **Problem definition** via [RegularizedProblems.jl](https://github.com/JuliaSmoothOptimizers/RegularizedProblems.jl).
- **Linear algebra operations** via [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl).
- **Prox-definition** via [ShiftedProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl).

This modularity makes it easy to prototype, benchmark, and extend regularization-based methods.

## Support for inexact subproblem solves

Solvers in RegularizedOptimization.jl allow inexact resolution of trust-region and cubic-regularized subproblems using first-order nonmsooth optimization methods such as R2.

This is crucial for large-scale problems where exact subproblem solutions are prohibitive.

## Research and teaching tool

The package is designed both as a research platform for developing new optimization methods and as a pedagogical tool for teaching modern non-smooth nonconvex optimization algorithms.
It provides reference implementations that are transparent and mathematically faithful, while being efficient enough for large-scale experiments.

# Examples

A simple example: solving a regularized quadratic problem with an $\ell_1$ penalty.

```julia
using LinearAlgebra, Random
using ProximalOperators
using NLPModels, NLPModelsModifiers, RegularizedProblems, RegularizedOptimization, SolverCore

# Set random seed for reproducibility
Random.seed!(123)   

# Define a basis pursuit denoising problem
compound = 10
nz = 10 * compound
bpdn, bpdn_nls, sol = bpdn_model(compound)

# Define the Hessian approximation
f = LSR1Model(bpdn)

# Define the nonsmooth regularizer (L1 norm) 
λ = 1.0
h = NormL1(λ)

# Define the regularized NLP model
reg_nlp = RegularizedNLPModel(f, h)

# Choose a solver (R2N) and execution statistics tracker
solver = R2NSolver(reg_nlp)
stats = RegularizedExecutionStats(reg_nlp)

# Solve the problem 
solve!(solver, reg_nlp, stats, x = f.meta.x0, σk = 1.0, atol = 1e-8, rtol = 1e-8, verbose = 1)
```

# Acknowledgements

Development of RegularizedOptimization.jl has been supported by the Natural Sciences and Engineering Research Council of Canada (NSERC), the Fonds de Recherche du Québec – Nature et Technologies (FRQNT).
The authors thank the JuliaSmoothOptimizers community for valuable feedback and contributions.

# References
