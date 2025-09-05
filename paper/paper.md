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
    orcid: 0009-0008-3158-7912
    affiliation: 1
  - name: Mohamed Laghdaf Habiboullah^[corresponding author]
    orcid: 0009-0005-3631-2799
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

[RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) is a Julia [@bezanson-edelman-karpinski-shah-2017] package that implements a family of regularization and trust-region type algorithms for solving nonsmooth optimization problems of the form:
\begin{equation}\label{eq:nlp}
    \underset{x \in \mathbb{R}^n}{\text{minimize}} \quad f(x) + h(x),
\end{equation}
where $f: \mathbb{R}^n \to \mathbb{R}$ is continuously differentiable on $\mathbb{R}^n$, and $h: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ is lower semi-continuous.
Both $f$ and $h$ may be nonconvex.

The library provides a modular and extensible framework for experimenting some nonsmooth nonconvex optimization algorithms, including:

- **Trust-region methods (TR, TRDH)** [@aravkin-baraldi-orban-2022] and [@leconte-orban-2023],
- **Quadratic regularization methods (R2, R2N)** [@diouane-habiboullah-orban-2024] and [@aravkin-baraldi-orban-2022],
- **Levenbergh-Marquardt methods (LM, LMTR)** [@aravkin-baraldi-orban-2024].

These methods rely solely on the gradient and Hessian(-vector) information of the smooth part $f$ and the proximal mapping of the nonsmooth part $h$ in order to compute steps.
Then, the objective function $f + h$ is used only to accept or reject trial points.
Moreover, they can handle cases where Hessian approximations are unbounded[@diouane-habiboullah-orban-2024] and [@leconte-orban-2023-2], making the package particularly suited for large-scale, ill-conditioned, and nonsmooth problems.

# Statement of need

## Unified framework for nonsmooth methods

There exists a way to solve \eqref{eq:nlp} in Julia via [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl).
It implements several proximal algorithms for nonsmooth optimization.
However, the available examples only consider convex instances of $h$, nmaely the $\ell_1$ norm and there are no tests for memory allocations.
Moreover, it implements only one quasi-Newton method (L-BFGS) and does not support Hessian approximations via linear operators.
In contrast, **RegularizedOptimization.jl** leverages [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl)[@leconte_linearoperators_jl_linear_operators_2023] to represent a variety of Hessian approximations, such as L-SR1, L-BFGS, and diagonal approximations.

**RegularizedOptimization.jl** implements a broad class of regularization-based algorithms for solving problems of the form $f(x) + h(x)$, where $f$ is smooth and $h$ is nonsmooth.
The package offers a consistent API to formulate optimization problems and apply different regularization methods.
It enables researchers to:

- Test and compare algorithms within a unified framework.
- Switch between exact Hessians, quasi-Newton updates, and diagonal Hessian approximations via [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl).
- Incorporate nonsmooth terms $h$ through proximal mappings.

The design of the package is motivated by recent advances in the complexity analysis of regularization and trust-region methods.

## Compatibility with JuliaSmoothOptimizers ecosystem

**RegularizedOptimization.jl** integrates seamlessly with other [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers) packages:

- **Definition of $f$** via [RegularizedProblems.jl](https://github.com/JuliaSmoothOptimizers/RegularizedProblems.jl), which provides efficient implementations of smooth problems $f$ together with their gradients.
- **Model Hessians (quasi-Newton, diagonal approximations)** via [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl), which represents Hessians as linear operators and implements efficient Hessian–vector products.
- **Definition of $h$** via [ProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ProximalOperators.jl), which offers a large collection of nonsmooth terms $h$, and [ShiftedProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl), which provides shifted proximal mappings.

This modularity makes it easy to prototype, benchmark, and extend regularization-based methods [@diouane-habiboullah-orban-2024],[@aravkin-baraldi-orban-2022],[@aravkin-baraldi-orban-2024],[@leconte-orban-2023-2] and [@diouane-gollier-orban-2024].

## Support for inexact subproblem solves

Solvers in **RegularizedOptimization.jl** allow inexact resolution of trust-region and cubic-regularized subproblems using first-order that are implemented in the package itself such as the quadratic regularization method R2[@aravkin-baraldi-orban-2022] and R2DH[@diouane-habiboullah-orban-2024] with trust-region variants TRDH[@leconte-orban-2023-2]

This is crucial for large-scale problems where exact subproblem solutions are prohibitive.

## Support for Hessians as Linear Operators

The second-order methods in **RegularizedOptimization.jl** can use Hessian approximations represented as linear operators via [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl).
Explicitly forming Hessians as dense or sparse matrices is often prohibitively expensive, both computationally and in terms of memory, especially in high-dimensional settings.
In contrast, many problems admit efficient implementations of Hessian–vector or Jacobian–vector products, either through automatic differentiation tools or limited-memory quasi-Newton updates, making the linear-operator approach more scalable and practical.

## In-place methods

All solvers in **RegularizedOptimization.jl** are implemented in an in-place fashion, minimizing memory allocations and improving performance.
This is particularly important for large-scale problems where memory usage can be a bottleneck.

# Examples

A simple example is the solution of a regularized quadratic problem with an $\ell_1$ penalty, as described in @[aravkin-baraldi-orban-2022].
Such problems are common in statistical learning and compressed sensing applications.The formulation is
$$
  \min_{x \in \mathbb{R}^n} \ \tfrac{1}{2}\|Ax-b\|_2^2+\lambda\|x\|_1,
$$
where $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^m$, and $\lambda>0$ is a regularization parameter.

```julia
using LinearAlgebra, Random
using ProximalOperators
using NLPModels, NLPModelsModifiers, RegularizedProblems, RegularizedOptimization, SolverCore

# Set random seed for reproducibility
Random.seed!(1234)   

# Define a basis pursuit denoising problem
compound = 10
bpdn, bpdn_nls, sol = bpdn_model(compound)

# Define the Hessian approximation
f = LSR1Model(bpdn)

# Define the nonsmooth regularizer (L1 norm) 
λ = 1.0
h = NormL1(λ)

# Define the regularized NLP model
reg_nlp = RegularizedNLPModel(f, h)

# Choose a solver (R2N) and execution statistics tracker
solver_r2N = R2NSolver(reg_nlp)
stats = RegularizedExecutionStats(reg_nlp)

# Solve the problem 
solve!(solver_r2N, reg_nlp, stats, x = f.meta.x0, σk = 1.0, atol = 1e-8, rtol = 1e-8, verbose = 1)

# Choose another solver (TR) and execution statistics tracker
solver_tr = TRSolver(reg_nlp)
stats_tr = RegularizedExecutionStats(reg_nlp)

# Solve the problem
solve!(solver_tr, reg_nlp, stats_tr, x = f.meta.x0, Δk = 1.0, atol = 1e-8, rtol = 1e-8, verbose = 1)
```

# Acknowledgements

Mohamed Laghdaf Habiboullah is supported by an excellence FRQNT grant,
and Youssef Diouane and Dominique Orban are partially supported by an NSERC Discovery Grant.

# References
