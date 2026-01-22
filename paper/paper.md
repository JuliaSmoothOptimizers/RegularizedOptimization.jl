---
title: 'RegularizedOptimization.jl: A Julia framework for regularized and nonsmooth optimization'
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

[RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) is a Julia package that implements families of quadratic regularization and trust-region methods for solving the nonsmooth optimization problem
\begin{equation}\label{eq:nlp}
    \underset{x \in \mathbb{R}^n}{\text{minimize}} \quad f(x) + h(x) \quad \text{subject to} \quad c(x) = 0,
\end{equation}
where $f: \mathbb{R}^n \to \mathbb{R}$ and $c: \mathbb{R}^n \to \mathbb{R}^m$ are continuously differentiable, and $h: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ is lower semi-continuous.
The nonsmooth objective $h$ can be a *regularizer*, such as a sparsity-inducing penalty, model simple constraints, such as $x$ belonging to a simple convex set, or be a combination of both.
All $f$, $h$ and $c$ can be nonconvex.
RegularizedOptimization.jl provides a modular and extensible framework for solving \eqref{eq:nlp}, and developing novel solvers.
Currently, the following solvers are implemented:

- **Trust-region solvers TR and TRDH** [@aravkin-baraldi-orban-2022;@leconte-orban-2023]
- **Quadratic regularization solvers R2, R2DH and R2N** [@diouane-habiboullah-orban-2024;@aravkin-baraldi-orban-2022]
- **Levenberg-Marquardt solvers LM and LMTR** [@aravkin-baraldi-orban-2024] used when $f$ is a least-squares residual.
- **Augmented Lagrangian solver AL** [@demarchi-jia-kanzow-mehlitz-2023].

All solvers rely on first derivatives of $f$ and $c$, and optionally on their second derivatives in the form of Hessian-vector products.
If second derivatives are not available, quasi-Newton approximations can be used.
In addition, the proximal mapping of the nonsmooth part $h$, or adequate models thereof, must be evaluated.
At each iteration, a step is computed by solving a subproblem of the form \eqref{eq:nlp} inexactly, in which $f$, $h$, and $c$ are replaced with appropriate models about the current iterate.
The solvers R2, R2DH and TRDH are particularly well suited to solve the subproblems, though they are general enough to solve \eqref{eq:nlp}.
All solvers are implemented in place, so re-solves incur no allocations.
To illustrate our claim of extensibility, a first version of the AL solver was implemented by an external contributor.
Furthermore, a nonsmooth penalty approach, described in [@diouane-gollier-orban-2024] is currently being developed, that relies on the library to efficiently solve the subproblems.

<!-- ## Requirements of the ShiftedProximalOperators.jl -->
<!---->
<!-- The nonsmooth part $h$ must have a computable proximal mapping, defined as -->
<!-- $$\text{prox}_{\nu h}(v) = \underset{x \in \mathbb{R}^n}{\arg\min} \frac{1}{2} \|x - v\|^2 + \nu h(x).$$ -->
<!---->
<!-- While [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl) provides many standard proximal mappings, [ShiftedProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl) also supplies **shifted** variants of these mappings which is not supported by [ProximalOperators.jl](https://www.github.com/JuliaFirstOrder/ProximalOperators.jl). -->

# Statement of need

## Model-based framework for nonsmooth methods

In Julia, \eqref{eq:nlp} can be solved using [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl), which implements splitting schemes and  line-search–based methods [@stella-themelis-sopasakis-patrinos-2017;@themelis-stella-patrinos-2017].
Among others, the **PANOC** [@stella-themelis-sopasakis-patrinos-2017] solver takes a step along a direction $d$, which depends on the L-BFGS quasi-Newton approximation of $f$, followed by proximal steps on $h$.

By contrast, [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) focuses on model-based trust-region and quadratic regularization methods, which typically require fewer evaluations of $f$ and its gradient than first-order line search methods, at the expense of more evaluations of proximal operators [@aravkin-baraldi-orban-2022].
However, each proximal computation is inexpensive for numerous commonly used choices of $h$, such as separable penalties and bound constraints, so that the overall approach is efficient for large-scale problems.

RegularizedOptimization.jl provides an API to formulate optimization problems and apply different solvers.
It integrates seamlessly with the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers)  [@jso] ecosystem.

The smooth objective $f$ can be defined via [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) [@orban-siqueira-nlpmodels-2020], which provides a standardized Julia API for representing nonlinear programming (NLP) problems.
The nonsmooth term $h$ can be modeled using [ProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ProximalOperators.jl).

Given $f$ and $h$, the companion package [RegularizedProblems.jl](https://github.com/JuliaSmoothOptimizers/RegularizedProblems.jl) provides a way to pair them into a *Regularized Nonlinear Programming Model*

```julia
reg_nlp = RegularizedNLPModel(f, h)
```

They can also be paired into a *Regularized Nonlinear Least-Squares Model* if $f(x) = \tfrac{1}{2} \|F(x)\|^2$ for some residual $F: \mathbb{R}^n \to \mathbb{R}^m$, in the case of the **LM** and **LMTR** solvers.

```julia
reg_nls = RegularizedNLSModel(F, h)
```

RegularizedProblems.jl also provides a set of instances commonly used in data science and in nonsmooth optimization, where several choices of $f$ can be paired with various regularizers.
This design makes for a convenient source of problem instances for benchmarking the solvers in [RegularizedOptimization.jl](https://www.github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl).

## Support for both exact and approximate Hessian

In contrast with [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl), [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl), methods such as **R2N** and **TR** methods support exact Hessians as well as several Hessian approximations of $f$.
Hessian–vector products $v \mapsto Hv$ can be obtained via automatic differentiation through [ADNLPModels.jl](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl) or implemented manually.
Limited-memory and diagonal quasi-Newton approximations can be selected from [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl).
This design allows solvers to exploit second-order information without explicitly forming dense or sparse Hessians, which is often expensive in time and memory, particularly at large scale.

# Example

We illustrate the capabilities of [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) on a Support Vector Machine (SVM) model with a $\ell_{1/2}^{1/2}$ penalty for image classification [@aravkin-baraldi-orban-2024].  

Below is a condensed example showing how to define and solve the problem, and perform a solve followed by a re-solve:

```julia
using LinearAlgebra, Random, ProximalOperators
using NLPModels, RegularizedProblems, RegularizedOptimization
using MLDatasets

Random.seed!(1234)
model, nls_model, _ = RegularizedProblems.svm_train_model()  # Build SVM model
f = LSR1Model(model)                                         # L-SR1 Hessian approximation
λ = 1.0                                                      # Regularization parameter
h = RootNormLhalf(λ)                                         # Nonsmooth term
reg_nlp = RegularizedNLPModel(f, h)                          # Regularized problem
solver = R2NSolver(reg_nlp)                                  # Choose solver
stats  = RegularizedExecutionStats(reg_nlp)
solve!(solver, reg_nlp, stats; atol=1e-4, rtol=1e-4, verbose=1, sub_kwargs=(max_iter=200,))
solve!(solver, reg_nlp, stats; atol=1e-5, rtol=1e-5, verbose=1, sub_kwargs=(max_iter=200,))
```

## Numerical results

We compare **TR**, **R2N**, **LM** and **LMTR** from our library on the SVM problem.
Experiments were performed on macOS (arm64) on an Apple M2 (8-core) machine, using Julia 1.11.7.

The table reports the convergence status of each solver, the number of evaluations of $f$, the number of evaluations of $\nabla f$, the number of proximal operator evaluations, the elapsed time and the final objective value.
For TR and R2N, we use limited-memory SR1 Hessian approximations.
The subproblem solver is **R2**.

\input{examples/Benchmark.tex}

For the **LM** and **LMTR** solvers, $\#\nabla f$ counts the number of Jacobian–vector and adjoint-Jacobian–vector products.

All methods successfully reduced the optimality measure below the specified tolerance of $10^{-4}$, and thus converged to an approximate first-order stationary point.
Note that the final objective values differ due to the nonconvexity of the problem.

**R2N** is the fastest in terms of time and number of gradient evaluations.
However, it requires more proximal evaluations, but these are inexpensive.
**LMTR** and **LM** require the fewest function evaluations, but incur many Jacobian–vector products, and are the slowest in terms of time.

Ongoing research aims to reduce the number of proximal evaluations.

# Acknowledgements

The authors would like to thank A. De Marchi for the Augmented Lagrangian solver.
M. L. Habiboullah is supported by an excellence FRQNT grant.
Y. Diouane, M. Gollier and D. Orban are partially supported by an NSERC Discovery Grant.

# References
