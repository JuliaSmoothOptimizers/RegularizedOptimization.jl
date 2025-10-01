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

[RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) is a Julia package that implements a family of quadratic regularization and trust-region type algorithms for solving nonsmooth optimization problem
\begin{equation}\label{eq:nlp}
    \underset{x \in \mathbb{R}^n}{\text{minimize}} \quad f(x) + h(x), \quad s.t. \quad c(x) = 0,
\end{equation}
where $f: \mathbb{R}^n \to \mathbb{R}$ is continuously differentiable, $h: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ is lower semi-continuous which that the regularizer such as sparsity-inducing penalties, bound constraints or a combination of both and $c: \mathbb{R}^n \to \mathbb{R}^m$ is continuously differentiable defining equality constraints.
All $f$, $h$, and $c$ can be nonconvex.
The library provides a modular and extensible framework for experimenting with nonsmooth and nonconvex optimization algorithms, including:

- **Trust-region solvers (TR, TRDH)** [@aravkin-baraldi-orban-2022;@leconte-orban-2023],
- **Quadratic regularization solvers (R2, R2N)** [@diouane-habiboullah-orban-2024;@aravkin-baraldi-orban-2022],
- **Levenberg-Marquardt solvers (LM, LMTR)** [@aravkin-baraldi-orban-2024].
- **Augmented Lagrangian solver (AL)** [@demarchi-jia-kanzow-mehlitz-2023].

Except of the **AL** solver, these methods rely on the gradient and optionally on the Hessian(-vector) information of the smooth part $f$ and the proximal mapping of the nonsmooth part $h$ in order to compute steps.
Then, the objective function $f + h$ is used only to accept or reject trial points.

Solvers in [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) allow inexact resolution of trust-region and quadratic-regularized subproblems using first-order that are implemented in the package itself such as the quadratic regularization method R2 [@aravkin-baraldi-orban-2022] and R2DH [@diouane-habiboullah-orban-2024] with trust-region variants TRDH [@leconte-orban-2023-2].

All solvers in [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) are implemented in an in-place fashion, minimizing memory allocations during the solution process.
Moreover, they implement non-monotone strategies to accept trial points, which can enhance algorithmic performance in practice [@leconte-orban-2023;@diouane-habiboullah-orban-2024].

## Requirements of the ShiftedProximalOperators.jl

The nonsmooth part $h$ must have a computable proximal mapping, defined as
$$\text{prox}_{\nu h}(v) = \underset{x \in \mathbb{R}^n}{\arg\min} \frac{1}{2} \|x - v\|^2 + \nu h(x).$$

While [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl) provides many standard proximal mappings, [ShiftedProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl) also supplies **shifted** variants of these mappings which is not supported by [ProximalOperators.jl](https://www.github.com/JuliaFirstOrder/ProximalOperators.jl).

# Statement of need

## Model-based framework for nonsmooth methods

In Julia, \eqref{eq:nlp} can be solved using [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl), which implements in-place, first-order, line-search–based methods [@stella-themelis-sopasakis-patrinos-2017;@themelis-stella-patrinos-2017].
Most of these methods are splitting schemes that either alternate between the proximal operators of $f$ and $h$, as in the **Douglas–Rachford** solver [@eckstein1992douglas], or take a step along a direction $d$, which depends on the gradient of $f$, possibly modified by a L-BFGS Quasi-Newton approximation followed by proximal steps on the nonsmooth part $h$. In some cases, such as with the **PANOC** [@stella-themelis-sopasakis-patrinos-2017] solver, this process is augmented with a line-search mechanism along $d$.

By contrast, [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) focuses on model-based approaches such as trust-region and quadratic regularization algorithms.
As shown in [@aravkin-baraldi-orban-2022], model-based methods typically require fewer evaluations of the objective and its gradient than first-order line search methods, at the expense of requiring a lot of proximal iterations to solve the subproblems.
Although these subproblems may require many proximal iterations, each proximal computation is inexpensive for nuumerous commonly used nonsmooth functions, such as separable penalties and bound constraints (see examples below), making the overall approach efficient for large-scale problems.

The package provides a consistent API to formulate optimization problems and apply different solvers.
It integrates seamlessly with the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers)  [@jso] ecosystem, an academic organization for nonlinear optimization software development, testing, and benchmarking.

On the one hand, the smooth objective $f$ can be defined via [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) [@orban-siqueira-nlpmodels-2020], which provides a standardized Julia API for representing nonlinear programming (NLP) problems.
Large collections of such problems are available in [CUTE.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl) [@orban-siqueira-cutest-2020] and [OptimizationProblems.jl](https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl) [@migot-orban-siqueira-optimizationproblems-2023].
Another option is to use [RegularizedProblems.jl](https://github.com/JuliaSmoothOptimizers/RegularizedProblems.jl), which provides problem instances commonly used in the nonsmooth optimization literature, where $f$ can be paired with various nonsmooth terms $h$.

On the other hand, nonsmooth terms $h$ can be modeled using [ProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ProximalOperators.jl), which provides a broad collection of nonsmooth functions, together with [ShiftedProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl), which provides shifted proximal mappings for nonsmooth functions.
Specifically, the package supports shifted proximal operators of the form
$$
    \underset{t \in \mathbb{R}^n}{\arg\min} \, { \tfrac{1}{2} ‖t - q‖₂² + ν \psi(t + s;x) + χ(s + t\mid Δ\mathbb{B})}
$$
where $\psi(;x)$ is a nonsmooth function that models $h$, in general we set $\psi(t;x) = h(x+t)$, $q$ is given, $x$ and $s$ are fixed shifts, $h$ is the nonsmooth term with respect
to which we are computing the proximal operator, and $χ(.| \Delta \mathbb{B})$ is the indicator of
a ball of radius $\Delta > 0$ defined by a certain norm.

These shifted operators allow us to (i) incorporate bound or trust-region constraints via the indicator term which is required for the **TR** and **TRDH** algorithms and (ii) evaluate the prox **in place**, without additional allocations, which integrates efficiently with our subproblem solvers.

## Support for Hessians and Hessian approximations of the smooth part $f$

In contrast to [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl), [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) methods such as **R2N** and **TR** methods support exact Hessians as well as several Hessian approximations of $f$.

Hessian–vector products $v \mapsto Hv$ can be obtained via automatic differentiation through [ADNLPModels.jl](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl) or implemented normally.

Hessian approximations (e.g., quasi-Newton and diagonal schemes) can be selected from via [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl).

This design allows algorithms to exploit second-order information **without** explicitly forming dense or sparse Hessian matrices, which is often expensive in time and memory, particularly at large scale.

## Requirements of the RegularizedProblems.jl

With $f$ and $h$ modeled as discussed above, the package [RegularizedProblems.jl](https://github.com/JuliaSmoothOptimizers/RegularizedProblems.jl) provides a straightforward way to pair them into a *Regularized Nonlinear Programming Model*

```julia
reg_nlp = RegularizedNLPModel(f, h)
```

They can also be paired into a *Regularized Nonlinear Least Squares Model* if $f(.) = \tfrac{1}{2} \|F(.)\|^2$ for some residual function $F: \mathbb{R}^n \to \mathbb{R}^m$, which is required for the **LM** and **LMTR** solvers.

```julia
reg_nls = RegularizedNLSModel(f, h)
```

This design makes for a convenient source of reproducible problem instances for testing and benchmarking the solvers in [RegularizedOptimization.jl](https://www.github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl).

## Testing and documentation

The package includes a comprehensive suite of unit tests that cover all functionalities, ensuring reliability and correctness.
Extensive documentation is provided, including a user guide, API reference, and examples to help users get started quickly.
Documentation is built using Documenter.jl.

## Application studies

The package is used to solve equality-constrained optimization problems by means of the exact penalty approach [@diouane-gollier-orban-2024] where the model of the nonsmooth part differs from the function $h$ itself.
This is not covered in the current version of the package [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl).

# Examples

We illustrate the capabilities of [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) on two nonsmooth and nonconvex problems:

- **Support Vector Machine (SVM) with $\ell_{1/2}^{1/2}$ penalty** for image classification [@aravkin-baraldi-orban-2024].  
- **Nonnegative Matrix Factorization (NNMF) with $\ell_0$ penalty and bound constraints** [@kim-park-2008].

Below is a condensed example showing how to define and solve SVM problem:

```julia
using LinearAlgebra, Random, ProximalOperators
using NLPModels, RegularizedProblems, RegularizedOptimization
using MLDatasets

Random.seed!(1234)
model, nls_model, _ = RegularizedProblems.svm_train_model()  # Build SVM model
f = LSR1Model(model)                                         # L-SR1 Hessian approximation
λ = 1.0                                                      # Regularization parameter
h = RootNormLhalf(1.0)                                       # Nonsmooth term
reg_nlp = RegularizedNLPModel(f, h)                          # Regularized problem
solver = R2NSolver(reg_nlp)                                  # Choose solver
stats  = RegularizedExecutionStats(reg_nlp)
solve!(solver, reg_nlp, stats; atol=1e-4, rtol=1e-4, verbose=0, sub_kwargs=(max_iter=200,))
solve!(solver, reg_nlp, stats; atol=1e-5, rtol=1e-5, verbose=0, sub_kwargs=(max_iter=200,))
```

The NNMF problem can be set up in a similar fashion:

```julia
Random.seed!(1234)
m, n, k = 100, 50, 5
model, nls_model, _, selected = nnmf_model(m, n, k)          # Build NNMF model
x0 = rand(model.meta.nvar)                                   # Initial point
λ = norm(grad(model, rand(model.meta.nvar)), Inf) / 200      # Regularization parameter
h = NormL0(λ)                                                # Nonsmooth term
reg_nls = RegularizedNLSModel(nls_model, h)                  # Regularized problem for LM
solver = LMSolver(reg_nls)                                   # Choose solver
```

## Numerical results

We compare **PANOC** [@stella-themelis-sopasakis-patrinos-2017] (from [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl)) against **TR**, **R2N**, and **LM** from our library.
In order to do so, we implemented a wrapper for **PANOC** to make it compatible with our problem definition.
The results are summarized in the combined table below:

\input{examples/Benchmark.tex}

* For the LM solver, gradient evaluations count equals the number of Jacobian–vector and adjoint-Jacobian–vector products.

## Discussion

According to **status**, all methods successfully reduced the optimality measure below the specified tolerance which is set to $10^{-4}$ and thus converged to a **first-order** stationary point.
However, the final objective values differ due to the nonconvexity of the problems.

- **SVM with $\ell^{1/2}$ penalty:** **TR** and **R2N** require far fewer function and gradient evaluations than **PANOC**, at the expense of more proximal iterations. Since each proximal step is inexpensive, **TR** and **R2N** are much faster overall.  
- **NNMF with constrained $\ell_0$ penalty:** **TR** outperforms **R2N**, while **LM** is competitive in terms of function calls but incurs many gradient evaluations.

Additional tests (e.g., other regularizers, constraint types, and scaling dimensions) have also been conducted, and a full benchmarking campaign is currently underway.

# Conclusion

The experiments highlight the effectiveness of the solvers implemented in [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) compared to **PANOC** from [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl).

On these examples, the performance of the solvers can be summarized as follows:

- **Function and gradient evaluations:** **TR** and **R2N** are the most efficient choices when aiming to minimize both.
- **Function evaluations only:** **LM** is preferable when the problem is a nonlinear least squares problem, as it achieves the lowest number of function evaluations.
- **Proximal iterations:** **PANOC** requires the fewest proximal iterations. However, in most nonsmooth applications, proximal steps are relatively inexpensive, so this criterion is of limited practical relevance.

In the future, the package will be extended with additional algorithms that enable to reduce the number of proximal evaluations, especially when the proximal mapping of $h$ is expensive to compute.

# Acknowledgements

The authors would like to thank Alberto Demarchi for his implementation of the Augmented Lagrangian solver.
Mohamed Laghdaf Habiboullah is supported by an excellence FRQNT grant.
Youssef Diouane and Dominique Orban are partially supported by an NSERC Discovery Grant.

# References
