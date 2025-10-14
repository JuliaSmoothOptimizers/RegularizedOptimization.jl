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
The nonsmooth objective $h$ can be a *regularizer* such as a sparsity-inducing penalty, model simple constraints such as $x$ belonging to a simple convex set, or be a combination of both.
All $f$, $h$ and $c$ can be nonconvex.
Together with the companion library [ShiftedProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl) described below, RegularizedOptimization.jl provides a modular and extensible framework for solving \eqref{eq:nlp}, and developing novel solvers.
Currently, the following solvers are implemented:

- **Trust-region solvers TR and TRDH** [@aravkin-baraldi-orban-2022;@leconte-orban-2023]
- **Quadratic regularization solvers R2, R2DH and R2N** [@diouane-habiboullah-orban-2024;@aravkin-baraldi-orban-2022]
- **Levenberg-Marquardt solvers LM and LMTR** [@aravkin-baraldi-orban-2024] used when $f$ is a least-squares residual
- **Augmented Lagrangian solver AL** [@demarchi-jia-kanzow-mehlitz-2023].

All solvers rely on first derivatives of $f$ and $c$, and optionally on their second derivatives in the form of Hessian-vector products.
If second derivatives are not available or too costly to compute, quasi-Newton approximations can be used.
In addition, the proximal mapping of the nonsmooth part $h$, or adequate models thereof, must be evaluated.
At each iteration, a step is computed by solving a subproblem of the form \eqref{eq:nlp} inexactly, in which $f$, $h$, and $c$ are replaced with appropriate models about the current iterate.
The solvers R2, R2DH and TRDH are particularly well suited to solve the subproblems, though they are general enough to solve \eqref{eq:nlp}.
All solvers have a non-monotone mode that enhance performance in practice on certain problems [@leconte-orban-2023;@diouane-habiboullah-orban-2024].
All are implemented in an in-place fashion, so that re-solves incur no allocations.
To illustrate our claim of extensibility, a first version of the AL solver was implemented and submitted by an external contributor.

<!-- ## Requirements of the ShiftedProximalOperators.jl -->
<!---->
<!-- The nonsmooth part $h$ must have a computable proximal mapping, defined as -->
<!-- $$\text{prox}_{\nu h}(v) = \underset{x \in \mathbb{R}^n}{\arg\min} \frac{1}{2} \|x - v\|^2 + \nu h(x).$$ -->
<!---->
<!-- While [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl) provides many standard proximal mappings, [ShiftedProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl) also supplies **shifted** variants of these mappings which is not supported by [ProximalOperators.jl](https://www.github.com/JuliaFirstOrder/ProximalOperators.jl). -->

# Statement of need

## Model-based framework for nonsmooth methods

In Julia, \eqref{eq:nlp} can be solved using [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl), which implements splitting schemes and  line-search–based methods [@stella-themelis-sopasakis-patrinos-2017;@themelis-stella-patrinos-2017].
Among others, the **PANOC** [@stella-themelis-sopasakis-patrinos-2017] solver takes a step along a direction $d$, which depends on the gradient of $f$ modified by a L-BFGS Quasi-Newton approximation, followed by proximal steps on the nonsmooth part $h$.

By contrast, [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) focuses on model-based trust-region and quadratic regularization methods, which typically require fewer evaluations of $f$ and its gradient than first-order line search methods, at the expense of more evaluations of proximal operators [@aravkin-baraldi-orban-2022].
However, each proximal computation is inexpensive for numerous commonly used choices of $h$, such as separable penalties and bound constraints (see examples below), so that the overall approach is efficient for large-scale problems.

When computing a step by (approximately) minimizing a model, [ShiftedProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl) implements efficient allocation-free shifted proximal mappings.
Specifically, it supports shifted proximal operators of the form
$$
    \underset{t \in \mathbb{R}^n}{\arg\min} \, { \tfrac{1}{2} \|t - q\|_2^2 + \nu \psi(t + s; x) + \chi(s + t \mid \Delta \mathbb{B})}
$$
where $q$ is given, $x$ and $s$ are fixed shifts, $\chi(\cdot \mid \Delta \mathbb{B})$ is the indicator of a ball of radius $\Delta > 0$ defined by a certain norm, and $\psi(\cdot; x)$ is a model of $h$ about $x$.
It is common to set $\psi(t + s; x) = h(x + s + t)$.

These shifted operators allow to (i) incorporate bound or trust-region constraints via the indicator, which is required for the **TR** and **TRDH** solvers, and (ii) evaluate the above in place, without additional allocations, which is currently not possible with ProximalOperators.jl.

RegularizedOptimization.jl provides a consistent API to formulate optimization problems and apply different solvers.
It integrates seamlessly with the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers)  [@jso] ecosystem, an academic organization for nonlinear optimization software development, testing, and benchmarking.

The smooth objective $f$ can be defined via [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) [@orban-siqueira-nlpmodels-2020], which provides a standardized Julia API for representing nonlinear programming (NLP) problems.
Large collections of such problems are available in [CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl) [@orban-siqueira-cutest-2020] and [OptimizationProblems.jl](https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl) [@migot-orban-siqueira-optimizationproblems-2023], but a use can easily interface or model their own smooth objective.

The nonsmooth term $h$ can be modeled using [ProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ProximalOperators.jl), which provides a broad collection of regularizers and indicators of simple sets.

With $f$ and $h$ modeled as discussed above, the companion package [RegularizedProblems.jl](https://github.com/JuliaSmoothOptimizers/RegularizedProblems.jl) provides a straightforward way to pair them into a *Regularized Nonlinear Programming Model*

```julia
reg_nlp = RegularizedNLPModel(f, h)
```

They can also be paired into a *Regularized Nonlinear Least Squares Model* if $f(x) = \tfrac{1}{2} \|F(x)\|^2$ for some residual $F: \mathbb{R}^n \to \mathbb{R}^m$, as would be the case with the **LM** and **LMTR** solvers.

```julia
reg_nls = RegularizedNLSModel(f, h)
```

RegularizedProblems.jl also provides a set of instances commonly used in data science and in the nonsmooth optimization literature, where several choices of $f$ can be paired with various nonsmooth terms $h$.
This design makes for a convenient source of reproducible problem instances for testing and benchmarking the solvers in [RegularizedOptimization.jl](https://www.github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl).

## Support for both exact and approximate Hessian

In contrast with [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl), [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl), methods such as **R2N** and **TR** methods support exact Hessians as well as several Hessian approximations of $f$.
Hessian–vector products $v \mapsto Hv$ can be obtained via automatic differentiation through [ADNLPModels.jl](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl) or implemented manually.
Limited-memory and diagonal quasi-Newton approximations can be selected from [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl).
This design allows solvers to exploit second-order information without explicitly forming dense or sparse Hessians, which is often expensive in time and memory, particularly at large scale.

## Testing and documentation

The package includes a comprehensive suite of unit tests that cover all functionalities, ensuring reliability and correctness.
Extensive documentation is provided, including a user guide, API reference, and examples to help users get started quickly.
Documentation is built using Documenter.jl.

## Application

A novel implementation of the exact penalty approach [@diouane-gollier-orban-2024] for equality-constrained smooth optimization is being developed based on RegularizedOptimization.jl.
In it, $h(x) = \|c(x)\|$ and the model $\psi(\cdot; x)$ differs from $h$ itself.
Specifically, $\psi(\cdot; x)$ is the norm of a linearization of $c$ about $x$.
This is not covered in the current version of [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl).

# Examples

We illustrate the capabilities of [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) on two nonsmooth and nonconvex problems:

- **Support Vector Machine (SVM) with $\ell_{1/2}^{1/2}$ penalty** for image classification [@aravkin-baraldi-orban-2024].  
- **Nonnegative Matrix Factorization (NNMF) with $\ell_0$ penalty and bound constraints** [@kim-park-2008].

Below is a condensed example showing how to define and solve SVM problem, and perform a solve followed by a re-solve:

```julia
using LinearAlgebra, Random, ProximalOperators
using NLPModels, RegularizedProblems, RegularizedOptimization
using MLDatasets

Random.seed!(1234)
model, nls_model, _ = RegularizedProblems.svm_train_model()  # Build SVM model
f = LSR1Model(model)                                         # L-SR1 Hessian approximation
λ = 1.0                                                      # Regularization parameter
h = RootNormLhalf(λ)                                       # Nonsmooth term
reg_nlp = RegularizedNLPModel(f, h)                          # Regularized problem
solver = R2NSolver(reg_nlp)                                  # Choose solver
stats  = RegularizedExecutionStats(reg_nlp)
solve!(solver, reg_nlp, stats; atol=1e-4, rtol=1e-4, verbose=1, sub_kwargs=(max_iter=200,))
solve!(solver, reg_nlp, stats; atol=1e-5, rtol=1e-5, verbose=1, sub_kwargs=(max_iter=200,))
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

We compare **TR**, **R2N**, **LM** and **LMTR** from our library.

We report the following solver statistics in the table: the convergence status of each solver, the number of evaluations of $f$, the number of evaluations of $\nabla f$, the number of proximal operator evaluations, the elapsed time in seconds and the final objective value.
On the SVM and NNMF problems, we use limited-memory SR1 and BFGS Hessian approximations, respectively.
The subproblem solver is **R2**.

\input{examples/Benchmark.tex}

- Note that for the **LM** and **LMTR** solvers, gradient evaluations count $\#\nabla f$ equals the number of Jacobian–vector and adjoint-Jacobian–vector products.

All methods successfully reduced the optimality measure below the specified tolerance of $10^{-4}$, and thus converged to an approximate first-order stationary point.
Note that, the final objective values differ due to the nonconvexity of the problems.

- **SVM with $\ell^{1/2}$ penalty:** **R2N** is the fastest, requiring the fewest gradient evaluations compared to all the other solvers.
However, it requires more proximal evaluations, but these are inexpensive.
**LMTR** and **LM** require the fewest function evaluations, but incur many Jacobian–vector products, and are the slowest.
Note that here, **LMTR** achieves the lowest objective value.
- **NNMF with constrained $\ell_0$ penalty:** **LMTR** is the fastest, and requires a fewer number of function evaluations than all the other solvers. Followed by **TR** which is the second fastest and requires the fewest gradient evaluations, however it achieves the highest objective value.
Note that both **LMTR** and **LM** achieve the lowest objective value.

Additional tests (e.g., other regularizers, constraint types, and scaling dimensions) have also been conducted, and a full benchmarking campaign is currently underway.

# Conclusion

The experiments highlight the effectiveness of the solvers implemented in [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl).

<!-- On these examples, the performance of the solvers can be summarized as follows: -->
<!---->
<!-- - **Function and gradient evaluations:** **TR** and **R2N** are the most efficient choices when aiming to minimize both. -->
<!-- - **Function evaluations only:** **LM** is preferable when the problem is a nonlinear least squares problem, as it achieves the lowest number of function evaluations. -->
<!-- - **Proximal iterations:** **PANOC** requires the fewest proximal iterations. However, in most nonsmooth applications, proximal steps are relatively inexpensive, so this criterion is of limited practical relevance. -->

In ongoing research, the package will be extended with algorithms that enable to reduce the number of proximal evaluations, especially when the proximal mapping of $h$ is expensive to compute.

# Acknowledgements

The authors would like to thank Alberto De Marchi for his implementation of the Augmented Lagrangian solver.
Mohamed Laghdaf Habiboullah is supported by an excellence FRQNT grant.
Youssef Diouane, Maxence Gollier and Dominique Orban are partially supported by an NSERC Discovery Grant.

# References
