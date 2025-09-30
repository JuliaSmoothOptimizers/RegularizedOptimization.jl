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

[RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) is a Julia package that implements a family of quadratic regularization and trust-region type algorithms for solving nonsmooth optimization problems of the form:
\begin{equation}\label{eq:nlp}
    \underset{x \in \mathbb{R}^n}{\text{minimize}} \quad f(x) + h(x),
\end{equation}
where $f: \mathbb{R}^n \to \mathbb{R}$ is continuously differentiable on $\mathbb{R}^n$, and $h: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ is lower semi-continuous.
Both $f$ and $h$ may be nonconvex.

The library provides a modular and extensible framework for experimenting with nonsmooth and nonconvex optimization algorithms, including:

- **Trust-region methods (TR, TRDH)** [@aravkin-baraldi-orban-2022;@leconte-orban-2023],
- **Quadratic regularization methods (R2, R2N)** [@diouane-habiboullah-orban-2024;@aravkin-baraldi-orban-2022],
- **Levenbergh-Marquardt methods (LM, LMTR)** [@aravkin-baraldi-orban-2024].
- **Augmented Lagrangian methods (AL)** [@demarchi-jia-kanzow-mehlitz-2023].

These methods rely on the gradient and optionnally on the Hessian(-vector) information of the smooth part $f$ and the proximal mapping of the nonsmooth part $h$ in order to compute steps.
Then, the objective function $f + h$ is used only to accept or reject trial points.
Moreover, they can handle cases where Hessian approximations are unbounded [@diouane-habiboullah-orban-2024;@leconte-orban-2023-2], making the package particularly suited for large-scale, ill-conditioned, and nonsmooth problems.

# Statement of need

## Model-based framework for nonsmooth methods

In Julia, \eqref{eq:nlp} can be solved using [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl), which implements in-place, first-order, line-search–based methods[@stella-themelis-sopasakis-patrinos-2017;@themelis-stella-patrinos-2017].
Most of these methods are generally splitting schemes that alternate between taking steps along the gradient of the smooth part $f$ (or quasi-Newton directions) and applying proximal steps on the nonsmooth part $h$.
Currently, [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl) provides only L-BFGS as a quasi-Newton option.
By contrast, [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) focuses on model-based approaches such as trust-region and quadratic regularization algorithms.
As shown in [@aravkin-baraldi-orban-2022], model-based methods typically require fewer evaluations of the objective and its gradient than first-order line search methods, at the expense of solving more involved subproblems.
Although these subproblems may require many proximal iterations, each proximal computation is inexpensive for several commonly used nonsmooth functions, such as separable penalties and bound constraints (see examples below), making the overall approach efficient for large-scale problems.

Building on this perspective, [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) implements state-of-the-art algorithms for solving problems of the form $f(x) + h(x)$, where $f$ is smooth and $h$ is nonsmooth.
The package provides a consistent API to formulate optimization problems and apply different regularization methods.
It integrates seamlessly with the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers) ecosystem, an academic organization for nonlinear optimization software development, testing, and benchmarking.

On the one hand, smooth problems $f$ can be defined via [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) [@orban-siqueira-nlpmodels-2020], which provides a standardized Julia API for representing nonlinear programming (NLP) problems.
Large collections of such problems are available in [Cutest.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl) [@orban-siqueira-cutest-2020] and [OptimizationProblems.jl](https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl) [@migot-orban-siqueira-optimizationproblems-2023].
Another option is to use [RegularizedProblems.jl](https://github.com/JuliaSmoothOptimizers/RegularizedProblems.jl), which provides problem instances commonly used in the nonsmooth optimization literature.

On the other hand, Hessian approximations of these functions, including quasi-Newton and diagonal schemes, can be specified through [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl), which represents Hessians as linear operators and implements efficient Hessian–vector products.

Finally, nonsmooth terms $h$ can be modeled using [ProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ProximalOperators.jl), which provides a broad collection of nonsmooth functions, together with [ShiftedProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl), which provides shifted proximal mappings for nonsmooth functions.

## Support for Hessians of the smooth part $f$

In contrast to [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl), [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) methods such as **R2N** and **TR** support Hessians of $f$, which can significantly improve convergence rates, especially for ill-conditioned problems.
Hessians can be obtained via automatic differentiation through [ADNLPModels.jl](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl) or supplied directly as Hessian–vector products $v \mapsto Hv$.
This enables algorithms to exploit second-order information without explicitly forming dense (or sparse) Hessians, which is often prohibitively expensive in both computation and memory, particularly in high-dimensional settings.

## Requirements of the RegularizedProblems.jl

To model the problem \eqref{eq:nlp}, one defines the smooth part $f$ and the nonsmooth part $h$ as discussed above.
The package [RegularizedProblems.jl](https://github.com/JuliaSmoothOptimizers/RegularizedProblems.jl) provides a straightforward way to create such instances, called *Regularized Nonlinear Programming Models*:

```julia
reg_nlp = RegularizedNLPModel(f, h)
```

This design makes it a convenient source of reproducible problem instances for testing and benchmarking algorithms in the repository [@diouane-habiboullah-orban-2024;@aravkin-baraldi-orban-2022;@aravkin-baraldi-orban-2024;@leconte-orban-2023-2].

## Requirements of the ShiftedProximalOperators.jl

The nonsmooth part $h$ must have a computable proximal mapping, defined as
$$\text{prox}_{h}(v) = \underset{x \in \mathbb{R}^n}{\arg\min} \left( h(x) + \frac{1}{2} \|x - v\|^2 \right).$$
This requirement is satisfied by a wide range of nonsmooth functions commonly used in practice, such as $\ell_1$ norm, $\ell_0$ "norm", indicator functions of convex sets, and group sparsity-inducing norms.
The package [ProximalOperators.jl](https://www.github.com/FirstOrder/ProximalOperators.jl) provides a comprehensive collection of such functions, along with their proximal mappings.
The main difference between the proximal operators implemented in
[ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl)
is that those implemented here involve a translation of the nonsmooth term.
Specifically, this package considers proximal operators defined as
$$
    \underset{t \in \mathbb{R}^n}{\arg\min} \, { \tfrac{1}{2} ‖t - q‖₂² + ν h(x + s + t) + χ(s + t|ΔB)}
$$
where $q$ is given, $x$ and $s$ are fixed shifts, $h$ is the nonsmooth term with respect
to which we are computing the proximal operator, and $χ(.; \Delta B)$ is the indicator of
a ball of radius $\Delta$ defined by a certain norm.
This package enables to encode this shifted proximal operator through without adding allocations and allowing to solve problem \eqref{eq:nlp} with bound constraints.

## Testing and documentation

The package includes a comprehensive suite of unit tests that cover all functionalities, ensuring reliability and correctness.
Extensive documentation is provided, including a user guide, API reference, and examples to help users get started quickly.
Aqua.jl is used to test the package dependencies.
Documentation is built using Documenter.jl.

## Solvers caracteristics

All solvers in [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) are implemented in an in-place fashion, minimizing memory allocations during the resolution process.
Moreover, they implement non-monotone strategies to accept trial points, which can enhance algorithmic performance in practice [@leconte-orban-2023;@diouane-habiboullah-orban-2024].

## Application studies

The package is used to solve equality-constrained optimization problems by means of the exact penalty approach [@diouane-gollier-orban-2024] where the model of the nonsmooth part differs from the function $h$ itself.
This is not covered in the current version of the package [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl).

## Support for inexact subproblem solves

Solvers in [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) allow inexact resolution of trust-region and quadratic-regularized subproblems using first-order that are implemented in the package itself such as the quadratic regularization method R2 [@aravkin-baraldi-orban-2022] and R2DH [@diouane-habiboullah-orban-2024] with trust-region variants TRDH [@leconte-orban-2023-2].

This is crucial for large-scale problems where exact subproblem solutions are prohibitive.
Moreover, one way to outperform line-search–based methods is to solve the subproblems more accurately by performing many proximal iterations, which are inexpensive to compute, rather than relying on numerous function and gradient evaluations.
We will illustrate this in the examples below.


# Examples


We consider two examples where the smooth part $f$ is nonconvex and the nonsmooth part $h$ is either $\ell^{1/2}$ or $\ell_0$ norm with constraints.

We compare the performance of our solvers with (**PANOC**) solver [@stella-themelis-sopasakis-patrinos-2017] implemented in [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl).

We illustrate the capabilities of [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) on two nonsmooth and nonconvex problems:

- **Support Vector Machine (SVM) with $\ell^{1/2}$ penalty** for image classification [@aravkin-baraldi-orban-2024].  
- **Nonnegative Matrix Factorization (NNMF) with $\ell_0$ penalty and constraints** [@kim-park-2008].

Both problems are of the form $\min f(x) + h(x)$ with $f$ nonconvex and $h$ nonsmooth.  
The NNMF problem can be set up in a similar way to the SVM case, with $h$ given by an $\ell_0$ norm and additional nonnegativity constraints.
Below is a condensed example showing how to define and solve such problems:

```julia
using LinearAlgebra, Random, ProximalOperators
using NLPModels, RegularizedProblems, RegularizedOptimization

Random.seed!(1234)
model, nls, _ = RegularizedProblems.svm_train_model()       # Build SVM model
f = LSR1Model(model)                                        # Hessian approximation
h = RootNormLhalf(1.0)                                      # Nonsmooth term
reg_nlp = RegularizedNLPModel(f, h)                        # Regularized problem
solver = R2NSolver(reg_nlp)                                 # Choose solver
stats  = RegularizedExecutionStats(reg_nlp)
solve!(solver, reg_nlp, stats; x=f.meta.x0, atol=1e-4, rtol=1e-4, verbose=0, sub_kwargs=(max_iter=200,))
solve!(solver, reg_nlp, stats; x=f.meta.x0, atol=1e-5, rtol=1e-5, verbose=0, sub_kwargs=(max_iter=200,))
```
The NNMF problem can be set up in a similar way, replacing the model by nnmf_model(...) with bound constraints, $h$ by an $\ell_0$ norm and use an L-BFGS Hessian approximation.

### Numerical results

We compare **PANOC** (from [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl)) with **TR**, **R2N**, and **LM** from our library.  
The results are summarized in the combined table below:

```
┌───────────────────┬─────────────┬──────────┬──────┬──────┬───────┐
│ Method            │   Status    │ Time (s) │   #f │  #∇f │ #prox │
├───────────────────┼─────────────┼──────────┼──────┼──────┼───────┤
│ PANOC (SVM)       │ first_order │   38.226 │ 3713 │ 3713 │  2269 │
│ TR (LSR1, SVM)    │ first_order │    5.912 │  347 │  291 │  4037 │
│ R2N (LSR1, SVM)   │ first_order │   1.2944 │   86 │   76 │  8586 │
│ TR (LBFGS, NNMF)  │ first_order │   0.0857 │   42 │   40 │  3160 │
│ R2N (LBFGS, NNMF) │ first_order │   0.2116 │   79 │   76 │  6273 │
│ LM (NNMF)         │ first_order │   0.1363 │    8 │ 7540 │  2981 │
└───────────────────┴─────────────┴──────────┴──────┴──────┴───────┘
```

### Discussion

- **SVM with $\ell^{1/2}$ penalty:** TR and R2N require far fewer function and gradient evaluations than PANOC, at the expense of more proximal iterations. Since each proximal step is inexpensive, TR and R2N are much faster overall.  
- **NNMF with constrained $\ell_0$ penalty:** R2N slightly outperforms TR, while LM is competitive in terms of function calls but incurs many gradient evaluations.  

Additional tests (e.g., other regularizers, constraint types, and scaling dimensions) have also been conducted, and a full benchmarking campaign is currently underway.

## Conclusion

The experiments highlight the effectiveness of the solvers implemented in [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) compared to **PANOC** from [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl).

On these examples, the performance of the solvers can be summarized as follows:

- **Function and gradient evaluations:** **TR** and **R2N** are the most efficient choices when aiming to minimize both.
- **Function evaluations only:** **LM** is preferable when the problem is a nonlinear least squares problem, as it achieves the lowest number of function evaluations.
- **Proximal iterations:** **PANOC** requires the fewest proximal iterations. However, in most nonsmooth applications, proximal steps are relatively inexpensive, so this criterion is of limited practical relevance.

# Acknowledgements

Mohamed Laghdaf Habiboullah is supported by an excellence FRQNT grant.
Youssef Diouane and Dominique Orban are partially supported by an NSERC Discovery Grant.

# References
