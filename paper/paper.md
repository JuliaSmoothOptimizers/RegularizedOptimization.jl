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

[RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) is a Julia [@bezanson-edelman-karpinski-shah-2017] package that implements a family of quadratic regularization and trust-region type algorithms for solving nonsmooth optimization problems of the form:
\begin{equation}\label{eq:nlp}
    \underset{x \in \mathbb{R}^n}{\text{minimize}} \quad f(x) + h(x),
\end{equation}
where $f: \mathbb{R}^n \to \mathbb{R}$ is continuously differentiable on $\mathbb{R}^n$, and $h: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ is lower semi-continuous.
Both $f$ and $h$ may be nonconvex.

The library provides a modular and extensible framework for experimenting with nonsmooth and nonconvex optimization algorithms, including:

- **Trust-region methods (TR, TRDH)** [@aravkin-baraldi-orban-2022;@leconte-orban-2023],
- **Quadratic regularization methods (R2, R2N)** [@diouane-habiboullah-orban-2024;@aravkin-baraldi-orban-2022],
- **Levenbergh-Marquardt methods (LM, LMTR)** [@aravkin-baraldi-orban-2024].
- **Augmented Lagrangian methods (ALTR)** (cite?).

These methods rely solely on the gradient and Hessian(-vector) information of the smooth part $f$ and the proximal mapping of the nonsmooth part $h$ in order to compute steps.
Then, the objective function $f + h$ is used only to accept or reject trial points.
Moreover, they can handle cases where Hessian approximations are unbounded [@diouane-habiboullah-orban-2024;@leconte-orban-2023-2], making the package particularly suited for large-scale, ill-conditioned, and nonsmooth problems.

# Statement of need

## Model-based framework for nonsmooth methods

There exists a way to solve \eqref{eq:nlp} in Julia using [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl), which implements in-place first-order line search–based methods for \eqref{eq:nlp}.
Most of these methods are generally splitting schemes that alternate between taking steps along the gradient of the smooth part $f$ (or quasi-Newton directions) and applying proximal steps on the nonsmooth part $h$.
Currently, [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl) provides only L-BFGS as a quasi-Newton option.
By contrast, [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) focuses on model-based approaches such as trust-region and regularization algorithms.
As shown in [@aravkin-baraldi-orban-2022], model-based methods typically require fewer evaluations of the objective and its gradient than first-order line search methods, at the expense of solving more involved subproblems.
Although these subproblems may require many proximal iterations, each proximal computation is inexpensive, making the overall approach efficient for large-scale problems.

Building on this perspective, [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) implements state-of-the-art regularization-based algorithms for solving problems of the form $f(x) + h(x)$, where $f$ is smooth and $h$ is nonsmooth.
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

## Requirements of the RegularizedProblems.jl package

To model the problem \eqref{eq:nlp}, one defines the smooth part $f$ and the nonsmooth part $h$ as discussed above.
The package [RegularizedProblems.jl](https://github.com/JuliaSmoothOptimizers/RegularizedProblems.jl) provides a straightforward way to create such instances, called *Regularized Nonlinear Programming Models*:

```julia
reg_nlp = RegularizedNLPModel(f, h)
```

This design makes it a convenient source of reproducible problem instances for testing and benchmarking algorithms in the repository [@diouane-habiboullah-orban-2024;@aravkin-baraldi-orban-2022;@aravkin-baraldi-orban-2024;@leconte-orban-2023-2].

## Requirements of the ShiftedProximalOperators.jl package

The nonsmooth part $h$ must have a computable proximal mapping, defined as
$$\text{prox}_{h}(v) = \underset{x \in \mathbb{R}^n}{\arg\min} \left( h(x) + \frac{1}{2} \|x - v\|^2 \right).$$
This requirement is satisfied by a wide range of nonsmooth functions commonly used in practice, such as $\ell_1$ norm, $\ell_0$ "norm", indicator functions of convex sets, and group sparsity-inducing norms.
The package [ProximalOperators.jl](https://www.github.com/FirstOrder/ProximalOperators.jl) provides a comprehensive collection of such functions, along with their proximal mappings.
The main difference between the proximal operators implemented in
[ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl)
is that those implemented here involve a translation of the nonsmooth term.
Specifically, this package considers proximal operators defined as
$$
    \underset{t \in \mathbb{R}^n}{\arg\min} \, { \tfrac{1}{2} ‖t - q‖₂² + ν h(x + s + t) + χ(s + t; ΔB) | t ∈ ℝⁿ },
$$
where $q$ is given, $x$ and $s$ are fixed shifts, $h$ is the nonsmooth term with respect
to which we are computing the proximal operator, and $χ(.; \Delta B)$ is the indicator of
a ball of radius $\Delta$ defined by a certain norm.

![Composition of JSO packages](jso-packages.pdf){ width=70% }


## Testing and documentation

The package includes a comprehensive suite of unit tests that cover all functionalities, ensuring reliability and correctness.
Extensive documentation is provided, including a user guide, API reference, and examples to help users get started quickly.
Aqua.jl is used to test the package dependencies.
Documentation is built using Documenter.jl.

## Non-monotone strategies

The solvers in [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) implement non-monotone strategies to accept trial points, which can enhance algorithmic performance in practice [@leconte-orban-2023;@diouane-habiboullah-orban-2024].

## Application studies

The package is used to solve equality-constrained optimization problems by means of the exact penalty approach [@diouane-gollier-orban-2024] where the model of the nonsmooth part differs from the function $h$ itself.
This is not covered in the current version of the package [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl).

## Support for inexact subproblem solves

Solvers in [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) allow inexact resolution of trust-region and quadratic-regularized subproblems using first-order that are implemented in the package itself such as the quadratic regularization method R2 [@aravkin-baraldi-orban-2022] and R2DH [@diouane-habiboullah-orban-2024] with trust-region variants TRDH [@leconte-orban-2023-2].

This is crucial for large-scale problems where exact subproblem solutions are prohibitive.
Moreover, one way to outperform line-search–based methods is to solve the subproblems more accurately by performing many proximal iterations, which are inexpensive to compute, rather than relying on numerous function and gradient evaluations.
We will illustrate this in the examples below.

## In-place methods

All solvers in [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) are implemented in an in-place fashion, minimizing memory allocations during the resolution process.

# Examples


We consider three examples where the smooth part $f$ is nonconvex and the nonsmooth part $h$ is either $\ell^{1/2}$ or $\ell_0$ norm with or without constraints.

We compare the performance of our solvers with (**PANOC**) solver [@stella-themelis-sopasakis-patrinos-2017] implemented in [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl).

## Problem of support vector machine with $\ell^{1/2}$ penalty


A first example addresses an image recognition task using a support vector machine (SVM) similar to those in [@aravkin-baraldi-orban-2024].
The formulation is
$$
\min_{x \in \mathbb{R}^n} \ \tfrac{1}{2} \|\mathbf{1} - \tanh(b \odot \langle A, x \rangle)\|^2 + \|x\|_{1/2}^{1/2},
$$  
where $A \in \mathbb{R}^{m \times n}$, with $n = 784$ representing the vectorized size of each image and $m = 13{,}007$ is the number of images in the training dataset.

```julia
using LinearAlgebra, Random
using ProximalOperators
using NLPModels, NLPModelsModifiers, RegularizedProblems, RegularizedOptimization
using MLDatasets

random_seed = 1234
Random.seed!(random_seed)

# Build the models
model, nls_train, _ = RegularizedProblems.svm_train_model()

# Define the Hessian approximation
f = LSR1Model(model)

# Define the nonsmooth regularizer (L0 norm)
λ = 1.0
h = RootNormLhalf(λ)

# Define the regularized NLP model
reg_nlp = RegularizedNLPModel(f, h)

# Choose a solver (R2DH) and execution statistics tracker
solver_r2n = R2NSolver(reg_nlp)
stats = RegularizedExecutionStats(reg_nlp)

# Max number of proximal iterations for subproblem solver
sub_kwargs = (max_iter=200,)

# Solve the problem 
solve!(solver_r2n, reg_nlp, stats, x = f.meta.x0, atol = 1e-4, rtol = 1e-4, verbose = 0, sub_kwargs = sub_kwargs)






```

````
┌───────────┬─────────────┬──────────┬──────┬──────┬───────┐
│ Method    │   Status    │ Time (s) │   #f │  #∇f │ #prox │
├───────────┼─────────────┼──────────┼──────┼──────┼───────┤
│ PANOC     │ first_order │  18.5413 │ 1434 │ 1434 │   934 │
│ TR(LSR1)  │ first_order │   5.8974 │  385 │  333 │ 11113 │
│ R2N(LSR1) │ first_order │   2.1251 │  175 │   95 │ 56971 │
└───────────┴─────────────┴──────────┴──────┴──────┴───────┘
````

We observe that both **TR** and **R2N** outperform **PANOC** in terms of the number of function and gradient evaluations and computational time, although they require more proximal iterations.
But since each proximal iteration is inexpensive, the overall performance is better.

## Problem of FitzHugh-Nagumo inverse with $\ell_0$ penalty

A second example is the FitzHugh-Nagumo inverse problem with an $\ell_0$ penalty, as described in [@aravkin-baraldi-orban-2022] and [@aravkin-baraldi-orban-2024].
This problem consists of recovering the parameters of a system of ordinary differential equations (ODEs) with sparsity constraints.
In general, the evaluation of the objective function and its gradient are costly because they require solving the ODEs compared to the proximal operator of the $\ell_0$ norm, which is inexpensive.

```julia
using LinearAlgebra
using ProximalOperators
using NLPModels, NLPModelsModifiers, RegularizedProblems, RegularizedOptimization
using DifferentialEquations, ADNLPModels

# Define the Fitzhugh-Nagumo problem
model, _, _ = RegularizedProblems.fh_model()
x0 = 0.1 * ones(model.meta.nvars) # initial guess

# Define the Hessian approximation
f = LBFGSModel(fh_model)

# Initialize the starting Hessian approximation scaling factor
f.op.data.scaling_factor = 1e4

# Define the nonsmooth regularizer (L1 norm)
λ = 1.0
h = NormL0(λ)

# Define the regularized NLP model
reg_nlp = RegularizedNLPModel(f, h)

# Choose a solver (TR) and execution statistics tracker
solver_tr = TRSolver(reg_nlp)
stats = RegularizedExecutionStats(reg_nlp)

# Max number of proximal iterations for subproblem solver
sub_kwargs = (max_iter=200,)

# Solve the problem
solve!(solver_tr, reg_nlp, stats, x = f.meta.x0, atol = 1e-3, rtol = 1e-3, verbose = 0, sub_kwargs = sub_kwargs)
```

````
┌────────────┬─────────────┬──────────┬─────┬─────┬───────┐
│ Method     │   Status    │ Time (s) │  #f │ #∇f │ #prox │
├────────────┼─────────────┼──────────┼─────┼─────┼───────┤
│ PANOC      │ first_order │   1.3279 │ 188 │ 188 │   107 │
│ TR(LBFGS)  │ first_order │   0.4075 │  83 │  60 │ 20983 │
│ R2N(LBFGS) │ first_order │   0.4001 │  63 │  62 │ 17061 │
└────────────┴─────────────┴──────────┴─────┴─────┴───────┘
  ````

Same observation as in the previous example: **TR** and **R2N** with LBFGS approximation of the Hessian of $f$ outperform **PANOC** in terms of the number of function and gradient evaluations and computational time, although they require more proximal iterations.

## Problem of Nonnegative least squares with $\ell_0$ penalty and constraints

The third experiment considers the sparse nonnegative matrix factorization (NNMF) problem introduced by [@kim-park-2008].
Let $A \in \mathbb{R}^{m \times n}$ be a nonnegative matrix whose columns correspond to observations drawn from a Gaussian mixture, with negative entries truncated to zero.

The goal is to obtain a factorization $A \approx WH$, where $W \in \mathbb{R}^{m \times k}$, $H \in \mathbb{R}^{k \times n}$, $k < \min(m,n)$, such that both factors are nonnegative and $H$ is sparse.

This leads to the optimization problem  

$$
\min_{W, H \geq 0} \; \tfrac{1}{2} \| A - WH \|_F^2 + \lambda \| \operatorname{vec}(H) \|_0,
$$  

where $\operatorname{vec}(H)$ denotes the column-stacked version of $H$.

Compared to the previous examples, we now consider a constrained problem with a nonsmooth and nonconvex term.

The library [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl) provides solvers that can handle constraints by separating the objective into three parts: a smooth term, a nonsmooth term, and the indicator function of the constraints. However, this approach assumes that the nonsmooth part is convex, which is not the case here.

Another approach is to merge the nonsmooth term with the indicator function of the constraints into a single nonsmooth function, and then apply **PANOC**, which is the strategy adopted here. However, the current library of proximal operators, [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl), on which [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl) relies, does not provide the proximal mapping of the sum of the $\ell_0$ norm and the indicator function of the nonnegative orthant. In contrast, [ShiftedProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl) does implement this operator.  

Therefore, to apply **PANOC** in this setting, one would first need to implement this combined proximal operator in [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl). For this reason, we do not include **PANOC** in this example.

Instead, we compare the performance of **TR** and **R2N** with that of **LM**.

```julia
using LinearAlgebra
using ProximalOperators
using NLPModels, NLPModelsModifiers, RegularizedProblems, RegularizedOptimization
using DifferentialEquations, ADNLPModels

# Build the models
m, n, k = 100, 50, 5
model, nls_model, A, selected = nnmf_model(m, n, k)

# Define the nonsmooth regularizer (L1 norm)
λ = norm(grad(model, rand(model.meta.nvar)), Inf) / 200
h = NormL0(λ)

# Define the regularized NLS model
reg_nlp = RegularizedNLSModel(nls_model, h)

# Choose a solver (TR) and execution statistics tracker
solver_lm = LMSolver(reg_nlp)
stats = RegularizedExecutionStats(reg_nlp)


# Solve the problem
solve!(solver_lm, reg_nlp, stats, x = f.meta.x0, atol = 1e-4, rtol = 1e-4, verbose = 0)
```

```
┌────────────┬─────────────┬──────────┬────┬──────┬───────┐
│ Method     │   Status    │ Time (s) │ #f │  #∇f │ #prox │
├────────────┼─────────────┼──────────┼────┼──────┼───────┤
│ TR(LBFGS)  │ first_order │   0.1727 │ 78 │   73 │ 10231 │
│ R2N(LBFGS) │ first_order │   0.1244 │ 62 │   62 │  5763 │
│ LM         │ first_order │   1.2796 │ 11 │ 2035 │   481 │
└────────────┴─────────────┴──────────┴────┴──────┴───────┘
```

We observe that **R2N** and **TR** achieve similar performance, with **R2N** being slightly better.
Both methods outperform **LM** in terms of computational time and the number of gradient evaluations.
However, **LM** requires significantly fewer function evaluations, which is expected since it is specifically designed for nonlinear least squares problems and can exploit the structure of the objective function more effectively.

## Conclusion

The experiments highlight the effectiveness of the solvers implemented in [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) compared to **PANOC** from [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl).

The performance can be summarized as follows:

- **Function and gradient evaluations:** **TR** and **R2N** are the most efficient choices when aiming to minimize both.
- **Function evaluations only:** **LM** is preferable when the problem is a nonlinear least squares problem, as it achieves the lowest number of function evaluations.
- **Proximal iterations:** **PANOC** requires the fewest proximal iterations. However, in most nonsmooth applications, proximal steps are relatively inexpensive, so this criterion is of limited practical relevance.

# Acknowledgements

Mohamed Laghdaf Habiboullah is supported by an excellence FRQNT grant.
Youssef Diouane and Dominique Orban are partially supported by an NSERC Discovery Grant.

# References
