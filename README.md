# RegularizedOptimization

[![CI](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl/actions/workflows/ci.yml)
[![](https://img.shields.io/badge/docs-latest-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/RegularizedOptimization.jl/dev)
[![codecov](https://codecov.io/gh/JuliaSmoothOptimizers/RegularizedOptimization.jl/branch/master/graph/badge.svg?token=lTbRmyBspS)](https://codecov.io/gh/JuliaSmoothOptimizers/RegularizedOptimization.jl)
[![DOI](https://zenodo.org/badge/160387219.svg)](https://zenodo.org/badge/latestdoi/160387219)

## How to cite

If you use RegularizedOptimization.jl in your work, please cite using the format given in [CITATION.bib](CITATION.bib).

## Synopsis

This package contains solvers to solve regularized optimization problems of the form

<p align="center">
minₓ f(x) + h(x)
</p>

where f: ℝⁿ → ℝ has Lipschitz-continuous gradient and h: ℝⁿ → ℝ is lower semi-continuous and proper.
The smooth term f describes the objective to minimize while the role of the regularizer h is to select
a solution with desirable properties: minimum norm, sparsity below a certain level, maximum sparsity, etc.
Both f and h can be nonconvex.

## Installation

To install the package, hit `]` from the Julia command line to enter the package manager and type
```julia
pkg> add RegularizedOptimization
```

## What is Implemented?

Please refer to the documentation.

## Related Software

* [RegularizedProblems.jl](https://github.com/JuliaSmoothOptimizers/RegularizedProblems.jl)
* [ShiftedProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl)

## References

1. A. Y. Aravkin, R. Baraldi and D. Orban, *A Proximal Quasi-Newton Trust-Region Method for Nonsmooth Regularized Optimization*, SIAM Journal on Optimization, 32(2), pp.900&ndash;929, 2022. Technical report: https://arxiv.org/abs/2103.15993
2. R. Baraldi, R. Kumar, and A. Aravkin (2019), [*Basis Pursuit De-noise with Non-smooth Constraints*](https://doi.org/10.1109/TSP.2019.2946029), IEEE Transactions on Signal Processing, vol. 67, no. 22, pp. 5811-5823.

```bibtex
@article{aravkin-baraldi-orban-2022,
  author = {Aravkin, Aleksandr Y. and Baraldi, Robert and Orban, Dominique},
  title = {A Proximal Quasi-{N}ewton Trust-Region Method for Nonsmooth Regularized Optimization},
  journal = {SIAM Journal on Optimization},
  volume = {32},
  number = {2},
  pages = {900--929},
  year = {2022},
  doi = {10.1137/21M1409536},
  abstract = { We develop a trust-region method for minimizing the sum of a smooth term (f) and a nonsmooth term (h), both of which can be nonconvex. Each iteration of our method minimizes a possibly nonconvex model of (f + h) in a trust region. The model coincides with (f + h) in value and subdifferential at the center. We establish global convergence to a first-order stationary point when (f) satisfies a smoothness condition that holds, in particular, when it has a Lipschitz-continuous gradient, and (h) is proper and lower semicontinuous. The model of (h) is required to be proper, lower semi-continuous and prox-bounded. Under these weak assumptions, we establish a worst-case (O(1/\epsilon^2)) iteration complexity bound that matches the best known complexity bound of standard trust-region methods for smooth optimization. We detail a special instance, named TR-PG, in which we use a limited-memory quasi-Newton model of (f) and compute a step with the proximal gradient method, resulting in a practical proximal quasi-Newton method. We establish similar convergence properties and complexity bound for a quadratic regularization variant, named R2, and provide an interpretation as a proximal gradient method with adaptive step size for nonconvex problems. R2 may also be used to compute steps inside the trust-region method, resulting in an implementation named TR-R2. We describe our Julia implementations and report numerical results on inverse problems from sparse optimization and signal processing. Both TR-PG and TR-R2 exhibit promising performance and compare favorably with two linesearch proximal quasi-Newton methods based on convex models. }
}
```

## Contributing 

Please refer to [this](https://jso.dev/contributing/) for contribution guidelines.
