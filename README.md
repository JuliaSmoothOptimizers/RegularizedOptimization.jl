# RegularizedOptimization

[![CI](https://github.com/UW-AMO/RegularizedOptimization.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/UW-AMO/RegularizedOptimization.jl/actions/workflows/ci.yml)
[![](https://img.shields.io/badge/docs-latest-3f51b5.svg)](https://UW-AMO.github.io/RegularizedOptimization.jl/dev)
[![codecov](https://codecov.io/gh/UW-AMO/RegularizedOptimization/branch/master/graph/badge.svg?token=LFPTDGDTP6)](https://codecov.io/gh/UW-AMO/RegularizedOptimization)

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
pkg> add https://github.com/UW-AMO/RegularizedOptimization.jl
```

## What is Implemented?

Please refer to the documentation.

## References

1. A. Y. Aravkin, R. Baraldi and D. Orban, *A Proximal Quasi-Newton Trust-Region Method for Nonsmooth Regularized Optimization*, Cahier du GERAD G-2021-12, GERAD, Montréal, Canada. https://arxiv.org/abs/2103.15993
2. R. Baraldi, R. Kumar, and A. Aravkin (2019), [*Basis Pursuit De-noise with Non-smooth Constraints*](https://doi.org/10.1109/TSP.2019.2946029), IEEE Transactions on Signal Processing, vol. 67, no. 22, pp. 5811-5823.

