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
2. A. Y. Aravkin, R. Baraldi and D. Orban, *A Levenberg-Marquardt Method for Nonsmooth Regularized Least Squares*, SIAM Journal on Scientific Computing, 46(4), pp.2557&ndash;2581, 2024. Technical report: https://arxiv.org/abs/2301.02347
3. G. Leconte and D. Orban, *The Indefinite Proximal Gradient Method*, Computational Optimization and Applications, 91(2), pp.861&ndash;903, 2025. Technical report: https://arxiv.org/abs/2309.08433

```bibtex
@Article{   aravkin-baraldi-orban-2022,
  Author  = {Aravkin, Aleksandr Y. and Baraldi, Robert and Orban, Dominique},
  Title   = {A Proximal Quasi-{N}ewton Trust-Region Method for Nonsmooth Regularized Optimization},
  Journal = {SIAM J. Optim.},
  Year    = 2022,
  Volume  = 32,
  Number  = 2,
  Pages   = {900--929},
  doi     = {10.1137/21M1409536},
}

@Article{   aravkin-baraldi-orban-2024,
  Author  = {A. Y. Aravkin and R. Baraldi and D. Orban},
  Title   = {A {L}evenberg–{M}arquardt Method for Nonsmooth Regularized Least Squares},
  Journal = {SIAM J. Sci. Comput.},
  Year    = 2024,
  Volume  = 46,
  Number  = 4,
  Pages   = {A2557--A2581},
  doi     = {10.1137/22M1538971},
}

@Article{   leconte-orban-2025,
  Author  = {G. Leconte and D. Orban},
  Title   = {The Indefinite Proximal Gradient Method},
  Journal = {Comput. Optim. Appl.},
  Year    = 2025,
  Volume  = 91,
  Number  = 2,
  Pages   = {861--903},
  doi     = {10.1007/s10589-024-00604-5},
}
```

## Contributing 

Please refer to [this](https://jso.dev/contributing/) for contribution guidelines.
