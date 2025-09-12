# RegularizedOptimization.jl

This package implements a family of algorithms that aim to solve nonsmooth optimization problems of the form
```math
\underset{x \in \mathbb{R}^n}{\text{minimize}} \quad f(x) + h(x),
```
where $f : \mathbb{R}^n \mapsto \mathbb{R}$ is continuously differentiable and $h : \mathbb{R}^n \mapsto \mathbb{R} \cup \{\infty\}$ is lower semi-continuous.
Both $f$ and $h$ can be nonconvex.

All solvers implemented in this package are JuliaSmoothOptimizers-compliant. They take the smooth part `f` as an [`AbstractNLPModel`](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) and the regularizer `h` is based on the [`ShiftedProximableFunction`](https://github.com/MaxenceGollier/ShiftedProximalOperators.jl) API. 
All solvers return a [`GenericExecutionStats`](https://github.com/JuliaSmoothOptimizers/SolverCore.jl/blob/16fc349908f46634f2c9acdddddb009b23634b71/src/stats.jl#L60).
We refer to [jso.dev](https://jso.dev) for tutorials on the NLPModel API. This framework allows the usage of models from Ampl (using [AmplNLReader.jl](https://github.com/JuliaSmoothOptimizers/AmplNLReader.jl)), CUTEst (using [CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl)), JuMP (using [NLPModelsJuMP.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsJuMP.jl)), PDE-constrained optimization problems (using [PDENLPModels.jl](https://github.com/JuliaSmoothOptimizers/PDENLPModels.jl)) and models defined with automatic differentiation (using [ADNLPModels.jl](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl)).

## Features

All solvers in RegularizedOptimization.jl have in-place versions. 
Users can preallocate a workspace for each solver and then use it to solve the problem without allocating memory.
This is useful if a problem has to be solved multiple times.
All solvers can work in any floating-point data type.

## How to Install

RegularizedOptimization can be installed through the Julia package manager:

```julia
julia> ]
pkg> add RegularizedOptimization
```

## Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/Krylov.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers) organization, so questions about any of our packages are welcome.





