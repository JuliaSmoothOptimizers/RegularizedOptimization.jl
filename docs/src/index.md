# RegularizedOptimization.jl

This package implements a family of algorithms to solve nonsmooth optimization problems of the form

```math
\underset{x \in \mathbb{R}^n}{\text{minimize}} \quad f(x) + h(x),
```

where $f : \mathbb{R}^n \to \mathbb{R}$ is continuously differentiable and $h : \mathbb{R}^n \to \mathbb{R} \cup \{\infty\}$ is lower semi-continuous and proper.
Both $f$ and $h$ may be **nonconvex**.

All solvers implemented in this package are **JuliaSmoothOptimizers-compliant**.  
They take a [`RegularizedNLPModel`](https://jso.dev/RegularizedProblems.jl/dev/reference#RegularizedProblems.RegularizedNLPModel) as input and return a [`GenericExecutionStats`](https://jso.dev/SolverCore.jl/stable/reference/#SolverCore.GenericExecutionStats).  

A [`RegularizedNLPModel`](https://jso.dev/RegularizedProblems.jl/stable/reference#RegularizedProblems.RegularizedNLPModel) contains:  

- a smooth component `f` represented as an [`AbstractNLPModel`](https://github.com/JuliaSmoothOptimizers/NLPModels.jl),  
- a nonsmooth regularizer `h`.  

We refer to [jso.dev](https://jso.dev) for tutorials on the `NLPModel` API. This framework allows the usage of models from  

- AMPL ([AmplNLReader.jl](https://github.com/JuliaSmoothOptimizers/AmplNLReader.jl)),  
- CUTEst ([CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl)),  
- JuMP ([NLPModelsJuMP.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsJuMP.jl)),  
- PDE-constrained problems ([PDENLPModels.jl](https://github.com/JuliaSmoothOptimizers/PDENLPModels.jl)),  
- models defined with automatic differentiation ([ADNLPModels.jl](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl)).

We refer to [ManualNLPModels.jl](https://github.com/JuliaSmoothOptimizers/ManualNLPModels.jl) for users interested in defining their own model.

---

## Algorithms

A presentation of each algorithm is given [here](@ref algorithms).

---

## Preallocating

All solvers in RegularizedOptimization.jl have **in-place versions**.  
Users can preallocate a workspace and reuse it across solves to avoid memory allocations, which is useful in repetitive scenarios.  

---

## How to Install

RegularizedOptimization can be installed through the Julia package manager:

```julia
julia> ]
pkg> add https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl
```

---

## Bug reports and discussions

If you think you found a bug, please open an [issue](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl/issues).  
Focused suggestions and requests can also be opened as issues. Before opening a pull request, we recommend starting an issue or a discussion first.  

For general questions not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions).  
This forum is for questions and discussions about any of the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers) packages.  