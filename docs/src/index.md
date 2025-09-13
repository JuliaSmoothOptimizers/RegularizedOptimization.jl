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

## Regularizers

Regularizers used in this package are based on the [ShiftedProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl) API, which is related to [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl). 

The solvers in this package work by approximating the regularizer with a *shifted model*.
That is, at each iterate $x_k$, we approximate $h(x_k + s)$ with a (simpler) function $\psi(s; x_k)$.
For example, if $h(x) = \|x\|$, then its *shifted model* is simply the function $h$ itself : $\psi(s; x_k) = \|x_k + s\|$.
On the other hand, if $h$ is the composition of a norm with a function, $h(x) = \|c(x)\|$, then its *shifted model* can be the approximation
```math
\psi(s; x_k) = \|c(x_k) + J(x_k)s\| \approx \|c(x_k + s) \| = h(x_k + s),
```
where $J(x_k)$ is the Jacobian of $c$ at the point $x_k$.

Basically, we expect a regularizer `h::Foo` to

- Be callable with vectors, i.e. to implement `(h::Foo)(x::AbstractVector)`.
- Be *shifteable*, that is, to implement a function `shifted(h::Foo, x::AbstractVector)` that returns the shifted model `ψ::ShiftedFoo`.

Next, we expect the shifted model `ψ::ShiftedFoo` to 

- Be callable with vectors, i.e. to implement `(ψ::ShiftedFoo)(x::AbstractVector)`.
- Be *shifteable*, that is, to implement a function `shifted(ψ::ShiftedFoo, x::AbstractVector)` that returns a shifted model `ψ'::ShiftedFoo`. Moreover, we should be able to change the shift in place, that is, the function `shift!(ψ::ShiftedFoo, x::AbstractVector)` should be implemented as well.
- Be *proximable*, that is, to implement the inplace proximal mapping `prox!(y::AbstractVector, ψ::ShiftedFoo, q::AbstractVector, σ::Real)`.

The proximal mapping is defined as 
```math
\text{prox}(\psi, q, \sigma) := \argmin_y \ \psi(y) + \frac{\sigma}{2} \|y - q\|_2^2.
```

!!! note
    The package [ShiftedProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl) mostly implements the shifted models `ψ`. 
    For the unshifted version, these are often implemented in [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl) so that you might actually need to install the latter. For example, if you wish to use the L0 norm as a regularizer, then you should define `h` as `h = NormL0(1.0)` with [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl), you don't need to do anything else in this case because the shifted model of the L0 norm is already implemented in [ShiftedProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl). 

!!! warning 
    The shifted model being proximable means that our solvers will not be able to automagically solve with any nonsmooth function that is given to it. Rather, the user is expected to provide an efficient solver for the proximal mapping.

The following table shows which regularizers are readily available and which dependency is required to use the regularizer (the shifted model is always in `ShiftedProximalOperators.jl`).

Regularizer | Shifted Model | Julia | Dependency
------------|---------------|-------|-----------
$\lambda ∥x∥_0$ | $\lambda ∥x + s∥_0$ | [`NormL0(λ)`](https://juliafirstorder.github.io/ProximalOperators.jl/stable/functions/#ProximalOperators.NormL0) | [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl)
$\lambda ∥x∥_1$ | $\lambda ∥x + s∥_1$ | [`NormL1(λ)`](https://juliafirstorder.github.io/ProximalOperators.jl/stable/functions/#ProximalOperators.NormL1) | [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl)
$\lambda ∥x∥_2$ | $\lambda ∥x + s∥_2$ | [`NormL2(λ)`](https://juliafirstorder.github.io/ProximalOperators.jl/stable/functions/#ProximalOperators.NormL2) | [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl)
$\lambda ∥c(x)∥_2$ | $\lambda ∥c(x) + J(x)s∥_2$ | [`CompositeNormL2(λ)`](https://jso.dev/ShiftedProximalOperators.jl/dev/reference/#ShiftedProximalOperators.CompositeNormL2) | [ShiftedProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl)

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