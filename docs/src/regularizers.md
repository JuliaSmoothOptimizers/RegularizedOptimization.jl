# [Regularizers](@id regularizers)

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

The following table shows some of which regularizers are readily available and which dependency is required to use the regularizer (the shifted model is always in `ShiftedProximalOperators.jl`).
The user should refer to [ShiftedProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl) for a complete overview of available regularizers.


Regularizer | Shifted Model | Julia | Dependency
------------|---------------|-------|-----------
$\lambda ∥x∥_0$ | $\lambda ∥x + s∥_0$ | [`NormL0(λ)`](https://juliafirstorder.github.io/ProximalOperators.jl/stable/functions/#ProximalOperators.NormL0) | [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl)
$\lambda ∥x∥_1$ | $\lambda ∥x + s∥_1$ | [`NormL1(λ)`](https://juliafirstorder.github.io/ProximalOperators.jl/stable/functions/#ProximalOperators.NormL1) | [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl)
$\lambda ∥x∥_2$ | $\lambda ∥x + s∥_2$ | [`NormL2(λ)`](https://juliafirstorder.github.io/ProximalOperators.jl/stable/functions/#ProximalOperators.NormL2) | [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl)
$\lambda ∥x∥_{1/2}^{1/2}$ | $\lambda ∥x + s∥_{1/2}^{1/2}$ | [`RootNormLhalf(λ)`](https://jso.dev/ShiftedProximalOperators.jl/dev/reference/#ShiftedProximalOperators.RootNormLhalf) | [ShiftedProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl)
$\lambda \text{rank}(X)$ | $\lambda \text{rank}(X + S)$ | [`Rank(λ)`](https://jso.dev/ShiftedProximalOperators.jl/dev/reference/#ShiftedProximalOperators.Rank) | [ShiftedProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl)
$\lambda ∥X∥_*$ | $\lambda ∥X + S∥_*$ | [`Nuclearnorm(λ)`](https://jso.dev/ShiftedProximalOperators.jl/dev/reference/#ShiftedProximalOperators.Nuclearnorm) | [ShiftedProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl)
$\lambda ∥c(x)∥_2$ | $\lambda ∥c(x) + J(x)s∥_2$ | [`CompositeNormL2(λ)`](https://jso.dev/ShiftedProximalOperators.jl/dev/reference/#ShiftedProximalOperators.CompositeNormL2) | [ShiftedProximalOperators.jl](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl)