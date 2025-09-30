# Custom regularizers

In the case where the regularizer for your application is not already implemented you can still implement one yourself and use our solvers.
In this tutorial we are going to model the (nonsmooth) function
```math
h : \mathbb{R}^n \mapsto \mathbb{R} \cup \{\infty\} : x \mapsto \chi(x; \mathbb{B}_\infty)
```
where $\chi(\cdot; A)$ is the indicator function on the set $A$ and $\mathbb{B}_\infty$ is the unit ball in the infinity norm, i.e,
```math
\chi(x; \mathbb{B}_\infty) = 
\begin{cases}
0 & \text{if } \|x\|_\infty \leq 1,\\
\infty & \text{otherwise}.
\end{cases}
```

```@example
using LinearAlgebra
using ShiftedProximalOperators
using RegularizedProblems
using RegularizedOptimization

# First we create a type for our indicator function
struct SimpleIndicator end

# Then, we add a evaluation function
function (h::SimpleIndicator)(x)
  return norm(x, Inf) <= 1 ? 0.0 : Inf
end

# Now, we add a type that represents the function t -> h(shift + t)
mutable struct ShiftedSimpleIndicator{V} <: ShiftedProximableFunction
  shift::V
  temp::V
end

# Add the evaluation function for the shifted type
function (ψ::ShiftedSimpleIndicator)(x)
  @. ψ.temp = ψ.shift + x
  return norm(ψ.temp, Inf) <= 1 ? 0.0 : Inf
end

# Add the shifted function
function ShiftedProximalOperators.shifted(h::SimpleIndicator, xk::Vector{T}) where{T}
  return ShiftedSimpleIndicator(xk, similar(xk))
end

# Add the shift! function
function ShiftedProximalOperators.shift!(ψ::ShiftedSimpleIndicator{Vector{T}}, xk::Vector{T}) where{T}
  ψ.shift .= xk
end

# Add the prox! function for both the unshifted and shifted types, we can solve the proximal problem analytically by hand.
function ShiftedProximalOperators.prox!(
  y::AbstractVector{T},
  h::SimpleIndicator,
  q::AbstractVector{T},
  σ::T,
) where{T}
  for i ∈ eachindex(y)
    y[i] = min(max(q[i], -1), 1)
  end
end

function ShiftedProximalOperators.prox!(
  y::AbstractVector{T},
  ψ::ShiftedSimpleIndicator{Vector{T}},
  q::AbstractVector{T},
  σ::T,
) where{T}
  for i ∈ eachindex(y)
    y[i] = min(max(q[i], -1 - ψ.shift[i]), 1 - ψ.shift[i])
  end
end

# We can try to use this new regularizer with R2 : 

# define a small smooth problem
using ADNLPModels
f_model = ADNLPModel(x -> (x[1] - 4)^2, 3*ones(1), name = "Simple quadratic model")
h = SimpleIndicator()
regularized_nlp = RegularizedNLPModel(f_model, h)

out = R2(regularized_nlp, verbose = 1)
println("R2 converged after $(out.iter) iterations to the solution x = $(out.solution)")
```