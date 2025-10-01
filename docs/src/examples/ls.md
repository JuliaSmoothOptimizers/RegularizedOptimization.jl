# A regularized least-square problem

In this tutorial, we will show how to model and solve the nonconvex nonsmooth least-square problem
```math
  \min_{x \in \mathbb{R}^n} \frac{1}{2} \|Ax - b\|_2^2 + \lambda \|x\|_0.
```

## Modelling the problem
We first formulate the objective function as the sum of a smooth function $f$ and a nonsmooth regularizer $h$:
```math
  \frac{1}{2} \|Ax - b\|_2^2 + \lambda \|x\|_0 = f(x) + h(x),
```
where 
```math
\begin{align*}
f(x) &:= \frac{1}{2} \|Ax - b\|_2^2,\\
h(x) &:= \lambda\|x\|_0.
\end{align*}
```

To model $f$, we are going to use [LLSModels.jl](https://github.com/JuliaSmoothOptimizers/LLSModels.jl).
For the nonsmooth regularizer, we observe that $h$ is actually readily available in [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl), you can refer to [this section](@ref regularizers) for a list of readily available regularizers.
We then wrap the smooth function and the regularizer in a `RegularizedNLPModel`.

```@example
using LLSModels
using ProximalOperators
using Random
using RegularizedProblems

Random.seed!(0)

# Generate A, b
m, n = 5, 10
A = randn((m, n))
b = randn(m)

# Choose a starting point for the optimization process
x0 = randn(n)

# Get an NLSModel corresponding to the smooth function f
f_model = LLSModel(A, b, x0 = x0, name = "NLS model of f") 

# Get the regularizer from ProximalOperators
λ  = 1.0   
h = NormL0(λ)

# Wrap into a RegularizedNLPModel
regularized_pb = RegularizedNLPModel(f_model, h)
```

## Solving the problem
We can now choose one of the solvers presented [here](@ref algorithms) to solve the problem we defined above.
In the case of least-squares, it is usually more appropriate to choose LM or LMTR.
```@example
using LLSModels
using ProximalOperators
using Random
using RegularizedProblems

Random.seed!(0)

m, n = 5, 10
λ  = 0.1
A = randn((m, n))
b = randn(m)

x0 = 10*randn(n)

f_model = LLSModel(A, b, x0 = x0, name = "NLS model of f") 
h = NormL0(λ)
regularized_pb = RegularizedNLPModel(f_model, h)

using RegularizedOptimization

# LM is a quadratic regularization method, we specify the verbosity and the tolerance of the solver
out = LM(regularized_pb, verbose = 1, atol = 1e-3)
println("LM converged after $(out.iter) iterations.")
println("--------------------------------------------------------------------------------------")

# We can choose LMTR instead which is a trust-region method
out = LMTR(regularized_pb, verbose = 1, atol = 1e-3)
println("LMTR converged after $(out.iter) iterations.")

```