# A regularized nonlinear least-square problem

In this tutorial, we will show how to model and solve the nonconvex nonsmooth least-square problem
```math
  \min_{x \in \mathbb{R}^2} \tfrac{1}{2} \sum_{i=1}^m \big(y_i - x_1 e^{x_2 t_i}\big)^2 + \lambda \|x\|_0.
```
This problem models the fitting of an exponential curve, given noisy data.

## Modelling the problem
We first formulate the objective function as the sum of a smooth function $f$ and a nonsmooth regularizer $h$:
```math
  \tfrac{1}{2} \sum_{i=1}^m \big(y_i - x_1 e^{x_2 t_i}\big)^2 + \lambda \|x\|_0 = f(x) + h(x),
```
where 
```math
\begin{align*}
f(x) &:= \tfrac{1}{2} \sum_{i=1}^m \big(y_i - x_1 e^{x_2 t_i}\big)^2,\\
h(x) &:= \lambda\|x\|_0.
\end{align*}
```

To model $f$, we are going to use [ADNLPModels.jl](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl).
For the nonsmooth regularizer, we observe that $h$ is actually readily available in [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl), you can refer to [this section](@ref regularizers) for a list of readily available regularizers.
We then wrap the smooth function and the regularizer in a `RegularizedNLPModel`.

```@example ls
using ADNLPModels
using ProximalOperators
using Random
using RegularizedProblems

Random.seed!(0)

# Generate synthetic nonlinear least-squares data
m = 100
t = range(0, 1, length=m)
a_true, b_true = 2.0, -1.0
y = [a_true * exp(b_true * ti) + 0.1*randn() for ti in t]

# Starting point
x0 = [1.0, 0.0]   # [a, b]

# Define nonlinear residuals
function F(x)
  a, b = x
  return [yi - a*exp(b*ti) for (ti, yi) in zip(t, y)]
end

# Build ADNLSModel
f_model = ADNLSModel(F, x0, m, name = "nonlinear LS model of f")

# Get the regularizer from ProximalOperators
λ  = 1.0   
h = NormL0(λ)

# Wrap into a RegularizedNLPModel
regularized_pb = RegularizedNLPModel(f_model, h)
```

## Solving the problem
We can now choose one of the solvers presented [here](@ref algorithms) to solve the problem we defined above.
In the case of least-squares, it is usually more appropriate to choose LM or LMTR.

```@example ls
using RegularizedOptimization

# LM is a quadratic regularization method.
out = LM(regularized_pb, verbose = 1, atol = 1e-4)
println("LM converged after $(out.iter) iterations.")
```

```@example ls
#We can choose LMTR instead which is a trust-region method
out = LMTR(regularized_pb, verbose = 1, atol = 1e-4)
println("LMTR converged after $(out.iter) iterations.")

```