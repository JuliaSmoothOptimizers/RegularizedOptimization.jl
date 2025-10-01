# A regularized optimization problem

In this tutorial, we will show how to model and solve the nonconvex nonsmooth optimization problem
```math
  \min_{x \in \mathbb{R}^2} x_1^2 + 100(x_2 - x_1^2 - 2x_1)^2 + |x_1| + |x_2|.
```

## Modelling the problem
We first formulate the objective function as the sum of a smooth function $f$ and a nonsmooth regularizer $h$:
```math
  x_1^2 + 100(x_2 - x_1^2 - 2x_1)^2 + |x_1| + |x_2| = f(x_1, x_2) + h(x_1, x_2),
```
where 
```math
\begin{align*}
f(x_1, x_2) &:= x_1^2 + 100(x_2 - x_1^2 - 2x_1)^2,\\
h(x_1, x_2) &:= \|x\|_1.
\end{align*}
``` 
To model $f$, we are going to use [ADNLPModels.jl](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl).
For the nonsmooth regularizer, we observe that $h$ is actually readily available in [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl), you can refer to [this section](@ref regularizers) for a list of readily available regularizers.
We then wrap the smooth function and the regularizer in a `RegularizedNLPModel`

```@example
using ADNLPModels
using ProximalOperators
using RegularizedProblems

# Model the function
f_fun = x -> x[1]^2 + 100*(x[2] - x[1]^2 - 2*x[1])^2

# Choose a starting point for the optimization process, for the sake of this example, we choose
x0 = [-1.0, 2.0]

# Get an NLPModel corresponding to the smooth function f
f_model = ADNLPModel(f_fun, x0, name = "AD model of f") 

# Get the regularizer from ProximalOperators
h = NormL1(1.0)

# Wrap into a RegularizedNLPModel
regularized_pb = RegularizedNLPModel(f_model, h)
```

## Solving the problem
We can now choose one of the solvers presented [here](@ref algorithms) to solve the problem we defined above.
Please refer to other sections of this documentation to make the wisest choice for your particular problem.
Depending on the problem structure and on requirements from the user, some solvers are more appropriate than others.
The following tries to give a quick overview of what choices one can make.
```@example
using ADNLPModels
using ProximalOperators
using RegularizedProblems

f_fun = x -> x[1]^2 + 100*(x[2] - x[1]^2 - 2*x[1])^2
x0 = [-1.0, 2.0]

f_model = ADNLPModel(f_fun, x0, name = "AD model of f") 
h = NormL1(1.0)
regularized_pb = RegularizedNLPModel(f_model, h)

using RegularizedOptimization

# Suppose for example that we don't want to use a quasi-Newton approach
# and that we don't have access to the Hessian of f, or that we don't want to incur the cost of computing it
# In this case, the most appropriate solver would be R2.
# For this example, we also choose a relatively small tolerance by specifying the keyword argument atol across all solvers.
out = R2(regularized_pb, verbose = 10, atol = 1e-3)
println("R2 converged after $(out.iter) iterations to the solution x = $(out.solution)")
println("--------------------------------------------------------------------------------------")

# Now, on this example, we can actually use second information on f. 
# To do so, we are going to use TR, a trust-region solver that can exploit second order information.
out = TR(regularized_pb, verbose = 10, atol = 1e-3)
println("TR converged after $(out.iter) iterations to the solution x = $(out.solution)")
println("--------------------------------------------------------------------------------------")

# Suppose for some reason we can not compute the Hessian. 
# In this case, we can try to switch to a quasi-Newton approximation, this can be done with NLPModelsModifiers.jl
# We could choose to use TR again but for the sake of this tutorial we run it with R2N

using NLPModelsModifiers

# Switch the model of the smooth function to a quasi-Newton approximation
f_model_lsr1 = LSR1Model(f_model)
regularized_pb_lsr1 = RegularizedNLPModel(f_model_lsr1, h)

# Solve with R2N
out = R2N(regularized_pb_lsr1, verbose = 10, atol = 1e-3)
println("R2N converged after $(out.iter) iterations to the solution x = $(out.solution)")
println("--------------------------------------------------------------------------------------")

# Finally, TRDH and R2DH are specialized for diagonal quasi-Newton approximations,
# and should be used instead of TR and R2N, respectively.
f_model_sg = SpectralGradientModel(f_model)
regularized_pb_sg = RegularizedNLPModel(f_model_sg, h)

# Solve with R2DH
out = R2DH(regularized_pb_sg, verbose = 10, atol = 1e-3)
println("R2DH converged after $(out.iter) iterations to the solution x = $(out.solution)")
println("--------------------------------------------------------------------------------------")

```
