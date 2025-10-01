# A regularized optimization problem

In this tutorial, we will show how to model and solve the nonconvex nonsmooth optimization problem
```math
  \min_{x \in \mathbb{R}^2} (1 - x_1)^2 + 100(x_2 - x_1^2)^2 + |x_1| + |x_2|,
```
which can be seen as a $$\ell_1$$ regularization of the Rosenbrock function. 
It can be shown that the solution to the problem is 
```math
  x^* = \begin{pmatrix}
  0.25\\
  0.0575
  \end{pmatrix}
```


## Modelling the problem
We first formulate the objective function as the sum of a smooth function $f$ and a nonsmooth regularizer $h$:
```math
  (1 - x_1)^2 + 100(x_2 - x_1^2)^2 + |x_1| + |x_2| = f(x_1, x_2) + h(x_1, x_2),
```
where 
```math
\begin{align*}
f(x_1, x_2) &:= (1 - x_1)^2 + 100(x_2 - x_1^2)^2,\\
h(x_1, x_2) &:= \|x\|_1.
\end{align*}
``` 
To model $f$, we are going to use [ADNLPModels.jl](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl).
For the nonsmooth regularizer, we use [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl). 
We then wrap the smooth function and the regularizer in a `RegularizedNLPModel`

```@example basic
using ADNLPModels
using ProximalOperators
using RegularizedProblems

# Model the function
f_fun = x -> (1 - x[1])^2 + 100*(x[2] - x[1]^2)^2

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

Suppose for example that we don't want to use a quasi-Newton approach and that we don't have access to the Hessian of f, or that we don't want to incur the cost of computing it. 
In this case, the most appropriate solver would be R2.
For this example, we also choose a relatively small tolerance by specifying the keyword arguments `atol` and `rtol` across all solvers.

```@example basic
using RegularizedOptimization
 
out = R2(regularized_pb, verbose = 10, atol = 1e-3, rtol = 1e-3)
println("R2 converged after $(out.iter) iterations to the solution x = $(out.solution)")
``` 

Now, we can actually use second information on f. 
To do so, we are going to use TR, a trust-region solver that can exploit second order information.
```@example basic

out = TR(regularized_pb, verbose = 10, atol = 1e-3, rtol = 1e-3)
println("TR converged after $(out.iter) iterations to the solution x = $(out.solution)")
```

Suppose for some reason we can not compute the Hessian. 
In this case, we can try to switch to a quasi-Newton approximation, this can be done with NLPModelsModifiers.jl
We could choose to use TR again but for the sake of this tutorial we run it with R2N
```@example basic
using NLPModelsModifiers

# Switch the model of the smooth function to a quasi-Newton approximation
f_model_lsr1 = LSR1Model(f_model)
regularized_pb_lsr1 = RegularizedNLPModel(f_model_lsr1, h)

# Solve with R2N
out = R2N(regularized_pb_lsr1, verbose = 10, atol = 1e-3, rtol = 1e-3)
println("R2N converged after $(out.iter) iterations to the solution x = $(out.solution)")
```

Finally, TRDH and R2DH are specialized for diagonal quasi-Newton approximations, and should be used instead of TR and R2N, respectively.
```@example basic

f_model_sg = SpectralGradientModel(f_model)
regularized_pb_sg = RegularizedNLPModel(f_model_sg, h)

# Solve with R2DH
out = R2DH(regularized_pb_sg, verbose = 10, atol = 1e-3, rtol = 1e-3)
println("R2DH converged after $(out.iter) iterations to the solution x = $(out.solution)")
```
