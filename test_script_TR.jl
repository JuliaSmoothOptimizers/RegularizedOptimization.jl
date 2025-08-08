using LinearAlgebra
using ProximalOperators
using SolverCore,NLPModels, SciMLSensitivity, NLPModelsModifiers, RegularizedProblems, RegularizedOptimization
using DifferentialEquations, ADNLPModels
using Random

Random.seed!(1234)
import RegularizedOptimization.solve!
# Define the Fitzhugh-Nagumo problem
model, model_nls, _ = RegularizedProblems.fh_model(backend = :optimized)
# Define the Hessian approximation


# Define the nonsmooth regularizer (L1 norm)
λ = 0.1
h = NormL1(λ)

# Define the regularized NLP model
reg_nlp = RegularizedNLPModel(model_nls, h)

# Choose a solver (TR) and execution statistics tracker
solver_tr = LMSolver(reg_nlp)
stats = RegularizedExecutionStats(reg_nlp)

# Solve the problem

solve!(solver_tr, reg_nlp, stats, x = model_nls.meta.x0, max_time = 10.0, atol = 1e-6, rtol = 1e-6, verbose = 1, σk = 5e2, σmin = 1e2)
