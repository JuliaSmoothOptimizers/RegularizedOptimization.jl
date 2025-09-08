## After merging the PRs on TR

using LinearAlgebra
using DifferentialEquations, ProximalOperators
using ADNLPModels, NLPModels, NLPModelsModifiers, RegularizedOptimization, RegularizedProblems

# Define the Fitzhugh-Nagumo problem
model, _, _ = RegularizedProblems.fh_model()

# Define the Hessian approximation
f = LBFGSModel(fh_model)

# Define the nonsmooth regularizer (L1 norm)
λ = 0.1
h = NormL1(λ)

# Define the regularized NLP model
reg_nlp = RegularizedNLPModel(f, h)

# Choose a solver (TR) and execution statistics tracker
solver_tr = TRSolver(reg_nlp)
stats = RegularizedExecutionStats(reg_nlp)

# Solve the problem
solve!(solver_tr, reg_nlp, stats, x = f.meta.x0, atol = 1e-3, rtol = 1e-4, verbose = 10, ν = 1.0e+2)

@test stats.status == :first_order
