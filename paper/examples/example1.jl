using LinearAlgebra, Random
using ProximalOperators
using NLPModels, NLPModelsModifiers, RegularizedProblems, RegularizedOptimization
using MLDatasets

random_seed = 1234
Random.seed!(random_seed)

# Load MNIST from MLDatasets
imgs, labels = MLDatasets.MNIST.traindata()

# Use RegularizedProblems' preprocessing
A, b = RegularizedProblems.generate_data(imgs, labels, (1, 7), false)

# Build the models
model, _, _ = RegularizedProblems.svm_model(A, b)

# Define the Hessian approximation
f = LBFGSModel(model)

# Define the nonsmooth regularizer (L0 norm)
λ = 1.0e-1
h = NormL0(λ)

# Define the regularized NLP model
reg_nlp = RegularizedNLPModel(f, h)

# Choose a solver (R2DH) and execution statistics tracker
solver_r2dh= R2DHSolver(reg_nlp)
stats = RegularizedExecutionStats(reg_nlp)

# Solve the problem 
solve!(solver_r2dh, reg_nlp, stats, x = f.meta.x0, σk = 1e-6, atol = 2e-5, rtol = 2e-5, verbose = 1)

@test stats.status == :first_order
