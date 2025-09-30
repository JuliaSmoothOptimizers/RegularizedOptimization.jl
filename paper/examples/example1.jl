using LinearAlgebra, Random, ProximalOperators
using NLPModels, RegularizedProblems, RegularizedOptimization
using MLDatasets

Random.seed!(1234)
model, nls_model, _ = RegularizedProblems.svm_train_model()  # Build SVM model
f = LSR1Model(model)                                         # L-SR1 Hessian approximation
λ = 1.0                                                      # Regularization parameter
h = RootNormLhalf(1.0)                                       # Nonsmooth term
reg_nlp = RegularizedNLPModel(f, h)                          # Regularized problem
solver = R2NSolver(reg_nlp)                                  # Choose solver
stats  = RegularizedExecutionStats(reg_nlp)
solve!(solver, reg_nlp, stats; atol=1e-4, rtol=1e-4, verbose=0, sub_kwargs=(max_iter=200,))
