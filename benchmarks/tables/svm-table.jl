include("regulopt-tables.jl")
using MLDatasets

Random.seed!(1234)
nlp_train, nls_train, sol_train  = RegularizedProblems.svm_train_model()
nlp_test, nls_test, sol_test = RegularizedProblems.svm_test_model()
f = LSR1Model(nlp_train)
f_test = LSR1Model(nlp_test)
λ = 10.0 #norm(grad(model, rand(model.meta.nvar)), Inf) / 10
h = NormL1(λ)

ν = 1.0e0
verbose = 0 #10
options = ROSolverOptions(ν = ν, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = verbose, maxIter = 500, spectral = true)
options2 = ROSolverOptions(spectral = false, psb = true, ϵa = 1e-6, ϵr = 1e-6, maxIter=40)
options3 = ROSolverOptions(spectral = false, psb = false, ϵa = 1e-6, ϵr = 1e-6, maxIter=40)
options4 = ROSolverOptions(spectral = true, ϵa = 1e-6, ϵr = 1e-6, maxIter=40)
options5 = ROSolverOptions(ν = ν, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = verbose, maxIter = 500, spectral = false, psb = true)
options6 = ROSolverOptions(ν = ν, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = verbose, maxIter = 500, spectral = false, psb = false)
options7 = ROSolverOptions(spectral = false, psb = true, reduce_TR = false, ϵa = 1e-6, ϵr = 1e-6, maxIter=40)

solvers = [:R2, :TRDH, :TRDH, :TRDH, :TR, :TR, :TR, :TR, :TR]
subsolvers = [:None, :None, :None, :None, :R2, :TRDH, :TRDH, :TRDH, :TRDH]
solver_options = [options, options, options5, options6, options, options, options, options, options]
subsolver_options = [options2, options2, options2, options2, options2, options2, options7, options3, options4]
subset = 1:9

benchmark_table(f, 1:f.meta.nvar, [], h, λ, solvers[subset], subsolvers[subset], solver_options[subset], subsolver_options[subset],
                "SVM with ν = $ν, λ = $λ")