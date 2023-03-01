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
ϵ = 1.0e-6
maxIter = 500
maxIter_inner = 40
options = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = verbose, maxIter = maxIter, spectral = true)
options2 = ROSolverOptions(spectral = false, psb = true, ϵa = ϵ, ϵr = ϵ, maxIter=maxIter_inner)
options3 = ROSolverOptions(spectral = false, psb = false, ϵa = ϵ, ϵr = ϵ, maxIter=maxIter_inner)
options4 = ROSolverOptions(spectral = true, ϵa = ϵ, ϵr = ϵ, maxIter=maxIter_inner)
options5 = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = verbose, maxIter = maxIter, spectral = false, psb = true)
options6 = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = verbose, maxIter = maxIter, spectral = false, psb = false)
options7 = ROSolverOptions(spectral = false, psb = true, reduce_TR = false, ϵa = ϵ, ϵr = ϵ, maxIter=maxIter_inner)

solvers = [:R2, :TRDH, :TRDH, :TRDH, :TR, :TR, :TR, :TR, :TR]
subsolvers = [:None, :None, :None, :None, :R2, :TRDH, :TRDH, :TRDH, :TRDH]
solver_options = [options, options, options5, options6, options, options, options, options, options]
subsolver_options = [options2, options2, options2, options2, options2, options2, options7, options3, options4]
subset = 1:9

benchmark_table(f, 1:f.meta.nvar, [], h, λ, solvers[subset], subsolvers[subset], solver_options[subset], subsolver_options[subset],
                "SVM with ν = $ν, λ = $λ")