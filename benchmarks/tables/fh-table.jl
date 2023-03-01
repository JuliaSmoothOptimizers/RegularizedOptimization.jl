include("regulopt-tables.jl")
using ADNLPModels, DifferentialEquations

Random.seed!(1234)
data, simulate, resid, misfit = RegularizedProblems.FH_smooth_term()
model = ADNLPModel(misfit, ones(5))
f = LBFGSModel(model)

λ = 1.0
h = NormL0(λ)
ν = 1.0e2
verbose = 0 #10
maxIter = 1000
maxIter_sub = 200 # max iter for subsolver
ϵ = 1.0e-6
options = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = verbose, maxIter = maxIter, spectral = true)
options2 = ROSolverOptions(spectral = false, psb = true, ϵa = ϵ, ϵr = ϵ, maxIter = maxIter_sub)
options3 = ROSolverOptions(spectral = false, psb = false, ϵa = ϵ, ϵr = ϵ, maxIter = maxIter_sub)
options4 = ROSolverOptions(spectral = true, ϵa = ϵ, ϵr = ϵ, maxIter = maxIter_sub)
options5 = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = verbose, maxIter = maxIter, spectral = false, psb = true)
options6 = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = verbose, maxIter = maxIter, spectral = false, psb = false)
options7 = ROSolverOptions(spectral = false, psb = true, reduce_TR = false, ϵa = ϵ, ϵr = ϵ, maxIter = maxIter_sub)
options8 = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = verbose, maxIter = maxIter, spectral = false, psb = false, reduce_TR = false)

solvers = [:R2, :TRDH, :TRDH, :TRDH, :TRDH, :TR, :TR, :TR, :TR, :TR]
subsolvers = [:None, :None, :None, :None, :None, :R2, :TRDH, :TRDH, :TRDH, :TRDH]
solver_options = [options, options, options5, options6, options8, options, options, options, options, options]
subsolver_options = [options2, options2, options2, options2, options2, options2, options7, options2, options3, options4]
subset = 2:10 # issues with R2 alone

benchmark_table(f, 1:f.meta.nvar, [], h, λ, solvers[subset], subsolvers[subset], solver_options[subset], subsolver_options[subset],
                "FH with ν = $ν, λ = $λ")
