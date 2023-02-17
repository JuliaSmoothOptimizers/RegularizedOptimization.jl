include("regulopt-tables.jl")

# model
Random.seed!(1234)
compound = 1
model, nls_model, sol = bpdn_model(compound, bounds = false)

# parameters
f = LSR1Model(model)
λ = norm(grad(model, zeros(model.meta.nvar)), Inf) / 10
h = NormL1(λ)

verbose = 0 # 10
options = ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = verbose, maxIter = 500, spectral = true)
options2 = ROSolverOptions(spectral = false, psb = true, ϵa = 1e-6, ϵr = 1e-6, maxIter=40)
options3 = ROSolverOptions(spectral = false, psb = false, ϵa = 1e-6, ϵr = 1e-6, maxIter=40)
options4 = ROSolverOptions(spectral = true, ϵa = 1e-6, ϵr = 1e-6, maxIter=40)
options5 = ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = verbose, maxIter = 500, spectral = false, psb = true)
options6 = ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = verbose, maxIter = 500, spectral = false, psb = false)
options7 = ROSolverOptions(spectral = false, psb = true, ϵa = 1e-6, ϵr = 1e-6, maxIter=40, reduce_TR = false)
options8 = ROSolverOptions(spectral = true, ϵa = 1e-6, ϵr = 1e-6, maxIter=40, reduce_TR = false)

solvers = [:R2, :TRDH, :TRDH, :TRDH, :TR, :TR, :TR, :TR, :TR, :TR]
subsolvers = [:None, :None, :None, :None, :R2, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH]
solver_options = [options, options, options5, options6, options, options, options, options, options, options]
subsolver_options = [options2, options2, options2, options2, options2, options7, options2, options3, options4, options8] # n'importe lequel si subsolver = :None

benchmark_table(f, 1:f.meta.nvar, sol, h, λ, solvers, subsolvers, solver_options, subsolver_options, "BPDN")