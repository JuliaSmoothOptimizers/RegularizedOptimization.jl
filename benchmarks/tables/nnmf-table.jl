include("regulopt-tables.jl")

Random.seed!(1234)
m, n, k = 100, 50, 5
model, A, selected = nnmf_model(m, n, k)
f = LSR1Model(model)
λ = norm(grad(model, rand(model.meta.nvar)), Inf) / 1000
h = NormL0(λ)
ν = 1.0
verbose = 0 #10
options = ROSolverOptions(ν = ν, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = verbose, maxIter = 500, spectral = true)
options2 = ROSolverOptions(spectral = false, psb = true, ϵa = 1e-6, ϵr = 1e-6, maxIter=40)
options3 = ROSolverOptions(spectral = false, psb = false, ϵa = 1e-6, ϵr = 1e-6, maxIter=40)
options4 = ROSolverOptions(spectral = true, ϵa = 1e-6, ϵr = 1e-6, maxIter=40)
options5 = ROSolverOptions(ν = ν, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = verbose, maxIter = 500, spectral = false, psb = true)
options6 = ROSolverOptions(ν = ν, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = verbose, maxIter = 500, spectral = false, psb = false)
options7 = ROSolverOptions(spectral = false, psb = true, reduce_TR = false, ϵa = 1e-6, ϵr = 1e-6, maxIter=40)
options8 = ROSolverOptions(ν = ν, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = verbose, maxIter = 500, spectral = false, psb = false, reduce_TR = false)

solvers = [:R2, :TRDH, :TRDH, :TRDH, :TRDH, :TR, :TR, :TR, :TR, :TR, :TR]
subsolvers = [:None, :None, :None, :None, :None, :R2, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH]
solver_options = [options, options, options5, options6, options8, options, options, options, options, options]
subsolver_options = [options2, options2, options2, options2, options2, options2, options7, options2, options3, options4]

benchmark_table(f, selected, [], h, λ, solvers, subsolvers, solver_options, subsolver_options,
                "NNMF with m = $m, n = $n, k = $k, ν = $ν, λ = $λ")