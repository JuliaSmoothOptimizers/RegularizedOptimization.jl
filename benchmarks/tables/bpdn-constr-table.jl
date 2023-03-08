include("regulopt-tables.jl")

# model
Random.seed!(1234)
compound = 1
model, nls_model, sol = bpdn_model(compound, bounds = true)

f = LSR1Model(model)
λ = norm(grad(model, zeros(model.meta.nvar)), Inf) / 10
h = NormL1(λ)

verbose = 0 # 10
ϵ = 1.0e-6
maxIter = 500
maxIter_inner = 20
options =
  ROSolverOptions(ν = 1.0, ϵa = ϵ, ϵr = ϵ, verbose = verbose, maxIter = maxIter, spectral = true)
options2 = ROSolverOptions(spectral = false, psb = true, ϵa = ϵ, ϵr = ϵ, maxIter = maxIter_inner)
options3 = ROSolverOptions(spectral = false, psb = false, ϵa = ϵ, ϵr = ϵ, maxIter = maxIter_inner)
options4 = ROSolverOptions(spectral = true, ϵa = ϵ, ϵr = ϵ, maxIter = maxIter_inner)
options6 = ROSolverOptions(
  ν = 1.0,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = false,
)
options5 = ROSolverOptions(
  ν = 1.0,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = true,
)
options7 = ROSolverOptions(
  spectral = false,
  psb = true,
  ϵa = ϵ,
  ϵr = ϵ,
  maxIter = maxIter_inner,
  reduce_TR = false,
)

solvers = [:R2, :TRDH, :TRDH, :TRDH, :TR, :TR, :TR, :TR, :TR]
subsolvers = [:None, :None, :None, :None, :R2, :TRDH, :TRDH, :TRDH, :TRDH]
solver_options = [options, options, options5, options6, options, options, options, options, options]
subsolver_options =
  [options2, options2, options2, options2, options2, options7, options2, options3, options4] # n'importe lequel si subsolver = :None

benchmark_table(
  f,
  1:(f.meta.nvar),
  sol,
  h,
  λ,
  solvers,
  subsolvers,
  solver_options,
  subsolver_options,
  "BPDN-cstr",
)
