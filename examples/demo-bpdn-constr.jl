using Random
using LinearAlgebra
using ProximalOperators
using NLPModels, NLPModelsModifiers#, RegularizedProblems, RegularizedOptimization
include("/Users/joshuawolff/Documents/GERAD/src/RegularizedOptimization.jl/src/RegularizedOptimization.jl")
include("/Users/joshuawolff/Documents/GERAD/src/RegularizedProblems.jl/src/RegularizedProblems.jl")
using Printf

Random.seed!(1234)

function demo_solver(f, sol, h, χ, suffix = "l0-linf")
  options = ROSolverOptions(ν = 1.0, β = 1e16, ϵ = 1e-6, verbose = 10)
  @info " using TR to solve with" h χ
  reset!(f)
  TR_out = TR(f, h, χ, options, x0 = f.meta.x0)
end

function demo_bpdn_constr(compound = 1)
  model, sol = bpdn_constr_model(compound)
  f = LSR1Model(model)
  λ = norm(grad(model, zeros(model.meta.nvar)), Inf) / 10
  res0 = demo_solver(f, sol, NormL0(λ), NormLinf(1.0))
  res1 = demo_solver(f, sol, NormL1(λ), NormLinf(1.0), "l1-linf")
  return sol, res0, res1
end

bpdn_true, bpdn_l0, bpdn_l1 = demo_bpdn_constr()