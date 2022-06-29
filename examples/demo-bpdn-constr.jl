using Random
using LinearAlgebra
using ProximalOperators
using NLPModels, NLPModelsModifiers, RegularizedProblems, RegularizedOptimization
using Printf

Random.seed!(1234)

function demo_solver(f, sol, h, χ, suffix = "l0-linf")
  options = ROSolverOptions(ν = 1.0, β = 1e16, ϵ = 1e-6, verbose = 10)
  @info " using TR to solve with" h χ
  reset!(f)
  TR_out = TR_constr(f, h, χ, options)
end

function demo_bpdn(compound = 1)
  model, model2, sol = bpdn_model(compound)
  f = LSR1Model(model)
  λ = norm(grad(model, zeros(model.meta.nvar)), Inf) / 10
  res0 = demo_solver(f, sol, NormL0(λ), NormLinf(1.0))
  res1 = demo_solver(f, sol, NormL1(λ), NormLinf(1.0), "l1-linf")
  return sol, res0, res1
end

bpdn_true, bpdn_l0, bpdn_l1 = demo_bpdn()