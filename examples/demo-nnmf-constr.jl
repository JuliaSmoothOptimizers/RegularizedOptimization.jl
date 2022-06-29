using Random
using LinearAlgebra
using ProximalOperators
using NLPModels, NLPModelsModifiers, RegularizedProblems, RegularizedOptimization
using Printf

Random.seed!(1234)

function demo_solver(f, h, χ, suffix = "l0-linf")
  options = ROSolverOptions(ν = 1.0, β = 1e16, ϵ = 1e-6, verbose = 10)
  @info " using TR to solve with" h χ
  reset!(f)
  TR_out = TR_constr(f, h, χ, options)
  TR_out
end

function demo_nnmf()
  model = nnmf_model(1000,1000,100)
  f = LSR1Model(model) # don't know if it is right
  λ = norm(grad(model, rand(model.meta.nvar)), Inf) / 1000000 # don't know if is right
  res0 = demo_solver(f, NormL0(λ), NormLinf(1.0))
  res1 = demo_solver(f, NormL1(λ), NormLinf(1.0), "l1-linf")
  return res0, res1
end

demo_nnmf()