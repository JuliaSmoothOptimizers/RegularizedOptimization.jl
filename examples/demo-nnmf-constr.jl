using Random
using LinearAlgebra
using ProximalOperators
using NLPModels, NLPModelsModifiers, RegularizedProblems, RegularizedOptimization
using Printf

include("plot-utils-nnmf.jl")

Random.seed!(1234)

function demo_solver(f, h, χ, selected, Avec, m, n, k, suffix = "l0-linf")
  options = ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10, maxIter = 500)
  @info " using TR to solve with" h χ
  reset!(f)
  TR_out = TR(f, h, χ, options, selected = selected)
  plot_nnmf(TR_out, Avec, m, n, k, "tr-r2-$suffix")
end

function demo_nnmf()
  m, n, k = 100, 50, 5
  model, A, selected = nnmf_model(m, n, k)
  f = LSR1Model(model)
  λ = norm(grad(model, rand(model.meta.nvar)), Inf) / 200
  demo_solver(f, NormL0(λ), NormLinf(1.0), selected, A, m, n, k, "l0-linf")
  λ = norm(grad(model, rand(model.meta.nvar)), Inf) / 10000
  demo_solver(f, NormL1(λ), NormLinf(1.0), selected, A, m, n, k, "l1-linf")
end

demo_nnmf()
