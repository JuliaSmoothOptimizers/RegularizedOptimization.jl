using Random
using LinearAlgebra
using ProximalOperators
using NLPModels, NLPModelsModifiers, RegularizedProblems, RegularizedOptimization
using Printf

include("plot-utils-bpdn.jl")

Random.seed!(1234)

function demo_solver(f, nls, sol, h, χ, suffix = "l0-linf")
  options = ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10)

  @info " using TR to solve with" h χ
  reset!(f)
  TR_out = TR(f, h, χ, options, x0 = f.meta.x0)
  @info "TR relative error" norm(TR_out.solution - sol) / norm(sol)
  plot_bpdn(TR_out, sol, "constr-tr-r2-$(suffix)")

  @info " using R2 to solve with" h
  reset!(f)
  R2_out = R2(f, h, options, x0 = f.meta.x0)
  @info "R2 relative error" norm(R2_out.solution - sol) / norm(sol)
  plot_bpdn(R2_out, sol, "constr-r2-$(suffix)")

  @info " using LMTR to solve with" h χ
  reset!(nls)
  LMTR_out = LMTR(nls, h, χ, options, x0 = f.meta.x0)
  @info "LMTR relative error" norm(LMTR_out.solution - sol) / norm(sol)
  plot_bpdn(LMTR_out, sol, "constr-lmtr-r2-$(suffix)")

  @info " using LM to solve with" h
  reset!(nls)
  LM_out = LM(nls, h, options, x0 = f.meta.x0)
  @info "LM relative error" norm(LM_out.solution - sol) / norm(sol)
  plot_bpdn(LM_out, sol, "constr-lm-r2-$(suffix)")
end

function demo_bpdn_constr(compound = 1)
  model, nls_model, sol = bpdn_model(compound, bounds = true)
  f = LSR1Model(model)
  λ = norm(grad(model, zeros(model.meta.nvar)), Inf) / 10
  demo_solver(f, nls_model, sol, NormL0(λ), NormLinf(1.0))
  λ = norm(grad(model, zeros(model.meta.nvar)), Inf) / 3
  demo_solver(f, nls_model, sol, NormL1(λ), NormLinf(1.0), "l1-linf")
end

demo_bpdn_constr()
