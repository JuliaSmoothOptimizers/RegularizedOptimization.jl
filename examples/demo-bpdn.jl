using Random
using LinearAlgebra
using ProximalOperators
using NLPModels, NLPModelsModifiers, RegularizedOptimization, RegularizedProblems

include("plot-utils-bpdn.jl")

Random.seed!(1234)

function demo_solver(model, nls_model, sol, h, χ, suffix = "l0-linf")
  options = ROSolverOptions(ν = 1.0, β = 1e16, ϵ = 1e-6, verbose = 10)

  @info "using R2 to solve with" h
  reset!(model)
  R2_out = R2(model, h, options, x0 = model.meta.x0)
  @info "R2 relative error" norm(R2_out.solution - sol) / norm(sol)
  # plot_bpdn(R2_out, sol, "r2-$(suffix)")

  @info " using TR to solve with" h χ
  reset!(model)
  TR_out = TR(model, h, χ, options, x0 = model.meta.x0)
  @info "TR relative error" norm(TR_out.solution - sol) / norm(sol)
  # plot_bpdn(TR_out, sol, "tr-r2-$(suffix)")

  @info "using LM to solve with" h
  LM_out = LM(nls_model, h, options, x0 = nls_model.meta.x0)
  @info "LM relative error" norm(LM_out.solution - sol) / norm(sol)
  # plot_bpdn(LM_out, sol, "lm-$(suffix)")

  @info "using LMTR to solve with" h
  reset!(nls_model)
  LMTR_out = LMTR(nls_model, h, χ, options, x0 = nls_model.meta.x0)
  @info "LMTR relative error" norm(LMTR_out.solution - sol) / norm(sol)
  # plot_bpdn(LMTR_out, sol, "lmtr-$(suffix)")

  plot_bpdn((R2_out, TR_out, LM_out, LMTR_out), sol, ("$(alg)-$(suffix)" for alg ∈ ("r2", "tr", "lm", "lmtr")))
end

function demo_bpdn(compound = 1)
  model, nls_model, sol = bpdn_model(compound)
  lsr1model = LSR1Model(model)
  λ = norm(grad(model, zeros(model.meta.nvar)), Inf) / 10
  r = 10 * compound

  demo_solver(lsr1model, nls_model, sol, NormL0(λ), NormLinf(1.0))
  # demo_solver(lsr1model, nls_model, sol, IndBallL0(r), NormLinf(1.0), "b0-linf")
  # demo_solver(lsr1model, nls_model, sol, NormL1(λ), NormL2(1.0), "l1-l2")
  # demo_solver(lsr1model, nls_model, sol, NormL1(λ), NormLinf(1.0), "l1-linf")
end

demo_bpdn()
