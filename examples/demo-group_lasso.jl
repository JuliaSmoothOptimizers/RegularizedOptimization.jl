using Random
using LinearAlgebra
using ProximalOperators, ShiftedProximalOperators
using NLPModels, NLPModelsModifiers, RegularizedOptimization, RegularizedProblems

include("plot-utils-bpdn.jl")

Random.seed!(1234)

function demo_solver(f, sol, h, χ, suffix = "l0-linf")
  options = ROSolverOptions(ν = 1.0, β = 1e16, ϵ = 1e-6, verbose = 10)

  @info "using LM to solve with" h
  reset!(f)
  LM_out = LM(f, h, options, x0 = f.meta.x0)
  @info "LM relative error" norm(LM_out.solution - sol) / norm(sol)
  plot_bpdn(LM_out, sol, "lm-$(suffix)")

  @info " using LM-TR to solve with" h χ
  reset!(f)
  LMTR_out = LMTR(f, h, χ, options, x0 = f.meta.x0)
  @info "LM-TR relative error" norm(LMTR_out.solution - sol) / norm(sol)
  plot_bpdn(LMTR_out, sol, "lmtr-r2-$(suffix)")
end

function demo_group(compound = 1)
  f, sol, g, active_groups, idx = group_lasso_nls_model(compound)
  # f = LSR1Model(model)
  idx = [idx[i,:] for i = 1:g]
  λ = .1*ones(g,) #norm(grad(model, zeros(model.meta.nvar)), Inf) / 1 * ones(g,)
  # λ[active_groups] .= .01
  demo_solver(f, sol, GroupNormL2(λ, idx), NormLinf(1.0), "g-l2-linf")

  # k = 10 * compound
  # demo_solver(f, sol, IndBallL0(10 * compound), NormLinf(1.0), "g-l2-linf")

end

demo_group()
