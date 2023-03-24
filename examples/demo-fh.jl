using DifferentialEquations, ProximalOperators
using ADNLPModels, NLPModels, NLPModelsModifiers, RegularizedOptimization, RegularizedProblems

include("plot-utils-fh.jl")

function demo_fh()
  data, simulate, resid, misfit, x0 = RegularizedProblems.FH_smooth_term()
  model = ADNLPModel(misfit, ones(5))
  h = NormL0(1.0)
  χ = NormLinf(1.0)
  # relax tolerances to speed up demo
  options = ROSolverOptions(; verbose = 10, ϵa = 1e-3, ϵr = 1e-3, β = 1e16, ν = 1.0e+2)

  lbfgs_model = LBFGSModel(model)
  TR_out = TR(lbfgs_model, h, χ, options)
  @info "TR relative error" norm(TR_out.solution - x0) / norm(x0)
  plot_fh(TR_out, simulate(TR_out.solution), data, "tr-r2")

  nls_model = ADNLSModel(resid, ones(5), 202)
  options.σmin = 1e-6
  LMTR_out = LMTR(nls_model, h, χ, options)
  @info "LMTR relative error" norm(LMTR_out.solution - x0) / norm(x0)
  plot_fh(LMTR_out, simulate(LMTR_out.solution), data, "lmtr-r2")

  reset!(nls_model)
  options.σmin = 1e+3
  LM_out = LM(nls_model, h, options)
  @info "LM relative error" norm(LM_out.solution - x0) / norm(x0)
  plot_fh(LM_out, simulate(LM_out.solution), data, "lm-r2")
end

demo_fh()
