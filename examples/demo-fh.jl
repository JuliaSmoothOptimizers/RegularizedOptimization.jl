using DifferentialEquations, ProximalOperators
using ADNLPModels, NLPModels, NLPModelsModifiers, RegularizedOptimization, RegularizedProblems

include("plot-utils-fh.jl")

function demo_fh()
  data, simulate, resid, misfit = RegularizedProblems.FH_smooth_term()
  model = ADNLPModel(misfit, ones(5))
  h = NormL0(1.0)
  χ = NormLinf(1.0)
  options = ROSolverOptions(; verbose = 10, ϵ = 1e-6, β = 1e16, ν = 1.0e+2)

  lbfgs_model = LBFGSModel(model)
  TR_out = TR(lbfgs_model, h, χ, options)
  plot_fh(TR_out, simulate(TR_out.solution), data, "tr-r2")

  nls_model = ADNLSModel(resid, ones(5), 202)
  LMTR_out = LMTR(nls_model, h, χ, options)
  plot_fh(LMTR_out, simulate(LMTR_out.solution), data, "lmtr-r2")

  reset!(nls_model)
  LM_out = LM(nls_model, h, options)
  plot_fh(LM_out, simulate(LM_out.solution), data, "lm-r2")
end

demo_fh()
