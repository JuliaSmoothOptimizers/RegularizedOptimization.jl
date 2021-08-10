using DifferentialEquations, ProximalOperators
using ADNLPModels, NLPModels, NLPModelsModifiers, RegularizedProblems, TRNC

include("plot-utils-fh.jl")

function demo_fh()
  data, simulate, resid, misfit = RegularizedProblems.FH_smooth_term()
  model = ADNLPModel(misfit, ones(5))

  h = NormL0(1.0)
  χ = NormLinf(1.0)
  options = TRNCoptions(; verbose = 10, ϵ = 1e-6, β = 1e16, ν = 1.0e+2)

  lbfgs_model = LBFGSModel(model)
  x, Ghist, Fhist, Hhist, Comp_pg = TR(lbfgs_model, h, χ, options)
  plot_fh(Comp_pg[2,:], Fhist+Hhist, simulate(x), data, "tr-r2")

  nls_model = ADNLSModel(resid, ones(5), 202)
  x, k, Fhist, Hhist, Comp_pg = LMTR(nls_model, h, χ, options)
  plot_fh(Comp_pg, Fhist+Hhist, simulate(x), data, "lmtr-r2")

  reset!(nls_model)
  x, k, Fhist, Hhist, Comp_pg = LM(nls_model, h, options)
  plot_fh(Comp_pg, Fhist+Hhist, simulate(x), data, "lm-r2")
end

demo_fh()

