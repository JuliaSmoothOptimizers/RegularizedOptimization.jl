using DifferentialEquations, ProximalOperators
using ADNLPModels, NLPModels, NLPModelsModifiers, RegularizedProblems, RegularizedOptimization
using Printf

function demo_fh()
  data, simulate, resid, misfit = RegularizedProblems.FH_smooth_term()
  model = ADNLPModel(misfit, ones(5))
  χ = NormLinf(1.0)
  options = ROSolverOptions(verbose = 10, ϵ = 1e-6, β = 1e16, ν = 1.0e+2)
  lbfgs_model = LBFGSModel(model)
  h = NormL0(1.0)
  @info " using TR to solve with" h χ
  res0 = TR(lbfgs_model, h, χ, options)
  h = NormL1(1.0)
  @info " using TR to solve with" h χ
  res1 = TR(lbfgs_model, h, χ, options)
  return res0, res1
end

fh_l0, fh_l1 = demo_fh()