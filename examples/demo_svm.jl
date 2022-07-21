using Random
using LinearAlgebra
using ProximalOperators, ShiftedProximalOperators, MLDatasets
using ADNLPModels, NLPModels, NLPModelsModifiers #ReverseADNLSModels
using RegularizedOptimization, RegularizedProblems

include("plot-utils-svm.jl")

Random.seed!(1234)

function demo_solver(nlp_tr, nls_tr, sol_tr, nlp_test, nls_test, sol_test, h, χ, suffix = "l0-linf")
  options = ROSolverOptions(ν = 1.0, β = 1e16, ϵ = 1e-4, verbose = 10, σmin = 1e-5);

  @info "using R2 to solve with" h
  reset!(nlp_tr)
  out = R2(nlp_tr, h, options, x0 = nlp_tr.meta.x0)
  @info "R2 relative training error" obj(nlp_tr, out.solution)
  @info "R2 relative testing error" obj(nlp_test, out.solution)
  @info "obj hist" out.solver_specific[:Fhist] + out.solver_specific[:Hhist]
  # plot_svm(out, sol, "r2-$(suffix)") ### include images?

  reset!(nlp_tr)
  @info "using TR to solve with" h χ
  out = TR(nlp_tr, h, χ, options, x0 = nlp_tr.meta.x0)

  @info " using TR to solve with" h χ
  reset!(f)
  nls_model = ADNLSModel(nlp_tr, )
  nls_model_t = ADNLSModel(resid_test, )
  TR_out = LMTR(nlp_tr, h, χ, options, x0 = f.meta.x0)
  # @info "TR relative error" norm(TR_out.solution - sol) / norm(sol)
  # plot_sv,(TR_out, sol, "tr-r2-$(suffix)")
end

function demo_tan()
  nlp_train, nls_train, sol_train  = RegularizedProblems.svm_train_model() #
  nlp_test, nls_test, sol_test = RegularizedProblems.svm_train_model()
  nlp_train = LSR1Model(nlp_train)
  f_test = LSR1Model(nlp_test)

  λ = 1e-2
  h = RootNormLhalf(λ)
  χ = NormLinf(1.0)

  out = TR(LBFGSModel(nlp_train), h, χ, options, x0 = nlp_train.meta.x0)
  out = LMTR(nls_train, h, χ,  options, x0 = nls_train.meta.x0)
  reset!(nls_train)
  out = LM(nls_train, h, options, x0 = nls_train.meta.x0)
end

demo_tan()

