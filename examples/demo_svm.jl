using Random
using LinearAlgebra
using ProximalOperators, ShiftedProximalOperators, MLDatasets
using ADNLPModels, NLPModels, NLPModelsModifiers, ReverseADNLSModels
using RegularizedOptimization, RegularizedProblems
using BenchmarkTools

# include("plot-utils-bpdn.jl")

Random.seed!(1234)

function demo_solver(f_tr, sol_tr, f_te, sol_te, h, χ, suffix = "l0-linf")
  options = ROSolverOptions(ν = 1.0, β = 1e16, ϵ = 1e-3, verbose = 10)

  @info "using R2 to solve with" h
  reset!(f_tr)
  # out = R2(f_tr, h, options, x0 = zero(f_tr.meta.x0))
  # out = TR(f_tr, h, χ, options, x0 = f_tr.meta.x0)

  # @info "R2 relative training error" obj(f_tr, out.solution)
  # @info "R2 relative testing error" obj(f_te, out.solution)
  # @info "obj hist" out.solver_specific[:Fhist] + out.solver_specific[:Hhist]
  # plot_tan(R2_out, sol, "r2-$(suffix)")

  # @info " using TR to solve with" h χ
  # reset!(f)
  nls_model = ADNLSModel(f_tr, )
  nls_model_t = ADNLSModel(resid_test, )
  TR_out = LMTR(f_tr, h, χ, options, x0 = f.meta.x0)
  # @info "TR relative error" norm(TR_out.solution - sol) / norm(sol)
  # plot_tan(TR_out, sol, "tr-r2-$(suffix)")
end

function demo_tan()
  nlp_train, nls_train, resid_train, obj_train, sol_train  = RegularizedProblems.tanh_train_model() #
  model_test, resid_test, sol_test = RegularizedProblems.tanh_train_model()
  # f_train = LSR1Model(model_train)
  # f_test = LSR1Model(model_test)
  # nls_train = ADNLSModel(resid_train, ones(size(nls_train.meta.x0)),size(sol_train,1) + size(nls_train.meta.x0,1))
  # f = ReverseADNLSModel(resid!, size(sol_train,1), ones(size(model_train.meta.x0)), name = "Dominique")

  options = ROSolverOptions(ν = 1.0, β = 1e16, ϵ = 1e-4, verbose = 10, σmin = 1e-6);
  λ = 1e-2
  h = RootNormLhalf(λ)
  # h = NormL0(λ)
  χ = NormLinf(1.0)

  out = TR(LBFGSModel(nlp_train), h, χ, options, x0 = nlp_train.meta.x0)
  out = LMTR(nls_train, h, χ,  options, x0 = nls_train.meta.x0)
  reset!(nls_train)
  out = LM(nls_train, h, options, x0 = nls_train.meta.x0)

  # demo_solver(f_train, sol_train, f_test, sol_test, RootNormLhalf(λ), NormLinf(1.0), "l1/2-linf")
  # demo_solver(f_train, sol_train, f_test, sol_test, NormL1(λ), NormLinf(1.0), "l1-linf")
  # demo_solver(f_train, sol_train, f_test, sol_test, NormL0(λ), NormLinf(1.0), "l1/2-linf")

  # demo_solver(resid_train, sol_train, resid_test, sol_test, RootNormLhalf(λ), NormLinf(1.0), "l1/2-linf")
end

# demo_tan()

function comp_derivs()
  model_train, fk, resid!, resid, sol_train = tanh_train_model() #
  fad = ReverseADNLSModel(resid!, size(sol_train,1), ones(size(model_train.meta.x0)), name = "Dominique")

  xk = 10*randn(size(fk.meta.x0));
  v = 10*randn(size(xk));
  Jvac = zeros(size(sol_train));
  Jvac_ad = similar(Jvac);
  @show @benchmark jprod_residual!($fk, $xk, $v, $Jvac)
  @show @benchmark jprod_residual!($fad, $xk, $v, $Jvac_ad)

  # @show norm(Jvac - Jvac_ad)

  v = 10*randn(size(sol_train));
  Jtvac = zero(xk);
  Jtvac_ad = zero(xk);

  @show @benchmark jtprod_residual!($fk, $xk, $v, $Jtvac)
  @show @benchmark jtprod_residual!($fad, $xk, $v, $Jtvac_ad)

  @show norm(Jtvac - Jtvac_ad)

end