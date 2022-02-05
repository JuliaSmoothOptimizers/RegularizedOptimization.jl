using Random
using LinearAlgebra
using ProximalOperators, ShiftedProximalOperators
using ADNLPModels, NLPModels, NLPModelsModifiers, ReverseADNLSModels
using RegularizedOptimization, RegularizedProblems

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
  model_train, f, resid!, resid, sol_train = tanh_train_model() #
  model_test, resid_test, sol_test = tanh_train_model()
  # f_train = LSR1Model(model_train)
  # f_test = LSR1Model(model_test)
  # f = ADNLSModel(resid, ones(size(model_train.meta.x0)),size(sol_train,1))
  # f = ReverseADNLSModel(resid!, size(sol_train,1), ones(size(model_train.meta.x0)), name = "Dominique")

  options = ROSolverOptions(ν = 1.0, β = 1e16, ϵ = 1e-3, verbose = 10);
  λ = 1e-3
  h = RootNormLhalf(λ)
  # h = NormL1(λ)
  χ = NormLinf(1.0)

  out = LMTR(f, h, χ,  options, x0 = f.meta.x0, subsolver = ReSp1)
  # out = LM(f, h, options, x0 = f.meta.x0, subsolver = R2)

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