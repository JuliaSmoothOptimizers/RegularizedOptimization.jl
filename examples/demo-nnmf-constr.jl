using Random
using LinearAlgebra
using ProximalOperators
using NLPModels, NLPModelsModifiers, RegularizedProblems, RegularizedOptimization
using Printf

include("plot-utils-nnmf.jl")

Random.seed!(1234)

function demo_solver(f, h, χ, selected, Avec, m, n, k, suffix = "l0-linf")
  options = ROSolverOptions(
    ν = 1.0e-3,
    β = 1e16,
    ϵa = 1e-6,
    ϵr = 1e-6,
    verbose = 1,
    maxIter = 500,
    spectral = false,
    psb = false,
    andrei = false,
  )
  @info " using TR to solve with" h χ
  reset!(f)
  TR_out = TR(f, h, χ, options, selected = selected)
  @info "TR objective" TR_out.objective
  @info "elapsed time" TR_out.elapsed_time
  @info "gradient call" neval_grad(f)
#  plot_nnmf(TR_out, Avec, m, n, k, "tr-r2-$suffix")

  @info " using R2 to solve with" h
  reset!(f)
  R2_out = R2(f, h, options, selected = selected)
  @info "R2 objective" R2_out.objective
  @info "elapsed time" R2_out.elapsed_time
  @info "gradient call" neval_grad(f)
 # plot_nnmf(R2_out, Avec, m, n, k, "r2-$suffix")

  subsolver_options = ROSolverOptions(spectral = false, psb = true, ϵa = options.ϵa)
  @info " using TR with TRDH as subproblem to solve with" h χ
  reset!(f)
  TR2_out = TR(
    f,
    h,
    χ,
    options,
    selected = selected,
    subsolver = TRDH,
    subsolver_options = subsolver_options,
  )
#  plot_nnmf(TR2_out, Avec, m, n, k, "tr-trdh-$suffix")
  @info "TR-TRDH objective" TR2_out.objective
  @info "elapsed time" TR2_out.elapsed_time
  @info "gradient call" neval_grad(f)

  @info " using TRDH to solve with" h χ
  reset!(f)
  TRDH_out = TRDH(f, h, χ, options, selected = selected)
 # plot_nnmf(TRDH_out, Avec, m, n, k, "trdh-$suffix")
  @info "TRDH objective" TRDH_out.objective
  @info "elapsed time" TRDH_out.elapsed_time
  @info "gradient call" neval_grad(f)
end

function demo_nnmf()
  m, n, k = 100, 50, 5
  model, A, selected = nnmf_model(m, n, k)
  f = LSR1Model(model)
  λ = 1.0e-1#norm(grad(model, rand(model.meta.nvar)), Inf) / 200
  demo_solver(f, NormL0(λ), NormLinf(1.0), selected, A, m, n, k, "l0-linf")
#  λ = norm(grad(model, rand(model.meta.nvar)), Inf) / 100000
#  demo_solver(f, NormL1(λ), NormLinf(1.0), selected, A, m, n, k, "l1-linf")
end

demo_nnmf()
