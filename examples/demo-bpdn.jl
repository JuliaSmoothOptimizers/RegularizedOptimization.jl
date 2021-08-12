using Random
using ProximalOperators
using NLPModels, NLPModelsModifiers, RegularizedProblems, TRNC

include("plot-utils-bpdn.jl")

Random.seed!(1234)

function demo_solver(f, sol, h, χ, suffix = "l0-linf")
  options = TRNCoptions(ν = 1.0,  β = 1e16, ϵ = 1e-6, verbose = 10)

  @info "using R2 to solve with" h
  reset!(f)
  xr2, Fhistr2, Hhistr2, Comp_pgr2 = R2(f, h, options, f.meta.x0)
  @info "R2 relative error" norm(xr2 - sol) / norm(sol)
  plot_bpdn(Comp_pgr2[2,:], Fhistr2+Hhistr2, xr2, sol, "r2-$(suffix)")

  @info " using TR to solve with" h χ
  reset!(f)
  xtr, Fhist, Hhist, Comp_pg = TR(f, h, χ, options, x0 = f.meta.x0)
  @info "TR relative error" norm(xtr - sol) / norm(sol)
  plot_bpdn(Comp_pg[2,:], Fhist+Hhist, xtr, sol, "tr-r2-$(suffix)")
end

function demo_bpdn(compound = 1)
  model, sol = bpdn_model(compound)
  f = LSR1Model(model)

  λ = norm(grad(model, zeros(model.meta.nvar)), Inf) / 10
  demo_solver(f, sol, NormL0(λ), NormLinf(1.0))

  k = 10 * compound
  demo_solver(f, sol, IndBallL0(10 * compound), NormLinf(1.0), "b0-linf")

  demo_solver(f, sol, NormL1(λ), NormL2(1.0), "l1-l2")

  demo_solver(f, sol, NormL1(λ), NormLinf(1.0), "l1-linf")
end

demo_bpdn()

