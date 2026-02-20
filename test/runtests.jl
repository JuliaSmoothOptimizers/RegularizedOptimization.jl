using LinearAlgebra: length
using LinearAlgebra, Random, Test
using ProximalOperators
using ADNLPModels,
  OptimizationProblems,
  OptimizationProblems.ADNLPProblems,
  NLPModels,
  NLPModelsModifiers,
  RegularizedProblems,
  RegularizedOptimization,
  SolverCore

Random.seed!(0)
const global compound = 1
const global nz = 10 * compound
const global options = ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10)
const global bpdn, bpdn_nls, sol = bpdn_model(compound)
const global bpdn2, bpdn_nls2, sol2 = bpdn_model(compound, bounds = true)
const global λ = norm(grad(bpdn, zeros(bpdn.meta.nvar)), Inf) / 10

include("test_AL.jl")

for (mod, mod_name) ∈ ((x -> x, "exact"), (LSR1Model, "lsr1"), (LBFGSModel, "lbfgs"))
  for (h, h_name) ∈ ((NormL0(λ), "l0"), (NormL1(λ), "l1"), (IndBallL0(10 * compound), "B0"))
    for solver_sym ∈ (:R2, :TR)
      solver_sym == :TR && mod_name == "exact" && continue
      solver_sym == :TR && h_name == "B0" && continue  # FIXME
      solver_name = string(solver_sym)
      solver = eval(solver_sym)
      @testset "bpdn-$(mod_name)-$(solver_name)-$(h_name)" begin
        x0 = zeros(bpdn.meta.nvar)
        p = randperm(bpdn.meta.nvar)[1:nz]
        x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
        args = solver_sym == :R2 ? () : (NormLinf(1.0),)
        out = solver(mod(bpdn), h, args..., options, x0 = x0)
        @test typeof(out.solution) == typeof(bpdn.meta.x0)
        @test length(out.solution) == bpdn.meta.nvar
        @test typeof(out.dual_feas) == eltype(out.solution)
        @test out.status == :first_order
        @test out.step_status == (out.iter > 0 ? :accepted : :unknown)
      end
    end
  end
end

for (mod, mod_name) ∈ ((SpectralGradientModel, "spg"),)
  # ((DiagonalPSBModel, "psb"),(DiagonalAndreiModel, "andrei"))   work but do not always terminate
  for (h, h_name) ∈ ((NormL0(λ), "l0"), (NormL1(λ), "l1"))  #, (IndBallL0(10 * compound), "B0"))
    @testset "bpdn-$(mod_name)-TRDH-$(h_name)" begin
      x0 = zeros(bpdn.meta.nvar)
      p = randperm(bpdn.meta.nvar)[1:nz]
      # x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
      χ = NormLinf(1.0)
      out = TRDH(mod(bpdn), h, χ, options, x0 = x0)
      @test typeof(out.solution) == typeof(bpdn.meta.x0)
      @test length(out.solution) == bpdn.meta.nvar
      @test typeof(out.dual_feas) == eltype(out.solution)
      @test out.status == :first_order
      @test out.step_status == (out.iter > 0 ? :accepted : :unknown)

      # Test with the different stopping criteria
      out = TRDH(mod(bpdn), h, χ, options, x0 = x0, atol_decr = 1e-6, rtol_decr = 1e-6, atol_step = 0.0, rtol_step = 0.0)
      @test typeof(out.solution) == typeof(bpdn.meta.x0)
      @test length(out.solution) == bpdn.meta.nvar
      @test typeof(out.dual_feas) == eltype(out.solution)
      @test out.status == :first_order

      out = TRDH(mod(bpdn), h, χ, options, x0 = x0, atol_decr = 0.0, rtol_decr = 0.0, atol_step = 1e-6, rtol_step = 1e-6)
      @test typeof(out.solution) == typeof(bpdn.meta.x0)
      @test length(out.solution) == bpdn.meta.nvar
      @test typeof(out.dual_feas) == eltype(out.solution)
      @test out.status == :first_order
    end
  end
end

# TR with h = L1 and χ = L2 is a special case
for (mod, mod_name) ∈ ((LSR1Model, "lsr1"), (LBFGSModel, "lbfgs"))
  for (h, h_name) ∈ ((NormL1(λ), "l1"),)
    @testset "bpdn-$(mod_name)-TR-$(h_name)" begin
      x0 = zeros(bpdn.meta.nvar)
      p = randperm(bpdn.meta.nvar)[1:nz]
      x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
      TR_out = TR(mod(bpdn), h, NormL2(1.0), options, x0 = x0)
      @test typeof(TR_out.solution) == typeof(bpdn.meta.x0)
      @test length(TR_out.solution) == bpdn.meta.nvar
      @test typeof(TR_out.dual_feas) == eltype(TR_out.solution)
      @test TR_out.status == :first_order
      @test TR_out.step_status == (TR_out.iter > 0 ? :accepted : :unknown)

      # Test with the different stopping criteria
      TR_out = TR(mod(bpdn), h, NormL2(1.0), options, x0 = x0, atol_decr = 1e-6, rtol_decr = 1e-6, atol_step = 0.0, rtol_step = 0.0)
      @test typeof(TR_out.solution) == typeof(bpdn.meta.x0)
      @test length(TR_out.solution) == bpdn.meta.nvar
      @test typeof(TR_out.dual_feas) == eltype(TR_out.solution)
      @test TR_out.status == :first_order

      TR_out = TR(mod(bpdn), h, NormL2(1.0), options, x0 = x0, atol_decr = 0.0, rtol_decr = 0.0, atol_step = 1e-6, rtol_step = 1e-6)
      @test typeof(TR_out.solution) == typeof(bpdn.meta.x0)
      @test length(TR_out.solution) == bpdn.meta.nvar
      @test typeof(TR_out.dual_feas) == eltype(TR_out.solution)
      @test TR_out.status == :first_order
    end
  end
end

for (h, h_name) ∈ ((NormL0(λ), "l0"), (NormL1(λ), "l1"), (IndBallL0(10 * compound), "B0"))
  for solver_sym ∈ (:LM, :LMTR)
    solver_name = string(solver_sym)
    solver = eval(solver_sym)
    solver_sym == :LMTR && h_name == "B0" && continue  # FIXME
    @testset "bpdn-ls-$(solver_name)-$(h_name)" begin
      x0 = zeros(bpdn_nls.meta.nvar)
      p = randperm(bpdn_nls.meta.nvar)[1:nz]
      x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
      args = solver_sym == :LM ? () : (NormLinf(1.0),)
      out = solver(bpdn_nls, h, args..., options, x0 = x0)
      @test typeof(out.solution) == typeof(bpdn.meta.x0)
      @test length(out.solution) == bpdn.meta.nvar
      @test typeof(out.dual_feas) == eltype(out.solution)
      @test out.status == :first_order
      @test out.step_status == (out.iter > 0 ? :accepted : :unknown)
    end
  end
end

# LMTR with h = L1 and χ = L2 is a special case
for (h, h_name) ∈ ((NormL1(λ), "l1"),)
  @testset "bpdn-ls-LMTR-$(h_name)" begin
    x0 = zeros(bpdn_nls.meta.nvar)
    p = randperm(bpdn_nls.meta.nvar)[1:nz]
    x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
    LMTR_out = LMTR(bpdn_nls, h, NormL2(1.0), options, x0 = x0)
    @test typeof(LMTR_out.solution) == typeof(bpdn.meta.x0)
    @test length(LMTR_out.solution) == bpdn.meta.nvar
    @test typeof(LMTR_out.dual_feas) == eltype(LMTR_out.solution)
    @test LMTR_out.status == :first_order
    @test LMTR_out.step_status == (LMTR_out.iter > 0 ? :accepted : :unknown)
  end
end

R2N_R2DH(args...; kwargs...) = R2N(args...; subsolver = R2DHSolver, kwargs...)
for (mod, mod_name) ∈ (
  (SpectralGradientModel, "spg"),
  (DiagonalPSBModel, "psb"),
  (LSR1Model, "lsr1"),
  (LBFGSModel, "lbfgs"),
)
  for (h, h_name) ∈ ((NormL0(λ), "l0"), (NormL1(λ), "l1"))
    for solver_sym ∈ (:R2DH, :R2N, :R2N_R2DH)
      solver_sym ∈ (:R2N, :R2N_R2DH) && mod_name ∈ ("spg", "psb") && continue
      solver_sym == :R2DH && mod_name != "spg" && continue
      solver_sym == :R2N_R2DH && h_name == "l1" && continue # this test seems to fail because s seems to be equal to zeros within the subsolver
      solver_name = string(solver_sym)
      solver = eval(solver_sym)
      @testset "bpdn-$(mod_name)-$(solver_name)-$(h_name)" begin
        x0 = zeros(bpdn.meta.nvar)
        out = solver(mod(bpdn), h, options, x0 = x0)
        @test typeof(out.solution) == typeof(bpdn.meta.x0)
        @test length(out.solution) == bpdn.meta.nvar
        @test typeof(out.dual_feas) == eltype(out.solution)
        @test out.status == :first_order
        @test out.step_status == (out.iter > 0 ? :accepted : :unknown)

        # Test with the different stopping criteria
        out = solver(mod(bpdn), h, options, x0 = x0, atol_decr = 1e-6, rtol_decr = 1e-6, atol_step = 0.0, rtol_step = 0.0)
        @test typeof(out.solution) == typeof(bpdn.meta.x0)
        @test length(out.solution) == bpdn.meta.nvar
        @test typeof(out.dual_feas) == eltype(out.solution)
        @test out.status == :first_order

        out = solver(mod(bpdn), h, options, x0 = x0, atol_decr = 0.0, rtol_decr = 0.0, atol_step = 1e-6, rtol_step = 1e-6)
        @test typeof(out.solution) == typeof(bpdn.meta.x0)
        @test length(out.solution) == bpdn.meta.nvar
        @test typeof(out.dual_feas) == eltype(out.solution)
        @test out.status == :first_order

      end
    end
  end
end

include("test_bounds.jl")
include("test_allocs.jl")
