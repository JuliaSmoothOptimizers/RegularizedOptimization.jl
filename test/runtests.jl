using LinearAlgebra: length
using LinearAlgebra, Random, Test
using ProximalOperators
using NLPModels, NLPModelsModifiers, RegularizedProblems, RegularizedOptimization

const global compound = 1
const global nz = 10 * compound
const global options = ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10)
const global bpdn, bpdn_nls, sol = bpdn_model(compound)
const global bpdn2, bpdn_nls2, sol2 = bpdn_model(compound, bounds = true)
const global λ = norm(grad(bpdn, zeros(bpdn.meta.nvar)), Inf) / 10

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
        @test typeof(out.solver_specific[:Fhist]) == typeof(out.solution)
        @test typeof(out.solver_specific[:Hhist]) == typeof(out.solution)
        @test typeof(out.solver_specific[:SubsolverCounter]) == Array{Int, 1}
        @test typeof(out.dual_feas) == eltype(out.solution)
        @test length(out.solver_specific[:Fhist]) == length(out.solver_specific[:Hhist])
        @test length(out.solver_specific[:Fhist]) == length(out.solver_specific[:SubsolverCounter])
        @test obj(bpdn, out.solution) == out.solver_specific[:Fhist][end]
        @test h(out.solution) == out.solver_specific[:Hhist][end]
        @test out.status == :first_order
      end
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
      @test typeof(TR_out.solver_specific[:Fhist]) == typeof(TR_out.solution)
      @test typeof(TR_out.solver_specific[:Hhist]) == typeof(TR_out.solution)
      @test typeof(TR_out.solver_specific[:SubsolverCounter]) == Array{Int, 1}
      @test typeof(TR_out.dual_feas) == eltype(TR_out.solution)
      @test length(TR_out.solver_specific[:Fhist]) == length(TR_out.solver_specific[:Hhist])
      @test length(TR_out.solver_specific[:Fhist]) ==
            length(TR_out.solver_specific[:SubsolverCounter])
      @test obj(bpdn, TR_out.solution) == TR_out.solver_specific[:Fhist][end]
      @test h(TR_out.solution) == TR_out.solver_specific[:Hhist][end]
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
      @test typeof(out.solution) == typeof(bpdn_nls.meta.x0)
      @test length(out.solution) == bpdn_nls.meta.nvar
      @test typeof(out.solver_specific[:Fhist]) == typeof(out.solution)
      @test typeof(out.solver_specific[:Hhist]) == typeof(out.solution)
      @test typeof(out.solver_specific[:SubsolverCounter]) == Array{Int, 1}
      @test typeof(out.dual_feas) == eltype(out.solution)
      @test length(out.solver_specific[:Fhist]) == length(out.solver_specific[:Hhist])
      @test length(out.solver_specific[:Fhist]) == length(out.solver_specific[:SubsolverCounter])
      @test obj(bpdn_nls, out.solution) == out.solver_specific[:Fhist][end]
      @test h(out.solution) == out.solver_specific[:Hhist][end]
      @test out.status == :first_order
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
    @test typeof(LMTR_out.solution) == typeof(bpdn_nls.meta.x0)
    @test length(LMTR_out.solution) == bpdn_nls.meta.nvar
    @test typeof(LMTR_out.solver_specific[:Fhist]) == typeof(LMTR_out.solution)
    @test typeof(LMTR_out.solver_specific[:Hhist]) == typeof(LMTR_out.solution)
    @test typeof(LMTR_out.solver_specific[:SubsolverCounter]) == Array{Int, 1}
    @test typeof(LMTR_out.dual_feas) == eltype(LMTR_out.solution)
    @test length(LMTR_out.solver_specific[:Fhist]) == length(LMTR_out.solver_specific[:Hhist])
    @test length(LMTR_out.solver_specific[:Fhist]) ==
          length(LMTR_out.solver_specific[:SubsolverCounter])
    @test obj(bpdn_nls, LMTR_out.solution) == LMTR_out.solver_specific[:Fhist][end]
    @test h(LMTR_out.solution) == LMTR_out.solver_specific[:Hhist][end]
    @test LMTR_out.status == :first_order
  end
end

include("test_bounds.jl")
