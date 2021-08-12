using LinearAlgebra, Random, Test
using ProximalOperators
using NLPModels, NLPModelsModifiers, RegularizedProblems, TRNC

const global compound = 1
const global nz = 10 * compound
const global options = TRNCoptions(ν = 1.0,  β = 1e16, ϵ = 1e-6, verbose = 0)
const global bpdn, sol = bpdn_model(compound)
const global λ = norm(grad(bpdn, zeros(bpdn.meta.nvar)), Inf) / 10

for (mod, mod_name) ∈ ((x -> x, "exact"), (LSR1Model, "lsr1"), (LBFGSModel, "lbfgs"))
  for (h, h_name) ∈ ((NormL0(λ), "l0"), (NormL1(λ), "l1"), (IndBallL0(10 * compound), "B0"))
    for solver_sym ∈ (:R2, :TR)
      solver_sym == :TR && mod_name == "exact" && continue
      solver_name = string(solver_sym)
      solver = eval(solver_sym)
      @testset "bpdn-$(mod_name)-$(solver_name)-$(h_name)" begin
        x0 = zeros(bpdn.meta.nvar)
        p  = randperm(bpdn.meta.nvar)[1:nz]
        x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
        args = solver_sym == :R2 ? () : (NormLinf(1.0),)
        x, Fhist, Hhist, Comp_pg, ξ = solver(mod(bpdn), h, args..., options, x0 = x0)
        @test typeof(x) == typeof(bpdn.meta.x0)
        @test length(x) == bpdn.meta.nvar
        @test typeof(Fhist) == typeof(x)
        @test typeof(Hhist) == typeof(x)
        @test typeof(Comp_pg) == Matrix{Int}
        @test typeof(ξ) == eltype(x)
        @test length(Fhist) == length(Hhist)
        @test length(Fhist) == size(Comp_pg, 2)
        @test obj(bpdn, x) == Fhist[end]
        @test h(x) == Hhist[end]
        @test sqrt(ξ) < options.ϵ
      end
    end
  end
end

# TR with h = L1 and χ = L2 is a special case
for (mod, mod_name) ∈ ((LSR1Model, "lsr1"), (LBFGSModel, "lbfgs"))
  for (h, h_name) ∈ ((NormL1(λ), "l1"),)
    @testset "bpdn-$(mod_name)-TR-$(h_name)" begin
      x0 = zeros(bpdn.meta.nvar)
      p  = randperm(bpdn.meta.nvar)[1:nz]
      x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
      x, Fhist, Hhist, Comp_pg, ξ = TR(mod(bpdn), h, NormL2(1.0), options, x0 = x0)
      @test typeof(x) == typeof(bpdn.meta.x0)
      @test length(x) == bpdn.meta.nvar
      @test typeof(Fhist) == typeof(x)
      @test typeof(Hhist) == typeof(x)
      @test typeof(Comp_pg) == Matrix{Int}
      @test typeof(ξ) == eltype(x)
      @test length(Fhist) == length(Hhist)
      @test length(Fhist) == size(Comp_pg, 2)
      @test obj(bpdn, x) == Fhist[end]
      @test h(x) == Hhist[end]
      @test sqrt(ξ) < options.ϵ
    end
  end
end

