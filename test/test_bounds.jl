const subsolver_options = deepcopy(options)
TR_TRDH(args...; kwargs...) = TR(args...; subsolver = TRDH, kwargs...)

for (mod, mod_name) ∈ ((x -> x, "exact"), (LSR1Model, "lsr1"), (LBFGSModel, "lbfgs"))
  for (h, h_name) ∈ ((NormL0(λ), "l0"), (NormL1(λ), "l1"))
    for solver_sym ∈ (:TR, :R2, :TR_TRDH)
      solver_sym ∈ (:TR, :TR_TRDH) && mod_name == "exact" && continue
      solver_name = string(solver_sym)
      solver = eval(solver_sym)
      @testset "bpdn-with-bounds-$(mod_name)-$(solver_name)-$(h_name)" begin
        x0 = zeros(bpdn2.meta.nvar)
        p = randperm(bpdn2.meta.nvar)[1:nz]
        args = solver_sym == :R2 ? () : (NormLinf(1.0),)
        @test has_bounds(mod(bpdn2))
        out = solver(mod(bpdn2), h, args..., options; x0 = x0)
        @test typeof(out.solution) == typeof(bpdn2.meta.x0)
        @test length(out.solution) == bpdn2.meta.nvar
        @test typeof(out.solver_specific[:Fhist]) == typeof(out.solution)
        @test typeof(out.solver_specific[:Hhist]) == typeof(out.solution)
        @test typeof(out.solver_specific[:SubsolverCounter]) == Array{Int, 1}
        @test typeof(out.dual_feas) == eltype(out.solution)
        @test length(out.solver_specific[:Fhist]) == length(out.solver_specific[:Hhist])
        @test length(out.solver_specific[:Fhist]) == length(out.solver_specific[:SubsolverCounter])
        @test obj(bpdn2, out.solution) == out.solver_specific[:Fhist][end]
        @test h(out.solution) == out.solver_specific[:Hhist][end]
        @test out.status == :first_order
      end
    end
  end
end

for (mod, mod_name) ∈ ((SpectralGradientModel, "spg"),)
  # ((DiagonalPSBModel, "psb"),(DiagonalAndreiModel, "andrei"))   work but do not always terminate
  for (h, h_name) ∈ ((NormL0(λ), "l0"), (NormL1(λ), "l1"))
    @testset "bpdn-with-bounds-$(mod_name)-TRDH-$(h_name)" begin
      x0 = zeros(bpdn2.meta.nvar)
      p = randperm(bpdn2.meta.nvar)[1:nz]
      χ = NormLinf(1.0)
      out = TRDH(mod(bpdn2), h, χ, options; x0 = x0)
      @test typeof(out.solution) == typeof(bpdn2.meta.x0)
      @test length(out.solution) == bpdn2.meta.nvar
      @test typeof(out.dual_feas) == eltype(out.solution)
      @test out.status == :first_order
    end
  end
end

for (h, h_name) ∈ ((NormL0(λ), "l0"), (NormL1(λ), "l1"))
  for solver_sym ∈ (:LMTR, :LM)
    solver_name = string(solver_sym)
    solver = eval(solver_sym)
    @testset "bpdn-with-bounds-ls-$(solver_name)-$(h_name)" begin
      x0 = zeros(bpdn_nls2.meta.nvar)
      args = solver_sym == :LM ? () : (NormLinf(1.0),)
      @test has_bounds(bpdn_nls2)
      out = solver(bpdn_nls2, h, args..., options, x0 = x0)
      @test typeof(out.solution) == typeof(bpdn.meta.x0)
      @test length(out.solution) == bpdn.meta.nvar
      @test typeof(out.dual_feas) == eltype(out.solution)
      @test out.status == :first_order
    end
  end
end

# LMTR with TRDH as subsolver
for (h, h_name) ∈ ((NormL0(λ), "l0"), (NormL1(λ), "l1"))
  @testset "bpdn-with-bounds-ls-LMTR-$(h_name)-TRDH" begin
    x0 = zeros(bpdn_nls2.meta.nvar)
    @test has_bounds(bpdn_nls2)
    LMTR_out = LMTR(
      bpdn_nls2,
      h,
      NormLinf(1.0),
      options,
      x0 = x0,
      subsolver = TRDH,
      subsolver_options = subsolver_options,
    )
    @test typeof(LMTR_out.solution) == typeof(bpdn_nls2.meta.x0)
    @test length(LMTR_out.solution) == bpdn_nls2.meta.nvar
    @test typeof(LMTR_out.solver_specific[:Fhist]) == typeof(LMTR_out.solution)
    @test typeof(LMTR_out.solver_specific[:Hhist]) == typeof(LMTR_out.solution)
    @test typeof(LMTR_out.solver_specific[:SubsolverCounter]) == Array{Int, 1}
    @test typeof(LMTR_out.dual_feas) == eltype(LMTR_out.solution)
    @test length(LMTR_out.solver_specific[:Fhist]) == length(LMTR_out.solver_specific[:Hhist])
    @test length(LMTR_out.solver_specific[:Fhist]) ==
          length(LMTR_out.solver_specific[:SubsolverCounter])
    @test length(LMTR_out.solver_specific[:Fhist]) == length(LMTR_out.solver_specific[:NLSGradHist])
    @test LMTR_out.solver_specific[:NLSGradHist][end] ==
          bpdn_nls2.counters.neval_jprod_residual + bpdn_nls2.counters.neval_jtprod_residual - 1
    @test obj(bpdn_nls2, LMTR_out.solution) == LMTR_out.solver_specific[:Fhist][end]
    @test h(LMTR_out.solution) == LMTR_out.solver_specific[:Hhist][end]
    @test LMTR_out.status == :first_order
  end
end

# LM with R2DH as subsolver
for (h, h_name) ∈ ((NormL0(λ), "l0"),)
  @testset "bpdn-with-bounds-ls-LM-$(h_name)-R2DH" begin
    x0 = zeros(bpdn_nls2.meta.nvar)
    @test has_bounds(bpdn_nls2)
    LM_out =
      LM(bpdn_nls2, h, options, x0 = x0, subsolver = R2DHSolver)#, subsolver_options = subsolver_options)
    @test typeof(LM_out.solution) == typeof(bpdn.meta.x0)
    @test length(LM_out.solution) == bpdn.meta.nvar
    @test typeof(LM_out.dual_feas) == eltype(LM_out.solution)
    @test LM_out.status == :first_order
  end
end
