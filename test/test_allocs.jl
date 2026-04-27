
# Test non allocating solve!
@testset "NLP allocs" begin
  for (h, h_name) ∈ ((NormL0(λ), "l0"),)
    for (solver, solver_name) ∈
        ((:R2Solver, "R2"), (:R2DHSolver, "R2DH"), (:R2NSolver, "R2N"), (:TRDHSolver, "TRDH"))
      @testset "$(solver_name)" begin
        reg_nlp = RegularizedNLPModel(LBFGSModel(bpdn), h)
        solver = eval(solver)(reg_nlp)
        stats = RegularizedExecutionStats(reg_nlp)
        solver_name == "R2" &&
          @test @wrappedallocs(solve!(solver, reg_nlp, stats, ν = 1.0, atol = 1e-6, rtol = 1e-6)) ==
                0
        (solver_name == "R2DH" || solver_name == "R2N") && @test @wrappedallocs(
          solve!(solver, reg_nlp, stats, σk = 1.0, atol = 1e-6, rtol = 1e-6)
        ) == 0
        (solver_name == "TRDH") &&
          @test @wrappedallocs(solve!(solver, reg_nlp, stats, atol = 1e-6, rtol = 1e-6)) == 0
        @test stats.status == :first_order
      end
    end
    @testset "Augmented Lagrangian" begin
      continue # FIXME : fails due to type instabilities in ADNLPModels...
      reg_nlp = RegularizedNLPModel(hs8(backend = :generic), h)
      solver = ALSolver(reg_nlp)
      stats = RegularizedExecutionStats(reg_nlp)
      @test @wrappedallocs(solve!(solver, reg_nlp, stats, atol = 1e-3)) == 0
      @test stats.status == :first_order
    end
  end
end

@testset "NLS allocs" begin
  for (h, h_name) ∈ ((NormL0(λ), "l0"),)
    for (solver, solver_name) ∈ ((:LMSolver, "LM"),)
      @testset "$(solver_name)" begin
        solver_name == "LM" && continue #FIXME
        reg_nlp = RegularizedNLPModel(bpdn_nls, h)
        solver = eval(solver)(reg_nlp)
        stats = RegularizedExecutionStats(reg_nlp)
        @test @wrappedallocs(solve!(solver, reg_nlp, stats, σk = 1.0, atol = 1e-6, rtol = 1e-6)) ==
              0
        @test stats.status == :first_order
      end
    end
  end
end

@testset "NLS allocs" begin
  for (h, h_name) ∈ ((NormL0(λ), "l0"),)
    for (solver, solver_name) ∈ ((:LMTRSolver, "LMTR"),)
      @testset "$(solver_name)" begin
        solver_name == "LMTR" && continue #FIXME
        reg_nlp = RegularizedNLPModel(bpdn_nls, h)
        solver = eval(solver)(reg_nlp)
        stats = RegularizedExecutionStats(reg_nlp)
        @test @wrappedallocs(solve!(solver, reg_nlp, stats, Δk = 1.0, atol = 1e-6, rtol = 1e-6)) ==
              0
        @test stats.status == :first_order
      end
    end
  end
end
