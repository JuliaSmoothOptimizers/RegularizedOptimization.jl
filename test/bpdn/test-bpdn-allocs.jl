# Test non allocating solve!
@testset "BPDN allocs" begin
  for (h, h_name) ∈ ((NormL0(λ), "l0"),)
    for (solver, solver_name) ∈ (
      (:R2Solver, "R2"),
      (:R2DHSolver, "R2DH"),
      (:R2NSolver, "R2N"),
      (:TRDHSolver, "TRDH"),
      (:TRSolver, "TR"),
    )
      @testset "$(solver_name)" begin
        (solver_name == "R2N" || solver_name == "TR") && continue #FIXME
        reg_nlp = RegularizedNLPModel(LBFGSModel(bpdn), h)
        solver = eval(solver)(reg_nlp)
        stats = RegularizedExecutionStats(reg_nlp)
        solver_name == "R2" && @test @wrappedallocs(
          solve!(solver, reg_nlp, stats, ν = 1.0, atol = 1e-6, rtol = 1e-6)
        ) == 0
        solver_name == "R2DH" && @test @wrappedallocs(
          solve!(solver, reg_nlp, stats, σk = 1.0, atol = 1e-6, rtol = 1e-6)
        ) == 0
        solver_name == "TRDH" &&
          @test @wrappedallocs(solve!(solver, reg_nlp, stats, atol = 1e-6, rtol = 1e-6)) ==
                0
        @test stats.status == :first_order
      end
    end
    
  end
  
  for (h, h_name) ∈ ((NormL0(λ), "l0"),)
    for (solver, solver_name) ∈ ((:LMSolver, "LM"),)
      @testset "$(solver_name)" begin
        solver_name == "LM" && continue #FIXME
        reg_nlp = RegularizedNLPModel(bpdn_nls, h)
        solver = eval(solver)(reg_nlp)
        stats = RegularizedExecutionStats(reg_nlp)
        @test @wrappedallocs(
          solve!(solver, reg_nlp, stats, σk = 1.0, atol = 1e-6, rtol = 1e-6)
        ) == 0
        @test stats.status == :first_order
      end
    end
  end
end