# Reject every trial. The GLL window may contain the initial objective only.
@testset "GLL memory ignores rejected steps" begin
  m = 4
  function check_window(reg_nlp, solver)
    stats = RegularizedExecutionStats(reg_nlp)
    solve!(solver, reg_nlp, stats; η1 = Inf, max_iter = 2, verbose = 0)
    @test stats.iter >= 1
    @test isfinite(solver.m_fh_hist[1])
    @test all(==( -Inf), @view solver.m_fh_hist[2:end])
  end

  # The first accepted value goes in the next slot, even after rejected steps.
  function check_next_slot(reg_nlp, solver)
    fx0 = Ref(NaN)
    stats = RegularizedExecutionStats(reg_nlp)
    cb = (reg_nlp, solver, stats) -> begin
      if stats.iter == 0
        fx0[] = solver.m_fh_hist[1]
      elseif stats.step_status == :accepted
        @test solver.m_fh_hist[1] == fx0[]
        @test isfinite(solver.m_fh_hist[2])
        @test all(==( -Inf), @view solver.m_fh_hist[3:end])
        stats.status = :user
      end
    end
    solve!(solver, reg_nlp, stats; callback = cb, max_iter = 30, verbose = 0)
    @test stats.status == :user
  end

  reg = RegularizedNLPModel(bpdn, NormL1(λ))
  @testset "R2" begin
    solver = R2Solver(reg; m_monotone = m)
    check_window(reg, solver)
    check_next_slot(reg, solver)
  end
  @testset "R2DH" begin
    solver = R2DHSolver(reg; m_monotone = m)
    check_window(reg, solver)
    check_next_slot(reg, solver)
  end

  reg_qn = RegularizedNLPModel(LBFGSModel(bpdn), NormL1(λ))
  @testset "R2N" begin
    solver = R2NSolver(reg_qn; m_monotone = m)
    check_window(reg_qn, solver)
    check_next_slot(reg_qn, solver)
  end
  @testset "TR" begin
    solver = TRSolver(reg_qn; m_monotone = m)
    check_window(reg_qn, solver)
    check_next_slot(reg_qn, solver)
  end

  reg_nls = RegularizedNLPModel(bpdn_nls, NormL1(λ))
  @testset "LM" begin
    solver = LMSolver(reg_nls; m_monotone = m)
    check_window(reg_nls, solver)
    check_next_slot(reg_nls, solver)
  end
end
