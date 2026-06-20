function test_solver(
  reg_nlp::R,
  solver::F;
  expected_status = :first_order,
  solver_constructor_kwargs = (;),
  solver_kwargs = (;),
) where {R, F}

  # Test output with allocating calling form
  stats_basic = solver(
    reg_nlp;
    solver_constructor_kwargs...,
    solver_kwargs...,
  )

  x0 = get(solver_kwargs, :x, reg_nlp.model.meta.x0)
  @test typeof(stats_basic.solution) == typeof(x0)
  @test length(stats_basic.solution) == reg_nlp.model.meta.nvar
  @test typeof(stats_basic.dual_feas) == eltype(stats_basic.solution)
  @test stats_basic.status == expected_status
  @test obj(reg_nlp, stats_basic.solution) == stats_basic.objective
  @test stats_basic.objective <= obj(reg_nlp, x0)

  # Test output with optimized calling form
  solver_constructor = getfield(RegularizedOptimization, Symbol(string(solver) * "Solver"))
  solver_object = solver_constructor(reg_nlp; solver_constructor_kwargs...)
  stats_optimized = RegularizedExecutionStats(reg_nlp)

  solve!(solver_object, reg_nlp, stats_optimized; solver_kwargs...)
  @test typeof(stats_optimized.solution) == typeof(x0)
  @test length(stats_optimized.solution) == reg_nlp.model.meta.nvar
  @test typeof(stats_optimized.dual_feas) == eltype(stats_optimized.solution)
  @test stats_optimized.status == expected_status
  @test obj(reg_nlp, stats_optimized.solution) == stats_optimized.objective
  @test stats_optimized.objective <= obj(reg_nlp, x0)
  @test all(solver_object.mν∇fk + solver_object.∇fk/stats_optimized.solver_specific[:sigma_cauchy] .≤ eps(eltype(solver_object.mν∇fk)))

  # TODO: test that the optimized entries in stats_optimized and stats_basic are the same.

end
