function test_solver(
  reg_nlp::R,
  solver_name::String;
  expected_status = :first_order,
  solver_constructor_kwargs = (;),
  solver_kwargs = (;),
) where {R}

  # Test output with allocating calling form
  solver_fun = getfield(RegularizedOptimization, Symbol(solver_name))
  stats_basic = solver_fun(
    reg_nlp.model,
    reg_nlp.h,
    ROSolverOptions();
    solver_constructor_kwargs...,
    solver_kwargs...,
  )

  x0 = get(solver_kwargs, :x0, reg_nlp.model.meta.x0)
  @test typeof(stats_basic.solution) == typeof(x0)
  @test length(stats_basic.solution) == reg_nlp.model.meta.nvar
  @test typeof(stats_basic.dual_feas) == eltype(stats_basic.solution)
  @test stats_basic.status == expected_status
  @test obj(reg_nlp, stats_basic.solution) == stats_basic.objective
  @test stats_basic.objective <= obj(reg_nlp, x0)

  # Test output with optimized calling form
  solver_constructor = getfield(RegularizedOptimization, Symbol(solver_name * "Solver"))
  solver = solver_constructor(reg_nlp; solver_constructor_kwargs...)
  stats_optimized = RegularizedExecutionStats(reg_nlp)

  # Remove the x0 entry from solver_kwargs
  optimized_solver_kwargs = Base.structdiff(solver_kwargs, NamedTuple{(:x0,)})
  solve!(solver, reg_nlp, stats_optimized; x = x0, optimized_solver_kwargs...) # It would be interesting to check for allocations here as well but depending on 
  # the structure of solver_kwargs, some variables might get boxed, resulting in 
  # false positives, for example if tol = 1e-3; solver_kwargs = (atol = tol),
  # then wrappedallocs would give a > 0 answer...
  @test typeof(stats_optimized.solution) == typeof(x0)
  @test length(stats_optimized.solution) == reg_nlp.model.meta.nvar
  @test typeof(stats_optimized.dual_feas) == eltype(stats_optimized.solution)
  @test stats_optimized.status == expected_status
  @test obj(reg_nlp, stats_optimized.solution) == stats_optimized.objective
  @test stats_optimized.objective <= obj(reg_nlp, x0)
  @test all(solver.mν∇fk + solver.∇fk/stats_optimized.solver_specific[:sigma_cauchy] .≤ eps(eltype(solver.mν∇fk)))

  # TODO: test that the optimized entries in stats_optimized and stats_basic are the same.

end
