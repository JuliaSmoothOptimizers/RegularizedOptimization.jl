@testset "R2N" begin
  # BASIC TESTS
  # Test basic NLP with 2-norm
  @testset "BASIC" begin
    rosenbrock_nlp = construct_rosenbrock_nlp()
    rosenbrock_reg_nlp = RegularizedNLPModel(rosenbrock_nlp, NormL2(0.01))

    # Test first order status
    first_order_kwargs = (atol = 1e-6, rtol = 1e-6)
    test_solver(
      rosenbrock_reg_nlp,
      "R2N",
      expected_status = :first_order,
      solver_kwargs = first_order_kwargs,
    )
    solver, stats = R2NSolver(rosenbrock_reg_nlp), RegularizedExecutionStats(rosenbrock_reg_nlp)

    # Test max time status
    max_time_kwargs = (x0 = [π, -π], atol = 1e-16, rtol = 1e-16, max_time = 1e-12)
    test_solver(
      rosenbrock_reg_nlp,
      "R2N",
      expected_status = :max_time,
      solver_kwargs = max_time_kwargs,
    )

    # Test max iter status
    max_iter_kwargs = (x0 = [π, -π], atol = 1e-16, rtol = 1e-16, max_iter = 1)
    test_solver(
      rosenbrock_reg_nlp,
      "R2N",
      expected_status = :max_iter,
      solver_kwargs = max_iter_kwargs,
    )

    # Test max eval status
    max_eval_kwargs = (x0 = [π, -π], atol = 1e-16, rtol = 1e-16, max_eval = 1)
    test_solver(
      rosenbrock_reg_nlp,
      "R2N",
      expected_status = :max_eval,
      solver_kwargs = max_eval_kwargs,
    )

    callback = (nlp, solver, stats) -> begin
      # We could add some tests here as well.
      
      # Check user status
      if stats.iter == 4
        stats.status = :user
      end
    end
    callback_kwargs = (x0 = [π, -π], atol = 1e-16, rtol = 1e-16, callback = callback)
    test_solver(
      rosenbrock_reg_nlp,
      "R2N",
      expected_status = :user,
      solver_kwargs = callback_kwargs,
    )
  end

  # BPDN TESTS
  # Test bpdn with L-BFGS and 1-norm
  @testset "BPDN" begin
    bpdn_kwargs = (x0 = zeros(bpdn.meta.nvar), σk = 1.0, β = 1e16, atol = 1e-6, rtol = 1e-6)
    reg_nlp = RegularizedNLPModel(LBFGSModel(bpdn), NormL1(λ))
    test_solver(reg_nlp, "R2N", expected_status = :first_order, solver_kwargs = bpdn_kwargs)
    solver, stats = R2NSolver(reg_nlp), RegularizedExecutionStats(reg_nlp)
    @test @wrappedallocs(
      solve!(solver, reg_nlp, stats, σk = 1.0, β = 1e16, atol = 1e-6, rtol = 1e-6)
    ) == 0

    #test_solver(reg_nlp,  # FIXME: divide by 0 error in the LBFGS approximation
    #            "R2N", 
    #            expected_status = :first_order,
    #            solver_kwargs=bpdn_kwargs,
    #            solver_constructor_kwargs=(subsolver=R2DHSolver,))

    # Test bpdn with L-SR1 and 0-norm
    reg_nlp = RegularizedNLPModel(LSR1Model(bpdn), NormL0(λ))
    test_solver(reg_nlp, "R2N", expected_status = :first_order, solver_kwargs = bpdn_kwargs)
    # FIXME: allocations fail with LSR1 -> a PR is awaiting on LinearOperators.jl
    # solver, stats = R2NSolver(reg_nlp), RegularizedExecutionStats(reg_nlp)
    # @test @wrappedallocs(
    #   solve!(solver, reg_nlp, stats, σk = 1.0, β = 1e16, atol = 1e-6, rtol = 1e-6)
    # ) == 0
     
    test_solver(
      reg_nlp,
      "R2N",
      expected_status = :first_order,
      solver_kwargs = bpdn_kwargs,
      solver_constructor_kwargs = (subsolver = R2DHSolver,),
    )
    # FIXME: allocations fail with LSR1 -> a PR is awaiting on LinearOperators.jl
    # solver, stats = R2NSolver(reg_nlp, subsolver = R2DHSolver), RegularizedExecutionStats(reg_nlp)
    # @test @wrappedallocs(
    #   solve!(solver, reg_nlp, stats, σk = 1.0, β = 1e16, atol = 1e-6, rtol = 1e-6)
    # ) == 0
  end
end
