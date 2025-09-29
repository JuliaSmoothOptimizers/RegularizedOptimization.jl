compound = 1
nz = 10 * compound
bpdn, bpdn_nls, sol = bpdn_model(compound)
bpdn_bounded, bpdn_nls_bounded, sol_bounded = bpdn_model(compound, bounds = true)
λ = norm(grad(bpdn, zeros(bpdn.meta.nvar)), Inf) / 10
x0 = 10*randn(bpdn.meta.nvar)

hessian_modifiers = [LSR1Model, LBFGSModel, SpectralGradientModel, DiagonalPSBModel] # TODO: should add exact hessians once we implement hess_op for bpdn
regularizers = [NormL0(λ), NormL1(λ), NormL2(λ), IndBallL0(10 * compound)]
bounded_regularizers = [NormL0(λ), NormL1(λ)]

@testset "BPDN" verbose=true begin
  @testset "NLP" begin
    @testset "R2" begin
      # Test on unbounded problem
      constructor_parameters_set = [NamedTuple(), NamedTuple(), NamedTuple(), NamedTuple(), NamedTuple()]
      solve_parameters_set = [(ν = 1.0, atol = 1e-6, rtol = 1e-6, x = x0, verbose = 10),
                              (ν = 1.0, atol = 1e-6, rtol = 1e-6, x = x0, verbose = 10, max_iter = 1,),
                              (ν = 1.0, atol = 1e-6, rtol = 1e-6, x = x0, verbose = 10, max_time = -1.0,),
                              (ν = 1.0, atol = 1e-6, rtol = 1e-6, x = x0, verbose = 10, max_eval = 2,),
                              (ν = 1.0, atol = 1e-6, rtol = 1e-6, x = x0, verbose = 10, callback = basic_callback,)]
      expected_output = [:first_order, :max_iter, :max_time, :max_eval, :user]
      for h in regularizers
        reg_nlp = RegularizedNLPModel(bpdn, h)
        @testset "R2-unbounded-$(typeof(h))" begin
          for (solver_kwargs, constructor_kwargs, expected_output) in zip(solve_parameters_set, constructor_parameters_set, expected_output)
            test_solver_basic(reg_nlp, :R2; expected_output = expected_output, constructor_kwargs = constructor_kwargs, solver_kwargs = solver_kwargs)
          end
        end
      end

      # Test on bounded problem
      for h in bounded_regularizers
        reg_nlp = RegularizedNLPModel(bpdn_bounded, h)
        @testset "R2-bounded-$(typeof(h))" begin
          for (solver_kwargs, constructor_kwargs, expected_output) in zip(solve_parameters_set, constructor_parameters_set, expected_output)
            test_solver_basic(reg_nlp, :R2; expected_output = expected_output, constructor_kwargs = constructor_kwargs, solver_kwargs = solver_kwargs)
          end
        end
      end
    end

    @testset "R2N" begin
      # Test on unbounded problem
      constructor_parameters_set = [NamedTuple(), (subsolver = R2DHSolver,)]
      solve_parameters_set = [(σk = 1.0, atol = 1e-6, rtol = 1e-6, x = x0, verbose = 10),
                              (σk = 1.0, atol = 1e-6, rtol = 1e-6, x = x0, verbose = 10, max_iter = 1,),
                              (σk = 1.0, atol = 1e-6, rtol = 1e-6, x = x0, verbose = 10, max_time = -1.0,),
                              (σk = 1.0, atol = 1e-6, rtol = 1e-6, x = x0, verbose = 10, max_eval = 2,),
                              (σk = 1.0, atol = 1e-6, rtol = 1e-6, x = x0, verbose = 10, callback = basic_callback,)]
      expected_output = [:first_order, :max_iter, :max_time, :max_eval, :user]
      for modifier in hessian_modifiers
        for h in regularizers
          reg_nlp = RegularizedNLPModel(modifier(bpdn), h)
          @testset "R2N-unbounded-$modifier-$(typeof(h))" begin
            for constructor_kwargs in constructor_parameters_set
              for (solver_kwargs, expected_output) in zip(solve_parameters_set, expected_output)
                test_solver_basic(reg_nlp, :R2N; expected_output = expected_output, constructor_kwargs = constructor_kwargs, solver_kwargs = solver_kwargs)
              end
            end
          end
        end
      end

      # Test on bounded problem
      for modifier in hessian_modifiers
        for h in bounded_regularizers
          reg_nlp = RegularizedNLPModel(bpdn_bounded, h)
          @testset "R2N-bounded-$modifier-$(typeof(h))" begin
            for constructor_kwargs in constructor_parameters_set
              for (solver_kwargs, expected_output) in zip(solve_parameters_set, expected_output)
                test_solver_basic(reg_nlp, :R2N; expected_output = expected_output, constructor_kwargs = constructor_kwargs, solver_kwargs = solver_kwargs)
              end
            end
          end
        end
      end
    end

    @testset "TR" begin
      # Test on unbounded problem
      constructor_parameters_set = [NamedTuple(), (subsolver = TRDHSolver,)]
      solve_parameters_set = [(Δk = 1.0, atol = 1e-6, rtol = 1e-6, x = x0, verbose = 10),
                              (Δk = 1.0, atol = 1e-6, rtol = 1e-6, x = x0, verbose = 10, max_iter = 1,),
                              (Δk = 1.0, atol = 1e-6, rtol = 1e-6, x = x0, verbose = 10, max_time = -1.0,),
                              (Δk = 1.0, atol = 1e-6, rtol = 1e-6, x = x0, verbose = 10, max_eval = 2,),
                              (Δk = 1.0, atol = 1e-6, rtol = 1e-6, x = x0, verbose = 10, callback = basic_callback,)]
      expected_output = [:first_order, :max_iter, :max_time, :max_eval, :user]
      for modifier in hessian_modifiers
        for h in regularizers
          reg_nlp = RegularizedNLPModel(modifier(bpdn), h)
          @testset "TR-unbounded-$modifier-$(typeof(h))" begin
            for constructor_kwargs in constructor_parameters_set
              for (solver_kwargs, expected_output) in zip(solve_parameters_set, expected_output)
                test_solver_basic(reg_nlp, :TR; expected_output = expected_output, constructor_kwargs = constructor_kwargs, solver_kwargs = solver_kwargs)
              end
            end
          end
        end
      end

      # Test on bounded problem
      for modifier in hessian_modifiers
        for h in bounded_regularizers
          reg_nlp = RegularizedNLPModel(bpdn_bounded, h)
          @testset "R2N-bounded-$modifier-$(typeof(h))" begin
            for constructor_kwargs in constructor_parameters_set
              for (solver_kwargs, expected_output) in zip(solve_parameters_set, expected_output)
                test_solver_basic(reg_nlp, :TR; expected_output = expected_output, constructor_kwargs = constructor_kwargs, solver_kwargs = solver_kwargs)
              end
            end
          end
        end
      end
    end
  end
end