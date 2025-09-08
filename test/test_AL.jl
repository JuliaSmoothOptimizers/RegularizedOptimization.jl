problem_list = ["HS8",]

@testset "Augmented Lagrangian" begin
  for problem_name in problem_list
    nlp = CUTEstModel(problem_name)
    for h in (NormL1(1.0), NormL2(1.0))
      stats = AL(nlp, h, atol = 1e-3, verbose = 1)
      @test stats.status == :first_order
      @test stats.primal_feas <= 1e-2
      @test stats.dual_feas <= 1e-2
      @test length(stats.solution) == nlp.meta.nvar
      @test typeof(stats.solution) == typeof(nlp.meta.x0)
    end
    finalize(nlp)
  end
end
