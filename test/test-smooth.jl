@testset "smooth NLP" begin
  @testset "unconstrained" begin
    for solver in [R2, R2N, R2DH, TR, TRDH]
      unconstrained_nlp(solver, atol = 1e-3, rtol = 1e-3)
    end
  end
  @testset "bound-constrained" begin
    for solver in [R2, R2N, R2DH, TR, TRDH]
      bound_constrained_nlp(solver, atol = 1e-3, rtol = 1e-3)
    end
  end
end

@testset "smooth NLS" begin
  @testset "unconstrained" begin
    for solver in [LM, LMTR]
      unconstrained_nls(solver, atol = 1e-3, rtol = 1e-3)
    end
  end
  @testset "bound-constrained" begin
    for solver in [LM, LMTR]
      bound_constrained_nls(solver, atol = 1e-3, rtol = 1e-3)
    end
  end
end