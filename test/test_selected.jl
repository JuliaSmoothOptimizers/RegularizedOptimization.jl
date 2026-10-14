# Regularize only some of the variables of a problem without bounds.
# Solvers used to apply the prox of h to all variables in that case.
using ShiftedProximalOperators: Nuclearnorm

@testset "selected variables without bounds" begin
  # minimize ½‖x - a‖² + λ‖x[selected]‖₁,
  # whose solution is the soft-thresholding of a on `selected` and a elsewhere.
  @testset "NormL1" begin
    a = [2.0, -0.3, 2.0, -0.3]
    λ = 0.5
    selected = 1:2
    xstar = [1.5, 0.0, 2.0, -0.3]
    objstar = sum((xstar .- a) .^ 2) / 2 + λ * sum(abs, xstar[selected])
    h = NormL1(λ)

    nlp = ADNLPModel(x -> sum((x .- a) .^ 2) / 2, zeros(4))
    @test !has_bounds(nlp)
    # selecting variables does not create bounds
    @test !R2Solver(RegularizedNLPModel(nlp, h, selected)).has_bnds

    # the other tests run TR with quasi-Newton models only
    for (solver, model) ∈
        ((R2, nlp), (R2N, nlp), (R2DH, nlp), (TR, LBFGSModel(nlp)), (TRDH, nlp))
      @testset "$(solver)" begin
        out = solver(RegularizedNLPModel(model, h, selected); atol = 1e-6, rtol = 1e-6)
        @test out.status == :first_order
        @test isapprox(out.solution, xstar, atol = 1e-4)
        @test isapprox(out.objective, objstar, atol = 1e-4)
      end
    end

    nls = ADNLSModel(x -> x .- a, zeros(4), 4)
    for solver ∈ (LM, LMTR)
      @testset "$(solver)" begin
        out = solver(RegularizedNLSModel(nls, h, selected); atol = 1e-6, rtol = 1e-6)
        @test out.status == :first_order
        @test isapprox(out.solution, xstar, atol = 1e-4)
      end
    end
  end

  # minimize ½‖x - a‖² + λ‖X‖_*, where X is the 2 × 2 matrix made of x[selected],
  # whose solution is the singular value thresholding of a on `selected` and a elsewhere.
  # The nuclear norm has no box variant, so trust-region methods do not support it with selected.
  @testset "Nuclearnorm" begin
    a = [3.0, 1.0, -1.0, 2.0, 0.5, -0.7]
    λ = 0.5
    selected = [1, 3, 4, 6]
    A = reshape(a[selected], 2, 2)
    F = svd(A)
    Astar = F.U * Diagonal(max.(F.S .- λ, 0)) * F.Vt
    xstar = copy(a)
    xstar[selected] .= vec(Astar)
    h = Nuclearnorm(λ, zeros(2, 2))

    nlp = ADNLPModel(x -> sum((x .- a) .^ 2) / 2, zeros(6))
    for solver ∈ (R2, R2N)
      @testset "$(solver)" begin
        out = solver(RegularizedNLPModel(nlp, h, selected); atol = 1e-6, rtol = 1e-6)
        @test out.status == :first_order
        @test isapprox(out.solution, xstar, atol = 1e-4)
      end
    end

    nls = ADNLSModel(x -> x .- a, zeros(6), 6)
    @testset "LM" begin
      out = LM(RegularizedNLSModel(nls, h, selected); atol = 1e-6, rtol = 1e-6)
      @test out.status == :first_order
      @test isapprox(out.solution, xstar, atol = 1e-4)
    end
  end
end
