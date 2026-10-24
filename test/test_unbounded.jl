# Tests of the :unbounded flag of R2 and of its use by R2N (subproblem unbounded below)
using LinearOperators, QuadraticModels

# f(x) = -½‖x‖² + ½‖x‖₁ is unbounded below: R2 must flag it instead of iterating to overflow
@testset "R2 flags an objective unbounded below" begin
  n = 5
  qm = QuadraticModel(zeros(n), LinearOperator(-Matrix(1.0I, n, n)), x0 = ones(n))
  stats = R2(RegularizedNLPModel(qm, NormL1(0.5)), atol = 1e-6, rtol = 1e-6, max_iter = 10000)
  @test stats.status == :unbounded
end

# R2N with an indefinite Hessian: f(x) = Σ(xᵢ² - 1)² is bounded below.
@testset "R2N rejects the step when its subproblem is unbounded" begin
  n = 10
  x0 = [isodd(i) ? sqrt(1 / 6) : sqrt(1 / 2) for i = 1:n]
  nlp = ADNLPModel(x -> sum((x .^ 2 .- 1) .^ 2), x0)
  out = R2N(RegularizedNLPModel(nlp, NormL1(0.1)), atol = 1e-6, rtol = 1e-6)
  @test out.status == :first_order
end
