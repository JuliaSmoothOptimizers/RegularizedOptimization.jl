function test_opnorm_upper_bound(B, m, n)
  T = eltype(B)
  is_upper_bound = true
  for _ = 1:m
    upper_bound, found =  RegularizedOptimization.opnorm_upper_bound(B)
    if opnorm(Matrix(B)) > upper_bound || !found
      is_upper_bound = false
      break
    end

    push!(B, randn(T, n), randn(T, n))
  end

  nallocs = @allocated RegularizedOptimization.opnorm_upper_bound(B)
  @test nallocs == 0
  @test is_upper_bound == true
end

# Test norm functions

@testset "Test opnorm upper bound functions" begin
  n = 10
  m = 40
  @testset "LBFGS" begin   
    B = LBFGSOperator(Float64, n, scaling = false)
    test_opnorm_upper_bound(B, m, n)

    B = LBFGSOperator(Float64, n, scaling = true)
    test_opnorm_upper_bound(B, m, n)

    B = LBFGSOperator(Float32, n, scaling = false)
    test_opnorm_upper_bound(B, m, n)

    B = LBFGSOperator(Float32, n, scaling = true)
    test_opnorm_upper_bound(B, m, n)
  end

  @testset "LSR1" begin
    B = LSR1Operator(Float64, n, scaling = false)
    test_opnorm_upper_bound(B, m, n)

    B = LSR1Operator(Float64, n, scaling = true)
    test_opnorm_upper_bound(B, m, n)

    B = LSR1Operator(Float32, n, scaling = false)
    test_opnorm_upper_bound(B, m, n)

    B = LSR1Operator(Float32, n, scaling = true)
    test_opnorm_upper_bound(B, m, n)
  end

  @testset "Diagonal" begin
    B = SpectralGradient(randn(Float64), n)
    test_opnorm_upper_bound(B, m, n)

    B = SpectralGradient(randn(Float32), n)
    test_opnorm_upper_bound(B, m, n)

    B = DiagonalPSB(randn(Float64, n))
    test_opnorm_upper_bound(B, m, n)

    B = DiagonalPSB(randn(Float32, n))
    test_opnorm_upper_bound(B, m, n)

    B = DiagonalAndrei(randn(Float64, n))
    test_opnorm_upper_bound(B, m, n)

    B = DiagonalAndrei(randn(Float32, n))
    test_opnorm_upper_bound(B, m, n)
  end
end