
using Test
using LinearAlgebra, Random, Printf

Random.seed!(0)

@testset "TRNC" begin

@testset "Descent Methods" begin
#   include("test_proxalgs.jl")
end

@testset "Hard Prox Computations" begin 
    # include("test_hardprox.jl")
end

@testset "LASSO" begin
  include("test_tr.jl")
end

# @testset "Nonlinear Problems" begin
#   include("test_problem.jl")
#   include("test_build_minimize.jl")
# end

# @testset "Various Utilities" begin 
#     include("test_utilities.jl")
# end
end
