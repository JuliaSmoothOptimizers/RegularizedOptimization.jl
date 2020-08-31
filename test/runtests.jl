
using Test
using LinearAlgebra, Random

Random.seed!(0)

@testset "TRNC" begin

@testset "Descent Methods" begin
  include("test_proxalgs.jl")
end

# @testset "LASSO" begin
#   include("test_variables.jl")
#   include("test_expressions.jl")
#   include("test_AbstractOp_binding.jl")
#   include("test_terms.jl")
# end

# @testset "Nonlinear Problems" begin
#   include("test_problem.jl")
#   include("test_build_minimize.jl")
# end

end
