using Test, TRNC
using LinearAlgebra, Random

include("test_bpdn_nobarrier_tr.jl")

Random.seed!(0)

@testset "TRNC" begin
	@testset "Descent Methods" begin
		# include("test_proxalgs.jl")
	end

	@testset "Hard Prox Computations" begin 
		# include("test_hardprox.jl")
	end

	@testset "LASSO" begin
		include("test_tr.jl")
	end

	# @testset "Various Utilities" begin 
	#   include("test_utilities.jl")
	# end
end
