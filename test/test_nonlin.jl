@testset "TRNC - Nonlinear Examples" begin 
# Julia Testing function
using TRNC
using DifferentialEquations, Zygote, DiffEqSensitivity
using Roots
using DataFrames
include("nonlin/nonlintable.jl")
include("nonlinfig_gen.jl")
include("fig_gen.jl")

	@testset "Lotka-Volterra: ||F(p) - b||² + λ||p||₁; ||⋅||₂≤Δ" begin

		println("Testing Lotka-Volterra; ||⋅||² + λ||⋅||₁; ||⋅||₂≤Δ")
		include("nonlin/test_lotka.jl")
		partest, objtest = LotkaVolt()

		# test against true values - note that these are operator-weighted (norm(x - x0))
		@test partest < .2 #20% error i guess 
		@test objtest < .2


	end

include("nonlin/nonlintablef.jl")
    @testset "Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₁; ||⋅||₂  ≤Δ" begin

        println("Testing Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₁; ||⋅||₂  ≤Δ")
        include("nonlin/test_FH_l1.jl")
		partest, objtest = FHNONLINl1()
				# test against true values - note that these are operator-weighted (norm(x - x0))
		@test partest < .15 #15% error i guess 
		@test objtest < .15


    end


    @testset "Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₀; ||⋅||_∞  ≤Δ" begin

        println("Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₀; ||⋅||_∞  ≤Δ")
        include("nonlin/test_FN_l0.jl")

        num_runs = 0
        partest = 10
        objtest = Float64
        while num_runs<10 && partest > .15
            partest, objtest = FHNONLINl0()
            num_runs+=1
        end
        @printf("Non-CVX problem required %1.2d runs\n", num_runs)
        # test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)

        @test num_runs < 9
		@test partest < .15 #10% error I guess 
		@test objtest < .15



    end

    @testset "Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₁; ||⋅||₂  ≤Δ" begin
        println("Testing with BFGS Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₁; ||⋅||₂  ≤Δ")
		include("nonlin/test_FH_l1_bfgs.jl")

        partest, objtest  = FHNONLINl1LBFGS()

        # test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)
		@test partest < 1.0
		@test objtest < 1.0



    end
end