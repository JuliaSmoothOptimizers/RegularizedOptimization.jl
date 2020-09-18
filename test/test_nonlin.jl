@testset "TRNC - Nonlinear Examples" begin 
# Julia Testing function
using TRNC
using DifferentialEquations, Zygote, DiffEqSensitivity
using Roots
using DiffEqParamEstim, Optim, PenaltyFunctions, DataFrames
include("nonlintable.jl")

    @testset "Lotka-Volterra: ||F(p) - b||² + λ||p||₁; ||⋅||₂≤Δ" begin

        println("Testing Lotka-Volterra; ||⋅||² + λ||⋅||₁; ||⋅||₂≤Δ")
        include("nonlin/test_lotka.jl")
        partest, objtest = LotkaVolt()

        # test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)
        @test partest < .15 #15% error i guess 
        @test objtest < .15


    end


#     @testset "Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₁; ||⋅||₂  ≤Δ" begin

#         println("Testing Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₁; ||⋅||₂  ≤Δ")
#         include("nonlin/test_FH_l1.jl")
#         p, ptrue, objtest, ftest, htest, fp, fpt  = FHNONLINl1()

#         # test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)
#         @test norm(ptrue - p) < .1 #10% error i guess 
#         @show p
#         @show ptrue
#         @show fp 
#         @show fpt 
#         @show norm(ptrue - p)

#         @test abs(objtest) < .05

#         @test abs(ftest) < .05

#         @test abs(htest)<.1


#     end


#     @testset "Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₀; ||⋅||_∞  ≤Δ" begin

#         println("Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₀; ||⋅||_∞  ≤Δ")
#         include("nonlin/test_FN_l0.jl")

#         num_runs = 0
#         htest = 5
#         p = Array{Float64}
#         ptrue = Array{Float64}
#         objtest = Float64
#         ftest = Float64
#         fp = 0.0
#         fpt = 0.0 
#         while num_runs<10 && abs(htest) > .05
#             p, ptrue, objtest, ftest, htest, fp, fpt  = FHNONLINl0()
#             num_runs+=1
#         end
#         @printf("Non-CVX problem required %1.2d runs\n", num_runs)
#         # test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)

#         @test num_runs < 9
#         @test norm(ptrue - p) < .1 #10% error I guess 
#         @show p
#         @show ptrue
#         @show fp 
#         @show fpt 
#         @show norm(ptrue - p)

#         @test abs(objtest) < .05

#         @test abs(ftest) < .05

#         @test abs(htest)<.05


#     end

#     @testset "Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₁; ||⋅||₂  ≤Δ" begin
#         println("Testing with BFGS Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₁; ||⋅||₂  ≤Δ")
#         include("nonlin/test_FH_l1_bfgs.jl")
#         p, ptrue, objtest, ftest, htest, fp, fpt  = FHNONLINl1LBFGS()

#         # test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)
#         @test norm(ptrue - p) < .05 #5% error i guess 
#         @show p
#         @show ptrue
#         @show fp 
#         @show fpt 
#         @show norm(ptrue - p)

#         @test abs(objtest) < .05

#         @test abs(ftest) < .05

#         @test abs(htest)<.05


#     end
end