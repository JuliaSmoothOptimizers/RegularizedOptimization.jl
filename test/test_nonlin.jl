@testset "TRNC - Nonlinear Examples" begin 
# Julia Testing function
using TRNC
using LinearAlgebra
using DifferentialEquations, Zygote, DiffEqSensitivity
using Printf, Roots, Plots

    @testset "Lotka-Volterra: ||F(p) - b||² + λ||p||₁; ||⋅||₂≤Δ" begin

        println("Testing Lotka-Volterra; ||⋅||² + λ||⋅||₁; ||⋅||₂≤Δ")
        include("nonlin/test_lotka.jl")
        p, ptrue, objtest, ftest, htest = LotkaVolt()

        # test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)
        @test norm(ptrue - p) < .05 #5% error i guess 

        @test abs(objtest) < .05
    
        @test abs(ftest) < .05
    
        @test abs(htest)<.05


    end

    @testset "Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₁; ||⋅||₂  ≤Δ" begin

        println("Testing Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₁; ||⋅||₂  ≤Δ")
        include("nonlin/test_FH_l1.jl")
        p, ptrue, objtest, ftest, htest = FHNONLINl1()

        # test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)
        @test norm(ptrue - p) < .05 #5% error i guess 

        @test abs(objtest) < .05

        @test abs(ftest) < .05

        @test abs(htest)<.05


    end


    @testset "Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₀; ||⋅||_∞  ≤Δ" begin

        println("Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₀; ||⋅||_∞  ≤Δ")
        include("nonlin/test_FN_l0.jl")
        p, ptrue, objtest, ftest, htest = FHNONLINl0()

        # test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)
        @test norm(ptrue - p) < .05 #5% error i guess 

        @test abs(objtest) < .05

        @test abs(ftest) < .05

        @test abs(htest)<.05


    end

    @testset "Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₁; ||⋅||₂  ≤Δ" begin
        println("Testing with BFGS Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₁; ||⋅||₂  ≤Δ")
        include("nonlin/test_FH_l1_bfgs.jl")
        p, ptrue, objtest, ftest, htest = FHNONLINl1LBFGS()

        # test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)
        @test norm(ptrue - p) < .05 #5% error i guess 

        @test abs(objtest) < .05

        @test abs(ftest) < .05

        @test abs(htest)<.05


    end
end