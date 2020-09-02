@testset "TRNC - Linear/BPDN Examples ($compound)" for compound=1:3
    using LinearAlgebra
    using TRNC 
    using Plots, Roots
    using Convex, SCS

    m,n = compound*25, compound*64
    p = randperm(n)
    k = compound*2

    @testset "LS, h=0" begin

        #set up the problem 
        A = rand(m,n)
        x0  = rand(n,)
        b0 = A*x0
        b = b0 + 0.5*rand(m,)
        cutoff = 0.0

        include("test_LS_nobarrier.jl")
        trslim, tr, cvx, trs_v_cvx, tr_v_cvx = LSnobar(A, x0, b, b0)

        # test against true values 
        @test trslim < .01
        @test tr < .01
        @test cvx > tr && cvx > trslim 
        @test trs_v_cvx<.01
        @test tr_v_cvx < .01


    end
end