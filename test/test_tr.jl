@testset "TRNC - Linear/BPDN Examples ($compound)" for compound=1:3
    using LinearAlgebra
    using TRNC 
    using Plots, Roots
    using Convex, SCS

    # # m,n = compound*25, compound*64
    m,n = compound*200,compound*512
    k = compound*10
    A = 5*randn(m,n)
    x0  = rand(n,)
    b0 = A*x0
    b = b0 + 0.05*randn(m,)

    @testset "LS, h=0; full hessian" begin

        include("LS/test_LS_nobarrier.jl")
        trslim, tr, cvx, trs_v_cvx, tr_v_cvx = LSnobar(A, x0, b, b0, compound)

        # test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)
        @test trslim < .001
        @test tr < .001
        @test cvx > tr && cvx > trslim 
        @test trs_v_cvx<.001
        @test tr_v_cvx < .001


    end

    @testset "LS, h=0; bfgs" begin

        include("LS/test_LS_nobarrier_lbfgs.jl")
        trslim, tr, cvx, trs_v_cvx, tr_v_cvx = LSnobarBFGS(A, x0, b, b0, compound)

        # test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)
        @test trslim < .001
        @test tr < .001
        @test cvx > tr && cvx > trslim 
        @test trs_v_cvx<.001
        @test tr_v_cvx < .001


    end




    #start bpdn stuff 
    x0 = zeros(n)
    p   = randperm(n)[1:k]
    x0 = zeros(n,)
    x0[p[1:k]]=sign.(randn(k))

    A,_ = qr(randn(n,m))
    B = Array(A)'
    A = Array(B)

    b0 = A*x0
    b = b0 + 0.005*randn(m,)


    @testset "LS, h=l1; binf" begin

        #set up the problem 
        #initialize x

        

        include("BPDN/test_bpdn_nobarrier_trBinf.jl")
        xcomp, objcomp, fcomp, hcomp = bpdnNoBarTrBinf(A, x0, b, b0, compound)

        # test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)
        @test xcomp[1] < .1*compound #tr 
        @test abs(xcomp[2] - xcomp[1]) <.001 #should be close to PG - true 
        @test abs(xcomp[3] - xcomp[1]) <.1 #should be kind of close to CVX 
        @test abs(xcomp[4] - xcomp[1]) <.05 #should be close to CVX - true 

        #test objective values 
        @test abs(objcomp[1] - objcomp[4]) <.1 #tr close to true
        @test abs(objcomp[2] - objcomp[1]) <.1 #should be close to PG
        @test abs(objcomp[3] - objcomp[1]) <.1 #should be close to CVX 

        #test objective values 
        @test abs(fcomp[1] - fcomp[4]) <.1 #tr close to true
        @test abs(fcomp[2] - fcomp[1]) <.1 #should be close to PG
        @test abs(fcomp[3] - fcomp[1]) <.1 #should be close to CVX 

        #test objective values 
        @test abs(hcomp[1] - hcomp[4]) <1 #tr close to true
        @test abs(hcomp[2] - hcomp[1]) <1 #should be close to PG
        @test abs(hcomp[3] - hcomp[1]) <1*compound #should be close to CVX (scaled by number of data points)


    end

    @testset "LS, h=l1; b2" begin


        include("BPDN/test_bpdn_nobarrier_trB2.jl")
        xcomp, objcomp, fcomp, hcomp = bpdnNoBarTrB2(A, x0, b, b0, compound)

        # test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)
        @test xcomp[1] < .1*compound #tr 
        @test abs(xcomp[2] - xcomp[1]) <.001 #should be close to PG - true 
        @test abs(xcomp[3] - xcomp[1]) <.1 #should be kind of close to CVX 
        @test abs(xcomp[4] - xcomp[1]) <.05 #should be close to CVX - true 

        #test objective values 
        @test abs(objcomp[1] - objcomp[4]) <.1 #tr close to true
        @test abs(objcomp[2] - objcomp[1]) <.1 #should be close to PG
        @test abs(objcomp[3] - objcomp[1]) <.1 #should be close to CVX 

        #test objective values 
        @test abs(fcomp[1] - fcomp[4]) <.1 #tr close to true
        @test abs(fcomp[2] - fcomp[1]) <.1 #should be close to PG
        @test abs(fcomp[3] - fcomp[1]) <.1 #should be close to CVX 

        #test objective values 
        @test abs(hcomp[1] - hcomp[4]) <1 #tr close to true
        @test abs(hcomp[2] - hcomp[1]) <1 #should be close to PG
        @test abs(hcomp[3] - hcomp[1]) <1*compound #should be close to CVX 


    end


    @testset "LS, h=l0; binf" begin

        include("BPDN/test_bpdnl0_nobarrier_trBinf.jl")
        ncvx_test = 0
        num_runs = 0
        xcomp = Array{Float64, 1}
        objcomp = Array{Float64, 1}
        fcomp = Array{Float64, 1}
        hcomp = Array{Float64, 1}
        while ncvx_test==0 && num_runs < 10
            xcomp, objcomp, fcomp, hcomp = bpdnNoBarTrl0Binf(A, x0, b, b0, compound)
            if xcomp[1] < .1
                ncvx_test=1
            end
            num_runs+=1
        end
        @printf("Non-CVX problem required %1.2d runs\n", num_runs)
        @test num_runs < 9 
        # test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)
        @test xcomp[1] < .1 #tr 
        @test abs(xcomp[2] - xcomp[1]) <.5 #should be kind of close to CVX 
        @test abs(xcomp[3] - xcomp[1]) <.5 #should be close to CVX - true 

        #test objective values 
        @test abs(objcomp[1] - objcomp[3]) <.1 #tr close to true
        @test abs(objcomp[2] - objcomp[1]) <.1 #should be close to CVX 

        #test objective values 
        @test abs(fcomp[1] - fcomp[3]) <.1 #tr close to true
        @test abs(fcomp[2] - fcomp[1]) <.1 #should be close to CVX 

        #test objective values 
        @test abs(hcomp[1] - hcomp[3]) <.1 #tr close to true


    end

    # @testset "LS, h=B0; binf" begin


    #     include("BPDN/test_bpdnB0_nobarrier_trBinf.jl")
    #     ncvx_test = 0
    #     num_runs = 0
    #     xcomp = Array{Float64, 1}
    #     objcomp = Array{Float64, 1}
    #     fcomp = Array{Float64, 1}
    #     hcomp = Array{Float64, 1}
    #     while ncvx_test==0 && num_runs < 1
    #         xcomp, objcomp, fcomp, hcomp = bpdnNoBarTrB0Binf(A, x0, b, b0, compound, k)
    #         if xcomp[1] < .1
    #             ncvx_test=1
    #         end
    #         num_runs+=1
    #     end
    #     @printf("Non-CVX problem required %1.2d runs\n", num_runs)
    #     @test num_runs < 9 
    #     # test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)
    #     @test xcomp[1] < .1 #tr 

    #     #test objective values 
    #     @test abs(objcomp[1] - objcomp[2]) <.1 #tr close to true

    #     #test objective values 
    #     @test abs(fcomp[1] - fcomp[2]) <.1 #tr close to true

    #     #test objective values 
    #     @test abs(hcomp[1] - hcomp[2]) <.1 #tr close to true


    # end
end