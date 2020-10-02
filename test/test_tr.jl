@testset "TRNC - Linear/BPDN Examples ($compound)" for compound=1
	using LinearAlgebra
	using TRNC 
	using Plots, Roots

	# # m,n = compound*25, compound*64
	m,n = compound*200,compound*512
	k = compound*10
	A = 5*randn(m,n)
	x0  = rand(n,)
	b0 = A*x0
	b = b0 + 0.05*randn(m,)

	@testset "LS, h=0; full hessian" begin

		include("LS/test_LS_nobarrier.jl")
		partest, objtest = LSnobar(A, x0, b, b0, compound)

		# test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)
		@test partest < .01 #15% error i guess 
		@test objtest < .01


	end

	@testset "LS, h=0; bfgs" begin

		include("LS/test_LS_nobarrier_lbfgs.jl")
		partest, objtest = LSnobarBFGS(A, x0, b, b0, compound)

		# test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)
		@test partest < .01 #15% error i guess 
		@test objtest < .01

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
		partest, objtest  = bpdnNoBarTrBinf(A, x0, b, b0, compound)

		# test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)
		@test partest < .01 #15% error i guess 
		@test objtest < .01

	end

	@testset "LS, h=l1; b2" begin


		include("BPDN/test_bpdn_nobarrier_trB2.jl")
		partest, objtest = bpdnNoBarTrB2(A, x0, b, b0, compound)

		# test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)
		@test partest < .01 #15% error i guess 
		@test objtest < .01

	end


	@testset "LS, h=l0; binf" begin

		include("BPDN/test_bpdnl0_nobarrier_trBinf.jl")
		ncvx_test = 0
		num_runs = 0
		partest = Array{Float64, 1}
		objtest = Array{Float64, 1}
		while ncvx_test==0 && num_runs < 10
			partest, objtest = bpdnNoBarTrl0Binf(A, x0, b, b0, compound)
			if partest < .1
				ncvx_test=1
			end
			num_runs+=1
		end
		@printf("Non-CVX problem required %1.2d runs\n", num_runs)
		@test num_runs < 9 
		@test partest < .01 #15% error i guess 
		@test objtest < .01


	end

	@testset "LS, h=B0; binf" begin


		include("BPDN/test_bpdnB0_nobarrier_trBinf.jl")
		ncvx_test = 0
		num_runs = 0
		partest = Array{Float64, 1}
		objtest = Array{Float64, 1}
		while ncvx_test==0 && num_runs < 9
			partest, objtest = bpdnNoBarTrB0Binf(A, x0, b, b0, compound, k)
			if partest < .1
				ncvx_test=1
			end
			num_runs+=1
		end
		@printf("Non-CVX problem required %1.2d runs\n", num_runs)
		@test num_runs < 9 
		@test partest < .01 #15% error i guess 
		@test objtest < .01



	end
end