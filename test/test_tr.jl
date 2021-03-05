

using ProximalOperators, ProximalAlgorithms
include("test_bpdn_nobarrier_tr.jl")
include("Lin_table.jl")
include("fig_gen.jl")
include("modded_panoc.jl")


@testset "TRNC - Linear/BPDN Examples ($compound)" for compound=1


	# # m,n = compound*25, compound*64
	m,n = compound*200,compound*512
	k = compound*10
	A = 5*randn(m,n)
	x0  = rand(n,)
	xi = zeros(size(x0))
	b0 = A*x0
	α = .01
	b = b0 + α*randn(m,)

	m,n= size(A)
	MI = 1000
	TOL = 1e-10
	λ = 1.0 
    #set all options
    Doptions = s_options(1/eigmax(A'*A); optTol = TOL, maxIter=MI, verbose=0)
	options = TRNCoptions(;verbose=0, ϵD=TOL, maxIter = MI)
	solver = ProximalAlgorithms.PANOC(tol = TOL, verbose=true, freq=1, maxit=MI)
    

	# @testset "LS, h=0" begin
	# 	folder = string("figs/ls/", compound, "/")

	# 	function f_obj(x)
	# 		f = .5*norm(A*x-b)^2
	# 		g = A'*(A*x - b)
	# 		return f, g, A'*A
	# 	end
	
	# 	function h_obj(x)
	# 		return 0
	# 	end
	# 	function tr_norm(z,σ, x, Δ)
	# 		return z./max(1, norm(z, 2)/Δ)
	# 	end

	# 	@info "running LS, h=0"
	# 	g = IndBallL2(100)
	# 	ϕ = LeastSquaresObjective((z)->[.5*norm(A*z-b)^2, A'*(A*z-b)], b)
	# 	parameters = TRNCstruct(f_obj, h_obj, λ; FO_options = Doptions, s_alg=PG, χk=(s)->norm(s, 2), ψχprox=tr_norm)
    
	# 	partest, objtest = bpdnNoBar(x0,xi, A, f_obj, h_obj,ϕ, g, λ, parameters, options, solver, folder, "ls")

	# 	# test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A))
	# 	@test partest < norm(x0)*.1 #same as noise levels 
	# 	@test objtest < α

	# end


	@testset "LS, h=0; bfgs" begin
		folder = string("figs/ls_bfgs/", compound, "/")
		xi = zeros(size(x0))
		function f_obj(x)
			f = .5*norm(A*x-b)^2
			g = A'*(A*x - b)
			return f, g
		end
	
		function h_obj(x)
			return 0
		end
		function tr_norm(z,σ, x, Δ)
			return z./max(1, norm(z, 2)/Δ)
		end

		@info "running LS bfgs, h=0"
		g = IndBallL2(100)
		ϕ = LeastSquaresObjective((z)->[norm(A*z-b)^2, A'*(A*z-b)], b)

		parameters = TRNCstruct(f_obj, h_obj, λ; FO_options = Doptions, s_alg=PG, χk=(s)->norm(s, 2), ψχprox=tr_norm)
    
		partest, objtest = bpdnNoBar(x0, xi, A, f_obj, h_obj,ϕ, g, λ, parameters, options, solver, folder, "ls")

		# test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A))
		@test partest < norm(x0)*.1 #50% x noise?  
		@test objtest < α

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
	b = b0 + α*randn(m,)
	λ = norm(A'*b, Inf)/100 #this can change around 

	Doptions.λ = λ
	Doptions.optTol=TOL
	@testset "LS, h=l1; binf" begin
		xi = zeros(size(x0))
		folder = string("figs/bpdn/LS_l1_Binf/", compound, "/")
		function f_obj(x)
			f = .5*norm(A*x-b)^2
			g = A'*(A*x - b)
			return f, g
		end
		function h_obj(x)
			return norm(x,1)
		end
		function prox(q, σ, xk, Δ)
			ProjB(wp) = min.(max.(wp,q.-σ), q.+σ)
			ProjΔ(yp) = min.(max.(yp, -Δ), Δ)
			s = ProjΔ(ProjB(-xk))
			return s
		end

		@info "running LS bfgs, h=l1, tr = linf"
		g = NormL1(λ)
		ϕ = LeastSquaresObjective((z)->[norm(A*z-b)^2, A'*(A*z-b)], b)

		parameters = TRNCstruct(f_obj, h_obj, λ; FO_options = Doptions, s_alg=PG, χk=(s)->norm(s, Inf), ψχprox=prox)
    
		partest, objtest = bpdnNoBar(x0, xi, A, f_obj, h_obj,ϕ, g, λ, parameters, options, solver, folder, "l1binf")

		# test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A))
		@test partest < norm(x0)*.1
		@test objtest < α

	end

	@testset "LS, h=l1, tr = l2" begin
		xi = zeros(size(x0))
		folder = string("figs/bpdn/LS_l1_B2/", compound, "/")

		function f_obj(x)
			f = .5*norm(A*x-b)^2
			g = A'*(A*x - b)
			return f, g
		end
		function h_obj(x)
			return norm(x,1)
		end
		function prox(q, σ, xk, Δ) #q = s - ν*g, ν*λ, xk, Δ - > basically inputs the value you need

			ProjB(y) = min.(max.(y, q.-σ), q.+σ)
			froot(η) = η - norm(ProjB((-xk).*(η/Δ)))
	
			# %do the 2 norm projection
			y1 = ProjB(-xk) #start with eta = tau
	
			if (norm(y1)<= Δ)
				y = y1  # easy case
			else
				η = fzero(froot, 1e-10, Inf)
				y = ProjB((-xk).*(η/Δ))
			end
			if (norm(y)<=Δ)
				snew = y
			else
				snew = Δ.*y./norm(y)
			end
			return snew
		end 

		@info "running LS bfgs, h=l1, tr = l2"
		g = NormL1(λ)
		ϕ = LeastSquaresObjective((z)->[norm(A*z-b)^2, A'*(A*z-b)], b)

		parameters = TRNCstruct(f_obj, h_obj, λ; FO_options = Doptions, s_alg=PG, χk=(s)->norm(s, Inf), ψχprox=prox)
    
		partest, objtest = bpdnNoBar(x0, xi, A, f_obj, h_obj,ϕ, g, λ, parameters, options, solver, folder, "l1b2")

		# test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A))
		@test partest < norm(x0)*.1 #5x noise
		@test objtest < α

	end


	@testset "LS, h=l0; binf" begin
		xi = zeros(size(x0))
		folder = string("figs/bpdn/LS_l0_Binf/", compound, "/")

		function f_obj(x)
			f = .5*norm(A*x-b)^2
			g = A'*(A*x - b)
			return f, g
		end
		function h_obj(x)
			return norm(x,0)
		end
		function prox(q, σ, xk, Δ)
			# @show σ/λ, λ
			c = sqrt(2*σ)
			w = xk+q
			st = zeros(size(w))
		
			for i = 1:length(w)
				absx = abs(w[i])
				if absx <=c
					st[i] = 0
				else
					st[i] = w[i]
				end
			end
			s = st - xk
			return s 
		end

		@info "running LS bfgs, h=l0, tr = linf"
		g = NormL0(λ)
		ϕ = LeastSquaresObjective((z)->[norm(A*z-b)^2, A'*(A*z-b)], b)

		parameters = TRNCstruct(f_obj, h_obj, λ; FO_options = Doptions, s_alg=PG, χk=(s)->norm(s, Inf), ψχprox=prox)
    
		partest, objtest = bpdnNoBar(x0, xi, A, f_obj, h_obj,ϕ, g, λ, parameters, options, solver, folder, "l0binf")

		@test partest < norm(x0)*.1 #5x noise
		@test objtest < α


	end


	λ = k
	@testset "LS, h=B0; binf" begin
		xi = zeros(size(x0))
		folder = string("figs/bpdn/LS_B0_Binf/", compound, "/")

		function f_obj(x)
			f = .5*norm(A*x-b)^2
			g = A'*(A*x - b)
			return f, g
		end
		function h_obj(x)
			if norm(x,0) ≤ λ
				h = 0
			else
				h = Inf
			end
			return h 
		end

		function prox(q, σ, xk, Δ)
			ProjB(w) = min.(max.(w, -Δ), +Δ)
			w = q + xk 
			#find largest entries
			p = sortperm(abs.(w), rev = true)
			w[p[λ+1:end]].=0 #set smallest to zero 
			s = ProjB(w-xk)#put all entries in projection?
			return s 
		end

		@info "running LS bfgs, h=B0, tr = linf"
		g = IndBallL0(λ)
		ϕ = LeastSquaresObjective((z)->[norm(A*z-b)^2, A'*(A*z-b)], b)

		parameters = TRNCstruct(f_obj, h_obj, λ; FO_options = Doptions, s_alg=PG, χk=(s)->norm(s, Inf), ψχprox=prox, HessApprox = LSR1Operator)
    
		partest, objtest = bpdnNoBar(x0, xi, A, f_obj, h_obj,ϕ, g, λ, parameters, options, solver, folder, "B0binf")

		@test partest < norm(x0)*.1 #5x noise
		@test objtest < α

	end
end