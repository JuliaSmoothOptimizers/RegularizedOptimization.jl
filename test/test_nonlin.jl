using DifferentialEquations, Zygote, DiffEqSensitivity
using Random, LinearAlgebra, TRNC, Printf,Roots, Plots
using ProximalOperators, ProximalAlgorithms, LinearOperators
include("Nonlin_table.jl")
include("nonlin/test_FN_l0.jl")
include("nonlinfig_gen.jl")
# include("modded_panoc.jl")

# @testset "TRNC - Nonlinear Examples" begin 
# Julia Testing function
#Here we solve the Fitzhugh-Nagumo (FHN) Model with some extra terms we know to be zero
	#The FHN model is a set of coupled ODE's 
	#V' = (f(V) - W + I)/μ for f(V) = V - V^3 / 3
	#W' = μ(aV - bW + c) for μ = 0.08,  b = 0.8, c = 0.7

   #so we need a model solution, a gradient, and a Hessian of the system (along with some data to fit)
   function FH_ODE(dx, x, p, t)
		#p is parameter vector [I,μ, a, b, c]
		V,W = x 
		I, μ, a, b, c = p
		dx[1] = (V - V^3/3 -  W + I)/μ
		dx[2] = μ*(a*V - b*W+c)
	end


	u0 = [2.0; 0.0]
	tspan = (0.0, 20.0)
	savetime = .2

	pars_FH = [0.5, 0.08, 1.0, 0.8, 0.7]
	prob_FH = ODEProblem(FH_ODE, u0, tspan, pars_FH)


	#So this is all well and good, but we need a cost function and some parameters to fit. First, we take care of the parameters
	#We start by noting that the FHN model is actually the van-der-pol oscillator with some parameters set to zero
	#x' = μ(x - x^3/3 - y)
	#y' = x/μ -> here μ = 12.5
	#changing the parameters to p = [0, .08, 1.0, 0, 0]
	x0 = [0, .2, 1.0, 0, 0]
	prob_VDP = ODEProblem(FH_ODE, u0, tspan, x0)
	sol_VDP = solve(prob_VDP,reltol=1e-6, saveat=savetime)


	#also make some noie to fit later
	t = sol_VDP.t
	b = hcat(sol_VDP.u...)
	noise = .1*randn(size(b))
	data = noise + b

	#so now that we have data, we want to formulate our optimization problem. This is going to be 
	#min_p ||f(p) - b||₂^2 + λ||p||₀
	#define your smooth objective function
	#First, make the function you are going to manipulate
	function Gradprob(p)
		temp_prob = remake(prob_FH, p = p)
		temp_sol = solve(temp_prob, reltol=1e-6, saveat=savetime, verbose=false)
		tot_loss = 0.

		if any((temp_sol.retcode!= :Success for s in temp_sol))
			tot_loss = Inf
		else
			temp_v = convert(Array, temp_sol)

			tot_loss = sum((temp_v - data).^2)/2
		end

		return tot_loss
	end
	function f_obj(x) #gradient and hessian info are smooth parts, m also includes nonsmooth part
		fk = Gradprob(x)
		# @show fk
		if fk==Inf 
			grad = Inf*ones(size(x))
			Hess = Inf*ones(size(x,1), size(x,1))
		else
			grad = Zygote.gradient(Gradprob, x)[1] 
			# Hess = Zygote.hessian(Gradprob, x)
		end

		return fk, grad
	end



	ϵ = 1e-2
	MI = 10
	TOL = 1e-10
	λ = 1.0 
	#set all options
	Doptions=s_options(1.0; λ=λ, optTol = TOL, verbose = 0)
	options = TRNCoptions(;verbose=0, ϵD=ϵ, maxIter = MI)
	solver = ProximalAlgorithms.PANOC(tol = ϵ, verbose=true, freq=1, maxit=MI)

	function A(x, ξ)
		if ξ!=0
			s = solve(remake(prob_FH, p = x), reltol=1e-6, saveat=savetime)
		else
			s = data
		end
	end

	# @testset "Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₀; ||⋅||_∞  ≤Δ" begin

		@info "Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₀; ||⋅||_∞  ≤Δ"
		folder = "figs/nonlin/FH/l0/"

		function h_obj(x)
			return norm(x,0) 
		end
		#this is for l0 norm 
		function prox(q, σ, xk, Δ)

			ProjB(y) = min.(max.(y, xk.-Δ),xk.+Δ) # define outside? 
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
			s = ProjB(st) - xk
			return s 
		end

		params= TRNCstruct(f_obj, h_obj, λ; FO_options = Doptions, s_alg=PG, ψχprox=prox, χk=(s)->norm(s, Inf), HessApprox = LSR1Operator)

		#put in your initial guesses
		xi = ones(size(pars_FH))


		# ϕ = LeastSquaresObjective(f_obj, data)
		g = NormL0(λ)
		ϕ = f_obj 

		partest, objtest = FHNONLIN(x0,xi, A, f_obj, h_obj,ϕ,g,λ,params, options, solver, folder, "fhl0")

		# test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)
		@test partest < .15 #10% error I guess 
		@test objtest < .15



	# end

	# @testset "Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + χ_{||p||₀≤δ}; ||⋅||_∞  ≤Δ" begin

	# 	println("Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + χ_{||p||₀≤δ}; ||⋅||_∞  ≤Δ")
	# 	include("nonlin/test_FN_B0.jl")

	# 	num_runs = 0
	# 	partest = 10
	# 	objtest = Float64
	# 	while num_runs<10 && partest > .15
	# 		partest, objtest = FHNONLINB0()
	# 		num_runs+=1
	# 	end
	# 	@printf("Non-CVX problem required %1.2d runs\n", num_runs)
	# 	# test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)

	# 	@test num_runs < 9
	# 	@test partest < .15 #10% error I guess 
	# 	@test objtest < .15
	# end


	# @testset "Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + χ_{||p||₀≤δ}; ||⋅||_∞  ≤Δ" begin

	# 	println("Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + χ_{||p||₀≤δ}; ||⋅||_∞  ≤Δ")
	# 	include("nonlin/test_FN_B0_bfgs.jl")

	# 	num_runs = 0
	# 	partest = 10
	# 	objtest = Float64
	# 	while num_runs<10 && partest > .15
	# 		partest, objtest = FHNONLINB0BFGS()
	# 		num_runs+=1
	# 	end
	# 	@printf("Non-CVX problem required %1.2d runs\n", num_runs)
	# 	# test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)

	# 	@test num_runs < 9
	# 	@test partest < .15 #10% error I guess 
	# 	@test objtest < .15

	# end

	# @testset "Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₁; ||⋅||₂  ≤Δ" begin
	# 	println("Testing with BFGS Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₁; ||⋅||₂  ≤Δ")
	# 	include("nonlin/test_FH_l1_bfgs.jl")

	# 	partest, objtest  = FHNONLINl1LBFGS()

	# 	# test against true values - note that these are operator-weighted (norm(x - x0)/opnorm(A)^2)
	# 	@test partest < 1.0
	# 	@test objtest < 1.0



	# end
# end