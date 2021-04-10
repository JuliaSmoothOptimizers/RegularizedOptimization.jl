# Implements Algorithm 4.2 in "Interior-Point Trust-Region Method for Composite Optimization".

using LinearOperators, LinearAlgebra, Arpack
export TR

"""Interior method for Trust Region problem
	TR(x, params, options)
Arguments
----------
x : Array{Float64,1}
	Initial guess for the x value used in the trust region
params : mutable structure TR_params with:
	--
	-ϵ, tolerance for primal convergence
	-Δk Float64, trust region radius
	-verbose Int, print every # options
	-maxIter Float64, maximum number of inner iterations (note: does not influence TotalCount)
options : mutable struct TR_methods
	-f_obj, smooth objective struct; eval, grad, Hess
	-h_obj, nonsmooth objective struct; h, ψ, ψχprox - function projecting onto the trust region ball or ψ+χ
	--
	-FO_options, options for first order algorithm, see DescentMethods.jl for more
	-s_alg, algorithm for descent direction, see DescentMethods.jl for more
	-χk, function that is the TR norm ball; defaults to l2 

Returns
-------
x   : Array{Float64,1}
	Final value of Algorithm 4.2 trust region
k   : Int
	number of iterations used
Fobj_hist: Array{Float64,1}
	smooth function history 
Hobj_hist: Array{Float64, 1}
	nonsmooth function history
Complex_hist: Array{Float64, 1}
	inner algorithm iteration count 

"""
function TR(
	x0,
	params,
	options)

	# initialize passed options
	ϵ = options.ϵ
	Δk = options.Δk
	verbose = options.verbose
	maxIter = options.maxIter
	η1 = options.η1
	η2 = options.η2 
	γ = options.γ
	τ = options.τ
	θ = options.θ
	β = options.β
	mem = options.mem

	if verbose == 0
		ptf = Inf
	elseif verbose == 1
		ptf = round(maxIter / 10)
	elseif verbose == 2
		ptf = round(maxIter / 100)
	else
		ptf = 1
	end

	# other parameters
	FO_options = params.FO_options
	s_alg = params.s_alg
	χk = params.χk 
	f_obj = params.f
	h_obj = params.h


	# initialize parameters
	xk = copy(x0)

	k = 0
	Fobj_hist = zeros(maxIter)
	Hobj_hist = zeros(maxIter)
	Complex_hist = zeros(maxIter)
	headerstr = "-"^155
	verbose != 0 && @printf(
		"%s\n", headerstr
	)
	verbose != 0 && @printf(
		"%10s | %10s | %11s | %12s | %10s | %10s | %11s | %10s | %10s | %10s | %9s | %9s\n",
		"Iter",
		"PG-Iter",
		"‖Gν‖",
		"Ratio: ρk",
		"x status ",
		"TR: Δk",
		"Δk status",
		"‖x‖   ",
		"‖s‖   ",
		"‖Bk‖   ",
		"f(x)   ",
		"h(x)   ",
	)
	verbose != 0 && @printf(
		"%s\n", headerstr
	)

	k = 0
	ρk = -1.0
	α = 1.0
	TR_stat = ""
	x_stat = ""

	# main algorithm initialization
	# test to see if user provided a hessian
	quasiNewtTest = (f_obj.Hess == LSR1Operator) || (f_obj.Hess==LBFGSOperator)
	if quasiNewtTest
		Bk = f_obj.Hess(size(xk, 1); mem=mem)
	else
		Bk = f_obj.Hess(xk)
	end
	# define the Hessian 
	H = Symmetric(Matrix(Bk))
	# νInv = eigmax(H) #make a Matrix? ||B_k|| = λ(B_k) # change to opNorm(Bk, 2), arPack? 
	νInv = (1 + θ) * maximum(abs.(eigs(H;nev=1, which=:LM)[1]))

	# keep track of old values, initialize functions
	∇fk =  f_obj.grad(xk)
	fk = f_obj.eval(xk)
	hk = h_obj.eval(xk)
	s = zeros(size(xk))
	xk⁻ = xk 
	Gν = ∇fk*νInv
	∇fk⁻ = ∇fk
	funEvals = 1

	kktNorm = χk(Gν)^2 / νInv
	optimal = kktNorm < ϵ

	while !optimal && k < maxIter && Δk ≥ 1e-16 
		# update count
		k = k + 1 


		Fobj_hist[k] = fk
		Hobj_hist[k] = hk
		# Print values
		k % ptf == 0 && 
		@printf(
			"%11d|  %9d |  %10.5e   %10.5e   %9s   %10.5e   %10s   %9.4e   %9.4e   %9.4e   %9.4e  %9.4e\n",
				k, funEvals,  kktNorm[1], ρk,   x_stat,  Δk, TR_stat,   χk(xk), χk(s), νInv,    fk,    hk )

		TR_stat = ""
		x_stat = ""

		# define inner function 
		φ(d) = [0.5 * (d' * H * d) + ∇fk' * d + fk, H * d + ∇fk, H] # (φ, ∇φ, ∇²φ)

		# define model and update ρ
		mk(d) = 0.5 * (d' * (H * d)) + ∇fk' * d + fk + h_obj.ψk(xk + d) # psik = h -> psik = h(x+d)

		# take initial step s1 and see if you can do more 
		FO_options.ν = min(1 / νInv, Δk)
		s1 = h_obj.ψχprox(-FO_options.ν * ∇fk, FO_options.λ * FO_options.ν, xk, Δk) # -> PG on one step s1
		Gν .= s1 * νInv
		χGν = χk(Gν)

		if kktNorm[1] > ϵ # final stopping criteria
			FO_options.optTol = min(.01, sqrt(χGν)) * χGν # stopping criteria for inner algorithm 
			FO_options.FcnDec = fk + hk - mk(s1)
			(s, hist, funEvals) = s_alg(φ, (d) -> h_obj.ψk(xk + d), s1, (d, λν) -> h_obj.ψχprox(d, λν, xk, min(β * χk(s1), Δk)), FO_options)
			Gν .= s / FO_options.ν
		else
			s .= s1
			funEvals = 1 
		end

		# update Complexity history 
		Complex_hist[k] += funEvals # doesn't really count because of quadratic model 

		fkn = f_obj.eval(xk + s)
		hkn = h_obj.eval(xk + s)

		Numerator = fk + hk - (fkn + hkn)
		Denominator = fk + hk - mk(s)

		ρk = (Numerator + 1e-16) / (Denominator + 1e-16)

		if ρk > η2 
			TR_stat = "increase"
			Δk = max(Δk, γ * χk(s))
			# Δk = γ*Δk
		else
			TR_stat = "kept"
		end

		if ρk >= η1 
			x_stat = "update"
			xk .+= s

			#update functions
			fk = fkn
			hk = hkn

			#update gradient & hessian 
			∇fk = f_obj.grad(xk)
			if quasiNewtTest
				push!(Bk, s, ∇fk - ∇fk⁻)
			else
				Bk = f_obj.Hess(xk)
			end

			Complex_hist[k] += 1
		end

		if ρk < η1
			x_stat = "reject"
			TR_stat = "shrink"
			α = .5
			Δk = α * Δk	# * norm(s, Inf) #change to reflect trust region 
		end

		# define the Hessian 
		H = Symmetric(Matrix(Bk))
		# β = eigmax(H) #make a Matrix? ||B_k|| = λ(B_k) # change to opNorm(Bk, 2), arPack? 
		νInv = (1 + θ) * maximum(abs.(eigs(H;nev=1, which=:LM)[1]))
		
		# update Gν with new direction
		if x_stat == "update"
			# kktNorm = χk(Gν)^2/νInv
			kktNorm = Denominator
		end
		
		# store previous iterates
		xk⁻ .= xk 
		∇fk⁻ .= ∇fk
		optimal = kktNorm<ϵ


	end

	return xk, k, Fobj_hist[Fobj_hist .!= 0], Hobj_hist[Fobj_hist .!= 0], Complex_hist[Complex_hist .!= 0]
end