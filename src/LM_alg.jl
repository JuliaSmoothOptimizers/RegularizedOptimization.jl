# Implements Algorithm 4.2 in "Interior-Point Trust-Region Method for Composite Optimization".

using LinearOperators, LinearAlgebra, Arpack
export LM

"""Interior method for Trust Region problem
	TR(x, params, options)
Arguments
----------
x : Array{Float64,1}
	Initial guess for the x value used in the trust region
params : mutable structure TR_params with:
	--
	-ϵD, tolerance for primal convergence
	-Δk Float64, trust region radius
	-verbose Int, print every # options
	-maxIter Float64, maximum number of inner iterations (note: does not influence TotalCount)
options : mutable struct TR_methods
	-f_obj, smooth objective function; takes in x and outputs [f, g, Bk]
	-h_obj, nonsmooth objective function; takes in x and outputs h
	--
	-FO_options, options for first order algorithm, see DescentMethods.jl for more
	-s_alg, algorithm for descent direction, see DescentMethods.jl for more
	-ψχprox, function projecting onto the trust region ball or ψ+χ
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
function LM(
	x0,
	params,
	options)

	# initialize passed options
	ϵD = options.ϵD
	verbose = options.verbose
	maxIter = options.maxIter
	η1 = options.η1
	η2 = options.η2 
	σk = options.σk 
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
	ψχprox = params.ψχprox
	HessApprox = params.HessApprox
	ψk = params.ψk
	χk = params.χk 
	f_obj = params.f_obj
	h_obj = params.h_obj
	λ = params.λ


	# initialize parameters
	xk = copy(x0)

	k = 0
	Fobj_hist = zeros(maxIter)
	Hobj_hist = zeros(maxIter)
	Complex_hist = zeros(maxIter)
	verbose != 0 && @printf(
		"---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n",
	)
	verbose != 0 && @printf(
		"%10s | %10s | %11s | %12s | %10s | %10s | %11s | %10s | %10s | %10s | %9s | %9s\n",
		"Iter",
		"PG-Iter",
		"||Gν||",
		"Ratio: ρk",
		"x status ",
		"LM: σk",
		"σk status",
		"LnSrch: α",
		"||x||   ",
		"||s||   ",
		"f(x)   ",
		"h(x)   ",
	)
	verbose != 0 && @printf(
		"---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n",
	)
	# make sure you only take the first output of the objective value of the true function you are minimizing
	ObjOuter(x) = f_obj(x)[1] + λ * h_obj(x) # make the same name as in the paper f+h 


	k = 0
	ρk = -1.0
	α = 1.0
	TR_stat = ""
	x_stat = ""
	# main algorithm initialization
	Fsmth_out = f_obj(xk)
	# test number of outputs to see if user provided a hessian

	if length(Fsmth_out) == 2
		(fk, ∇fk) = Fsmth_out
	else
		error("Smooth Function must provide at least 2 outputs - fk and ∇fk. Can also provide Hessian.  ")
	end

	# define the Hessian 
    ν =1/σk
	# νInv = (1+θ)*maximum(abs.(eigs(H;nev=1, which=:LM)[1]))

	# keep track of old subgradient for LnSrch purposes
	Gν =  ∇fk * ν
	s = zeros(size(xk))
	funEvals = 1

	kktInit = χk(Gν)
	kktNorm = 100 * kktInit


	while kktNorm[1] > ϵD && k < maxIter
		# update count
		k = k + 1 # inner

		# store previous iterates
		xk⁻ = xk 
		∇fk⁻ = ∇fk
		sk⁻ = s

		Fobj_hist[k] = fk
		Hobj_hist[k] = h_obj(xk) * λ
		# Print values
		k % ptf == 0 && 
		@printf(
			"%11d|  %9d |  %10.5e   %10.5e   %9s   %10.5e   %10s   %10.4e   %9.4e   %9.4e   %9.4e  %9.4e\n",
				k, funEvals,  kktNorm[1], ρk,   x_stat,  σk, TR_stat,   α,   χk(xk), χk(s),   fk,    λ * h_obj(xk) )


		TR_stat = ""
		x_stat = ""
		# define inner function 
		φ(d) = [∇fk' * d + fk, ∇fk] # (φ, ∇φ)
		# define model and update ρ
		mk(d) = φ(d)[1] + λ * ψk(xk + d) # psik = h -> psik = h(x+d)


		s = ψχprox(-ν * ∇fk, ν * λ, xk, σk) # -> PG on one step s
		# Gν = s./ν
        funEvals = 1 

		# update Complexity history 
		Complex_hist[k] += funEvals# doesn't really count because of quadratic model 

		# @show ObjOuter(xk), ObjOuter(xk + s), mk(zeros(size(s))), mk(s)
		Numerator = ObjOuter(xk) - ObjOuter(xk + s)
		Denominator = mk(zeros(size(s))) - mk(s)
 

		ρk = (Numerator + 1e-16) / (Denominator + 1e-16)
		if (ρk > η2 && !(ρk == Inf || isnan(ρk) || Numerator < 0)) 
			TR_stat = "increase"
			σk = σk / γ # here γ>1, we shrink σk
		else
			TR_stat = "kept"
		end

		if (ρk >= η1 && !(ρk == Inf || isnan(ρk) || Numerator < 0))
			x_stat = "update"
			xk = xk + s
		end

		if (ρk < η1 || (ρk == Inf || isnan(ρk) || Numerator < 0))
            # @show ρk, ρk<η1, Numerator, Denominator
			x_stat = "reject"
			TR_stat = "shrink"
			σk = σk * γ # dominique changes to this? increase σk
		end

		Fsmth_out = f_obj(xk)
		
		if length(Fsmth_out) == 2
			(fk, ∇fk) = Fsmth_out
		else
			error("Smooth function must provide at least 2 outputs - fk and ∇fk. Can also provide Hessian.  ")
		end
		# update Gν with new direction
		# kktNorm = χk(s/ν)^2*ν
		if x_stat == "update"
			kktNorm = Denominator
		end

		# define the Hessian 
		# ν =1/ max(σk, (1+θ)*maximum(abs.(eigs(H;nev=1, which=:LM)[1])))
		ν = 1/σk


		Complex_hist[k] += 1

	end

	return xk, k, Fobj_hist[Fobj_hist .!= 0], Hobj_hist[Fobj_hist .!= 0], Complex_hist[Complex_hist .!= 0]
end