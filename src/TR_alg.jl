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
function TR(
	x0,
	params,
	options)

	# initialize passed options
	ϵD = options.ϵD
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
		"%10s | %10s | %11s | %12s | %10s | %10s | %11s | %10s | %10s | %10s | %10s | %9s | %9s\n",
		"Iter",
		"PG-Iter",
		"||Gν||",
		"Ratio: ρk",
		"x status ",
		"TR: Δk",
		"Δk status",
		"LnSrch: α",
		"||x||   ",
		"||s||   ",
		"||Bk||   ",
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
	if length(Fsmth_out) == 3
		(fk, ∇fk, Bk) = Fsmth_out
	elseif length(Fsmth_out) == 2 && k == 0
		(fk, ∇fk) = Fsmth_out
		Bk = HessApprox(size(xk, 1); mem=mem)
	elseif length(Fsmth_out) == 2
		(fk, ∇fk) = Fsmth_out
		push!(Bk, s,  ∇fk - ∇fk⁻)
	else
		error("Smooth Function must provide at least 2 outputs - fk and ∇fk. Can also provide Hessian.  ")
	end

	# define the Hessian 
	H = Symmetric(Matrix(Bk))
	# νInv = eigmax(H) #make a Matrix? ||B_k|| = λ(B_k) # change to opNorm(Bk, 2), arPack? 
	νInv = (1 + θ) * maximum(abs.(eigs(H;nev=1, which=:LM)[1]))

	# keep track of old subgradient for LnSrch purposes
	Gν =  ∇fk
	s = zeros(size(xk))
	funEvals = 1

	kktInit = χk(Gν)^2 / νInv
	kktNorm = 100 * kktInit

	while kktNorm[1] > ϵD && k < maxIter && Δk ≥ 1e-16 # is this kosher? 
		# update count
		k = k + 1 
		# store previous iterates
		xk⁻ = xk 
		∇fk⁻ = ∇fk
		sk⁻ = s


		Fobj_hist[k] = fk
		Hobj_hist[k] = h_obj(xk) * λ
		# Print values
		k % ptf == 0 && 
		@printf(
			"%11d|  %9d |  %10.5e   %10.5e   %9s   %10.5e   %10s   %10.4e   %9.4e   %9.4e   %9.4e   %9.4e  %9.4e\n",
				k, funEvals,  kktNorm[1], ρk,   x_stat,  Δk, TR_stat,   α,   χk(xk), χk(s), νInv,    fk,    λ * h_obj(xk) )

		TR_stat = ""
		x_stat = ""
		# define inner function 
		φ(d) = [0.5 * (d' * H * d) + ∇fk' * d + fk, H * d + ∇fk, H] # (φ, ∇φ, ∇²φ)
		# φν(d) = [fk + ∇fk'*d + νInv/2*(d'*d), ∇fk + νInv.*d] # necessary?
		# define model and update ρ
		mk(d) = φ(d)[1] + λ * ψk(xk + d) # psik = h -> psik = h(x+d)

		FO_options.ν = min(1 / νInv, Δk)

		s1 = ψχprox(-FO_options.ν * ∇fk, FO_options.λ * FO_options.ν, xk, Δk) # -> PG on one step s1
		Gν = s1 * νInv
		# if χk(Gν)^2 / νInv > ϵD # final stopping criteria 
		if kktNorm[1] > ϵD
			FO_options.optTol = min(.01, sqrt(χk(Gν))) * χk(Gν) # stopping criteria for inner algorithm 
			FO_options.FcnDec = mk(zeros(size(s))) - mk(s1)
			(s, s⁻, hist, funEvals) = s_alg(φ, (d) -> ψk(xk + d), s1, (d, λν) -> ψχprox(d, λν, xk, min(β * χk(s1), Δk)), FO_options) 
			# (s, s⁻, hist, funEvals) = s_alg(φ, (d)->ψk(xk + d), s1, (d, λν)->ψχprox(d, λν, xk, Δk), FO_options) #2*||s_1|| = Δk (same norm as TR) min(ξ*||s_1||, Δk)min(10*norm(s,2), Δk)
			Gν = s / FO_options.ν
		else
			funEvals = 1 
		end

		# update Complexity history 
		Complex_hist[k] += funEvals# doesn't really count because of quadratic model 



		α = 1.0
		# @show ObjOuter(xk), ObjOuter(xk + s), mk(zeros(size(s))), mk(s)
		Numerator = ObjOuter(xk) - ObjOuter(xk + s)
		Denominator = mk(zeros(size(s))) - mk(s)

		# @show β*norm(s1)^2, norm(s)^2, norm(s)^2>norm(s1)^2, Numerator, Denominator, Δk

		ρk = (Numerator + 1e-16) / (Denominator + 1e-16)

		if (ρk > η2 && !(ρk == Inf || isnan(ρk) || Numerator < 0))
			TR_stat = "increase"
			Δk = max(Δk, γ * χk(s))
			# Δk = γ*Δk
		else
			TR_stat = "kept"
		end

		if (ρk >= η1 && !(ρk == Inf || isnan(ρk) || Numerator < 0))
			x_stat = "update"
			xk = xk + s
		end

		if (ρk < η1 || (ρk == Inf || isnan(ρk) || Numerator < 0))
			x_stat = "reject"
			TR_stat = "shrink"
			α = .5
			Δk = α * Δk	# * norm(s, Inf) #change to reflect trust region 
		end

		Fsmth_out = f_obj(xk)
		
		if length(Fsmth_out) == 3
			(fk, ∇fk, Bk) = Fsmth_out
		elseif length(Fsmth_out) == 2
			(fk, ∇fk) = Fsmth_out
			push!(Bk, s, ∇fk - ∇fk⁻) # should this be Gν⁺ - Gν? 
		else
			error("Smooth function must provide at least 2 outputs - fk and ∇fk. Can also provide Hessian.  ")
		end

		# define the Hessian 
		H = Symmetric(Matrix(Bk))
		# β = eigmax(H) #make a Matrix? ||B_k|| = λ(B_k) # change to opNorm(Bk, 2), arPack? 
		νInv = (1 + θ) * maximum(abs.(eigs(H;nev=1, which=:LM)[1]))
		
		# update Gν with new direction
		# kktNorm = χk(Gν)^2/νInv
		if x_stat == "update"
			kktNorm = Denominator
		end

		Complex_hist[k] += 1

	end

	return xk, k, Fobj_hist[Fobj_hist .!= 0], Hobj_hist[Fobj_hist .!= 0], Complex_hist[Complex_hist .!= 0]
end