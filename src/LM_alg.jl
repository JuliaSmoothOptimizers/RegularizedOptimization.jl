# Implements Algorithm 4.2 in "Interior-Point Trust-Region Method for Composite Optimization".

using LinearOperators, LinearAlgebra, Arpack
export QR

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
function QR(x0, params, options)

	# initialize passed options
	ϵD = options.ϵD
	verbose = options.verbose
	maxIter = options.maxIter
	η1 = options.η1
	η2 = options.η2 
	σk = options.σk 
	γ = options.γ

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
	ψχprox = params.ψχprox
	ψk = params.ψk
	χk = params.χk 
	f_obj = params.f_obj
	f_grad = params.f_grad
	h_obj = params.h_obj
	λ = params.λ

	# initialize parameters
	xk = copy(x0)
	k = 0
	Fobj_hist = zeros(maxIter)
	Hobj_hist = zeros(maxIter)
	Complex_hist = zeros(Int, maxIter)
	@info @sprintf "%6s %8s %8s %7s %8s %7s %7s %7s %1s" "iter" "f(x)" "h(x)" "ξ" "ρ" "σ" "‖x‖" "‖s‖" ""

	k = 0
	ρk = -1.0
	TR_stat = ""

	fk = f_obj(xk)
	∇fk = f_grad(xk)
	hk = λ * h_obj(xk)

	ν = 1 / σk

	sNorm = 0.0
	ξ = 0.0
	funEvals = 1

	optimal = false
	tired = k ≥ maxIter

	while !(optimal || tired) 
		k = k + 1
		Fobj_hist[k] = fk
		Hobj_hist[k] = hk
		k % ptf == 0 && @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e %7.1e %7.1e %1s" k fk hk ξ ρk σk χk(xk) sNorm TR_stat

		# define model
		φk(d) = ∇fk' * d + fk
		mk(d) = φk(d) + λ * ψk(xk + d) # psik = h -> psik = h(x+d)

		s = ψχprox(-ν * ∇fk, ν * λ, xk, σk) # -> PG on one step s
		sNorm = χk(s)

		# update Complexity history 
		Complex_hist[k] += 1# doesn't really count because of quadratic model 

		fkps = f_obj(xk + s)
		hkps = λ * h_obj(xk + s)
		Δobj = (fk + hk) - (fkps + hkps)
		ξ = (fk + hk) - mk(s)

		if (ξ ≤ 0 || isnan(ξ))
			error("failed to compute a step")
		end
 
		ρk = (Δobj + 1e-16) / (ξ + 1e-16)
		if η2 ≤ ρk < Inf
			TR_stat = "↗"
			σk = σk / γ
		else
			TR_stat = "="
		end

		if η1 ≤ ρk < Inf
			xk .+= s
			fk = fkps
			hk = hkps
			optimal = ξ < ϵD
			if !optimal
				∇fk = f_grad(xk)
			end
		end

		if ρk < η1 || ρk == Inf
			TR_stat = "↘"
			σk = σk * γ
		end

		ν = 1 / σk
		tired = k ≥ maxIter
	end

	return xk, k, Fobj_hist[Fobj_hist .!= 0], Hobj_hist[Fobj_hist .!= 0], Complex_hist[Complex_hist .!= 0]
end
