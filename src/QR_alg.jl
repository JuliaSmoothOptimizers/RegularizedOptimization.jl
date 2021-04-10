# Implements Algorithm 4.2 in "Interior-Point Trust-Region Method for Composite Optimization".

using LinearOperators, LinearAlgebra, Arpack
export QuadReg

"""Interior method for Trust Region problem
	QuadReg(x, params, options)
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
function QuadReg(
	x0,
	params,
	options)

	# initialize passed options
	ϵ = options.ϵ
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
	f_obj = params.f
	h_obj = params.h
	χk = params.χk


	# initialize parameters
	xk = copy(x0)

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
		"σk",
		"σk status",
		" α",
		"‖x‖   ",
		"‖s‖   ",
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
	∇fk =  f_obj.grad(xk)
	∇fk⁻ = ∇fk
	xk⁻ = xk 
	fk = f_obj.eval(xk)
	hk = h_obj.eval(xk)
    ν = 1 / σk
	# νInv = (1+θ)*maximum(abs.(eigs(H;nev=1, which=:LM)[1]))
	Gν =  ∇fk * σk
	s = zeros(size(xk))
	funEvals = 1

	kktNorm = χk(Gν)^2 * σk
	optimal = kktNorm < ϵ


	while !optimal && k < maxIter
		# update count
		k = k + 1 # inner

		Fobj_hist[k] = fk
		Hobj_hist[k] = hk
		# Print values
		k % ptf == 0 && 
		@printf(
			"%11d|  %9d |  %10.5e   %10.5e   %9s   %10.5e   %10s   %10.4e   %9.4e   %9.4e   %9.4e  %9.4e\n",
				k, funEvals,  kktNorm[1], ρk,   x_stat,  σk, TR_stat,   α,   χk(xk), χk(s),   fk,    hk )

		TR_stat = ""
		x_stat = ""

		# define model and update ρ
		mk(d) = ∇fk' * d + fk + h_obj.ψk(xk + d) # psik = h -> psik = h(x+d)

		s = h_obj.ψχprox(-ν * ∇fk, ν * FO_options.λ, xk, σk) # -> PG on one step s

		fkn = f_obj.eval(xk + s)
		hkn = h_obj.eval(xk + s)
		Numerator = fk + hk - (fkn + hkn)
		Denominator = fk + hk - mk(s)
 

		ρk = (Numerator + 1e-16) / (Denominator + 1e-16)
		if ρk > η2 
			TR_stat = "increase"
			σk = σk / γ # here γ>1, we shrink σk
		else
			TR_stat = "kept"
		end

		if ρk >= η1 
			x_stat = "update"
			xk .+= s 

			#update functions 
			fk = fkn
			hk = hkn

			#update gradient 
			∇fk .= f_obj.grad(xk)
			Complex_hist[k] += 1

		end

		if ρk < η1 
			x_stat = "reject"
			TR_stat = "shrink"
			σk = max(σk * γ, 1e-6) # dominique σmin ok? 
		end

		
		# update Gν with new direction
		# kktNorm = χk(s/ν)^2*ν
		if x_stat == "update"
			kktNorm = Denominator
		end

		# define the Hessian 
		ν = 1 / σk

		# store previous iterates
		xk⁻ .= xk 
		∇fk⁻ .= ∇fk
		optimal = kktNorm < ϵ
		
	end

	return xk, k, Fobj_hist[Fobj_hist .!= 0], Hobj_hist[Fobj_hist .!= 0], Complex_hist[Complex_hist .!= 0]
end