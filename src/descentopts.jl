export s_params, TRNCparams, TRNCmethods, FObj #, HObj

mutable struct s_params
	ν
	λ
	optTol
	maxIter
	verbose
	p
	FcnDec

	function s_params(ν, λ; optTol=1f-6, maxIter=10000, verbose=0, p=1.1, FcnDec=1e10)
		return new(ν,λ, optTol, maxIter, verbose, p, FcnDec)
	end
end

mutable struct TRNCparams
	ϵ # termination criteria
	Δk # trust region radius
	verbose # print every so often
	maxIter # maximum amount of inner iterations
	η1 # ρ lower bound 
	η2 # ρ upper bound 
	τ # linesearch buffer parameter 
	σk # quadratic model linesearch buffer parameter
	γ # trust region buffer 
	mem # Bk iteration memory
	θ # TR inner loop "closeness" to Bk
	β # TR size for PG steps j>1

	function TRNCparams(
		;
		ϵ=1e-2,
		Δk=1.0,
		verbose=0,
		maxIter=10000,
		η1=1.0e-3, # ρ lower bound
		η2=0.9,  # ρ upper bound
		τ=0.01, # linesearch buffer parameter
		σk=1.0e-3, # LM parameter
		γ=3.0, # trust region buffer
		mem=5, # L-BFGS memory
		θ=1e-3,
		β=10.0
		) # default values for trust region parameters in algorithm 4.2
		return new(ϵ, Δk, verbose, maxIter, η1, η2, τ, σk, γ, mem, θ, β)
	end
end

mutable struct FObj
	eval
	grad
	Hess # Hessian (approximation) choosen. Defaults to LSR1, unless the user provides a Hessian in the smooth function 
	
	function FObj(
		eval; 
		grad=(x) -> ForwardDiff.gradient(eval, x), 
		Hess=LSR1Operator,
		)
		return new(eval, grad, Hess)
	end
end

# mutable struct HObj
# 	eval
# 	ψk # nonsmooth model of h that you are trying to solve - default is ψ=h.
# 	ψχprox # ψ_k + χ_k where χ_k is the Δ - norm ball that you project onto. 	
# 	# Note that the basic case is that ψ_k = 0 with l2 TR norm, 
# 	# so it is just prox of l2-norm ball projection. 
# 	# In the LM algorithm, this is just the prox of ψₖ

# 	function HObj(
# 		eval; 
# 		ψk=eval, 
# 		ψχprox=(z, σ, xt, Dk) -> z ./ max(1, norm(z, 2) / σ),
# 		)
# 		return new(eval, ψk, ψχprox)
# 	end
# end

mutable struct TRNCmethods
	FO_options # options for minimization routine you use for s; based on minconf_spg
	s_alg # algorithm passed that determines descent direction
	# ψχprox # ψ_k + χ_k where χ_k is the Δ - norm ball that you project onto. 
	# Note that the basic case is that ψ_k = 0 with l2 TR norm, 
	# so it is just prox of l2-norm ball projection. 
	# In the LM algorithm, this is just the prox of ψₖ
	# ψk # nonsmooth model of h that you are trying to solve - default is ψ=h. 
	χk # TR norm one computes for the trust region radius - default is l2 
	# HessApprox # Hessian Approximation choosen. Defaults to LBFGS, unless the user provides a Hessian in the smooth function 
	f # objective function (unaltered) that you want to minimize
	ψ # objective function that is nonsmooth - > only used for evaluation

	function TRNCmethods(
		f,
		ψ;
		FO_options=s_params(1.0, 1.0),
		s_alg=PG,
		# ψχprox=(z, σ, xt, Dk) -> z ./ max(1, norm(z, 2) / σ),
		# ψk=h,
		χk=s -> norm(s, 2),
	)
		return new(FO_options, s_alg, χk, f, ψ)
	end
end