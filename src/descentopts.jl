using ForwardDiff, LinearOperators
export s_options, TRNCoptions, TRNCstruct, FObj, HObj

mutable struct s_params
	ν
	optTol
	maxIter
	verbose
	restart
	λ
	p
	FcnDec
end


function s_options(ν; optTol=1f-6, maxIter=10000, verbose=0, restart=10, λ=1.0, p=1.1, FcnDec=1e10)

	return s_params(ν, optTol, maxIter, verbose, restart, λ, p, FcnDec)

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
end

function TRNCoptions(
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
	return TRNCparams(ϵ, Δk, verbose, maxIter, η1, η2, τ, σk, γ, mem, θ, β)
end


mutable struct Fsmooth
	eval
	grad
	Hess # Hessian (approximation) choosen. Defaults to LSR1, unless the user provides a Hessian in the smooth function 
end
function FObj(eval; grad=(x) -> ForwardDiff.gradient(eval, x), Hess=LSR1Operator)
	return Fsmooth(eval, grad, Hess)
end

mutable struct Hnonsmooth
	eval
	ψk # nonsmooth model of h that you are trying to solve - default is ψ=h.
	ψχprox # ψ_k + χ_k where χ_k is the Δ - norm ball that you project onto. 	
	# Note that the basic case is that ψ_k = 0 with l2 TR norm, 
	# so it is just prox of l2-norm ball projection. 
	# In the LM algorithm, this is just the prox of ψₖ
end

function HObj(eval; ψk=eval, ψχprox=(z, σ, xt, Dk) -> z ./ max(1, norm(z, 2) / σ),)
	return Hnonsmooth(eval, ψk, ψχprox)
end

mutable struct TRNCmethods
	FO_options # options for minimization routine you use for s; based on minconf_spg
	s_alg # algorithm passed that determines descent direction
	χk # TR norm one computes for the trust region radius - default is l2 
	f # struct of objective function (unaltered) that you want to minimize
	h # objective function that is nonsmooth - > only used for evaluation
end


function TRNCstruct(
	f,
	h;
	FO_options=s_options(1.0;),
	s_alg=PG,
	χk=(s) -> norm(s, 2)
)
	return TRNCmethods(FO_options, s_alg, χk, f, h)
end