#Implements Algorithm 4.2 in "Interior-Point Trust-Region Method for Composite Optimization".
#Note that some of the file inclusions are for testing purposes (ie minconf_spg)

using LinearOperators
export IP_options, IntPt_TR, IP_struct #export necessary values to file that calls these functions
# include("proxGD.jl")

mutable struct IP_params
	ϵD #termination criteria
	ϵC #dual termination criteria
	Δk #trust region radius
	verbose #print every so often
	maxIter #maximum amount of inner iterations
	η1 #ρ lower bound 
	η2 #ρ upper bound 
	τ # linesearch buffer parameter 
	σ #quadratic model linesearch buffer parameter
	γ #trust region buffer 
	mem #Bk iteration memory
	θ #TR inner loop "closeness" to Bk
end

mutable struct IP_methods
	FO_options #options for minConf_SPG/minimization routine you use for s
	s_alg #algorithm passed that determines descent direction
	Rkprox # ψ_k + χ_k where χ_k is the Δ - norm ball that you project onto. Note that the basic case is that ψ_k = 0
	ψk #nonsmooth model of h that you are trying to solve - it is possible that ψ=h. 
	f_obj #objective function (unaltered) that you want to minimize
	h_obj #objective function that is nonsmooth - > only used for evaluation
	λ #objective nonsmooth tuning parameter
end

function IP_options(
	;
	ϵD = 1e-2,
	ϵC = 1e-2,
	Δk = 1.0,
	verbose = 0,
	maxIter = 10000,
	η1 = 1.0e-3, #ρ lower bound
	η2 = 0.9,  #ρ upper bound
	τ = 0.01, #linesearch buffer parameter
	σ = 1.0e-3, # quadratic model linesearch buffer parameter
	γ = 3.0, #trust region buffer
	mem = 5, #L-BFGS memory
	θ = 1/8
) #default values for trust region parameters in algorithm 4.2
	return IP_params(ϵD, ϵC, Δk, verbose, maxIter,η1, η2, τ, σ, γ, mem, θ)
end

function IP_struct(
	f_obj,
	h,
	λ;
	FO_options = s_options(1.0;),
	s_alg = PG,
	Rkprox = (z, σ, xt, Dk) → z./max(1, norm(z, 2)/σ),
	ψk = h
)
	return IP_methods(FO_options, s_alg, Rkprox, ψk, f_obj, h, λ)
end



"""Interior method for Trust Region problem
	IntPt_TR(x, TotalCount,params, options)
Arguments
----------
x : Array{Float64,1}
	Initial guess for the x value used in the trust region
TotalCount: Float64
	overall count on total iterations
params : mutable structure IP_params with:
	--
	-ϵD, tolerance for primal convergence
	-ϵC, tolerance for dual convergence
	-Δk Float64, trust region radius
	-verbose Int, print every # options
	-maxIter Float64, maximum number of inner iterations (note: does not influence TotalCount)
options : mutable struct IP_methods
	-f_obj, smooth objective function; takes in x and outputs [f, g, Bk]
	-h_obj, nonsmooth objective function; takes in x and outputs h
	--
	-FO_options, options for first order algorithm, see DescentMethods.jl for more
	-s_alg, algorithm for descent direction, see DescentMethods.jl for more
	-Rkprox, function projecting onto the trust region ball or ψ+χ
	-InnerFunc, inner objective or proximal operator of ψk+χk+1/2||u - sⱼ + ∇qk|²
l : Vector{Float64} size of x, defaults to -Inf
u : Vector{Float64} size of x, defaults to Inf
μ : Float64, initial barrier parameter, defaults to 1.0

Returns
-------
x   : Array{Float64,1}
	Final value of Algorithm 4.2 trust region
k   : Int
	number of iterations used
"""
function IntPt_TR(
	x0,
	params,
	options)

	#initialize passed options
	ϵD = options.ϵD
	ϵC = options.ϵC
	Δk = options.Δk
	verbose = options.verbose
	maxIter = options.maxIter
	η1 = options.η1
	η2 = options.η2 
	σ = options.σ 
	γ = options.γ
	τ = options.τ
	θ = options.θ
	mem = options.mem

	if verbose==0
		ptf = Inf
	elseif verbose==1
		ptf = round(maxIter/10)
	elseif verbose==2
		ptf = round(maxIter/100)
	else
		ptf = 1
	end

	#other parameters
	FO_options = params.FO_options
	s_alg = params.s_alg
	Rkprox = params.Rkprox
	ψk = params.ψk
	f_obj = params.f_obj
	h_obj = params.h_obj
	λ = params.λ


	#initialize parameters
	xk = copy(x0)

	k = 0
	Fobj_hist = zeros(maxIter)
	Hobj_hist = zeros(maxIter)
	Complex_hist = zeros(maxIter)
	verbose!=0 && @printf(
		"------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n",
	)
	verbose!=0 && @printf(
		"%10s | %11s | %11s | %10s | %11s | %11s | %10s | %10s | %10s | %10s | %10s | %10s\n",
		"Iter",
		"||Gν||",
		"Ratio: ρk",
		"x status ",
		"TR: Δk",
		"Δk status",
		"LnSrch: α",
		"||x||",
		"||s||",
		"||Bk||",
		"f(x)",
		"h(x)",
	)
	verbose!=0 && @printf(
		"------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n",
	)
	#make sure you only take the first output of the objective value of the true function you are minimizing
	ObjOuter(x) = f_obj(x)[1] + λ*h_obj(x) 


	k = 0
	ρk = -1
	α = 1.0

	#main algorithm initialization
	Fsmth_out = f_obj(xk)
	#test number of outputs to see if user provided a hessian

	if length(Fsmth_out)==3
		(fk, ∇fk, Bk) = Fsmth_out
	elseif length(Fsmth_out)==2 && k==0
		(fk, ∇fk) = Fsmth_out
		Bk = LBFGSOperator(size(xk,1); mem = mem)
	elseif length(Fsmth_out)==2
		(fk, ∇fk) = Fsmth_out
		push!(Bk, s,  ∇fk-∇fk⁻)
	else
		error("Smooth Function must provide at least 2 outputs - fk and ∇fk. Can also provide Hessian.  ")
	end

	# initialize qk
	qk = fk 
	∇qk = ∇fk 
	H = Bk

	
	#keep track of old subgradient for LnSrch purposes
	Gν =  ∇fk
	s = zeros(size(xk))

	∇qksj = copy(∇qk) 
	g_old = Gν

	kktInit = norm(g_old)
	kktNorm = 100*kktInit

	while kktNorm[1] > ϵD && k < maxIter
		#update count
		k = k + 1 #inner
		TR_stat = ""
		x_stat = ""

		#store previous iterates
		xk⁻ = xk 
		∇fk⁻ = ∇fk
		sk⁻ = s

		#define the Hessian 
		∇²qk = Matrix(H) 
		β = eigmax(∇²qk) #make a Matrix? ||B_k|| = λ(B_k)

		#define inner function 
		# objInner(d) = [0.5*(d'*∇²qk(d)) + ∇qk'*d + qk, ∇²qk(d) + ∇qk] #(mkB, ∇mkB)
		objInner(d) = [0.5*(d'*∇²qk*d) + ∇qk'*d + qk, ∇²qk*d + ∇qk] #(mkB, ∇mkB)
		s⁻ = zeros(size(xk))
		


		νmin = (1-sqrt(1-4*θ))/(2*β)
		νmax = (1+sqrt(1-4*θ))/(2*β)
		FO_options.ν = (νmin+νmax)/2 #nu min later? λ(B_k)/2
		if h_obj(xk)==0 #i think this is for h==0? 
			FO_options.λ = Δk * FO_options.β
		end

		s = Rkprox(-FO_options.ν*∇qk, FO_options.λ*FO_options.ν, xk, Δk) #-> PG on step s1
		Gν = s/FO_options.ν
		if norm(Gν)>ϵD #final stopping criteria 
			(s, s⁻, hist, funEvals) = s_alg(objInner, (d)->ψk(xk + d), s, (d, λν)->Rkprox(d, λν, xk, Δk), FO_options)
		else
			funEvals = 1 
		end

		#update Complexity history 
		Complex_hist[k]+=funEvals# doesn't really count because of quadratic model 




		α = 1.0
		#define model and update ρ
		# mk(d) = 0.5*(d'*∇²qk*d) + ∇qk'*d + qk + ψk(xk + d)
		mk(d) = objInner(d)[1] + ψk(xk+d) #psik = h -> psik = h(x+d)
		# look up how to test if two functions are equivalent? 
		ρk = (ObjOuter(xk) - ObjOuter(xk + s)) / (mk(zeros(size(s)))-mk(s))

		if (ρk > η2)
			TR_stat = "increase"
			# Δk = max(Δk, γ * norm(s, 1)) #for safety
			Δk = γ*Δk
		else
			TR_stat = "kept"
		end

		if (ρk >= η1 && !(ρk==Inf || isnan(ρk)))
			x_stat = "update"
			xk = xk + s
			zkl = zkl + dzl
			zku = zku + dzu
		end

		if (ρk < η1 || (ρk ==Inf || isnan(ρk)))

			x_stat = "reject"
			TR_stat = "shrink"
			α = .5
			Δk = α*Δk	#* norm(s, Inf) #change to reflect trust region 
		end

		Fsmth_out = f_obj(xk)
		
		if length(Fsmth_out)==3
			(fk, ∇fk, Bk) = Fsmth_out
		elseif length(Fsmth_out)==2
			(fk, ∇fk) = Fsmth_out
			push!(Bk, s, ∇fk-∇fk⁻)
		else
			error("Smooth function must provide at least 2 outputs - fk and ∇fk. Can also provide Hessian.  ")
		end


		#update qk with new direction
		qk = fk 
		∇qk = ∇fk


		#update Gν with new direction
		g_old = Gν
		kktNorm = norm(Gν)

		#Print values
		k % ptf == 0 && 
		@printf(
			"%11d|  %10.5e   %10.5e   %10s   %10.5e   %10s   %10.5e   %10.5e   %10.5e   %10.5e   %10.5e  %10.5e\n",
			   k,   kktNorm[1], ρk,   x_stat,  Δk, TR_stat,   α,   norm(xk, 2), norm(s, 2), β,    fk,    ψk(xk))

		Fobj_hist[k] = fk
		Hobj_hist[k] = h_obj(xk)
		Complex_hist[k]+=1

	end

	return xk, k, Fobj_hist[Fobj_hist.!=0], Hobj_hist[Fobj_hist.!=0], Complex_hist[Complex_hist.!=0]
end