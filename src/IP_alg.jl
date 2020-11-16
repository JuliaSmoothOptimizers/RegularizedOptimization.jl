#Implements Algorithm 4.2 in "Interior-Point Trust-Region Method for Composite Optimization".
#Note that some of the file inclusions are for testing purposes (ie minconf_spg)

using LinearOperators
export IP_options, IntPt_TR, IP_struct #export necessary values to file that calls these functions


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
end

mutable struct IP_methods
	FO_options #options for minConf_SPG/minimization routine you use for s
	s_alg #algorithm passed that determines descent direction
	Rkprox # ψ_k + χ_k where χ_k is the Δ - norm ball that you project onto. Note that the basic case is that ψ_k = 0
	ψk #nonsmooth model of h that you are trying to solve - it is possible that ψ=h. 
	f_obj #objective function (unaltered) that you want to minimize
	h_obj #objective function that is nonsmooth - > only used for evaluation
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
) #default values for trust region parameters in algorithm 4.2
	return IP_params(ϵD, ϵC, Δk, verbose, maxIter,η1, η2, τ, σ, γ)
end

function IP_struct(
	f_obj,
	h;
	FO_options = s_options(1.0;),
	s_alg = PG,
	Rkprox = (z, σ, xt, Dk) → z./max(1, norm(z, 2)/σ),
	ψk = h
)
	return IP_methods(FO_options, s_alg, Rkprox, ψk, f_obj, h)
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
	options;
	l = -1.0e16 * ones(size(x0)),
	u = 1.0e16 * ones(size(x0)),
	μ = 0.0,
	BarIter = 1)

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


	#initialize parameters
	xk = copy(x0)
	#initialize them to positive values for x=l and negative for x=u
	if μ ==0.0
		zkl = zeros(size(x0))
		zku = zeros(size(x0))
	else
		zkl = ones(size(x0))
		zku = -ones(size(x0))
	end
	k = 0
	Fobj_hist = zeros(maxIter * BarIter)
	Hobj_hist = zeros(maxIter * BarIter)
	Complex_hist = zeros(maxIter * BarIter)
	verbose!=0 && @printf(
		"------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n",
	)
	verbose!=0 && @printf(
		"%10s | %11s | %11s | %11s | %11s | %11s | %10s | %11s | %11s | %10s | %10s | %10s | %10s   | %10s | %10s\n",
		"Iter",
		"μ",
		"||(Gν-∇q) + ∇ϕ⁺)-zl+zu||",
		"||zl(x-l) - μ||",
		"||zu(u-x) - μ||",
		"Ratio: ρk",
		"x status ",
		"TR: Δk",
		"Δk status",
		"LnSrch: α",
		"||x||",
		"||s||",
		"β",
		"f(x)",
		"h(x)",
	)
	verbose!=0 && @printf(
		"------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n",
	)

	#Barrier Loop
	while k < BarIter && (μ > 1e-6 || μ==0) #create options for this
		#make sure you only take the first output of the objective value of the true function you are minimizing
		ObjOuter(x) = f_obj(x)[1] + h_obj(x) - μ*sum(log.((x-l).*(u-x)))# - μ * sum(log.(x - l)) - μ * sum(log.(u - x)) #


		k_i = 0
		ρk = -1
		α = 1.0

		#main algorithm initialization
		Fsmth_out = f_obj(xk)
		#test number of outputs to see if user provided a hessian

		if length(Fsmth_out)==3
			(fk, ∇fk, Bk) = Fsmth_out
		elseif length(Fsmth_out)==2 && k_i==0
			(fk, ∇fk) = Fsmth_out
			Bk = LBFGSOperator(size(xk,1))
		elseif length(Fsmth_out)==2
			(fk, ∇fk) = Fsmth_out
			# Bk = bfgs_update(Bk, s, ∇fk-∇fk⁻)
			push!(Bk, s,  ∇fk-∇fk⁻)
		else
			# throw(ArgumentError(f_obj, "Function must provide at least 2 outputs - fk and ∇fk. Can also provide Hessian.  "))
			error("Smooth Function must provide at least 2 outputs - fk and ∇fk. Can also provide Hessian.  ")
		end

		# initialize qk
		qk = fk - μ * sum(log.(xk - l)) - μ * sum(log.(u - xk))
		∇qk = ∇fk - μ ./ (xk - l) + μ ./ (u - xk)

		#changed for now to have full representation
		if isempty(methods(Bk))
			# H(d) = Bk*d
			H = Bk
		else 
			H = Bk 
		end

		
		#keep track of old subgradient for LnSrch purposes
		Gν =  ∇fk
		s = zeros(size(xk))
		dzl = zeros(size(zkl))
		dzu = zeros(size(zku))
		∇qksj = copy(∇qk) 
		g_old = ((Gν - ∇qksj) + ∇qk) #this is just ∇fk at first 

		kktInit = [norm(g_old - zkl + zku), norm(zkl .* (xk - l) .- μ), norm(zku .* (u - xk) .- μ)]
		kktNorm = 100*kktInit

		# while (kktNorm[1]/kktInit[1] > ϵD || kktNorm[2]/kktInit[2] > ϵC || kktNorm[3]/kktInit[3] > ϵC) && k_i < maxIter
		while (kktNorm[1] > ϵD || kktNorm[2] > ϵC || kktNorm[3] > ϵC) && k_i < maxIter
			#update count
			k_i = k_i + 1 #inner
			k = k + 1  #outer
			TR_stat = ""
			x_stat = ""

			#store previous iterates
			xk⁻ = xk 
			∇fk⁻ = ∇fk
			sk⁻ = s
			dzl⁻ = dzl 
			dzu⁻ = dzu 

			#define the Hessian 
			# ∇²qk(d) = H(d) + Diagonal(zkl ./ (xk - l))*d + Diagonal(zku ./ (u - xk))*d
			# β = power_iteration(∇²qk,randn(size(xk)))[1] #computes ||B_k||_2^2
			∇²qk = H + Diagonal(zkl ./ (xk - l)) + Diagonal(zku ./ (u - xk))
			β = eigmax(Matrix(∇²qk)) #make a Matrix? 

			#define inner function 
			# objInner(d) = [0.5*(d'*∇²qk(d)) + ∇qk'*d + qk, ∇²qk(d) + ∇qk] #(mkB, ∇mkB)
			objInner(d) = [0.5*(d'*∇²qk*d) + ∇qk'*d + qk, ∇²qk*d + ∇qk] #(mkB, ∇mkB)
			s⁻ = zeros(size(xk))
			
			FO_options.β = β
			if h_obj(xk)==0 #i think this is for h==0? 
				FO_options.λ = Δk * FO_options.β
			end
			(s, s⁻, hist, funEvals) = s_alg(objInner, (d)->ψk(xk + d), s⁻, (d, λν)->Rkprox(d, λν, xk, Δk), FO_options)
			# @show hist
			#update Complexity history 
			Complex_hist[k]+=funEvals # doesn't really count because of quadratic model 

			#compute qksj for the previous iterate 
			Gν = (s⁻ - s) * β
			∇qksj = ∇qk + ∇²qk(s⁻)



			α = 1.0
			mult = 0.5
			# gradient for z
			dzl = μ ./ (xk - l) - zkl - zkl .* s ./ (xk - l)
			dzu = μ ./ (u - xk) - zku + zku .* s ./ (u - xk)
			# linesearch for step size?
			# if μ!=0
				# α = directsearch(xk - l, u - xk, zkl, zku, s, dzl, dzu)
				α = lsTR(xk, s,l,u; mult=mult, tau =τ)
				# α = linesearch(xk, zkl, zku, s, dzl, dzu,l,u ;mult=mult, tau = τ)
			# end
			#update search direction
			s = s * α
			dzl = dzl * α
			dzu = dzu * α

			#define model and update ρ
			# mk(d) = 0.5*(d'*∇²qk(d)) + ∇qk'*d + qk + ψk(xk + d) #needs to be xk in the model -> ask user to specify that? 
			mk(d) = 0.5*(d'*∇²qk*d) + ∇qk'*d + qk + ψk(xk + d)
			# look up how to test if two functions are equivalent? 
			ρk = (ObjOuter(xk) - ObjOuter(xk + s)) / (mk(zeros(size(xk))) - mk(s))
			# @show ObjOuter(xk)
			# @show ObjOuter(xk + s)
			# @show mk(zeros(size(xk)))
			# @show mk(s)
			# @show f_obj(xk)[1]
			# @show h_obj(xk)
			# @show f_obj(xk+s)[1]
			# @show h_obj(xk+s)
			if (ρk > η2)
				TR_stat = "increase"
				Δk = max(Δk, γ * norm(s, 1)) #for safety
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

				x_stat = "shrink"

				#changed back linesearch
				α = 1.0
				#this needs to be the previous search direction and iterate? 
				while(ObjOuter(xk + α*s) > ObjOuter(xk) + σ*α*(g_old'*s) && α>1e-16) #compute a directional derivative of ψ CHECK LINESEARCH
					α = α*mult
					# @show α
				end
				# α = 0.1 #was 0.1; can be whatever
				#step should be rejected
				xk = xk + α*s
				zkl = zkl + α*dzl
				zku = zku + α*dzu
				Δk = α * norm(s, 1)
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
			qk = fk - μ * sum(log.(xk - l)) - μ * sum(log.(u - xk))
			∇qk = ∇fk - μ ./ (xk - l) + μ ./ (u - xk)
			# ∇²qk = H + Diagonal(zkl ./ (xk - l)) + Diagonal(zku ./ (u - xk))


			#update Gν with new direction
			# Gν = (s⁻ - s) * β #is affine scaling of s (αs) still in the subgradient? 
			g_old = (Gν - ∇qksj) + ∇qk
			kktNorm = [
				norm(g_old - zkl + zku) #check this
				norm(zkl .* (xk - l) .- μ)
				norm(zku .* (u - xk) .- μ)
			]
	
			#Print values
			k % ptf == 0 && 
			@printf(
				"%11d|  %10.5e  %19.5e   %18.5e   %17.5e   %10.5e   %10s   %10.5e   %10s   %10.5e   %10.5e   %10.5e   %10.5e   %10.5e   %10.5e \n",
				# k, μ, kktNorm[1]/kktInit[1],  kktNorm[2]/kktInit[2],  kktNorm[3]/kktInit[3], ρk, x_stat, Δk, TR_stat, α, norm(xk, 2), norm(s, 2), β, fk, ψk(xk))
				k, μ, kktNorm[1],  kktNorm[2],  kktNorm[3], ρk, x_stat, Δk, TR_stat, α, norm(xk, 2), norm(s, 2), β, fk, ψk(xk))

			Fobj_hist[k] = fk
			Hobj_hist[k] = h_obj(xk)
			Complex_hist[k]+=1
			# if k % ptf == 0
			# 	FO_options.optTol = FO_options.optTol * 0.1
			# end
		end
		# mu = norm(zl.*(x.-l)) + norm(zu.*(u.-x))
		μ = 0.1 * μ
		k = k + 1
		ϵD = ϵD * μ
		ϵC = ϵC * μ

	end
	return xk, k, Fobj_hist[Fobj_hist.!=0], Hobj_hist[Fobj_hist.!=0], Complex_hist[Complex_hist.!=0]
end