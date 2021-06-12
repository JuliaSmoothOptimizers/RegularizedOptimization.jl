export PG, PGLnsch, PGΔ, PGE

using Printf, ShiftedProximalOperators
"""
		Proximal Gradient Descent  for
		min_x ϕ(x) = f(x) + g(x), with f(x) cvx and β-smooth, g(x) closed cvx

		Input:
				f: function handle that returns f(x) and ∇f(x)
				h: function handle that returns g(x)
				s: initial point
				proxG: function handle that calculates prox_{νg}
				options: see descentopts.jl
		Output:
				s⁺: s update
				s : s^(k-1)
				his : function history
				feval : number of function evals (total objective )
"""
function PG(
	f, 
	∇f,
	h, 
	options;
	x0::AbstractVector=f.meta.x0,
	)

	ϵ=options.ϵ
	maxIter=options.maxIter

	if options.verbose==0
		ptf = Inf
	elseif options.verbose==1
		ptf = round(maxIter/10)
	elseif options.verbose==2
		ptf = round(maxIter/100)
	else
		ptf = 1
	end

	#Problem Initialize
	ν = options.ν
	x = x0
	x⁺ = deepcopy(x)
	Fobj_hist = zeros(maxIter)
	Hobj_hist = zeros(maxIter)
	Complex_hist = zeros(Int, maxIter)

	# Iteration set up
	g = ∇f(x⁺) #objInner/ quadratic model
	k = 1
	fk = f(x⁺)
	hk = h(x⁺)

	#do iterations
	optimal = false
	tired = k ≥ maxIter

	if options.verbose != 0
		@info @sprintf "%6s %8s %8s %7s %8s %7s" "iter" "f(x)" "h(x)" "‖∂ϕ‖" "ν" "‖x‖"
	end

	while !(optimal || tired)

		Fobj_hist[k] = fk
		Hobj_hist[k] = hk
		Complex_hist[k] += 1

		gold = g
		x = x⁺

		x⁺ = ShiftedProximalOperators.prox(h, x - ν*g, ν) 

		g = ∇f(x⁺)
		fk = f(x⁺)
		hk = h(x⁺)

		k+=1
		err = norm(g-gold - (x⁺-x)/ν)
		optimal = err < ϵ 
		tired = k ≥ maxIter

		k % ptf == 0 && @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e " k fk hk err ν norm(xk)
 
	end
	return x⁺, k, Fobj_hist[Fobj_hist .!= 0], Hobj_hist[Fobj_hist .!= 0], Complex_hist[Complex_hist .!= 0]
end

function PGΔ(
	f, 
	∇f, 
	h,
	options;
	x::AbstractVector=f.meta.x0,
	)

	ε=options.optTol
	maxIter=options.maxIter

	if options.verbose==0
			ptf = Inf
	elseif options.verbose==1
			ptf = round(maxIter/10)
	elseif options.verbose==2
			ptf = round(maxIter/100)
	else
			ptf = 1
	end
	#Problem Initialize
	ν = options.ν
	p = options.p 
	fDec = options.fDec

	k = 1
	feval = 1
	x⁺ = deepcopy(x)

	# Iteration set up
	g = ∇f(x⁺) #objInner/ quadratic model
	fk = f(x⁺)

	#do iterations
	optimal = false
	FD = false
	tired = k ≥ maxIter

	while !(optimal || tired || FD)

		gold = g
		fold = f
		x = x⁺

		x⁺ = ShiftedProximalOperators.prox(h, x - ν*g, ν)
		# update function info
		g = ∇f(x⁺)
		f = f(x⁺)

		feval+=1
		k+=1
		err = norm(g-gold - (x⁺-x)/ν)
		optimal =  err < ε
		tired = k ≥ maxIter

		k % ptf == 0 && @info @sprintf "%4d ‖xᵏ⁺¹ - xᵏ‖=%1.5e ν = %1.5e" k err ν

		Difff = fold + h(x) - f - h(x⁺) # these do not work 
		FD = abs(Difff)<p*norm(fDec)

	end
	return x⁺, feval
end



function PGE(f, h, s, options)

	ε=options.optTol
	maxIter=options.maxIter

	if options.verbose==0
			ptf = Inf
	elseif options.verbose==1
			ptf = round(maxIter/10)
	elseif options.verbose==2
			ptf = round(maxIter/100)
	else
			ptf = 1
	end
	#Problem Initialize
	ν = options.ν
	p = options.p 
	fDec = options.fDec

	k = 1
	feval = 1
	s⁺ = deepcopy(s)

	# Iteration set up
	g = ∇f(s⁺) #objInner/ quadratic model

	#do iterations
	FD = false 
	optimal = false
	tired = k ≥ maxIter 
	
	#do iterations
	while !(optimal || tired || FD)

		gold = g
		s = s⁺

		ν = min(g'*g/(g'*Bk*g), ν) #no BK, will not work 
		#prox step
		s⁺ = ShiftedProximalOperators.prox(h, s - ν*g, ν)
		# update function info
		g = ∇f(s⁺)
		f = f(s⁺)

		feval+=1
		k+=1
		err = norm(g-gold - (s⁺-s)/ν)
		optimal =  err < ε
		tired = k ≥ maxIter

		k % ptf == 0 && @info @sprintf "%4d ‖xᵏ⁺¹ - xᵏ‖=%1.5e ν = %1.5e" k err ν

		
		Difff = fold + h(s) - f - h(s⁺) # these do not work 
		FD = abs(Difff)<p*norm(fDec)
	end
	return s⁺, feval
end

function PGLnsch(f, ∇f, h, s, options)

	ε=options.optTol
	maxIter=options.maxIter

	if options.verbose==0
			ptf = Inf
	elseif options.verbose==1
			ptf = round(maxIter/10)
	elseif options.verbose==2
			ptf = round(maxIter/100)
	else
			ptf = 1
	end
	
	#Problem Initialize
	p = options.p
	ν₀ = options.ν
	k = 1
	s⁺ = deepcopy(s)

	# Iteration set up
	feval = 1
	g = ∇f(s⁺) #objInner/ quadratic model
	fk = f(s⁺)

	#do iterations
	optimal = false
	tired = k ≥ maxIter

	while !(optimal || tired)

		gold = g
		s = s⁺

		s⁺ = ShiftedProximalOperators.prox(h, s - ν*g, ν)
		#linesearch but we don't pass in f? 
		while f(s⁺) ≥ fk + g'*(s⁺ - s) + 1/(ν*2)*norm(s⁺ - s)^2
				ν *= p*ν
				s⁺ = prox(h, s - ν*g, ν)
				feval+=1
		end
		# update function info
		g = ∇f(s⁺) #objInner/ quadratic model
		fk = f(s⁺)

		feval+=1
		k+=1
		err = norm(g-gold - (s⁺-s)/ν)
		optimal = err < ε 
		tired = k ≥ maxIter

		k % ptf == 0 && @info @sprintf "%4d ‖xᵏ⁺¹ - xᵏ‖=%1.5e ν = %1.5e" k err ν
		ν = ν₀

	end
	return s⁺, feval
end