export PG, PG!, PGLnsch, PGΔ

"""
	Proximal Gradient Descent  for
	min_x ϕ(x) = f(x) + g(x), with f(x) cvx and β-smooth, g(x) closed cvx

	Input:
        Fcn: function handle that returns f(x) and ∇f(x)
        Gcn: function handle that returns g(x)
		s: initial point
		proxG: function handle that calculates prox_{νg}
		options: see descentopts.jl
	Output:
		s⁺: s update
		s : s^(k-1)
        his : function history
        feval : number of function evals (total objective )
"""
function PG(Fcn, Gcn, s,  proxG, options)

	ε=options.optTol
	max_iter=options.maxIter

	if options.verbose==0
		print_freq = Inf
	elseif options.verbose==1
		print_freq = round(max_iter/10)
	elseif options.verbose==2
		print_freq = round(max_iter/100)
	else
		print_freq = 1
	end
	#Problem Initialize
	m = length(s)
	ν = options.ν
	λ = options.λ
	k = 1
	err = 100.0
	his = zeros(max_iter)
	s⁺ = deepcopy(s)

	# Iteration set up
	f, g = Fcn(s⁺) #objInner/ quadratic model
	fstart = f
	feval = 1

	#do iterations
	while err >= ε && k<max_iter && abs(f)>1e-16 #another stopping criteria abs(f - fstart)>TOL*||Δf(s1)||

		his[k] = f + Gcn(s⁺)*λ #Gcn = h(x)
		#sheet on which to freq
		k % print_freq ==0 && @printf("Iter %4d, Obj Val %1.5e, ‖xᵏ⁺¹ - xᵏ‖ %1.5e, ν = %1.5e\n", k, his[k], err, ν)

		gold = g
		s = s⁺

		#prox step
		s⁺ = proxG(s - ν*g, λ*ν) #combination regularizer + TR
		# update function info
		f, g = Fcn(s⁺)
		feval+=1

		# err = norm((s-s⁺)/ν) #stopping criteria
		err = norm(g-gold - (s⁺-s)/ν) #(Bk - ν^-1I)(s⁺ -s ) ----> equation 17 in paper 
		# err = norm((s-s⁺)/ν - gold) #equation 16 in paper
	
		k+=1
	end
	return s⁺,s, his[1:k-1], feval
end


function PG!(Fcn!,Gcn!, s,  proxG!, options)
	ε=options.optTol
	max_iter=options.maxIter

	if options.verbose==0
		print_freq = Inf
	elseif options.verbose==1
		print_freq = round(max_iter/10)
	elseif options.verbose==2
		print_freq = round(max_iter/100)
	else
		print_freq = 1
	end
	#get types 
	T = eltype(s)
	R = real(T)
	#Problem Initialize
	m = length(s)
	ν = options.ν
	λ = options.λ
	s⁻ = copy(s)
	g = zeros(T, m)

	k = 1
	err = 100.0
	his = zeros(max_iter)
	# Iteration set up
	f = Fcn!(s,g)
	feval = 1
	#do iterations
	while err > ε && abs(f)> 1e-16 && k < max_iter
		copy!(s⁻,s)
		his[k] = f + Gcn!(s) #shouldn't actually modify anything, just produce output 
		#prox step
		BLAS.axpy!(-ν,g,s)
		proxG!(s, ν*λ)
		err = norm(s-s⁻)
		# update function info
		f= Fcn!(s,g)
		feval+=1
		k+=1
		#sheet on which to freq
		k % print_freq ==0 && @printf("Iter %4d, Obj Val %1.5e, ‖xᵏ⁺¹ - xᵏ‖ %1.5e\n", k, f, err)
	end
	return s⁻, his[1:k-1], feval
end

function PGLnsch(Fcn,Gcn, s,  proxG, options)

	ε=options.optTol
	max_iter=options.maxIter

	if options.verbose==0
		print_freq = Inf
	elseif options.verbose==1
		print_freq = round(max_iter/10)
	elseif options.verbose==2
		print_freq = round(max_iter/100)
	else
		print_freq = 1
    end
    
    T = eltype(s)
    R = real(T)
    
    
	#Problem Initialize
    m = length(s)
    α = options.α
	# ν = R(1.0)
	ν = options.ν
	λ = options.λ
	k = 1
	err = 100.0
	his = zeros(max_iter)
	s⁺ = deepcopy(s)


	# Iteration set up
	f, g = Fcn(s⁺)
	feval = 1
	#do iterations
	while err >= ε && k<max_iter && abs(f)>1e-16
		s = s⁺
		his[k] = f + Gcn(s⁺)
		k % print_freq ==0 && @printf("Iter %4d, Obj Val %1.5e, ‖xᵏ⁺¹ - xᵏ‖ %1.5e\n", k, his[k], err)
		#prox step
        s⁺ = proxG(s - ν*g, λ*ν)
        #linesearch
        while Fcn(s⁺)[1] ≥ f + g'*(s⁺ - s) + 1/(ν*2)*norm(s⁺ - s)^2
            ν *= α*ν
            s⁺ = proxG(s - ν*g, λ*ν)
            feval+=1
        end
        # update function info
        ν = options.ν
        f, g = Fcn(s⁺)

		feval+=1
		err = norm(s-s⁺)
		k+=1

	end
	return s⁺,s, his[1:k-1], feval
end



function PGΔ(Fcn, Gcn, s,  proxG, options)

	ε=options.optTol
	max_iter=options.maxIter

	if options.verbose==0
		print_freq = Inf
	elseif options.verbose==1
		print_freq = round(max_iter/10)
	elseif options.verbose==2
		print_freq = round(max_iter/100)
	else
		print_freq = 1
	end
	#Problem Initialize
	m = length(s)
	ν = options.ν
	λ = options.λ


	p = options.p 
	FcnDec = options.FcnDec


	k = 1
	err = 100.0
	his = zeros(max_iter)
	s⁺ = deepcopy(s)

	# Iteration set up
	f, g = Fcn(s⁺) #objInner/ quadratic model
	fstart = f*100
	feval = 1

	#do iterations
	while err >= ε && k<max_iter && abs(f)>1e-16 && abs(f - fstart)<p*norm(FcnDec) #another stopping criteria abs(f - fstart)>TOL*||Δf(s1)||

		his[k] = f + Gcn(s⁺)*λ #Gcn = h(x)
		#sheet on which to freq
		k % print_freq ==0 && @printf("Iter %4d, Obj Val %1.5e, ‖xᵏ⁺¹ - xᵏ‖ %1.5e, ν = %1.5e\n", k, his[k], err, ν)

		gold = g
		s = s⁺

		#prox step
		s⁺ = proxG(s - ν*g, λ*ν) #combination regularizer + TR
		# update function info
		f, g = Fcn(s⁺)
		feval+=1

		# err = norm((s-s⁺)/ν) #stopping criteria
		err = norm(g-gold - (s⁺-s)/ν) #(Bk - ν^-1I)(s⁺ -s ) ----> equation 17 in paper 
		# err = norm((s-s⁺)/ν - gold) #equation 16 in paper
	
		k+=1
	end
	return s⁺,s, his[1:k-1], feval
end