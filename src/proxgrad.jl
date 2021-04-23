export PG, PGLnsch, PGΔ, PGE, PGnew

using Printf
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
	f, g, _ = Fcn(s⁺) #objInner/ quadratic model
	fstart = f
	feval = 1

	#do iterations
	while err >= ε && k<max_iter #another stopping criteria abs(f - fstart)>TOL*||Δf(s1)||

		his[k] = f + Gcn(s⁺)*λ #Gcn = h(x)
		#sheet on which to freq
		
		gold = g
		s = s⁺

		#prox step
		s⁺ = proxG(s - ν*g, λ*ν) #combination regularizer + TR
		# update function info
		f, g = Fcn(s⁺)[1:2]
		feval+=1

		# err = norm((s-s⁺)/ν) #stopping criteria
		err = norm(g-gold - (s⁺-s)/ν) #(Bk - ν^-1I)(s⁺ -s ) ----> equation 17 in paper 
		# err = norm((s-s⁺)/ν - gold) #equation 16 in paper
	
		k+=1
	end
	return s⁺, his[1:k-1], feval
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
    p = options.p
	# ν = R(1.0)
	ν = options.ν
	λ = options.λ
	k = 1
	err = 100.0
	his = zeros(max_iter)
	s⁺ = deepcopy(s)


	# Iteration set up
	f, g = Fcn(s⁺)[1:2]
	feval = 1
	#do iterations
	while err >= ε && k<max_iter
		s = s⁺
		his[k] = f + Gcn(s⁺)
		#prox step
        s⁺ = proxG(s - ν*g, λ*ν)
        #linesearch
        while Fcn(s⁺)[1] ≥ f + g'*(s⁺ - s) + 1/(ν*2)*norm(s⁺ - s)^2
            ν *= p*ν
            s⁺ = proxG(s - ν*g, λ*ν)
            feval+=1
        end
        # update function info
        ν = options.ν
		f, g = Fcn(s⁺)[1:2]

		feval+=1
		err = norm(s-s⁺)
		k+=1

	end
	return s⁺, his[1:k-1], feval
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
	f, g = Fcn(s⁺)[1:2] #objInner/ quadratic model
    his[k] = f + Gcn(s⁺)*λ #Gcn = h(x)
	DiffFcn = 0.0 
    feval = 1
	#do iterations
	while err >= ε && k<max_iter && abs(DiffFcn)<p*norm(FcnDec) #another stopping criteria abs(f - fstart)>TOL*||Δf(s1)||

		#sheet on which to freq

		gold = g
		s = s⁺

		#prox step
		s⁺ = proxG(s - ν*g, λ*ν) #combination regularizer + TR
		# update function info
		f, g = Fcn(s⁺)[1:2]
		feval+=1

		# err = norm((s-s⁺)/ν) #stopping criteria
		err = norm(g-gold - (s⁺-s)/ν) #(Bk - ν^-1I)(s⁺ -s ) ----> equation 17 in paper 
		# err = norm((s-s⁺)/ν - gold) #equation 16 in paper
        k+=1

        
        his[k] = f + Gcn(s⁺)*λ #Gcn = h(x)
        DiffFcn = his[k-1] - his[k]
	end
	return s⁺, his[1:k-1], feval
end



function PGE(Fcn, Gcn, s,  proxG, options)

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
    f, g, Bk = Fcn(s⁺) #objInner/ quadratic model
    his[k] = f + Gcn(s⁺)*λ #Gcn = h(x)
	DiffFcn = 0.0 
    feval = 1
	#do iterations
	while err >= ε && k<max_iter && abs(DiffFcn)<p*norm(FcnDec) #another stopping criteria abs(f - fstart)>TOL*||Δf(s1)||

		#sheet on which to freq
		gold = g
		s = s⁺

		ν = min(g'*g/(g'*Bk*g), ν)
		#prox step
		s⁺ = proxG(s - ν*g, λ*ν) #combination regularizer + TR
		# update function info
		f, g, Bk = Fcn(s⁺)
		feval+=1

		# err = norm((s-s⁺)/ν) #stopping criteria
		err = norm(g-gold - (s⁺-s)/ν) #(Bk - ν^-1I)(s⁺ -s ) ----> equation 17 in paper 
		# err = norm((s-s⁺)/ν - gold) #equation 16 in paper
        k+=1

        
        his[k] = f + Gcn(s⁺)*λ #Gcn = h(x)
        DiffFcn = his[k-1] - his[k]
	end
	return s⁺, his[1:k-1], feval
end

function PGnew(GradFcn, Gcn, s, options)

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
	g = GradFcn(s⁺) #objInner/ quadratic model
	feval = 1
	k % print_freq == 0 && @info @sprintf "%4d ‖xᵏ⁺¹ - xᵏ‖=%1.5e ν = %1.5e" k err ν

	#do iterations
	while err >= ε && k<max_iter #another stopping criteria abs(f - fstart)>TOL*||Δf(s1)||

		#sheet on which to freq
		
		gold = g
		s = s⁺

		#prox step
		s⁺ = prox(Gcn, s - ν*g, ν) #combination regularizer + TR
		# update function info
		g = GradFcn(s⁺)
		feval+=1

		# err = norm((s-s⁺)/ν) #stopping criteria
		err = norm(g-gold - (s⁺-s)/ν) #(Bk - ν^-1I)(s⁺ -s ) ----> equation 17 in paper 
		# err = norm((s-s⁺)/ν - gold) #equation 16 in paper
		k % print_freq == 0 && @info @sprintf "%4d ‖xᵏ⁺¹ - xᵏ‖=%1.5e ν = %1.5e" k err ν

		k+=1
	end
	return s⁺, feval
end