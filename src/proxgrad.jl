export PG, PG!, PGLnsch

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
	θ = options.θ
	β = options.β
	νmin = 1-sqrt(1-4*θ)/(2*β)
	νmax = 1+sqrt(1-4*θ)/(2*β)

	ν = νmax
	λ = options.λ
	k = 1
	err = 100
	his = zeros(max_iter)
	s⁺ = deepcopy(s)


	# Iteration set up
	f, g = Fcn(s⁺)
	fstart = f
	feval = 1
	#do iterations
	while err >= ε && k<max_iter && abs(f)>1e-16
		s = s⁺
		his[k] = f + Gcn(s⁺)

		#prox step
		s⁺ = proxG(s - ν*g, λ*ν)
		# update function info
		f, g = Fcn(s⁺)
		feval+=1
		err = norm(s-s⁺)
		k+=1
		if f>fstart || isnan(norm(s⁺))
			s⁺ = s
			ν = νmin #can you make this larger if ||Bk|| sucks? 
			err = 100
		end
		#sheet on which to freq
		k % print_freq ==0 && @printf("Iter %4d, Obj Val %1.5e, ‖xᵏ⁺¹ - xᵏ‖ %1.5e, ν = %1.5e\n", k, f, err, ν)
	end
	@show fstart - f
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
	ν = options.β^(-1)
	λ = options.λ
	s⁻ = copy(s)
	g = zeros(T, m)

	k = 1
	err = 100
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
	ν = R(1.0)
	λ = options.λ
	k = 1
	err = 100
	his = zeros(max_iter)
	s⁺ = deepcopy(s)


	# Iteration set up
	f, g = Fcn(s⁺)
	feval = 1
	#do iterations
	while err >= ε && k<max_iter && abs(f)>1e-16
		s = s⁺
		his[k] = f + Gcn(s⁺)
		#prox step
        s⁺ = proxG(s - ν*g, λ*ν)
        #linesearch
        while Fcn(s⁺)[1] ≥ f + g'*(s⁺ - s) + 1/(ν*2)*norm(s⁺ - s)^2
            ν *= α*ν
            s⁺ = proxG(s - ν*g, λ*ν)
            feval+=1
        end
        # update function info
        ν = R(1.0)
        f, g = Fcn(s⁺)

		feval+=1
		err = norm(s-s⁺)
		k+=1
		#sheet on which to freq
		k % print_freq ==0 && @printf("Iter %4d, Obj Val %1.5e, ‖xᵏ⁺¹ - xᵏ‖ %1.5e\n", k, f, err)
	end
	return s⁺,s, his[1:k-1], feval
end