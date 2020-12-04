export FISTA, FISTA!, FISTAD
"""
	FISTA for
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
		feval : number of function evals (total objective)
"""
function FISTA(Fcn, Gcn, s,  proxG, options)
	ε=options.optTol
	max_iter=options.maxIter
	restart = options.restart
	ν = options.ν
	λ = options.λ
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
	y = deepcopy(s)
	s⁺ = zeros(T, m)
	#initialize parameters
	t = R(1.0)
	# Iteration set up
	k = 1
	err = R(100.0)
	his = zeros(max_iter)

	#do iterations
	f, g = Fcn(y)
	feval = 1
	while err >= ε && k<max_iter && abs(f)>1e-16
		copy!(s,s⁺)
		his[k] = f + Gcn(y)
		s⁺ = proxG(y - ν*g, ν*λ)

		#update step
		t⁻ = t
		# t = 0.5*(1.0 + sqrt(1.0+4.0*t⁻^2))
		t = R(0.5)*(R(1.0) + sqrt(R(1.0)+R(4.0)*t⁻^2))

		#update y
		y = s⁺ + ((t⁻ - R(1.0))/t)*(s⁺-s)

		#check convergence
		err = norm(s - s⁺)


		#sheet on which to freq
		k % print_freq ==0 && @printf("Iter %4d, Obj Val %1.5e, ‖xᵏ⁺¹ - xᵏ‖ %1.5e\n", k, f, err)

		#update parameters
		f, g = Fcn(y)

		feval+=1
		k+=1
	end
	return s⁺,s, his[1:k-1], feval

end

function FISTA!(Fcn!,Gcn!, s,  proxG!, options)
	ε=options.optTol
	max_iter=options.maxIter
	restart = options.restart
	ν = options.ν
	λ = options.λ
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
	y = deepcopy(s)
	s⁻ = zeros(T, m)
	gradF = zeros(T, m)


	#initialize parameters
	t = R(1.0)
	# Iteration set up
	k = 1
	err = R(100.0)
	his = zeros(max_iter)

	#do iterations
	f = Fcn!(y, gradF)
	feval = 1
	while ε<err && abs(f) >1e-16 && k<max_iter
		copy!(s⁻, s)

		his[k] = f + Gcn!(y) #note: shouldn't actually modify anything, just produce output
		BLAS.axpy!(-ν,gradF,y)
		copy!(s, y)
		proxG!(s, ν*λ)

		#update step
		t⁺ = R(0.5)*(R(1.0) + sqrt(R(1.0)+R(4.0)*t^2))

		#update y
		y = s + ((t - R(1.0))/t⁺)*(s-s⁻)

		#check convergence
		err = norm(s - s⁻)

		#sheet on which to freq
		k % print_freq ==0 && @printf("Iter %4d, Obj Val %1.5e, ‖xᵏ⁺¹ - xᵏ‖ %1.5e\n", k, f, err)
		#update parameters
		f = Fcn!(y, gradF)
		t = t⁺

		feval+=1
		k+=1
	end

	return s⁻, his[1:k-1], feval
end

#enforces strict descent  for FISTA 
function FISTAD(Fcn, Gcn, s,  proxG, options)
	ε=options.optTol
	max_iter=options.maxIter
	restart = options.restart
	ν = options.ν
	λ = options.λ
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
	v = deepcopy(s)
	s⁺ = zeros(T, m)
	#initialize parameters
	t = R(1.0)
	# Iteration set up
	k = 1
	err = R(100.0)
	his = zeros(max_iter)

	#do iterations
	y = (R(1.0)-t)*s + t*v
	f, g = Fcn(y) 

	feval = 1
	while err >= ε && k<max_iter && abs(f)>1e-16
		copy!(s,s⁺)
		his[k] = f + Gcn(y)

		#complete prox step 
		u = proxG(y - ν*g, ν*λ)

		if Fcn(u)[1] ≤ f
			s⁺ = u
		else
			s⁺ = s
		end

		#update step
		# t⁻ = t
		# t = R(0.5)*(R(1.0) + sqrt(R(1.0)+R(4.0)*t⁻^2))
		t = R(2/(k + 1))

		#update y
		# v = s⁺ + ((t⁻ - R(1.0))/t)*(s⁺-s)
		v = s⁺ + (R(1.0)/t)*(u - s⁺)
		y = (R(1.0)-t)*s⁺ + t*v #I think this shold be s⁺ since it's at the end of the loop 

		#check convergence
		err = norm(s - s⁺)


		#sheet on which to freq
		k % print_freq ==0 && @printf("Iter %4d, Obj Val %1.5e, ‖xᵏ⁺¹ - xᵏ‖ %1.5e\n", k, f, err)

		#update parameters
		f, g = Fcn(y)

		feval+=1
		k+=1
	end
	return s⁺,s, his[1:k-1], feval

end