export s_options


mutable struct s_params
	optTol
	maxIter
	verbose
	restart
	β 
end


function s_options(β;optTol=1f-5, maxIter=10000, verbose=2, restart=100)

	return s_params(optTol, maxIter, verbose, restart, β)

end


#===========================================================================
	Proximal Gradient Descent for
	min_x ϕ(x) = f(x) + g(x), with f(x) cvx and β-smooth, g(x) closed cvx

	Input:
		x: initial point
		β: Lipschitz constant for F
		Fcn: function handle that returns f(x) and ∇f(x)
		proxG: function handle that calculates prox_{ηg}
		ε: tolerance, where ||x^{k+1} - x^k ||⩽ ε
		* max_iter: self-explanatory
		* print_freq: # of freq's in the sheets
	Output:
		x: x update
		flag 0: exit normal
		flag 1: max iter exit
===========================================================================#
function PG(Fcn, x,  proxG, options)
	ε=options.optTol
	max_iter=options.maxIter
	
	if options.verbose==0
		print_freq = Inf
	elseif options.verbose==1
		print_freq = round(max_iter/10)
	elseif options.verbose==2
		print_freq = round(max_iter/100)
	else 
		print_freq = round(max_iter/200)
	end
	#Problem Initialize
	m = length(x)
	η = 1.0/options.β
	k = 1
	err = 100
	his = zeros(max_iter)


	# Iteration set up
	f, gradF = Fcn(x)
	
	#do iterations
	while err ≥ ε && k <max_iter
		his[k] = f
		#take a gradient step: x-=η*∇f
		# BLAS.axpy!(-η, gradF, x1)
		#prox step
		xp = proxG(x - η*gradF, η)
		# update function info
		f, gradF = Fcn(xp)
		err = norm(x-xp)
		x = xp
		k+=1
		#sheet on which to freq
		k % print_freq ==0 && @printf("Iter %4d, Obj Val %1.5e, ‖xᵏ⁺¹ - xᵏ‖ %1.5e\n", k, f, err)

	end
	@printf("Error Criteria Reached! -> Obj Val %1.5e, ε = ‖xᵏ⁺¹ - xᵏ‖ %1.5e\n", f, err)
	return x, his[1:k-1]
end
#===========================================================================
	FISTA for
	min_x ϕ(x) = f(x) + g(x), with f(x) cvx and β-smooth, g(x) closed cvx

	Input:
		x: initial point
		β: Lipschitz constant for F
		Fcn!: function handle that returns f(x) and ∇f(x)
		proxG!: function handle that calculates prox_{ηg}
		ε: tolerance, where ||x^{k+1} - x^k ||⩽ ε
		* max_iter: self-explanatory
		* print_freq: # of freq's in the sheets
	Output:
		x: x update
===========================================================================# 

function FISTA(Fcn, x,  proxG, options)
	ε=options.optTol
	max_iter=options.maxIter
	restart = options.restart 
	η = options.β^(-1)
	if options.verbose==0
		print_freq = Inf
	elseif options.verbose==1
		print_freq = round(max_iter/10)
	elseif options.verbose==2
		print_freq = round(max_iter/100)
	else 
		print_freq = round(max_iter/200)
	end

	#Problem Initialize
	m = length(x)
	y = zeros(m)


	#initialize parameters
	t = 1.0
	tk = t
	# Iteration set up
	k = 1
	err = 100.0
	his = zeros(max_iter)
	
	#do iterations
	f, gradF = Fcn(y)

	while err >= ε && k<max_iter
		# if (mod(k, restart) == 1)
		# 		t = 1;
		# end

		his[k] = f
		xk = copy(x)
		x = proxG(y - η*gradF, η)

		#update x
		#		x = y - η*gradF;
		# BLAS.axpy!(-η, gradF, x)
		# x = proxG(y - η*gradF, η)

		#update step
		tk = t
		t = 0.5*(1.0 + sqrt(1.0+4.0*tk^2))

		#update y
		y = x + ((tk - 1.0)/t)*(x-xk)

		#check convergence
		err = norm(x - xk)


		#sheet on which to freq
		k % print_freq ==0 && @printf("Iter %4d, Obj Val %1.5e, ‖xᵏ⁺¹ - xᵏ‖ %1.5e\n", k, f, err)

		#update parameters
		f, gradF = Fcn(y)
		k+=1
	end
	# f, _ = Fcn(y)
	@printf("Error Criteria Reached! -> Obj Val %1.5e, ε = ‖xᵏ⁺¹ - xᵏ‖ %1.5e\n", f, err)
	return x, his[1:k-1]
end




function test_conv(x, Fcn, proxG,η, critin=1.0)
	m = length(x)
	gradF = zeros(m)
	xtest = copy(x)
	f, gradF = Fcn(xtest, gradF)
	BLAS.axpy!(-1.0, gradF, xtest)
	xtest = proxG(xtest, η)
	crit = norm(x - xtest)/critin

	return crit
end