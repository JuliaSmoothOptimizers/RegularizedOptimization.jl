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

function proxgrad(x, β, Fcn, proxG, ε; max_iter = 10000, print_freq=100)
	#Problem Initialize
	m = length(x)
	gradF = zeros(m)
	y = copy(x)
	η = 1.0/β
	# Iteration set up
	k = 1
	err = 100
	f, gradF = Fcn(x, gradF)
	his = zeros(max_iter)
	#do iterations
	while err ≥ ε
		his[k] = f
		#take a gradient step: x-=η*∇f
		BLAS.axpy!(-η, gradF, x)
		#prox step
		x = proxG(x, η)
		# update function info
		f, gradF = Fcn(x, gradF)
		err = norm(x-y)
		y = copy(x)
		k+=1
		#sheet on which to freq
		k % print_freq ==0 && @printf("Iter %4d, Obj Val %1.5e, ‖xᵏ⁺¹ - xᵏ‖ %1.5e\n", k, f, err)

		if k ≥ max_iter
			@printf("Maximum Iterations Reached!\n")
			return his
		end
	end
	@printf("Error Criteria Reached! -> Obj Val %1.5e, ε = ‖xᵏ⁺¹ - xᵏ‖ %1.5e\n", f, err)
	return his[1:k]
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

function FISTA(x, β, Fcn, proxG, ε; max_iter = 10000, print_freq=100, restart = 100)
	#Problem Initialize
	m = length(x)
	gradF = zeros(m)
	xs = copy(x)
	y = copy(x)
	#initialize parameters
	η = β^(-1)
	λ = 1.0
	λs = copy(λ)
	# Iteration set up
	k = 1
	err = 100.0
	his = zeros(max_iter)
	converged = false
	#do iterations
	f, gradF = Fcn(y, gradF)

	while ~converged && k<max_iter
		if (mod(k, restart) == 1)
				λ = 1;
		end

		his[k] = f
		xs = copy(x)

		#update x
		#		x = y - η*gradF;
		x = copy(y)
		BLAS.axpy!(-η, gradF, x)
		x = proxG(x, η)

		#update step
		λs = copy(λ)
		λ = 0.5*(1 + sqrt(1+4*λs^2));

		#update y
		y = x + ((λs - 1.0)/λ)*(x-xs)

		#check convergence
		err = norm(x - xs);
		converged = err <= ε

		#sheet on which to freq
		k % print_freq ==0 && @printf("Iter %4d, Obj Val %1.5e, ‖xᵏ⁺¹ - xᵏ‖ %1.5e\n", k, f, err)

		#update parameters
		f, gradF = Fcn(y, gradF)
		k+=1
	end
	f, _ = Fcn(y, gradF)
	# @printf("Error Criteria Reached! -> Obj Val %1.5e, ε = ‖xᵏ⁺¹ - xᵏ‖ %1.5e\n", f, crit)
	return his[1:k]
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