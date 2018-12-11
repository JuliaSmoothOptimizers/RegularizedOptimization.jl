#===========================================================================
	Proximal Gradient Descent for 
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
		flag 0: exit normal
		flag 1: max iter exit
===========================================================================#

function proxgrad!(x, β, Fcn!, proxG!, ε; max_iter = 10000, print_freq=100)
	#Problem Initialize
	m = length(x)
	gradF = zeros(m)
	y = copy(x)
	η = 1.0/β
	# Iteration set up 
	k = 1
	err = 100
	f = Fcn!(x, gradF)
	his = zeros(max_iter)
	#do iterations
	while err ≥ ε
		his[k] = f
		#take a gradient step
		BLAS.axpy!(-η, gradF, x)
		#prox step
		proxG!(x, η)
		# update function info
		f = Fcn!(x, gradF)
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

function FISTA!(x, β, Fcn!, proxG!, ε; max_iter = 10000, print_freq=100)
	#Problem Initialize
	m = length(x)
	gradF = zeros(m)
	xs = copy(x)
	y = copy(x)
	#initialize parameters
	η = β^(-1)
	λ = 0.0
	λs = copy(λ)
	# Iteration set up 
	k = 1
	err = 100.0
	f = Fcn!(y, gradF)
	his = zeros(max_iter)
	#do iterations
	while err ≥ ε
		his[k] = f
		#take a step in x
		BLAS.axpy!(-η, gradF, y)
		#update y
		x = copy(y)
		proxG!(x, η) #updates x
		#update x
		λ = (1.0+sqrt(1.0+4.0*λs^2))/2.0
		y = x + (λs - 1.0)/λ*(x-xs)
		# update function info
		f = Fcn!(x, gradF)
		err = norm(xs - x)
		#sheet on which to freq 
		k % print_freq ==0 && @printf("Iter %4d, Obj Val %1.5e, ‖xᵏ⁺¹ - xᵏ‖ %1.5e\n", k, f, err)
		
		#update parameters
		xs = copy(x)
		λs = copy(λ)
		f = Fcn!(y, gradF)
		k+=1

		if k ≥ max_iter
			@printf("Maximum Iterations Reached!\n")
			return his
		end
	end
	@printf("Error Criteria Reached! -> Obj Val %1.5e, ε = ‖xᵏ⁺¹ - xᵏ‖ %1.5e\n", f, err)
	return his[1:k]
end



