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
	xm = copy(x)
	y = copy(x)
	η = 1.0/β
	t = 1.0
	# Iteration set up 
	k = 1
	err = 100
	f = Fcn!(y, gradF)
	his = zeros(max_iter)
	#do iterations
	while err ≥ ε
		his[k] = f
		#take a step in x
		# proxG!(BLAS.axpy!(-η, gradF, y), η) #should reassign x, check to make sure this input or just y
		x = copy(y)
		proxG!(x, η)
		tm = copy(t)
		t = (1.0+sqrt(1.0+4.0*tm^2))/2.0
		#prox step
		y = x + ((tm-1.0)/t)*(x-xm)
		# update function info
		f = Fcn!(y, gradF)
		err = norm(y-x)
		xm = copy(x)
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



