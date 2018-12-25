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

function FISTA(x0, β, Fcn, proxG, ε; max_iter = 10000, print_freq=1000)

	#function parameters
	restart = 100
	η = 1/β
	m = length(x0)
	gradF = zeros(m)
	his = zeros(max_iter)
	#initialize function
	critin = test_conv(x0, Fcn, proxG, η)
	crit = copy(critin)
	#Problem Initialize
	k = 0
	fstep = 1.0
	x = copy(x0)
	y = copy(x0)
	converged=false
	while (~converged && k <max_iter)
		k +=1
		if k % restart ==1
			fstep = 1.0
		end

		f, gradF = Fcn(y, gradF)
		his[k]=f
		# x⁺=copy(y)
		#x⁺ -= η*∇f
		# BLAS.axpy!(-η, gradF, x⁺) 

		x⁺ = proxG(BLAS.axpy!(-η, gradF, y), η)
		stepNew = 0.5*(1.0+sqrt(1.0+4.0*fstep^2))
		y = x + ((fstep -1.0)/stepNew)*(x⁺ - x)
		fstep = copy(stepNew)
		x = copy(x⁺)


		#test to see if you've converged
		crit = test_conv(x, Fcn, proxG, η, critin)
		converged = (crit < ε)

		#sheet on which to freq
		k % print_freq ==0 && @printf("Iter %4d, Obj Val %1.5e, ‖xᵏ⁺¹ - xᵏ‖ %1.5e\n", k, f, crit)
	end
	f, _ = Fcn(y, gradF)
	@printf("Error Criteria Reached! -> Obj Val %1.5e, ε = ‖xᵏ⁺¹ - xᵏ‖ %1.5e\n", f, crit)
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