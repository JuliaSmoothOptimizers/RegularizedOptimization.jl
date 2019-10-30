export s_options


mutable struct s_params
	optTol
	maxIter
	verbose
	restart
	β 
	
end


function s_options(β ;optTol=1f-5, maxIter=10000, verbose=2, restart=100)

	return s_params(optTol, maxIter, verbose, restart, β )

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
function PG(Fcn, x,  projG, options)
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
	# λ = options.λ
	k = 1
	err = 100
	his = zeros(max_iter)


	# Iteration set up
	f, gradF = Fcn(x)
	feval = 1
	#do iterations
	while err ≥ ε && k <max_iter
		his[k] = f
		#take a gradient step: x-=η*∇f
		# BLAS.axpy!(-η, gradF, x1)
		#prox step
		xp = projG(x - η*gradF)
		# update function info
		f, gradF = Fcn(xp)
		feval+=1
		err = norm(x-xp)
		x = xp
		k+=1
		#sheet on which to freq
		k % print_freq ==0 && @printf("Iter %4d, Obj Val %1.5e, ‖xᵏ⁺¹ - xᵏ‖ %1.5e\n", k, f, err)

	end
	# @printf("Error Criteria Reached! -> Obj Val %1.5e, ε = ‖xᵏ⁺¹ - xᵏ‖ %1.5e\n", f, err)
	return x, his[1:k-1], feval
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

function FISTA(Fcn, x,  projG, options)
	ε=options.optTol
	max_iter=options.maxIter
	restart = options.restart 
	η = options.β^(-1)
	# λ = options.λ
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
	feval = 1
	while err >= ε && k<max_iter
		# if (mod(k, restart) == 1)
		# 		t = 1;
		# end

		his[k] = f
		xk = copy(x)
		x = projG(y - η*gradF)

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
		feval+=1
		k+=1
	end
	# f, _ = Fcn(y)
	# @printf("Error Criteria Reached! -> Obj Val %1.5e, ε = ‖xᵏ⁺¹ - xᵏ‖ %1.5e\n", f, err)
	return x, his[1:k-1], feval
end


function linesearch(x, zjl, zju, s, dzl, dzu ;mult=.9, tau = .01)
	α = 1.0
	     while( 
            any(x + α*s - l .< (1-tau)*(x-l)) || 
            any(u - x - α*s .< (1-tau)*(u-x)) ||
            any(zjl + α*dzl .< (1-tau)*zjl) || 
            any(zju + α*dzu .< (1-tau)*zju)
            )
            α = α*mult

        end
        return α
end

function directsearch(x, zjl, zju, s, dzl, dzu; tau = .01) #used to be .01
	temp = [(-tau *(x-l))./s; (-tau*(u-x))./-s; (-tau*zjl)./dzl; (-tau*zju)./dzu]
    temp=filter((a) -> 1>=a>0, temp)
    # @printf("%1.5e | %1.5e | %1.5e | %1.5e \n", maximum(abs.((u - x)./s)), maximum(abs.((x-l)./s)), maximum(abs.(-zjl./dzl)), maximum(abs.(-zju./dzu)))
    temp = minimum(vcat(temp, 1.0))

	return temp


end
function directsearch!(α, zjl, zju, s, dzl, dzu; tau = .01) #used to be .01
	temp = [(-tau *(x-l))./s; (-tau*(u-x))./-s; (-tau*zjl)./dzl; (-tau*zju)./dzu]
    temp=filter((a) -> 1>=a>0, temp)
    α = minimum(vcat(temp, 1.0))
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