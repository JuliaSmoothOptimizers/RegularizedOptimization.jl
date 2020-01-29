export barrier_alg



function barrier_alg(x,zl, zu,IPparams, IPoptions; is_cvx=0, mu_tol =1e-10)
	mu = 1.0
	IterCount = 0.0
	#pretty much the same deal as below
	l, u = IPparams.l, IPparams.u
	#take into account Moreau Envelope
	# f + h st c(x)≦0 - approximation of gradient 
	for iter=1:10000
		x, zl, zu, k = IntPt_TR(x, zl, zu,mu,IterCount, IPparams, IPoptions) #changed zl and zu to be inputs
		if (is_cvx==1)
			mu = norm(zl.*(x.-l)) + norm(zu.*(u.-x))
			# mu = sum(-zl.*(x-l) + zu.*(u-x)) #work in progress
		else
			mu = mu*.5
		end
		if(mu < mu_tol)
			break
		end
		IPparams.FO_options.optTol = .1*IPparams.FO_options.optTol #default is 1e-10
		IterCount = k #overall iteration
		IPoptions.epsC = 10*mu #updates epsC
		IPoptions.epsD = 10*mu #updates epsC
	end

	return x, zl, zu


end
