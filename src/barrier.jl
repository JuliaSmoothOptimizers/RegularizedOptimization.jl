export barrier_alg



function barrier_alg(x,zl, zu,IPparams, IPoptions; is_cvx=0)
	mu = 1.0
	IterCount = 0.0
	for iter=1:10000
		x, zl, zu, k = IntPt_TR(x, zl, zu,mu,IterCount, IPparams, IPoptions) #changed zl and zu to be inputs
		if (is_cvx==1)
			#pretty much the same deal as below
			# mu = norm(zjl.*(x-l)) + norm(zju.*(u-x))
			mu = sum(-zl.*(x-l) + zu.*(u-x))
		else
			mu = mu/10.0
		end
		# iter % 2 ==0 && @printf("Barrier Method: iter %4d, μ %1.5e\n", iter, mu)
		if(mu < 1e-6)
			break
		end
		IPparams.FO_options.optTol = .1*IPparams.FO_options.optTol
		IterCount = k
	end

	return x, zl, zu


end
