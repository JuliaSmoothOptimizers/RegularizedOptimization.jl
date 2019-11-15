export barrier_alg



function barrier_alg(x,zl, zu,IPparams, IPoptions; is_cvx=0)
	mu = 1.0

	for iter=1:10000
		x, zl, zu, j = IntPt_TR(x, zl, zu,mu, IPparams, IPoptions) #changed zl and zu to be inputs
		if (is_cvx==1)
			#pretty much the same deal as below
			# mu = norm(zjl.*(x-l)) + norm(zju.*(u-x))
			mu = sum(-zjl.*(x-l) + zju.*(u-x))
		else
			mu = .9*mu
		end
		iter % 2 ==0 && @printf("Barrier Method: iter %4d, μ %1.5e\n", iter, mu)
		if(mu < 1e-6)
			break
		end
		IPparams.tr_options.optTol = .1*IPparams.tr_options.optTol
	end

	return x, zl, zu


end
