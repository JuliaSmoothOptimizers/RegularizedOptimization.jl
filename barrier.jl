using LinearAlgebra, Printf #include necessary packages
# include("IP_alg.jl")



function barrier_alg(x,zl, zu,IPparams, IPoptions)
	mu = 1.0

	for iter=1:1000
		x, zj, zj, j = IntPt_TR(x, zl, zu,mu, parameters, options)
		mu = .9*mu
		iter % 2 ==0 && @printf("Barrier iter %4d\n", iter)
		if(mu < 1e-6)
			break
		end
	end

	return x, zl, zu


end
