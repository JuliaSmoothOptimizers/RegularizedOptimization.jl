export s_options

mutable struct s_params
	optTol
	maxIter
	verbose
	restart
	ν
	λ
	p
	FcnDec
end


function s_options(ν; optTol=1f-6, maxIter=10000, verbose=0, restart=10, λ=1.0, p = 1.1, FcnDec = 1)

	return s_params(optTol, maxIter, verbose, restart, ν, λ, p, FcnDec)

end
