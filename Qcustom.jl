using LinearAlgebra

export Q_params, QCustom

mutable struct params
	grad
	Hess
end

function Q_params(; grad = Array{Float64,1}(undef,0), Hess = Array{Float64,2}(undef,0,0))
	return params(grad, Hess)
end




function QCustom(x, par)

	f = 0.5*x'*par.Hess*x + x'*par.grad;
    g = par.Hess*x +par.grad;
    h = par.Hess;

	return f, g, h
end

# function LScustom(x, grad, Hess)
# 	f = 0.5*norm(A*x-b)^2;

# 	if(nargout > 1)
#     	g = A'*(A*x - b);
# 	end

# 	if(nargout > 2)
#     	h = A'*A;
# 	end

# 	return f, g, H
# end