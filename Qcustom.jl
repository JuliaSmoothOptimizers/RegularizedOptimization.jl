using LinearAlgebra, Printf

export Q_params, QCustom

mutable struct params
	grad
	Hess
end

function Q_params(; grad = Vector{Float64}(undef,0), Hess = Array{Float64,2}(undef,0,0))
	return params( grad, Hess)
end




function QCustom(s, par)
	"""
	QCustom is the quadratic approximation for the smooth part of the function f(x)

	Parameters
    ----------
    s : Array{Float64,1}
        Search direction computed by the TR method
    par.grad : Array{Float64,1}
        gradient of ϕ at x
    par.Hess : Array{Float64,2}
        Hessian (or Hessian approximation) of TR method 
	"""
	
	Hess = par.Hess;
	grad = par.grad;
	f = 0.5*(s'*(Hess*s)) + grad'*s; #technically don't need obj but for completeness sake
    g = Hess*s +grad;
    h = Hess;
	return f, g, h
end
