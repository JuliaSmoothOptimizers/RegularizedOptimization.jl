export qk


function qk(s, grad, Hess)
	"""
	qk is the quadratic approximation for the smooth part of the function f(x)

	Arguments
    ----------
    s : Array{Float64,1}
        Search direction computed by the TR method
    grad : Array{Float64,1}
        gradient of q at x
    Hess : Array{Float64,2}
        Hessian (or Hessian approximation) of TR method
	"""

	f = 0.5*(s'*(Hess*s)) + grad'*s
    g = Hess*s +grad
    h = Hess
	return f, g, h
end

#next function is already in ProxProj.jl
# function prox_dual(x, step, prox_primal)
# 	"""
# 	Function for easily computing proximal operator of conjugate based on the Moreau decomposition

# 	Arguments
# 	----------

# 	x: Array{float64,1}
# 		input for which you are evaluating the prox
# 	η: float64
# 		step size of the prox
# 	prox_primal: Function
# 		primal proximal operator of function whose conjugate you are trying to evaluate
# 		form - TO ADD

# 	"""
#     out = x - η*prox_primal((1/η)*x, 1/η)
#     return out
# end
