import ProximalAlgorithms: LBFGS, Maybe, ZeroFPR, ZeroFPR_iterable, ZeroFPR_state
import ProximalAlgorithms.IterationTools: halt, sample, tee, loop
using Base.Iterators  # for take
# mutable struct LeastSquaresObjective
# 	smooth
#     nonsmooth
# 	hist 
#     count 
# end
	  
# definition that will allow to simply evaluate the objective: ϕ(x)
function (f::LeastSquaresObjective)(x)
    r = f.smooth(x)[1]
	append!(f.hist, r + λ*f.nonsmooth(x))
    f.count +=1 
	return r
end

# first import gradient and gradient! to extend them
import ProximalOperators.gradient, ProximalOperators.gradient!

function gradient!(∇fx, f::LeastSquaresObjective, x)
	r, g = f.smooth(x)
	∇fx .= g
    append!(f.hist, r + λ*f.nonsmooth(x))
    f.count +=1 
	return r
end

function gradient(f::LeastSquaresObjective, x)
	∇fx = similar(x)
	fx = gradient!(∇fx, f, x)
	return ∇fx, fx
end

# PANOC checks if the objective is quadratic
import ProximalOperators.is_quadratic
is_quadratic(::LeastSquaresObjective) = true

import ProximalOperators.prox, ProximalOperators.prox!
function (h::ProxOp)(x)
	r = h.func(x)
	h.count += 1
	return f.λ*r
end
# state.g_z = prox!(state.z, iter.g, state.y, state.gamma)
function prox!(z, h::ProxOp, y, gamma)
	h.count+=1
	z.= h.proxh(y, gamma)
	return h.λ*h.func(z)
end

# # z, g_z = prox(iter.g, y, gamma)
function prox(h::ProxOp, y, gamma)
	z = similar(y)
	h.count+=1
	nsval = prox!(z, h, y, gamma)
	return z, nsval 
end

function my_zerofpr(solver::ZeroFPR{R},
    x0::AbstractArray{C};
    f = Zero(),
    A = I,
    g = Zero(),
    L::Maybe{R} = nothing,
    ) where {R,C<:Union{R,Complex{R}}}
    stop(state::ZeroFPR_state) = norm(state.res, Inf) / state.gamma <= solver.tol
    function disp((it, state))
        @printf(
                "%5d | %.3e | %.3e | %.3e | %9.2e | %9.2e\n",
                it,
                state.gamma,
                norm(state.res, Inf) / state.gamma,
                (state.tau === nothing ? 0.0 : state.tau),
                state.f_Ax,  # <-- added this
                state.g_xbar    # <-- and this
            )
    end

    gamma = if solver.gamma === nothing && L !== nothing
        solver.alpha / L
    else
        solver.gamma
    end

    iter = ZeroFPR_iterable(
        f,
        A,
        g,
        x0,
        solver.alpha,
        solver.beta,
        gamma,
        solver.adaptive,
        LBFGS(x0, solver.memory),
    )
    iter = take(halt(iter, stop), solver.maxit)
    iter = enumerate(iter)
    if solver.verbose
        iter = tee(sample(iter, solver.freq), disp)
    end

    num_iters, state_final = loop(iter)

    return state_final.x, num_iters
end