# Tseng, "On Accelerated Proximal Gradient Methods for Convex-Concave
# Optimization" (2008).
#
# Beck, Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm
# for Linear Inverse Problems", SIAM Journal on Imaging Sciences, vol. 2,
# no. 1, pp. 183-202 (2009).

using Base.Iterators
using LinearAlgebra
using Printf
include("iterationtools.jl")
using Main.IterationTools

struct FBS_iterable{R<:Real,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,Tpg,Tg}
    f::Tf             # smooth term
    prox::Tpg
    g::Tg             # (possibly) nonsmooth, proximable term
    x0::Tx            # initial point
    gamma::R   # stepsize parameter of forward and backward steps
    adaptive::Bool    # enforce adaptive stepsize even if L is provided
    fast::Bool
end

Base.IteratorSize(::Type{<:FBS_iterable}) = Base.IsInfinite()

mutable struct FBS_state{R<:Real,Tx}
    x::Tx             # iterate
    f_x::R           # value of smooth term
    grad_f_x::Tx    # gradient of f at Ax
    gamma::R          # stepsize parameter of forward and backward steps
    y::Tx             # forward point
    z::Tx             # forward-backward point
    g_z::R            # value of nonsmooth term (at z)
    res::Tx           # fixed-point residual at iterate (= z - x)
    theta::R
    z_prev::Tx
end

f_model(state::FBS_state) = f_model(state.f_x, state.grad_f_x, state.res, state.gamma)

function Base.iterate(iter::FBS_iterable{R}) where {R}
    x = iter.x0
    f_x, grad_f_x= iter.f(x)
    @show size(x)
    gamma = iter.gamma

    if gamma === nothing
        # compute lower bound to Lipschitz constant of the gradient of x ↦ f(Ax)
        xeps = x .+ R(1)
        f_xeps, grad_f_xeps = iter.f(xeps)
        L = norm(grad_f_xeps - grad_f_x) / R(sqrt(length(x)))
        gamma = R(1) / L
    end

    # compute initial forward-backward step
    y = x - gamma .* grad_f_x
    z = iter.prox(y, gamma)
    g_z = iter.g(z)
    # compute initial fixed-point residual
    res = x - z

    state = FBS_state(
        x,
        f_x,
        grad_f_x,
        gamma,
        y,
        z,
        g_z,
        res,
        R(1),
        copy(x),
    )

    return state, state
end

function Base.iterate(iter::FBS_iterable{R}, state::FBS_state{R,Tx}) where {R,Tx}
    z, f_z, grad_f_z= nothing, nothing, nothing
    a, b, c = nothing, nothing, nothing

    # backtrack gamma (warn and halt if gamma gets too small)
    while iter.gamma === nothing || iter.adaptive == true
        if state.gamma < 1e-7 # TODO: make this a parameter, or dependent on R?
            @warn "parameter `gamma` became too small ($(state.gamma)), stopping the iterations"
            return nothing
        end
        f_z_upp = f_model(state)
        z =  state.z
        f_z, grad_f_z = iter.f(z)
        tol = 10 * eps(R) * (1 + abs(f_Az))
        if f_z <= f_z_upp + tol
            break
        end
        state.gamma *= 0.5
        state.y .= state.x .- state.gamma .* state.grad_f_x
        state.z = iter.prox(state.y, state.gamma)
        state.g_z = iter.g(state.z)
        state.res .= state.x .- state.z
    end

    if iter.fast == true
        theta1 = (R(1) + sqrt(R(1) + 4 * state.theta^2)) / R(2)
        extr = (state.theta - R(1)) / theta1
        state.theta = theta1
        state.x .= state.z .+ extr .* (state.z .- state.z_prev)
        state.z_prev, state.z = state.z, state.z_prev
    else
        state.x, state.z = state.z, state.x
    end

    # TODO: if iter.fast == true, in the adaptive case we should still be able
    # to save some computation by extrapolating Ax and (if f is quadratic)
    # f_Ax, grad_f_Ax, At_grad_f_Ax.
    if iter.fast == false && (iter.gamma === nothing || iter.adaptive == true)
        state.x = z
        state.f_x = f_z
        state.grad_f_x = grad_f_z
    else
        # mul!(state.Ax, iter.A, state.x)
        state.f_x, state.grad_f_x=iter.f(state.x)
    end

    # mul!(state.At_grad_f_Ax, adjoint(iter.A), state.grad_f_Ax)
    state.y .= state.x .- state.gamma .* state.grad_f_x
    state.z = iter.prox(state.y, state.gamma)
    state.g_z = iter.g(state.z)

    state.res .= state.x .- state.z

    return state, state
end

# Solver

struct ForwardBackward{R<:Real}
    gamma::R
    adaptive::Bool
    fast::Bool
    maxit::Int
    tol::R
    verbose::Bool
    freq::Int

    function ForwardBackward{R}(;
        gamma::R = R(1.0),
        adaptive::Bool = false,
        fast::Bool = false,
        maxit::Int = 10000,
        tol::R = R(1e-8),
        verbose::Bool = false,
        freq::Int = 1,
    ) where {R}
        @assert gamma === nothing || gamma > 0
        @assert maxit > 0
        @assert tol > 0
        @assert freq > 0
        new(gamma, adaptive, fast, maxit, tol, verbose, freq)
    end
end

function (solver::ForwardBackward{R})(
    x0::AbstractArray{C};
    f = Zero(),
    prox = Zero(),
    # A = I,
    g = Zero(),
    L::R = nothing,
) where {R,C<:Union{R,Complex{R}}}

    stop(state::FBS_state) = norm(state.res, Inf) / state.gamma <= solver.tol
    disp((it, state)) =
        @printf("%5d | %.3e | %.3e\n", it, state.gamma, norm(state.res, Inf) / state.gamma)

    gamma = if solver.gamma === nothing && L !== nothing
        R(1) / L
    else
        solver.gamma
    end

    iter = FBS_iterable(f, prox, g, x0, gamma, solver.adaptive, solver.fast)
    iter = take(IterationTools.halt(iter, stop), solver.maxit)
    iter = enumerate(iter)
    if solver.verbose
        iter = IterationTools.tee(IterationTools.sample(iter, solver.freq), disp)
    end

    num_iters, state_final = IterationTools.loop(iter)

    return state_final.z, num_iters

end

# Outer constructors

"""
    ForwardBackward([gamma, adaptive, fast, maxit, tol, verbose, freq])
Instantiate the Forward-Backward splitting algorithm (see [1, 2]) for solving
optimization problems of the form
    minimize f(Ax) + g(x),
where `f` is smooth and `A` is a linear mapping (for example, a matrix).
If `solver = ForwardBackward(args...)`, then the above problem is solved with
    solver(x0, [f, A, g, L])
Optional keyword arguments:
* `gamma::Real` (default: `nothing`), the stepsize to use; defaults to `1/L` if not set (but `L` is).
* `adaptive::Bool` (default: `false`), if true, forces the method stepsize to be adaptively adjusted.
* `fast::Bool` (default: `false`), if true, uses Nesterov acceleration.
* `maxit::Integer` (default: `10000`), maximum number of iterations to perform.
* `tol::Real` (default: `1e-8`), absolute tolerance on the fixed-point residual.
* `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
* `freq::Integer` (default: `10`), frequency of verbosity.
If `gamma` is not specified at construction time, the following keyword
argument can be used to set the stepsize parameter:
* `L::Real` (default: `nothing`), the Lipschitz constant of the gradient of x ↦ f(Ax).
References:
[1] Tseng, "On Accelerated Proximal Gradient Methods for Convex-Concave
Optimization" (2008).
[2] Beck, Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm
for Linear Inverse Problems", SIAM Journal on Imaging Sciences, vol. 2, no. 1,
pp. 183-202 (2009).
"""
ForwardBackward(::Type{R}; kwargs...) where {R} = ForwardBackward{R}(; kwargs...)
ForwardBackward(; kwargs...) = ForwardBackward(Float64; kwargs...)