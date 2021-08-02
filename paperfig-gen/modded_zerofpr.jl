import ProximalAlgorithms: LBFGS, Maybe, ZeroFPR, ZeroFPR_iterable, ZeroFPR_state
import ProximalAlgorithms.IterationTools: halt, sample, tee, loop
using Base.Iterators  # for take

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

    return state_final.x, num_iters, state_final.xbar
end