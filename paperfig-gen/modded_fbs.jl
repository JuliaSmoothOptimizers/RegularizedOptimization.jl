import ProximalAlgorithms: Maybe, ForwardBackward, FBS_iterable, FBS_state
import ProximalAlgorithms.IterationTools: halt, sample, tee, loop
using Base.Iterators  # for take

function my_fbs(solver::ForwardBackward{R},
    x0::AbstractArray{C};
    f = Zero(),
    A = I,
    g = Zero(),
    L::Maybe{R} = nothing,
    ) where {R,C<:Union{R,Complex{R}}}
    stop(state::FBS_state) = norm(state.res, Inf) / state.gamma <= solver.tol
    function disp((it, state))
        @printf(
                "%5d | %.3e | %.3e | %9.2e | %9.2e\n",
                it,
                state.gamma,
                norm(state.res, Inf) / state.gamma,
                state.f_Ax,  # <-- added this
                state.g_z    # <-- and this
            )
    end

    gamma = if solver.gamma === nothing && L !== nothing
        R(1) / L
    else
        solver.gamma
    end

    iter = FBS_iterable(
        f,
        A,
        g,
        x0,
        gamma,
        solver.adaptive, 
        solver.fast
    )
    iter = take(halt(iter, stop), solver.maxit)
    iter = enumerate(iter)
    if solver.verbose
        iter = tee(sample(iter, solver.freq), disp)
    end

    num_iters, state_final = loop(iter)

    return state_final.z, num_iters
end