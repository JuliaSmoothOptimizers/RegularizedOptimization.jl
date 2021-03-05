import ProximalAlgorithms: LBFGS, Maybe, PANOC, PANOC_iterable, PANOC_state
import ProximalAlgorithms.IterationTools: halt, sample, tee, loop
using Base.Iterators  # for take
mutable struct LeastSquaresObjective
	nonlin
	b
end
	  
# definition that will allow to simply evaluate the objective: ϕ(x)
function (f::LeastSquaresObjective)(x)
	r = f.nonlin(x)[1]
	return r
end

# first import gradient and gradient! to extend them
import ProximalOperators.gradient, ProximalOperators.gradient!

function gradient!(∇fx, f::LeastSquaresObjective, x)
	r, g = f.nonlin(x)
	∇fx .= g
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



function my_panoc(solver::PANOC{R},
		x0::AbstractArray{C};
		f = Zero(),
		A = I,
		g = Zero(),
		L::Maybe{R} = nothing,
		Fhist = zeros(0),
		Hhist = zeros(0),) where {R,C<:Union{R,Complex{R}}}
	stop(state::PANOC_state) = norm(state.res, Inf) / state.gamma <= solver.tol
	function disp((it, state))
		append!(Fhist, state.f_Ax)
		append!(Hhist, state.g_z)
		@printf(
				"%5d | %.3e | %.3e | %.3e | %9.2e | %9.2e\n",
				it,
				state.gamma,
				norm(state.res, Inf) / state.gamma,
				(state.tau === nothing ? 0.0 : state.tau),
				state.f_Ax,  # <-- added this
				state.g_z    # <-- and this
			)
	end
	# disp((it, state)) = @printf(
	# 	"%5d | %.3e | %.3e | %.3e | %9.2e | %9.2e\n",
	# 	it,
	# 	state.gamma,
	# 	norm(state.res, Inf) / state.gamma,
	# 	(state.tau === nothing ? 0.0 : state.tau),
	# 	state.f_Ax,  # <-- added this
	# 	state.g_z    # <-- and this
	# )
	
	gamma = if solver.gamma === nothing && L !== nothing
		solver.alpha / L
	else
		solver.gamma
	end

	iter = PANOC_iterable(
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
	if isinf(sum(state_final.z))
		x = state_final.x
	else
		x = state_final.z
	end

	return x, num_iters, Fhist, Hhist
end