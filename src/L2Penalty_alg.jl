export L2Penalty, L2PenaltySolver, solve!

import SolverCore.solve!

mutable struct L2PenaltySolver{T <: Real, V <: AbstractVector{T}, S <: AbstractOptimizationSolver} <: AbstractOptimizationSolver
  x::V
	s::V
	s0::V
  ψ::ShiftedCompositeNormL2
	sub_ψ::CompositeNormL2
	sub_solver::S
	sub_stats::GenericExecutionStats{T, V, V, Any}
end

function L2PenaltySolver(
  nlp::AbstractNLPModel{T, V};
	sub_solver = R2Solver
	) where{T, V}
	x0 = nlp.meta.x0
	x = similar(x0)
	s = similar(x0)
	s0 = zero(x0)

	# Allocating variables for the ShiftedProximalOperator structure
	(rows, cols) = jac_structure(nlp)
  vals = similar(rows,eltype(x0))
	A = SparseMatrixCOO(nlp.meta.ncon, nlp.meta.nvar, rows, cols, vals)
	b = similar(x0, eltype(x0), nlp.meta.ncon)

	
	# Allocate ψ = ||c(x) + J(x)s|| to compute θ
	ψ = ShiftedCompositeNormL2(1.0, 
		(c, x) -> cons!(nlp, x, c), 
		(j, x) -> jac_coord!(nlp, x, j.vals), 
		A, 
		b
	)

	# Allocate sub_ψ = ||c(x)|| to solve min f(x) + τ||c(x)||
	sub_ψ = CompositeNormL2(1.0, 
		(c, x) -> cons!(nlp, x, c), 
		(j, x) -> jac_coord!(nlp, x, j.vals), 
		A, 
		b
	)
	sub_nlp = RegularizedNLPModel(nlp, sub_ψ)
	sub_stats = GenericExecutionStats(nlp)
	if sub_solver == R2NSolver
		Solver = sub_solver(sub_nlp,sub_solver = L2_R2N_subsolver)
	else
		Solver = sub_solver(sub_nlp)
	end

  return L2PenaltySolver(
		x,
		s,
		s0,
		ψ,
		sub_ψ,
		Solver,
		sub_stats
	)
end


"""
    L2Penalty(nlp; kwargs…)

An exact ℓ₂-penalty method for the problem

    min f(x) 	s.t c(x) = 0

where f: ℝⁿ → ℝ and c: ℝⁿ → ℝᵐ respectively have a Lipschitz-continuous gradient and Jacobian.

At each iteration k, an iterate is computed as 

    xₖ ∈ argmin f(x) + τₖ‖c(x)‖₂

where τₖ is some penalty parameter.
This nonsmooth problem is solved using `R2` (see `R2` for more information) with the first order model ψ(s;x) = τₖ‖c(x) + J(x)s‖₂

For advanced usage, first define a solver "L2PenaltySolver" to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = L2PenaltySolver(nlp)
    solve!(solver, nlp)

    stats = GenericExecutionStats(nlp)
    solver = L2PenaltySolver(nlp)
    solve!(solver, nlp, stats)

# Arguments
* `nlp::AbstractNLPModel{T, V}`: the problem to solve, see `RegularizedProblems.jl`, `NLPModels.jl`.

# Keyword arguments 
- `x::V = nlp.meta.x0`: the initial guess;
- `atol::T = √eps(T)`: absolute tolerance;
- `rtol::T = √eps(T)`: relative tolerance;
- `neg_tol::T = eps(T)^(1 / 4)`: negative tolerance
- `ktol::T = eps(T)^(1 / 4)`: the initial tolerance sent to the subsolver
- `max_eval::Int = -1`: maximum number of evaluation of the objective function (negative number means unlimited);
- `sub_max_eval::Int = -1`: maximum number of evaluation for the subsolver (negative number means unlimited);
- `max_time::Float64 = 30.0`: maximum time limit in seconds;
- `max_iter::Int = 10000`: maximum number of iterations;
- `sub_max_iter::Int = 10000`: maximum number of iterations for the subsolver;
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration;
- `sub_verbose::Int = 0`: if > 0, display subsolver iteration details every `verbose` iteration;
- `τ::T = T(100)`: initial penalty parameter;
- `β1::T = τ`: penalty update parameter: τₖ <- τₖ + β1;	
- `β2::T = T(0.1)`: tolerance decreasing factor, at each iteration, ktol <- β2*ktol;
- `β3::T = 1/τ`: initial regularization parameter σ₀ = β3/τₖ at each iteration;
- `β4::T = eps(T)`: minimal regularization parameter σ for `R2`;
other 'kwargs' are passed to `R2` (see `R2` for more information).

The algorithm stops either when `√θₖ < atol + rtol*√θ₀ ` or `θₖ < 0` and `√(-θₖ) < neg_tol` where θₖ := ‖c(xₖ)‖₂ - ‖c(xₖ) + J(xₖ)sₖ‖₂, and √θₖ is a stationarity measure.

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
The callback is called at each iteration.
The expected signature of the callback is `callback(nlp, solver, stats)`, and its output is ignored.
Changing any of the input arguments will affect the subsequent iterations.
In particular, setting `stats.status = :user` will stop the algorithm.
All relevant information should be available in `nlp` and `solver`.
Notably, you can access, and modify, the following:
- `solver.x`: current iterate;
- `solver.sub_solver`: a `R2Solver` structure holding relevant information on the subsolver state, see `R2` for more information;
- `stats`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
  - `stats.iter`: current iteration counter;
  - `stats.objective`: current objective function value;
  - `stats.status`: current status of the algorithm. Should be `:unknown` unless the algorithm has attained a stopping criterion. Changing this to anything will stop the algorithm, but you should use `:user` to properly indicate the intention.
  - `stats.elapsed_time`: elapsed time in seconds.
You can also use the `sub_callback` keyword argument which has exactly the same structure and in sent to `R2`.
"""
function L2Penalty(
  nlp::AbstractNLPModel{T, V};
	sub_solver = R2Solver,
  kwargs...) where{ T <: Real, V }
	if !equality_constrained(nlp) 
		error("L2Penalty: This algorithm only works for equality contrained problems.")
	end
	solver = L2PenaltySolver(nlp,sub_solver = sub_solver)
	stats = GenericExecutionStats(nlp)
	solve!(
		solver,
		nlp,
		stats;
		kwargs...
	)
	return stats
end

function SolverCore.solve!(
  solver::L2PenaltySolver{T, V},
	nlp::AbstractNLPModel{T, V},
	stats::GenericExecutionStats{T, V};
	callback = (args...) -> nothing,
	sub_callback = (args...) -> nothing,
	x::V = nlp.meta.x0,
	atol::T = √eps(T),
	rtol::T = √eps(T),
	neg_tol = eps(T)^(1/4),
	ktol::T = eps(T)^(1/4),
	max_iter::Int = 10000,
	sub_max_iter::Int = 10000,
	max_time::T = T(30.0),
	max_eval::Int = -1,
	sub_max_eval::Int = -1,
	verbose = 0,
	sub_verbose = 0,
	τ::T = T(100),
	β1::T = τ,
	β2::T = T(0.1),
	β3::T = 1/τ,
	β4::T = eps(T),
	kwargs...,
  ) where {T, V}
    
	reset!(stats)

	# Retrieve workspace
	h = NormL2(1.0)
	ψ = solver.ψ
	sub_ψ = solver.sub_ψ
	sub_ψ.h = NormL2(τ)
	solver.sub_solver.ψ.h = NormL2(τ)
	
	x = solver.x .= x
	s = solver.s
	s0 = solver.s0
	shift!(ψ, x)
	fx = obj(nlp, x)
	hx = h(ψ.b)

	if verbose > 0
    @info log_header(
      [:iter, :sub_iter, :fx, :hx, :theta, :xi, :epsk, :tau, :normx],
      [Int, Int, Float64, Float64, Float64, Float64, Float64, Float64, Float64],
      hdr_override = Dict{Symbol,String}(   # TODO: Add this as constant dict elsewhere
        :iter => "outer",
				:sub_iter => "inner",
        :fx => "f(x)",
        :hx => "h(x)",
				:theta => "√θ",
        :xi => "√(ξ/ν)",
				:epsk => "ϵₖ",
        :tau => "τ",
        :normx => "‖x‖"
      ),
      colsep = 1,
    )
  end

	set_iter!(stats, 0)
	rem_eval = max_eval
	start_time = time()
	set_time!(stats, 0.0)
	set_objective!(stats,fx + hx)
	set_solver_specific!(stats,:smooth_obj,fx)
	set_solver_specific!(stats,:nonsmooth_obj, hx)

	local θ::T 
	prox!(s, ψ, s0, 1.0)
	θ = hx - ψ(s)

	sqrt_θ = θ ≥ 0 ? sqrt(θ) : sqrt(-θ)
	θ < 0 && sqrt_θ ≥ neg_tol && error("L2Penalty: prox-gradient step should produce a decrease but θ = $(θ)")
	
  atol += rtol * sqrt_θ # make stopping test absolute and relative
	ktol = max(ktol,atol) # Keep ϵ₀ ≥ ϵ
	tol_init = ktol # store value of ϵ₀ 

	done = false

	while !done
		model = RegularizedNLPModel(nlp, sub_ψ)
		solve!(
			solver.sub_solver,
			model,
			solver.sub_stats;
			callback = sub_callback,
			x = x,
			atol = ktol,
			rtol = T(0),
			neg_tol = neg_tol,
			verbose = sub_verbose,
			max_iter = sub_max_iter,
			max_time = max_time - stats.elapsed_time,
			max_eval = min(rem_eval,sub_max_eval),
			σmin = β4,
			ν = 1/max(β4,β3*τ),
			kwargs...,
		)

		x .= solver.sub_stats.solution
		fx = solver.sub_stats.solver_specific[:smooth_obj]
		hx = solver.sub_stats.solver_specific[:nonsmooth_obj]/τ
		sqrt_ξ_νInv  =  solver.sub_stats.solver_specific[:xi]

		shift!(ψ, x)
		prox!(s, ψ, s0, 1.0)

		θ = hx - ψ(s)
		sqrt_θ = θ ≥ 0 ? sqrt(θ) : sqrt(-θ)
		θ < 0 && sqrt_θ ≥ neg_tol && error("L2Penalty: prox-gradient step should produce a decrease but θ = $(θ)")

		if sqrt_θ > ktol
			τ = τ + β1
			sub_ψ.h = NormL2(τ)
			solver.sub_solver.ψ.h = NormL2(τ)
		else 
			ktol = max(β2^(ceil(log(β2,sqrt_ξ_νInv/tol_init)))*ktol,atol) #the β^... allows to directly jump to a sufficiently small ϵₖ
		end

		solved = (sqrt_θ ≤ atol && solver.sub_stats.status == :first_order) || (θ < 0 && sqrt_θ ≤ neg_tol && solver.sub_stats.status == :first_order)
		(θ < 0 && sqrt_θ > neg_tol) && error("L2Penalty: prox-gradient step should produce a decrease but θ = $(θ)")

		verbose > 0 &&
			stats.iter % verbose == 0 &&
				@info log_row(Any[stats.iter, solver.sub_stats.iter, fx, hx ,sqrt_θ, sqrt_ξ_νInv, ktol, τ, norm(x)], colsep = 1)

		set_iter!(stats, stats.iter + 1)
		rem_eval = max_eval - neval_obj(nlp)
		set_time!(stats, time() - start_time)
		set_objective!(stats,fx + hx)
		set_solver_specific!(stats,:smooth_obj,fx)
		set_solver_specific!(stats,:nonsmooth_obj, hx)
		set_solver_specific!(stats, :theta, sqrt_θ)

		set_status!(
      stats,
      get_status(
        nlp,
        elapsed_time = stats.elapsed_time,
        iter = stats.iter,
        optimal = solved,
        max_eval = max_eval,
        max_time = max_time,
        max_iter = max_iter
      ),
    )

		callback(nlp, solver, stats)

		done = stats.status != :unknown
	end

	set_solution!(stats, x)
	return stats

end

function get_status(
  nlp::M;
  elapsed_time = 0.0,
  iter = 0,
  optimal = false,
  max_eval = Inf,
  max_time = Inf,
  max_iter = Inf,
) where{ M <: AbstractNLPModel }
  if optimal
    :first_order
  elseif iter > max_iter
    :max_iter
  elseif elapsed_time > max_time
    :max_time
  elseif neval_obj(nlp) > max_eval && max_eval > -1
    :max_eval
  else
    :unknown
  end
end

mutable struct L2_R2N_subsolver{T <: Real, V <: AbstractVector{T}} <: AbstractOptimizationSolver
  u1::V
	u2::V
end

function L2_R2N_subsolver(
  reg_nlp::AbstractRegularizedNLPModel{T, V};
	) where{T, V}
	x0 = reg_nlp.model.meta.x0
	n = reg_nlp.model.meta.nvar
	m = length(reg_nlp.h.b)
	#x = zero(x0)
	u1 = similar(x0, n+m)
	u2 = zeros(eltype(x0), n+m)


  return L2_R2N_subsolver(
		u1,
		u2,
	)
end

function solve!(
	solver::L2_R2N_subsolver{T, V},
	reg_nlp::AbstractRegularizedNLPModel{T, V},
	stats::GenericExecutionStats{T, V, V, Any},
	∇fk::V,
	Q::L,
	σk::T;
	x = reg_nlp.model.meta.x0,
	atol = eps(T)^(0.5),
	max_time = T(30),
	max_iter = 10000
	) where{T <: Real, V <: AbstractVector{T} , L <: AbstractLinearOperator}

	start_time = time()
	set_time!(stats, 0.0)
	set_iter!(stats, 0)
	
	n = reg_nlp.model.meta.nvar
	m = length(reg_nlp.h.b)
	Δ = reg_nlp.h.h.lambda

	u1 = solver.u1
	u2 = solver.u2

	# Create problem
	@. u1[1:n] = ∇fk
	@. u1[n+1:n+m] = -reg_nlp.h.b

	αₖ = 0.0

	H1 = [-Q reg_nlp.h.A']
	H2 = [reg_nlp.h.A αₖ*opEye(m,m)]
	H = [H1;H2]
	x1,_ = minres_qlp(H,u1)

	if norm(x1[n+1:n+m]) <= Δ
		set_solution!(stats,x1[1:n])
		return
	end
	u2[n+1:n+m] .= x1[n+1:n+m]
	x2,_ = minres_qlp(H,u2)
	αₖ += norm(x1[n+1:n+m])^2/(x1[n+1:n+m]'x2[n+1:n+m])*(norm(x1[n+1:n+m])- Δ)/Δ
	k = 0

	while abs(norm(x1[n+1:n+m]) - Δ) > eps(T)^(0.75) && stats.iter < max_iter && stats.elapsed_time < max_time
		H2 = [reg_nlp.h.A αₖ*opEye(m,m)]
		H = [H1;H2]

		x1,_ = minres_qlp(H,u1)
		u2[n+1:n+m] .= x1[n+1:n+m]
		x2,_ = minres_qlp(H,u2)
		αₖ += norm(x1[n+1:n+m])^2/(x1[n+1:n+m]'x2[n+1:n+m])*(norm(x1[n+1:n+m])- Δ)/Δ
		set_iter!(stats,stats.iter + 1)
		set_time!(stats,time()-start_time)
	end
	set_solution!(stats,x1[1:n])

end
