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

  return L2PenaltySolver(
		x,
		s,
		s0,
		ψ,
		sub_ψ,
		sub_solver(sub_nlp),
		sub_stats
	)
end


"""
    #TODO
"""
function L2Penalty(
  nlp::AbstractNLPModel{T, V};
  kwargs...) where{ T <: Real, V }

	solver = L2PenaltySolver(nlp)
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
	θ ≥ 0 || error("L2Penalty: prox-gradient step should produce a decrease but θ = $(θ)")
	sqrt_θ = sqrt(θ)
	
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
			neg_tol = T(0),
			verbose = sub_verbose,
			max_iter = sub_max_iter,
			max_time = max_time - stats.elapsed_time,
			max_eval = min(rem_eval,sub_max_eval),
			σmin = β4,
			#ν = 1/max(β4,β3*τ),
			kwargs...,
		)

		x .= solver.sub_stats.solution
		fx = solver.sub_stats.solver_specific[:smooth_obj]
		hx = solver.sub_stats.solver_specific[:nonsmooth_obj]
		sqrt_ξ_νInv  =  solver.sub_stats.solver_specific[:xi]

		shift!(ψ, x)
		prox!(s, ψ, s0, 1.0)

		θ = hx - ψ(s)
		θ ≥ 0 || error("L2Penalty: prox-gradient step should produce a decrease but θ = $(θ)")
		sqrt_θ = sqrt(θ)

		if sqrt_θ > ktol
			τ = τ + β1
			sub_ψ.h = NormL2(τ)
			solver.sub_solver.ψ.h = NormL2(τ)
		else 
			ktol = max(β2^(ceil(log(β2,sqrt_ξ_νInv/tol_init)))*ktol,atol) #the β^... allows to directly jump to a sufficiently small ϵₖ
		end

		solved = sqrt_θ ≤ atol && sqrt_ξ_νInv ≤ atol
		
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
