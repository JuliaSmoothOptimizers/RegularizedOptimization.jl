export AL, ALSolver, solve!

import SolverCore.solve!

function AL(nlp::AbstractNLPModel, h; kwargs...)
  kwargs_dict = Dict(kwargs...)
  selected = pop!(kwargs_dict, :selected, 1:(nlp.meta.nvar))
  reg_nlp = RegularizedNLPModel(nlp, h, selected)
  return AL(reg_nlp; kwargs...)
end

function AL(reg_nlp::AbstractRegularizedNLPModel; kwargs...)
  if unconstrained(reg_nlp.model) || bound_constrained(reg_nlp.model)
    return AL(Val(:unc), reg_nlp; kwargs...)
  elseif equality_constrained(reg_nlp.model)
    return AL(Val(:equ), reg_nlp; kwargs...)
  else # has inequalities
    return AL(Val(:ineq), reg_nlp; kwargs...)
  end
end

function AL(
  ::Val{:unc},
  reg_nlp::AbstractRegularizedNLPModel;
  subsolver = has_bounds(reg_nlp.model) ? TR : R2,
  kwargs...,
)
  if !(unconstrained(reg_nlp.model) || bound_constrained(reg_nlp.model))
    error(
      "AL(::Val{:unc}, ...) should only be called for unconstrained or bound-constrained problems. Use AL(...)",
    )
  end
  @warn "Problem does not have general explicit constraints; calling solver $(string(subsolver))"
  return subsolver(reg_nlp; kwargs...)
end

function AL(
  ::Val{:ineq},
  reg_nlp::AbstractRegularizedNLPModel;
  x::V = reg_nlp.model.meta.x0,
  kwargs...,
) where {V}
  nlp = reg_nlp.model
  if nlp.meta.ncon == 0 || equality_constrained(nlp)
    error(
      "AL(::Val{:ineq}, ...) should only be called for problems with inequalities. Use AL(...)",
    )
  end
  snlp = nlp isa AbstractNLSModel ? SlackNLSModel(nlp) : SlackModel(nlp)
  reg_snlp = RegularizedNLPModel(snlp, reg_nlp.h, reg_nlp.selected)
  if length(x) != snlp.meta.nvar
    xs = fill!(V(undef, snlp.meta.nvar), zero(eltype(V)))
    xs[1:(nlp.meta.nvar)] .= x
  else
    xs = x
  end
  output = AL(Val(:equ), reg_snlp; x = xs, kwargs...)
  output.solution = output.solution[1:(nlp.meta.nvar)]
  return output
end

"""
    AL(reg_nlp; kwargs...)

An augmented Lagrangian method for constrained regularized optimization, namely problems in the form

    minimize    f(x) + h(x)
    subject to  lvar ≤ x ≤ uvar,
                lcon ≤ c(x) ≤ ucon

where f: ℝⁿ → ℝ, c: ℝⁿ → ℝᵐ and their derivatives are Lipschitz continuous and h: ℝⁿ → ℝ is
lower semi-continuous, proper and prox-bounded.

At each iteration, an iterate x is computed as an approximate solution of the subproblem

    minimize    L(x;y,μ) + h(x)
    subject to  lvar ≤ x ≤ uvar

where y is an estimate of the Lagrange multiplier vector for the constraints lcon ≤ c(x) ≤ ucon,
μ is the penalty parameter and L(⋅;y,μ) is the augmented Lagrangian function defined by

    L(x;y,μ) := f(x) - yᵀc(x) + ½ μ ‖c(x)‖².

For advanced usage, first define a solver "ALSolver" to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = ALSolver(reg_nlp)
    solve!(solver, reg_nlp)

    stats = GenericExecutionStats(reg_nlp.model)
    solver = ALSolver(reg_nlp)
    solve!(solver, reg_nlp, stats)

# Arguments

- `reg_nlp::AbstractRegularizedNLPModel`: a regularized optimization problem, see `RegularizedProblems.jl`,
  consisting of `model` representing a smooth optimization problem, see `NLPModels.jl`, and a regularizer `h` such
  as those defined in `ProximalOperators.jl`.

The objective and gradient of `model` will be accessed.
The Hessian of `model` may be accessed or not, depending on the subsolver adopted.
If adopted, the Hessian is accessed as an abstract operator and need not be the exact Hessian.

# Keyword arguments

- `x::AbstractVector`: a primal initial guess (default: `reg_nlp.model.meta.x0`)
- `y::AbstractVector`: a dual initial guess (default: `reg_nlp.model.meta.y0`)
- `atol::T = √eps(T)`: absolute optimality tolerance;
- `ctol::T = atol`: absolute feasibility tolerance;
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration;
- `max_iter::Int = 10000`: maximum number of iterations;
- `max_time::Float64 = 30.0`: maximum time limit in seconds;
- `max_eval::Int = -1`: maximum number of evaluation of the objective function (negative number means unlimited);
- `subsolver::AbstractOptimizationSolver = has_bounds(nlp) ? TR : R2`: the procedure used to compute a step (e.g. `PG`, `R2`, `TR` or `TRDH`);
- `subsolver_logger::AbstractLogger`: a logger to pass to the subproblem solver;
- `init_penalty::T = T(10)`: initial penalty parameter;
- `factor_penalty_up::T = T(2)`: multiplicative factor to increase the penalty parameter;
- `factor_primal_linear_improvement::T = T(3/4)`: fraction to declare sufficient improvement of feasibility;
- `init_subtol::T = T(0.1)`: initial subproblem tolerance;
- `factor_decrease_subtol::T = T(1/4)`: multiplicative factor to decrease the subproblem tolerance;
- `dual_safeguard = (nlp::AugLagModel) -> nothing`: in-place function to modify, as needed, the dual estimate.

# Output

- `stats::GenericExecutionStats`: solution and other info, see `SolverCore.jl`.

# Callback
The callback is called at each iteration.
The expected signature of the callback is `callback(nlp, solver, stats)`, and its output is ignored.
Changing any of the input arguments will affect the subsequent iterations.
In particular, setting `stats.status = :user` will stop the algorithm.
All relevant information should be available in `reg_nlp` and `solver`.
Notably, you can access, and modify, the following:
- `solver.x`: current iterate;
- `solver.y`: current dual estimate;
- `stats`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
  - `stats.iter`: current iteration counter;
  - `stats.objective`: current objective function value;
  - `stats.status`: current status of the algorithm. Should be `:unknown` unless the algorithm has attained a stopping criterion. Changing this to anything will stop the algorithm, but you should use `:user` to properly indicate the intention;
  - `stats.elapsed_time`: elapsed time in seconds;
  - `stats.solver_specific[:smooth_obj]`: current value of the smooth part of the objective function;
  - `stats.solver_specific[:nonsmooth_obj]`: current value of the nonsmooth part of the objective function.
"""
mutable struct ALSolver{T, V, M, Pb, ST} <: AbstractOptimizationSolver
  x::V
  cx::V
  y::V
  has_bnds::Bool
  sub_problem::Pb
  sub_solver::ST
  sub_stats::GenericExecutionStats{T, V, V, T}
end

function ALSolver(reg_nlp::AbstractRegularizedNLPModel{T, V}; kwargs...) where {T, V}
  nlp = reg_nlp.model
  nvar, ncon = nlp.meta.nvar, nlp.meta.ncon
  x = V(undef, nvar)
  cx = V(undef, ncon)
  y = V(undef, ncon)
  has_bnds = has_bounds(nlp)
  sub_model = AugLagModel(nlp, V(undef, ncon), T(0), x, T(0), cx)
  sub_problem = RegularizedNLPModel(sub_model, reg_nlp.h, reg_nlp.selected)
  sub_solver = R2Solver(reg_nlp; kwargs...)
  sub_stats = RegularizedExecutionStats(sub_problem)
  M = typeof(nlp)
  ST = typeof(sub_solver)
  return ALSolver{T, V, M, typeof(sub_problem), ST}(
    x,
    cx,
    y,
    has_bnds,
    sub_problem,
    sub_solver,
    sub_stats,
  )
end

@doc (@doc ALSolver) function AL(
  ::Val{:equ},
  reg_nlp::AbstractRegularizedNLPModel;
  kwargs...,
)
  nlp = reg_nlp.model
  if !(nlp.meta.minimize)
    error("AL only works for minimization problems")
  end
  if nlp.meta.ncon == 0 || !equality_constrained(nlp)
    error(
      "AL(::Val{:equ}, ...) should only be called for equality-constrained problems with bounded variables. Use AL(...)",
    )
  end
  solver = ALSolver(reg_nlp)
  solve!(solver, reg_nlp; kwargs...)
end

function SolverCore.solve!(
  solver::AbstractOptimizationSolver,
  model::AbstractRegularizedNLPModel;
  kwargs...,
)
  stats = RegularizedExecutionStats(model)
  solve!(solver, model, stats; kwargs...)
end

function SolverCore.solve!(
  solver::ALSolver{T, V},
  reg_nlp::AbstractRegularizedNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = reg_nlp.model.meta.x0,
  y::V = reg_nlp.model.meta.y0,
  atol::T = √eps(T),
  verbose::Int = 0,
  max_iter::Int = 10000,
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  subsolver_verbose::Int = 0,
  subsolver_max_iter::Int = 100000,
  subsolver_max_eval::Int = -1,
  init_penalty::T = T(10),
  factor_penalty_up::T = T(2),
  ctol::T = atol,
  init_subtol::T = T(0.1),
  factor_primal_linear_improvement::T = T(3 // 4),
  factor_decrease_subtol::T = T(1 // 4),
  dual_safeguard = project_y!,
) where {T, V}
  reset!(stats)

  # Retrieve workspace
  nlp = reg_nlp.model
  h = reg_nlp.h
  selected = reg_nlp.selected

  # Sanity checks
  if !(nlp.meta.minimize)
    error("AL only works for minimization problems")
  end
  if nlp.meta.ncon == 0 || !equality_constrained(nlp)
    error(
      "AL(::Val{:equ}, ...) should only be called for equality-constrained problems. Use AL(...)",
    )
  end
  @assert length(solver.x) == nlp.meta.nvar
  @assert length(solver.y) == nlp.meta.ncon
  #TODO check solver.has_bnds with has_bounds(nlp) for solver.sub_solver

  reset!(stats)
  set_iter!(stats, 0)
  start_time = time()
  set_time!(stats, 0.0)

  # check parameter values
  @assert init_penalty > 0
  @assert factor_penalty_up > 1
  @assert 0 < factor_primal_linear_improvement < 1
  @assert 0 < factor_decrease_subtol < 1

  # initialization
  solver.x .= max.(nlp.meta.lvar, min.(x, nlp.meta.uvar))
  solver.y .= y
  set_solution!(stats, solver.x)
  set_constraint_multipliers!(stats, solver.y)
  subout = solver.sub_stats

  fx, _ = objcons!(nlp, solver.x, solver.cx)
  hx = @views h(solver.x[selected])
  objx = fx + hx
  set_objective!(stats, objx)
  set_solver_specific!(stats, :smooth_obj, fx)
  set_solver_specific!(stats, :nonsmooth_obj, hx)

  mu = init_penalty
  solver.sub_problem.model.y .= solver.y
  update_μ!(solver.sub_problem.model, mu)

  cviol = norm(solver.cx, Inf)
  cviol_old = Inf
  iter = 0
  subiters = 0
  done = false
  subtol = init_subtol
  rem_eval = max_eval

  if verbose > 0
    @info log_header(
      [:iter, :sub_it, :obj, :cviol, :μ, :normy, :sub_tol, :sub_status],
      [Int, Int, Float64, Float64, Float64, Float64, Float64, Symbol],
    )
    @info log_row(Any[iter, subiters, objx, cviol, mu, norm(solver.y), subtol])
  end

  callback(reg_nlp, solver, stats)

  while !done
    iter += 1

    # dual safeguard
    dual_safeguard(solver.sub_problem.model)

    subtol = max(subtol, atol)
    reset!(subout)
    solve!(
      solver.sub_solver,
      solver.sub_problem,
      subout,
      x = solver.x,
      atol = subtol,
      rtol = zero(T),
      max_time = max_time - stats.elapsed_time,
      max_eval = subsolver_max_eval < 0 ? rem_eval : min(subsolver_max_eval, rem_eval),
      max_iter = subsolver_max_iter,
      verbose = subsolver_verbose,
    )
    solver.x .= subout.solution
    solver.cx .= solver.sub_problem.model.cx
    subiters = subout.iter

    # objective
    fx = obj(nlp, solver.x)
    hx = @views h(solver.x[selected])
    objx = fx + hx
    set_objective!(stats, objx)
    set_solver_specific!(stats, :smooth_obj, fx)
    set_solver_specific!(stats, :nonsmooth_obj, hx)

    # dual estimate
    update_y!(solver.sub_problem.model)
    solver.y .= solver.sub_problem.model.y
    set_constraint_multipliers!(stats, solver.y)

    # stationarity measure
    # FIXME it seems that R2 returns no dual_feas in `dual_feas`
    # but in `solver_specific.xi`
    if subout.dual_residual_reliable
      set_dual_residual!(stats, subout.dual_feas)
    end

    # primal violation
    cviol = norm(solver.cx, Inf)
    set_primal_residual!(stats, cviol)

    # termination checks
    dual_ok = subout.status_reliable && subout.status == :first_order && subtol <= atol
    primal_ok = cviol <= ctol
    optimal = dual_ok && primal_ok

    set_iter!(stats, iter)
    set_time!(stats, time() - start_time)
    set_status!(
      stats,
      SolverCore.get_status(
        nlp,
        elapsed_time = stats.elapsed_time,
        iter = stats.iter,
        optimal = optimal,
        infeasible = false,
        parameter_too_large = false,
        unbounded = false,
        stalled = false,
        exception = false,
        max_eval = max_eval,
        max_time = max_time,
        max_iter = max_iter,
      ),
    )

    callback(reg_nlp, solver, stats)

    done = stats.status != :unknown

    if verbose > 0 && (mod(stats.iter, verbose) == 0 || done)
      @info log_row(
        Any[iter, subiters, objx, cviol, mu, norm(solver.y), subtol, subout.status],
      )
    end

    if !done
      if cviol > max(ctol, factor_primal_linear_improvement * cviol_old)
        mu *= factor_penalty_up
      end
      update_μ!(solver.sub_problem.model, mu)
      cviol_old = cviol
      subtol *= factor_decrease_subtol
      rem_eval = max_eval < 0 ? max_eval : max_eval - neval_obj(nlp)
    end
  end
  set_solution!(stats, solver.x)
  set_constraint_multipliers!(stats, solver.y)
  stats
end

"""
    project_y!(nlp)

Given an `AugLagModel`, project `nlp.y` into [ymin, ymax] and updates `nlp.μc_y` accordingly.
"""
project_y!(nlp::AugLagModel) = project_y!(nlp::AugLagModel, -1e20, 1e20)

function project_y!(nlp::AugLagModel, ymin::V, ymax::V) where {V}
  nlp.y .= max.(ymin, min.(nlp.y, ymax))
  nlp.μc_y .= nlp.μ .* nlp.cx .- nlp.y
end
