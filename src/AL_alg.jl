export AL

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

# a uniform solver interface is missing
# TR(nlp, h; kwargs...) = TR(nlp, h, NormLinf(1.0); kwargs...)

function AL(
  ::Val{:ineq},
  reg_nlp::AbstractRegularizedNLPModel;
  x0::V = reg_nlp.model.meta.x0,
  kwargs...,
) where {V}
  nlp = reg_nlp.model
  if nlp.meta.ncon == 0 || equality_constrained(nlp)
    error("AL(::Val{:ineq}, ...) should only be called for problems with inequalities. Use AL(...)")
  end
  snlp = nlp isa AbstractNLSModel ? SlackNLSModel(nlp) : SlackModel(nlp)
  reg_snlp = RegularizedNLPModel(snlp, reg_nlp.h, reg_nlp.selected)
  if length(x0) != snlp.meta.nvar
    x = fill!(V(undef, snlp.meta.nvar), zero(eltype(V)))
    x[1:(nlp.meta.nvar)] .= x0
  else
    x = x0
  end
  output = AL(Val(:equ), reg_snlp; x0 = x, kwargs...)
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

### Arguments

* `reg_nlp::AbstractRegularizedNLPModel`: a regularized optimization problem, see `RegularizedProblems.jl`, 
  consisting of `model` representing a smooth optimization problem, see `NLPModels.jl`, and a regularizer `h` such
  as those defined in `ProximalOperators.jl`.

The objective and gradient of `model` will be accessed.
The Hessian of `model` may be accessed or not, depending on the subsolver adopted.
If adopted, the Hessian is accessed as an abstract operator and need not be the exact Hessian.

### Keyword arguments

* `x0::AbstractVector`: a primal initial guess (default: `reg_nlp.model.meta.x0`)
* `y0::AbstractVector`: a dual initial guess (default: `reg_nlp.model.meta.y0`)
- `atol::T = √eps(T)`: absolute optimality tolerance;
- `ctol::T = atol`: absolute feasibility tolerance;
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration;
- `max_iter::Int = 10000`: maximum number of iterations;
- `max_time::Float64 = 30.0`: maximum time limit in seconds;
- `max_eval::Int = -1`: maximum number of evaluation of the objective function (negative number means unlimited);
* `subsolver::AbstractOptimizationSolver = has_bounds(nlp) ? TR : R2`: the procedure used to compute a step (e.g. `PG`, `R2`, `TR` or `TRDH`);
* `subsolver_logger::AbstractLogger`: a logger to pass to the subproblem solver;
- `init_penalty::T = T(10)`: initial penalty parameter;
- `factor_penalty_up::T = T(2)`: multiplicative factor to increase the penalty parameter;
- `factor_primal_linear_improvement::T = T(3/4)`: fraction to declare sufficient improvement of feasibility;
- `init_subtol::T = T(0.1)`: initial subproblem tolerance;
- `factor_decrease_subtol::T = T(1/4)`: multiplicative factor to decrease the subproblem tolerance;
- `dual_safeguard = (nlp::AugLagModel) -> nothing`: in-place function to modify, as needed, the dual estimate.

### Return values

* `stats::GenericExecutionStats`: solution and other info, see `SolverCore.jl`.

"""
function AL(
  ::Val{:equ},
  reg_nlp::AbstractRegularizedNLPModel{T, V};
  x0::V = reg_nlp.model.meta.x0,
  y0::V = reg_nlp.model.meta.y0,
  atol::T = √eps(T),
  verbose::Int = 0,
  max_iter::Int = 10000,
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  subsolver = has_bounds(reg_nlp.model) ? TR : R2,
  subsolver_logger::Logging.AbstractLogger = Logging.NullLogger(),
  init_penalty::T = T(10),
  factor_penalty_up::T = T(2),
  ctol::T = atol,
  init_subtol::T = T(0.1),
  factor_primal_linear_improvement::T = T(3 // 4),
  factor_decrease_subtol::T = T(1 // 4),
  dual_safeguard = project_y!,
) where {T, V}

  # Retrieve workspace
  nlp = reg_nlp.model
  h = reg_nlp.h
  selected = reg_nlp.selected

  if !(nlp.meta.minimize)
    error("AL only works for minimization problems")
  end
  if nlp.meta.ncon == 0 || !equality_constrained(nlp)
    error(
      "AL(::Val{:equ}, ...) should only be called for equality-constrained problems. Use AL(...)",
    )
  end

  stats = GenericExecutionStats(nlp)
  set_iter!(stats, 0)
  start_time = time()
  set_time!(stats, 0.0)

  # parameters
  @assert init_penalty > 0
  @assert factor_penalty_up > 1
  @assert 0 < factor_primal_linear_improvement < 1
  @assert 0 < factor_decrease_subtol < 1

  # initialization
  @assert length(x0) == nlp.meta.nvar
  @assert length(y0) == nlp.meta.ncon
  x = similar(nlp.meta.x0)
  y = similar(nlp.meta.y0)
  x .= x0
  y .= y0
  set_solution!(stats, x)
  set_constraint_multipliers!(stats, y)

  fx, cx = objcons(nlp, x)
  mu = init_penalty
  alf = AugLagModel(nlp, y, mu, x, fx, cx)

  cviol = norm(alf.cx, Inf)
  cviol_old = Inf
  iter = 0
  subiters = 0
  done = false
  subtol = init_subtol

  if verbose > 0
    @info log_header(
      [:iter, :subiter, :fx, :prim_res, :μ, :normy, :dual_tol, :inner_status],
      [Int, Int, Float64, Float64, Float64, Float64, Float64, Symbol],
    )
    @info log_row(Any[iter, subiters, fx, cviol, alf.μ, norm(y), subtol])
  end

  while !done
    iter += 1

    # dual safeguard
    dual_safeguard(alf)

    # AL subproblem
    subtol = max(subtol, atol)
    subout = with_logger(subsolver_logger) do
      subsolver(alf, h, x = x, atol = subtol, rtol = zero(T), selected = selected)
    end
    x .= subout.solution
    subiters += subout.iter

    # objective
    fx = obj(nlp, x)
    set_objective!(stats, fx)

    # dual estimate
    update_y!(alf)
    set_constraint_multipliers!(stats, alf.y)

    # stationarity measure
    # FIXME it seems that R2 returns no dual_feas in `dual_feas`
    # but in `solver_specific.xi`
    if subout.dual_residual_reliable
      set_dual_residual!(stats, subout.dual_feas)
    end

    # primal violation
    cviol = norm(alf.cx, Inf)
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

    done = stats.status != :unknown

    if verbose > 0 && (mod(stats.iter, verbose) == 0 || done)
      @info log_row(Any[iter, subiters, fx, cviol, alf.μ, norm(alf.y), subtol, subout.status])
    end

    if !done
      if cviol > max(ctol, factor_primal_linear_improvement * cviol_old)
        #@info "decreasing mu"
        mu *= factor_penalty_up
      end
      update_μ!(alf, mu)
      cviol_old = cviol
      subtol *= factor_decrease_subtol
    end
  end
  set_solution!(stats, x)
  set_constraint_multipliers!(stats, alf.y)
  return stats
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