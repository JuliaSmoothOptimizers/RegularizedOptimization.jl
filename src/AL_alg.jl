export AL

function AL(nlp::AbstractNLPModel, h::H, options::ROSolverOptions; kwargs...) where {H}
  if unconstrained(nlp) || bound_constrained(nlp)
    return AL(Val(:unc), nlp, h, options; kwargs...)
  elseif equality_constrained(nlp)
    return AL(Val(:equ), nlp, h, options; kwargs...)
  else # has inequalities
    return AL(Val(:ineq), nlp, h, options; kwargs...)
  end
end

function AL(
  ::Val{:unc},
  nlp::AbstractNLPModel,
  h::H,
  options::ROSolverOptions;
  subsolver = has_bounds(nlp) ? TR : R2,
  kwargs...,
) where {H}
  if !(unconstrained(nlp) || bound_constrained(nlp))
    error(
      "AL(::Val{:unc}, ...) should only be called for unconstrained or bound-constrained problems. Use AL(...)",
    )
  end
  @warn "Problem does not have general explicit constraints; calling solver $(string(subsolver))"
  return subsolver(nlp, h, options; kwargs...)
end

# a uniform solver interface is missing
# TR(nlp, h, options; kwargs...) = TR(nlp, h, NormLinf(1.0), options; kwargs...)

function AL(
  ::Val{:ineq},
  nlp::AbstractNLPModel,
  h::H,
  options::ROSolverOptions{T};
  x0::AbstractVector{T} = nlp.meta.x0,
  kwargs...,
) where {H, T}
  if nlp.meta.ncon == 0 || equality_constrained(nlp)
    error("AL(::Val{:ineq}, ...) should only be called for problems with inequalities. Use AL(...)")
  end
  snlp = nlp isa AbstractNLSModel ? SlackNLSModel(nlp) : SlackModel(nlp)
  if length(x0) != snlp.meta.nvar
    x0s = zeros(T, snlp.meta.nvar)
    x0s[1:(nlp.meta.nvar)] .= x0
  else
    x0s = x0
  end
  output = AL(Val(:equ), snlp, h, options; x0 = x0s, kwargs...)
  output.solution = output.solution[1:(nlp.meta.nvar)]
  return output
end

"""
    AL(nlp, h, options; kwargs...)

An augmented Lagrangian method for the problem

    min f(x) + h(x) subject to lvar ≤ x ≤ uvar, lcon ≤ c(x) ≤ ucon

where f: ℝⁿ → ℝ, c: ℝⁿ → ℝᵐ and their derivatives are Lipschitz continuous and h: ℝⁿ → ℝ is
lower semi-continuous, proper and prox-bounded.

At each iteration, an iterate x is computed as an approximate solution of

    min  L(x;y,μ) + h(x) subject to lvar ≤ x ≤ uvar

where y is an estimate of the Lagrange multiplier vector for the constraints lcon ≤ c(x) ≤ ucon, 
μ is the penalty parameter and L(⋅;y,μ) is the augmented Lagrangian function defined by

    L(x;y,μ) := f(x) - yᵀc(x) + ½ μ ‖c(x)‖².

### Arguments

* `nlp::AbstractNLPModel`: a smooth optimization problem
* `h`: a regularizer such as those defined in ProximalOperators
* `options::ROSolverOptions`: a structure containing algorithmic parameters

The objective and gradient of `nlp` will be accessed.
The Hessian of `nlp` may be accessed or not, depending on the subsolver adopted.
If adopted, the Hessian is accessed as an abstract operator and need not be the exact Hessian.

### Keyword arguments

* `x0::AbstractVector`: a primal initial guess (default: `nlp.meta.x0`)
* `y0::AbstractVector`: a dual initial guess (default: `nlp.meta.y0`)
* `subsolver`: the procedure used to compute a step (e.g. `PG`, `R2`, `TR` or `TRDH`)
* `subsolver_logger::AbstractLogger`: a logger to pass to the subproblem solver
* `subsolver_options::ROSolverOptions`: default options to pass to the subsolver.

### Return values

* `stats::GenericExecutionStats`: solution and other info.

"""
function AL(
  ::Val{:equ},
  nlp::AbstractNLPModel,
  h::H,
  options::ROSolverOptions{T};
  x0::AbstractVector{T} = nlp.meta.x0,
  y0::AbstractVector{T} = nlp.meta.y0,
  subsolver = has_bounds(nlp) ? TR : R2,
  subsolver_logger::Logging.AbstractLogger = Logging.NullLogger(),
  subsolver_options::ROSolverOptions{T} = ROSolverOptions{T}(),
  init_penalty::Real = T(10),
  factor_penalty_up::Real = T(2),
  ctol::Real = options.ϵa > 0 ? options.ϵa : options.ϵr,
  init_subtol::Real = T(0.1),
  factor_primal_linear_improvement::Real = T(3 // 4),
  factor_decrease_subtol::Real = T(1 // 4),
) where {H, T <: Real}
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
  ymin = -1e20
  ymax = 1e20
  @assert ymin <= 0
  @assert ymax >= 0
  verbose = options.verbose
  max_time = options.maxTime
  max_iter = options.maxIter

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

  V = norm(alf.cx, Inf)
  V_old = Inf
  iter = 0
  subiters = 0
  done = false

  suboptions = subsolver_options
  subtol = init_subtol

  if verbose > 0
    @info log_header(
      [:iter, :subiter, :fx, :prim_res, :μ, :normy, :dual_tol, :inner_status],
      [Int, Int, Float64, Float64, Float64, Float64, Float64, Symbol],
    )
    @info log_row(Any[iter, subiters, fx, V, alf.μ, norm(y), subtol])
  end

  while !done
    iter += 1

    # dual safeguard
    project_y!(alf, ymin, ymax)

    # AL subproblem
    suboptions.ϵa = max(subtol, options.ϵa)
    suboptions.ϵr = max(subtol, options.ϵr)
    subout = with_logger(subsolver_logger) do
      subsolver(alf, h, suboptions, x0 = x)
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
    if subout.dual_residual_reliable
      set_dual_residual!(stats, subout.dual_feas)
    end

    # primal violation
    V = norm(alf.cx, Inf)
    set_primal_residual!(stats, V)

    # termination checks
    dual_ok =
      subout.status_reliable &&
      subout.status == :first_order &&
      suboptions.ϵa <= options.ϵa &&
      suboptions.ϵr <= options.ϵr
    primal_ok = V <= ctol
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
        max_time = max_time,
        max_iter = max_iter,
      ),
    )

    done = stats.status != :unknown

    if verbose > 0 && (mod(stats.iter, verbose) == 0 || done)
      @info log_row(Any[iter, subiters, fx, V, alf.μ, norm(alf.y), subtol, subout.status])
    end

    if !done
      if V > max(ctol, factor_primal_linear_improvement * V_old)
        #@info "decreasing mu"
        mu *= factor_penalty_up
      end
      update_μ!(alf, mu)
      V_old = V
      subtol *= factor_decrease_subtol
    end
  end
  set_solution!(stats, x)
  set_constraint_multipliers!(stats, alf.y)
  return stats
end

"""
    project_y!(nlp, ymin, ymax)

Given an `AugLagModel`, project `nlp.y` into [ymin, ymax]and updates `nlp.μc_y` accordingly.
"""
function project_y!(
  nlp::AugLagModel,
  ymin::AbstractVector{T},
  ymax::AbstractVector{T},
) where {T <: Real}
  nlp.y .= max.(ymin, min.(nlp.y, ymax))
  nlp.μc_y .= nlp.μ .* nlp.cx .- nlp.y
end

function project_y!(nlp::AugLagModel, ymin::T, ymax::T) where {T <: Real}
  nlp.y .= max.(ymin, min.(nlp.y, ymax))
  nlp.μc_y .= nlp.μ .* nlp.cx .- nlp.y
end