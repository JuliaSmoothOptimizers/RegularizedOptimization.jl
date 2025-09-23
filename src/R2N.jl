export R2N, R2NSolver, solve!

import SolverCore.solve!

mutable struct R2NSolver{
  T <: Real,
  G <: ShiftedProximableFunction,
  V <: AbstractVector{T},
  ST <: AbstractOptimizationSolver,
  PB <: AbstractRegularizedNLPModel,
} <: AbstractOptimizationSolver
  xk::V
  ∇fk::V
  ∇fk⁻::V
  y::V
  mν∇fk::V
  ψ::G
  xkn::V
  s::V
  s1::V
  v0::V
  has_bnds::Bool
  l_bound::V
  u_bound::V
  l_bound_m_x::V
  u_bound_m_x::V
  m_fh_hist::V
  subsolver::ST
  subpb::PB
  substats::GenericExecutionStats{T, V, V, T}
end

function R2NSolver(
  reg_nlp::AbstractRegularizedNLPModel{T, V};
  subsolver = R2Solver,
  m_monotone::Int = 1,
) where {T, V}
  x0 = reg_nlp.model.meta.x0
  l_bound = reg_nlp.model.meta.lvar
  u_bound = reg_nlp.model.meta.uvar

  xk = similar(x0)
  ∇fk = similar(x0)
  ∇fk⁻ = similar(x0)
  y = similar(x0)
  mν∇fk = similar(x0)
  xkn = similar(x0)
  s = similar(x0)
  s1 = similar(x0)

  v0 = [(-1.0)^i for i in 0:(reg_nlp.model.meta.nvar-1)]
  v0 ./= sqrt(reg_nlp.model.meta.nvar)

  has_bnds = any(l_bound .!= T(-Inf)) || any(u_bound .!= T(Inf))
  if has_bnds
    l_bound_m_x = similar(xk)
    u_bound_m_x = similar(xk)
    @. l_bound_m_x = l_bound - x0
    @. u_bound_m_x = u_bound - x0
  else
    l_bound_m_x = similar(xk, 0)
    u_bound_m_x = similar(xk, 0)
  end
  m_fh_hist = fill(T(-Inf), m_monotone - 1)

  ψ =
    has_bnds ? shifted(reg_nlp.h, xk, l_bound_m_x, u_bound_m_x, reg_nlp.selected) :
    shifted(reg_nlp.h, xk)

  Bk = hess_op(reg_nlp.model, x0)
  sub_nlp = R2NModel(Bk, ∇fk, T(1), x0)
  subpb = RegularizedNLPModel(sub_nlp, ψ)
  substats = RegularizedExecutionStats(subpb)
  subsolver = subsolver(subpb)

  return R2NSolver{T, typeof(ψ), V, typeof(subsolver), typeof(subpb)}(
    xk,
    ∇fk,
    ∇fk⁻,
    y,
    mν∇fk,
    ψ,
    xkn,
    s,
    s1,
    v0,
    has_bnds,
    l_bound,
    u_bound,
    l_bound_m_x,
    u_bound_m_x,
    m_fh_hist,
    subsolver,
    subpb,
    substats,
  )
end

"""
    R2N(reg_nlp; kwargs…)

A second-order quadratic regularization method for the problem

    min f(x) + h(x)

where f: ℝⁿ → ℝ is C¹, and h: ℝⁿ → ℝ is
lower semi-continuous, proper and prox-bounded.

About each iterate xₖ, a step sₖ is computed as a solution of

    min  φ(s; xₖ) + ½ σₖ ‖s‖² + ψ(s; xₖ)

where φ(s ; xₖ) = f(xₖ) + ∇f(xₖ)ᵀs + ½ sᵀBₖs is a quadratic approximation of f about xₖ,
ψ(s; xₖ) is either h(xₖ + s) or an approximation of h(xₖ + s), ‖⋅‖ is the ℓ₂ norm and σₖ > 0 is the regularization parameter.

For advanced usage, first define a solver "R2NSolver" to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = R2NSolver(reg_nlp; m_monotone = 1)
    solve!(solver, reg_nlp)

    stats = RegularizedExecutionStats(reg_nlp)
    solve!(solver, reg_nlp, stats)
  
# Arguments
* `reg_nlp::AbstractRegularizedNLPModel{T, V}`: the problem to solve, see `RegularizedProblems.jl`, `NLPModels.jl`.

# Keyword arguments 
- `x::V = nlp.meta.x0`: the initial guess;
- `atol::T = √eps(T)`: absolute tolerance;
- `rtol::T = √eps(T)`: relative tolerance;
- `neg_tol::T = eps(T)^(1 / 4)`: negative tolerance;
- `max_eval::Int = -1`: maximum number of evaluation of the objective function (negative number means unlimited);
- `max_time::Float64 = 30.0`: maximum time limit in seconds;
- `max_iter::Int = 10000`: maximum number of iterations;
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration;
- `σmin::T = eps(T)`: minimum value of the regularization parameter;
- `σk::T = eps(T)^(1 / 5)`: initial value of the regularization parameter;
- `η1::T = √√eps(T)`: successful iteration threshold;
- `η2::T = T(0.9)`: very successful iteration threshold;
- `γ::T = T(3)`: regularization parameter multiplier, σ := σ/γ when the iteration is very successful and σ := σγ when the iteration is unsuccessful;
- `θ::T = 1/(1 + eps(T)^(1 / 5))`: is the model decrease fraction with respect to the decrease of the Cauchy model;
- `opnorm_maxiter::Int = 1`: how many iterations of the power method to use to compute the operator norm of Bₖ. If a negative number is provided, then Arpack is used instead;
- `m_monotone::Int = 1`: monotonicity parameter. By default, R2N is monotone but the non-monotone variant will be used if `m_monotone > 1`;
- `sub_kwargs::NamedTuple = NamedTuple()`: a named tuple containing the keyword arguments to be sent to the subsolver. The solver will fail if invalid keyword arguments are provided to the subsolver. For example, if the subsolver is `R2Solver`, you can pass `sub_kwargs = (max_iter = 100, σmin = 1e-6,)`.

The algorithm stops either when `√(ξₖ/νₖ) < atol + rtol*√(ξ₀/ν₀) ` or `ξₖ < 0` and `√(-ξₖ/νₖ) < neg_tol` where ξₖ := f(xₖ) + h(xₖ) - φ(sₖ; xₖ) - ψ(sₖ; xₖ), and √(ξₖ/νₖ) is a stationarity measure.

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
The callback is called at each iteration.
The expected signature of the callback is `callback(nlp, solver, stats)`, and its output is ignored.
Changing any of the input arguments will affect the subsequent iterations.
In particular, setting `stats.status = :user` will stop the algorithm.
All relevant information should be available in `nlp` and `solver`.
Notably, you can access, and modify, the following:
- `solver.xk`: current iterate;
- `solver.∇fk`: current gradient;
- `stats`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
  - `stats.iter`: current iteration counter;
  - `stats.objective`: current objective function value;
  - `stats.solver_specific[:smooth_obj]`: current value of the smooth part of the objective function;
  - `stats.solver_specific[:nonsmooth_obj]`: current value of the nonsmooth part of the objective function;
  - `stats.status`: current status of the algorithm. Should be `:unknown` unless the algorithm has attained a stopping criterion. Changing this to anything other than `:unknown` will stop the algorithm, but you should use `:user` to properly indicate the intention;
  - `stats.elapsed_time`: elapsed time in seconds.
Similarly to the callback, when using a quasi-Newton approximation, two functions, `qn_update_y!(nlp, solver, stats)` and `qn_copy!(nlp, solver, stats)` are called at each update of the approximation.
Namely, the former computes the `y` vector for which the pair `(s, y)` is pushed into the approximation.
By default, `y := ∇fk⁻ - ∇fk`.
The latter allows the user to tell which values should be copied for the next iteration.
By default, only the gradient is copied: `∇fk⁻ .= ∇fk`.
This might be useful when using R2N in a constrained optimization context, when the gradient of the Lagrangian function is pushed at each iteration rather than the gradient of the objective function.
"""
function R2N(
  nlp::AbstractNLPModel{T, V},
  h,
  options::ROSolverOptions{T};
  kwargs...,
) where {T <: Real, V}
  kwargs_dict = Dict(kwargs...)
  selected = pop!(kwargs_dict, :selected, 1:(nlp.meta.nvar))
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  reg_nlp = RegularizedNLPModel(nlp, h, selected)
  return R2N(
    reg_nlp,
    x = x0,
    atol = options.ϵa,
    rtol = options.ϵr,
    neg_tol = options.neg_tol,
    verbose = options.verbose,
    max_iter = options.maxIter,
    max_time = options.maxTime,
    σmin = options.σmin,
    σk = options.σk,
    η1 = options.η1,
    η2 = options.η2,
    γ = options.γ;
    kwargs_dict...,
  )
end

function R2N(reg_nlp::AbstractRegularizedNLPModel; kwargs...)
  kwargs_dict = Dict(kwargs...)
  m_monotone = pop!(kwargs_dict, :m_monotone, 1)
  subsolver = pop!(kwargs_dict, :subsolver, R2Solver)
  solver = R2NSolver(reg_nlp, subsolver = subsolver, m_monotone = m_monotone)
  stats = GenericExecutionStats(reg_nlp.model)
  solve!(solver, reg_nlp, stats; kwargs_dict...)
  return stats
end

function SolverCore.solve!(
  solver::R2NSolver{T, G, V},
  reg_nlp::AbstractRegularizedNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  qn_update_y!::Function = _qn_grad_update_y!,
  qn_copy!::Function = _qn_grad_copy!,
  x::V = reg_nlp.model.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  neg_tol::T = eps(T)^(1 / 4),
  verbose::Int = 0,
  max_iter::Int = 10000,
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  σk::T = eps(T)^(1 / 5),
  σmin::T = eps(T),
  η1::T = √√eps(T),
  η2::T = T(0.9),
  γ::T = T(3),
  β::T = 1 / eps(T),
  θ::T = 1/(1 + eps(T)^(1 / 5)),
  opnorm_maxiter::Int = 1,
  sub_kwargs::NamedTuple = NamedTuple(),
) where {T, V, G}
  reset!(stats)

  # Retrieve workspace
  selected = reg_nlp.selected
  h = reg_nlp.h
  nlp = reg_nlp.model

  xk = solver.xk .= x

  # Make sure ψ has the correct shift 
  shift!(solver.ψ, xk)

  ∇fk = solver.∇fk
  ∇fk⁻ = solver.∇fk⁻
  mν∇fk = solver.mν∇fk
  ψ = solver.ψ
  xkn = solver.xkn
  s = solver.s
  s1 = solver.s1
  m_fh_hist = solver.m_fh_hist .= T(-Inf)
  has_bnds = solver.has_bnds

  if has_bnds
    l_bound_m_x = solver.l_bound_m_x
    u_bound_m_x = solver.u_bound_m_x
    l_bound = solver.l_bound
    u_bound = solver.u_bound
  end
  m_monotone = length(m_fh_hist) + 1

  # initialize parameters
  improper = false
  hk = @views h(xk[selected])
  if hk == Inf
    verbose > 0 && @info "R2N: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, xk, T(1))
    hk = @views h(xk[selected])
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "R2N: found point where h has value" hk
  end
  improper = (hk == -Inf)
  improper == true && @warn "R2N: Improper term detected"
  improper == true && return stats

  if verbose > 0
    @info log_header(
      [:outer, :inner, :fx, :hx, :xi, :ρ, :σ, :normx, :norms, :normB, :arrow],
      [Int, Int, T, T, T, T, T, T, T, T, Char],
      hdr_override = Dict{Symbol, String}(
        :fx => "f(x)",
        :hx => "h(x)",
        :xi => "√(ξ1/ν)",
        :normx => "‖x‖",
        :norms => "‖s‖",
        :normB => "‖B‖",
        :arrow => "R2N",
      ),
      colsep = 1,
    )
  end

  local ξ1::T
  local ρk::T = zero(T)

  fk = obj(nlp, xk)
  grad!(nlp, xk, ∇fk)
  qn_copy!(nlp, solver, stats)

  quasiNewtTest = isa(nlp, QuasiNewtonModel)
  λmax::T = T(1)
  found_λ = true
  solver.subpb.model.B = hess_op(nlp, xk)

  if opnorm_maxiter ≤ 0
    λmax, found_λ = opnorm(solver.subpb.model.B)
  else
    λmax = power_method!(solver.subpb.model.B, solver.v0, solver.subpb.model.v, opnorm_maxiter)
  end
  
  found_λ || error("operator norm computation failed")

  ν₁ = θ / (λmax + σk)

  sqrt_ξ1_νInv = one(T)

  @. mν∇fk = -ν₁ * ∇fk

  set_iter!(stats, 0)
  start_time = time()
  set_time!(stats, 0.0)
  set_objective!(stats, fk + hk)
  set_solver_specific!(stats, :smooth_obj, fk)
  set_solver_specific!(stats, :nonsmooth_obj, hk)
  set_solver_specific!(stats, :sigma, σk)
  set_solver_specific!(stats, :sigma_cauchy, 1/ν₁)
  m_monotone > 1 && (m_fh_hist[stats.iter % (m_monotone - 1) + 1] = fk + hk)

  φ1 = let ∇fk = ∇fk
    d -> dot(∇fk, d)
  end

  mk1 = let ψ = ψ
    d -> φ1(d) + ψ(d)::T
  end

  mk = let ψ = ψ, solver = solver
    d -> obj(solver.subpb.model, d) + ψ(d)::T
  end

  prox!(s1, ψ, mν∇fk, ν₁)
  mks = mk1(s1)

  ξ1 = hk - mks + max(1, abs(hk)) * 10 * eps()
  sqrt_ξ1_νInv = ξ1 ≥ 0 ? sqrt(ξ1 / ν₁) : sqrt(-ξ1 / ν₁)
  solved = (ξ1 < 0 && sqrt_ξ1_νInv ≤ neg_tol) || (ξ1 ≥ 0 && sqrt_ξ1_νInv ≤ atol)
  (ξ1 < 0 && sqrt_ξ1_νInv > neg_tol) &&
    error("R2N: prox-gradient step should produce a decrease but ξ1 = $(ξ1)")
  atol += rtol * sqrt_ξ1_νInv # make stopping test absolute and relative

  set_status!(
    stats,
    get_status(
      reg_nlp,
      elapsed_time = stats.elapsed_time,
      iter = stats.iter,
      optimal = solved,
      improper = improper,
      max_eval = max_eval,
      max_time = max_time,
      max_iter = max_iter,
    ),
  )

  callback(nlp, solver, stats)

  done = stats.status != :unknown

  while !done
    sub_atol = stats.iter == 0 ? 1.0e-3 : min(sqrt_ξ1_νInv ^ (1.5), sqrt_ξ1_νInv * 1e-3)

    solver.subpb.model.σ = σk
    isa(solver.subsolver, R2DHSolver) && (solver.subsolver.D.d[1] = 1/ν₁)
    if isa(solver.subsolver, R2Solver) #FIXME
      solve!(
        solver.subsolver,
        solver.subpb,
        solver.substats;
        x = s1,
        ν = ν₁,
        atol = sub_atol,
        sub_kwargs...,
      )
    else
      solve!(
        solver.subsolver,
        solver.subpb,
        solver.substats;
        x = s1,
        σk = σk,
        atol = sub_atol,
        sub_kwargs...,
      )
    end

    s .= solver.substats.solution

    if norm(s) > β * norm(s1)
      s .= s1
    end

    xkn .= xk .+ s
    fkn = obj(nlp, xkn)
    hkn = @views h(xkn[selected])
    mks = mk(s)

    fhmax = m_monotone > 1 ? maximum(m_fh_hist) : fk + hk
    Δobj = fhmax - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    Δmod = fhmax - (fk + mks) + max(1, abs(fhmax)) * 10 * eps()
    ξ = hk - mks + max(1, abs(hk)) * 10 * eps()

    if (ξ < 0 || isnan(ξ))
      error("R2N: failed to compute a step: ξ = $ξ")
    end

    ρk = Δobj / Δmod

    verbose > 0 &&
      stats.iter % verbose == 0 &&
      @info log_row(
        Any[
          stats.iter,
          solver.substats.iter,
          fk,
          hk,
          sqrt_ξ1_νInv,
          ρk,
          σk,
          norm(xk),
          norm(s),
          λmax,
          (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "="),
        ],
        colsep = 1,
      )

    if η1 ≤ ρk < Inf
      xk .= xkn
      if has_bnds
        @. l_bound_m_x = l_bound - xk
        @. u_bound_m_x = u_bound - xk
        set_bounds!(ψ, l_bound_m_x, u_bound_m_x)
      end
      #update functions
      fk = fkn
      hk = hkn

      shift!(ψ, xk)
      grad!(nlp, xk, ∇fk)

      if quasiNewtTest
        qn_update_y!(nlp, solver, stats)
        push!(nlp, s, solver.y)
        qn_copy!(nlp, solver, stats)
      end
      solver.subpb.model.B = hess_op(nlp, xk)

      if opnorm_maxiter ≤ 0
        λmax, found_λ = opnorm(solver.subpb.model.B)
      else
        λmax = power_method!(solver.subpb.model.B, solver.v0, solver.subpb.model.v, opnorm_maxiter)
      end
      
      found_λ || error("operator norm computation failed")
    end

    if η2 ≤ ρk < Inf
      σk = max(σk/γ, σmin)
    end

    if ρk < η1 || ρk == Inf
      σk = σk * γ
    end

    ν₁ = θ / (λmax + σk)
    m_monotone > 1 && (m_fh_hist[stats.iter % (m_monotone - 1) + 1] = fk + hk)

    set_objective!(stats, fk + hk)
    set_solver_specific!(stats, :smooth_obj, fk)
    set_solver_specific!(stats, :nonsmooth_obj, hk)
    set_solver_specific!(stats, :sigma, σk)
    set_solver_specific!(stats, :sigma_cauchy, 1/ν₁)
    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)

    @. mν∇fk = - ν₁ * ∇fk
    prox!(s1, ψ, mν∇fk, ν₁)
    mks = mk1(s1)

    ξ1 = hk - mks + max(1, abs(hk)) * 10 * eps()

    sqrt_ξ1_νInv = ξ1 ≥ 0 ? sqrt(ξ1 / ν₁) : sqrt(-ξ1 / ν₁)
    solved = (ξ1 < 0 && sqrt_ξ1_νInv ≤ neg_tol) || (ξ1 ≥ 0 && sqrt_ξ1_νInv ≤ atol)

    (ξ1 < 0 && sqrt_ξ1_νInv > neg_tol) &&
      error("R2N: prox-gradient step should produce a decrease but ξ1 = $(ξ1)")
    set_status!(
      stats,
      get_status(
        reg_nlp,
        elapsed_time = stats.elapsed_time,
        iter = stats.iter,
        optimal = solved,
        improper = improper,
        max_eval = max_eval,
        max_time = max_time,
        max_iter = max_iter,
      ),
    )

    callback(nlp, solver, stats)

    done = stats.status != :unknown
  end

  if verbose > 0 && stats.status == :first_order
    @info log_row(
      Any[
        stats.iter,
        0,
        fk,
        hk,
        sqrt_ξ1_νInv,
        ρk,
        σk,
        norm(xk),
        norm(s),
        λmax,
        (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "="),
      ],
      colsep = 1,
    )
    @info "R2N: terminating with √(ξ1/ν) = $(sqrt_ξ1_νInv)"
  end

  set_solution!(stats, xk)
  set_residuals!(stats, zero(eltype(xk)), sqrt_ξ1_νInv)
  return stats
end

function _qn_grad_update_y!(
  nlp::AbstractNLPModel{T, V},
  solver::R2NSolver{T, G, V},
  stats::GenericExecutionStats,
) where {T, V, G}
  @. solver.y = solver.∇fk - solver.∇fk⁻
end

function _qn_grad_copy!(
  nlp::AbstractNLPModel{T, V},
  solver::R2NSolver{T, G, V},
  stats::GenericExecutionStats,
) where {T, V, G}
  solver.∇fk⁻ .= solver.∇fk
end
