export TR, TRSolver, solve!

import SolverCore.solve!

mutable struct TRSolver{
  T <: Real,
  G <: ShiftedProximableFunction,
  V <: AbstractVector{T},
  N,
  ST <: AbstractOptimizationSolver,
  PB <: AbstractRegularizedNLPModel,
} <: AbstractOptimizationSolver
  xk::V
  ∇fk::V
  ∇fk⁻::V
  mν∇fk::V
  ψ::G
  χ::N
  xkn::V
  s::V
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

function TRSolver(
  reg_nlp::AbstractRegularizedNLPModel{T, V};
  χ::X = NormLinf(one(T)),
  subsolver = R2Solver,
  m_monotone::Int = 1,
) where {T, V, X}
  x0 = reg_nlp.model.meta.x0
  l_bound = reg_nlp.model.meta.lvar
  u_bound = reg_nlp.model.meta.uvar

  xk = similar(x0)
  ∇fk = similar(x0)
  ∇fk⁻ = similar(x0)
  mν∇fk = similar(x0)
  xkn = similar(x0)
  s = similar(x0)
  has_bnds = any(l_bound .!= T(-Inf)) || any(u_bound .!= T(Inf))
  if has_bnds || subsolver == TRDHSolver
    l_bound_m_x = similar(xk)
    u_bound_m_x = similar(xk)
    @. l_bound_m_x = l_bound - x0
    @. u_bound_m_x = u_bound - x0
  else
    l_bound_m_x = similar(xk, 0)
    u_bound_m_x = similar(xk, 0)
  end

  m_fh_hist = fill(T(-Inf), m_monotone - 1)

  v0 = [(-1.0)^i for i = 0:(reg_nlp.model.meta.nvar - 1)]
  v0 ./= sqrt(reg_nlp.model.meta.nvar)

  ψ =
    has_bnds || subsolver == TRDHSolver ?
    shifted(reg_nlp.h, xk, l_bound_m_x, u_bound_m_x, reg_nlp.selected) :
    shifted(reg_nlp.h, xk, T(1), χ)

  Bk = hess_op(reg_nlp.model, xk)
  sub_nlp = R2NModel(Bk, ∇fk, zero(T), x0) #FIXME 
  subpb = RegularizedNLPModel(sub_nlp, ψ)
  substats = RegularizedExecutionStats(subpb)
  subsolver = subsolver(subpb)

  return TRSolver(
    xk,
    ∇fk,
    ∇fk⁻,
    mν∇fk,
    ψ,
    χ,
    xkn,
    s,
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
    TR(reg_nlp; kwargs…)
    TR(nlp, h, χ, options; kwargs...)

A trust-region method for the problem

    min f(x) + h(x)

where f: ℝⁿ → ℝ has a Lipschitz-continuous gradient, and h: ℝⁿ → ℝ is
lower semi-continuous and proper.

About each iterate xₖ, a step sₖ is computed as an approximate solution of

    min  φ(s; xₖ) + ψ(s; xₖ)  subject to  ‖s‖ ≤ Δₖ

where φ(s ; xₖ) = f(xₖ) + ∇f(xₖ)ᵀs + ½ sᵀ Bₖ s  is a quadratic approximation of f about xₖ,
ψ(s; xₖ) = h(xₖ + s), ‖⋅‖ is a user-defined norm and Δₖ > 0 is the trust-region radius.

For advanced usage, first define a solver "TRSolver" to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = TRSolver(reg_nlp; χ =  NormLinf(1), subsolver = R2Solver, m_monotone = 1)
    solve!(solver, reg_nlp)

    stats = RegularizedExecutionStats(reg_nlp)
    solve!(solver, reg_nlp, stats)

# Arguments
* `reg_nlp::AbstractRegularizedNLPModel{T, V}`: the problem to solve, see `RegularizedProblems.jl`, `NLPModels.jl`.

# Keyword arguments
- `x::V = nlp.meta.x0`: the initial guess;
- `atol::T = √eps(T)`: absolute tolerance;
- `rtol::T = √eps(T)`: relative tolerance;
- `neg_tol::T = eps(T)^(1 / 4)`: negative tolerance (see stopping conditions below);
- `max_eval::Int = -1`: maximum number of evaluation of the objective function (negative number means unlimited);
- `max_time::Float64 = 30.0`: maximum time limit in seconds;
- `max_iter::Int = 10000`: maximum number of iterations;
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration;
- `Δk::T = T(1)`: initial value of the trust-region radius;
- `η1::T = √√eps(T)`: successful iteration threshold;
- `η2::T = T(0.9)`: very successful iteration threshold;
- `γ::T = T(3)`: trust-region radius parameter multiplier. Must satisfy `γ > 1`. The trust-region radius is updated as Δ := Δ*γ when the iteration is very successful and Δ := Δ/γ when the iteration is unsuccessful;
- `m_monotone::Int = 1`: monotonicity parameter. By default, TR is monotone but the non-monotone variant will be used if `m_monotone > 1`;
- `opnorm_maxiter::Int = 5`: how many iterations of the power method to use to compute the operator norm of Bₖ. If a negative number is provided, then Arpack is used instead;
- `χ::F =  NormLinf(1)`: norm used to define the trust-region;`
- `subsolver::S = R2Solver`: subsolver used to solve the subproblem that appears at each iteration.
- `compute_obj::Bool = true`: (advanced) whether `f(x₀)` should be computed or not. If set to false, then the value is retrieved from `stats.solver_specific[:smooth_obj]`;
- `compute_grad::Bool = true`: (advanced) whether `∇f(x₀)` should be computed or not. If set to false, then the value is retrieved from `solver.∇fk`;
- `sub_kwargs::NamedTuple = NamedTuple()`: a named tuple containing the keyword arguments to be sent to the subsolver. The solver will fail if invalid keyword arguments are provided to the subsolver. For example, if the subsolver is `R2Solver`, you can pass `sub_kwargs = (max_iter = 100, σmin = 1e-6,)`.

The algorithm stops when `‖sᶜᵖ‖/ν < atol + rtol*‖s₀ᶜᵖ‖/ν ` where sᶜᵖ ∈ argminₛ f(xₖ) + ∇f(xₖ)ᵀs + ψ(s; xₖ) ½ ν⁻¹ ‖s‖².

#  Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
$(callback_docstring)
"""
function TR(
  f::AbstractNLPModel,
  h::H,
  χ::X,
  options::ROSolverOptions{R};
  x0::AbstractVector{R} = f.meta.x0,
  subsolver_options = ROSolverOptions(ϵa = options.ϵa),
  selected::AbstractVector{<:Integer} = 1:(f.meta.nvar),
  kwargs...,
) where {H, X, R}
  reg_nlp = RegularizedNLPModel(f, h, selected)
  stats = TR(
    reg_nlp;
    x = x0,
    atol = options.ϵa,
    sub_atol = subsolver_options.ϵa,
    rtol = options.ϵr,
    neg_tol = options.neg_tol,
    verbose = options.verbose,
    max_iter = options.maxIter,
    max_time = options.maxTime,
    Δk = options.Δk,
    η1 = options.η1,
    η2 = options.η2,
    γ = options.γ,
    kwargs...,
  )
  return stats
end

function TR(reg_nlp::AbstractRegularizedNLPModel{T, V}; kwargs...) where {T, V}
  kwargs_dict = Dict(kwargs...)
  subsolver = pop!(kwargs_dict, :subsolver, R2Solver)
  χ = pop!(kwargs_dict, :χ, NormLinf(one(T)))
  m_monotone = pop!(kwargs_dict, :m_monotone, 1)
  solver = TRSolver(reg_nlp, subsolver = subsolver, χ = χ, m_monotone = m_monotone)
  stats = RegularizedExecutionStats(reg_nlp)
  solve!(solver, reg_nlp, stats; kwargs_dict...)
  return stats
end

function SolverCore.solve!(
  solver::TRSolver{T, G, V},
  reg_nlp::AbstractRegularizedNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = reg_nlp.model.meta.x0,
  atol::T = √eps(T),
  sub_atol::T = √eps(T),
  rtol::T = √eps(T),
  neg_tol::T = eps(T)^(1 / 4),
  verbose::Int = 0,
  subsolver_logger::Logging.AbstractLogger = Logging.SimpleLogger(),
  max_iter::Int = 10000,
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  Δk::T = T(1),
  η1::T = √√eps(T),
  η2::T = T(0.9),
  γ::T = T(3),
  sub_kwargs::NamedTuple = NamedTuple(),
  opnorm_maxiter::Int = 5,
  compute_obj::Bool = true,
  compute_grad::Bool = true,
) where {T, G, V}
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
  χ = solver.χ
  m_fh_hist = solver.m_fh_hist .= T(-Inf)
  has_bnds = solver.has_bnds

  m_monotone = length(m_fh_hist) + 1

  if has_bnds || isa(solver.subsolver, TRDHSolver) #TODO elsewhere ?
    l_bound_m_x, u_bound_m_x = solver.l_bound_m_x, solver.u_bound_m_x
    l_bound, u_bound = solver.l_bound, solver.u_bound
    update_bounds!(l_bound_m_x, u_bound_m_x, false, l_bound, u_bound, xk, Δk)
    set_bounds!(ψ, l_bound_m_x, u_bound_m_x)
    set_bounds!(solver.subsolver.ψ, l_bound_m_x, u_bound_m_x)
  else
    set_radius!(ψ, Δk)
    set_radius!(solver.subsolver.ψ, Δk)
  end

  # initialize parameters
  improper = false
  hk = @views h(xk[selected])
  if hk == Inf
    verbose > 0 && @info "TR: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, xk, T(1))
    hk = @views h(xk[selected])
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "TR: found point where h has value" hk
  end
  improper = (hk == -Inf)
  improper == true && @warn "TR: Improper term detected"

  if verbose > 0
    @info log_header(
      [:outer, :inner, :fx, :hx, :norm_s_cauchydν, :ρ, :Δ, :normx, :norms, :normB, :arrow],
      [Int, Int, T, T, T, T, T, T, T, T, Char],
      hdr_override = Dict{Symbol, String}(   # TODO: Add this as constant dict elsewhere
        :fx => "f(x)",
        :hx => "h(x)",
        :norm_s_cauchydν => "‖sᶜᵖ‖/ν",
        :normx => "‖x‖",
        :norms => "‖s‖",
        :normB => "‖B‖",
        :arrow => "TR",
      ),
      colsep = 1,
    )
  end

  local ρk = zero(T)
  local prox_evals::Int = 0

  α = 1 / eps(T)
  β = 1 / eps(T)

  fk = compute_obj ? obj(nlp, xk) : stats.solver_specific[:smooth_obj]
  compute_grad && grad!(nlp, xk, ∇fk)
  ∇fk⁻ .= ∇fk

  quasiNewtTest = isa(nlp, QuasiNewtonModel)
  λmax::T = T(1)
  found_λ = true

  if opnorm_maxiter ≤ 0
    λmax, found_λ = opnorm(solver.subpb.model.B)
  else
    λmax = power_method!(solver.subpb.model.B, solver.v0, solver.subpb.model.v, opnorm_maxiter)
  end
  found_λ || error("operator norm computation failed")

  ν₁ = α * Δk / (1 + λmax * (α * Δk + 1))

  @. mν∇fk = -ν₁ * ∇fk

  set_iter!(stats, 0)
  start_time = time()
  set_time!(stats, 0.0)
  set_objective!(stats, fk + hk)
  set_solver_specific!(stats, :smooth_obj, fk)
  set_solver_specific!(stats, :nonsmooth_obj, hk)
  set_solver_specific!(stats, :prox_evals, prox_evals + 1)
  m_monotone > 1 && (m_fh_hist[stats.iter % (m_monotone - 1) + 1] = fk + hk)

  # models
  mk = let ψ = ψ, solver = solver
    d -> obj(solver.subpb.model, d) + ψ(d)::T
  end

  prox!(s, ψ, mν∇fk, ν₁)
  norm_s_cauchy = norm(s)
  norm_s_cauchydν = norm_s_cauchy / ν₁
  
  atol += rtol * norm_s_cauchydν # make stopping test absolute and relative
  sub_atol += rtol * norm_s_cauchydν

  solved = norm_s_cauchydν ≤ atol

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
    sub_atol = stats.iter == 0 ? 1e-5 : max(sub_atol, min(1e-2, norm_s_cauchydν))
    ∆_effective = min(β * χ(s), Δk)

    if has_bnds || isa(solver.subsolver, TRDHSolver) #TODO elsewhere ?
      update_bounds!(l_bound_m_x, u_bound_m_x, false, l_bound, u_bound, xk, Δk)
      set_bounds!(ψ, l_bound_m_x, u_bound_m_x)
      set_bounds!(solver.subsolver.ψ, l_bound_m_x, u_bound_m_x)
    else
      set_radius!(solver.subsolver.ψ, ∆_effective)
      set_radius!(ψ, ∆_effective)
    end
    with_logger(subsolver_logger) do
      if isa(solver.subsolver, TRDHSolver) #FIXME
        solver.subsolver.D.d[1] = 1/ν₁
        solve!(
          solver.subsolver,
          solver.subpb,
          solver.substats;
          x = s,
          atol = stats.iter == 0 ? 1e-5 : max(sub_atol, min(1e-2, norm_s_cauchydν)),
          Δk = ∆_effective / 10,
          sub_kwargs...,
        )
      else
        solve!(
          solver.subsolver,
          solver.subpb,
          solver.substats;
          x = s,
          atol = stats.iter == 0 ? 1e-5 : max(sub_atol, min(1e-2, norm_s_cauchydν)),
          ν = ν₁,
          sub_kwargs...,
        )
      end
    end

    prox_evals += solver.substats.iter
    s .= solver.substats.solution

    xkn .= xk .+ s
    fkn = obj(nlp, xkn)
    hkn = @views h(xkn[selected])
    sNorm = χ(s)

    fhmax = m_monotone > 1 ? maximum(m_fh_hist) : fk + hk
    Δobj = fhmax - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ξ = hk - mk(s) + max(1, abs(hk)) * 10 * eps()

    if (ξ ≤ 0 || isnan(ξ))
      error("TR: failed to compute a step: ξ = $ξ")
    end

    ρk = Δobj / ξ

    verbose > 0 &&
      stats.iter % verbose == 0 &&
      @info log_row(
        Any[
          stats.iter,
          solver.substats.iter,
          fk,
          hk,
          norm_s_cauchydν,
          ρk,
          Δk,
          χ(xk),
          sNorm,
          λmax,
          (η2 ≤ ρk < Inf) ? '↗' : (ρk < η1 ? '↘' : '='),
        ],
        colsep = 1,
      )

    if η2 ≤ ρk < Inf
      Δk = max(Δk, γ * sNorm)
      if !(has_bnds || isa(solver.subsolver, TRDHSolver))
        set_radius!(ψ, Δk)
        set_radius!(solver.subsolver.ψ, Δk)
      end
    end

    if η1 ≤ ρk < Inf
      xk .= xkn
      if has_bnds || isa(solver.subsolver, TRDHSolver)
        update_bounds!(l_bound_m_x, u_bound_m_x, false, l_bound, u_bound, xk, Δk)
        set_bounds!(ψ, l_bound_m_x, u_bound_m_x)
        set_bounds!(solver.subsolver.ψ, l_bound_m_x, u_bound_m_x)
      end
      fk = fkn
      hk = hkn

      shift!(ψ, xk)
      grad!(nlp, xk, ∇fk)

      if quasiNewtTest
        @. ∇fk⁻ = ∇fk - ∇fk⁻
        push!(nlp, s, ∇fk⁻) # update QN operator
      end

      if opnorm_maxiter ≤ 0
        λmax, found_λ = opnorm(solver.subpb.model.B)
      else
        λmax = power_method!(solver.subpb.model.B, solver.v0, solver.subpb.model.v, opnorm_maxiter)
      end
      found_λ || error("operator norm computation failed")

      ∇fk⁻ .= ∇fk
    end

    if ρk < η1 || ρk == Inf
      Δk = Δk / 2
      if has_bnds || isa(solver.subsolver, TRDHSolver)
        update_bounds!(l_bound_m_x, u_bound_m_x, false, l_bound, u_bound, xk, Δk)
        set_bounds!(ψ, l_bound_m_x, u_bound_m_x)
        set_bounds!(solver.subsolver.ψ, l_bound_m_x, u_bound_m_x)
      else
        set_radius!(ψ, Δk)
        set_radius!(solver.subsolver.ψ, Δk)
      end
    end

    m_monotone > 1 && (m_fh_hist[stats.iter % (m_monotone - 1) + 1] = fk + hk)

    set_objective!(stats, fk + hk)
    set_solver_specific!(stats, :smooth_obj, fk)
    set_solver_specific!(stats, :nonsmooth_obj, hk)
    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)
    set_solver_specific!(stats, :prox_evals, prox_evals + 1)

    ν₁ = α * Δk / (1 + λmax * (α * Δk + 1))
    @. mν∇fk = -ν₁ * ∇fk

    prox!(s, ψ, mν∇fk, ν₁)
    norm_s_cauchy = norm(s)
    norm_s_cauchydν = norm_s_cauchy / ν₁
    
    solved = norm_s_cauchydν ≤ atol

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
      Any[stats.iter, solver.substats.iter, fk, hk, norm_s_cauchydν, ρk, Δk, χ(xk), χ(s), λmax, ""],
      colsep = 1,
    )
    @info "TR: terminating with ‖sᶜᵖ‖/ν = $(norm_s_cauchydν)"
  end

  set_solution!(stats, xk)
  set_residuals!(stats, zero(eltype(xk)), norm_s_cauchydν)
end
