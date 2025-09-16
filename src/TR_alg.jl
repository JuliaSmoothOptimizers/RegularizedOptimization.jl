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
  has_bnds::Bool
  l_bound::V
  u_bound::V
  l_bound_m_x::V
  u_bound_m_x::V
  subsolver::ST
  subpb::PB
  substats::GenericExecutionStats{T, V, V, T}
end

function TRSolver(
  reg_nlp::AbstractRegularizedNLPModel{T, V};
  χ::X = NormLinf(one(T)),
  subsolver = R2Solver,
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

  ψ =
    has_bnds || subsolver == TRDHSolver ?
    shifted(reg_nlp.h, xk, l_bound_m_x, u_bound_m_x, reg_nlp.selected) :
    shifted(reg_nlp.h, xk, T(1), χ)

  Bk =
    isa(reg_nlp.model, QuasiNewtonModel) ? hess_op(reg_nlp.model, xk) :
    hess_op!(reg_nlp.model, xk, similar(xk))
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
    has_bnds,
    l_bound,
    u_bound,
    l_bound_m_x,
    u_bound_m_x,
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

    solver = TR(reg_nlp; χ =  NormLinf(1), subsolver = R2Solver)
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
- `χ::F =  NormLinf(1)`: norm used to define the trust-region;`
- `subsolver::S = R2Solver`: subsolver used to solve the subproblem that appears at each iteration.
- `sub_kwargs::NamedTuple`: a named tuple containing the keyword arguments to be sent to the subsolver. The solver will fail if invalid keyword arguments are provided to the subsolver.

The algorithm stops either when `√(ξₖ/νₖ) < atol + rtol*√(ξ₀/ν₀) ` or `ξₖ < 0` and `√(-ξₖ/νₖ) < neg_tol` where ξₖ := f(xₖ) + h(xₖ) - φ(sₖ; xₖ) - ψ(sₖ; xₖ), and √(ξₖ/νₖ) is a stationarity measure.

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
  solver = TRSolver(reg_nlp, subsolver = subsolver, χ = χ)
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
  has_bnds = solver.has_bnds

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
      [:outer, :inner, :fx, :hx, :xi, :ρ, :Δ, :normx, :norms, :normB, :arrow],
      [Int, Int, T, T, T, T, T, T, T, T, Char],
      hdr_override = Dict{Symbol, String}(   # TODO: Add this as constant dict elsewhere
        :fx => "f(x)",
        :hx => "h(x)",
        :xi => "√(ξ1/ν)",
        :normx => "‖x‖",
        :norms => "‖s‖",
        :normB => "‖B‖",
        :arrow => "TR",
      ),
      colsep = 1,
    )
  end

  local ξ1::T
  local ρk = zero(T)

  α = 1 / eps(T)
  β = 1 / eps(T)

  fk = obj(nlp, xk)
  grad!(nlp, xk, ∇fk)
  ∇fk⁻ .= ∇fk

  quasiNewtTest = isa(nlp, QuasiNewtonModel)
  λmax = T(1)

  λmax, found_λ = opnorm(solver.subpb.model.B)
  found_λ || error("operator norm computation failed")

  ν₁ = α * Δk / (1 + λmax * (α * Δk + 1))
  sqrt_ξ1_νInv = one(T)

  @. mν∇fk = -ν₁ * ∇fk

  set_iter!(stats, 0)
  start_time = time()
  set_time!(stats, 0.0)
  set_objective!(stats, fk + hk)
  set_solver_specific!(stats, :smooth_obj, fk)
  set_solver_specific!(stats, :nonsmooth_obj, hk)

  # models
  φ1 = let ∇fk = ∇fk
    d -> dot(∇fk, d)
  end
  mk1 = let ψ = ψ, φ1 = φ1
    d -> φ1(d) + ψ(d)
  end

  mk = let ψ = ψ, solver = solver
    d -> obj(solver.subpb.model, d) + ψ(d)::T
  end

  prox!(s, ψ, mν∇fk, ν₁)
  ξ1 = hk - mk1(s) + max(1, abs(hk)) * 10 * eps()
  ξ1 > 0 || error("TR: first prox-gradient step should produce a decrease but ξ1 = $(ξ1)")
  sqrt_ξ1_νInv = sqrt(ξ1 / ν₁)

  solved = (ξ1 < 0 && sqrt_ξ1_νInv ≤ neg_tol) || (ξ1 ≥ 0 && sqrt_ξ1_νInv ≤ atol)
  (ξ1 < 0 && sqrt_ξ1_νInv > neg_tol) &&
    error("TR: prox-gradient step should produce a decrease but ξ1 = $(ξ1)")
  atol += rtol * sqrt_ξ1_νInv # make stopping test absolute and relative
  sub_atol += rtol * sqrt_ξ1_νInv

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
    sub_atol = stats.iter == 0 ? 1e-5 : max(sub_atol, min(1e-2, sqrt_ξ1_νInv))
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
          atol = stats.iter == 0 ? 1e-5 : max(sub_atol, min(1e-2, sqrt_ξ1_νInv)),
          Δk = ∆_effective / 10,
          sub_kwargs...
        )
      else
        solve!(
          solver.subsolver,
          solver.subpb,
          solver.substats;
          x = s,
          atol = stats.iter == 0 ? 1e-5 : max(sub_atol, min(1e-2, sqrt_ξ1_νInv)),
          ν = ν₁,
          sub_kwargs...
        )
      end
    end

    s .= solver.substats.solution

    xkn .= xk .+ s
    fkn = obj(nlp, xkn)
    hkn = @views h(xkn[selected])
    sNorm = χ(s)

    Δobj = fk + hk - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
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
          sqrt_ξ1_νInv,
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

      λmax, found_λ = opnorm(solver.subpb.model.B)
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

    set_objective!(stats, fk + hk)
    set_solver_specific!(stats, :smooth_obj, fk)
    set_solver_specific!(stats, :nonsmooth_obj, hk)
    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)

    ν₁ = α * Δk / (1 + λmax * (α * Δk + 1))
    @. mν∇fk = -ν₁ * ∇fk

    prox!(s, ψ, mν∇fk, ν₁)
    ξ1 = hk - mk1(s) + max(1, abs(hk)) * 10 * eps()
    sqrt_ξ1_νInv = sqrt(ξ1 / ν₁)

    solved = (ξ1 < 0 && sqrt_ξ1_νInv ≤ neg_tol) || (ξ1 ≥ 0 && sqrt_ξ1_νInv ≤ atol)
    (ξ1 < 0 && sqrt_ξ1_νInv > neg_tol) &&
      error("TR: prox-gradient step should produce a decrease but ξ1 = $(ξ1)")

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
      Any[stats.iter, solver.substats.iter, fk, hk, sqrt_ξ1_νInv, ρk, Δk, χ(xk), χ(s), λmax, ""],
      colsep = 1,
    )
    @info "TR: terminating with √(ξ1/ν) = $(sqrt_ξ1_νInv)"
  end

  set_solution!(stats, xk)
  set_residuals!(stats, zero(eltype(xk)), sqrt_ξ1_νInv)
end
