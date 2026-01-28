export LMTR, LMTRSolver, solve!

import SolverCore.solve!

mutable struct LMTRSolver{
  T <: Real,
  G <: ShiftedProximableFunction,
  V <: AbstractVector{T},
  N,
  ST <: AbstractOptimizationSolver,
  PB <: AbstractRegularizedNLPModel,
} <: AbstractOptimizationSolver
  xk::V
  ∇fk::V
  mν∇fk::V
  Fk::V
  Fkn::V
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

function LMTRSolver(
  reg_nls::AbstractRegularizedNLPModel{T, V};
  subsolver = R2Solver,
  χ = NormLinf(one(T)),
) where {T, V}
  x0 = reg_nls.model.meta.x0
  l_bound = reg_nls.model.meta.lvar
  u_bound = reg_nls.model.meta.uvar

  xk = similar(x0)
  ∇fk = similar(x0)
  mν∇fk = similar(x0)
  Fk = similar(x0, reg_nls.model.nls_meta.nequ)
  Fkn = similar(Fk)
  xkn = similar(x0)
  s = similar(x0)
  has_bnds = any(l_bound .!= T(-Inf)) || any(u_bound .!= T(Inf)) || subsolver == TRDHSolver
  if has_bnds
    l_bound_m_x = similar(xk)
    u_bound_m_x = similar(xk)
    @. l_bound_m_x = l_bound - x0
    @. u_bound_m_x = u_bound - x0
  else
    l_bound_m_x = similar(xk, 0)
    u_bound_m_x = similar(xk, 0)
  end

  ψ =
    has_bnds ?
    shifted(
      reg_nls.h,
      xk,
      max.(-one(T), l_bound_m_x),
      min.(one(T), u_bound_m_x),
      reg_nls.selected,
    ) : shifted(reg_nls.h, xk, one(T), χ)

  Jk = jac_op_residual(reg_nls.model, xk)
  sub_nlp = LMModel(Jk, Fk, T(0), xk)
  subpb = RegularizedNLPModel(sub_nlp, ψ)
  substats = RegularizedExecutionStats(subpb)
  subsolver = subsolver(subpb)

  return LMTRSolver{T, typeof(ψ), V, typeof(χ), typeof(subsolver), typeof(subpb)}(
    xk,
    ∇fk,
    mν∇fk,
    Fk,
    Fkn,
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
    LMTR(reg_nls; kwargs...)
    LMTR(nls, h, χ, options; kwargs...)

A trust-region Levenberg-Marquardt method for the problem

    min ½ ‖F(x)‖² + h(x)

where F: ℝⁿ → ℝᵐ and its Jacobian J are Lipschitz continuous and h: ℝⁿ → ℝ is
lower semi-continuous and proper.

At each iteration, a step s is computed as an approximate solution of

    min  ½ ‖J(x) s + F(x)‖₂² + ψ(s; x)  subject to  ‖s‖ ≤ Δ

where F(x) and J(x) are the residual and its Jacobian at x, respectively, ψ(s; x) = h(x + s),
‖⋅‖ is a user-defined norm and Δ > 0 is a trust-region radius.

For advanced usage, first define a solver "LMSolver" to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = LMTRSolver(reg_nls; χ = NormLinf(one(T)), subsolver = R2Solver)
    solve!(solver, reg_nls)

    stats = RegularizedExecutionStats(reg_nls)
    solve!(solver, reg_nls, stats)
  
# Arguments
* `reg_nls::AbstractRegularizedNLPModel{T, V}`: the problem to solve, see `RegularizedProblems.jl`, `NLPModels.jl`.

# Keyword arguments
- `x::V = nlp.meta.x0`: the initial guess;
- `atol::T = √eps(T)`: absolute tolerance;
- `sub_atol::T = atol`: subsolver absolute tolerance;
- `rtol::T = √eps(T)`: relative tolerance;
- `neg_tol::T = zero(T): negative tolerance;
- `max_eval::Int = -1`: maximum number of evaluation of the objective function (negative number means unlimited);
- `max_time::Float64 = 30.0`: maximum time limit in seconds;
- `max_iter::Int = 10000`: maximum number of iterations;
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration;
- `Δk::T = eps(T)`: initial value of the trust-region radius;
- `η1::T = √√eps(T)`: successful iteration threshold;
- `η2::T = T(0.9)`: very successful iteration threshold;
- `γ::T = T(3)`: trust-region radius parameter multiplier, Δ := Δ*γ when the iteration is very successful and Δ := Δ/γ when the iteration is unsuccessful;
- `α::T = 1/eps(T)`: TODO
- `β::T = 1/eps(T)`: TODO
- `χ =  NormLinf(1)`: norm used to define the trust-region;`
- `subsolver::S = R2Solver`: subsolver used to solve the subproblem that appears at each iteration.
- `sub_kwargs::NamedTuple = NamedTuple()`: a named tuple containing the keyword arguments to be sent to the subsolver. The solver will fail if invalid keyword arguments are provided to the subsolver. For example, if the subsolver is `R2Solver`, you can pass `sub_kwargs = (max_iter = 100, σmin = 1e-6,)`.

The algorithm stops when `‖sᶜᵖ‖/ν < atol + rtol*‖s₀ᶜᵖ‖/ν ` where sᶜᵖ ∈ argminₛ f(xₖ) + ∇f(xₖ)ᵀs + ψ(s; xₖ) ½ ν⁻¹ ‖s‖².

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
$(callback_docstring)
"""
function LMTR(nls::AbstractNLSModel, h::H, χ::X, options::ROSolverOptions; kwargs...) where {H, X}
  kwargs_dict = Dict(kwargs...)
  selected = pop!(kwargs_dict, :selected, 1:(nls.meta.nvar))
  reg_nls = RegularizedNLPModel(nls, h, selected)
  x0 = pop!(kwargs_dict, :x0, nls.meta.x0)
  return LMTR(
    reg_nls;
    x = x0,
    χ = χ,
    atol = options.ϵa,
    rtol = options.ϵr,
    neg_tol = options.neg_tol,
    verbose = options.verbose,
    max_iter = options.maxIter,
    max_time = options.maxTime,
    Δk = options.Δk,
    η1 = options.η1,
    η2 = options.η2,
    γ = options.γ,
    α = options.α,
    β = options.β,
    kwargs_dict...,
  )
end

function LMTR(reg_nls::AbstractRegularizedNLPModel{T}; kwargs...) where {T}
  kwargs_dict = Dict(kwargs...)
  subsolver = pop!(kwargs_dict, :subsolver, R2Solver)
  χ = pop!(kwargs_dict, :χ, NormLinf(one(T)))
  solver = LMTRSolver(reg_nls, χ = χ, subsolver = subsolver)
  stats = RegularizedExecutionStats(reg_nls)
  solve!(solver, reg_nls, stats; kwargs_dict...)
end

function SolverCore.solve!(
  solver::LMTRSolver{T, G, V},
  reg_nls::AbstractRegularizedNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = reg_nls.model.meta.x0,
  atol::T = √eps(T),
  sub_atol::T = atol,
  rtol::T = √eps(T),
  neg_tol::T = zero(T),
  verbose::Int = 0,
  max_iter::Int = 10000,
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  Δk::T = T(1),
  η1::T = √√eps(T),
  η2::T = T(0.9),
  γ::T = T(3),
  α::T = 1 / eps(T),
  β::T = 1 / eps(T),
  sub_kwargs::NamedTuple = NamedTuple(),
) where {T, G, V}
  reset!(stats)

  # Retrieve workspace
  selected = reg_nls.selected
  h = reg_nls.h
  nls = reg_nls.model

  xk = solver.xk .= x

  # Make sure ψ has the correct shift 
  shift!(solver.ψ, xk)

  Fk = solver.Fk
  Fkn = solver.Fkn
  ∇fk = solver.∇fk
  mν∇fk = solver.mν∇fk
  ψ = solver.ψ
  χ = solver.χ
  xkn = solver.xkn
  s = solver.s
  has_bnds = solver.has_bnds
  if has_bnds
    l_bound_m_x, u_bound_m_x = solver.l_bound_m_x, solver.u_bound_m_x
    l_bound, u_bound = solver.l_bound, solver.u_bound
    update_bounds!(l_bound_m_x, u_bound_m_x, false, l_bound, u_bound, xk, Δk)
    set_bounds!(ψ, l_bound_m_x, u_bound_m_x)
    set_bounds!(solver.subsolver.ψ, l_bound_m_x, u_bound_m_x)
  end

  # initialize parameters
  improper = false
  hk = @views h(xk[selected])
  if hk == Inf
    verbose > 0 && @info "LM: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, xk, one(T))
    hk = @views h(xk[selected])
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "LM: found point where h has value" hk
  end
  improper = (hk == -Inf)
  improper == true && @warn "LM: Improper term detected"
  improper == true && return stats

  if verbose > 0
    @info log_header(
      [:outer, :inner, :fx, :hx, :norm_s_cauchydν, :ρ, :Δ, :normx, :norms, :ν, :arrow],
      [Int, Int, T, T, T, T, T, T, T, T, Char],
      hdr_override = Dict{Symbol, String}(
        :fx => "f(x)",
        :hx => "h(x)",
        :norm_s_cauchydν => "‖sᶜᵖ‖/ν",
        :normx => "‖x‖",
        :norms => "‖s‖",
        :arrow => "LMTR",
      ),
      colsep = 1,
    )
  end

  local ξ1::T
  local ρk::T = zero(T)
  local prox_evals::Int = 0

  residual!(nls, xk, Fk)
  jtprod_residual!(nls, xk, Fk, ∇fk)
  fk = dot(Fk, Fk) / 2

  σmax, found_σ = opnorm(solver.subpb.model.J)
  found_σ || error("operator norm computation failed")
  ν = α * Δk / (1 + σmax^2 * (α * Δk + 1))
  @. mν∇fk = -∇fk * ν

  set_iter!(stats, 0)
  start_time = time()
  set_time!(stats, 0.0)
  set_objective!(stats, fk + hk)
  set_solver_specific!(stats, :smooth_obj, fk)
  set_solver_specific!(stats, :nonsmooth_obj, hk)
  set_solver_specific!(stats, :prox_evals, prox_evals + 1)

  mk = let ψ = ψ, solver = solver
    d -> obj(solver.subpb.model, d) + ψ(d)
  end

  # Take first proximal gradient step s1 and see if current xk is nearly stationary.
  # s1 minimizes φ1(d) + ‖d‖² / 2 / ν + ψ(d) ⟺ s1 ∈ prox{νψ}(-ν∇φ1(0))
  prox!(s, ψ, mν∇fk, ν)
  norm_s_cauchy = norm(s)
  norm_s_cauchydν = norm_s_cauchy / ν
  
  atol += rtol * norm_s_cauchydν # make stopping test absolute and relative
  sub_atol += rtol * norm_s_cauchydν

  solved = norm_s_cauchydν ≤ atol

  set_status!(
    stats,
    get_status(
      reg_nls,
      elapsed_time = stats.elapsed_time,
      iter = stats.iter,
      optimal = solved,
      improper = improper,
      max_eval = max_eval,
      max_time = max_time,
      max_iter = max_iter,
    ),
  )

  callback(nls, solver, stats)

  done = stats.status != :unknown

  while !done
    ∆_effective = min(β * χ(s), Δk)

    if has_bnds
      update_bounds!(l_bound_m_x, u_bound_m_x, false, l_bound, u_bound, xk, ∆_effective)
      set_bounds!(ψ, l_bound_m_x, u_bound_m_x)
      set_bounds!(solver.subsolver.ψ, l_bound_m_x, u_bound_m_x)
    else
      set_radius!(solver.subsolver.ψ, ∆_effective)
      set_radius!(ψ, ∆_effective)
    end

    if isa(solver.subsolver, TRDHSolver)
      solver.subsolver.D.d[1] = 1/ν
      solve!(
        solver.subsolver,
        solver.subpb,
        solver.substats;
        x = s,
        atol = stats.iter == 0 ? 1.0e-5 : max(sub_atol, min(1e-2, norm_s_cauchydν)),
        Δk = ∆_effective / 10,
        sub_kwargs...,
      )
    else
      solve!(
        solver.subsolver,
        solver.subpb,
        solver.substats;
        x = s,
        atol = stats.iter == 0 ? 1.0e-5 : max(sub_atol, min(1e-2, norm_s_cauchydν)),
        ν = ν,
        sub_kwargs...,
      )
    end

    prox_evals += solver.substats.iter
    s .= solver.substats.solution

    sNorm = χ(s)
    xkn .= xk .+ s
    residual!(nls, xkn, Fkn)
    fkn = dot(Fkn, Fkn) / 2
    hkn = @views h(xkn[selected])
    mks = mk(s)

    ξ = fk + hk - mks + max(1, abs(hk)) * 10 * eps()
    if (ξ ≤ 0 || isnan(ξ))
      error("LMTR: failed to compute a step: ξ = $ξ")
    end

    Δobj = fk + hk - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
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
          ∆_effective,
          χ(xk),
          sNorm,
          ν,
          (η2 ≤ ρk < Inf) ? '↗' : (ρk < η1 ? '↘' : '='),
        ],
        colsep = 1,
      )

    if η1 ≤ ρk < Inf
      xk .= xkn
      if has_bnds
        update_bounds!(l_bound_m_x, u_bound_m_x, false, l_bound, u_bound, xk, Δk)
        set_bounds!(ψ, l_bound_m_x, u_bound_m_x)
        set_bounds!(solver.subsolver.ψ, l_bound_m_x, u_bound_m_x)
      end

      #update functions
      Fk .= Fkn
      fk = fkn
      hk = hkn

      shift!(ψ, xk)
      jtprod_residual!(nls, xk, Fk, ∇fk)

      σmax, found_σ = opnorm(solver.subpb.model.J)
      found_σ || error("operator norm computation failed")
    end

    if η2 ≤ ρk < Inf
      Δk = max(Δk, γ * sNorm)
      if !has_bnds
        set_radius!(ψ, Δk)
        set_radius!(solver.subsolver.ψ, Δk)
      end
    end

    if ρk < η1 || ρk == Inf
      Δk = Δk / 2
      if has_bnds
        update_bounds!(l_bound_m_x, u_bound_m_x, false, l_bound, u_bound, xk, Δk)
        set_bounds!(ψ, l_bound_m_x, u_bound_m_x)
        set_bounds!(solver.subsolver.ψ, l_bound_m_x, u_bound_m_x)
      else
        set_radius!(solver.subsolver.ψ, Δk)
        set_radius!(ψ, Δk)
      end
    end

    set_objective!(stats, fk + hk)
    set_solver_specific!(stats, :smooth_obj, fk)
    set_solver_specific!(stats, :nonsmooth_obj, hk)
    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)
    set_solver_specific!(stats, :prox_evals, prox_evals + 1)

    ν = α * Δk / (1 + σmax^2 * (α * Δk + 1))
    @. mν∇fk = -∇fk * ν

    prox!(s, ψ, mν∇fk, ν)
    norm_s_cauchy = norm(s)
    norm_s_cauchydν = norm_s_cauchy / ν
  
    solved = norm_s_cauchydν ≤ atol

    set_status!(
      stats,
      get_status(
        reg_nls,
        elapsed_time = stats.elapsed_time,
        iter = stats.iter,
        optimal = solved,
        improper = improper,
        max_eval = max_eval,
        max_time = max_time,
        max_iter = max_iter,
      ),
    )

    callback(nls, solver, stats)

    done = stats.status != :unknown
  end

  if verbose > 0 && stats.status == :first_order
    @info log_row(Any[stats.iter, 0, fk, hk, norm_s_cauchydν, ρk, Δk, χ(xk), χ(s), ν, ""], colsep = 1)
    @info "LMTR: terminating with ‖sᶜᵖ‖/ν = $(norm_s_cauchydν)"
  end

  set_solution!(stats, xk)
  set_residuals!(stats, zero(T), norm_s_cauchydν)
  return stats
end
