export TRDH, TRDHSolver, solve!

import SolverCore.solve!

mutable struct TRDHSolver{
  T <: Real,
  G <: ShiftedProximableFunction,
  V <: AbstractVector{T},
  QN <: AbstractDiagonalQuasiNewtonOperator{T},
  N,
} <: AbstractOptimizationSolver
  xk::V
  ∇fk::V
  ∇fk⁻::V
  mν∇fk::V
  D::QN
  ψ::G
  χ::N
  xkn::V
  s::V
  dk::V
  has_bnds::Bool
  l_bound::V
  u_bound::V
  l_bound_m_x::V
  u_bound_m_x::V
end

function TRDHSolver(
  reg_nlp::AbstractRegularizedNLPModel{T, V};
  D::Union{Nothing, AbstractDiagonalQuasiNewtonOperator} = nothing,
  χ = NormLinf(one(T)),
) where {T, V}
  x0 = reg_nlp.model.meta.x0
  l_bound = reg_nlp.model.meta.lvar
  u_bound = reg_nlp.model.meta.uvar
  l_bound_k = similar(l_bound)
  u_bound_k = similar(u_bound)

  xk = similar(x0)
  ∇fk = similar(x0)
  ∇fk⁻ = similar(x0)
  mν∇fk = similar(x0)
  xkn = similar(x0)
  s = similar(x0)
  dk = similar(x0)
  has_bnds = any(l_bound .!= T(-Inf)) || any(u_bound .!= T(Inf))

  is_subsolver = reg_nlp.h isa ShiftedProximableFunction # case TRDH is used as a subsolver
  if is_subsolver
    ψ = shifted(reg_nlp.h, xk)
    @assert !has_bnds
    l_bound = copy(ψ.l)
    u_bound = copy(ψ.u)
    @. l_bound_k = max(x0 - one(T), l_bound)
    @. u_bound_k = min(x0 + one(T), u_bound)
    has_bnds = true
    set_bounds!(ψ, l_bound_k, u_bound_k)
  else
    if has_bnds
      @. l_bound_k = max(-one(T), l_bound - x0)
      @. u_bound_k = min(one(T), u_bound - x0)
      ψ = shifted(reg_nlp.h, xk, l_bound_k, u_bound_k, reg_nlp.selected)
    else
      ψ = shifted(reg_nlp.h, xk, one(T), χ)
    end
  end
  isnothing(D) && (
    D =
      isa(reg_nlp.model, AbstractDiagonalQNModel) ? hess_op(reg_nlp.model, x0) :
      SpectralGradient(T(1), reg_nlp.model.meta.nvar)
  )

  return TRDHSolver(
    xk,
    ∇fk,
    ∇fk⁻,
    mν∇fk,
    D,
    ψ,
    χ,
    xkn,
    s,
    dk,
    has_bnds,
    l_bound,
    u_bound,
    l_bound_k,
    u_bound_k,
  )
end

"""
    TRDH(reg_nlp; kwargs…)
    TRDH(nlp, h, χ, options; kwargs...)
    TRDH(f, ∇f!, h, options, x0)

A trust-region method with diagonal Hessian approximation for the problem

    min f(x) + h(x)

where f: ℝⁿ → ℝ has a Lipschitz-continuous gradient,, and h: ℝⁿ → ℝ is
lower semi-continuous, proper and prox-bounded.

About each iterate xₖ, a step sₖ is computed as an approximate solution of

    min  φ(s; xₖ) + ψ(s; xₖ)  subject to  ‖s‖ ≤ Δₖ

where φ(s ; xₖ) = f(xₖ) + ∇f(xₖ)ᵀs + ½ sᵀ Dₖ s  is a quadratic approximation of f about xₖ,
ψ(s; xₖ) = h(xₖ + s), ‖⋅‖ is a user-defined norm, Dₖ is a diagonal Hessian approximation
and Δₖ > 0 is the trust-region radius.

For advanced usage, first define a solver "TRDHSolver" to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = TRDH(reg_nlp; D = nothing, χ =  NormLinf(1))
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
- `Δk::T = T(1)`: initial value of the trust-region radius;
- `η1::T = √√eps(T)`: successful iteration threshold;
- `η2::T = T(0.9)`: very successful iteration threshold;
- `γ::T = T(3)`: trust-region radius parameter multiplier. Must satisfy `γ > 1`. The trust-region radius is updated as Δ := Δ*γ when the iteration is very successful and Δ := Δ/γ when the iteration is unsuccessful;
- `reduce_TR::Bool = true`: see explanation on the stopping criterion below;
- `χ::F =  NormLinf(1)`: norm used to define the trust-region;`
- `D::L = nothing`: diagonal quasi-Newton approximation used for the model φ. If nothing is provided and `reg_nlp.model` is not a diagonal quasi-Newton approximation, a spectral gradient approximation is used.`

The algorithm stops either when `√(ξₖ/νₖ) < atol + rtol*√(ξ₀/ν₀) ` or `ξₖ < 0` and `√(-ξₖ/νₖ) < neg_tol` where ξₖ := f(xₖ) + h(xₖ) - φ(sₖ; xₖ) - ψ(sₖ; xₖ), and √(ξₖ/νₖ) is a stationarity measure.
Alternatively, if `reduce_TR = true`, then ξₖ₁ := f(xₖ) + h(xₖ) - φ(sₖ₁; xₖ) - ψ(sₖ₁; xₖ) is used instead of ξₖ, where sₖ₁ is the Cauchy point.

#  Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
$(callback_docstring)
"""
function TRDH(
  nlp::AbstractDiagonalQNModel{T, V},
  h,
  χ,
  options::ROSolverOptions{T};
  kwargs...,
) where {T, V}
  kwargs_dict = Dict(kwargs...)
  selected = pop!(kwargs_dict, :selected, 1:(nlp.meta.nvar))
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  reg_nlp = RegularizedNLPModel(nlp, h, selected)
  stats = TRDH(
    reg_nlp;
    x = x0,
    D = nlp.op,
    χ = χ,
    atol = options.ϵa,
    rtol = options.ϵr,
    neg_tol = options.neg_tol,
    verbose = options.verbose,
    max_iter = options.maxIter,
    max_time = options.maxTime,
    reduce_TR = options.reduce_TR,
    Δk = options.Δk,
    η1 = options.η1,
    η2 = options.η2,
    γ = options.γ,
    kwargs_dict...,
  )
  return stats
end

function TRDH(
  f::F,
  ∇f!::G,
  h::H,
  D::DQN,
  options::ROSolverOptions{R},
  x0::AbstractVector{R};
  χ::X = NormLinf(one(R)),
  selected::AbstractVector{<:Integer} = 1:length(x0),
  kwargs...,
) where {R <: Real, F, G, H, DQN <: AbstractDiagonalQuasiNewtonOperator, X}
  nlp = NLPModel(x0, f, grad = ∇f!)
  reg_nlp = RegularizedNLPModel(nlp, h, selected)
  stats = TRDH(
    reg_nlp;
    x = x0,
    D = D,
    χ = χ,
    atol = options.ϵa,
    rtol = options.ϵr,
    neg_tol = options.neg_tol,
    verbose = options.verbose,
    max_iter = options.maxIter,
    max_time = options.maxTime,
    reduce_TR = options.reduce_TR,
    Δk = options.Δk,
    η1 = options.η1,
    η2 = options.η2,
    γ = options.γ,
    kwargs...,
  )
  return stats.solution, stats.iter, stats
end

function TRDH(reg_nlp::AbstractRegularizedNLPModel{T, V}; kwargs...) where {T, V}
  kwargs_dict = Dict(kwargs...)
  D = pop!(kwargs_dict, :D, nothing)
  χ = pop!(kwargs_dict, :χ, NormLinf(one(T)))
  solver = TRDHSolver(reg_nlp, D = D, χ = χ)
  stats = RegularizedExecutionStats(reg_nlp)
  solve!(solver, reg_nlp, stats; kwargs_dict...)
  return stats
end

function SolverCore.solve!(
  solver::TRDHSolver{T, G, V},
  reg_nlp::AbstractRegularizedNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = reg_nlp.model.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  neg_tol::T = eps(T)^(1 / 4),
  verbose::Int = 0,
  max_iter::Int = 10000,
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  reduce_TR::Bool = true,
  Δk::T = T(1),
  η1::T = √√eps(T),
  η2::T = T(0.9),
  γ::T = T(3),
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
  D = solver.D
  dk = solver.dk
  ψ = solver.ψ
  xkn = solver.xkn
  s = solver.s
  χ = solver.χ
  has_bnds = solver.has_bnds

  if has_bnds
    l_bound_m_x = solver.l_bound_m_x
    u_bound_m_x = solver.u_bound_m_x
    l_bound = solver.l_bound
    u_bound = solver.u_bound
  end

  # initialize parameters
  improper = false
  hk = @views h(xk[selected])
  if hk == Inf
    verbose > 0 && @info "TRDH: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, xk, T(1))
    hk = @views h(xk[selected])
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "TRDH: found point where h has value" hk
  end
  improper = (hk == -Inf)
  improper == true && @warn "TRDH: Improper term detected"
  improper == true && return stats

  is_subsolver = h isa ShiftedProximableFunction # case TRDH is used as a subsolver

  if is_subsolver
    l_bound .= ψ.l
    u_bound .= ψ.u
  end

  if verbose > 0
    @info log_header(
      [:iter, :fx, :hx, :xi, :ρ, :Δ, :normx, :norms, :normD, :arrow],
      [Int, T, T, T, T, T, T, T, T, Char],
      hdr_override = Dict{Symbol, String}(   # TODO: Add this as constant dict elsewhere
        :fx => "f(x)",
        :hx => "h(x)",
        :xi => "√(ξ/ν)",
        :normx => "‖x‖",
        :norms => "‖s‖",
        :normD => "‖D‖",
        :arrow => "TRDH",
      ),
      colsep = 1,
    )
  end

  local ξ1::T
  local ρk::T = zero(T)

  α = 1 / eps(T)
  β = 1 / eps(T)

  fk = obj(nlp, xk)
  grad!(nlp, xk, ∇fk)
  ∇fk⁻ .= ∇fk

  dk .= D.d
  DNorm = norm(D.d, Inf)

  ν = (α * Δk)/(DNorm + one(T))
  sqrt_ξ_νInv = one(T)

  @. mν∇fk = -ν * ∇fk

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
  mk1 = let ψ = ψ
    d -> φ1(d) + ψ(d)::T
  end

  φ = let ∇fk = ∇fk, dk = dk
    d -> begin
      result = zero(T)
      n = length(d)
      @inbounds for i = 1:n
        result += d[i]^2 * dk[i] / 2 + ∇fk[i] * d[i]
      end
      result
    end
  end
  mk = let ψ = ψ
    d -> φ(d) + ψ(d)::T
  end

  if reduce_TR
    prox!(s, ψ, mν∇fk, ν)
    mks = mk1(s)

    ξ1 = hk - mks + max(1, abs(hk)) * 10 * eps()
    sqrt_ξ_νInv = ξ1 ≥ 0 ? sqrt(ξ1 / ν) : sqrt(-ξ1 / ν)
    solved = (ξ1 < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ1 ≥ 0 && sqrt_ξ_νInv ≤ atol)
    (ξ1 < 0 && sqrt_ξ_νInv > neg_tol) &&
      error("TR: prox-gradient step should produce a decrease but ξ = $(ξ)")
    atol += rtol * sqrt_ξ_νInv # make stopping test absolute and relative
  end

  Δ_effective = reduce_TR ? min(β * χ(s), Δk) : Δk

  # update radius
  if has_bnds
    update_bounds!(l_bound_m_x, u_bound_m_x, is_subsolver, l_bound, u_bound, xk, Δ_effective)
    set_bounds!(ψ, l_bound_m_x, u_bound_m_x)
  else
    set_radius!(ψ, Δ_effective)
  end

  iprox!(s, ψ, ∇fk, dk)
  ξ = hk - mk(s) + max(1, abs(hk)) * 10 * eps()
  sNorm = χ(s)

  if !reduce_TR
    sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / ν) : sqrt(-ξ / ν)
    solved = (ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ ≥ 0 && sqrt_ξ_νInv < atol)
    (ξ < 0 && sqrt_ξ_νInv > neg_tol) &&
      error("TRDH: prox-gradient step should produce a decrease but ξ = $(ξ)")
    atol += rtol * sqrt_ξ_νInv # make stopping test absolute and relative #TODO : this is redundant code with the other case of the test.
  end

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
    xkn .= xk .+ s
    fkn = obj(nlp, xkn)
    hkn = @views h(xkn[selected])

    Δobj = fk + hk - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ρk = Δobj / ξ

    verbose > 0 &&
      stats.iter % verbose == 0 &&
      @info log_row(
        Any[
          stats.iter,
          fk,
          hk,
          sqrt_ξ_νInv,
          ρk,
          Δk,
          χ(xk),
          sNorm,
          norm(D.d),
          (η2 ≤ ρk < Inf) ? "↗" : (ρk < η1 ? "↘" : "="),
        ],
        colsep = 1,
      )

    if η1 ≤ ρk < Inf
      xk .= xkn
      if has_bnds
        update_bounds!(l_bound_m_x, u_bound_m_x, is_subsolver, l_bound, u_bound, xk, Δk)
        has_bnds && set_bounds!(ψ, l_bound_m_x, u_bound_m_x)
      end
      fk = fkn
      hk = hkn
      shift!(ψ, xk)
      grad!(nlp, xk, ∇fk)
      @. ∇fk⁻ = ∇fk - ∇fk⁻
      push!(D, s, ∇fk⁻) # update QN operator
      dk .= D.d
      DNorm = norm(D.d, Inf)
      ∇fk⁻ .= ∇fk
    end

    if η2 ≤ ρk < Inf
      Δk = max(Δk, γ * sNorm)
      !has_bnds && set_radius!(ψ, Δk)
    end

    if ρk < η1 || ρk == Inf
      Δk = Δk / 2
      if has_bnds
        update_bounds!(l_bound_m_x, u_bound_m_x, is_subsolver, l_bound, u_bound, xk, Δ_effective)
        set_bounds!(ψ, l_bound_m_x, u_bound_m_x)
      else
        set_radius!(ψ, Δk)
      end
    end

    set_objective!(stats, fk + hk)
    set_solver_specific!(stats, :smooth_obj, fk)
    set_solver_specific!(stats, :nonsmooth_obj, hk)
    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)

    ν = reduce_TR ? (α * Δk)/(DNorm + one(T)) : α / (DNorm + one(T))
    mν∇fk .= -ν .* ∇fk

    if reduce_TR
      prox!(s, ψ, mν∇fk, ν)
      ξ1 = hk - mk1(s) + max(1, abs(hk)) * 10 * eps()
      sqrt_ξ_νInv = ξ1 ≥ 0 ? sqrt(ξ1 / ν) : sqrt(-ξ1 / ν)
      solved = (ξ1 < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ1 ≥ 0 && sqrt_ξ_νInv < atol)
      (ξ1 < 0 && sqrt_ξ_νInv > neg_tol) &&
        error("TRDH: prox-gradient step should produce a decrease but ξ = $(ξ)")
    end

    iprox!(s, ψ, ∇fk, dk)

    sNorm = χ(s)

    if !reduce_TR
      ξ = hk - mk(s) + max(1, abs(hk)) * 10 * eps()
      sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / ν) : sqrt(-ξ / ν)
      solved = (ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ ≥ 0 && sqrt_ξ_νInv < atol)
      (ξ < 0 && sqrt_ξ_νInv > neg_tol) &&
        error("TRDH: prox-gradient step should produce a decrease but ξ = $(ξ)")
    end

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
      Any[stats.iter, fk, hk, sqrt_ξ_νInv, ρk, Δk, χ(xk), sNorm, norm(D.d), ""],
      colsep = 1,
    )
    @info "TRDH: terminating with √(ξ/ν) = $(sqrt_ξ_νInv)"
  end

  set_solution!(stats, xk)

  return stats
end
