export TRDH, TRDHSolver, solve!

import SolverCore.solve!

mutable struct TRDHSolver{
  T <: Real,
  G <: ShiftedProximableFunction,
  V <: AbstractVector{T},
  QN <: AbstractDiagonalQuasiNewtonOperator{T},
  N
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
  χ = NormLinf(one(T))
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
    @. l_bound_k = max(xk - one(T), l_bound)
    @. u_bound_k = min(xk + one(T), u_bound)
    has_bnds = true
    set_bounds!(ψ, l_bound_k, u_bound_k)
  else
    if has_bnds
      @. l_bound_k = max(-one(T), l_bound - xk)
      @. u_bound_k = min(one(T), u_bound - xk)
      ψ = shifted(reg_nlp.h, xk, l_bound_k, u_bound_k, selected)
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

function SolverCore.solve!(
  solver::TRDHSolver{T, G, V},
  reg_nlp::AbstractRegularizedNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = reg_nlp.model.meta.x0,
  #χ = NormLinf(one(T)),
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
  α::T = 1 / eps(T),
  β::T = 1 / eps(T)
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
        set_radius!(ψ, Δ_effective)
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
          "",
        ],
        colsep = 1,
      )
    @info "TRDH: terminating with √(ξ/ν) = $(sqrt_ξ_νInv)"
  end
end

"""
    TRDH(nlp, h, χ, options; kwargs...)
    TRDH(f, ∇f!, h, options, x0)

A trust-region method with diagonal Hessian approximation for the problem

    min f(x) + h(x)

where f: ℝⁿ → ℝ has a Lipschitz-continuous Jacobian, and h: ℝⁿ → ℝ is
lower semi-continuous and proper.

About each iterate xₖ, a step sₖ is computed as an approximate solution of

    min  φ(s; xₖ) + ψ(s; xₖ)  subject to  ‖s‖ ≤ Δₖ

where φ(s ; xₖ) = f(xₖ) + ∇f(xₖ)ᵀs + ½ sᵀ Dₖ s  is a quadratic approximation of f about xₖ,
ψ(s; xₖ) = h(xₖ + s), ‖⋅‖ is a user-defined norm, Dₖ is a diagonal Hessian approximation
and Δₖ > 0 is the trust-region radius.

### Arguments

* `nlp::AbstractDiagonalQNModel`: a smooth optimization problem
* `h`: a regularizer such as those defined in ProximalOperators
* `χ`: a norm used to define the trust region in the form of a regularizer
* `options::ROSolverOptions`: a structure containing algorithmic parameters

The objective and gradient of `nlp` will be accessed.

In the second form, instead of `nlp`, the user may pass in

* `f` a function such that `f(x)` returns the value of f at x
* `∇f!` a function to evaluate the gradient in place, i.e., such that `∇f!(g, x)` store ∇f(x) in `g`
* `x0::AbstractVector`: an initial guess.

### Keyword arguments

* `x0::AbstractVector`: an initial guess (default: `nlp.meta.x0`)
* `selected::AbstractVector{<:Integer}`: (default `1:f.meta.nvar`)

### Return values

* `xk`: the final iterate
* `Fobj_hist`: an array with the history of values of the smooth objective
* `Hobj_hist`: an array with the history of values of the nonsmooth objective
* `Complex_hist`: an array with the history of number of inner iterations.
"""
function TRDH(
  nlp::AbstractDiagonalQNModel{R, S},
  h,
  χ,
  options::ROSolverOptions{R};
  kwargs...,
) where {R <: Real, S}
  kwargs_dict = Dict(kwargs...)
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  xk, k, outdict = TRDH(
    x -> obj(nlp, x),
    (g, x) -> grad!(nlp, x, g),
    h,
    hess_op(nlp, x0),
    options,
    x0;
    χ = χ,
    l_bound = nlp.meta.lvar,
    u_bound = nlp.meta.uvar,
    kwargs...,
  )
  sqrt_ξ_νInv = outdict[:sqrt_ξ_νInv]
  stats = GenericExecutionStats(nlp)
  set_status!(stats, outdict[:status])
  set_solution!(stats, xk)
  set_objective!(stats, outdict[:fk] + outdict[:hk])
  set_residuals!(stats, zero(eltype(xk)), sqrt_ξ_νInv)
  set_iter!(stats, k)
  set_time!(stats, outdict[:elapsed_time])
  set_solver_specific!(stats, :radius, outdict[:radius])
  set_solver_specific!(stats, :Fhist, outdict[:Fhist])
  set_solver_specific!(stats, :Hhist, outdict[:Hhist])
  set_solver_specific!(stats, :NonSmooth, outdict[:NonSmooth])
  set_solver_specific!(stats, :SubsolverCounter, outdict[:Chist])
  return stats
end

# update l_bound_k and u_bound_k
function update_bounds!(l_bound_k, u_bound_k, is_subsolver, l_bound, u_bound, xk, Δ)
  if is_subsolver
    @. l_bound_k = max(xk - Δ, l_bound)
    @. u_bound_k = min(xk + Δ, u_bound)
  else
    @. l_bound_k = max(-Δ, l_bound - xk)
    @. u_bound_k = min(Δ, u_bound - xk)
  end
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
  start_time = time()
  elapsed_time = 0.0
  ϵ = options.ϵa
  ϵr = options.ϵr
  Δk = options.Δk
  neg_tol = options.neg_tol
  verbose = options.verbose
  maxIter = options.maxIter
  maxTime = options.maxTime
  η1 = options.η1
  η2 = options.η2
  γ = options.γ
  α = options.α
  β = options.β
  reduce_TR = options.reduce_TR

  local l_bound, u_bound
  has_bnds = false
  kw_keys = keys(kwargs)
  if :l_bound in kw_keys
    l_bound = kwargs[:l_bound]
    has_bnds = has_bnds || any(l_bound .!= R(-Inf))
  end
  if :u_bound in kw_keys
    u_bound = kwargs[:u_bound]
    has_bnds = has_bnds || any(u_bound .!= R(Inf))
  end

  if verbose == 0
    ptf = Inf
  elseif verbose == 1
    ptf = round(maxIter / 10)
  elseif verbose == 2
    ptf = round(maxIter / 100)
  else
    ptf = 1
  end

  # initialize parameters
  xk = copy(x0)
  hk = h(xk[selected])
  if hk == Inf
    verbose > 0 && @info "TRDH: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, x0, one(eltype(x0)))
    hk = h(xk[selected])
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "TRDH: found point where h has value" hk
  end
  hk == -Inf && error("nonsmooth term is not proper")

  xkn = similar(xk)
  s = zero(xk)
  l_bound_k = similar(xk)
  u_bound_k = similar(xk)
  is_subsolver = h isa ShiftedProximableFunction # case TRDH is used as a subsolver
  if is_subsolver
    ψ = shifted(h, xk)
    @assert !has_bnds
    l_bound = copy(ψ.l)
    u_bound = copy(ψ.u)
    @. l_bound_k = max(xk - Δk, l_bound)
    @. u_bound_k = min(xk + Δk, u_bound)
    has_bnds = true
    set_bounds!(ψ, l_bound_k, u_bound_k)
  else
    if has_bnds
      @. l_bound_k = max(-Δk, l_bound - xk)
      @. u_bound_k = min(Δk, u_bound - xk)
      ψ = shifted(h, xk, l_bound_k, u_bound_k, selected)
    else
      ψ = shifted(h, xk, Δk, χ)
    end
  end

  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, maxIter)
  if verbose > 0
    #! format: off
    if reduce_TR
      @info @sprintf "%6s %8s %8s %7s %7s %8s %7s %7s %7s %7s %1s" "outer" "f(x)" "h(x)" "√(ξ1/ν)" "√ξ" "ρ" "Δ" "‖x‖" "‖s‖" "‖Dₖ‖" "TRDH"
    else
      @info @sprintf "%6s %8s %8s %7s %8s %7s %7s %7s %7s %1s" "outer" "f(x)" "h(x)" "√(ξ/ν)" "ρ" "Δ" "‖x‖" "‖s‖" "‖Dₖ‖" "TRDH"
    end
    #! format: off
  end

  local ξ1
  local ξ
  k = 0

  fk = f(xk)
  ∇fk = similar(xk)
  ∇f!(∇fk, xk)
  ∇fk⁻ = copy(∇fk)
  DNorm = norm(D.d, Inf)
  ν = (α * Δk)/(DNorm + one(R))
  mν∇fk = -ν .* ∇fk
  sqrt_ξ_νInv = one(R)

  optimal = false
  tired = maxIter > 0 && k ≥ maxIter || elapsed_time > maxTime

  while !(optimal || tired)
    k = k + 1
    elapsed_time = time() - start_time
    Fobj_hist[k] = fk
    Hobj_hist[k] = hk

    # model for prox-gradient step to update Δk if ||s|| is too small and ξ1
    φ1(d) = ∇fk' * d
    mk1(d) = φ1(d) + ψ(d)

    if reduce_TR
      prox!(s, ψ, mν∇fk, ν)
      Complex_hist[k] += 1
      ξ1 = hk - mk1(s) + max(1, abs(hk)) * 10 * eps()
      sqrt_ξ_νInv = ξ1 ≥ 0 ? sqrt(ξ1 / ν) : sqrt(-ξ1 / ν)

      if ξ1 ≥ 0 && k == 1
        ϵ += ϵr * sqrt_ξ_νInv  # make stopping test absolute and relative
      end

      if (ξ1 < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ1 ≥ 0 && sqrt_ξ_νInv < ϵ)
        # the current xk is approximately first-order stationary
        optimal = true
        continue
      end

      ξ1 > 0 || error("TR: first prox-gradient step should produce a decrease but ξ1 = $(ξ1)")
    end

    Δ_effective = reduce_TR ? min(β * χ(s), Δk) : Δk
    # update radius
    if has_bnds
      update_bounds!(l_bound_k, u_bound_k, is_subsolver, l_bound, u_bound, xk, Δ_effective)
      set_bounds!(ψ, l_bound_k, u_bound_k)
    else
      set_radius!(ψ, Δ_effective)
    end

    # model with diagonal hessian
    φ(d) = ∇fk' * d + (d' * (D.d .* d)) / 2
    mk(d) = φ(d) + ψ(d)

    iprox!(s, ψ, ∇fk, D)

    sNorm = χ(s)
    xkn .= xk .+ s
    fkn = f(xkn)
    hkn = h(xkn[selected])
    hkn == -Inf && error("nonsmooth term is not proper")

    Δobj = fk + hk - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ξ = hk - mk(s) + max(1, abs(hk)) * 10 * eps()

    if !reduce_TR

      sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / ν) : sqrt(-ξ / ν)
      if ξ ≥ 0 && k == 1
        ϵ += ϵr * sqrt_ξ_νInv  # make stopping test absolute and relative
      end

      if (ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ ≥ 0 && sqrt_ξ_νInv < ϵ)
        # the current xk is approximately first-order stationary
        optimal = true
        continue
      end
    end

    if (ξ ≤ 0 || isnan(ξ))
      error("TRDH: failed to compute a step: ξ = $ξ")
    end

    ρk = Δobj / ξ

    TR_stat = (η2 ≤ ρk < Inf) ? "↗" : (ρk < η1 ? "↘" : "=")

    if (verbose > 0) && (k % ptf == 0)
      #! format: off
      if reduce_TR
        @info @sprintf "%6d %8.1e %8.1e %7.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s" k fk hk sqrt_ξ_νInv sqrt(ξ) ρk Δk χ(xk) sNorm norm(D.d) TR_stat
      else
        @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s" k fk hk sqrt_ξ_νInv ρk Δk χ(xk) sNorm norm(D.d) TR_stat
      end
      #! format: on
    end

    if η2 ≤ ρk < Inf
      Δk = max(Δk, γ * sNorm)
      !has_bnds && set_radius!(ψ, Δk)
    end

    if η1 ≤ ρk < Inf
      xk .= xkn
      has_bnds && update_bounds!(l_bound_k, u_bound_k, is_subsolver, l_bound, u_bound, xk, Δk)
      has_bnds && set_bounds!(ψ, l_bound_k, u_bound_k)
      #update functions
      fk = fkn
      hk = hkn
      shift!(ψ, xk)
      ∇f!(∇fk, xk)
      push!(D, s, ∇fk - ∇fk⁻) # update QN operator
      DNorm = norm(D.d, Inf)
      ∇fk⁻ .= ∇fk
    end

    if ρk < η1 || ρk == Inf
      Δk = Δk / 2
      has_bnds && update_bounds!(l_bound_k, u_bound_k, is_subsolver, l_bound, u_bound, xk, Δk)
      has_bnds ? set_bounds!(ψ, l_bound_k, u_bound_k) : set_radius!(ψ, Δk)
    end

    ν = reduce_TR ? (α * Δk)/(DNorm + one(R)) : α / (DNorm + one(R))
    mν∇fk .= -ν .* ∇fk

    tired = k ≥ maxIter || elapsed_time > maxTime
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8s %8.1e %8.1e" k "" fk hk
    elseif optimal
      #! format: off
      if reduce_TR
        @info @sprintf "%6d %8.1e %8.1e %7.1e %7.1e %8s %7.1e %7.1e %7.1e %7.1e" k fk hk sqrt_ξ_νInv sqrt(ξ1) "" Δk χ(xk) χ(s) norm(D.d)
        #! format: on
        @info "TRDH: terminating with √(ξ1/ν) = $(sqrt_ξ_νInv)"
      else
        #! format: off
        @info @sprintf "%6d %8.1e %8.1e %7.1e %8s %7.1e %7.1e %7.1e %7.1e" k fk hk sqrt_ξ_νInv "" Δk χ(xk) χ(s) norm(D.d)
        #! format: on
        @info "TRDH: terminating with √(ξ/ν) = $(sqrt_ξ_νInv)"
      end
    end
  end

  !reduce_TR && (ξ1 = ξ) # for output dict

  status = if optimal
    :first_order
  elseif elapsed_time > maxTime
    :max_time
  elseif tired
    :max_iter
  else
    :exception
  end
  outdict = Dict(
    :Fhist => Fobj_hist[1:k],
    :Hhist => Hobj_hist[1:k],
    :Chist => Complex_hist[1:k],
    :NonSmooth => h,
    :status => status,
    :fk => fk,
    :hk => hk,
    :sqrt_ξ_νInv => sqrt_ξ_νInv,
    :elapsed_time => elapsed_time,
    :radius => Δk,
  )

  return xk, k, outdict
end
