export R2DH, R2DH_mine, R2DHSolver, solve!

mutable struct R2DHSolver{
  T <: Real,
  G <: Union{ShiftedProximableFunction, Nothing},
  V <: AbstractVector{T},
} <: AbstractOptimizationSolver
  xk::V
  ∇fk::V
  ∇fk⁻::V
  mν∇fk::V
  ψ::G
  xkn::V
  s::V
  dkσk::V
  has_bnds::Bool
  l_bound::V
  u_bound::V
  l_bound_m_x::V
  u_bound_m_x::V
  m_fh_hist::V
end

function R2DHSolver(reg_nlp::AbstractRegularizedNLPModel{T, V}; Dσ::T = T(1), m_monotone::Int = 1) where{T, V}
  x0 = reg_nlp.model.meta.x0
  l_bound = reg_nlp.model.meta.lvar
  u_bound = reg_nlp.model.meta.uvar

  xk = similar(x0)
  ∇fk = similar(x0)
  ∇fk⁻ = similar(x0)
  mν∇fk = similar(x0)
  xkn = similar(x0)
  s = similar(x0)
  dkσk = similar(x0)
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

  ψ = has_bnds ? shifted(reg_nlp.h, xk, l_bound_m_x, u_bound_m_x, reg_nlp.selected) : shifted(reg_nlp.h, xk)

  return R2DHSolver(
    xk,
    ∇fk,
    ∇fk⁻,
    mν∇fk,
    ψ,
    xkn,
    s,
    dkσk,
    has_bnds,
    l_bound,
    u_bound,
    l_bound_m_x,
    u_bound_m_x,
    m_fh_hist
  )
end
  

"""
    R2DH(nlp, h, options)
    R2DH(f, ∇f!, h, options, x0)

A second-order quadratic regularization method for the problem

    min f(x) + h(x)

where f: ℝⁿ → ℝ is C¹ and h: ℝⁿ → ℝ is lower semi-continuous, proper, and prox-bounded.

About each iterate xₖ, a step sₖ is computed as a solution of

    min  φ(s; xₖ) + ψ(s; xₖ)

where φ(s ; xₖ) = f(xₖ) + ∇f(xₖ)ᵀs + ½ sᵀ(Dₖ + σₖI)s is a quadratic approximation of f about xₖ,
ψ(s; xₖ) = h(xₖ + s), Dₖ is a diagonal Hessian approximation and σₖ > 0 is the regularization parameter.

### Arguments

* `nlp::AbstractDiagonalQNModel`: a smooth optimization problem
* `h`: a regularizer such as those defined in ProximalOperators
* `options::ROSolverOptions`: a structure containing algorithmic parameters
* `x0::AbstractVector`: an initial guess (in the second calling form)

### Keyword Arguments

* `x0::AbstractVector`: an initial guess (in the first calling form: default = `nlp.meta.x0`)
* `selected::AbstractVector{<:Integer}`: subset of variables to which `h` is applied (default `1:length(x0)`).
* `D`: Diagonal quasi-Newton operator.
* `Mmonotone::Int`: number of previous values of the objective to consider for the non-monotone variant (default: 6).

The objective and gradient of `nlp` will be accessed.

In the second form, instead of `nlp`, the user may pass in

* `f` a function such that `f(x)` returns the value of f at x
* `∇f!` a function to evaluate the gradient in place, i.e., such that `∇f!(g, x)` store ∇f(x) in `g`

### Return values

* `xk`: the final iterate
* `Fobj_hist`: an array with the history of values of the smooth objective
* `Hobj_hist`: an array with the history of values of the nonsmooth objective
* `Complex_hist`: an array with the history of number of inner iterations.
"""

function R2DH(
  nlp::AbstractDiagonalQNModel{T, V},
  h,
  options::ROSolverOptions{T};
  kwargs...,
) where{T, V}
  kwargs_dict = Dict(kwargs...)
  selected = pop!(kwargs_dict, :selected, 1:(nlp.meta.nvar))
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  reg_nlp = RegularizedNLPModel(nlp, h, selected)
  return R2DH(
    reg_nlp,
    x = x0,
    atol = options.ϵa,
    rtol = options.ϵr,
    neg_tol = options.neg_tol,
    verbose = options.verbose,
    max_iter = options.maxIter,
    max_time = options.maxTime,
    σmin = options.σmin,
    η1 = options.η1,
    η2 = options.η2,
    ν = options.ν,
    γ = options.γ,
    θ = options.θ,
    kwargs_dict...,
  )
end

function R2DH(
  reg_nlp::AbstractRegularizedNLPModel{T, V};
  kwargs...
) where{T, V}
  kwargs_dict = Dict(kwargs...)
  m_monotone = pop!(kwargs_dict, :m_monotone, 1)
  solver = R2DHSolver(reg_nlp, m_monotone = m_monotone)
  stats = GenericExecutionStats(reg_nlp.model)
  solve!(solver, reg_nlp, stats; kwargs_dict...)
  return stats
end

function solve!(
  solver::R2DHSolver{T},
  reg_nlp::AbstractRegularizedNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = reg_nlp.model.meta.x0,
  D::AbstractDiagonalQuasiNewtonOperator{T} = isa(reg_nlp.model, AbstractDiagonalQNModel) ? hess_op(reg_nlp.model, x) : SpectralGradient(T(1), reg_nlp.model.meta.nvar),
  atol::T = √eps(T),
  rtol::T = √eps(T),
  neg_tol::T = eps(T)^(1 / 4),
  verbose::Int = 0,
  max_iter::Int = 10000,
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  σmin::T = eps(T),
  η1::T = √√eps(T),
  η2::T = T(0.9),
  ν::T = eps(T)^(1 / 5),
  γ::T = T(3),
  θ::T = eps(T)^(1/5),
) where{T, V}
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
  dkσk = solver.dkσk
  ψ = solver.ψ
  xkn = solver.xkn
  s = solver.s
  m_fh_hist = solver.m_fh_hist
  has_bnds = solver.has_bnds

  if has_bnds
    l_bound = solver.l_bound
    u_bound = solver.u_bound
  end
  m_fh_hist = solver.m_fh_hist
  m_monotone = length(m_fh_hist) + 1


  # initialize parameters
  improper = false
  hk = @views h(xk[selected])
  if hk == Inf
    verbose > 0 && @info "R2: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, xk, one(eltype(x0)))
    hk = @views h(xk[selected])
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "R2: found point where h has value" hk
  end
  improper = (hk == -Inf)

  if verbose > 0
    @info log_header(
      [:iter, :fx, :hx, :xi, :ρ, :σ, :normx, :norms, :arrow],
      [Int, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Char],
      hdr_override = Dict{Symbol, String}(   # TODO: Add this as constant dict elsewhere
        :iter => "iter",
        :fx => "f(x)",
        :hx => "h(x)",
        :xi => "√(ξ/ν)",
        :ρ => "ρ",
        :σ => "σ",
        :normx => "‖x‖",
        :norms => "‖s‖",
        :arrow => " ",
      ),
      colsep = 1,
    )
  end

  local ξ::T
  local ρk::T = zero(T)

  σk = max(1 / ν, σmin)

  fk = obj(nlp, xk)
  grad!(nlp, xk, ∇fk)
  ∇fk⁻ .= ∇fk
  spectral_test = isa(D, SpectralGradient)

  @. dkσk = D.d .+ σk
  DNorm = norm(D.d, Inf)

  ν₁ = 1 / ((DNorm + σk) * (1 + θ))
  sqrt_ξ_νInv = one(T)

  @. mν∇fk = -ν₁ * ∇fk
  m_monotone > 1 && (m_fh_hist[mod(stats.iter+1, m_monotone - 1) + 1] = fk + hk)

  set_iter!(stats, 0)
  start_time = time()
  set_time!(stats, 0.0)
  set_objective!(stats, fk + hk)
  set_solver_specific!(stats, :smooth_obj, fk)
  set_solver_specific!(stats, :nonsmooth_obj, hk)

  φ(d) = ∇fk' * d + (d' * (dkσk .* d)) / 2 #TODO This probably allocated a little 
  mk(d) = φ(d) + ψ(d)

  spectral_test ? prox!(s, ψ, mν∇fk, ν₁) : iprox!(s, ψ, ∇fk, dkσk)

  mks = mk(s) 
  while mks == -Inf #TODO add test coverage for this
    σk = σk * γ
    dkσk .= D.d .+ σk
    DNorm = norm(D.d, Inf)
    ν₁ = 1 / ((DNorm + σk) * (1 + θ))
    @. mν∇fk = -ν₁ * ∇fk
    spectral_test ? prox!(s, ψ, mν∇fk, ν₁) : iprox!(s, ψ, ∇fk, dkσk)
  end

  ξ = hk - mks + max(1, abs(hk)) * 10 * eps()
  sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / ν₁) : sqrt(-ξ / ν₁)
  solved = (ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ ≥ 0 && sqrt_ξ_νInv ≤ atol)
  (ξ < 0 && sqrt_ξ_νInv > neg_tol) &&
    error("R2DH: prox-gradient step should produce a decrease but ξ1 = $(ξ1)")
  atol += rtol * sqrt_ξ_νInv # make stopping test absolute and relative

  set_solver_specific!(stats, :xi, sqrt_ξ_νInv)
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
    # Update xk, sigma_k
    xkn .= xk .+ s
    fkn = obj(nlp, xkn)
    hkn = @views h(xkn[selected])
    improper = (hkn == -Inf)

    fhmax = m_monotone > 1 ? maximum(m_fh_hist) : fk + hk
    Δobj = fhmax - (fkn + hkn) + max(1, abs(fhmax)) * 10 * eps()
    Δmod = fhmax - (fk + mks) + max(1, abs(hk)) * 10 * eps()

    ρk = Δobj / Δmod

    verbose > 0 &&
      stats.iter % verbose == 0 &&
      @info log_row(
        Any[
          stats.iter,
          fk,
          hk,
          sqrt_ξ_νInv,
          ρk,
          σk,
          norm(xk),
          norm(s),
          (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "="),
        ],
        colsep = 1,
      )
    if η2 ≤ ρk < Inf
      σk = max(σk / γ, σmin)
    end

    if η1 ≤ ρk < Inf
      xk .= xkn
      if has_bnds
        @. l_bound_m_x = l_bound - xk
        @. u_bound_m_x = u_bound - xk
        set_bounds!(ψ, l_bound_m_x, u_bound_m_x)
      end
      fk = fkn
      hk = hkn
      shift!(ψ, xk)
      grad!(nlp, xk, ∇fk)
      push!(D, s, ∇fk - ∇fk⁻) # update QN operator
      ∇fk⁻ .= ∇fk
    end

    if ρk < η1 || ρk == Inf
      σk = σk * γ
    end

    set_objective!(stats, fk + hk)
    set_solver_specific!(stats, :smooth_obj, fk)
    set_solver_specific!(stats, :nonsmooth_obj, hk)
    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)

    @. dkσk = D.d .+ σk
    DNorm = norm(D.d, Inf)

    ν₁ = 1 / ((DNorm + σk) * (1 + θ))

    @. mν∇fk = -ν₁ * ∇fk
    m_monotone > 1 && (m_fh_hist[mod(stats.iter+1, m_monotone - 1) + 1] = fk + hk)

    spectral_test ? prox!(s, ψ, mν∇fk, ν₁) : iprox!(s, ψ, ∇fk, dkσk)
    mks = mk(s)

    while mks == -Inf  #TODO add test coverage for this
      σk = σk * γ
      dkσk .= D.d .+ σk
      DNorm = norm(D.d, Inf)
      ν₁ = 1 / ((DNorm + σk) * (1 + θ))
      @. mν∇fk = -ν₁ * ∇fk
      spectral_test ? prox!(s, ψ, mν∇fk, ν₁) : iprox!(s, ψ, ∇fk, dkσk)
    end

    ξ = hk - mks + max(1, abs(hk)) * 10 * eps()
    sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / ν₁) : sqrt(-ξ / ν₁)
    solved = (ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ ≥ 0 && sqrt_ξ_νInv ≤ atol)
    (ξ < 0 && sqrt_ξ_νInv > neg_tol) &&
      error("R2DH: prox-gradient step should produce a decrease but ξ1 = $(ξ1)")

    set_solver_specific!(stats, :xi, sqrt_ξ_νInv)
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
        sqrt_ξ_νInv,
        ρk,
        σk,
        norm(xk),
        norm(s),
        (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "="),
      ],
      colsep = 1,
    )
    @info "R2DH: terminating with √(ξ/ν) = $(sqrt_ξ_νInv)"
  end

  set_solution!(stats,xk)
  return stats
end
