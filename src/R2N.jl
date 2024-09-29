export R2N, R2NSolver, solve!

import SolverCore.solve!

mutable struct R2NSolver{
  T <: Real,
  G <: ShiftedProximableFunction,
  V <: AbstractVector{T},
  S <: AbstractOptimizationSolver,
} <: AbstractOptimizationSolver
  xk::V
  ∇fk::V
  ∇fk⁻::V
  mν∇fk::V
  ψ::G
  sub_ψ::G
  xkn::V
  s::V
  s1::V
  has_bnds::Bool
  l_bound::V
  u_bound::V
  sub_solver::S
	sub_stats::GenericExecutionStats{T, V, V, Any}
end

function R2NSolver(reg_nlp::AbstractRegularizedNLPModel{T, V}; sub_solver = R2Solver) where {T, V}
  x0 = reg_nlp.model.meta.x0
  l_bound = reg_nlp.model.meta.lvar
  u_bound = reg_nlp.model.meta.uvar

  xk = similar(x0)
  ∇fk = similar(x0)
  ∇fk⁻ = similar(x0)
  mν∇fk = similar(x0)
  xkn = similar(x0)
  s = zero(x0)
  s1 = similar(x0)
  has_bnds = any(l_bound .!= T(-Inf)) || any(u_bound .!= T(Inf))

  ψ = has_bnds ? shifted(reg_nlp.h, xk, l_bound_m_x, u_bound_m_x, reg_nlp.selected) : shifted(reg_nlp.h, xk)
  sub_ψ = has_bnds ? shifted(reg_nlp.h, xk, l_bound_m_x, u_bound_m_x, reg_nlp.selected) : shifted(reg_nlp.h, xk)

	sub_nlp = RegularizedNLPModel(reg_nlp.model, sub_ψ)
	sub_stats = GenericExecutionStats(reg_nlp.model)

  return R2NSolver(
    xk,
    ∇fk,
    ∇fk⁻,
    mν∇fk,
    ψ,
    sub_ψ,
    xkn,
    s,
    s1,
    has_bnds,
    l_bound,
    u_bound,
    sub_solver(sub_nlp),
    sub_stats
  )
end

function SolverCore.solve!(
  solver::R2NSolver{T},
  reg_nlp::AbstractRegularizedNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = reg_nlp.model.meta.x0,
  atol::T = √eps(T),
  sub_atol::T = atol,
  rtol::T = √eps(T),
  neg_tol::T = eps(T)^(1 / 4),
  verbose::Int = 0,
  max_iter::Int = 10000,
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  σmin::T = eps(T),
  η1::T = √√eps(T),
  η2::T = T(0.9),
  ν = eps(T)^(1/5),
  γ::T = T(3),
  β = 1/eps(T),
  θ = eps(T)^(1/5),
  kwargs...
) where {T, V}

  reset!(stats)

  # Retrieve workspace
  selected = reg_nlp.selected
  h = reg_nlp.h
  nlp = reg_nlp.model

  xk = solver.xk .= x

  # Make sure ψ has the correct shift 
  shift!(solver.ψ, xk)

  σk = 1/ν
  ∇fk = solver.∇fk
  ∇fk⁻ = solver.∇fk⁻
  mν∇fk = solver.mν∇fk
  ψ = solver.ψ
  xkn = solver.xkn
  s = solver.s
  s1 = solver.s1
  has_bnds = solver.has_bnds
  if has_bnds
    l_bound = solver.l_bound
    u_bound = solver.u_bound
  end
  sub_atol_init = copy(sub_atol)
  sub_solver = solver.sub_solver
  sub_stats = solver.sub_stats
  sub_ψ = solver.sub_ψ

  # initialize parameters
  improper = false
  hk = @views h(xk[selected])
  if hk == Inf
    verbose > 0 && @info "R2N: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, xk, one(eltype(x0)))
    hk = @views h(xk[selected])
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "R2N: found point where h has value" hk
  end
  improper = (hk == -Inf)

  if verbose > 0
    @info log_header(
      [:outer, :inner, :fx, :hx, :xi, :ρ, :σ, :normx, :norms, :normB, :arrow],
      [Int, Int, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Char],
      hdr_override = Dict{Symbol, String}(   # TODO: Add this as constant dict elsewhere
        :outer => "outer",
        :inner => "inner",
        :fx => "f(x)",
        :hx => "h(x)",
        :xi => "√(ξ1/ν)",
        :ρ => "ρ",
        :σ => "σ",
        :normx => "‖x‖",
        :norms => "‖s‖",
        :normB => "‖B‖",
        :arrow => "R2N",
      ),
      colsep = 1,
    )
  end

  local ξ1::T
  local ρk::T

  fk = obj(nlp, xk)
  grad!(nlp, xk, ∇fk)
  ∇fk⁻ .= ∇fk

  quasiNewtTest = isa(nlp, QuasiNewtonModel)
  Bk = hess_op(nlp, xk)
  local λmax::T
  try
    λmax = opnorm(Bk)
  catch LAPACKException
    λmax = opnorm(Matrix(Bk))
  end

  νInv = (1 + θ) *( σk + λmax)
  sqrt_ξ1_νInv = one(T)
  ν_subsolver = 1/νInv

  @. mν∇fk = -ν_subsolver * ∇fk

  set_iter!(stats, 0)
  start_time = time()
  set_time!(stats, 0.0)
  set_objective!(stats, fk + hk)
  set_solver_specific!(stats, :smooth_obj, fk)
  set_solver_specific!(stats, :nonsmooth_obj, hk)

  # model for first prox-gradient step and ξ1
  φ1(d) = ∇fk' * d
  mk1(d) = φ1(d) + ψ(d)

  # model for subsequent prox-gradient steps and ξ
  φ(d) = (d' * (Bk * d)) / 2 + ∇fk' * d + σk * dot(d, d) / 2

  ∇φ!(g, d) = begin
    mul!(g, Bk, d)
    g .+= ∇fk
    g .+= σk * d
    g
  end

  mk(d) = φ(d) + ψ(d)

  prox!(s, ψ, mν∇fk, ν_subsolver)
  mks = mk1(s)

  ξ1 = hk - mks + max(1, abs(hk)) * 10 * eps()

  sqrt_ξ1_νInv = ξ1 ≥ 0 ? sqrt(ξ1 / ν_subsolver) : sqrt(-ξ1 / ν_subsolver)
  solved = (ξ1 < 0 && sqrt_ξ1_νInv ≤ neg_tol) || (ξ1 ≥ 0 && sqrt_ξ1_νInv ≤ atol)
  (ξ1 < 0 && sqrt_ξ1_νInv > neg_tol) &&
    error("R2N: prox-gradient step should produce a decrease but ξ1 = $(ξ1)")
  atol += rtol * sqrt_ξ1_νInv # make stopping test absolute and relative
  sub_atol += rtol * sqrt_ξ1_νInv

  set_solver_specific!(stats, :xi, sqrt_ξ1_νInv)
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

    s1 .= s

    sub_atol = stats.iter == 0 ? 1.0e-3 : max(sub_atol, min(1e-3, sqrt_ξ1_νInv)) # 1.0e-5 default
    #@debug "setting inner stopping tolerance to" subsolver_options.optTol
    #subsolver_args = subsolver == R2DH ? (SpectralGradient(1., f.meta.nvar),) : ()
    nlp_model = FirstOrderModel(φ,∇φ!,s)
    model = RegularizedNLPModel(nlp_model, ψ)
    #model.selected .= reg_nlp.selected
    if sub_solver == R2Solver
      solve!(
        sub_solver, 
        model, 
        sub_stats, 
        x = s, 
        atol = sub_atol,
        ν = ν_subsolver,
        kwargs...)  
    else
      solve!(
        sub_solver, 
        model, 
        sub_stats, 
        ∇fk,
        Bk/σk + opEye(nlp.meta.nvar,nlp.meta.nvar),
        σk,
        atol = sub_atol,
        max_time = max_time - stats.elapsed_time,
        kwargs...)  
    end  
    s .= sub_stats.solution

    if norm(s) > β * norm(s1)
      s .= s1
    end
    if mk(s) > mk(s1)
      s .= s1
    end

    sub_atol = sub_atol_init

    xkn .= xk .+ s
    fkn = obj(nlp, xkn)
    hkn = h(xkn[selected])
    hkn == -Inf && error("nonsmooth term is not proper")

    mks = mk(s)
    Δobj = fk + hk - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ξ = hk - mks + max(1, abs(hk)) * 10 * eps()

    ρk = Δobj / ξ

    verbose > 0 &&
      stats.iter % verbose == 0 &&
      @info log_row(
        Any[
          stats.iter,
          sub_stats.iter,
          fk,
          hk,
          sqrt_ξ1_νInv,
          ρk,
          σk,
          norm(xk),
          norm(s),
          λmax,
          (η2 ≤ ρk < Inf) ? "↗" : (ρk < η1 ? "↘" : "="),
        ],
        colsep = 1,
      )

    if η2 ≤ ρk < Inf
        σk = max(σk/γ, σmin)
    end

    if η1 ≤ ρk < Inf
      xk .= xkn
      has_bounds(nlp) && set_bounds!(ψ, l_bound - xk, u_bound - xk)

      #update functions
      fk = fkn
      hk = hkn

      shift!(ψ, xk)
      ∇fk = grad!(nlp, xk, ∇fk)
      @. mν∇fk = -ν_subsolver * ∇fk
      if quasiNewtTest
        push!(nlp, s, ∇fk - ∇fk⁻)
      end
      Bk = hess_op(nlp, xk)
      try 
        λmax = opnorm(Bk)
      catch LAPACKException
        λmax = opnorm(Matrix(Bk))
      end
      ∇fk⁻ .= ∇fk
    end

    if ρk < η1 || ρk == Inf
      σk = σk * γ
    end
    νInv = (1 + θ) *( σk + λmax)

    set_objective!(stats, fk + hk)
    set_solver_specific!(stats, :smooth_obj, fk)
    set_solver_specific!(stats, :nonsmooth_obj, hk)
    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)

    prox!(s, ψ, mν∇fk, ν_subsolver)
    mks = mk1(s)

    ξ1 = hk - mks + max(1, abs(hk)) * 10 * eps()
    sqrt_ξ1_νInv = ξ1 ≥ 0 ? sqrt(ξ1 / ν_subsolver) : sqrt(-ξ1 / ν_subsolver)
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
        sub_stats.iter,
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

  set_solution!(stats,xk)
  return stats
end
