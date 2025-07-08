export TR, TRSolver, solve!

import SolverCore.solve!

mutable struct TRSolver{
  T <: Real,
  G <: ShiftedProximableFunction,
  V <: AbstractVector{T},
  N,
  ST <: AbstractOptimizationSolver,
  PB <: AbstractRegularizedNLPModel
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
  reg_nlp::AbstractRegularizedNLPModel{T, V},
  χ::X;
  subsolver = R2Solver
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
    has_bnds || subsolver == TRDHSolver ? shifted(reg_nlp.h, xk, l_bound_m_x, u_bound_m_x, reg_nlp.selected) :
    shifted(reg_nlp.h, xk, T(1), χ)

  Bk = hess_op(reg_nlp.model, x0)
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
    TR(nlp, h, χ, options; kwargs...)

A trust-region method for the problem

    min f(x) + h(x)

where f: ℝⁿ → ℝ has a Lipschitz-continuous Jacobian, and h: ℝⁿ → ℝ is
lower semi-continuous and proper.

About each iterate xₖ, a step sₖ is computed as an approximate solution of

    min  φ(s; xₖ) + ψ(s; xₖ)  subject to  ‖s‖ ≤ Δₖ

where φ(s ; xₖ) = f(xₖ) + ∇f(xₖ)ᵀs + ½ sᵀ Bₖ s  is a quadratic approximation of f about xₖ,
ψ(s; xₖ) = h(xₖ + s), ‖⋅‖ is a user-defined norm and Δₖ > 0 is the trust-region radius.
The subproblem is solved inexactly by way of a first-order method such as the proximal-gradient
method or the quadratic regularization method.

### Arguments

* `nlp::AbstractNLPModel`: a smooth optimization problem
* `h`: a regularizer such as those defined in ProximalOperators
* `χ`: a norm used to define the trust region in the form of a regularizer
* `options::ROSolverOptions`: a structure containing algorithmic parameters

The objective, gradient and Hessian of `nlp` will be accessed.
The Hessian is accessed as an abstract operator and need not be the exact Hessian.

### Keyword arguments

* `x0::AbstractVector`: an initial guess (default: `nlp.meta.x0`)
* `subsolver_logger::AbstractLogger`: a logger to pass to the subproblem solver (default: the null logger)
* `subsolver`: the procedure used to compute a step (`PG`, `R2` or `TRDH`)
* `subsolver_options::ROSolverOptions`: default options to pass to the subsolver (default: all defaut options)
* `selected::AbstractVector{<:Integer}`: (default `1:f.meta.nvar`).

### Return values

* `xk`: the final iterate
* `Fobj_hist`: an array with the history of values of the smooth objective
* `Hobj_hist`: an array with the history of values of the nonsmooth objective
* `Complex_hist`: an array with the history of number of inner iterations.
"""
function TR(
  f::AbstractNLPModel,
  h::H,
  χ::X,
  options::ROSolverOptions{R};
  x0::AbstractVector{R} = f.meta.x0,
  subsolver_logger::Logging.AbstractLogger = Logging.NullLogger(),
  subsolver = R2,
  subsolver_options = ROSolverOptions(ϵa = options.ϵa),
  selected::AbstractVector{<:Integer} = 1:(f.meta.nvar),
) where {H, X, R}
  start_time = time()
  elapsed_time = 0.0
  # initialize passed options
  ϵ = options.ϵa
  ϵ_subsolver = subsolver_options.ϵa
  ϵr = options.ϵr
  Δk = options.Δk
  verbose = options.verbose
  maxIter = options.maxIter
  maxTime = options.maxTime
  η1 = options.η1
  η2 = options.η2
  γ = options.γ
  α = options.α
  θ = options.θ
  β = options.β

  # store initial values of the subsolver_options fields that will be modified
  ν_subsolver = subsolver_options.ν
  ϵa_subsolver = subsolver_options.ϵa
  Δk_subsolver = subsolver_options.Δk

  local l_bound, u_bound
  if has_bounds(f) || subsolver == TRDH
    l_bound = f.meta.lvar
    u_bound = f.meta.uvar
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
    verbose > 0 && @info "TR: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, x0, one(eltype(x0)))
    hk = h(xk[selected])
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "TR: found point where h has value" hk
  end
  hk == -Inf && error("nonsmooth term is not proper")

  xkn = similar(xk)
  s = zero(xk)
  ψ =
    (has_bounds(f) || subsolver == TRDH) ?
    shifted(h, xk, max.(-Δk, l_bound - xk), min.(Δk, u_bound - xk), selected) :
    shifted(h, xk, Δk, χ)

  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, maxIter)
  if verbose > 0
    #! format: off
    @info @sprintf "%6s %8s %8s %8s %7s %7s %8s %7s %7s %7s %7s %1s" "outer" "inner" "f(x)" "h(x)" "√(ξ1/ν)" "√ξ" "ρ" "Δ" "‖x‖" "‖s‖" "‖Bₖ‖" "TR"
    #! format: on
  end

  local ξ1
  k = 0

  fk = obj(f, xk)
  ∇fk = grad(f, xk)
  ∇fk⁻ = copy(∇fk)

  quasiNewtTest = isa(f, QuasiNewtonModel)
  Bk = hess_op(f, xk)

  λmax, found_λ = opnorm(Bk)
  found_λ || error("operator norm computation failed")
  α⁻¹Δ⁻¹ = 1 / (α * Δk)
  ν = 1 / (α⁻¹Δ⁻¹ + λmax * (α⁻¹Δ⁻¹ + 1))
  sqrt_ξ1_νInv = one(R)

  optimal = false
  tired = k ≥ maxIter || elapsed_time > maxTime

  while !(optimal || tired)
    k = k + 1
    elapsed_time = time() - start_time
    Fobj_hist[k] = fk
    Hobj_hist[k] = hk

    # model for first prox-gradient step and ξ1
    φ1(d) = ∇fk' * d
    mk1(d) = φ1(d) + ψ(d)

    # model for subsequent prox-gradient steps and ξ
    φ(d) = (d' * (Bk * d)) / 2 + ∇fk' * d

    ∇φ!(g, d) = begin
      mul!(g, Bk, d)
      g .+= ∇fk
      g
    end

    mk(d) = φ(d) + ψ(d)

    # Take first proximal gradient step s1 and see if current xk is nearly stationary.
    # s1 minimizes φ1(s) + ‖s‖² / 2 / ν + ψ(s) ⟺ s1 ∈ prox{νψ}(-ν∇φ1(0)).
    prox!(s, ψ, -ν * ∇fk, ν)
    ξ1 = hk - mk1(s) + max(1, abs(hk)) * 10 * eps()
    ξ1 > 0 || error("TR: first prox-gradient step should produce a decrease but ξ1 = $(ξ1)")
    sqrt_ξ1_νInv = sqrt(ξ1 / ν)

    if ξ1 ≥ 0 && k == 1
      ϵ_increment = ϵr * sqrt_ξ1_νInv
      ϵ += ϵ_increment  # make stopping test absolute and relative
      ϵ_subsolver += ϵ_increment
    end

    if sqrt_ξ1_νInv < ϵ
      # the current xk is approximately first-order stationary
      optimal = true
      continue
    end

    subsolver_options.ϵa = k == 1 ? 1.0e-5 : max(ϵ_subsolver, min(1e-2, sqrt_ξ1_νInv))
    ∆_effective = min(β * χ(s), Δk)
    (has_bounds(f) || subsolver == TRDH) ?
    set_bounds!(ψ, max.(-∆_effective, l_bound - xk), min.(∆_effective, u_bound - xk)) :
    set_radius!(ψ, ∆_effective)
    subsolver_options.Δk = ∆_effective / 10
    subsolver_options.ν = ν
    subsolver_args = subsolver == TRDH ? (SpectralGradient(1 / ν, f.meta.nvar),) : ()

    #s, iter, outdict = with_logger(subsolver_logger) do
      s, iter, outdict = subsolver(φ, ∇φ!, ψ, subsolver_args..., subsolver_options, s)
    #end

    # restore initial values of subsolver_options here so that it is not modified
    # if there is an error
    subsolver_options.ν = ν_subsolver
    subsolver_options.ϵa = ϵa_subsolver
    subsolver_options.Δk = Δk_subsolver

    Complex_hist[k] = sum(outdict[:Chist])

    sNorm = χ(s)
    xkn .= xk .+ s
    fkn = obj(f, xkn)
    hkn = h(xkn[selected])
    hkn == -Inf && error("nonsmooth term is not proper")

    Δobj = fk + hk - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ξ = hk - mk(s) + max(1, abs(hk)) * 10 * eps()

    if (ξ ≤ 0 || isnan(ξ))
      error("TR: failed to compute a step: ξ = $ξ")
    end

    ρk = Δobj / ξ

    TR_stat = (η2 ≤ ρk < Inf) ? "↗" : (ρk < η1 ? "↘" : "=")

    if (verbose > 0) && (k % ptf == 0)
      #! format: off
      @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s" k iter fk hk sqrt_ξ1_νInv sqrt(ξ) ρk ∆_effective χ(xk) sNorm λmax TR_stat
      #! format: on
    end

    if η2 ≤ ρk < Inf
      Δk = max(Δk, γ * sNorm)
      !(has_bounds(f) || subsolver == TRDH) && set_radius!(ψ, Δk)
    end

    if η1 ≤ ρk < Inf
      xk .= xkn
      (has_bounds(f) || subsolver == TRDH) &&
        set_bounds!(ψ, max.(-Δk, l_bound - xk), min.(Δk, u_bound - xk))

      #update functions
      fk = fkn
      hk = hkn
      shift!(ψ, xk)
      ∇fk = grad(f, xk)
      # grad!(f, xk, ∇fk)
      if quasiNewtTest
        push!(f, s, ∇fk - ∇fk⁻)
      end
      Bk = hess_op(f, xk)
      λmax, found_λ = opnorm(Bk)
      found_λ || error("operator norm computation failed")
      ∇fk⁻ .= ∇fk
    end

    if ρk < η1 || ρk == Inf
      Δk = Δk / 2
      (has_bounds(f) || subsolver == TRDH) ?
      set_bounds!(ψ, max.(-Δk, l_bound - xk), min.(Δk, u_bound - xk)) : set_radius!(ψ, Δk)
    end
    α⁻¹Δ⁻¹ = 1 / (α * Δk)
    ν = 1 / (α⁻¹Δ⁻¹ + λmax * (α⁻¹Δ⁻¹ + 1))
    tired = k ≥ maxIter || elapsed_time > maxTime
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8s %8.1e %8.1e" k "" fk hk
    elseif optimal
      #! format: off
      @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8s %7.1e %7.1e %7.1e %7.1e" k 1 fk hk sqrt_ξ1_νInv sqrt(ξ1) "" Δk χ(xk) χ(s) λmax
      #! format: on
      @info "TR: terminating with √(ξ1/ν) = $(sqrt_ξ1_νInv)"
    end
  end

  status = if optimal
    :first_order
  elseif elapsed_time > maxTime
    :max_time
  elseif tired
    :max_iter
  else
    :exception
  end

  stats = GenericExecutionStats(f)
  set_status!(stats, status)
  set_solution!(stats, xk)
  set_objective!(stats, fk + hk)
  set_residuals!(stats, zero(eltype(xk)), sqrt_ξ1_νInv)
  set_iter!(stats, k)
  set_time!(stats, elapsed_time)
  set_solver_specific!(stats, :radius, Δk)
  set_solver_specific!(stats, :Fhist, Fobj_hist[1:k])
  set_solver_specific!(stats, :Hhist, Hobj_hist[1:k])
  set_solver_specific!(stats, :NonSmooth, h)
  set_solver_specific!(stats, :SubsolverCounter, Complex_hist[1:k])
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
  ψ = solver.ψ
  xkn = solver.xkn
  s = solver.s
  χ = solver.χ
  has_bnds = solver.has_bnds

  if has_bnds || isa(solver.subsolver, TRDHSolver) #TODO elsewhere ?
    l_bound_m_x = solver.l_bound_m_x
    u_bound_m_x = solver.u_bound_m_x
    l_bound = solver.l_bound
    u_bound = solver.u_bound
    @. l_bound_m_x = l_bound - xk
    @. u_bound_m_x = u_bound - xk
    set_bounds!(ψ, l_bound_m_x, u_bound_m_x)
  else
    set_radius!(ψ, Δk)
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
  improper == true && return stats

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
  local ρk::T = zero(T)

  fk = obj(nlp, xk)
  grad!(nlp, xk, ∇fk)
  ∇fk⁻ .= ∇fk

  quasiNewtTest = isa(nlp, QuasiNewtonModel)
  λmax::T = T(1)
  solver.subpb.model.B = hess_op(nlp, xk)

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
    
    if isa(solver.subsolver, TRDHSolver) #FIXME
      solver.subsolver.D.d[1] = 1/ν₁
      solve!(
        solver.subsolver, 
        solver.subpb, 
        solver.substats; 
        x = s, 
        atol = stats.iter == 0 ? 1e-5 : max(sub_atol, min(1e-2, sqrt_ξ1_νInv)),
        Δk = min(β * χ(s), Δk) / 10
        )
    else 
      solve!(
        solver.subsolver, 
        solver.subpb, 
        solver.substats; 
        x = s, 
        atol = stats.iter == 0 ? 1e-5 : max(sub_atol, min(1e-2, sqrt_ξ1_νInv)),
        ν  = ν₁,
      )
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
          (η2 ≤ ρk < Inf) ? "↗" : (ρk < η1 ? "↘" : "="),
        ],
        colsep = 1,
      )
  
    if η1 ≤ ρk < Inf
      xk .= xkn
      ∆_effective = min(β * χ(s), Δk)
      if has_bnds || isa(solver.subsolver, TRDHSolver)
        @. l_bound_m_x = l_bound - xk
        @. u_bound_m_x = u_bound - xk
        set_bounds!(ψ, l_bound_m_x, u_bound_m_x)
      end
      fk = fkn
      hk = hkn

      shift!(ψ, xk)
      grad!(nlp, xk, ∇fk)

      if quasiNewtTest
        @. ∇fk⁻ = ∇fk - ∇fk⁻
        push!(nlp, s, ∇fk⁻) # update QN operator
      end

      solver.subpb.model.B = hess_op(nlp, xk)

      λmax, found_λ = opnorm(solver.subpb.model.B)
      found_λ || error("operator norm computation failed")

      ∇fk⁻ .= ∇fk
    end

    if η2 ≤ ρk < Inf
      Δk = max(Δk, γ * sNorm)
      if !(has_bnds || isa(solver.subsolver, TRDHSolver))
        set_radius!(ψ, Δk)
      end
    end

    if η2 ≤ ρk < Inf
      Δk = max(Δk, γ * sNorm)
      !(has_bnds || isa(solver.subsolver, TRDHSolver)) && set_radius!(ψ, Δk)
    end

    if ρk < η1 || ρk == Inf
      Δk = Δk / 2
      if has_bnds || isa(solver.subsolver, TRDHSolver) #TODO elsewhere ?
      @. l_bound_m_x = l_bound - xk
      @. u_bound_m_x = u_bound - xk
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
        Any[
          stats.iter,
          solver.substats.iter,
          fk,
          hk,
          sqrt_ξ1_νInv,
          ρk,
          Δk,
          χ(xk),
          χ(s),
          λmax,
          "",
        ],
        colsep = 1,
      )
    @info "TR: terminating with √(ξ1/ν) = $(sqrt_ξ1_νInv)"
  end
end
