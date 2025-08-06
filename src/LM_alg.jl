export LM, LMSolver, solve!

import SolverCore.solve!

mutable struct TRDHSolver # FIXME
end

mutable struct LMSolver{
  T <: Real,
  G <: ShiftedProximableFunction,
  V <: AbstractVector{T},
  M <: AbstractLinearOperator{T},
  ST <: AbstractOptimizationSolver,
  PB <: AbstractRegularizedNLPModel,
} <: AbstractOptimizationSolver
  xk::V
  ∇fk::V
  mν∇fk::V
  Fk::V
  Fkn::V
  Jk::M
  ψ::G
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

function LMSolver(
  reg_nls::AbstractRegularizedNLPModel{T, V};
  subsolver = R2Solver,
) where{T, V}
  x0 = reg_nls.model.meta.x0
  l_bound = reg_nls.model.meta.lvar
  u_bound = reg_nls.model.meta.uvar

  xk = similar(x0)
  ∇fk = similar(x0)
  mν∇fk = similar(x0)
  Fk = similar(x0, reg_nls.model.nls_meta.nequ)
  Fkn = similar(Fk)
  Jk = jac_op_residual(reg_nls.model, xk)
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
    has_bnds ? shifted(reg_nls.h, xk, l_bound_m_x, u_bound_m_x, reg_nls.selected) :
    shifted(reg_nls.h, xk)
  
  sub_nlp = LMModel(Jk, Fk, T(1), x0)
  subpb = RegularizedNLPModel(sub_nlp, ψ)
  substats = RegularizedExecutionStats(subpb)
  subsolver = subsolver(subpb)

  return LMSolver(
    xk,
    ∇fk,
    mν∇fk,
    Fk,
    Fkn,
    Jk,
    ψ,
    xkn,
    s,
    has_bnds,
    l_bound,
    u_bound,
    l_bound_m_x,
    u_bound_m_x,
    subsolver,
    subpb,
    substats
  )
end

function SolverCore.solve!(
  solver::LMSolver{T, G, V},
  reg_nls::AbstractRegularizedNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing, 
  x::V = reg_nls.model.meta.x0,
  nonlinear::Bool = true,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  neg_tol::T = zero(T),
  verbose::Int = 0,
  max_iter::Int = 10000,
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  σk::T = eps(T)^(1 / 5),
  σmin::T = eps(T),
  η1::T = √√eps(T),
  η2::T = T(0.9),
  γ::T = T(3),
  θ::T = 1/(1 + eps(T)^(1 / 5)),
) where {T, V, G}
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
  Jk = solver.Jk
  ∇fk = solver.∇fk
  mν∇fk = solver.mν∇fk
  ψ = solver.ψ
  xkn = solver.xkn
  s = solver.s

  has_bnds = solver.has_bnds
  if has_bnds
    l_bound = solver.l_bound
    u_bound = solver.u_bound
    l_bound_m_x = solver.l_bound_m_x
    u_bound_m_x = solver.u_bound_m_x
  end

  # initialize parameters
  improper = false
  hk = @views h(xk[selected])
  if hk == Inf
    verbose > 0 && @info "LM: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, xk, one(eltype(x0)))
    hk = @views h(xk[selected])
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "LM: found point where h has value" hk
  end
  improper = (hk == -Inf)
  improper == true && @warn "LM: Improper term detected"
  improper == true && return stats

  if verbose > 0
    @info log_header(
      [:outer, :inner, :fx, :hx, :xi, :ρ, :σ, :normx, :norms, :normJ, :arrow],
      [Int, Int, T, T, T, T, T, T, T, T, Char],
      hdr_override = Dict{Symbol, String}(
        :fx => "f(x)",
        :hx => "h(x)",
        :xi => "√(ξ1/ν)",
        :normx => "‖x‖",
        :norms => "‖s‖",
        :normJ => "‖J‖²",
        :arrow => "LM",
      ),
      colsep = 1,
    )
  end

  local ξ1::T
  local ρk::T = zero(T)

  residual!(nls, xk, Fk)
  Jk = jac_op_residual(nls, xk)
  jtprod_residual!(nls, xk, Fk, ∇fk)
  fk = dot(Fk, Fk) / 2

  σmax, found_σ = opnorm(Jk)
  found_σ || error("operator norm computation failed")
  ν = θ / (σmax^2 + σk) # ‖J'J + σₖ I‖ = ‖J‖² + σₖ
  sqrt_ξ1_νInv = one(T)

  @. mν∇fk = -ν * ∇fk

  set_iter!(stats, 0)
  start_time = time()
  set_time!(stats, 0.0)
  set_objective!(stats, fk + hk)
  set_solver_specific!(stats, :smooth_obj, fk)
  set_solver_specific!(stats, :nonsmooth_obj, hk)

  φ1 = let Fk = Fk, ∇fk = ∇fk
    d -> dot(Fk, Fk) / 2 + dot(∇fk, d) # ∇fk = Jk^T Fk
  end

  mk1 = let φ1 = φ1, ψ = ψ
    d -> φ1(d) + ψ(d)
  end

  mk = let ψ = ψ, solver = solver
    d -> obj(solver.subpb.model, d) + ψ(d)
  end

  prox!(s, ψ, mν∇fk, ν)
  ξ1 = fk + hk - mk1(s) + max(1, abs(fk + hk)) * 10 * eps()
  sqrt_ξ1_νInv = ξ1 ≥ 0 ? sqrt(ξ1 / ν) : sqrt(-ξ1 / ν)
  solved = (ξ1 < 0 && sqrt_ξ1_νInv ≤ neg_tol) || (ξ1 ≥ 0 && sqrt_ξ1_νInv ≤ atol)
  (ξ1 < 0 && sqrt_ξ1_νInv > neg_tol) &&
    error("LM: prox-gradient step should produce a decrease but ξ1 = $(ξ1)")
  atol += rtol * sqrt_ξ1_νInv # make stopping test absolute and relative

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

    sub_atol = stats.iter == 0 ? 1.0e-3 : min(sqrt_ξ1_νInv ^ (1.5), sqrt_ξ1_νInv * 1e-3)
    solver.subpb.model.σ = σk
    isa(solver.subsolver, R2DHSolver) && (solver.subsolver.D.d[1] = 1/ν)
    if isa(solver.subsolver, R2Solver) #FIXME
      solve!(
        solver.subsolver,
        solver.subpb,
        solver.substats,
        x = s,
        atol = sub_atol,
        ν = ν,
      )
    else
      solve!(
        solver.subsolver,
        solver.subpb,
        solver.substats,
        x = s,
        atol = sub_atol,
        σk = σk, #FIXME
      )
    end

    s .= solver.substats.solution

    xkn .= xk .+ s
    residual!(nls, xkn, Fkn)
    fkn = dot(Fkn, Fkn) / 2
    hkn = @views h(xkn[selected])
    mks = mk(s)
    ξ = fk + hk - mks + max(1, abs(hk)) * 10 * eps()

    if (ξ ≤ 0 || isnan(ξ))
      error("LM: failed to compute a step: ξ = $ξ")
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
          sqrt_ξ1_νInv,
          ρk,
          σk,
          norm(xk),
          norm(s),
          1 / ν,
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
      Fk .= Fkn
      fk = fkn
      hk = hkn

      # update gradient & Hessian
      shift!(ψ, xk)
      Jk = jac_op_residual(nls, xk)
      jtprod_residual!(nls, xk, Fk, ∇fk)

      # update opnorm if not linear least squares
      if nonlinear == true
        σmax, found_σ = opnorm(Jk)
        found_σ || error("operator norm computation failed")
      end
    end

    if η2 ≤ ρk < Inf
      σk = max(σk / γ, σmin)
    end

    if ρk < η1 || ρk == Inf
      σk = σk * γ
    end

    set_objective!(stats, fk + hk)
    set_solver_specific!(stats, :smooth_obj, fk)
    set_solver_specific!(stats, :nonsmooth_obj, hk)
    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)

    ν = θ / (σmax^2 + σk) # ‖J'J + σₖ I‖ = ‖J‖² + σₖ

    @. mν∇fk = - ν * ∇fk
    prox!(s, ψ, mν∇fk, ν)
    mks = mk1(s)

    ξ1 = fk + hk - mks + max(1, abs(hk)) * 10 * eps()

    sqrt_ξ1_νInv = ξ1 ≥ 0 ? sqrt(ξ1 / ν) : sqrt(-ξ1 / ν)
    solved = (ξ1 < 0 && sqrt_ξ1_νInv ≤ neg_tol) || (ξ1 ≥ 0 && sqrt_ξ1_νInv ≤ atol)

    (ξ1 < 0 && sqrt_ξ1_νInv > neg_tol) &&
      error("LM: prox-gradient step should produce a decrease but ξ1 = $(ξ1)")

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
        1 / ν,
        "",
      ],
      colsep = 1,
    )
    @info "LM: terminating with √(ξ1/ν) = $(sqrt_ξ1_νInv)"
  end

  set_solution!(stats, xk)
  set_residuals!(stats, zero(T), sqrt_ξ1_νInv)
  return stats
end

"""
    LM(nls, h, options; kwargs...)

A Levenberg-Marquardt method for the problem

    min ½ ‖F(x)‖² + h(x)

where F: ℝⁿ → ℝᵐ and its Jacobian J are Lipschitz continuous and h: ℝⁿ → ℝ is
lower semi-continuous, proper and prox-bounded.

At each iteration, a step s is computed as an approximate solution of

    min  ½ ‖J(x) s + F(x)‖² + ½ σ ‖s‖² + ψ(s; x)

where F(x) and J(x) are the residual and its Jacobian at x, respectively, ψ(s; x) = h(x + s),
and σ > 0 is a regularization parameter.

### Arguments

* `nls::AbstractNLSModel`: a smooth nonlinear least-squares problem
* `h`: a regularizer such as those defined in ProximalOperators
* `options::ROSolverOptions`: a structure containing algorithmic parameters

### Keyword arguments

* `x0::AbstractVector`: an initial guess (default: `nls.meta.x0`)
* `subsolver_logger::AbstractLogger`: a logger to pass to the subproblem solver
* `subsolver`: the procedure used to compute a step (`PG`, `R2` or `TRDH`)
* `subsolver_options::ROSolverOptions`: default options to pass to the subsolver.
* `selected::AbstractVector{<:Integer}`: (default `1:nls.meta.nvar`).

### Return values

* `xk`: the final iterate
* `Fobj_hist`: an array with the history of values of the smooth objective
* `Hobj_hist`: an array with the history of values of the nonsmooth objective
* `Complex_hist`: an array with the history of number of inner iterations.
"""
function LM(
  nls::AbstractNLSModel,
  h::H,
  options::ROSolverOptions;
  x0::AbstractVector = nls.meta.x0,
  subsolver_logger::Logging.AbstractLogger = Logging.NullLogger(),
  subsolver = R2,
  subsolver_options = ROSolverOptions(ϵa = options.ϵa),
  selected::AbstractVector{<:Integer} = 1:(nls.meta.nvar),
  nonlinear::Bool = true,
) where {H}
  start_time = time()
  elapsed_time = 0.0
  # initialize passed options
  ϵ = options.ϵa
  ϵ_subsolver = subsolver_options.ϵa
  ϵr = options.ϵr
  verbose = options.verbose
  maxIter = options.maxIter
  maxTime = options.maxTime
  η1 = options.η1
  η2 = options.η2
  γ = options.γ
  θ = options.θ
  σmin = options.σmin
  σk = options.σk

  # store initial values of the subsolver_options fields that will be modified
  ν_subsolver = subsolver_options.ν
  ϵa_subsolver = subsolver_options.ϵa

  local l_bound, u_bound
  treats_bounds = has_bounds(nls) || subsolver == TRDH
  if treats_bounds
    l_bound = nls.meta.lvar
    u_bound = nls.meta.uvar
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
    verbose > 0 && @info "LM: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, x0, one(eltype(x0)))
    hk = h(xk[selected])
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "LM: found point where h has value" hk
  end
  hk == -Inf && error("nonsmooth term is not proper")
  ψ = treats_bounds ? shifted(h, xk, l_bound - xk, u_bound - xk, selected) : shifted(h, xk)

  xkn = similar(xk)

  local ξ1
  local sqrt_ξ1_νInv
  k = 0
  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, maxIter)
  Grad_hist = zeros(Int, maxIter)
  Resid_hist = zeros(Int, maxIter)

  if verbose > 0
    #! format: off
    @info @sprintf "%6s %8s %8s %8s %7s %7s %8s %7s %7s %7s %7s %1s" "outer" "inner" "f(x)" "h(x)" "√(ξ1/ν)" "√ξ" "ρ" "σ" "‖x‖" "‖s‖" "‖Jₖ‖²" "reg"
    #! format: on
  end

  # main algorithm initialization
  Fk = residual(nls, xk)
  Fkn = similar(Fk)
  fk = dot(Fk, Fk) / 2
  Jk = jac_op_residual(nls, xk)
  ∇fk = Jk' * Fk
  JdFk = similar(Fk)   # temporary storage
  Jt_Fk = similar(∇fk)

  σmax, found_σ = opnorm(Jk)
  found_σ || error("operator norm computation failed")
  ν = θ / (σmax^2 + σk) # ‖J'J + σₖ I‖ = ‖J‖² + σₖ

  s = zero(xk)

  optimal = false
  tired = k ≥ maxIter || elapsed_time > maxTime

  while !(optimal || tired)
    k = k + 1
    elapsed_time = time() - start_time
    Fobj_hist[k] = fk
    Hobj_hist[k] = hk
    Grad_hist[k] = nls.counters.neval_jtprod_residual + nls.counters.neval_jprod_residual
    Resid_hist[k] = nls.counters.neval_residual

    # model for first prox-gradient iteration
    φ1(d) = begin # || Fk ||^2/2 + d*Jk'*Fk
      jtprod_residual!(nls, xk, Fk, Jt_Fk)
      dot(Fk, Fk) / 2 + dot(Jt_Fk, d)
    end

    mk1(d) = φ1(d) + ψ(d)

    # TODO: reuse residual computation
    # model for subsequent prox-gradient iterations
    φ(d) = begin
      jprod_residual!(nls, xk, d, JdFk)
      JdFk .+= Fk
      return dot(JdFk, JdFk) / 2 + σk * dot(d, d) / 2
    end

    ∇φ!(g, d) = begin
      jprod_residual!(nls, xk, d, JdFk)
      JdFk .+= Fk
      jtprod_residual!(nls, xk, JdFk, g)
      g .+= σk * d
      return g
    end

    mk(d) = begin
      jprod_residual!(nls, xk, d, JdFk)
      JdFk .+= Fk
      return dot(JdFk, JdFk) / 2 + σk * dot(d, d) / 2 + ψ(d)
    end

    # take first proximal gradient step s1 and see if current xk is nearly stationary
    # s1 minimizes φ1(s) + ‖s‖² / 2 / ν + ψ(s) ⟺ s1 ∈ prox{νψ}(-ν∇φ1(0)).
    ∇fk .*= -ν  # reuse gradient storage
    prox!(s, ψ, ∇fk, ν)
    ξ1 = fk + hk - mk1(s) + max(1, abs(fk + hk)) * 10 * eps()  # TODO: isn't mk(s) returned by subsolver?
    ξ1 > 0 || error("LM: first prox-gradient step should produce a decrease but ξ1 = $(ξ1)")
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

    #  subsolver_options.ϵa = k == 1 ? 1.0e-1 : max(ϵ_subsolver, min(1.0e-2, ξ1 / 10))
    subsolver_options.ϵa = k == 1 ? 1.0e-3 : min(sqrt_ξ1_νInv^(1.5), sqrt_ξ1_νInv * 1e-3) # 1.0e-5 default
    subsolver_options.ν = ν
    subsolver_args = subsolver == R2DH ? (SpectralGradient(1 / ν, nls.meta.nvar),) : ()
    @debug "setting inner stopping tolerance to" subsolver_options.optTol
    #s, iter, _ = with_logger(subsolver_logger) do
    #subsolver_options.verbose = 1
    s, iter, _ = subsolver(φ, ∇φ!, ψ, subsolver_args..., subsolver_options, s)
    #end
    # restore initial subsolver_options here so that it is not modified if there is an error
    subsolver_options.ν = ν_subsolver
    subsolver_options.ϵa = ϵa_subsolver

    Complex_hist[k] = iter

    xkn .= xk .+ s
    residual!(nls, xkn, Fkn)
    fkn = dot(Fkn, Fkn) / 2
    hkn = h(xkn[selected])
    hkn == -Inf && error("nonsmooth term is not proper")
    mks = mk(s)
    ξ = fk + hk - mks + max(1, abs(hk)) * 10 * eps()

    if (ξ ≤ 0 || isnan(ξ))
      error("LM: failed to compute a step: ξ = $ξ")
    end

    Δobj = fk + hk - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ρk = Δobj / ξ

    σ_stat = (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")

    if (verbose > 0) && (k % ptf == 0)
      #! format: off
      @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s" k iter fk hk sqrt_ξ1_νInv sqrt(ξ) ρk σk norm(xk) norm(s) 1/ν σ_stat
      #! format: off
    end

    if η2 ≤ ρk < Inf
      σk = max(σk / γ, σmin)
    end

    if η1 ≤ ρk < Inf
      xk .= xkn
      treats_bounds && set_bounds!(ψ, l_bound - xk, u_bound - xk)

      # update functions
      Fk .= Fkn
      fk = fkn
      hk = hkn

      # update gradient & Hessian
      shift!(ψ, xk)
      Jk = jac_op_residual(nls, xk)
      jtprod_residual!(nls, xk, Fk, ∇fk)

      # update opnorm if not linear least squares
      if nonlinear == true
        σmax, found_σ = opnorm(Jk)
        found_σ || error("operator norm computation failed")
      end

      Complex_hist[k] += 1
    end

    if ρk < η1 || ρk == Inf
      σk = σk * γ
    end
    ν = θ / (σmax^2 + σk) # ‖J'J + σₖ I‖ = ‖J‖² + σₖ
    tired = k ≥ maxIter || elapsed_time > maxTime
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8s %8.1e %8.1e" k "" fk hk
    elseif optimal
      #! format: off
      @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8s %7.1e %7.1e %7.1e %7.1e" k 1 fk hk sqrt_ξ1_νInv sqrt(ξ1) "" σk norm(xk) norm(s) 1/ν
      #! format: on
      @info "LM: terminating with √(ξ1/ν) = $(sqrt_ξ1_νInv)"
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

  stats = GenericExecutionStats(nls)
  set_status!(stats, status)
  set_solution!(stats, xk)
  set_objective!(stats, fk + hk)
  set_residuals!(stats, zero(eltype(xk)), ξ1 ≥ 0 ? sqrt_ξ1_νInv : ξ1)
  set_iter!(stats, k)
  set_time!(stats, elapsed_time)
  set_solver_specific!(stats, :sigma, σk)
  set_solver_specific!(stats, :Fhist, Fobj_hist[1:k])
  set_solver_specific!(stats, :Hhist, Hobj_hist[1:k])
  set_solver_specific!(stats, :NonSmooth, h)
  set_solver_specific!(stats, :SubsolverCounter, Complex_hist[1:k])
  set_solver_specific!(stats, :NLSGradHist, Grad_hist[1:k])
  set_solver_specific!(stats, :ResidHist, Resid_hist[1:k])
  return stats
end
