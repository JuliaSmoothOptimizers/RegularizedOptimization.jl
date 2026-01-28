export LM, LMSolver, solve!

import SolverCore.solve!

mutable struct LMSolver{
  T <: Real,
  G <: ShiftedProximableFunction,
  V <: AbstractVector{T},
  ST <: AbstractOptimizationSolver,
  PB <: AbstractRegularizedNLPModel,
} <: AbstractOptimizationSolver
  xk::V
  ∇fk::V
  mν∇fk::V
  Fk::V
  Fkn::V
  Jv::V
  Jtv::V
  ψ::G
  xkn::V
  s::V
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

function LMSolver(
  reg_nls::AbstractRegularizedNLPModel{T, V};
  subsolver = R2Solver,
  m_monotone::Int = 1,
) where {T, V}
  x0 = reg_nls.model.meta.x0
  l_bound = reg_nls.model.meta.lvar
  u_bound = reg_nls.model.meta.uvar

  xk = similar(x0)
  ∇fk = similar(x0)
  mν∇fk = similar(x0)
  Fk = similar(x0, reg_nls.model.nls_meta.nequ)
  Fkn = similar(Fk)
  Jv = similar(Fk)
  Jtv = similar(x0)
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

  m_fh_hist = fill(T(-Inf), m_monotone - 1)

  ψ =
    has_bnds ? shifted(reg_nls.h, xk, l_bound_m_x, u_bound_m_x, reg_nls.selected) :
    shifted(reg_nls.h, xk)

  Jk = jac_op_residual(reg_nls.model, xk)
  sub_nlp = LMModel(Jk, Fk, T(1), xk)
  subpb = RegularizedNLPModel(sub_nlp, ψ)
  substats = RegularizedExecutionStats(subpb)
  subsolver = subsolver(subpb)

  return LMSolver{T, typeof(ψ), V, typeof(subsolver), typeof(subpb)}(
    xk,
    ∇fk,
    mν∇fk,
    Fk,
    Fkn,
    Jv,
    Jtv,
    ψ,
    xkn,
    s,
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
    LM(reg_nls; kwargs...)

A Levenberg-Marquardt method for the problem

    min ½ ‖F(x)‖² + h(x)

where F: ℝⁿ → ℝᵐ and its Jacobian J are Lipschitz continuous and h: ℝⁿ → ℝ is
lower semi-continuous, proper and prox-bounded.

At each iteration, a step s is computed as an approximate solution of

    min  ½ ‖J(x) s + F(x)‖² + ½ σ ‖s‖² + ψ(s; x)

where F(x) and J(x) are the residual and its Jacobian at x, respectively, ψ(s; xₖ) is either h(xₖ + s) or an approximation of h(xₖ + s),
‖⋅‖ is the ℓ₂ norm and σₖ > 0 is the regularization parameter.

For advanced usage, first define a solver "LMSolver" to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = LMSolver(reg_nls; subsolver = R2Solver, m_monotone = 1)
    solve!(solver, reg_nls)

    stats = RegularizedExecutionStats(reg_nls)
    solve!(solver, reg_nls, stats)
  
# Arguments
* `reg_nls::AbstractRegularizedNLPModel{T, V}`: the problem to solve, see `RegularizedProblems.jl`, `NLPModels.jl`.

# Keyword arguments
- `x::V = nlp.meta.x0`: the initial guess;
- `nonlinear::Bool = true`: whether the function `F` is nonlinear or not.
- `atol::T = √eps(T)`: absolute tolerance;
- `rtol::T = √eps(T)`: relative tolerance;
- `neg_tol::T = zero(T): negative tolerance;
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
- `m_monotone::Int = 1`: monotonicity parameter. By default, LM is monotone but the non-monotone variant will be used if `m_monotone > 1`;
- `subsolver = R2Solver`: the solver used to solve the subproblems.
- `sub_kwargs::NamedTuple = NamedTuple()`: a named tuple containing the keyword arguments to be sent to the subsolver. The solver will fail if invalid keyword arguments are provided to the subsolver. For example, if the subsolver is `R2Solver`, you can pass `sub_kwargs = (max_iter = 100, σmin = 1e-6,)`.

The algorithm stops when `‖sᶜᵖ‖/ν < atol + rtol*‖s₀ᶜᵖ‖/ν ` where sᶜᵖ ∈ argminₛ f(xₖ) + ∇f(xₖ)ᵀs + ψ(s; xₖ) ½ ν⁻¹ ‖s‖².

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
$(callback_docstring)
"""
function LM(nls::AbstractNLSModel, h::H, options::ROSolverOptions; kwargs...) where {H}
  kwargs_dict = Dict(kwargs...)
  selected = pop!(kwargs_dict, :selected, 1:(nls.meta.nvar))
  x0 = pop!(kwargs_dict, :x0, nls.meta.x0)
  reg_nls = RegularizedNLPModel(nls, h, selected)
  return LM(
    reg_nls;
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
    γ = options.γ,
    kwargs_dict...,
  )
end

function LM(reg_nls::AbstractRegularizedNLPModel; kwargs...)
  kwargs_dict = Dict(kwargs...)
  subsolver = pop!(kwargs_dict, :subsolver, R2Solver)
  m_monotone = pop!(kwargs_dict, :m_monotone, 1)
  solver = LMSolver(reg_nls, subsolver = subsolver, m_monotone = m_monotone)
  stats = RegularizedExecutionStats(reg_nls)
  solve!(solver, reg_nls, stats; kwargs_dict...)
  return stats
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
  sub_kwargs::NamedTuple = NamedTuple(),
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
  Jv = solver.Jv
  Jtv = solver.Jtv
  ∇fk = solver.∇fk
  mν∇fk = solver.mν∇fk
  ψ = solver.ψ
  xkn = solver.xkn
  s = solver.s
  m_fh_hist = solver.m_fh_hist .= T(-Inf)
  has_bnds = solver.has_bnds

  m_monotone = length(m_fh_hist) + 1

  if has_bnds
    l_bound, u_bound = solver.l_bound, solver.u_bound
    l_bound_m_x, u_bound_m_x = solver.l_bound_m_x, solver.u_bound_m_x
    update_bounds!(l_bound_m_x, u_bound_m_x, l_bound, u_bound, xk)
    set_bounds!(ψ, l_bound_m_x, u_bound_m_x)
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
      [:outer, :inner, :fx, :hx, :norm_s_cauchydν, :ρ, :σ, :normx, :norms, :normJ, :arrow],
      [Int, Int, T, T, T, T, T, T, T, T, Char],
      hdr_override = Dict{Symbol, String}(
        :fx => "f(x)",
        :hx => "h(x)",
        :norm_s_cauchydν => "‖sᶜᵖ‖/ν",
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
  local prox_evals::Int = 0

  residual!(nls, xk, Fk)
  jtprod_residual!(nls, xk, Fk, ∇fk)
  fk = dot(Fk, Fk) / 2

  σmax, found_σ = opnorm(solver.subpb.model.J)
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
  set_solver_specific!(stats, :prox_evals, prox_evals + 1)
  m_monotone > 1 && (m_fh_hist[stats.iter % (m_monotone - 1) + 1] = fk + hk)

  mk = let ψ = ψ, solver = solver
    d -> obj(solver.subpb.model, d, skip_sigma = true) + ψ(d)
  end

  prox!(s, ψ, mν∇fk, ν)
  norm_s_cauchy = norm(s)
  norm_s_cauchydν = norm_s_cauchy / ν
  
  atol += rtol * norm_s_cauchydν # make stopping test absolute and relative

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
    sub_atol = stats.iter == 0 ? 1.0e-3 : min(norm_s_cauchydν ^ (1.5), norm_s_cauchydν * 1e-3)
    solver.subpb.model.σ = σk
    isa(solver.subsolver, R2DHSolver) && (solver.subsolver.D.d[1] = 1/ν)
    if isa(solver.subsolver, R2Solver) #FIXME
      solve!(
        solver.subsolver,
        solver.subpb,
        solver.substats;
        x = s,
        atol = sub_atol,
        ν = ν,
        sub_kwargs...,
      )
    else
      solve!(
        solver.subsolver,
        solver.subpb,
        solver.substats;
        x = s,
        atol = sub_atol,
        σk = σk, #FIXME
        sub_kwargs...,
      )
    end

    prox_evals += solver.substats.iter
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

    fhmax = m_monotone > 1 ? maximum(m_fh_hist) : fk + hk
    Δobj = fhmax - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
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
          (η2 ≤ ρk < Inf) ? '↘' : (ρk < η1 ? '↗' : '='),
        ],
        colsep = 1,
      )

    if η1 ≤ ρk < Inf
      xk .= xkn

      if has_bnds
        update_bounds!(l_bound_m_x, u_bound_m_x, l_bound, u_bound, xk)
        set_bounds!(ψ, l_bound_m_x, u_bound_m_x)
      end

      #update functions
      Fk .= Fkn
      fk = fkn
      hk = hkn

      # update gradient & Hessian
      shift!(ψ, xk)
      jtprod_residual!(nls, xk, Fk, ∇fk)

      # update opnorm if not linear least squares
      if nonlinear == true
        σmax, found_σ = opnorm(solver.subpb.model.J)
        found_σ || error("operator norm computation failed")
      end
    end

    if η2 ≤ ρk < Inf
      σk = max(σk / γ, σmin)
    end

    if ρk < η1 || ρk == Inf
      σk = σk * γ
    end

    m_monotone > 1 && (m_fh_hist[stats.iter % (m_monotone - 1) + 1] = fk + hk)

    set_objective!(stats, fk + hk)
    set_solver_specific!(stats, :smooth_obj, fk)
    set_solver_specific!(stats, :nonsmooth_obj, hk)
    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)
    set_solver_specific!(stats, :prox_evals, prox_evals + 1)

    ν = θ / (σmax^2 + σk) # ‖J'J + σₖ I‖ = ‖J‖² + σₖ

    @. mν∇fk = - ν * ∇fk
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
    @info log_row(
      Any[stats.iter, 0, fk, hk, norm_s_cauchydν, ρk, σk, norm(xk), norm(s), 1 / ν, ""],
      colsep = 1,
    )
    @info "LM: terminating with ‖sᶜᵖ‖/ν = $(norm_s_cauchydν)"
  end

  set_solution!(stats, xk)
  set_residuals!(stats, zero(T), norm_s_cauchydν)
  return stats
end
