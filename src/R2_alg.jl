export R2, R2Solver, solve!

import SolverCore.solve!

mutable struct R2Solver{
  R <: Real,
  G <: Union{ShiftedProximableFunction, Nothing},
  S <: AbstractVector{R},
} <: AbstractOptimizationSolver
  xk::S
  ∇fk::S
  mν∇fk::S
  ψ::G
  xkn::S
  s::S
  has_bnds::Bool
  l_bound::S
  u_bound::S
  l_bound_m_x::S
  u_bound_m_x::S
  Fobj_hist::Vector{R}
  Hobj_hist::Vector{R}
  Complex_hist::Vector{Int}
end

function R2Solver(
  x0::S,
  options::ROSolverOptions,
  l_bound::S,
  u_bound::S;
  ψ = nothing,
) where {R <: Real, S <: AbstractVector{R}}
  maxIter = options.maxIter
  xk = similar(x0)
  ∇fk = similar(x0)
  mν∇fk = similar(x0)
  xkn = similar(x0)
  s = zero(x0)
  has_bnds = any(l_bound .!= R(-Inf)) || any(u_bound .!= R(Inf))
  if has_bnds
    l_bound_m_x = similar(xk)
    u_bound_m_x = similar(xk)
  else
    l_bound_m_x = similar(xk, 0)
    u_bound_m_x = similar(xk, 0)
  end
  Fobj_hist = zeros(R, maxIter + 2)
  Hobj_hist = zeros(R, maxIter + 2)
  Complex_hist = zeros(Int, maxIter + 2)
  return R2Solver(
    xk,
    ∇fk,
    mν∇fk,
    ψ,
    xkn,
    s,
    has_bnds,
    l_bound,
    u_bound,
    l_bound_m_x,
    u_bound_m_x,
    Fobj_hist,
    Hobj_hist,
    Complex_hist,
  )
end

function R2Solver(reg_nlp::AbstractRegularizedNLPModel{T, V}; max_iter::Int = 10000) where {T, V}
  x0 = reg_nlp.model.meta.x0
  l_bound = reg_nlp.model.meta.lvar
  u_bound = reg_nlp.model.meta.uvar

  xk = similar(x0)
  ∇fk = similar(x0)
  mν∇fk = similar(x0)
  xkn = similar(x0)
  s = zero(x0)
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
  Fobj_hist = zeros(T, max_iter + 2)
  Hobj_hist = zeros(T, max_iter + 2)
  Complex_hist = zeros(Int, max_iter + 2)

  ψ =
    has_bnds ? shifted(reg_nlp.h, xk, l_bound_m_x, u_bound_m_x, reg_nlp.selected) :
    shifted(reg_nlp.h, xk)
  return R2Solver(
    xk,
    ∇fk,
    mν∇fk,
    ψ,
    xkn,
    s,
    has_bnds,
    l_bound,
    u_bound,
    l_bound_m_x,
    u_bound_m_x,
    Fobj_hist,
    Hobj_hist,
    Complex_hist,
  )
end

"""
    R2(reg_nlp; kwargs…)

A first-order quadratic regularization method for the problem

    min f(x) + h(x)

where f: ℝⁿ → ℝ has a Lipschitz-continuous gradient, and h: ℝⁿ → ℝ is
lower semi-continuous, proper and prox-bounded.

About each iterate xₖ, a step sₖ is computed as a solution of

    min  φ(s; xₖ) + ½ σₖ ‖s‖² + ψ(s; xₖ)

where φ(s ; xₖ) = f(xₖ) + ∇f(xₖ)ᵀs is the Taylor linear approximation of f about xₖ,
ψ(s; xₖ) is either h(xₖ + s) or an approximation of h(xₖ + s), ‖⋅‖ is a user-defined norm and σₖ > 0 is the regularization parameter.

For advanced usage, first define a solver "R2Solver" to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = R2Solver(reg_nlp)
    solve!(solver, reg_nlp)

    stats = RegularizedExecutionStats(reg_nlp)
    solver = R2Solver(reg_nlp)
    solve!(solver, reg_nlp, stats)

# Arguments
* `reg_nlp::AbstractRegularizedNLPModel{T, V}`: the problem to solve, see `RegularizedProblems.jl`, `NLPModels.jl`.

# Keyword arguments 
- `x::V = nlp.meta.x0`: the initial guess;
- `atol::T = √eps(T)`: absolute tolerance;
- `rtol::T = √eps(T)`: relative tolerance;
- `neg_tol::T = eps(T)^(1 / 4)`: negative tolerance
- `max_eval::Int = -1`: maximum number of evaluation of the objective function (negative number means unlimited);
- `max_time::Float64 = 30.0`: maximum time limit in seconds;
- `max_iter::Int = 10000`: maximum number of iterations;
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration;
- `σmin::T = eps(T)`: minimum value of the regularization parameter;
- `η1::T = √√eps(T)`: very successful iteration threshold;
- `η2::T = T(0.9)`: successful iteration threshold;
- `ν::T = eps(T)^(1 / 5)`: multiplicative inverse of the regularization parameter: ν = 1/σ;
- `γ::T = T(3)`: regularization parameter multiplier, σ := σ/γ when the iteration is very successful and σ := σγ when the iteration is unsuccessful.
- `compute_obj::Bool = true`: (advanced) whether `f(x₀)` should be computed or not. If set to false, then the value is retrieved from `stats.solver_specific[:smooth_obj]`;
- `compute_grad::Bool = true`: (advanced) whether `∇f(x₀)` should be computed or not. If set to false, then the value is retrieved from `solver.∇fk`;

The algorithm stops when `‖sᶜᵖ‖/ν < atol + rtol*‖s₀ᶜᵖ‖/ν ` where sᶜᵖ ∈ argminₛ f(xₖ) + ∇f(xₖ)ᵀs + ψ(s; xₖ) ½ ν⁻¹ ‖s‖².

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
$(callback_docstring)
"""
function R2(
  nlp::AbstractNLPModel{R, V},
  h,
  options::ROSolverOptions{R};
  kwargs...,
) where {R <: Real, V}
  kwargs_dict = Dict(kwargs...)
  selected = pop!(kwargs_dict, :selected, 1:(nlp.meta.nvar))
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  reg_nlp = RegularizedNLPModel(nlp, h, selected)
  return R2(
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
    γ = options.γ;
    kwargs_dict...,
  )
end

function R2(
  nlp::AbstractNLPModel{R, V},
  h;
  selected::AbstractVector{<:Integer} = 1:(nlp.meta.nvar),
  kwargs...,
) where {R, V}
  reg_nlp = RegularizedNLPModel(nlp, h, selected)
  return R2(reg_nlp; kwargs...)
end

function R2(
  f::F,
  ∇f!::G,
  h::H,
  options::ROSolverOptions{R},
  x0::AbstractVector{R};
  selected::AbstractVector{<:Integer} = 1:length(x0),
  kwargs...,
) where {F <: Function, G <: Function, H, R <: Real}
  nlp = FirstOrderModel(f, ∇f!, x0)
  reg_nlp = RegularizedNLPModel(nlp, h, selected)
  stats = R2(
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
    γ = options.γ;
    kwargs...,
  )
  outdict = Dict(
    :Fhist => stats.solver_specific[:Fhist],
    :Hhist => stats.solver_specific[:Hhist],
    :Chist => stats.solver_specific[:SubsolverCounter],
    :NonSmooth => h,
    :status => stats.status,
    :fk => stats.solver_specific[:smooth_obj],
    :hk => stats.solver_specific[:nonsmooth_obj],
    :ξ => stats.dual_feas,
    :elapsed_time => stats.elapsed_time,
  )
  return stats.solution, stats.iter, outdict
end

function R2(
  f::F,
  ∇f!::G,
  h::H,
  options::ROSolverOptions{R},
  x0::AbstractVector{R},
  l_bound::AbstractVector{R},
  u_bound::AbstractVector{R};
  selected::AbstractVector{<:Integer} = 1:length(x0),
  kwargs...,
) where {F <: Function, G <: Function, H, R <: Real}
  nlp = FirstOrderModel(f, ∇f!, x0, lcon = l_bound, ucon = u_bound)
  reg_nlp = RegularizedNLPModel(nlp, h, selected)
  stats = R2(
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
    γ = options.γ;
    kwargs...,
  )
  outdict = Dict(
    :Fhist => stats.solver_specific[:Fhist],
    :Hhist => stats.solver_specific[:Hhist],
    :Chist => stats.solver_specific[:SubsolverCounter],
    :NonSmooth => h,
    :status => stats.status,
    :fk => stats.solver_specific[:smooth_obj],
    :hk => stats.solver_specific[:nonsmooth_obj],
    :ξ => stats.solver_specific[:xi],
    :elapsed_time => stats.elapsed_time,
  )
  return stats.solution, stats.iter, outdict
end

function R2(reg_nlp::AbstractRegularizedNLPModel; kwargs...)
  kwargs_dict = Dict(kwargs...)
  max_iter = pop!(kwargs_dict, :max_iter, 10000)
  solver = R2Solver(reg_nlp, max_iter = max_iter)
  stats = GenericExecutionStats(reg_nlp.model) # TODO: change this to `stats = RegularizedExecutionStats(reg_nlp)` when FHist etc. is ruled out.
  cb = pop!(
    kwargs_dict,
    :callback,
    (nlp, solver, stats) -> begin
      solver.Fobj_hist[stats.iter + 1] = stats.solver_specific[:smooth_obj]
      solver.Hobj_hist[stats.iter + 1] = stats.solver_specific[:nonsmooth_obj]
      solver.Complex_hist[stats.iter + 1] += 1
    end,
  )
  solve!(solver, reg_nlp, stats; callback = cb, max_iter = max_iter, kwargs...)
  set_solver_specific!(stats, :Fhist, solver.Fobj_hist[1:(stats.iter + 1)])
  set_solver_specific!(stats, :Hhist, solver.Hobj_hist[1:(stats.iter + 1)])
  set_solver_specific!(stats, :SubsolverCounter, solver.Complex_hist[1:(stats.iter + 1)])
  return stats
end

function SolverCore.solve!(
  solver::R2Solver{T},
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
  σmin::T = eps(T),
  η1::T = √√eps(T),
  η2::T = T(0.9),
  ν::T = eps(T)^(1 / 5),
  γ::T = T(3),
  compute_obj::Bool = true,
  compute_grad::Bool = true,
) where {T, V}
  reset!(stats)

  # Retrieve workspace
  selected = reg_nlp.selected
  h = reg_nlp.h
  nlp = reg_nlp.model

  xk = solver.xk .= x

  # Make sure ψ has the correct shift 
  shift!(solver.ψ, xk)

  ∇fk = solver.∇fk
  mν∇fk = solver.mν∇fk
  ψ = solver.ψ
  xkn = solver.xkn
  s = solver.s
  has_bnds = solver.has_bnds
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
    verbose > 0 && @info "R2: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, xk, one(eltype(xk)))
    hk = @views h(xk[selected])
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "R2: found point where h has value" hk
  end
  improper = (hk == -Inf)

  if verbose > 0
    @info log_header(
      [:iter, :fx, :hx, :norm_s_cauchydν, :ρ, :σ, :normx, :norms, :arrow],
      [Int, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Char],
      hdr_override = Dict{Symbol, String}(   # TODO: Add this as constant dict elsewhere
        :iter => "iter",
        :fx => "f(x)",
        :hx => "h(x)",
        :norm_s_cauchydν => "‖sᶜᵖ‖/ν",
        :ρ => "ρ",
        :σ => "σ",
        :normx => "‖x‖",
        :norms => "‖s‖",
        :arrow => "R2",
      ),
      colsep = 1,
    )
  end

  local ξ::T
  local ρk::T
  σk = max(1 / ν, σmin)
  ν = 1 / σk
  sqrt_ξ_νInv = one(T)

  fk = compute_obj ? obj(nlp, xk) : stats.solver_specific[:smooth_obj]
  compute_grad && grad!(nlp, xk, ∇fk)
  @. mν∇fk = -ν * ∇fk

  set_iter!(stats, 0)
  start_time = time()
  set_time!(stats, 0.0)
  set_objective!(stats, fk + hk)
  set_solver_specific!(stats, :smooth_obj, fk)
  set_solver_specific!(stats, :nonsmooth_obj, hk)
  set_solver_specific!(stats, :sigma, σk)

  φk(d) = dot(∇fk, d)
  mk(d)::T = φk(d) + ψ(d)::T

  prox!(s, ψ, mν∇fk, ν)
  mks = mk(s)
  norm_s_cauchy = norm(s)
  norm_s_cauchydν = norm_s_cauchy / ν

  ξ = hk - mks + max(1, abs(hk)) * 10 * eps()

  atol += rtol * norm_s_cauchydν # make stopping test absolute and relative

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

    # Update xk, sigma_k
    xkn .= xk .+ s
    fkn = obj(nlp, xkn)
    hkn = @views h(xkn[selected])
    improper = (hkn == -Inf)

    Δobj = (fk + hk) - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ρk = Δobj / ξ

    verbose > 0 &&
      stats.iter % verbose == 0 &&
      @info log_row(
        Any[
          stats.iter,
          fk,
          hk,
          norm_s_cauchydν,
          ρk,
          σk,
          norm(xk),
          norm_s_cauchy,
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
      fk = fkn
      hk = hkn
      grad!(nlp, xk, ∇fk)
      shift!(ψ, xk)
    end

    if η2 ≤ ρk < Inf
      σk = max(σk / γ, σmin)
    end
    if ρk < η1 || ρk == Inf
      σk = σk * γ
    end

    ν = 1 / σk
    @. mν∇fk = -ν * ∇fk

    set_objective!(stats, fk + hk)
    set_solver_specific!(stats, :smooth_obj, fk)
    set_solver_specific!(stats, :nonsmooth_obj, hk)
    set_solver_specific!(stats, :sigma, σk)
    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)

    prox!(s, ψ, mν∇fk, ν)
    mks = mk(s)
    norm_s_cauchy = norm(s)
    norm_s_cauchydν = norm_s_cauchy / ν

    ξ = hk - mks + max(1, abs(hk)) * 10 * eps()
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
    @info log_row(Any[stats.iter, fk, hk, norm_s_cauchydν, ρk, σk, norm(xk), norm_s_cauchy, ""], colsep = 1)
    @info "R2: terminating with ‖sᶜᵖ‖/ν = $(norm_s_cauchydν)"
  end

  set_solution!(stats, xk)
  set_residuals!(stats, zero(eltype(xk)), norm_s_cauchy)
  return stats
end
