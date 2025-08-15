export iR2, iR2Solver, solve!

import SolverCore.solve!

mutable struct iR2Solver{
  R <: Real,
  G <: Union{ShiftedProximableFunction, InexactShiftedProximableFunction, Nothing},
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

# !!!!!! Not used anywhere !!!!!
# function iR2Solver(
#   x0::S,
#   options::ROSolverOptions,
#   l_bound::S,
#   u_bound::S;
#   ψ = nothing,
# ) where {R <: Real, S <: AbstractVector{R}}
#   maxIter = options.maxIter
#   xk = similar(x0)
#   ∇fk = similar(x0)
#   mν∇fk = similar(x0)
#   xkn = similar(x0)
#   s = zero(x0)
#   has_bnds = any(l_bound .!= R(-Inf)) || any(u_bound .!= R(Inf))
#   if has_bnds
#     l_bound_m_x = similar(xk)
#     u_bound_m_x = similar(xk)
#   else
#     l_bound_m_x = similar(xk, 0)
#     u_bound_m_x = similar(xk, 0)
#   end
#   Fobj_hist = zeros(R, maxIter + 2)
#   Hobj_hist = zeros(R, maxIter + 2)
#   Complex_hist = zeros(Int, maxIter + 2)
#   dualGap = options.dualGap
#   κξ = options.κξ
#   mk1 = options.mk1
#   shift = similar(x0)
#   s_k_unshifted = similar(x0)
#   callback_pointer = options.callback_pointer
#   context = AlgorithmContextCallback(
#     dualGap = options.dualGap,
#     κξ = options.κξ,
#     shift = ψ.xk + ψ.sj,
#     s_k_unshifted = s_k_unshifted,
#     mk = ModelFunction(similar(x0), ψ),
#   )
#   return iR2Solver(
#     xk,
#     ∇fk,
#     mν∇fk,
#     ψ,
#     xkn,
#     s,
#     has_bnds,
#     l_bound,
#     u_bound,
#     l_bound_m_x,
#     u_bound_m_x,
#     Fobj_hist,
#     Hobj_hist,
#     Complex_hist,
#     callback_pointer,
#     context,
#   )
# end

function iR2Solver(reg_nlp::AbstractRegularizedNLPModel{T, V}; max_iter::Int = 10000) where {T, V}
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

  ψ = shifted(reg_nlp.h, xk)
  # has_bnds ? shifted(reg_nlp.h, xk, l_bound_m_x, u_bound_m_x, reg_nlp.selected) :
  # shifted(reg_nlp.h, xk)

  return iR2Solver(
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
    iR2(reg_nlp; kwargs…)

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

The algorithm stops either when `√(ξₖ/νₖ) < atol + rtol*√(ξ₀/ν₀) ` or `ξₖ < 0` and `√(-ξₖ/νₖ) < neg_tol` where ξₖ := f(xₖ) + h(xₖ) - φ(sₖ; xₖ) - ψ(sₖ; xₖ), and √(ξₖ/νₖ) is a stationarity measure.

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
The callback is called at each iteration.
The expected signature of the callback is `callback(nlp, solver, stats)`, and its output is ignored.
Changing any of the input arguments will affect the subsequent iterations.
In particular, setting `stats.status = :user` will stop the algorithm.
All relevant information should be available in `nlp` and `solver`.
Notably, you can access, and modify, the following:
- `solver.xk`: current iterate;
- `solver.∇fk`: current gradient;
- `stats`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
  - `stats.iter`: current iteration counter;
  - `stats.objective`: current objective function value;
  - `stats.solver_specific[:smooth_obj]`: current value of the smooth part of the objective function
  - `stats.solver_specific[:nonsmooth_obj]`: current value of the nonsmooth part of the objective function
  - `stats.status`: current status of the algorithm. Should be `:unknown` unless the algorithm has attained a stopping criterion. Changing this to anything will stop the algorithm, but you should use `:user` to properly indicate the intention.
  - `stats.elapsed_time`: elapsed time in seconds.
"""
function iR2(
  nlp::AbstractNLPModel{R, V},
  h,
  options::ROSolverOptions{R};
  kwargs...,
) where {R <: Real, V}
  kwargs_dict = Dict(kwargs...)
  selected = pop!(kwargs_dict, :selected, 1:(nlp.meta.nvar))
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  reg_nlp = RegularizedNLPModel(nlp, h, selected)
  return iR2(
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
  )
end

function iR2(
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
  stats = iR2(
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
  )
  outdict = Dict(
    :Fhist => stats.solver_specific[:Fhist],
    :Hhist => stats.solver_specific[:Hhist],
    :Chist => stats.solver_specific[:SubsolverCounter],
    :ItersProx => stats.solver_specific[:ItersProx],
    :NonSmooth => h,
    :status => stats.status,
    :fk => stats.solver_specific[:smooth_obj],
    :hk => stats.solver_specific[:nonsmooth_obj],
    :ξ => stats.solver_specific[:xi],
    :elapsed_time => stats.elapsed_time,
  )
  return stats.solution, stats.iter, outdict
end

function iR2(
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
  stats = iR2(
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
  )
  outdict = Dict(
    :Fhist => stats.solver_specific[:Fhist],
    :Hhist => stats.solver_specific[:Hhist],
    :Chist => stats.solver_specific[:SubsolverCounter],
    :ItersProx => stats.solver_specific[:ItersProx],
    :NonSmooth => h,
    :status => stats.status,
    :fk => stats.solver_specific[:smooth_obj],
    :hk => stats.solver_specific[:nonsmooth_obj],
    :ξ => stats.solver_specific[:xi],
    :elapsed_time => stats.elapsed_time,
  )
  return stats.solution, stats.iter, outdict
end

function iR2(reg_nlp::AbstractRegularizedNLPModel; kwargs...)

  # if h has exact prox, switch to R2 
  # if !(shifted(reg_nlp.h, reg_nlp.model.meta.x0) isa InexactShiftedProximableFunction)
  #   @warn "h has exact prox, switching to R2"
  #   return R2(reg_nlp; kwargs...)
  # end

  kwargs_dict = Dict(kwargs...)
  max_iter = pop!(kwargs_dict, :max_iter, 10000)

  solver = iR2Solver(reg_nlp, max_iter = max_iter)
  stats = GenericExecutionStats(reg_nlp.model)
  cb =
    (nlp, solver, stats) -> begin
      solver.Fobj_hist[stats.iter + 1] = stats.solver_specific[:smooth_obj]
      solver.Hobj_hist[stats.iter + 1] = stats.solver_specific[:nonsmooth_obj]
      solver.Complex_hist[stats.iter + 1] += 1
    end

  solve!(solver, reg_nlp, stats; callback = cb, max_iter = max_iter, kwargs_dict...)
  set_solver_specific!(stats, :Fhist, solver.Fobj_hist[1:(stats.iter + 1)])
  set_solver_specific!(stats, :Hhist, solver.Hobj_hist[1:(stats.iter + 1)])
  set_solver_specific!(stats, :SubsolverCounter, solver.Complex_hist[1:(stats.iter + 1)])
  return stats
end

function SolverCore.solve!(
  solver::iR2Solver{T},
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
) where {T, V}
  reset!(stats)

  # Retrieve workspace
  selected = reg_nlp.selected
  h = reg_nlp.h
  nlp = reg_nlp.model

  xk = solver.xk .= x

  # Make sure ψ has the correct shift 
  shift!(solver.ψ, xk)

  # ∇fk = solver.∇fk
  mν∇fk = solver.mν∇fk
  ψ = solver.ψ
  κξ = one(T)
  dualGap = zero(T)
  if ψ isa InexactShiftedProximableFunction
    let ctx = ψ.h.context
      κξ = ctx.κξ
      dualGap = ctx.dualGap
    end
  end

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
    verbose > 0 && @info "iR2: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, xk, one(eltype(xk)))
    hk = @views h(xk[selected])
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "R2: found point where h has value" hk
  end
  improper = (hk == -Inf)

  if verbose > 0
    @info log_header(
      [:iter, :fx, :hx, :xi, :ρ, :σ, :normx, :norms, :dualgap, :arrow],
      [Int, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Char],
      hdr_override = Dict{Symbol, String}(   # TODO: Add this as constant dict elsewhere
        :iter => "iter",
        :fx => "f(x)",
        :hx => "h(x)",
        :xi => "√(ξ/ν)",
        :ρ => "ρ",
        :σ => "σ",
        :normx => "‖x‖",
        :norms => "‖s‖",
        :objgap => "dualGap_iR2",
        :arrow => " ",
      ),
      colsep = 1,
    )
  end

  local ξ::T
  local ρk::T
  σk = max(1 / ν, σmin)
  ν = 1 / σk
  sqrt_ξ_νInv = one(T)

  fk = obj(nlp, xk)
  grad!(nlp, xk, solver.∇fk)
  @. mν∇fk = -ν * solver.∇fk

  set_iter!(stats, 0)
  start_time = time()
  set_time!(stats, 0.0)
  set_objective!(stats, fk + hk)
  set_solver_specific!(stats, :smooth_obj, fk)
  set_solver_specific!(stats, :nonsmooth_obj, hk)

  φk(d) = dot(solver.∇fk, d)
  mk(d)::T = φk(d) + ψ(d)::T

  update_prox_context!(solver, stats, ψ)
  prox!(s, ψ, mν∇fk, ν)

  mks = mk(s)

  ξ = hk - mks + max(1, abs(hk)) * 10 * eps()

  sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / ν) : sqrt(-ξ / ν)
  atol += rtol * sqrt_ξ_νInv # make stopping test absolute and relative

  solved = (ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ ≥ 0 && sqrt_ξ_νInv ≤ atol * √κξ)
  (ξ < 0 && sqrt_ξ_νInv > neg_tol) && error(
    "iR2: first prox-gradient step should produce a decrease but ξ = $(ξ) and sqrt_ξ_νInv = $(sqrt_ξ_νInv) > $(neg_tol)",
  )

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

    Δobj = (fk + hk) - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
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
          σk,
          norm(xk),
          norm(s),
          dualGap,
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
      fk = fkn
      hk = hkn
      grad!(nlp, xk, solver.∇fk)
      shift!(ψ, xk)
    end

    if η2 ≤ ρk < Inf
      σk = max(σk / γ, σmin)
    end
    if ρk < η1 || ρk == Inf
      σk = σk * γ
    end

    ν = 1 / σk
    @. mν∇fk = -ν * solver.∇fk

    set_objective!(stats, fk + hk)
    set_solver_specific!(stats, :smooth_obj, fk)
    set_solver_specific!(stats, :nonsmooth_obj, hk)
    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)

    # prepare callback context and pointer to callback function
    update_prox_context!(solver, stats, ψ)
    prox!(s, ψ, mν∇fk, ν)
    mks = mk(s)

    ξ = hk - mks + max(1, abs(hk)) * 10 * eps()

    sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / ν) : sqrt(-ξ / ν)
    solved = (ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ ≥ 0 && sqrt_ξ_νInv ≤ atol * √κξ)
    (ξ < 0 && sqrt_ξ_νInv > neg_tol) && error(
      "iR2: prox-gradient step should produce a decrease but ξ = $(ξ) and sqrt_ξ_νInv = $(sqrt_ξ_νInv) > $(neg_tol)",
    )

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
        fk,
        hk,
        sqrt_ξ_νInv,
        ρk,
        σk,
        norm(xk),
        norm(s),
        dualGap,
        (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "="),
      ],
      colsep = 1,
    )
    @info "iR2: terminating with √(ξ/ν) = $(sqrt_ξ_νInv)"
  end

  if ψ isa InexactShiftedProximableFunction
    set_solver_specific!(stats, :ItersProx, ψ.h.context.prox_stats[3])
  end
  set_solution!(stats, xk)
  return stats
end