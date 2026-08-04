export NMPG, NMPGSolver, solve!

import SolverCore.solve!

const NMPG_header = Dict{Symbol, String}(
  :iter => "iter",
  :fx => "f(x)",
  :hx => "h(x)",
  :xi => "√(ξ/ν)",
  :σ => "σ",
  :normx => "‖x‖",
  :norms => "‖s‖",
)

mutable struct NMPGSolver{R <: Real, G <: ShiftedProximableFunction, S <: AbstractVector{R}} <:
               AbstractOptimizationSolver
  xk::S
  ∇fk::S
  mν∇fk::S
  ψ::G
  xkn::S
  s::S
end

function NMPGSolver(reg_nlp::AbstractRegularizedNLPModel{T, V}) where {T, V}
  x0 = reg_nlp.model.meta.x0
  xk = similar(x0)
  ∇fk = similar(x0)
  mν∇fk = similar(x0)
  xkn = similar(x0)
  s = zero(x0)
  ψ = shifted(reg_nlp.h, xk)
  return NMPGSolver(xk, ∇fk, mν∇fk, ψ, xkn, s)
end

@deprecate NMPG(nlp, h, options::ROSolverOptions; kwargs...) NMPG(nlp, h; kwargs...)

function NMPG(
  nlp::AbstractNLPModel{R, V},
  h,
  options::ROSolverOptions{R};
  kwargs...,
) where {R <: Real, V}
  kwargs_dict = Dict(kwargs...)
  selected = pop!(kwargs_dict, :selected, 1:(nlp.meta.nvar))
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  reg_nlp = RegularizedNLPModel(nlp, h, selected)
  return NMPG(
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

function NMPG(
  nlp::AbstractNLPModel{R, V},
  h;
  selected::AbstractVector{<:Integer} = 1:(nlp.meta.nvar),
  kwargs...,
) where {R, V}
  reg_nlp = RegularizedNLPModel(nlp, h, selected)
  return NMPG(reg_nlp; kwargs...)
end

"""
    NMPG(reg_nlp; kwargs…)

A proximal-gradient method for the problem

    min_x f(x) + h(x)

where f: ℝⁿ → ℝ has a continuous gradient, and h: ℝⁿ → ℝ is proper, lower semi-continuous, and prox-bounded.

About each iterate x, an update x₊ is computed as a solution of

    min_z  f(x) + <∇f(x), z-x> + ½ σ ‖z-x‖² + h(z)

where σ > 0 is the regularization parameter (inverse of the stepsize).

For advanced usage, first define a solver "NMPGSolver" to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = NMPGSolver(reg_nlp)
    solve!(solver, reg_nlp)

    stats = RegularizedExecutionStats(reg_nlp)
    solver = NMPGSolver(reg_nlp)
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
- `η1::T = √√eps(T)`: successful iteration threshold;
- `η2::T = T(0.9)`: very successful iteration threshold;
- `ν::T = eps(T)^(1 / 5)`: multiplicative inverse of the regularization parameter: ν = 1/σ;
- `γ::T = T(3)`: regularization parameter multiplier, σ := σ/γ when the iteration is very successful and σ := σγ when the iteration is unsuccessful.
- `w_monotone::T = T(1.0)`: monotonicity parameter. By default, NMPG is monotone but non-monotone if `w_monotone ∈ (0,1)`;
- `compute_obj::Bool = true`: (advanced) whether `f(x₀)` should be computed or not. If set to false, then the value is retrieved from `stats.solver_specific[:smooth_obj]`;
- `compute_grad::Bool = true`: (advanced) whether `∇f(x₀)` should be computed or not. If set to false, then the value is retrieved from `solver.∇fk`;
- `compute_res::Bool = true`: (advanced) whether `norm_res` should be computed or not. If set to false, then the value is retrieved from `solver.∇fk`;
- `check_res_only::Bool = false`: (advanced) whether termination should be based on `norm_res` alone or not. If set to true, then only `norm_res < tol` is checked;
- `spectral_stepsize::Bool = false`: (advanced) whether to use adaptive Barzilai-Borwein stepsize or not. If set to true, then `compute_res` is activated too.

The algorithm stops based on several criteria:
- if `norm_res < tol`
- or if `√(ξₖ/νₖ) < tol`
- or if `ξₖ < 0` and `√(-ξₖ/νₖ) < neg_tol`
where `tol := atol + rtol*√(ξ₀/ν₀)`
`ξₖ := h(xₖ) - <∇f(xₖ),sₖ> - ψ(sₖ; xₖ)`
and `norm_res := ||sₖ ./ νₖ + ∇fₖ₊₁ - ∇fₖ||`.

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
$(callback_docstring)
"""
function NMPG(reg_nlp::AbstractRegularizedNLPModel; kwargs...)
  solver = NMPGSolver(reg_nlp)
  stats = RegularizedExecutionStats(reg_nlp)
  solve!(solver, reg_nlp, stats; kwargs...)
  return stats
end

function SolverCore.solve!(
  solver::NMPGSolver{T},
  reg_nlp::AbstractRegularizedNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = reg_nlp.model.meta.x0,
  atol::T = eps(T)^(1 / 3),
  rtol::T = eps(T)^(1 / 3),
  neg_tol::T = eps(T)^(1 / 4),
  verbose::Int = 0,
  max_iter::Int = 10_000,
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  σmin::T = √eps(T),
  σmax::T = 1/√eps(T),
  η1::T = √√eps(T),
  η2::T = T(0.9),
  ν::T = eps(T)^(1 / 5),
  γ::T = T(3),
  w_monotone::T = T(1),
  compute_obj::Bool = true,
  compute_grad::Bool = true,
  compute_res::Bool = true,
  check_res_only::Bool = false,
  spectral_stepsize::Bool = false,
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

  # initialize parameters
  improper = false
  hk = @views h(xk[selected])
  if hk == Inf
    verbose > 0 && @info "NMPG: finding initial guess where nonsmooth term is finite"
    gmm = 100 * eps(eltype(xk))
    prox!(xk, h, xk, gmm)
    hk = @views h(xk[selected])
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "NMPG: found point where h has value" hk
  end
  improper = (hk == -Inf)

  if verbose > 0
    @info log_header(
      [:iter, :fx, :hx, :xi, :σ, :normx, :norms],
      [Int, Float64, Float64, Float64, Float64, Float64, Float64],
      hdr_override = NMPG_header,
      colsep = 1,
    )
  end

  local ξ::T
  local sqrt_ξ_νInv::T
  local σk::T
  local fhmerit::T
  local epstol::T
  local norm_s::T
  local norm_res::T
  local is_monotone::Bool

  (!compute_res && check_res_only) &&
    error("NMPG: must compute residual to check termination with it")
  if spectral_stepsize
    compute_res = true
    if !compute_res
      @warn "Activating compute_res"
    end
  end

  # TODO preallocate in solver
  if compute_res
    ∇fkn = similar(x)
    res = similar(x)
  end

  σk = max(1 / ν, σmin)
  ν = 1 / σk
  sqrt_ξ_νInv = one(T)
  is_monotone = w_monotone == T(1)

  fk = compute_obj ? obj(nlp, xk) : stats.solver_specific[:smooth_obj]
  compute_grad && grad!(nlp, xk, ∇fk)
  @. mν∇fk = -ν * ∇fk

  fhmerit = fk + hk # initialize merit = objective
  epstol = max(1, abs(fhmerit)) * 10 * eps()

  set_iter!(stats, 0)
  start_time = time()
  set_time!(stats, 0.0)
  set_objective!(stats, fk+hk)
  set_solver_specific!(stats, :smooth_obj, fk)
  set_solver_specific!(stats, :nonsmooth_obj, hk)
  set_solver_specific!(stats, :sigma, σk)

  prox!(s, ψ, mν∇fk, ν)
  mks = dot(∇fk, s) + ψ(s)

  ξ = hk - mks + epstol
  sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / ν) : sqrt(-ξ / ν)
  tol = atol + rtol * sqrt_ξ_νInv # make stopping test absolute and relative
  norm_res = 2*tol
  if check_res_only
    solved = norm_res <= tol
  else
    solved = (ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ ≥ 0 && sqrt_ξ_νInv ≤ tol) || (norm_res <= tol)
  end
  (ξ < 0 && sqrt_ξ_νInv > neg_tol) &&
    error("NMPG: prox-gradient step should produce a decrease but ξ = $(ξ)")

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

  callback(reg_nlp, solver, stats)

  done = stats.status != :unknown

  while !done

    # Update xk, sigma_k
    xkn .= xk .+ s
    fkn = obj(nlp, xkn)
    hkn = @views h(xkn[selected])
    improper = (hkn == -Inf)
    norm_s = norm(s, 2)

    verbose > 0 &&
      stats.iter % verbose == 0 &&
      @info log_row(Any[stats.iter, fk, hk, sqrt_ξ_νInv, σk, norm(xk), norm_s], colsep = 1)

    # step acceptance
    if (fkn + hkn <= fhmerit - η1*ξ + epstol) || (fkn + hkn <= fhmerit - (η1/ν)*norm_s^2 + epstol)
      xk .= xkn
      fk = fkn
      hk = hkn
      if compute_res
        grad!(nlp, xk, ∇fkn)
        res .= s ./ ν + ∇fkn - ∇fk
        norm_res = norm(res, 2)
        if spectral_stepsize
          res .= ∇fkn - ∇fk # for spectral stepsize only
        end
        ∇fk .= ∇fkn
      else
        grad!(nlp, xk, ∇fk)
      end
      shift!(ψ, xk)
      set_step_status!(stats, :accepted)
      # adapt stepsize
      if spectral_stepsize
        σk = LinearAlgebra.dot(s, res) / LinearAlgebra.dot(s, s)
        σk = max(σmin, min(σk, σmax))
      elseif (fkn + hkn <= fhmerit - η2*ξ + epstol) # if very successful
        σk = max(σmin, σk / γ)
      end
      # update merit
      fhmerit = if is_monotone
        fk+hk
      else
        w_monotone*(fk+hk)+(1-w_monotone)*fhmerit
      end
      epstol = max(1, abs(fhmerit)) * 10 * eps()
    else
      set_step_status!(stats, :rejected)
      # backtrack stepsize
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
    norm_s = norm(s, 2)
    mks = dot(∇fk, s) + ψ(s)

    ξ = hk - mks + epstol
    sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / ν) : sqrt(-ξ / ν)
    if check_res_only
      solved = norm_res <= tol
    else
      solved = (ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ ≥ 0 && sqrt_ξ_νInv ≤ tol) || (norm_res <= tol)
    end
    (ξ < 0 && sqrt_ξ_νInv > neg_tol) &&
      error("NMPG: prox-gradient step should produce a decrease but ξ = $(ξ)")

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

    callback(reg_nlp, solver, stats)

    done = stats.status != :unknown
  end

  if verbose > 0 && stats.status == :first_order
    @info log_row(Any[stats.iter, fk, hk, sqrt_ξ_νInv, σk, norm(xk), norm_s], colsep = 1)
    if compute_res
      if sqrt_ξ_νInv < norm_res
        @info "NMPG: terminating with √(ξ/ν) = $(sqrt_ξ_νInv) and ||res|| = $(norm_res)"
      else
        @info "NMPG: terminating with ||res|| = $(norm_res) and √(ξ/ν) = $(sqrt_ξ_νInv)"
      end
    else
      @info "NMPG: terminating with √(ξ/ν) = $(sqrt_ξ_νInv)"
    end
  end

  set_solution!(stats, xk)
  if check_res_only
    set_residuals!(stats, zero(eltype(xk)), norm_res)
  else
    set_residuals!(stats, zero(eltype(xk)), min(sqrt_ξ_νInv, norm_res))
  end
  return stats
end
