export R2

mutable struct R2Solver{R, S <: AbstractVector{R}} <: AbstractOptimizationSolver
  xk::S
  ∇fk::S
  mν∇fk::S
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
  u_bound::S,
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
  Fobj_hist = zeros(R, maxIter)
  Hobj_hist = zeros(R, maxIter)
  Complex_hist = zeros(Int, maxIter)
  return R2Solver(
    xk,
    ∇fk,
    mν∇fk,
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
    R2(nlp, h, options)
    R2(f, ∇f!, h, options, x0)

A first-order quadratic regularization method for the problem

    min f(x) + h(x)

where f: ℝⁿ → ℝ has a Lipschitz-continuous gradient, and h: ℝⁿ → ℝ is
lower semi-continuous, proper and prox-bounded.

About each iterate xₖ, a step sₖ is computed as a solution of

    min  φ(s; xₖ) + ½ σₖ ‖s‖² + ψ(s; xₖ)

where φ(s ; xₖ) = f(xₖ) + ∇f(xₖ)ᵀs is the Taylor linear approximation of f about xₖ,
ψ(s; xₖ) = h(xₖ + s), ‖⋅‖ is a user-defined norm and σₖ > 0 is the regularization parameter.

### Arguments

* `nlp::AbstractNLPModel`: a smooth optimization problem
* `h`: a regularizer such as those defined in ProximalOperators
* `options::ROSolverOptions`: a structure containing algorithmic parameters
* `x0::AbstractVector`: an initial guess (in the second calling form)

### Keyword Arguments

* `x0::AbstractVector`: an initial guess (in the first calling form: default = `nlp.meta.x0`)
* `selected::AbstractVector{<:Integer}`: (default `1:length(x0)`).

The objective and gradient of `nlp` will be accessed.

In the second form, instead of `nlp`, the user may pass in

* `f` a function such that `f(x)` returns the value of f at x
* `∇f!` a function to evaluate the gradient in place, i.e., such that `∇f!(g, x)` store ∇f(x) in `g`.

### Return values

* `xk`: the final iterate
* `Fobj_hist`: an array with the history of values of the smooth objective
* `Hobj_hist`: an array with the history of values of the nonsmooth objective
* `Complex_hist`: an array with the history of number of inner iterations.
"""

function R2(nlp::AbstractNLPModel, args...; kwargs)
  kwargs_dict = Dict(kwargs...)
  selected = pop!(kwargs_dict, :selected, 1:nlp.meta.nvar)
  reg_nlp = RegularizedNLPModel(nlp, args.h, selected)
  return R2(reg_nlp, args...; kwargs...)
end

function R2(reg_nlp::AbstractRegularizedNLPModel, args...; kwargs...)
  kwargs_dict = Dict(kwargs...)
  x0 = pop!(kwargs_dict, :x0, reg_nlp.model.meta.x0)
  xk, k, outdict = R2(
    x -> obj(reg_nlp.model, x),
    (g, x) -> grad!(reg_nlp.model, x, g),
    reg_nlp.h,
    args...,
    x0,
    nlp.meta.lvar,
    nlp.meta.uvar;
    selected = reg_nlp.selected,
    kwargs_dict...,
  )
  ξ = outdict[:ξ]
  stats = GenericExecutionStats(nlp)
  set_status!(stats, outdict[:status])
  set_solution!(stats, xk)
  set_objective!(stats, outdict[:fk] + outdict[:hk])
  set_residuals!(stats, zero(eltype(xk)), ξ)
  set_iter!(stats, k)
  set_time!(stats, outdict[:elapsed_time])
  set_solver_specific!(stats, :Fhist, outdict[:Fhist])
  set_solver_specific!(stats, :Hhist, outdict[:Hhist])
  set_solver_specific!(stats, :NonSmooth, outdict[:NonSmooth])
  set_solver_specific!(stats, :SubsolverCounter, outdict[:Chist])
  return stats
end

# method without bounds
function R2(
  f::F,
  ∇f!::G,
  h::H,
  options::ROSolverOptions{R},
  x0::AbstractVector{R};
  selected::AbstractVector{<:Integer} = 1:length(x0),
  kwargs...,
) where {F <: Function, G <: Function, H, R <: Real}
  start_time = time()
  elapsed_time = 0.0
  solver = R2Solver(x0, options, similar(x0, 0), similar(x0, 0))
  k, status, fk, hk, ξ = R2!(solver, f, ∇f!, h, options, x0; selected = selected)
  elapsed_time = time() - start_time
  outdict = Dict(
    :Fhist => solver.Fobj_hist[1:k],
    :Hhist => solver.Hobj_hist[1:k],
    :Chist => solver.Complex_hist[1:k],
    :NonSmooth => h,
    :status => status,
    :fk => fk,
    :hk => hk,
    :ξ => ξ,
    :elapsed_time => elapsed_time,
  )
  return solver.xk, k, outdict
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
  start_time = time()
  elapsed_time = 0.0
  solver = R2Solver(x0, options, l_bound, u_bound)
  k, status, fk, hk, ξ = R2!(solver, f, ∇f!, h, options, x0; selected = selected)
  elapsed_time = time() - start_time
  outdict = Dict(
    :Fhist => solver.Fobj_hist[1:k],
    :Hhist => solver.Hobj_hist[1:k],
    :Chist => solver.Complex_hist[1:k],
    :NonSmooth => h,
    :status => status,
    :fk => fk,
    :hk => hk,
    :ξ => ξ,
    :elapsed_time => elapsed_time,
  )
  return solver.xk, k, outdict
end

function R2!(
  solver::R2Solver{R},
  f::F,
  ∇f!::G,
  h::H,
  options::ROSolverOptions{R},
  x0::AbstractVector{R};
  selected::AbstractVector{<:Integer} = 1:length(x0),
) where {F <: Function, G <: Function, H, R <: Real}
  start_time = time()
  elapsed_time = 0.0
  ϵ = options.ϵa
  ϵr = options.ϵr
  neg_tol = options.neg_tol
  verbose = options.verbose
  maxIter = options.maxIter
  maxTime = options.maxTime
  σmin = options.σmin
  η1 = options.η1
  η2 = options.η2
  ν = options.ν
  γ = options.γ

  # retrieve workspace
  xk = solver.xk
  xk .= x0
  ∇fk = solver.∇fk
  mν∇fk = solver.mν∇fk
  xkn = solver.xkn
  s = solver.s
  has_bnds = solver.has_bnds
  if has_bnds
    l_bound = solver.l_bound
    u_bound = solver.u_bound
    l_bound_m_x = solver.l_bound_m_x
    u_bound_m_x = solver.u_bound_m_x
  end
  Fobj_hist = solver.Fobj_hist
  Hobj_hist = solver.Hobj_hist
  Complex_hist = solver.Complex_hist

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
  hk = @views h(xk[selected])
  if hk == Inf
    verbose > 0 && @info "R2: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, x0, one(eltype(x0)))
    hk = @views h(xk[selected])
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "R2: found point where h has value" hk
  end
  hk == -Inf && error("nonsmooth term is not proper")

  if has_bnds
    @. l_bound_m_x = l_bound - xk
    @. u_bound_m_x = u_bound - xk
    ψ = shifted(h, xk, l_bound_m_x, u_bound_m_x, selected)
  else
    ψ = shifted(h, xk)
  end

  if verbose > 0
    #! format: off
    @info @sprintf "%6s %8s %8s %7s %8s %7s %7s %7s %1s" "iter" "f(x)" "h(x)" "√(ξ/ν)" "ρ" "σ" "‖x‖" "‖s‖" ""
    #! format: off
  end

  local ξ::R
  k = 0
  σk = max(1 / ν, σmin)
  ν = 1 / σk
  sqrt_ξ_νInv = one(R)

  fk = f(xk)
  ∇f!(∇fk, xk)
  @. mν∇fk = -ν * ∇fk

  optimal = false
  tired = maxIter > 0 && k ≥ maxIter || elapsed_time > maxTime

  while !(optimal || tired)
    k = k + 1
    elapsed_time = time() - start_time
    Fobj_hist[k] = fk
    Hobj_hist[k] = hk

    # define model
    φk(d) = dot(∇fk, d)
    mk(d)::R = φk(d) + ψ(d)::R

    prox!(s, ψ, mν∇fk, ν)
    Complex_hist[k] += 1
    mks = mk(s)
    ξ = hk - mks + max(1, abs(hk)) * 10 * eps()
    sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / ν) : sqrt(-ξ / ν)

    if ξ ≥ 0 && k == 1
      ϵ += ϵr * sqrt_ξ_νInv # make stopping test absolute and relative
    end
    
    if (ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ ≥ 0 && sqrt_ξ_νInv ≤ ϵ)
      optimal = true
      continue
    end

    ξ > 0 || error("R2: prox-gradient step should produce a decrease but ξ = $(ξ)")
    xkn .= xk .+ s
    fkn = f(xkn)
    hkn = @views h(xkn[selected])
    hkn == -Inf && error("nonsmooth term is not proper")

    Δobj = (fk + hk) - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ρk = Δobj / ξ

    if (verbose > 0) && (k % ptf == 0)
      #! format: off
      σ_stat = (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")
      @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e %7.1e %7.1e %1s" k fk hk sqrt_ξ_νInv ρk σk norm(xk) norm(s) σ_stat

      #! format: on
    end

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
      ∇f!(∇fk, xk)
      shift!(ψ, xk)
    end

    if ρk < η1 || ρk == Inf
      σk = σk * γ
    end

    ν = 1 / σk
    tired = maxIter > 0 && k ≥ maxIter
    if !tired
      @. mν∇fk = -ν * ∇fk
    end
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8.1e %8.1e" k fk hk
    elseif optimal
      #! format: off
      @info @sprintf "%6d %8.1e %8.1e %7.1e %8s %7.1e %7.1e %7.1e" k fk hk sqrt(ξ/ν) "" σk norm(xk) norm(s)
      #! format: on
      @info "R2: terminating with √(ξ/ν) = $(sqrt_ξ_νInv)"
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

  return k, status, fk, hk, sqrt_ξ_νInv
end
