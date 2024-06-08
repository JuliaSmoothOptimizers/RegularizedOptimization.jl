export R2,R2Solver,solve!

mutable struct R2Solver{R <: Real,G <: Union{ShiftedProximableFunction, Nothing}, S <: AbstractVector{R}} <: AbstractOptimizationSolver
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
  ψ = nothing
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

function R2Solver(
  reg_nlp::AbstractRegularizedNLPModel{T,V};
  max_iter::Int = 10000,
  kwargs...
) where {T,V}
  kwargs_dict = Dict(kwargs...)
  x0 = pop!(kwargs_dict, :x0, reg_nlp.model.meta.x0)
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
  Fobj_hist = zeros(T, max_iter)
  Hobj_hist = zeros(T, max_iter)
  Complex_hist = zeros(Int, max_iter)
  
  ψ = has_bnds ? shifted(reg_nlp.h, x0, l_bound_m_x, u_bound_m_x, reg_nlp.selected) : shifted(reg_nlp.h, x0)
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
    R2(reg_nlp)

A first-order quadratic regularization method for the problem

    min f(x) + h(x)

where f: ℝⁿ → ℝ has a Lipschitz-continuous gradient, and h: ℝⁿ → ℝ is
lower semi-continuous, proper and prox-bounded.

About each iterate xₖ, a step sₖ is computed as a solution of

    min  φ(s; xₖ) + ½ σₖ ‖s‖² + ψ(s; xₖ)

where φ(s ; xₖ) = f(xₖ) + ∇f(xₖ)ᵀs is the Taylor linear approximation of f about xₖ,
ψ(s; xₖ) = h(xₖ + s), ‖⋅‖ is a user-defined norm and σₖ > 0 is the regularization parameter.

For advanced usage, first define a solver "R2Solver" to preallocate the memory used in the algorithm, and then call `solve!`:

### Arguments
* `reg_nlp::AbstractRegularizedNLPModel`: a smooth, regularized optimization problem: min f(x) + h(x)
* `options::ROSolverOptions`: a structure containing algorithmic parameters
* `nlp::AbstractNLPModel`: a smooth optimization problem (in the second calling form)
* `h`: a regularizer such as those defined in ProximalOperators (in the second calling form)
* `x0::AbstractVector`: an initial guess (in the third calling form)

### Keyword Arguments

* `x0::AbstractVector`: an initial guess (in the first and second calling form resp.: default = `reg_nlp.model.meta.x0` and `nlp.meta.x0`)
* `selected::AbstractVector{<:Integer}`: (default `1:length(x0)`) (in the first calling form, this should be stored in reg_nlp)

The objective and gradient of the smooth part will be accessed.

In the third form, instead of `nlp`, the user may pass in

* `f` a function such that `f(x)` returns the value of f at x
* `∇f!` a function to evaluate the gradient in place, i.e., such that `∇f!(g, x)` store ∇f(x) in `g`.

### Return values

* `xk`: the final iterate
* `Fobj_hist`: an array with the history of values of the smooth objective
* `Hobj_hist`: an array with the history of values of the nonsmooth objective
* `Complex_hist`: an array with the history of number of inner iterations.
"""

function R2(
  nlp::AbstractNLPModel{R, V}, 
  h, 
  options::ROSolverOptions{R};
  kwargs...) where{ R <: Real, V }
  kwargs_dict = Dict(kwargs...)
  selected = pop!(kwargs_dict, :selected, 1:nlp.meta.nvar) 
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  reg_nlp = RegularizedNLPModel(nlp, h, selected)
  return R2(
    reg_nlp,
    x = x0,
    a_tol = options.ϵa,
    r_tol = options.ϵr,
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
  return stats
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
  nlp = FirstOrderModel(f,∇f!,x0)
  reg_nlp = RegularizedNLPModel(nlp,h,selected) 
  stats = R2(
  reg_nlp,
  x=x0,
  a_tol = options.ϵa,
  r_tol = options.ϵr,
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
    :NonSmooth => h,
    :status => stats.status,
    :fk => stats.solver_specific[:smooth_obj],
    :hk => stats.solver_specific[:nonsmooth_obj],
    :ξ => stats.solver_specific[:xi],
    :elapsed_time => stats.elapsed_time,
  )
return stats.solution,stats.iter,outdict
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
  nlp = FirstOrderModel(f,∇f!,x0,lcon = l_bound, ucon = u_bound)
  reg_nlp = RegularizedNLPModel(nlp,h,selected) 
  stats = R2(
    reg_nlp,
    x=x0,
    a_tol = options.ϵa,
    r_tol = options.ϵr,
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
    :NonSmooth => h,
    :status => stats.status,
    :fk => stats.solver_specific[:smooth_obj],
    :hk => stats.solver_specific[:nonsmooth_obj],
    :ξ => stats.solver_specific[:xi],
    :elapsed_time => stats.elapsed_time,
  )
  return stats.solution,stats.iter,outdict
end


function R2(reg_nlp::AbstractRegularizedNLPModel; kwargs...)
  kwargs_dict = Dict(kwargs...)
  x0 = pop!(kwargs_dict, :x, reg_nlp.model.meta.x0)
  solver = R2Solver(reg_nlp,x0=x0)
  stats = GenericExecutionStats(reg_nlp.model)
  cb = (nlp, solver, stats) -> begin
    solver.Fobj_hist[stats.iter+1] = stats.solver_specific[:smooth_obj]
    solver.Hobj_hist[stats.iter+1] = stats.solver_specific[:nonsmooth_obj]
    solver.Complex_hist[stats.iter+1] += 1
  end
  solve!(
    solver,
    reg_nlp,
    stats;
    x = x0,
    callback = cb,
    kwargs...
  )
  set_solver_specific!(stats, :Fhist, solver.Fobj_hist[1:stats.iter+1])
  set_solver_specific!(stats, :Hhist, solver.Hobj_hist[1:stats.iter+1])
  set_solver_specific!(stats, :SubsolverCounter, solver.Complex_hist[1:stats.iter+1])
  return stats
end

function SolverCore.solve!(
  solver::R2Solver{T}, 
  reg_nlp::AbstractRegularizedNLPModel{T, V}, 
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = reg_nlp.model.meta.x0,
  a_tol::T = √eps(T),
  r_tol::T = √eps(T),
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
      hdr_override = Dict{Symbol,String}(   # TODO: Add this as constant dict elsewhere
        :iter => "iter",
        :fx => "f(x)",
        :hx => "h(x)",
        :xi => "√(ξ/ν)",
        :ρ => "ρ",
        :σ => "σ",
        :normx => "‖x‖",
        :norms => "‖s‖",
        :arrow => " "
      ),
      colsep = 1,
    )
  end

  local ξ::T
  σk = max(1 / ν, σmin)
  ν = 1 / σk
  sqrt_ξ_νInv = one(T)

  fk = obj(nlp, xk)
  grad!(nlp, xk, ∇fk)
  @. mν∇fk = -ν * ∇fk

  set_iter!(stats, 0)
  start_time = time()
  set_time!(stats, 0.0)
  set_objective!(stats,fk + hk)
  set_solver_specific!(stats,:smooth_obj,fk)
  set_solver_specific!(stats,:nonsmooth_obj, hk)

  φk(d) = dot(∇fk, d)   
  mk(d)::T = φk(d) + ψ(d)::T

  prox!(s, ψ, mν∇fk, ν)
  mks = mk(s)

  ξ = hk - mks + max(1, abs(hk)) * 10 * eps()
  ξ > 0 || error("R2: prox-gradient step should produce a decrease but ξ = $(ξ)")
  sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / ν) : sqrt(-ξ / ν)
  a_tol += r_tol * sqrt_ξ_νInv # make stopping test absolute and relative

  solved = (ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ ≥ 0 && sqrt_ξ_νInv ≤ a_tol)

  set_solver_specific!(stats,:xi,sqrt_ξ_νInv)
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
      max_iter = max_iter
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

    if η1 ≤ ρk < Inf
      xk .= xkn
      if has_bnds
        @. l_bound_m_x = l_bound - xk
        @. u_bound_m_x = u_bound - xk
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
    set_solver_specific!(stats,:smooth_obj,fk)
    set_solver_specific!(stats,:nonsmooth_obj, hk)
    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)

    prox!(s, ψ, mν∇fk, ν)
    mks = mk(s)

    ξ = hk - mks + max(1, abs(hk)) * 10 * eps()
    sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / ν) : sqrt(-ξ / ν)
    solved = (ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ ≥ 0 && sqrt_ξ_νInv ≤ a_tol)
  
    verbose > 0 && 
      stats.iter % verbose == 0 &&
        @info log_row(Any[stats.iter, fk, hk, sqrt_ξ_νInv, ρk, σk, norm(xk), norm(s), (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")], colsep = 1)

    set_solver_specific!(stats,:xi,sqrt_ξ_νInv)
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
        max_iter = max_iter
      ),
    )

    callback(nlp, solver, stats)

    done = stats.status != :unknown
  end

  verbose > 0 &&
    stats.status == :first_order &&
        @info "R2: terminating with √(ξ/ν) = $(sqrt_ξ_νInv)"

  set_solution!(stats,xk)
  return stats
end

function get_status(
  reg_nlp;
  elapsed_time = 0.0,
  iter = 0,
  optimal = false,
  improper = false,
  max_eval = Inf,
  max_time = Inf,
  max_iter = Inf,
)
  if optimal
    :first_order
  elseif improper
    :improper
  elseif iter > max_iter
    :max_iter
  elseif elapsed_time > max_time
    :max_time
  elseif neval_obj(reg_nlp.model) > max_eval && max_eval != -1
    :max_eval
  else
    :unknown
  end
end