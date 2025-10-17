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
  JtF::V  # Add JtF to store the gradient J'F
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
  # Preallocated QuadraticModel components to avoid reallocations
  x0_quad::V
  reg_hess_wrapper::ShiftedHessian{T}
  reg_hess_op::LinearOperator
  v0::V  # workspace for power method
  tmp_res::V # temporary residual storage for power method
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
  # residuals will be allocated after creating Jacobian operator (size may vary)
  Fk = similar(x0, 0)
  Fkn = similar(x0, 0)
  Jv = similar(x0, 0)
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

  # Defer Jacobian/residual construction to solve!; preallocate JtF (length n)
  JtF = similar(xk)
  # residuals will be allocated in solve! when the Jacobian operator is available
  Fk = similar(x0, 0)
  Fkn = similar(x0, 0)
  Jv = similar(x0, 0)
  n = length(xk)
  c0_val = dot(Fk, Fk) / 2  # Initial constant term = 1/2||F||²
  x0_quad = zeros(T, n)  # Pre-allocated x0 for QuadraticModel
  # Create mutable wrapper around Hessian; use JacobianGram to avoid forming J'J
  gram = JacobianGram{T}(nothing, zeros(T, 0))
  # Try to initialize JacobianGram.tmp with the row size of the Jacobian at x0
  try
    J_init = jac_op_residual(reg_nls.model, x0)
    gram.J = J_init
    gram.tmp = zeros(T, size(J_init, 1))
  catch
    # If jac_op_residual is not available for this model type, leave tmp empty
    gram.J = nothing
    gram.tmp = zeros(T, 0)
  end
  reg_hess_wrapper = ShiftedHessian{T}(gram, T(1))
  reg_hess_op = LinearOperator{T}(n, n, false, false,
    (y, x) -> mul!(y, reg_hess_wrapper, x),
    (y, x) -> mul!(y, adjoint(reg_hess_wrapper), x),
    (y, x) -> mul!(y, adjoint(reg_hess_wrapper), x),
  )
  sub_nlp = QuadraticModel(JtF, reg_hess_op, c0 = c0_val, x0 = x0_quad, name = "LM-subproblem")
  subpb = RegularizedNLPModel(sub_nlp, ψ)
  substats = RegularizedExecutionStats(subpb)
  subsolver = subsolver(subpb)
  v0 = [(-1.0)^i for i = 0:(n - 1)]
  v0 ./= sqrt(n)
  tmp_res = copy(gram.tmp)

  return LMSolver{T, typeof(ψ), V, typeof(subsolver), typeof(subpb)}(
    xk,
    ∇fk,
    mν∇fk,
    Fk,
    Fkn,
    Jv,
    Jtv,
    JtF,  # Add JtF to constructor
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
    x0_quad,
    reg_hess_wrapper,
    reg_hess_op,
    v0,
    tmp_res,
  )
end

"""
    LM(reg_nls; kwargs...)

A Levenberg-Marquardt method for the problem

    min ½ ‖F(x)‖² + h(x)

where F: ℝⁿ → ℝᵐ and its Jacobian J are Lipschitz continuous and h: ℝⁿ → ℝ is
lower semi-continuous, proper and prox-bounded.

At each iteration, a step s is computed as an approximate solution of

    min  ½ ‖J(x) s + F(x)‖² + ½ σ ‖s‖² + ψ(s; xₖ) 

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

The algorithm stops either when `√(ξₖ/νₖ) < atol + rtol*√(ξ₀/ν₀) ` or `ξₖ < 0` and `√(-ξₖ/νₖ) < neg_tol` where ξₖ := f(xₖ) + h(xₖ) - φ(sₖ; xₖ) - ψ(sₖ; xₖ), and √(ξₖ/νₖ) is a stationarity measure.

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
$(callback_docstring)
"""
function LM(nls::AbstractNLSModel, h::H, options::ROSolverOptions; kwargs...) where {H}
  kwargs_dict = Dict(kwargs...)
  selected = pop!(kwargs_dict, :selected, 1:(nls.meta.nvar))
  x0 = pop!(kwargs_dict, :x0, nls.meta.x0)
  # allow callers to request skipping σ in the quadratic regularizer
  skip_sigma = pop!(kwargs_dict, :skip_sigma, false)
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
    skip_sigma = skip_sigma,
    kwargs_dict...,
  )
end

function LM(reg_nls::AbstractRegularizedNLPModel; kwargs...)
  kwargs_dict = Dict(kwargs...)
  subsolver = pop!(kwargs_dict, :subsolver, R2Solver)
  m_monotone = pop!(kwargs_dict, :m_monotone, 1)
  # propagate skip_sigma to the solver
  skip_sigma = pop!(kwargs_dict, :skip_sigma, false)
  solver = LMSolver(reg_nls, subsolver = subsolver, m_monotone = m_monotone)
  stats = RegularizedExecutionStats(reg_nls)
  solve!(solver, reg_nls, stats; kwargs_dict..., skip_sigma = skip_sigma)
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
  skip_sigma::Bool = false,   # <-- new kwarg (default: false)
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
  local prox_evals::Int = 0
  local inner_stall_count::Int = 0
  INNER_STALL_THRESHOLD = 500
  # Track subsolver stalls and keep a backup step
  stall_counter = 0
  s_backup = similar(s)

  # Ensure residual vectors are sized to match the Jacobian rows
  Jk = jac_op_residual(nls, xk)
  m = size(Jk, 1)
  if length(Fk) != m
    solver.Fk = zeros(T, m)
    solver.Fkn = similar(solver.Fk)
    solver.Jv = similar(solver.Fk)
    Fk = solver.Fk
    Fkn = solver.Fkn
    Jv = solver.Jv
  end
  residual!(nls, xk, Fk)
  jtprod_residual!(nls, xk, Fk, ∇fk)
  fk = dot(Fk, Fk) / 2

  # Compute Jacobian norm for normalization
  Jk = jac_op_residual(nls, xk)
  # Estimate Jacobian operator norm (largest singular value) via power method to avoid heavy allocations
  if length(solver.tmp_res) != size(Jk, 1)
    solver.tmp_res = zeros(T, size(Jk, 1))
  end
  σmax = power_method_singular!(Jk, solver.v0, solver.subpb.model.data.v, solver.tmp_res, 5)
  # if the caller requested skipping σ, compute ν without adding σk into the denominator
  if skip_sigma
    ν = θ / (σmax^2)
  else
    ν = θ / (σmax^2 + σk) # ‖J'J + σₖ I‖ = ‖J‖² + σₖ
  end
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
    
  # Update the QuadraticModel in-place: update J'F and mutate the Hessian wrapper
  Jk = jac_op_residual(nls, xk)
  mul!(solver.JtF, Jk', Fk)  # Update gradient J'F in-place
  # Update underlying JacobianGram wrapper to reference the new J and resize tmp
  solver.reg_hess_wrapper.B.J = Jk
  if length(solver.tmp_res) != size(Jk, 1)
    solver.reg_hess_wrapper.B.tmp = zeros(T, size(Jk, 1))
  else
    solver.reg_hess_wrapper.B.tmp .= 0
  end
  # respect skip_sigma: when true the shifted Hessian should not include the outer σk
  solver.reg_hess_wrapper.sigma = skip_sigma ? zero(T) : σk
  # Update QuadraticModel's gradient/counters in-place
  c0_val = dot(Fk, Fk) / 2  # Constant term = 1/2||F||²
  solver.subpb.model.data.c0 = c0_val  # Ensure the constant term is stored
  update_quadratic_model!(solver.subpb.model, solver.JtF)
  # If the subsolver requires any special scaling (e.g. R2DHSolver), set it
  isa(solver.subsolver, R2DHSolver) && (solver.subsolver.D.d[1] = 1/ν)
  # Backup current candidate step before calling subsolver
  copyto!(s_backup, s)
  if isa(solver.subsolver, R2Solver) #FIXME
    solve!(solver.subsolver, solver.subpb, solver.substats, x = s, atol = sub_atol, ν = ν, max_iter = 1000)
  else
    solve!(
      solver.subsolver,
      solver.subpb,
      solver.substats,
      x = s,
      atol = sub_atol,
      σk = σk, #FIXME
      max_iter = 1000,
    )
  end

  prox_evals += solver.substats.iter

  # If the subsolver spent too many iterations (stalled), restore previous step
  # and use a cheap proximal-gradient fallback to make progress instead of
  # spinning on the inner solver. Also adjust σk to try a different
  # regularization direction and cap its growth.
  if solver.substats.iter > INNER_STALL_THRESHOLD
    stall_counter += 1
    @warn "LM: subsolver iter count high - restoring previous step" iter=solver.substats.iter stall=stall_counter
    # restore previous candidate step
    copyto!(s, s_backup)
    # Fallback: take a single prox-gradient step (cheap) to make outer loop progress
    try
      prox!(s, ψ, mν∇fk, ν)
    catch e
      @warn "LM: prox fallback failed" err=string(e)
    end
    # reset substats.solution to the fallback for downstream logic
    solver.substats.solution .= s
    # Adjust σk to try a different regularization direction and cap its growth
    σk = max(σk / γ, σmin)
    σ_cap = 1e40
    if σk > σ_cap
      σk = σ_cap
    end
  else
    # Accept subsolver solution
    s .= solver.substats.solution
    stall_counter = 0
  end

  # If we've observed repeated stalls, accept the (prox) fallback step and
  # advance the outer iteration to avoid spinning on the inner solver.
  if stall_counter >= 3
    @warn "LM: repeated inner stalls — accepting prox fallback and continuing" stall=stall_counter
    # apply step
    xk .= xk .+ s
    # update residuals and gradient for the new xk
    residual!(nls, xk, Fk)
    fk = dot(Fk, Fk) / 2
    hk = @views h(xk[selected])
    shift!(ψ, xk)
    jtprod_residual!(nls, xk, Fk, ∇fk)
    # reset counters so we don't immediately re-enter this branch
    stall_counter = 0
    prox_evals += 1
    # proceed to next outer iteration
    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)
    continue
  end

    # Diagnostic: if the subsolver used a very large number of iterations, log state
    if solver.substats.iter > 1000
      @warn "LM: subsolver iter count high" iter=solver.substats.iter fk=fk hk=hk σk=σk ν=ν
      @warn "QuadraticModel gradient (first 10)" grad=solver.subpb.model.data.c[1:min(10,end)]
      @warn "QuadraticModel H type" H_type=typeof(solver.subpb.model.data.H)
      try
        @warn "H* s (first 10)" Hs=begin
          tmp = similar(s)
          mul!(tmp, solver.subpb.model.data.H, s)
          tmp[1:min(10,end)]
        end
      catch e
        @warn "Failed to mul! H* s" err=string(e)
      end
    end

    xkn .= xk .+ s
    residual!(nls, xkn, Fkn)
    fkn = dot(Fkn, Fkn) / 2
    hkn = @views h(xkn[selected])
    mks = mk(s)
    ξ = fk + hk - mks + max(1, abs(hk)) * 10 * eps()

    # Diagnostic: if ξ is not finite, emit helpful warnings with nearby state
    if !isfinite(ξ)
      @warn "LM diagnostic: ξ not finite" ξ=ξ fk=fk hk=hk mks=mks
      try
        @warn "QuadraticModel data.c (first 8)" c_first = solver.subpb.model.data.c[1:min(end,8)]
      catch
      end
      try
        @warn "QuadraticModel c0" c0 = getfield(solver.subpb.model.data, :c0)
      catch
      end
      try
        @warn "JtF contains NaN?" any_nan = any(isnan, solver.JtF)
      catch
      end
      try
        Jk_dbg = jac_op_residual(nls, xk)
        @warn "Jacobian operator size" m=size(Jk_dbg,1) n=size(Jk_dbg,2)
      catch
      end
    end

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
        Jk = jac_op_residual(nls, xk)
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
    mks = mk1(s)


    ξ = fk + hk - mks + max(1, abs(hk)) * 10 * eps()

    if isnan(ξ)
      @info "QuadraticModel gradient" solver.subpb.model.data.c
      @info "QuadraticModel Hessian operator" solver.subpb.model.data.H
      @info "QuadraticModel full" solver.subpb.model
      @info "JacobianGram J" solver.reg_hess_wrapper.B.J
      @info "JacobianGram tmp" solver.reg_hess_wrapper.B.tmp
      @info "Residual Fk" Fk
      @info "Residual Fkn" Fkn
      @info "JtF" solver.JtF
      error("LM: failed to compute a step: ξ = NaN")
    elseif ξ ≤ 0
      error("LM: failed to compute a step: ξ = $ξ")
    end
    (ξ < 0 && sqrt_ξ1_νInv > neg_tol) &&
      error("LM: prox-gradient step should produce a decrease but ξ = $(ξ)")

    # Recompute stationarity measure and solved flag using current ξ and ν
    sqrt_ξ1_νInv = ξ ≥ 0 ? sqrt(ξ / ν) : sqrt(-ξ / ν)
    solved = (ξ < 0 && sqrt_ξ1_νInv ≤ neg_tol) || (ξ ≥ 0 && sqrt_ξ1_νInv ≤ atol)

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
      Any[stats.iter, 0, fk, hk, sqrt_ξ1_νInv, ρk, σk, norm(xk), norm(s), 1 / ν, ""],
      colsep = 1,
    )
    @info "LM: terminating with √(ξ1/ν) = $(sqrt_ξ1_νInv)"
  end

  set_solution!(stats, xk)
  set_residuals!(stats, zero(T), sqrt_ξ1_νInv)
  return stats
end
