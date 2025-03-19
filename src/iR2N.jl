export iR2N
"""
iR2N(nlp, h, χ, options; kwargs...)

A regularized quasi-Newton method for the problem

    min f(x) + h(x)

where f: ℝⁿ → ℝ has a Lipschitz-continuous Jacobian, and h: ℝⁿ → ℝ is
lower semi-continuous and proper.

About each iterate xₖ, a step sₖ is computed as an approximate solution of

    min  φ(s; xₖ) + ½ σₖ ‖s‖² + ψ(s; xₖ) 

where φ(s ; xₖ) = f(xₖ) + ∇f(xₖ)ᵀs + ½ sᵀ Bₖ s  is a quadratic approximation of f about xₖ,
ψ(s; xₖ) = h(xₖ + s) and σₖ > 0 is the regularization parameter.
The subproblem is solved inexactly by way of a first-order method such as the proximal-gradient
method or the quadratic regularization method.

### Arguments

* `nlp::AbstractNLPModel`: a smooth optimization problem
* `h`: a regularizer such as those defined in ProximalOperators
* `options::ROSolverOptions`: a structure containing algorithmic parameters

The objective, gradient and Hessian of `nlp` will be accessed.
The Hessian is accessed as an abstract operator and need not be the exact Hessian.

### Keyword arguments

* `x0::AbstractVector`: an initial guess (default: `nlp.meta.x0`)
* `subsolver_logger::AbstractLogger`: a logger to pass to the subproblem solver (default: the null logger)
* `subsolver`: the procedure used to compute a step (`PG` or `R2`)
* `subsolver_options::ROSolverOptions`: default options to pass to the subsolver (default: all default options)
* `selected::AbstractVector{<:Integer}`: (default `1:f.meta.nvar`),

### Return values

* `xk`: the final iterate
* `Fobj_hist`: an array with the history of values of the smooth objective
* `Hobj_hist`: an array with the history of values of the nonsmooth objective
* `Complex_hist`: an array with the history of number of inner iterations.
"""
function iR2N(
  f::AbstractNLPModel,
  h::H,
  options::ROSolverOptions{R};
  x0::AbstractVector = f.meta.x0,
  subsolver_logger::Logging.AbstractLogger = Logging.NullLogger(),
  subsolver = iR2,
  subsolver_options = ROSolverOptions(ϵa = options.ϵa),
  Mmonotone::Int = 0,
  selected::AbstractVector{<:Integer} = 1:(f.meta.nvar),
  prox_callback_flag::Int = 0,
  κξ_flag::Int = 0,
) where {H, R}
  start_time = time()
  elapsed_time = 0.0
  # initialize passed options
  ϵ = options.ϵa
  ϵ_subsolver_init = subsolver_options.ϵa
  ϵ_subsolver = copy(ϵ_subsolver_init)
  ϵr = options.ϵr
  Δk = options.Δk
  verbose = options.verbose
  maxIter = options.maxIter
  maxTime = options.maxTime
  η1 = options.η1
  η2 = options.η2
  γ = options.γ
  θ = options.θ
  σmin = options.σmin
  α = options.α
  β = options.β
  σk = options.σk
  dualGap = options.dualGap
  κξ = options.κξ

  # initialize callback and pointer to callback function
  if prox_callback_flag == 0
    options.callback_pointer =
      @cfunction(default_prox_callback, Cint, (Ptr{Cdouble}, Csize_t, Cdouble, Ptr{Cvoid}))
  elseif prox_callback_flag == 1
    options.callback_pointer =
      @cfunction(default_prox_callback_v2, Cint, (Ptr{Cdouble}, Csize_t, Cdouble, Ptr{Cvoid}))
  else
    options.callback_pointer =
      @cfunction(default_prox_callback_v3, Cint, (Ptr{Cdouble}, Csize_t, Cdouble, Ptr{Cvoid}))
  end

  function update_κξ(a, b, k)
    return a + (b - a) * k / options.maxIter
  end

  # initialize strategy for κξ
  if κξ_flag == 0
    # default strategy : κξ remains constant
    a = κξ
    b = κξ
  elseif κξ_flag == 1
    # κξ is increased at each iteration (i.e we become more and more demanding on the quality of the solution)
    a = κξ
    b = 1.0
  else
    # κξ is decreased at each iteration (i.e we become less and less demanding on the quality of the solution)
    a = κξ
    b = 1 / 2
  end

  # store initial values of the subsolver_options fields that will be modified
  ν_subsolver = subsolver_options.ν
  ϵa_subsolver = subsolver_options.ϵa

  local l_bound, u_bound
  if has_bounds(f)
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
  #σk = max(1 / options.ν, σmin) #SVM
  xk = copy(x0)
  hk = h(xk[selected])
  if hk == Inf # TODO 
    verbose > 0 && @info "iR2N: finding initial guess where nonsmooth term is finite"
    prox!(
      xk,
      h,
      x0,
      one(eltype(x0)),
      AlgorithmContextCallback(dualGap = dualGap, flag_projLp = 1, iters_prox_projLp = 100),
      options.callback_pointer,
    )
    hk = h(xk[selected])
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "iR2N: found point where h has value" hk
  end
  hk == -Inf && error("nonsmooth term is not proper")

  xkn = similar(xk)
  s = zero(xk)

  ψ = has_bounds(f) ? shifted(h, xk, l_bound - xk, u_bound - xk, selected) : shifted(h, xk) # TODO : implement shifted bounds for inexact prox 

  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  FHobj_hist = fill!(Vector{R}(undef, Mmonotone), R(-Inf))
  Complex_hist = zeros(Int, maxIter)
  if verbose > 0
    #! format: off
    @info @sprintf "%6s %8s %8s %8s %7s %7s %8s %7s %7s %7s %7s %7s %7s %1s" "outer" "inner" "f(x)" "h(x)" "√(ξ1/ν)" "√ξ" "ρ" "σ" "‖x‖" "‖s‖" "‖Bₖ‖" "dualGap" "κξ" "iR2N"
    #! format: on
  end

  # main algorithm initialization

  local ξ1
  k = 0

  fk = obj(f, xk)
  ∇fk = grad(f, xk)
  ∇fk⁻ = copy(∇fk)

  quasiNewtTest = isa(f, QuasiNewtonModel)
  Bk = hess_op(f, xk)

  λmax = opnorm(Bk)
  # found_λ || error("operator norm computation failed")
  νInv = (1 + θ) * (σk + λmax)
  sqrt_ξ1_νInv = one(R)

  # initialize context for prox_callback
  context = AlgorithmContextCallback(
    hk = hk,
    κξ = κξ,
    shift = similar(xk),
    s_k_unshifted = similar(xk),
    dualGap = dualGap,
    iters_prox_projLp = 100,
  )

  optimal = false
  tired = k ≥ maxIter || elapsed_time > maxTime

  while !(optimal || tired)
    k = k + 1
    elapsed_time = time() - start_time
    Fobj_hist[k] = fk
    Hobj_hist[k] = hk
    Mmonotone > 0 && (FHobj_hist[mod(k - 1, Mmonotone) + 1] = fk + hk)

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

    # take first proximal gradient step s1 and see if current xk is nearly stationary
    # s1 minimizes φ1(s) + ‖s‖² / 2 / ν + ψ(s) ⟺ s1 ∈ prox{νψ}(-ν∇φ1(0)).

    subsolver_options.ν = 1 / νInv

    # prepare callback and pointer to callback function
    context.hk = hk
    context.mk = mk1
    context.κξ = update_κξ(a, b, k)
    context.shift = ψ.xk + ψ.sj
    context.dualGap = dualGap # reset dualGap to its initial value at each iteration

    # call prox computation to get s_{k,cp}
    prox!(s, ψ, -subsolver_options.ν * ∇fk, subsolver_options.ν, context, options.callback_pointer)

    # compute ξ1 : associated with s_{k,cp}
    ξ1 = hk - mk1(s) + max(1, abs(hk)) * 10 * eps()

    sqrt_ξ1_νInv = ξ1 ≥ 0 ? sqrt(ξ1 * νInv) : sqrt(-ξ1 * νInv)

    (ξ1 < 0 && sqrt_ξ1_νInv > options.neg_tol) && error(
      "iR2N: first prox-gradient step should produce a decrease but ξ1 = $(ξ1) and √(ξ1/ν) = $(sqrt_ξ1_νInv) > $(options.neg_tol)",
    )

    if ξ1 ≥ 0 && k == 1
      ϵ_increment = ϵr * sqrt_ξ1_νInv
      ϵ += ϵ_increment  # make stopping test absolute and relative
      ϵ_subsolver += ϵ_increment
    end

    if sqrt_ξ1_νInv < ϵ * sqrt(κξ)
      # the current xk is approximately first-order stationary
      optimal = true
      continue
    end
    s1 = copy(s)

    subsolver_options.ϵa = k == 1 ? 1.0e-3 : min(sqrt_ξ1_νInv^(1.5), sqrt_ξ1_νInv * 1e-3) # 1.0e-5 default
    subsolver_options.neg_tol = options.neg_tol
    subsolver_options.dualGap = dualGap
    subsolver_options.κξ = context.κξ
    subsolver_options.verbose = 100
    subsolver_options.callback_pointer = options.callback_pointer
    subsolver_options.mk1 = mk # tests on value of the model
    @debug "setting inner stopping tolerance to" subsolver_options.optTol
    subsolver_args = subsolver == R2DH ? (SpectralGradient(νInv, f.meta.nvar),) : ()
    s, iter, outdict = with_logger(subsolver_logger) do
      subsolver(φ, ∇φ!, ψ, subsolver_args..., subsolver_options, s)
    end
    push!(context.prox_stats[2], iter)
    push!(context.prox_stats[3], outdict[:ItersProx])

    if norm(s) > β * norm(s1)
      s .= s1
      println("iR2N: using s1")
    end
    # restore initial subsolver_options.ϵa here so that subsolver_options.ϵa
    # is not modified if there is an error

    subsolver_options.ν = ν_subsolver
    subsolver_options.ϵa = ϵ_subsolver_init
    Complex_hist[k] = iter

    xkn .= xk .+ s
    fkn = obj(f, xkn)
    hkn = h(xkn[selected])
    hkn == -Inf && error("nonsmooth term is not proper")
    mks = mk(s) #- σk * dot(s, s) / 2

    fhmax = Mmonotone > 0 ? maximum(FHobj_hist) : fk + hk
    Δobj = fhmax - (fkn + hkn) + max(1, abs(fhmax)) * 10 * eps()
    Δmod = fhmax - (fk + mks) + max(1, abs(hk)) * 10 * eps()
    ξ = hk - mks + max(1, abs(hk)) * 10 * eps()
    sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ * νInv) : sqrt(-ξ * νInv)

    # check assumptions on ξ and ξ1 (cf Zulip for discussion)
    aux_assert = (1 - 1 / (1 + θ)) * ξ1
    if (ξ < aux_assert && sqrt_ξ_νInv > options.neg_tol) ||
       ((ξ < 0 && sqrt_ξ_νInv > options.neg_tol) || isnan(ξ))
      if (ξ < 0 && sqrt_ξ_νInv > options.neg_tol) || isnan(ξ)
        error("iR2N: failed to compute a step: ξ = $ξ and sqrt_ξ_νInv = $sqrt_ξ_νInv")
      elseif ξ < aux_assert
        error(
          "iR2N: ξ should be ≥ (1 - 1/(1+θ)) * ξ1 but ξ = $ξ and (1 - 1/(1+θ)) * ξ1 = $aux_assert.",
        )
      end
    end

    if (ξ < 0 && sqrt_ξ_νInv > options.neg_tol) || isnan(ξ)
      error("iR2N: failed to compute a step: ξ = $ξ and sqrt_ξ_νInv = $sqrt_ξ_νInv")
    end

    ρk = Δobj / Δmod

    iR2N_stat = (η2 ≤ ρk < Inf) ? "↗" : (ρk < η1 ? "↘" : "=")

    if (verbose > 0) && ((k % ptf == 0) || (k == 1))
      #! format: off
      @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %7.1e %7.1e %1s" k iter fk hk sqrt_ξ1_νInv sqrt(abs(ξ1)) ρk σk norm(xk) norm(s) λmax context.dualGap context.κξ iR2N_stat
      #! format: off
    end

    if η2 ≤ ρk < Inf
        σk = max(σk/γ, σmin)
    end

    if η1 ≤ ρk < Inf
      xk .= xkn
      has_bounds(f) && set_bounds!(ψ, l_bound - xk, u_bound - xk)

      #update functions
      fk = fkn
      hk = hkn

      # update gradient & Hessian
      shift!(ψ, xk)
      ∇fk = grad(f, xk)
      if quasiNewtTest
        push!(f, s, ∇fk - ∇fk⁻)
      end
      Bk = hess_op(f, xk)
      λmax = opnorm(Bk)
      # found_λ || error("operator norm computation failed")
      ∇fk⁻ .= ∇fk
    end

    if ρk < η1 || ρk == Inf
        σk = σk * γ
    end
    νInv = (1 + θ) * (σk + λmax)
    tired = k ≥ maxIter || elapsed_time > maxTime
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8s %8.1e %8.1e" k "" fk hk
    elseif optimal
      #! format: off
      @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8s %7.1e %7.1e %7.1e %7.1e %7.1e %7.1e" k 1 fk hk sqrt_ξ1_νInv sqrt(abs(ξ1)) "" σk norm(xk) norm(s) λmax context.dualGap context.κξ
      #! format: on
      @info "iR2N: terminating with √(ξ1/ν) = $(sqrt_ξ1_νInv)"
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

  context.prox_stats[1] = k

  stats = GenericExecutionStats(f)
  set_status!(stats, status)
  set_solution!(stats, xk)
  set_objective!(stats, fk + hk)
  set_residuals!(stats, zero(eltype(xk)), sqrt_ξ1_νInv)
  set_iter!(stats, k)
  set_time!(stats, elapsed_time)
  set_solver_specific!(stats, :Fhist, Fobj_hist[1:k])
  set_solver_specific!(stats, :Hhist, Hobj_hist[1:k])
  set_solver_specific!(stats, :NonSmooth, h)
  set_solver_specific!(stats, :SubsolverCounter, Complex_hist[1:k])
  set_solver_specific!(stats, :ItersProx, context.prox_stats)

  return stats
end