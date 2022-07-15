export TR

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
* `subsolver`: the procedure used to compute a step (`PG` or `R2`)
* `subsolver_options::ROSolverOptions`: default options to pass to the subsolver (default: all defaut options).

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
  options::ROSolverOptions;
  x0::AbstractVector = f.meta.x0,
  subsolver_logger::Logging.AbstractLogger = Logging.NullLogger(),
  subsolver = R2,
  subsolver_options = ROSolverOptions(),
) where {H, X}
  start_time = time()
  elapsed_time = 0.0
  # initialize passed options
  ϵ = options.ϵ
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

  has_bounds(f) ? (l_bound = f.meta.lvar ; u_bound = f.meta.uvar) : nothing

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
  hk = h(xk)
  if hk == Inf
    verbose > 0 && @info "TR: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, x0, one(eltype(x0)))
    hk = h(xk)
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "TR: found point where h has value" hk
  end
  hk == -Inf && error("nonsmooth term is not proper")

  xkn = similar(xk)
  s = zero(xk)
  has_bounds(f) ? (ψ = shifted(h, xk, max.(-Δk,l_bound-xk), min.(Δk, u_bound-xk))) : (ψ = shifted(h, xk, Δk, χ))

  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, maxIter)
  if verbose > 0
    #! format: off
    if has_bounds(f)
      @info @sprintf "%6s %8s %8s %8s %7s %7s %8s %7s %7s %7s %7s %7s %1s" "outer" "inner" "f(x)" "h(x)" "√ξ1" "√ξ" "ρ" "‖x‖" "‖s‖" "‖Bₖ‖" "TR"
    else
      @info @sprintf "%6s %8s %8s %8s %7s %7s %8s %7s %7s %7s %7s %1s" "outer" "inner" "f(x)" "h(x)" "√ξ1" "√ξ" "ρ" "Δ" "‖x‖" "‖s‖" "‖Bₖ‖" "TR"
    end
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
  νInv = (1 + θ) * λmax

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
    subsolver_options.ν = 1 / (νInv + 1 / (Δk * α))
    prox!(s, ψ, -subsolver_options.ν * ∇fk, subsolver_options.ν)
    ξ1 = hk - mk1(s) + max(1, abs(hk)) * 10 * eps()
    ξ1 > 0 || error("TR: first prox-gradient step should produce a decrease but ξ1 = $(ξ1)")

    if sqrt(ξ1) < ϵ
      # the current xk is approximately first-order stationary
      optimal = true
      continue
    end

    subsolver_options.ϵ = k == 1 ? 1.0e-5 : max(ϵ, min(1e-2, sqrt(ξ1)) * ξ1)
    has_bounds(f) ? set_bounds!(ψ, max.(-min(β * χ(s), Δk), l_bound-xk), min.(min(β * χ(s), Δk), u_bound-xk)) : set_radius!(ψ, min(β * χ(s), Δk))
    s, iter, _ = with_logger(subsolver_logger) do
      subsolver(φ, ∇φ!, ψ, subsolver_options, s)
    end
    Complex_hist[k] = iter

    sNorm = χ(s)
    xkn .= xk .+ s
    fkn = obj(f, xkn)
    hkn = h(xkn)
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
      if has_bounds(f)
        @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8.1e %7.1e %7.1e %7.1e %1s" k iter fk hk sqrt(ξ1) sqrt(ξ) ρk  χ(xk) sNorm νInv TR_stat
      else
        @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s" k iter fk hk sqrt(ξ1) sqrt(ξ) ρk ψ.Δ χ(xk) sNorm νInv TR_stat
      end
      #! format: on
    end

    if η2 ≤ ρk < Inf
      Δk = max(Δk, γ * sNorm)
      has_bounds(f) ? set_bounds!(ψ, max.(-Δk, l_bound-xk), min.(Δk, u_bound-xk)) : set_radius!(ψ, Δk)
    end

    if η1 ≤ ρk < Inf
      xk .= xkn

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
      νInv = (1 + θ) * λmax
      ∇fk⁻ .= ∇fk
    end

    if ρk < η1 || ρk == Inf
      Δk = Δk / 2
      has_bounds(f) ? set_bounds!(ψ, max.(-Δk, l_bound-xk), min.(Δk, u_bound-xk)) : set_radius!(ψ, Δk)
    end
    tired = k ≥ maxIter || elapsed_time > maxTime
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8s %8.1e %8.1e" k "" fk hk
    elseif optimal
      #! format: off
      if has_bounds(f)
        @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8s %7.1e %7.1e %7.1e" k 1 fk hk sqrt(ξ1) sqrt(ξ1) "" χ(xk) χ(s) νInv
      else
        @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8s %7.1e %7.1e %7.1e %7.1e" k 1 fk hk sqrt(ξ1) sqrt(ξ1) "" ψ.Δ χ(xk) χ(s) νInv
      end
      #! format: on
      @info "TR: terminating with √ξ1 = $(sqrt(ξ1))"
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

  return GenericExecutionStats(
    status,
    f,
    solution = xk,
    objective = fk + hk,
    dual_feas = sqrt(ξ1),
    iter = k,
    elapsed_time = elapsed_time,
    solver_specific = Dict(
      :Fhist => Fobj_hist[1:k],
      :Hhist => Hobj_hist[1:k],
      :NonSmooth => h,
      :SubsolverCounter => Complex_hist[1:k],
    ),
  )
end
