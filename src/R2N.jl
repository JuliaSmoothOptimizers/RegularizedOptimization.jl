export R2N

"""
R2N(nlp, h, χ, options; kwargs...)

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
* `subsolver_options::ROSolverOptions`: default options to pass to the subsolver (default: all defaut options)
* `selected::AbstractVector{<:Integer}`: (default `1:f.meta.nvar`).

### Return values

* `xk`: the final iterate
* `Fobj_hist`: an array with the history of values of the smooth objective
* `Hobj_hist`: an array with the history of values of the nonsmooth objective
* `Complex_hist`: an array with the history of number of inner iterations.
"""
function R2N(
  f::AbstractNLPModel,
  h::H,
  options::ROSolverOptions{R};
  x0::AbstractVector = f.meta.x0,
  subsolver_logger::Logging.AbstractLogger = Logging.NullLogger(),
  subsolver = R2,
  subsolver_options = ROSolverOptions(ϵa = options.ϵa),
  selected::AbstractVector{<:Integer} = 1:(f.meta.nvar),
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
  σk = σmin
  xk = copy(x0)
  hk = h(xk[selected])
  if hk == Inf
    verbose > 0 && @info "R2N: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, x0, one(eltype(x0)))
    hk = h(xk[selected])
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "R2N: found point where h has value" hk
  end
  hk == -Inf && error("nonsmooth term is not proper")

  xkn = similar(xk)
  s = zero(xk)
  ψ = has_bounds(f) ? shifted(h, xk, l_bound - xk, u_bound - xk, selected) : shifted(h, xk)

  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, maxIter)
  if verbose > 0
    #! format: off
    @info @sprintf "%6s %8s %8s %8s %7s %7s %8s %7s %7s %7s %7s %1s" "outer" "inner" "f(x)" "h(x)" "√(ξ1/ν)" "√ξ" "ρ" "σ" "‖x‖" "‖s‖" "‖Bₖ‖" "R2N"
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
  νInv = (1 + θ) *( σk + λmax)
  sqrt_ξ1_νInv = one(R)

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
    prox!(s, ψ, -subsolver_options.ν * ∇fk, subsolver_options.ν)
    ξ1 = hk - mk1(s) + max(1, abs(hk)) * 10 * eps()
    ξ1 > 0 || error("R2N: first prox-gradient step should produce a decrease but ξ1 = $(ξ1)")
    sqrt_ξ1_νInv = sqrt(ξ1 * νInv)
    # println("sqrt_ξ1_νInv: ", sqrt_ξ1_νInv)
    # println("ξ1: ", ξ1)
    # println("νInv: ", νInv)

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
    s1 = copy(s)

  #  subsolver_options.ϵa = k == 1 ? 1.0e-1 : max(ϵ_subsolver, min(1.0e-2, ξ1 / 10))
    subsolver_options.ϵa = k == 1 ? 1.0e-3 : max(ϵ_subsolver, min(1e-3, sqrt_ξ1_νInv)) # 1.0e-5 default
    @debug "setting inner stopping tolerance to" subsolver_options.optTol
    subsolver_args = subsolver == R2DH ? (SpectralGradient(1., f.meta.nvar),) : ()
    s, iter, _ = with_logger(subsolver_logger) do
      subsolver(φ, ∇φ!, ψ, subsolver_args..., subsolver_options, s)
    end

    if norm(s) > β * norm(s1)
      s .= s1
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
    Δobj = fk + hk - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ξ = hk - mks + max(1, abs(hk)) * 10 * eps()

    if (ξ ≤ 0 || isnan(ξ))
      error("R2N: failed to compute a step: ξ = $ξ")
    end

    ρk = Δobj / ξ

    R2N_stat = (η2 ≤ ρk < Inf) ? "↗" : (ρk < η1 ? "↘" : "=")

    if (verbose > 0) && (k % ptf == 0)
      #! format: off
      @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s" k iter fk hk sqrt_ξ1_νInv sqrt(ξ1) ρk σk norm(xk) norm(s) λmax R2N_stat
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
      ∇fk⁻ .= ∇fk
    end

    if ρk < η1 || ρk == Inf
        σk = σk * γ
    end
    νInv = (1 + θ) *( σk + λmax)

    tired = k ≥ maxIter || elapsed_time > maxTime
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8s %8.1e %8.1e" k "" fk hk
    elseif optimal
      #! format: off
      @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8s %7.1e %7.1e %7.1e %7.1e" k 1 fk hk sqrt_ξ1_νInv sqrt(ξ1) "" σk norm(xk) norm(s) λmax
      #! format: on
      @info "R2N: terminating with √(ξ1/ν) = $(sqrt_ξ1_νInv)"
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
  return stats
end