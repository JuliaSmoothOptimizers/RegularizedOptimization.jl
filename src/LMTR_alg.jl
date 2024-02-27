export LMTR

"""
    LMTR(nls, h, χ, options; kwargs...)

A trust-region Levenberg-Marquardt method for the problem

    min ½ ‖F(x)‖² + h(x)

where F: ℝⁿ → ℝᵐ and its Jacobian J are Lipschitz continuous and h: ℝⁿ → ℝ is
lower semi-continuous and proper.

At each iteration, a step s is computed as an approximate solution of

    min  ½ ‖J(x) s + F(x)‖₂² + ψ(s; x)  subject to  ‖s‖ ≤ Δ

where F(x) and J(x) are the residual and its Jacobian at x, respectively, ψ(s; x) = h(x + s),
‖⋅‖ is a user-defined norm and Δ > 0 is a trust-region radius.

### Arguments

* `nls::AbstractNLSModel`: a smooth nonlinear least-squares problem
* `h`: a regularizer such as those defined in ProximalOperators
* `χ`: a norm used to define the trust region in the form of a regularizer
* `options::ROSolverOptions`: a structure containing algorithmic parameters

### Keyword arguments

* `x0::AbstractVector`: an initial guess (default: `nls.meta.x0`)
* `subsolver_logger::AbstractLogger`: a logger to pass to the subproblem solver
* `subsolver`: the procedure used to compute a step (`PG`, `R2` or `TRDH`)
* `subsolver_options::ROSolverOptions`: default options to pass to the subsolver.
* `selected::AbstractVector{<:Integer}`: (default `1:nls.meta.nvar`).

### Return values

* `xk`: the final iterate
* `Fobj_hist`: an array with the history of values of the smooth objective
* `Hobj_hist`: an array with the history of values of the nonsmooth objective
* `Complex_hist`: an array with the history of number of inner iterations.
"""
function LMTR(
  nls::AbstractNLSModel,
  h::H,
  χ::X,
  options::ROSolverOptions;
  x0::AbstractVector = nls.meta.x0,
  subsolver_logger::Logging.AbstractLogger = Logging.NullLogger(),
  subsolver = R2,
  subsolver_options = ROSolverOptions(ϵa = options.ϵa),
  selected::AbstractVector{<:Integer} = 1:(nls.meta.nvar),
) where {H, X}
  start_time = time()
  elapsed_time = 0.0
  # initialize passed options
  ϵ = options.ϵa
  ϵ_subsolver = subsolver_options.ϵa
  ϵr = options.ϵr
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

  # store initial values of the subsolver_options fields that will be modified
  ν_subsolver = subsolver_options.ν
  ϵa_subsolver = subsolver_options.ϵa
  Δk_subsolver = subsolver_options.Δk

  local l_bound, u_bound
  treats_bounds = has_bounds(nls) || subsolver == TRDH
  if treats_bounds
    l_bound = nls.meta.lvar
    u_bound = nls.meta.uvar
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
  xk = copy(x0)
  hk = h(xk[selected])
  if hk == Inf
    verbose > 0 && @info "LMTR: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, x0, one(eltype(x0)))
    hk = h(xk[selected])
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "LMTR: found point where h has value" hk
  end
  hk == -Inf && error("nonsmooth term is not proper")

  xkn = similar(xk)
  s = zero(xk)
  ψ =
    treats_bounds ? shifted(h, xk, max.(-Δk, l_bound - xk), min.(Δk, u_bound - xk), selected) :
    shifted(h, xk, Δk, χ)

  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, maxIter)
  Grad_hist = zeros(Int, maxIter)
  Resid_hist = zeros(Int, maxIter)

  if verbose > 0
    #! format: off
    @info @sprintf "%6s %8s %8s %8s %7s %7s %8s %7s %7s %7s %7s %1s" "outer" "inner" "f(x)" "h(x)" "√ξ1" "√ξ" "ρ" "Δ" "‖x‖" "‖s‖" "1/ν" "TR"
    #! format: on
  end

  local ξ1
  k = 0

  Fk = residual(nls, xk)
  Fkn = similar(Fk)
  fk = dot(Fk, Fk) / 2
  Jk = jac_op_residual(nls, xk)
  ∇fk = Jk' * Fk
  JdFk = similar(Fk)   # temporary storage
  Jt_Fk = similar(∇fk)   # temporary storage

  σmax = opnorm(Jk)
  νInv = (1 + θ) * σmax^2  # ‖J'J‖ = ‖J‖²

  mν∇fk = -∇fk / νInv

  optimal = false
  tired = k ≥ maxIter || elapsed_time > maxTime

  while !(optimal || tired)
    k = k + 1
    elapsed_time = time() - start_time
    Fobj_hist[k] = fk
    Hobj_hist[k] = hk
    Grad_hist[k] = nls.counters.neval_jtprod_residual + nls.counters.neval_jprod_residual
    Resid_hist[k] = nls.counters.neval_residual

    # model for first prox-gradient iteration
    φ1(d) = begin
      jtprod_residual!(nls, xk, Fk, Jt_Fk)
      dot(Fk, Fk) / 2 + dot(Jt_Fk, d)
    end

    mk1(d) = φ1(d) + ψ(d)

    # TODO: reuse residual computation
    # model for subsequent prox-gradient iterations
    φ(d) = begin
      jprod_residual!(nls, xk, d, JdFk)
      JdFk .+= Fk
      dot(JdFk, JdFk) / 2
    end

    ∇φ!(g, d) = begin
      jprod_residual!(nls, xk, d, JdFk)
      JdFk .+= Fk
      jtprod_residual!(nls, xk, JdFk, g)
      g
    end

    mk(d) = φ(d) + ψ(d)

    # Take first proximal gradient step s1 and see if current xk is nearly stationary.
    # s1 minimizes φ1(d) + ‖d‖² / 2 / ν + ψ(d) ⟺ s1 ∈ prox{νψ}(-ν∇φ1(0))
    ν = 1 / (νInv + 1 / (Δk * α))
    prox!(s, ψ, mν∇fk, ν)
    ξ1 = fk + hk - mk1(s) + max(1, abs(fk + hk)) * 10 * eps()
    ξ1 > 0 || error("LMTR: first prox-gradient step should produce a decrease but ξ1 = $(ξ1)")

    if ξ1 ≥ 0 && k == 1
      ϵ_increment = ϵr * sqrt(ξ1)
      ϵ += ϵ_increment  # make stopping test absolute and relative
      ϵ_subsolver += ϵ_increment
    end

    if sqrt(ξ1) < ϵ
      # the current xk is approximately first-order stationary
      optimal = true
      continue
    end

    subsolver_options.ϵa = k == 1 ? 1.0e-5 : max(ϵ_subsolver, min(1.0e-1, ξ1 / 10))
    ∆_effective = min(β * χ(s), Δk)
    treats_bounds ?
    set_bounds!(ψ, max.(-∆_effective, l_bound - xk), min.(∆_effective, u_bound - xk)) :
    set_radius!(ψ, ∆_effective)
    subsolver_options.Δk = ∆_effective / 10
    subsolver_options.ν = ν
    subsolver_args = subsolver == TRDH ? (SpectralGradient(1 / ν, nls.meta.nvar),) : ()
    s, iter, _ = with_logger(subsolver_logger) do
      subsolver(φ, ∇φ!, ψ, subsolver_args..., subsolver_options, s)
    end
    # restore initial values of subsolver_options here so that it is not modified
    # if there is an error
    subsolver_options.ν = ν_subsolver
    subsolver_options.ϵa = ϵa_subsolver
    subsolver_options.Δk = Δk_subsolver

    Complex_hist[k] = iter

    sNorm = χ(s)
    xkn .= xk .+ s
    residual!(nls, xkn, Fkn)
    fkn = dot(Fkn, Fkn) / 2
    hkn = h(xkn[selected])
    hkn == -Inf && error("nonsmooth term is not proper")

    Δobj = fk + hk - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ξ = fk + hk - mk(s) + max(1, abs(fk + hk)) * 10 * eps() # TODO: isn't mk(s) returned by subsolver?

    @debug "computed step" s norm(s, Inf) Δk

    if (ξ ≤ 0 || isnan(ξ))
      error("LMTR: failed to compute a step: ξ = $ξ")
    end

    ρk = Δobj / ξ

    TR_stat = (η2 ≤ ρk < Inf) ? "↗" : (ρk < η1 ? "↘" : "=")

    if (verbose > 0) && (k % ptf == 0)
      #! format: off
      @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s" k iter fk hk sqrt(ξ1) sqrt(ξ) ρk ∆_effective χ(xk) sNorm νInv TR_stat
      #! format: on
    end

    if η2 ≤ ρk < Inf
      Δk = max(Δk, γ * sNorm)
      !treats_bounds && set_radius!(ψ, Δk)
    end

    if η1 ≤ ρk < Inf
      xk .= xkn
      treats_bounds && set_bounds!(ψ, max.(-Δk, l_bound - xk), min.(Δk, u_bound - xk))

      #update functions
      Fk .= Fkn
      fk = fkn
      hk = hkn

      shift!(ψ, xk)
      Jk = jac_op_residual(nls, xk)
      jtprod_residual!(nls, xk, Fk, ∇fk)
      σmax = opnorm(Jk)
      νInv = (1 + θ) * σmax^2  # ‖J'J‖ = ‖J‖²
      @. mν∇fk = -∇fk / νInv
    end

    if ρk < η1 || ρk == Inf
      Δk = Δk / 2
      treats_bounds ? set_bounds!(ψ, max.(-Δk, l_bound - xk), min.(Δk, u_bound - xk)) :
      set_radius!(ψ, Δk)
    end

    tired = k ≥ maxIter || elapsed_time > maxTime
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8s %8.1e %8.1e" k "" fk hk
    elseif optimal
      #! format: off
      @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8s %7.1e %7.1e %7.1e %7.1e" k 1 fk hk sqrt(ξ1) sqrt(ξ1) "" Δk χ(xk) χ(s) νInv
      #! format: on
      @info "LMTR: terminating with √ξ1 = $(sqrt(ξ1))"
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

  stats = GenericExecutionStats(nls)
  set_status!(stats, status)
  set_solution!(stats, xk)
  set_objective!(stats, fk + hk)
  set_residuals!(stats, zero(eltype(xk)), ξ1 ≥ 0 ? sqrt(ξ1) : ξ1)
  set_iter!(stats, k)
  set_time!(stats, elapsed_time)
  set_solver_specific!(stats, :Fhist, Fobj_hist[1:k])
  set_solver_specific!(stats, :Hhist, Hobj_hist[1:k])
  set_solver_specific!(stats, :NonSmooth, h)
  set_solver_specific!(stats, :SubsolverCounter, Complex_hist[1:k])
  set_solver_specific!(stats, :NLSGradHist, Grad_hist[1:k])
  set_solver_specific!(stats, :ResidHist, Resid_hist[1:k])
  return stats
end
