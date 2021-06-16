export LM
"""
    LM(nls, options, options; x0=nls.meta.x0, subsolver_logger=Logging.NullLogger())

A Levenberg-Marquardt method for the problem

    min ½ ‖F(x)‖² + h(x)

where F: ℜⁿ → ℜᵐ and its Jacobian J are Lipschitz continuous and h: ℜⁿ → ℜ is lower semi-continuous.

At each iteration, a step s is computed as an approximate solution of

    min  ½ ‖J(x) s + F(x)‖² + ½ σ ‖s‖² + ψ(s; x)

where F(x) and J(x) are the residual and its Jacobian at x, respectively, ψ(s; x) = h(x + s),
and σ > 0 is a regularization parameter.

### Arguments

* `nls::AbstractNLSModel`: a smooth nonlinear least-squares problem
* `h::ProximableFunction`: a regularizer
* `options::TRNCMethods`: insert description here

### Keyword arguments

* `x0::AbstractVector`: an initial guess (default: the initial guess stored in `nls`)
* `subsolver_logger::AbstractLogger`: a logger to pass to the subproblem solver
* `s_alg`: the procedure used to compute a step (`PG` or `QRalg`)
* `subsolver_options::TRNCoptions`: default options to pass to the subsolver.

### Return values

* `xk`: the final iterate
* `k`: the overall number of iterations
* `Fobj_hist`: an array with the history of values of the smooth objective
* `Hobj_hist`: an array with the history of values of the nonsmooth objective
* `Complex_hist`: an array with the history of number of inner iterations.
"""

function LM(
  nls::AbstractNLSModel, 
  h::ProximableFunction, 
  options; 
  x0::AbstractVector=nls.meta.x0, 
  subsolver_logger=Logging.NullLogger(), 
  s_alg=QRalg, 
  subsolver_options = TRNCoptions()
  )
  # initialize passed options
  ϵ = options.ϵ
  σk = 1 / options.ν
  verbose = options.verbose
  maxIter = options.maxIter
  η1 = options.η1
  η2 = options.η2
  γ = options.γ
  θ = options.θ

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
  xkn = similar(xk)
  ψ = shifted(h, xk)

  k = 0
  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, maxIter)
  verbose == 0 || @info @sprintf "%6s %8s %8s %8s %7s %7s %8s %7s %7s %7s %7s %1s" "iter" "PG iter" "f(x)" "h(x)" "ξ" "Δm" "ρ" "σ" "‖x‖" "‖s‖" "‖Jₖ‖²" "reg"

  k = 0
  ρk = -1.0
  α = 1.0
  σ_stat = ""

  # main algorithm initialization
  Fk = residual(nls, xk)
  fk = dot(Fk, Fk) / 2
  Jk = jac_residual(nls, xk)
  ∇fk = Jk' * Fk
  svd_info = svds(Jk, nsv=1, ritzvec=false)
  νInv = (1 + θ) * (maximum(svd_info[1].S)^2 + σk)  # ‖J'J + σₖ I‖ = ‖J‖² + σₖ
  hk = h(xk)
  s = zero(xk)
  funEvals = 1

  ξ1 = 0.0
  ξ = 0.0
  optimal = false
  tired = k ≥ maxIter

  while !(optimal || tired)
    k = k + 1

    Fobj_hist[k] = fk
    Hobj_hist[k] = hk
    k % ptf == 0 && @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s" k funEvals fk hk ξ1 ξ ρk σk norm(xk) norm(s) νInv σ_stat

    # define inner function
    # TODO: use Jacobian operator and do not compute gradient unless necessary
    φ(d) = begin
      JdFk = Jk * d + Fk
      return dot(JdFk, JdFk) / 2 + σk * dot(d, d) / 2
      # return [dot(JdFk, JdFk) / 2 + σk * dot(d, d) / 2, Jk' * JdFk + σk * d, nothing]
    end

    ∇φ(d) = begin
      JdFk = Jk * d + Fk
      return Jk' * JdFk + σk * d
    end

    # define model and update ρ
    mk(d) = begin
      JdFk = Jk * d + Fk
      return dot(JdFk, JdFk) / 2 + ψ(d)
    end

    # take first proximal gradient step s1 and see if current xk is nearly stationary
    subsolver_options.ν = 1 / νInv
    s1 = ShiftedProximalOperators.prox(ψ, -subsolver_options.ν * ∇fk, subsolver_options.ν)
    ξ1 = fk + hk - mk(s1) + max(1, abs(fk + hk)) * 10 * eps()  # TODO: isn't mk(s) returned by s_alg?
    ξ1 > 0 || error("LM: first prox-gradient step should produce a decrease!")

    if ξ1 < ϵ
      # the current xk is approximately first-order stationary
      verbose == 0 || @info "LM: terminating with ξ1 = $(ξ1)"
      optimal = true
      continue
    end

    # subsolver_options.ϵ = k == 1 ? 1.0e-4 : max(ϵ, min(1.0e-4, ξ1 / 5))
    subsolver_options.ϵ = k == 1 ? 1.0e-1 : max(ϵ, min(1.0e-2, ξ1 / 10))
    @debug "setting inner stopping tolerance to" subsolver_options.optTol
    s, funEvals, _, _, _ = with_logger(subsolver_logger) do
      s_alg(φ, ∇φ, ψ, subsolver_options, x0 = s1)
    end

    Complex_hist[k] += funEvals

    xkn .= xk .+ s
    Fkn = residual(nls, xkn)  # TODO: call residual!()
    fkn = dot(Fkn, Fkn) / 2
    hkn = h(xkn)

    ξ = fk + hk - mk(s) + max(1, abs(fk + hk)) * 10 * eps()  # TODO: isn't mk(s) returned by s_alg?

    if (ξ ≤ 0 || isnan(ξ))
      error("LM: failed to compute a step: ξ = $ξ")
    end

    Δobj = fk + hk - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ρk = Δobj / ξ

    if η2 ≤ ρk < Inf
      σ_stat = "↘"
      σk = σk / γ
    else
      σ_stat = "="
    end

    if η1 ≤ ρk < Inf
      xk .= xkn

      # update functions
      Fk .= Fkn
      fk = fkn
      hk = hkn

      #update gradient & hessian
      shift!(ψ, xk)
      Jk = jac_residual(nls, xk)
      mul!(∇fk, Jk', Fk)
      svd_info = svds(Jk, nsv=1, ritzvec=false)
      νInv = (1 + θ) * (maximum(svd_info[1].S)^2 + σk)  # ‖J'J + σₖ I‖ = ‖J‖² + σₖ

      Complex_hist[k] += 1
    end

    if ρk < η1 || ρk == Inf
      σ_stat = "↗"
      σk = max(σk * γ, 1e-6)
    end

    tired = k ≥ maxIter
  end

  return xk, k, Fobj_hist[Fobj_hist .!= 0], Hobj_hist[Fobj_hist .!= 0], Complex_hist[Complex_hist .!= 0]
end
