export LMTR

"""
    LMTR(nls, params, options; x0=nls.meta.x0, subsolver_logger=Logging.NullLogger())

A trust-region Levenberg-Marquardt method for the problem

    min ½ ‖F(x)‖² + h(x)

where F: ℜⁿ → ℜᵐ and its Jacobian J are Lipschitz continuous and h: ℜⁿ → ℜ is lower semi-continuous.

At each iteration, a step s is computed as an approximate solution of

    min  ½ ‖J(x) s + F(x)‖₂² + ψ(s; x)  subject to  ‖s‖ ≤ Δ

where F(x) and J(x) are the residual and its Jacobian at x, respectively, ψ(s; x) = h(x + s),
‖⋅‖ is a user-defined norm and Δ > 0 is a trust-region radius.

### Arguments

* `nls::AbstractNLSModel`: a smooth nonlinear least-squares problem
* `h::ProximableFunction`: a regularizer
* `χ::ProximableFunction`: a norm used to define the trust region
* `params::TRNCMethods`: insert description here

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
function LMTR(
  nls::AbstractNLSModel,
  h::ProximableFunction,
  χ::ProximableFunction,
  params;
  x0::AbstractVector=nls.meta.x0,
  subsolver_logger::Logging.AbstractLogger=Logging.NullLogger(),
  s_alg=QRalg,
  subsolver_options=TRNCoptions()
 )
  # initialize passed options
  ϵ = params.ϵ
  Δk = params.Δk
  verbose = params.verbose
  maxIter = params.maxIter
  η1 = params.η1
  η2 = params.η2
  γ = params.γ
  θ = params.θ
  β = params.β

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
  s = zero(xk)
  ψ = shifted(h, xk, Δk, χ)

  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, (2, maxIter))
  verbose == 0 || @info @sprintf "%6s %8s %8s %8s %7s %7s %8s %7s %7s %7s %7s %1s" "iter" "PG iter" "f(x)" "h(x)" "√ξ1" "√ξ" "ρ" "Δ" "‖x‖" "‖s‖" "1/ν" "TR"

  k = 0
  ρk = -1.0
  α = 1.0
  sNorm = 0.0
  TR_stat = ""

  # main algorithm initialization
  Fk = residual(nls, xk)  # TODO: use residual!()
  fk = dot(Fk, Fk) / 2
  Jk = jac_residual(nls, xk)  # TODO: use operator
  ∇fk = Jk' * Fk
  svd_info = svds(Jk, nsv=1, ritzvec=false)
  νInv = (1 + θ) * maximum(svd_info[1].S)^2  # ‖J'J‖ = ‖J‖²
  hk = h(xk)
  funEvals = 1

  ξ = 0.0
  ξ1 = 0.0
  optimal = false
  tired = k ≥ maxIter

  while !(optimal || tired)
    k = k + 1

    Fobj_hist[k] = fk
    Hobj_hist[k] = hk
    k % ptf == 0 && @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s" k length(funEvals) fk hk sqrt(ξ1) sqrt(ξ) ρk Δk χ(xk) sNorm νInv TR_stat

    # define inner function
    φ(d) = begin
      JdFk = Jk * d + Fk
      return dot(JdFk, JdFk) / 2
    end

    ∇φ!(g,d) = begin
      JdFk = Jk * d + Fk
      g .= Jk' * JdFk
      return g
    end

    mk(d) = φ(d) + ψ(d)

    # take first proximal gradient step s1 and see if current xk is nearly stationary
    subsolver_options.ν = 1 / (νInv + 1/(Δk*α))
    s1 = ShiftedProximalOperators.prox(ψ, -subsolver_options.ν * ∇fk, subsolver_options.ν)
    ξ1 = fk + hk - mk(s1) + max(1, abs(fk + hk)) * 10 * eps()
    ξ1 > 0 || error("LMTR: first prox-gradient step should produce a decrease but ξ1 = $(ξ1)")

    if sqrt(ξ1) < ϵ
      # the current xk is approximately first-order stationary
      optimal = true
      verbose == 0 || @info "LMTR: terminating with ξ1 = $(ξ1)"
      continue
    end

    subsolver_options.ϵ = k == 1 ? 1.0e-5 : max(ϵ, min(1.0e-1, ξ1 / 10))
    set_radius!(ψ, min(β * χ(s1), Δk))
    s, funEvals, _, _, _ = with_logger(subsolver_logger) do
      s_alg(φ, ∇φ!, ψ, subsolver_options, s1)
    end

    Complex_hist[2,k] += length(funEvals)

    sNorm = χ(s)
    xkn .= xk .+ s
    Fkn = residual(nls, xkn)  # TODO: use residual!()
    fkn = dot(Fkn, Fkn) / 2
    hkn = h(xkn)

    Δobj = fk + hk - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ξ = fk + hk - mk(s) + max(1, abs(fk + hk)) * 10 * eps() # TODO: isn't mk(s) returned by s_alg?

    @debug "computed step" s norm(s, Inf) Δk

    if (ξ ≤ 0 || isnan(ξ))
      error("LMTR: failed to compute a step: ξ = $ξ")
    end

    ρk = Δobj / ξ

    if η2 ≤ ρk < Inf
      TR_stat = "↗"
      Δk = max(Δk, γ * sNorm)
      set_radius!(ψ, Δk)
    else
      TR_stat = "="
    end

    if η1 ≤ ρk < Inf
      xk .= xkn

      #update functions
      Fk .= Fkn
      fk = fkn
      hk = hkn

      shift!(ψ, xk)
      Jk = jac_residual(nls, xk)
      # ∇fk = Jk' * Fk
      mul!(∇fk, Jk', Fk)
      svd_info = svds(Jk, nsv=1, ritzvec=false)
      νInv = (1 + θ) * maximum(svd_info[1].S)^2

      Complex_hist[1,k] += 1
    end

    if ρk < η1 || ρk == Inf
      TR_stat = "↘"
      # Δk = sNorm / γ
      α = .5
      Δk = α * Δk
      set_radius!(ψ, Δk)
    end

    tired = k ≥ maxIter
  end

  return xk, k, Fobj_hist[Fobj_hist .!= 0], Hobj_hist[Fobj_hist .!= 0], Complex_hist[Complex_hist .!= 0]
end