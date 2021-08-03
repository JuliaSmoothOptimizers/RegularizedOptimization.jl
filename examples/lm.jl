using Printf
using Logging
using Arpack
using ShiftedProximalOperators

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
* `params::TRNCMethods`: insert description here
* `options::TRNCParams`: insert description here
### Keyword arguments
* `x0::AbstractVector`: an initial guess (default: the initial guess stored in `nls`)
* `subsolver_logger::AbstractLogger`: a logger to pass to the subproblem solver.
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
  params,
  options;
  x0::AbstractVector=nls.meta.x0,
  subsolver_logger::Logging.AbstractLogger=Logging.NullLogger()
 )
  # initialize passed options
  ϵ = options.ϵ
  Δk = options.Δk
  verbose = options.verbose
  maxIter = options.maxIter
  η1 = options.η1
  η2 = options.η2
  γ = options.γ
  θ = options.θ
  β = options.β

  if verbose == 0
    ptf = Inf
  elseif verbose == 1
    ptf = round(maxIter / 10)
  elseif verbose == 2
    ptf = round(maxIter / 100)
  else
    ptf = 1
  end

  # other parameters
  FO_options = params.FO_options
  s_alg = params.s_alg
  χ = params.χ

  # initialize parameters
  xk = copy(x0)
  xkn = similar(xk)
  s = zero(xk)
  ψ = shifted(h, xk, Δk, χ)

  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, maxIter)
  @info @sprintf "%6s %8s %8s %8s %7s %7s %8s %7s %7s %7s %7s %1s" "iter" "PG iter" "f(x)" "h(x)" "ξ" "Δm" "ρ" "Δ" "‖x‖" "‖s‖" "1/ν" "TR"

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
    k % ptf == 0 && @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s" k funEvals fk hk ξ1 ξ ρk Δk χ(xk) sNorm νInv TR_stat

    # define inner function
    φ(d) = begin
      JdFk = Jk * d + Fk
      return dot(JdFk, JdFk) / 2
      # return [dot(JdFk, JdFk) / 2, Jk' * JdFk, nothing]
    end

    ∇φ(d) = begin
      JdFk = Jk * d + Fk
      return Jk' * JdFk
    end

    mk(d) = φ(d) + ψ(d)

    # take first proximal gradient step s1 and see if current xk is nearly stationary
    FO_options.ν = min(1 / νInv, Δk)
    s1 = ShiftedProximalOperators.prox(ψ, -FO_options.ν * ∇fk, FO_options.ν)
    ξ1 = fk + hk - mk(s1)
    ξ1 > 0 || error("LMTR: first prox-gradient step should produce a decrease but ξ1 = $(ξ1)")

    if ξ1 < ϵ
      # the current xk is approximately first-order stationary
      optimal = true
      @info "LMTR: terminating with ξ1 = $(ξ1)"
      continue
    end

    FO_options.optTol = k == 1 ? 1.0e-5 : max(ϵ, min(1.0e-1, ξ1 / 10))
    set_radius!(ψ, min(β * χ(s1), Δk))
    inner_params = TRNCmethods(FO_options=params.FO_options, χ=χ)
    inner_options = TRNCparams(; maxIter=100, verbose=10, ϵ=FO_options.optTol, σk=1 / Δk)
    s, funEvals, _, _, _ = QRalg2(φ, ∇φ, ψ, s1, inner_params, inner_options)
    # s, funEvals, _, _, _ = with_logger(subsolver_logger) do
    #   QRalg2(φ, ∇φ, ψ, s1, inner_params, inner_options)
    # end
    # (s, funEvals) = s_alg(φ, ∇φ, ψ, s1, FO_options)

    Complex_hist[k] += funEvals

    sNorm = χ(s)
    xkn .= xk .+ s
    Fkn = residual(nls, xkn)  # TODO: use residual!()
    fkn = dot(Fkn, Fkn) / 2
    hkn = h(xkn)

    Δobj = fk + hk - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ξ = fk + hk - mk(s)  # TODO: isn't mk(s) returned by s_alg?

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

      Complex_hist[k] += 1
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


"""
    LM(nls, params, options; x0=nls.meta.x0, subsolver_logger=Logging.NullLogger())
A Levenberg-Marquardt method for the problem
    min ½ ‖F(x)‖² + h(x)
where F: ℜⁿ → ℜᵐ and its Jacobian J are Lipschitz continuous and h: ℜⁿ → ℜ is lower semi-continuous.
At each iteration, a step s is computed as an approximate solution of
    min  ½ ‖J(x) s + F(x)‖² + ½ σ ‖s‖² + ψ(s; x)
where F(x) and J(x) are the residual and its Jacobian at x, respectively, ψ(s; x) = h(x + s),
and σ > 0 is a regularization parameter.
### Arguments
* `nls::AbstractNLSModel`: a smooth nonlinear least-squares problem
* `params::TRNCMethods`: insert description here
* `options::TRNCParams`: insert description here
### Keyword arguments
* `x0::AbstractVector`: an initial guess (default: the initial guess stored in `nls`)
* `subsolver_logger::AbstractLogger`: a logger to pass to the subproblem solver.
### Return values
* `xk`: the final iterate
* `k`: the overall number of iterations
* `Fobj_hist`: an array with the history of values of the smooth objective
* `Hobj_hist`: an array with the history of values of the nonsmooth objective
* `Complex_hist`: an array with the history of number of inner iterations.
"""
# params <- methods
# options <- params
function LM(nls::AbstractNLSModel, h::ProximableFunction, params, options; x0::AbstractVector=nls.meta.x0, subsolver_logger=Logging.NullLogger())
  # initialize passed options
  ϵ = options.ϵ
  σk = options.σk
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

  # other parameters
  FO_options = params.FO_options
  s_alg = params.s_alg
  χ = params.χ

  # initialize parameters
  xk = copy(x0)
  xkn = similar(xk)
  ψ = shifted(h, xk)

  k = 0
  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, maxIter)
  @info @sprintf "%6s %8s %8s %8s %7s %7s %8s %7s %7s %7s %7s %1s" "iter" "PG iter" "f(x)" "h(x)" "ξ" "Δm" "ρ" "σ" "‖x‖" "‖s‖" "‖Jₖ‖²" "reg"

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
    k % ptf == 0 && @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s" k funEvals fk hk ξ1 ξ ρk σk χ(xk) χ(s) νInv σ_stat

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
    FO_options.ν = 1 / νInv
    s1 = ShiftedProximalOperators.prox(ψ, -FO_options.ν * ∇fk, FO_options.ν)
    ξ1 = fk + hk - mk(s1)  # TODO: isn't mk(s) returned by s_alg?
    ξ1 > 0 || error("LM: first prox-gradient step should produce a decrease!")

    if ξ1 < ϵ
      # the current xk is approximately first-order stationary
      @info "LM: terminating with ξ1 = $(ξ1)"
      optimal = true
      continue
    end

    FO_options.optTol = k == 1 ? 1.0e-4 : max(ϵ, min(1.0e-4, ξ1 / 5))
    FO_options.FcnDec = ξ1
    inner_params = TRNCmethods(FO_options=params.FO_options, χ=χ)
    inner_options = TRNCparams(; maxIter=20000, verbose=10, ϵ=FO_options.optTol, σk=1.0)
    s, funEvals, _, _, _ = with_logger(subsolver_logger) do
      QRalg(φ, ∇φ, ψ, s1, inner_params, inner_options)
    end

    Complex_hist[k] += funEvals

    xkn .= xk .+ s
    Fkn = residual(nls, xkn)  # TODO: call residual!()
    fkn = dot(Fkn, Fkn) / 2
    hkn = h(xkn)

    ξ = fk + hk - mk(s)  # TODO: isn't mk(s) returned by s_alg?

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
