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
* `h::ProximableFunction`: a regularizer
* `χ::ProximableFunction`: a norm used to define the trust region
* `options::TRNCoptions`: a structure containing algorithmic parameters

### Keyword arguments

* `x0::AbstractVector`: an initial guess (default: `nls.meta.x0`)
* `subsolver_logger::AbstractLogger`: a logger to pass to the subproblem solver
* `subsolver`: the procedure used to compute a step (`PG` or `R2`)
* `subsolver_options::TRNCoptions`: default options to pass to the subsolver.

### Return values

* `xk`: the final iterate
* `Fobj_hist`: an array with the history of values of the smooth objective
* `Hobj_hist`: an array with the history of values of the nonsmooth objective
* `Complex_hist`: an array with the history of number of inner iterations.
"""
function LMTR(
  nls::AbstractNLSModel,
  h::ProximableFunction,
  χ::ProximableFunction,
  options::TRNCoptions;
  x0::AbstractVector = nls.meta.x0,
  subsolver_logger::Logging.AbstractLogger = Logging.NullLogger(),
  subsolver = R2,
  subsolver_options = TRNCoptions()
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

  # initialize parameters
  xk = copy(x0)
  xkn = similar(xk)
  s = zero(xk)
  ψ = shifted(h, xk, Δk, χ)

  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, (2, maxIter))
  verbose == 0 || @info @sprintf "%6s %8s %8s %8s %7s %7s %8s %7s %7s %7s %7s %1s" "outer" "inner" "f(x)" "h(x)" "√ξ1" "√ξ" "ρ" "Δ" "‖x‖" "‖s‖" "1/ν" "TR"

  local ξ1
  k = 0
  α = 1.0

  # main algorithm initialization
  Fk = residual(nls, xk)
  Fkn = similar(Fk)
  fk = dot(Fk, Fk) / 2
  Jk = jac_op_residual(nls, xk)
  ∇fk = Jk' * Fk
  JdFk = similar(Fk)   # temporary storage
  svd_info = svds(Jk, nsv=1, ritzvec=false)
  νInv = (1 + θ) * maximum(svd_info[1].S)^2  # ‖J'J‖ = ‖J‖²
  mν∇fk = -∇fk/νInv
  hk = h(xk)
  funEvals = 1

  optimal = false
  tired = k ≥ maxIter

  while !(optimal || tired)
    k = k + 1

    Fobj_hist[k] = fk
    Hobj_hist[k] = hk

    # TODO: reuse residual computation
    φ(d) = begin
      jprod_residual!(nls, xk, d, JdFk)
      JdFk .+= Fk
      return dot(JdFk, JdFk) / 2
    end

    ∇φ!(g,d) = begin
      jprod_residual!(nls, xk, d, JdFk)
      JdFk .+= Fk
      jtprod_residual!(nls, xk, JdFk, g) #profiler
      return g
    end

    mk(d) = φ(d) + ψ(d)

    # take first proximal gradient step s1 and see if current xk is nearly stationary
    subsolver_options.ν = 1 / (νInv + 1/(Δk*α))
    s1 = ShiftedProximalOperators.prox(ψ, mν∇fk, subsolver_options.ν)
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
    s, sub_fhist, sub_hhist, sub_cmplx, sub_ξ = with_logger(subsolver_logger) do
      subsolver(φ, ∇φ!, ψ, subsolver_options, s1)
    end

    Complex_hist[2,k] += length(sub_fhist)

    sNorm = χ(s)
    xkn .= xk .+ s
    residual!(nls, xkn, Fkn)
    fkn = dot(Fkn, Fkn) / 2
    hkn = h(xkn)

    Δobj = fk + hk - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ξ = fk + hk - mk(s) + max(1, abs(fk + hk)) * 10 * eps() # TODO: isn't mk(s) returned by subsolver?

    @debug "computed step" s norm(s, Inf) Δk

    if (ξ ≤ 0 || isnan(ξ))
      error("LMTR: failed to compute a step: ξ = $ξ")
    end

    ρk = Δobj / ξ

    TR_stat = (η2 ≤ ρk < Inf) ? "↗" : (ρk < η1 ? "↘" : "=")

    if (verbose > 0) && (k % ptf == 0)
      @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s" k length(sub_fhist) fk hk sqrt(ξ1) sqrt(ξ) ρk Δk χ(xk) sNorm νInv TR_stat
    end

    if η2 ≤ ρk < Inf
      Δk = max(Δk, γ * sNorm)
      set_radius!(ψ, Δk)
    end

    if η1 ≤ ρk < Inf
      xk .= xkn

      #update functions
      Fk .= Fkn
      fk = fkn
      hk = hkn

      shift!(ψ, xk)
      Jk = jac_op_residual(nls, xk)
      jtprod_residual!(nls, xk, Fk, ∇fk)
      svd_info = svds(Jk, nsv=1, ritzvec=false)
      νInv = (1 + θ) * maximum(svd_info[1].S)^2
      @. mν∇fk = -∇fk/νInv
      Complex_hist[1,k] += 1
    end

    if ρk < η1 || ρk == Inf
      α = .5
      Δk = α * Δk
      set_radius!(ψ, Δk)
    end

    tired = k ≥ maxIter
  end

  if (verbose > 0) && (k == 1)
    @info @sprintf "%6d %8s %8.1e %8.1e" k "" fk hk
  end

  return xk, Fobj_hist[1:k], Hobj_hist[1:k], Complex_hist[:,1:k], ξ1
end
