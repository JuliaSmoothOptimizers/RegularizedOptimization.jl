export TR
"""
    TR(nlp, params, options; x0=nls.meta.x0, subsolver_logger=Logging.NullLogger())

A trust-region method for the problem

    min f(x) + h(x)

where f: ℜⁿ → ℜ and h: ℜⁿ → ℜ is lower semi-continuous and proper.

At each iteration, a step s is computed as an approximate solution of

    min  ½ ‖s - (sⱼ - ν∇φ(sⱼ)‖₂² + ψ(s; x)  subject to  ‖s‖ ≤ Δ

where ∇φ is the gradient of the quadratic approximation of f(x) at x, ψ(s; x) = h(x + s),
‖⋅‖ is a user-defined norm and Δ > 0 is a trust-region radius.

### Arguments

* `nlp::AbstractNLPModel`: a smooth optimization problem (only the objective will be accessed)
* `h::ProximableFunction`: a regularizer
* `χ::ProximableFunction`: a norm used to define the trust region
* `params::TRNCMethods`: insert description here

### Keyword arguments

* `x0::AbstractVector`: an initial guess (default: the initial guess stored in `nlp`)
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
function TR(
  f::AbstractNLPModel,
  h::ProximableFunction,
  χ::ProximableFunction,
  options;
  x0::AbstractVector=f.meta.x0,
  subsolver_logger::Logging.AbstractLogger=Logging.NullLogger(),
  s_alg = QRalg,
  subsolver_options = TRNCoptions(),
  )

  # initialize passed options
  ϵ = options.ϵ
  Δk = options.Δk
  verbose = options.verbose
  maxIter = options.maxIter
  η1 = options.η1
  η2 = options.η2
  γ = options.γ
  α = options.α
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
  m = length(xk)
  s = zero(xk)
  ψ = shifted(h, xk, Δk, χ)

  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, (2, maxIter))
  @info @sprintf "%6s %8s %8s %8s %7s %7s %8s %7s %7s %7s %7s %1s" "iter" "PG iter" "f(x)" "h(x)" "√ξ1" "√ξ" "ρ" "Δ" "‖x‖" "‖s‖" "‖Bₖ‖" "TR"

  k = 0
  ρk = -1.0
  sNorm = 0.0
  TR_stat = ""

  # keep track of old values, initialize functions
  ∇fk = grad(f, xk)
  fk = obj(f, xk)
  hk = ψ.h(xk)
  s = zero(xk)
  ∇fk⁻ = copy(∇fk)
  funEvals = 1
  Hist_gradeval = [fk + hk]

  quasiNewtTest = isa(f, QuasiNewtonModel)
  Bk = hess_op(f, xk)
  νInv = (1 + θ) * abs(eigs(Bk; nev=1, v0 = randn(m,), which=:LM)[1][1])

  ξ = 0.0
  ξ1 = 0.0
  optimal = false
  tired = k ≥ maxIter

  while !(optimal || tired)
    k = k + 1
    Fobj_hist[k] = fk
    Hobj_hist[k] = hk
    # Print values
    k % ptf == 0 && @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s" k length(funEvals) fk hk sqrt(ξ1) sqrt(ξ) ρk Δk χ(xk) sNorm νInv TR_stat

    φ(d) = begin
        return 0.5 * (d' * (Bk * d)) + ∇fk' * d
    end

    ∇φ!(g, d) = begin
      mul!(g, Bk, d)
      # g = Bk * d
      g .+= ∇fk
      g
    end

    mk(d) = φ(d) + ψ(d)

    # take first proximal gradient step s1 and see if current xk is nearly stationary
    subsolver_options.ν = 1 / (νInv + 1/(Δk*α))
    s1 = ShiftedProximalOperators.prox(ψ, -subsolver_options.ν * ∇fk, subsolver_options.ν) # -> PG on one step s1
    ξ1 = hk - mk(s1) + max(1, abs(hk)) * 10 * eps()
    ξ1 > 0 || error("TR: first prox-gradient step should produce a decrease but ξ1 = $(ξ1)")

    if sqrt(ξ1) < ϵ
      # the current xk is approximately first-order stationary
      optimal = true
      @info "TR: terminating with ξ1 = $(sqrt(ξ1))"
      continue
    end
    subsolver_options.ϵ = k == 1 ? 1.0e-5 : max(ϵ, min(1e-2, sqrt(ξ1)) * ξ1)
    set_radius!(ψ, min(β * χ(s1), Δk))
    s, funEvals, _, _, _ = with_logger(subsolver_logger) do
      s_alg(φ, ∇φ!, ψ, subsolver_options, s1)
    end
    
    Complex_hist[2,k] += length(funEvals)
    sNorm =  χ(s)
    xkn .= xk .+ s
    fkn = obj(f, xkn)
    hkn = h(xkn)

    Δobj = fk + hk - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ξ = hk - mk(s) + max(1, abs(hk)) * 10 * eps()

    if (ξ ≤ 0 || isnan(ξ))
      error("TR: failed to compute a step: ξ = $ξ")
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
      fk = fkn
      hk = hkn
      shift!(ψ, xk)
      ∇fk = grad(f, xk)
      # grad!(f, xk, ∇fk)
      if quasiNewtTest
        push!(f, s, ∇fk - ∇fk⁻)
      end
      Bk = hess_op(f, xk)
      νInv = (1 + θ) * abs(eigs(Bk; nev=1,  v0 = randn(m,), which=:LM)[1][1])
      # store previous iterates
      ∇fk⁻ .= ∇fk

      #hist update
      Complex_hist[1,k] += 1
      append!(Hist_gradeval, fk+hk)
    end

    if ρk < η1 || ρk == Inf
      TR_stat = "↘"
      Δk = .5 * Δk	# change to reflect trust region
      set_radius!(ψ, Δk)
    end
    tired = k ≥ maxIter

  end

  return xk, Hist_gradeval, Fobj_hist[1:k], Hobj_hist[1:k], Complex_hist[:,1:k]
end
