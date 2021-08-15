export R2

"""
    R2(nlp, h, options)
    R2(f, ∇f!, h, options, x0)

A first-order quadratic regularization method for the problem

    min f(x) + h(x)

where f: ℝⁿ → ℝ has a Lipschitz-continuous gradient, and h: ℝⁿ → ℝ is
lower semi-continuous, proper and prox-bounded.

About each iterate xₖ, a step sₖ is computed as a solution of

    min  φ(s; xₖ) + ½ σₖ ‖s‖² + ψ(s; xₖ)

where φ(s ; xₖ) = f(xₖ) + ∇f(xₖ)ᵀs is the Taylor linear approximation of f about xₖ,
ψ(s; xₖ) = h(xₖ + s), ‖⋅‖ is a user-defined norm and σₖ > 0 is the regularization parameter.

### Arguments

* `nlp::AbstractNLPModel`: a smooth optimization problem
* `h::ProximableFunction`: a regularizer
* `options::TRNCoptions`: a structure containing algorithmic parameters
* `x0::AbstractVector`: an initial guess (in the second calling form)

### Keyword Arguments

* `x0::AbstractVector`: an initial guess (in the first calling form: default = `nlp.meta.x0`)

The objective and gradient of `nlp` will be accessed.

In the second form, instead of `nlp`, the user may pass in

* `f` a function such that `f(x)` returns the value of f at x
* `∇f!` a function to evaluate the gradient in place, i.e., such that `∇f!(g, x)` store ∇f(x) in `g`.

### Return values

* `xk`: the final iterate
* `Fobj_hist`: an array with the history of values of the smooth objective
* `Hobj_hist`: an array with the history of values of the nonsmooth objective
* `Complex_hist`: an array with the history of number of inner iterations.
"""
function R2(nlp::AbstractNLPModel, args...; kwargs...)
  kwargs_dict = Dict(kwargs...)
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  R2(x -> obj(nlp, x), (g, x) -> grad!(nlp, x, g), args..., x0; kwargs_dict...)
end

function R2(
  f::F,
  ∇f!::G,
  h::ProximableFunction,
  options::TRNCoptions,
  x0::AbstractVector
  ) where {F <: Function, G <: Function}
  ϵ = options.ϵ
  verbose = options.verbose
  maxIter = options.maxIter
  η1 = options.η1
  η2 = options.η2
  ν = options.ν
  γ = options.γ

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
  ψ = shifted(h, xk)

  k = 0
  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, (2,maxIter))
  verbose == 0 || @info @sprintf "%6s %8s %8s %7s %8s %7s %7s %7s %1s" "iter" "f(x)" "h(x)" "√ξ" "ρ" "σ" "‖x‖" "‖s‖" ""

  local ξ
  k = 0
  σk = 1/ν

  fk = f(xk)
  ∇fk = similar(xk)
  ∇f!(∇fk, xk)
  mν∇fk = -ν * ∇fk
  hk = h(xk)

  optimal = false
  tired = maxIter > 0 && k ≥ maxIter

  while !(optimal || tired)
    k = k + 1

    Fobj_hist[k] = fk
    Hobj_hist[k] = hk

    # define model
    φk(d) = dot(∇fk, d)
    mk(d) = φk(d) + ψ(d)

    s = ShiftedProximalOperators.prox(ψ, mν∇fk, ν)
    Complex_hist[2,k] += 1
    mks = mk(s)
    ξ = hk - mks + max(1, abs(hk)) * 10 * eps()

    if (sqrt(ξ) < 0 || isnan(ξ))
      error("QR: failed to compute a step: ξ = $ξ")
    end

    if sqrt(ξ) < ϵ
      optimal = true
      verbose == 0 || @info "QR: terminating with ξ = $ξ"
      continue
    end

    xkn .= xk .+ s
    fkn = f(xkn)
    hkn = h(xkn)
    Δobj = (fk + hk) - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ρk = Δobj / ξ

    σ_stat = (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")

    if (verbose > 0) && (k % ptf == 0)
      @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e %7.1e %7.1e %1s" k fk hk sqrt(ξ) ρk σk norm(xk) norm(s) σ_stat
    end

    if η2 ≤ ρk < Inf
      σk = σk / γ
    end

    if η1 ≤ ρk < Inf
      xk .= xkn
      fk = fkn
      hk = hkn
      ∇f!(∇fk, xk)
      shift!(ψ, xk)
      Complex_hist[1,k]+=1
    end

    if ρk < η1 || ρk == Inf
      σk = σk * γ
    end

    ν = 1 / σk
    tired = maxIter > 0 && k ≥ maxIter
    if !tired
      @. mν∇fk = -ν * ∇fk
    end
  end

  if (verbose > 0) && (k == 1)
    @info @sprintf "%6d %8.1e %8.1e" k fk hk
  end

  return xk, Fobj_hist[1:k], Hobj_hist[1:k], Complex_hist[:,1:k], ξ
end
