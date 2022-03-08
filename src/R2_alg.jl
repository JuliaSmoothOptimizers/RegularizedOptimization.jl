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
* `options::ROSolverOptions`: a structure containing algorithmic parameters
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
  xk, k, outdict = R2(x -> obj(nlp, x), (g, x) -> grad!(nlp, x, g), args..., x0; kwargs_dict...)

  return GenericExecutionStats(
    outdict[:status],
    nlp,
    solution = xk,
    objective = outdict[:fk] + outdict[:hk],
    dual_feas = sqrt(outdict[:ξ]),
    iter = k,
    elapsed_time = outdict[:elapsed_time],
    solver_specific = Dict(
      :Fhist => outdict[:Fhist],
      :Hhist => outdict[:Hhist],
      :NonSmooth => outdict[:NonSmooth],
      :SubsolverCounter => outdict[:Chist],
    ),
  )
end

function R2(
  f::F,
  ∇f!::G,
  h::ProximableFunction,
  options::ROSolverOptions,
  x0::AbstractVector,
) where {F <: Function, G <: Function}
  start_time = time()
  elapsed_time = 0.0
  ϵ = options.ϵ
  verbose = options.verbose
  maxIter = options.maxIter
  maxTime = options.maxTime
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
  hk = h(xk)
  if hk == Inf
    verbose > 0 && @info "R2: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, x0, one(eltype(x0)))
    hk = h(xk)
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "R2: found point where h has value" hk
  end
  hk == -Inf && error("nonsmooth term is not proper")

  xkn = similar(xk)
  s = zero(xk)
  ψ = shifted(h, xk)

  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, maxIter)
  verbose == 0 ||
    @info @sprintf "%6s %8s %8s %7s %8s %7s %7s %7s %1s" "iter" "f(x)" "h(x)" "√ξ" "ρ" "σ" "‖x‖" "‖s‖" ""

  local ξ
  k = 0
  σk = 1 / ν

  fk = f(xk)
  ∇fk = similar(xk)
  ∇f!(∇fk, xk)
  mν∇fk = -ν * ∇fk

  optimal = false
  tired = maxIter > 0 && k ≥ maxIter || elapsed_time > maxTime

  while !(optimal || tired)
    k = k + 1
    elapsed_time = time() - start_time
    Fobj_hist[k] = fk
    Hobj_hist[k] = hk

    # define model
    φk(d) = dot(∇fk, d)
    mk(d) = φk(d) + ψ(d)

    prox!(s, ψ, mν∇fk, ν)
    Complex_hist[k] += 1
    mks = mk(s)
    ξ = hk - mks - σk * dot(s, s) / 2 + max(1, abs(hk)) * 10 * eps()
    ξ > 0 || error("R2: prox-gradient step should produce a decrease but ξ = $(ξ)")

    if sqrt(ξ) < ϵ
      optimal = true
      verbose == 0 || @info "R2: terminating with ξ = $ξ"
      continue
    end

    xkn .= xk .+ s
    fkn = f(xkn)
    hkn = h(xkn)
    hkn == -Inf && error("nonsmooth term is not proper")

    Δobj = (fk + hk) - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ρk = Δobj / (hk - mks +  max(1, abs(hk)) * 10 * eps())

    σ_stat = (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")

    if (verbose > 0) && (k % ptf == 0)
      @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e %7.1e %7.1e %1s" k fk hk sqrt(ξ) ρk σk norm(
        xk,
      ) norm(s) σ_stat
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
    end

    if ρk < η1 || ρk == Inf
      σk = σk * γ
    end

    ν = 1 / σk
    tired = maxIter > 0 && k ≥ maxIter || elapsed_time > maxTime
    if !tired
      @. mν∇fk = -ν * ∇fk
    end
  end

  if (verbose > 0) && (k == 1)
    @info @sprintf "%6d %8.1e %8.1e" k fk hk
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
  outdict = Dict(
    :Fhist => Fobj_hist[1:k],
    :Hhist => Hobj_hist[1:k],
    :Chist => Complex_hist[1:k],
    :NonSmooth => h,
    :status => status,
    :fk => fk,
    :hk => hk,
    :ξ => ξ,
    :elapsed_time => elapsed_time,
  )

  return xk, k, outdict
end
