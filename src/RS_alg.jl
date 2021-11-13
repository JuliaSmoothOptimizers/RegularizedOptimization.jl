export RS1, RS2

"""
    RS1(nlp, h, options)
    RS1(f, ∇f!, h, options, x0)

A first-order quadratic regularization method for the problem

    min ‖Jx + F‖² + ψ(x)

where φ: ℝⁿ → ℝ has a Lipschitz-continuous gradient, and ψ: ℝⁿ → ℝ is
lower semi-continuous, proper and prox-bounded.

We augment the cost function to be

    min  ‖Jx + F‖² + ½ γₖ ‖x - z‖² + ψ(z)

where we then solver for x directly, z is the prox step in the new x,
 γₖ → 0 is the regularization parameter.

### Arguments

* `nlp::AbstractNLPModel`: a smooth optimization problem
* `h::ProximableFunction`: a regularizer
* `options::RegOptoptions`: a structure containing algorithmic parameters
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
function RS1(nlp::AbstractNLPModel, args...; kwargs...)
  kwargs_dict = Dict(kwargs...)
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  xk, k, outdict = RS1(x -> obj(nlp, x), (g, x) -> grad!(nlp, x, g), args..., x0; kwargs_dict...)

  return GenericExecutionStats(
    outdict[:status],
    nlp,
    solution = xk,
    objective = outdict[:fk] + outdict[:hk],
    dual_feas = sqrt(outdict[:ξ]),
    iter = k,
    elapsed_time = outdict[:elapsed_time],
    solver_specific = Dict(:Fhist=>outdict[:Fhist], :Hhist=>outdict[:Hhist], :NonSmooth=>outdict[:NonSmooth], :SubsolverCounter=>outdict[:Chist])
  )
end

function RS1(
  f::F,
  ∇f!::G,
  h::ProximableFunction,
  options::RegOptoptions,
  x0::AbstractVector;
  JtJ,
  JtF
  ) where {F <: Function, G <: Function}
  ϵ = options.ϵ
  verbose = options.verbose
  maxIter = options.maxIter
  ν = options.ν
  γ = options.γ

  if options.verbose==0
      print_freq = Inf
  elseif options.verbose==1
      print_freq = round(max_iter/10)
  elseif options.verbose==2
      print_freq = round(max_iter/100)
  else
      print_freq = 1
  end

  xk = copy(x0)
  zk = zero(xk)

  k = 0
  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, (2,maxIter))
  verbose == 0 || @info @sprintf "%6s %8s %8s %7s %8s %7s %7s" "iter" "f(x)" "h(x)" "‖∂ϕ‖" "ν" "‖x‖" "‖x - z‖"

  fk = f(xk)
  hk = h(xk)
  ∇fk = similar(xk)
  ∇f!(∇fk, xk)
  optimal = false
  tired = maxIter > 0 && k ≥ maxIter

  while !(optimal || tired)
    k = k + 1

    Fobj_hist[k] = fk
    Hobj_hist[k] = hk

    xk .= (JtJ + I/ν)\( z./ν+  JtF)
    z = ShiftedProximalOperators.prox(h, xk, ν)

    fk = f(z)
    hk = h(z)
    ν /= γ
    k % ptf == 0 && @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e %7.1e " k fk hk err ν norm(xk) norm(xk - z)

    err = norm(∇f!(∇fk, xk) + (z - x)/ν) #optimality conditions?
    optimal = err < ϵ && h.χ(xk) < h.Δ
    tired = k ≥ maxIter
    end

    return xk, Fobj_hist[1:k]+Hobj_hist[1:k], Fobj_hist[1:k], Hobj_hist[1:k], Complex_hist[:,1:k]
end

function RS2(nlp::AbstractNLPModel, args...; kwargs...)
  kwargs_dict = Dict(kwargs...)
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  R2(x -> obj(nlp, x), (g, x) -> grad!(nlp, x, g), args..., x0; kwargs_dict...)
end

function  RS2(
  f::F,
  ∇f!::G,
  h::ProximableFunction,
  χ::ProximableFunction,
  options::RegOptoptions,
  x0::AbstractVector;
  JtJ,
  JtF
  ) where {F <: Function, G <: Function}
  ϵ = options.ϵ
  verbose = options.verbose
  maxIter = options.maxIter

  ν = options.ν
  γ = options.γ

  if verbose==0
      print_freq = Inf
  elseif verbose==1
      print_freq = round(max_iter/10)
  elseif verbose==2
      print_freq = round(max_iter/100)
  else
      print_freq = 1
  end

  #Problem Initialize
  xk = copy(x0)
  z1 = zero(xk)
  z2 = zero(xk)
  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, (2,maxIter))

  # Iteration set up
  fk = f(xk)
  hk = h(xk)
  ∇fk = similar(xk)
  ∇f!(∇fk, xk)
  optimal = false
  tired = maxIter > 0 && k ≥ maxIter

  #do iterations
  optimal = false
  tired = k ≥ maxIter

  verbose == 0 || @info @sprintf "%6s %8s %8s %7s %8s %7s %7s" "iter" "f(x)" "h(x)" "‖∂ϕ‖" "ν" "‖x‖" "∑ᵢ ‖x - zᵢ‖"

  while !(optimal || tired)
    k = k + 1

    Fobj_hist[k] = fk
    Hobj_hist[k] = hk

    #solve for xk
    xk = (JtJ + 2*I/ν)\(z1./ν + z2./ν - JtF)

    #update w1
    z1 = ShiftedProximalOperators.prox(h, xk, ν)

    #w2 update
    z2 = ShiftedProximalOperators.prox(χ, xk, h.Δ) #need Δ here

    fk = f(z1)
    hk = h(z1)
    ν /= γ
    k % ptf == 0 && @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e %7.1e " k fk hk err ν norm(xk) norm(xk - z1) + norm(xk - z2)

    err = norm(∇f!(∇fk, xk) + (z - x)/ν + (z - x)ν) #optimality conditions?
    optimal = err < ϵ && h.χ(xk) < h.Δ
    tired = k ≥ maxIter
    end

    return xk, Fobj_hist[1:k]+Hobj_hist[1:k], Fobj_hist[1:k], Hobj_hist[1:k], Complex_hist[:,1:k]

end