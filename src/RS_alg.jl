export ReSp1, ReSp2

"""
    ReSp1(nlp, h, options)
    ReSp1(f, ∇f!, h, options, x0)

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
function ReSp1(nlp::AbstractNLPModel, args...; kwargs...)
  kwargs_dict = Dict(kwargs...)
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  xk, k, outdict = ReSp1(x -> obj(nlp, x), (g, x) -> grad!(nlp, x, g), args..., x0; kwargs_dict...)

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

function ReSp1(
  f::F,
  ∇f!::G,
  h::ProximableFunction,
  options::ROSolverOptions,
  x0::AbstractVector;
  JtJ,
  JtF,
  ξ1,
  fkhk
  ) where {F <: Function, G <: Function}
  start_time = time()
  elapsed_time = 0.0
  ϵ = options.ϵ
  verbose = options.verbose
  maxIter = options.maxIter
  maxTime = options.maxTime
  η1 = options.η1
  η2 = options.η2
  ν = 1/options.ν #||Bk|| -> being very large at k=0
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

  #initialize parameters
  xk = zero(x0)
  hk = h(xk)
  if hk == Inf
    verbose > 0 && @info "ReSp1: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, x0, one(eltype(x0)))
    hk = h(xk)
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "ReSp1: found point where h has value" hk
  end
  hk == -Inf && error("nonsmooth term is not proper")

  xkn = similar(xk)
  z = zero(xk)
  ψ = shifted(h, xk)

  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, maxIter)
  verbose == 0 || @info @sprintf "%6s %8s %8s %7s %8s %7s %7s" "iter" "f(x)" "h(x)" "ξ" "ν" "‖x‖" "‖x - z‖"

  local ξ
  k = 0

  fk = f(xk)
  # ∇fk = similar(xk) # can we use these instead of passing more arguments?
  # ∇f!(∇fk, xk)

  optimal = false
  tired = maxIter > 0 && k ≥ maxIter || elapsed_time > maxTime

  while !(optimal || tired)
    k = k + 1
    elapsed_time = time() - start_time
    Fobj_hist[k] = fk
    Hobj_hist[k] = hk

    op = JtJ + opDiagonal(size(xk,1), size(xk,1), ones(size(xk))./ν)
    # xk, stats = cg(op, z./ν - JtF) #which one should be faster?
    xk = Matrix(op)\(z./ν - JtF)
    prox!(z, h, xk, ν)
    Complex_hist[k] += 1
    ξ = fkhk - (f(z) + h(z))
    # This isn't a good metric for determining error
    # ξ > 0 || error("RS1: prox-gradient step should produce a decrease but ξ = $(ξ)")
    @show ξ, fkhk - (f(xk) + h(xk)), ξ1, h.Δ - h.χ(z), fkhk, f(z)+h(z)
    if ξ > .01*ξ1 && h.Δ - h.χ(z) ≥ 0
      optimal = true
      verbose == 0 || @info "R2: terminating with ξ = $ξ"
      continue
    end

    fk = f(z)
    hk = h(z)

    if (verbose > 0) && (k % ptf == 0)
      @info @sprintf "%6s %8s %8s %7s %8s %7s %7s" k fk hk sqrt(ξ) ν norm(xk) norm(z - xk)
    end

    ν = max(ν/γ, 1e-6) # put floor of 1e-6, more exit crit
    tired = maxIter > 0 && k ≥ maxIter || elapsed_time > maxTime
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
  if norm(z)==0.0
    return x0, k, outdict
  else
    return z, k, outdict
  end
end

function ReSp2(nlp::AbstractNLPModel, args...; kwargs...)
  kwargs_dict = Dict(kwargs...)
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  R2(x -> obj(nlp, x), (g, x) -> grad!(nlp, x, g), args..., x0; kwargs_dict...)
end

function  ReSp2(
  f::F,
  ∇f!::G,
  h::ProximableFunction,
  χ::ProximableFunction,
  options::ROSolverOptions,
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