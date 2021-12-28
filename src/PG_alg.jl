export PG, PGLnsch, PGΔ, PGE

"""
Proximal Gradient Descent  for

  min_x ϕ(x) = f(x) + g(x), with f(x) β-smooth, g(x) closed, lsc

Input:
  f: function handle that returns f(x) and ∇f(x)
  h: function handle that returns g(x)
  s: initial point
  proxG: function handle that calculates prox_{νg}
  options: see descentopts.jl
Output:
  s⁺: s update
  s : s^(k-1)
  his : function history
  feval : number of function evals (total objective )
"""
function PG(nlp::AbstractNLPModel, args...; kwargs...)
  kwargs_dict = Dict(kwargs...)
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  xk, k, outdict = PG(x -> obj(nlp, x), (g, x) -> grad!(nlp, x, g), args..., x0; kwargs_dict...)

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

function PG(
  f::F,
  ∇f!::G,
  h::ProximableFunction,
  options::ROSolverOptions,
  x0::AbstractVector,
) where {F <: Function, G <: Function}
  start_time = time()
  elapsed_time = 0.0
  ϵ = options.ϵ
  maxIter = options.maxIter
  maxTime = options.maxTime
  ν = options.ν
  verbose = options.verbose

  if options.verbose == 0
    ptf = Inf
  elseif options.verbose == 1
    ptf = round(maxIter / 10)
  elseif options.verbose == 2
    ptf = round(maxIter / 100)
  else
    ptf = 1
  end

  #Problem Initialize
  xk = copy(x0)
  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, maxIter)

  # Iteration set up
  ∇fk = similar(xk)
  ∇f!(∇fk, xk)
  fk = f(xk)
  hk = h(xk)
  ∇fkn = similar(∇fk)
  xkn = similar(x0)
  fstep = xk .- ν .* ∇fk

  #do iterations
  local ξ
  k = 0
  optimal = false
  tired = k ≥ maxIter || elapsed_time ≥ maxTime

  if options.verbose != 0
    @info @sprintf "%6s %8s %8s %7s %8s %7s" "iter" "f(x)" "h(x)" "‖∂ϕ‖" "ν" "‖x‖"
  end

  while !(optimal || tired)
    k = k + 1
    elapsed_time = time() - start_time
    Fobj_hist[k] = fk
    Hobj_hist[k] = hk
    Complex_hist[k] += 1

    ∇fkn .= ∇fk
    xkn .= xk
    fstep .= xk .- ν .* ∇fk
    prox!(xk, h, fstep, ν)

    ∇f!(∇fk, xk)
    fk = f(xk)
    hk = h(xk)

    k += 1
    ∇fkn .= ∇fk .- ∇fkn .- (xk .- xkn) ./ ν
    ξ = norm(∇fkn)
    optimal = ξ < ϵ
    tired =  maxIter > 0 && k ≥ maxIter || elapsed_time > maxTime

    if (verbose > 0) && (k % ptf == 0)
      @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e " k fk hk ξ ν norm(xk)
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

function PGΔ(f, ∇f, h, options; x::AbstractVector = f.meta.x0)
  ε = options.optTol
  maxIter = options.maxIter

  if options.verbose == 0
    ptf = Inf
  elseif options.verbose == 1
    ptf = round(maxIter / 10)
  elseif options.verbose == 2
    ptf = round(maxIter / 100)
  else
    ptf = 1
  end
  #Problem Initialize
  ν = options.ν
  p = options.p
  fDec = options.fDec

  k = 1
  feval = 1
  x⁺ = deepcopy(x)

  # Iteration set up
  g = ∇f(x⁺) #objInner/ quadratic model
  fk = f(x⁺)

  #do iterations
  optimal = false
  FD = false
  tired = k ≥ maxIter

  while !(optimal || tired || FD)
    gold = g
    fold = f
    x = x⁺

    x⁺ = ShiftedProximalOperators.prox(h, x - ν * g, ν)
    # update function info
    g = ∇f(x⁺)
    f = f(x⁺)

    feval += 1
    k += 1
    err = norm(g - gold - (x⁺ - x) / ν)
    optimal = err < ε
    tired = k ≥ maxIter

    k % ptf == 0 && @info @sprintf "%4d ‖xᵏ⁺¹ - xᵏ‖=%1.5e ν = %1.5e" k err ν

    Difff = fold + h(x) - f - h(x⁺) # these do not work
    FD = abs(Difff) < p * norm(fDec)
  end
  return x⁺, feval
end

function PGE(f, h, s, options)
  ε = options.optTol
  maxIter = options.maxIter

  if options.verbose == 0
    ptf = Inf
  elseif options.verbose == 1
    ptf = round(maxIter / 10)
  elseif options.verbose == 2
    ptf = round(maxIter / 100)
  else
    ptf = 1
  end
  #Problem Initialize
  ν = options.ν
  p = options.p
  fDec = options.fDec

  k = 1
  feval = 1
  s⁺ = deepcopy(s)

  # Iteration set up
  g = ∇f(s⁺) #objInner/ quadratic model

  #do iterations
  FD = false
  optimal = false
  tired = k ≥ maxIter

  #do iterations
  while !(optimal || tired || FD)
    gold = g
    s = s⁺

    ν = min(g' * g / (g' * Bk * g), ν) #no BK, will not work
    #prox step
    s⁺ = ShiftedProximalOperators.prox(h, s - ν * g, ν)
    # update function info
    g = ∇f(s⁺)
    f = f(s⁺)

    feval += 1
    k += 1
    err = norm(g - gold - (s⁺ - s) / ν)
    optimal = err < ε
    tired = k ≥ maxIter

    k % ptf == 0 && @info @sprintf "%4d ‖xᵏ⁺¹ - xᵏ‖=%1.5e ν = %1.5e" k err ν

    Difff = fold + h(s) - f - h(s⁺) # these do not work
    FD = abs(Difff) < p * norm(fDec)
  end
  return s⁺, feval
end

function PGLnsch(f, ∇f, h, s, options)
  ε = options.optTol
  maxIter = options.maxIter

  if options.verbose == 0
    ptf = Inf
  elseif options.verbose == 1
    ptf = round(maxIter / 10)
  elseif options.verbose == 2
    ptf = round(maxIter / 100)
  else
    ptf = 1
  end

  #Problem Initialize
  p = options.p
  ν₀ = options.ν
  k = 1
  s⁺ = deepcopy(s)

  # Iteration set up
  feval = 1
  g = ∇f(s⁺) #objInner/ quadratic model
  fk = f(s⁺)

  #do iterations
  optimal = false
  tired = k ≥ maxIter

  while !(optimal || tired)
    gold = g
    s = s⁺

    s⁺ = ShiftedProximalOperators.prox(h, s - ν * g, ν)
    #linesearch but we don't pass in f?
    while f(s⁺) ≥ fk + g' * (s⁺ - s) + 1 / (ν * 2) * norm(s⁺ - s)^2
      ν *= p * ν
      s⁺ = prox(h, s - ν * g, ν)
      feval += 1
    end
    # update function info
    g = ∇f(s⁺) #objInner/ quadratic model
    fk = f(s⁺)

    feval += 1
    k += 1
    err = norm(g - gold - (s⁺ - s) / ν)
    optimal = err < ε
    tired = k ≥ maxIter

    k % ptf == 0 && @info @sprintf "%4d ‖xᵏ⁺¹ - xᵏ‖=%1.5e ν = %1.5e" k err ν
    ν = ν₀
  end
  return s⁺, feval
end
