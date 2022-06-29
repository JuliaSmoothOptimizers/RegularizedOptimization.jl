export TR_constr

function TR_constr(
  f::AbstractNLPModel,
  h::ProximableFunction,
  χ::ProximableFunction,
  options::ROSolverOptions;
  x0::AbstractVector = f.meta.x0,
  subsolver_logger::Logging.AbstractLogger = Logging.NullLogger(),
  subsolver = R2,
  subsolver_options = ROSolverOptions(),
)
  start_time = time()
  elapsed_time = 0.0
  # initialize passed options
  ϵ = options.ϵ
  Δk = options.Δk
  verbose = options.verbose
  maxIter = options.maxIter
  maxTime = options.maxTime
  η1 = options.η1
  η2 = options.η2
  γ = options.γ
  α = options.α
  θ = options.θ
  β = options.β
  l_bound = f.meta.lvar
  u_bound = f.meta.uvar

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
    verbose > 0 && @info "TR: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, x0, one(eltype(x0)))
    hk = h(xk)
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "TR: found point where h has value" hk
  end
  hk == -Inf && error("nonsmooth term is not proper")

  xkn = similar(xk)
  s = zero(xk)
  ψ = shifted(h, xk, max.(-Δk,l_bound-xk), min.(Δk, u_bound-xk))

  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, maxIter)
  if verbose > 0
    @info @sprintf "%6s %8s %8s %8s %7s %7s %8s %7s %7s %7s %7s %1s" "outer" "inner" "F(x)" "h(x)" "√ξ1" "√ξ" "ρ" "Δk" "‖x‖" "‖s‖" "‖Bₖ‖" "TR"
  end

  local ξ1
  k = 0

  fk = obj(f, xk)
  ∇fk = grad(f, xk)
  ∇fk⁻ = copy(∇fk)

  quasiNewtTest = isa(f, QuasiNewtonModel)
  Bk = hess_op(f, xk)

  λmax, found_λ = opnorm(Bk)
  found_λ || error("operator norm computation failed")
  νInv = (1 + θ) * λmax

  optimal = false
  tired = k ≥ maxIter || elapsed_time > maxTime

  while !(optimal || tired)
    k = k + 1
    elapsed_time = time() - start_time
    Fobj_hist[k] = fk
    Hobj_hist[k] = hk

    # model for first prox-gradient step and ξ1
    φ1(d) = ∇fk' * d
    mk1(d) = φ1(d) + ψ(d)

    # model for subsequent prox-gradient steps and ξ
    φ(d) = (d' * (Bk * d)) / 2 + ∇fk' * d

    ∇φ!(g, d) = begin
      mul!(g, Bk, d)
      g .+= ∇fk
      g
    end

    mk(d) = φ(d) + ψ(d)

    # Take first proximal gradient step s1 and see if current xk is nearly stationary.
    # s1 minimizes φ1(s) + ‖s‖² / 2 / ν + ψ(s) ⟺ s1 ∈ prox{νψ}(-ν∇φ1(0)).
    subsolver_options.ν = 1 / (νInv + 1 / (Δk * α))
    prox!(s, ψ, -subsolver_options.ν * ∇fk, subsolver_options.ν)
    ξ1 = hk - mk1(s) + max(1, abs(hk)) * 10 * eps()
    ξ1 > 0 || error("TR: first prox-gradient step should produce a decrease but ξ1 = $(ξ1)")

    if sqrt(ξ1) < ϵ
      # the current xk is approximately first-order stationary
      optimal = true
      continue
    end

    subsolver_options.ϵ = k == 1 ? 1.0e-5 : max(ϵ, min(1e-2, sqrt(ξ1)) * ξ1)
    set_bounds!(ψ, max.(-min(β * χ(s), Δk), l_bound-xk), min.(min(β * χ(s), Δk), u_bound-xk))
    s, iter, _ = with_logger(subsolver_logger) do
      subsolver(φ, ∇φ!, ψ, subsolver_options, s)
    end
    Complex_hist[k] = iter

    sNorm = χ(s)
    xkn .= xk .+ s
    fkn = obj(f, xkn)
    hkn = h(xkn)
    hkn == -Inf && error("nonsmooth term is not proper")

    Δobj = fk + hk - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ξ = hk - mk(s) + max(1, abs(hk)) * 10 * eps()

    if (ξ ≤ 0 || isnan(ξ))
      error("TR: failed to compute a step: ξ = $ξ")
    end

    ρk = Δobj / ξ

    TR_stat = (η2 ≤ ρk < Inf) ? "↗" : (ρk < η1 ? "↘" : "=")

    if (verbose > 0) && (k % ptf == 0)
      @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s" k iter fk hk sqrt(ξ1) sqrt(ξ) ρk Δk χ(xk) sNorm νInv TR_stat
    end

    if η2 ≤ ρk < Inf
      Δk = max(Δk, γ * sNorm)
      set_bounds!(ψ, max.(-Δk, l_bound-xk), min.(Δk, u_bound-xk))
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
      λmax, found_λ = opnorm(Bk)
      found_λ || error("operator norm computation failed")
      νInv = (1 + θ) * λmax
      ∇fk⁻ .= ∇fk
    end

    if ρk < η1 || ρk == Inf
      Δk = Δk / 2
      set_bounds!(ψ, max.(-Δk, l_bound-xk), min.(Δk, u_bound-xk))
    end
    tired = k ≥ maxIter || elapsed_time > maxTime
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8s %8.1e %8.1e" k "" fk hk
    elseif optimal
      @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8s %7.1e %7.1e %7.1e %7.1e" k 1 fk hk sqrt(ξ1) sqrt(ξ1) "" Δk χ(xk) χ(s) νInv
      @info "TR: terminating with √ξ1 = $(sqrt(ξ1))"
    end
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

  return GenericExecutionStats(
    status,
    f,
    solution = xk,
    objective = fk + hk,
    dual_feas = sqrt(ξ1),
    iter = k,
    elapsed_time = elapsed_time,
    solver_specific = Dict(
      :Fhist => Fobj_hist[1:k],
      :Hhist => Hobj_hist[1:k],
      :NonSmooth => h,
      :SubsolverCounter => Complex_hist[1:k],
    ),
  )
end