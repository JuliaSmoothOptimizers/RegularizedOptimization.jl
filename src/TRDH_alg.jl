export TRDH

"""
    TRDH(nlp, h, χ, options; kwargs...)
    TRDH(f, ∇f!, h, options, x0)

A trust-region method with diagonal Hessian approximation for the problem

    min f(x) + h(x)

where f: ℝⁿ → ℝ has a Lipschitz-continuous Jacobian, and h: ℝⁿ → ℝ is
lower semi-continuous and proper.

About each iterate xₖ, a step sₖ is computed as an approximate solution of

    min  φ(s; xₖ) + ψ(s; xₖ)  subject to  ‖s‖ ≤ Δₖ

where φ(s ; xₖ) = f(xₖ) + ∇f(xₖ)ᵀs + ½ sᵀ Dₖ s  is a quadratic approximation of f about xₖ,
ψ(s; xₖ) = h(xₖ + s), ‖⋅‖ is a user-defined norm, Dₖ is a diagonal Hessian approximation
and Δₖ > 0 is the trust-region radius.

### Arguments

* `nlp::AbstractNLPModel`: a smooth optimization problem
* `h`: a regularizer such as those defined in ProximalOperators
* `χ`: a norm used to define the trust region in the form of a regularizer
* `options::ROSolverOptions`: a structure containing algorithmic parameters

The objective and gradient of `nlp` will be accessed.

In the second form, instead of `nlp`, the user may pass in

* `f` a function such that `f(x)` returns the value of f at x
* `∇f!` a function to evaluate the gradient in place, i.e., such that `∇f!(g, x)` store ∇f(x) in `g`
* `x0::AbstractVector`: an initial guess.

### Keyword arguments

* `x0::AbstractVector`: an initial guess (default: `nlp.meta.x0`)
* `selected::AbstractVector{<:Integer}`: (default `1:f.meta.nvar`)
* `Bk`: initial diagonal Hessian approximation (default: `(one(R) / options.ν) * I`).

### Return values

* `xk`: the final iterate
* `Fobj_hist`: an array with the history of values of the smooth objective
* `Hobj_hist`: an array with the history of values of the nonsmooth objective
* `Complex_hist`: an array with the history of number of inner iterations.
"""
function TRDH(
  nlp::AbstractNLPModel{R},
  h,
  χ,
  options::ROSolverOptions{R};
  kwargs...,
) where {R <: Real}
  kwargs_dict = Dict(kwargs...)
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  xk, k, outdict = TRDH(
    x -> obj(nlp, x),
    (g, x) -> grad!(nlp, x, g),
    h,
    options,
    x0;
    χ = χ,
    l_bound = nlp.meta.lvar,
    u_bound = nlp.meta.uvar,
    kwargs...,
  )
  ξ = outdict[:ξ]
  stats = GenericExecutionStats(nlp)
  set_status!(stats, outdict[:status])
  set_solution!(stats, xk)
  set_objective!(stats, outdict[:fk] + outdict[:hk])
  set_residuals!(stats, zero(eltype(xk)), ξ ≥ 0 ? sqrt(ξ) : ξ)
  set_iter!(stats, k)
  set_time!(stats, outdict[:elapsed_time])
  set_solver_specific!(stats, :Fhist, outdict[:Fhist])
  set_solver_specific!(stats, :Hhist, outdict[:Hhist])
  set_solver_specific!(stats, :NonSmooth, outdict[:NonSmooth])
  set_solver_specific!(stats, :SubsolverCounter, outdict[:Chist])
  return stats
end

# update l_bound_k and u_bound_k
function update_bounds!(l_bound_k, u_bound_k, is_subsolver, l_bound, u_bound, xk, Δ)
  if is_subsolver
    @. l_bound_k = max(xk - Δ, l_bound)
    @. u_bound_k = min(xk + Δ, u_bound)
  else
    @. l_bound_k = max(-Δ, l_bound - xk)
    @. u_bound_k = min(Δ, u_bound - xk)
  end
end

function TRDH(
  f::F,
  ∇f!::G,
  h::H,
  options::ROSolverOptions{R},
  x0::AbstractVector{R};
  χ::X = NormLinf(one(R)),
  selected::AbstractVector{<:Integer} = 1:length(x0),
  Bk = (one(R) / options.ν) * I,
  kwargs...,
) where {R <: Real, F, G, H, X}
  start_time = time()
  elapsed_time = 0.0
  ϵ = options.ϵa
  ϵr = options.ϵr
  Δk = options.Δk
  neg_tol = options.neg_tol
  verbose = options.verbose
  maxIter = options.maxIter
  maxTime = options.maxTime
  η1 = options.η1
  η2 = options.η2
  γ = options.γ
  α = options.α
  β = options.β
  spectral = options.spectral
  psb = options.psb
  hess_init_val = (Bk isa UniformScaling) ? Bk.λ : (one(R) / options.ν)
  reduce_TR = options.reduce_TR

  local l_bound, u_bound
  has_bnds = false
  kw_keys = keys(kwargs)
  if :l_bound in kw_keys
    l_bound = kwargs[:l_bound]
    has_bnds = has_bnds || any(l_bound .!= R(-Inf))
  end
  if :u_bound in kw_keys
    u_bound = kwargs[:u_bound]
    has_bnds = has_bnds || any(u_bound .!= R(Inf))
  end

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
  hk = h(xk[selected])
  if hk == Inf
    verbose > 0 && @info "TRDH: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, x0, one(eltype(x0)))
    hk = h(xk[selected])
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "TRDH: found point where h has value" hk
  end
  hk == -Inf && error("nonsmooth term is not proper")

  xkn = similar(xk)
  s = zero(xk)
  l_bound_k = similar(xk)
  u_bound_k = similar(xk)
  if h isa ShiftedProximableFunction # case TRDH is used as a subsolver
    is_subsolver = true
    ψ = shifted(h, xk)
    @assert !has_bnds
    l_bound = copy(ψ.l)
    u_bound = copy(ψ.u)
    @. l_bound_k = max(xk - Δk, l_bound)
    @. u_bound_k = min(xk + Δk, u_bound)
    has_bnds = true
    set_bounds!(ψ, l_bound_k, u_bound_k)
  else
    is_subsolver = false
    if has_bnds
      @. l_bound_k = max(-Δk, l_bound - xk)
      @. u_bound_k = min(Δk, u_bound - xk)
      ψ = shifted(h, xk, l_bound_k, u_bound_k, selected)
    else
      ψ = shifted(h, xk, Δk, χ)
    end
  end

  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, maxIter)
  if verbose > 0
    #! format: off
    if reduce_TR
      @info @sprintf "%6s %8s %8s %7s %7s %8s %7s %7s %7s %7s %1s" "outer" "f(x)" "h(x)" "√ξ1" "√ξ" "ρ" "Δ" "‖x‖" "‖s‖" "‖Bₖ‖" "TRDH"
    else
      @info @sprintf "%6s %8s %8s %7s %8s %7s %7s %7s %7s %1s" "outer" "f(x)" "h(x)" "√ξ" "ρ" "Δ" "‖x‖" "‖s‖" "‖Bₖ‖" "TRDH"
    end
    #! format: off
  end

  local ξ1
  local ξ
  k = 0

  fk = f(xk)
  ∇fk = similar(xk)
  ∇f!(∇fk, xk)
  ∇fk⁻ = copy(∇fk)
  Dk = spectral ? SpectralGradient(hess_init_val, length(xk)) :
    ((Bk isa UniformScaling) ? DiagonalQN(fill!(similar(xk), hess_init_val), psb) : DiagonalQN(diag(Bk), psb))
  DkNorm = norm(Dk.d, Inf)
  νInv = (DkNorm + one(R) / (α * Δk))
  ν = one(R) / νInv
  mν∇fk = -ν .* ∇fk

  optimal = false
  tired = maxIter > 0 && k ≥ maxIter || elapsed_time > maxTime

  while !(optimal || tired)
    k = k + 1
    elapsed_time = time() - start_time
    Fobj_hist[k] = fk
    Hobj_hist[k] = hk

    # model for prox-gradient step to update Δk if ||s|| is too small and ξ1
    φ1(d) = ∇fk' * d
    mk1(d) = φ1(d) + ψ(d)

    if reduce_TR
      prox!(s, ψ, mν∇fk, ν)
      Complex_hist[k] += 1
      ξ1 = hk - mk1(s) + max(1, abs(hk)) * 10 * eps()

      if ξ1 ≥ 0 && k == 1
        ϵ += ϵr * sqrt(ξ1)  # make stopping test absolute and relative
      end

      if (ξ1 < 0 && sqrt(-ξ1) ≤ neg_tol) || (ξ1 ≥ 0 && sqrt(ξ1) < ϵ)
        # the current xk is approximately first-order stationary
        optimal = true
        continue
      end

      ξ1 > 0 || error("TR: first prox-gradient step should produce a decrease but ξ1 = $(ξ1)")
    end

    Δ_effective = reduce_TR ? min(β * χ(s), Δk) : Δk
    # update radius
    if has_bnds
      update_bounds!(l_bound_k, u_bound_k, is_subsolver, l_bound, u_bound, xk, Δ_effective)
      set_bounds!(ψ, l_bound_k, u_bound_k)
    else
      set_radius!(ψ, Δ_effective)
    end

    # model with diagonal hessian 
    φ(d) = ∇fk' * d + (d' * (Dk.d .* d)) / 2
    mk(d) = φ(d) + ψ(d)

    iprox!(s, ψ, ∇fk, Dk)
    Complex_hist[k] += 1

    sNorm = χ(s)
    xkn .= xk .+ s
    fkn = f(xkn)
    hkn = h(xkn[selected])
    hkn == -Inf && error("nonsmooth term is not proper")

    Δobj = fk + hk - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ξ = hk - mk(s) + max(1, abs(hk)) * 10 * eps()

    if !reduce_TR
      if ξ ≥ 0 && k == 1
        ϵ += ϵr * sqrt(ξ)  # make stopping test absolute and relative
      end

      if (ξ < 0 && sqrt(-ξ) ≤ neg_tol) || (ξ ≥ 0 && sqrt(ξ) < ϵ)
        # the current xk is approximately first-order stationary
        optimal = true
        continue
      end
    end

    if (ξ ≤ 0 || isnan(ξ))
      error("TRDH: failed to compute a step: ξ = $ξ")
    end

    ρk = Δobj / ξ

    TR_stat = (η2 ≤ ρk < Inf) ? "↗" : (ρk < η1 ? "↘" : "=")

    if (verbose > 0) && (k % ptf == 0)
      #! format: off
      if reduce_TR
        @info @sprintf "%6d %8.1e %8.1e %7.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s" k fk hk sqrt(ξ1) sqrt(ξ) ρk Δk χ(xk) sNorm norm(Dk.d) TR_stat
      else
        @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s" k fk hk sqrt(ξ) ρk Δk χ(xk) sNorm norm(Dk.d) TR_stat
      end
      #! format: on
    end

    if η2 ≤ ρk < Inf
      Δk = max(Δk, γ * sNorm)
      !has_bnds && set_radius!(ψ, Δk)
    end

    if η1 ≤ ρk < Inf
      xk .= xkn
      has_bnds && update_bounds!(l_bound_k, u_bound_k, is_subsolver, l_bound, u_bound, xk, Δk)
      has_bnds && set_bounds!(ψ, l_bound_k, u_bound_k)
      #update functions
      fk = fkn
      hk = hkn
      shift!(ψ, xk)
      ∇f!(∇fk, xk)
      push!(Dk, s, ∇fk - ∇fk⁻) # update QN operator
      DkNorm = norm(Dk.d, Inf)
      ∇fk⁻ .= ∇fk
    end

    if ρk < η1 || ρk == Inf
      Δk = Δk / 2
      has_bnds && update_bounds!(l_bound_k, u_bound_k, is_subsolver, l_bound, u_bound, xk, Δk)
      has_bnds ? set_bounds!(ψ, l_bound_k, u_bound_k) : set_radius!(ψ, Δk)
    end

    νInv = (DkNorm + one(R) / (α * Δk))
    ν = one(R) / νInv
    mν∇fk .= -ν .* ∇fk

    tired = k ≥ maxIter || elapsed_time > maxTime
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8s %8.1e %8.1e" k "" fk hk
    elseif optimal
      #! format: off
      if reduce_TR
        @info @sprintf "%6d %8.1e %8.1e %7.1e %7.1e %8s %7.1e %7.1e %7.1e %7.1e" k fk hk sqrt(ξ1) sqrt(ξ1) "" Δk χ(xk) χ(s) norm(Dk.d)
        #! format: on
        @info "TRDH: terminating with √ξ1 = $(sqrt(ξ1))"
      else
        @info @sprintf "%6d %8.1e %8.1e %7.1e %8s %7.1e %7.1e %7.1e %7.1e" k fk hk sqrt(ξ) "" Δk χ(
          xk,
        ) χ(s) norm(Dk.d)
        @info "TRDH: terminating with √ξ = $(sqrt(ξ))"
      end
    end
  end

  !reduce_TR && (ξ1 = ξ) # for output dict

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
    :ξ => ξ1,
    :elapsed_time => elapsed_time,
  )

  return xk, k, outdict
end
