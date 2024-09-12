export R2DH

"""
    R2DH(nlp, h, options)
    R2DH(f, ∇f!, h, options, x0)

A first-order quadratic regularization method for the problem

    min f(x) + h(x)

where f: ℝⁿ → ℝ has a Lipschitz-continuous gradient, and h: ℝⁿ → ℝ is
lower semi-continuous, proper and prox-bounded.

About each iterate xₖ, a step sₖ is computed as a solution of

    min  φ(s; xₖ) + ψ(s; xₖ)

where φ(s ; xₖ) = f(xₖ) + ∇f(xₖ)ᵀs + ½ sᵀ (σₖ+Dₖ) s (if `summation = true`) and φ(s ; xₖ) = f(xₖ) + ∇f(xₖ)ᵀs + ½ sᵀ (σₖ+Dₖ) s (if `summation = false`) is a quadratic approximation of f about xₖ,
ψ(s; xₖ) = h(xₖ + s), ‖⋅‖ is a user-defined norm, Dₖ is a diagonal Hessian approximation
and σₖ > 0 is the regularization parameter.

### Arguments

* `nlp::AbstractDiagonalQNModel`: a smooth optimization problem
* `h`: a regularizer such as those defined in ProximalOperators
* `options::ROSolverOptions`: a structure containing algorithmic parameters
* `x0::AbstractVector`: an initial guess (in the second calling form)

### Keyword Arguments

* `x0::AbstractVector`: an initial guess (in the first calling form: default = `nlp.meta.x0`)
* `selected::AbstractVector{<:Integer}`: (default `1:length(x0)`).
* `Bk`: initial diagonal Hessian approximation (default: `(one(R) / options.ν) * I`).
* `summation`: boolean used to choose between the two versions of R2DH (see above, default : `true`).

The objective and gradient of `nlp` will be accessed.

In the second form, instead of `nlp`, the user may pass in

* `f` a function such that `f(x)` returns the value of f at x
* `∇f!` a function to evaluate the gradient in place, i.e., such that `∇f!(g, x)` store ∇f(x) in `g`

### Return values

* `xk`: the final iterate
* `Fobj_hist`: an array with the history of values of the smooth objective
* `Hobj_hist`: an array with the history of values of the nonsmooth objective
* `Complex_hist`: an array with the history of number of inner iterations.
"""
function R2DH(
    nlp::AbstractDiagonalQNModel{R, S},
    h,
    options::ROSolverOptions{R};
    kwargs...,
  ) where {R <: Real, S}
    kwargs_dict = Dict(kwargs...)
    x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
    xk, k, outdict = R2DH(
      x -> obj(nlp, x),
      (g, x) -> grad!(nlp, x, g),
      h,
      hess_op(nlp, x0),
      options,
      x0;
      kwargs...,
    )
    ξ = outdict[:ξ]
    stats = GenericExecutionStats(nlp)
    set_status!(stats, outdict[:status])
    set_solution!(stats, xk)
    set_objective!(stats, outdict[:fk] + outdict[:hk])
    set_residuals!(stats, zero(eltype(xk)), ξ)
    set_iter!(stats, k)
    set_time!(stats, outdict[:elapsed_time])
    set_solver_specific!(stats, :Fhist, outdict[:Fhist])
    set_solver_specific!(stats, :Hhist, outdict[:Hhist])
    set_solver_specific!(stats, :NonSmooth, outdict[:NonSmooth])
    set_solver_specific!(stats, :SubsolverCounter, outdict[:Chist])
    return stats
  end

function R2DH(
  f::F,
  ∇f!::G,
  h::H,
  D::DQN,
  options::ROSolverOptions{R},
  x0::AbstractVector{R};
  Mmonotone::Int = 5,
  selected::AbstractVector{<:Integer} = 1:length(x0),
  summation::Bool = true,
  kwargs...,
) where {F <: Function, G <: Function, H, R <: Real, DQN <: AbstractDiagonalQuasiNewtonOperator}
  start_time = time()
  elapsed_time = 0.0
  ϵ = options.ϵa
  ϵr = options.ϵr
  neg_tol = options.neg_tol
  verbose = options.verbose
  maxIter = options.maxIter
  maxTime = options.maxTime
  σmin = options.σmin
  η1 = options.η1
  η2 = options.η2
  ν = options.ν
  γ = options.γ

  local l_bound, u_bound
  has_bnds = false
  for (key, val) in kwargs
    if key == :l_bound
      l_bound = val
      has_bnds = has_bnds || any(l_bound .!= R(-Inf))
    elseif key == :u_bound
      u_bound = val
      has_bnds = has_bnds || any(u_bound .!= R(Inf))
    end
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
    verbose > 0 && @info "R2DH: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, x0, one(eltype(x0)))
    hk = h(xk[selected])
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "R2DH: found point where h has value" hk
  end
  hk == -Inf && error("nonsmooth term is not proper")

  xkn = similar(xk)
  s = zero(xk)
  ψ = has_bnds ? shifted(h, xk, l_bound - xk, u_bound - xk, selected) : shifted(h, xk)

  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  FHobj_hist = fill!(Vector{R}(undef, Mmonotone), R(-Inf))
  Complex_hist = zeros(Int, maxIter)
  if verbose > 0
    #! format: off
    @info @sprintf "%6s %8s %8s %7s %8s %7s %7s %7s %1s" "iter" "f(x)" "h(x)" "√ξ" "ρ" "σ" "‖x‖" "‖s‖" ""
    #! format: off
  end

  local ξ
  k = 0
  σk = summation ?  σmin : max(1 / ν, σmin)

  fk = f(xk)
  ∇fk = similar(xk)
  ∇f!(∇fk, xk)
  ∇fk⁻ = copy(∇fk) 
  spectral_test = isa(D, SpectralGradient)
  D.d .= summation ? D.d .+ σk : D.d  .* σk
  DNorm = norm(D.d, Inf)


  ν = 1 / DNorm
  mν∇fk = -ν * ∇fk
  sqrt_ξ_νInv = one(R)  

  optimal = false
  tired = maxIter > 0 && k ≥ maxIter || elapsed_time > maxTime

  while !(optimal || tired)
    k = k + 1
    elapsed_time = time() - start_time
    Fobj_hist[k] = fk
    Hobj_hist[k] = hk
    Mmonotone > 0 && (FHobj_hist[mod(k-1, Mmonotone) + 1] = fk + hk)

    D.d .= max.(D.d, eps(R))


    # model with diagonal hessian
    φ(d) = ∇fk' * d + (d' * (D.d .* d)) / 2
    mk(d) = φ(d) + ψ(d)

    if spectral_test
      prox!(s, ψ, mν∇fk, ν)
    else
      iprox!(s, ψ, ∇fk, D)
    end

  #  iprox!(s, ψ, ∇fk, D)

    Complex_hist[k] += 1
    xkn .= xk .+ s
    fkn = f(xkn)
    hkn = h(xkn[selected])
    hkn == -Inf && error("nonsmooth term is not proper")
    
    fhmax = Mmonotone > 0 ? maximum(FHobj_hist) : fk + hk
    Δobj = fhmax - (fkn + hkn) + max(1, abs(fhmax)) * 10 * eps()
    Δmod = fhmax - (fk + mk(s)) + max(1, abs(hk)) * 10 * eps()
    ξ = hk - mk(s) + max(1, abs(hk)) * 10 * eps()
    sqrt_ξ_νInv = ξ ≥ 0 ? sqrt(ξ / ν) : sqrt(-ξ / ν)

    if ξ ≥ 0 && k == 1
      ϵ += ϵr * sqrt_ξ_νInv  # make stopping test absolute and relative
    end
    
    if (ξ < 0 && sqrt_ξ_νInv ≤ neg_tol) || (ξ ≥ 0 && sqrt_ξ_νInv < ϵ)
        # the current xk is approximately first-order stationary
      optimal = true
      continue
    end

    if (ξ ≤ 0 || isnan(ξ))
        error("R2DH: failed to compute a step: ξ = $ξ")
    end

    ρk = Δobj / Δmod

    σ_stat = (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")

    if (verbose > 0) && (k % ptf == 0)
      #! format: off
      @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e %7.1e %7.1e %1s" k fk hk sqrt_ξ_νInv ρk σk norm(xk) norm(s) σ_stat
      #! format: on
    end

    if η2 ≤ ρk < Inf
      σk = max(σk / γ, σmin)
    end

    if η1 ≤ ρk < Inf
      xk .= xkn
      has_bnds && set_bounds!(ψ, l_bound - xk, u_bound - xk)
      fk = fkn
      hk = hkn
      shift!(ψ, xk)
      ∇f!(∇fk, xk)
      push!(D, s, ∇fk - ∇fk⁻) # update QN operator
      DNorm = norm(D.d, Inf) 
      ∇fk⁻ .= ∇fk
    end

    if ρk < η1 || ρk == Inf
      σk = σk * γ
    end

    D.d .= summation ? D.d .+ σk : D.d  .* σk 
    DNorm = norm(D.d, Inf)
    ν = 1 / DNorm
    
    tired = maxIter > 0 && k ≥ maxIter
    if !tired
      @. mν∇fk = -ν * ∇fk
    end
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8.1e %8.1e" k fk hk
    elseif optimal
      #! format: off
      @info @sprintf "%6d %8.1e %8.1e %7.1e %8s %7.1e %7.1e %7.1e" k fk hk sqrt_ξ_νInv "" σk norm(xk) norm(s)
      #! format: on
      @info "R2DH: terminating with √(ξ/ν) = $(sqrt_ξ_νInv))"
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
