export RIPM, RIPMDH

function proj!(
  z_l_kj::AbstractVector{R},
  z_l_kj⁻::AbstractVector{R},
  μk::R,
  xkj_m_lvar::AbstractVector{R},
  Kzul::R,
  Kzuu::R,
) where {R <: Real} 
  for i in eachindex(z_l_kj)
    low = Kzul * min(one(R), z_l_kj⁻[i], μk / xkj_m_lvar[i])
    (z_l_kj[i] < low) && (z_l_kj[i] = low)
    upp = max(Kzuu, z_l_kj⁻[i], Kzuu / μk, Kzuu * μk / xkj_m_lvar[i])
    (z_l_kj[i] > upp) && (z_l_kj[i] = upp)
  end
end

function crossover!(x, x_m_lvar, z, ilow, μ, lvar)
  nlow = length(x_m_lvar)
  thresh = μ^(1/4)
  for i=1:nlow
    idx = ilow[i]
    # println("thresh = $(thresh), x_m_lvar[$i] = $(x_m_lvar[i]), z[$i] = $(z[i])")
    # if x_m_lvar[i] ≤ 10 * z[i]
    if x_m_lvar[i] ≤ thresh
      x[idx] = lvar[idx]
    end
    # if z[i] ≤ 10 * x_m_lvar[i]
    if z[i] ≤ thresh
      z[idx] = 0
    end
  end
end


function crossover!(x, z_l, z_u, ilow, iupp, lvar, uvar, μ)
  n = length(x)
  R = eltype(x)
  nlow, nupp = length(ilow), length(iupp)
  thresh = μ^(1/2)
  thresh2 = μ^(1/4)
  c_l = 1
  c_u = 1
  for i=1:n
    idx_l = (c_l ≤ nlow) ? ilow[c_l] : 0
    idx_u = (c_u ≤ nupp) ? iupp[c_u] : 0
    dist_x_l_i = (i == idx_l) ? x[i] - lvar[i] : R(Inf)
    dist_x_u_i = (i == idx_u) ? uvar[i] - x[i] : R(Inf)
    z_l_i = (i == idx_l) ? z_l[c_l] : zero(R)
    z_u_i = (i == idx_u) ? z_u[c_u] : zero(R)

    if dist_x_l_i ≤ thresh && i == idx_l
      x[i] = lvar[i]
    end
    if dist_x_u_i ≤ thresh && i == idx_u
      x[i] = uvar[i]
    end
    # if z[i] ≤ 10 * x_m_lvar[i]
    if z_l_i ≤ thresh && i == idx_l
      z_l[c_l] = 0
    end
    if z_u_i ≤ thresh && i == idx_u
      z_u[c_l] = 0
    end
    if dist_x_l_i ≤ thresh2 && z_l_i ≤ thresh2 && i == idx_l
      x[i] = lvar[i]
      z_l[c_l] = 0
    end
    if dist_x_u_i ≤ thresh2 && z_u_i ≤ thresh2 && i == idx_u
      x[i] = uvar[i]
      z_u[c_u] = 0
    end
    (i == idx_l) && (c_l += 1)
    (i == idx_u) && (c_u += 1)
  end
end

function mul_B_Θ!(res, B, Θ, d)
  mul!(res, B, d)
  res .+= Θ .* d
  return res
end

"""
    RIPM(nlp, h, χ, options; kwargs...)

An interior point method for the regularized problem

    min f(x) + h(x) s.t. x ≥ 0

where f: ℝⁿ → ℝ has a Lipschitz-continuous Jacobian, and h: ℝⁿ → ℝ is
lower semi-continuous and proper.

About each iterate, a step xₖ is computed as an approximate solution of
the barrier subproblem

    min  f(xₖ) + h(xₖ) + φₖ(xₖ)

where φₖ(x) = - μₖ Σᵢ log (xᵢ).

### Arguments

* `nlp::AbstractNLPModel`: a smooth optimization problem
* `h`: a regularizer such as those defined in ProximalOperators
* `χ`: a norm used to define the trust region in the form of a regularizer
* `options::ROSolverOptions`: a structure containing algorithmic parameters

The objective, gradient and Hessian of `nlp` will be accessed.
The Hessian is accessed as an abstract operator and need not be the exact Hessian.

### Keyword arguments

* `x0::AbstractVector`: an initial guess (default: `nlp.meta.x0`)
* `subsolver_logger::AbstractLogger`: a logger to pass to the subproblem solver (default: the null logger)
* `subsolver`: the procedure used to compute a step (`PG` or `R2`)
* `subsolver_options::ROSolverOptions`: default options to pass to the subsolver (default: all defaut options)
* `selected::AbstractVector{<:Integer}`: (default `1:f.meta.nvar`).

### Return values

* `xk`: the final iterate
* `Fobj_hist`: an array with the history of values of the smooth objective
* `Hobj_hist`: an array with the history of values of the nonsmooth objective
* `Complex_hist`: an array with the history of number of inner iterations.
"""
function RIPM(
  f::AbstractNLPModel{R},
  h::H,
  χ::X,
  options::ROSolverOptions;
  x0::AbstractVector = f.meta.x0,
  subsolver_logger::Logging.AbstractLogger = Logging.NullLogger(),
  subsolver = R2,
  subsolver_options = ROSolverOptions(),
  selected::AbstractVector{<:Integer} = 1:(f.meta.nvar),
) where {R, H, X}
  start_time = time()
  elapsed_time = 0.0
  # initialize passed options
  ϵ = options.ϵa
  ϵμ = copy(ϵ)
  ϵ_subsolver = copy(subsolver_options.ϵa)
  ϵr = options.ϵr
  ϵri = options.opt_RIPM.ϵri
  ϵri1 = options.opt_RIPM.ϵri1
  useξz = options.opt_RIPM.useξz
  stop_proj_gd = options.opt_RIPM.stop_proj_gd
  Δk = options.Δk
  Δkj = Δk
  verbose = options.verbose
  maxIter = options.maxIter
  maxIter_inner = options.opt_RIPM.maxIter_inner
  maxIter_outer = options.opt_RIPM.maxIter_outer
  maxTime = options.maxTime
  threshXinvZ = options.opt_RIPM.threshXinvZ
  resetQN = options.opt_RIPM.resetQN
  scaleTR = options.opt_RIPM.scaleTR
  crossover = options.opt_RIPM.crossover
  η1 = options.η1
  η2 = options.η2
  γ = options.γ
  α = options.α
  θ = options.θ
  β = options.β
  stop_ν = options.stop_ν
  subsolver_stop_ν = copy(subsolver_options.stop_ν)
  subsolver_options.stop_ν = stop_ν

  ν_subsolver = subsolver_options.ν
  ϵa_subsolver = subsolver_options.ϵa
  Δk_subsolver = subsolver_options.Δk

  lvar = f.meta.lvar
  uvar = f.meta.uvar

  ilow, iupp = sort!([f.meta.ilow; f.meta.irng]), sort!([f.meta.iupp; f.meta.irng])
  nlow = length(ilow)
  nupp = length(iupp)

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
  xkj = copy(x0)
  n = length(xkj)
  δk = options.opt_RIPM.δ0
  δmin = options.opt_RIPM.δmin
  μk = options.opt_RIPM.μ0
  μmin = options.opt_RIPM.μmin
  # tmp fix to not strictly feasible point
  for i in 1:n
    if !(lvar[i] < xkj[i])
      xkj[i] = lvar[i] + μk
    end
    if !(xkj[i] < uvar[i])
      xkj[i] = uvar[i] - μk
    end
    if !(lvar[i] < xkj[i] < uvar[i])
      xkj[i] = (lvar[i] + uvar[i]) / 2
    end
  end
  hkj = h(xkj[selected])
  if hkj == Inf
    verbose > 0 && @info "TR: finding initial guess where nonsmooth term is finite"
    prox!(xkj, h, xkj, one(eltype(x0)))
    hkj = h(xkj[selected])
    hkj < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "TR: found point where h has value" hk
  end
  hkj == -Inf && error("nonsmooth term is not proper")

  xkjn = copy(xkj)
  s = zero(xkj)
  s2 = similar(s)
  s3 = similar(s)
  xkj_m_lvar = similar(xkj, nlow)
  uvar_m_xkj = similar(xkj, nupp)
  @. xkj_m_lvar = @views xkj[ilow] - lvar[ilow]
  @. uvar_m_xkj = @views uvar[iupp] - xkj[iupp]
  l_bound_kj = lvar - xkj
  u_bound_kj = uvar - xkj
  z_l_kj = μk ./ xkj_m_lvar
  z_u_kj = μk ./ uvar_m_xkj
  B_Θ = similar(xkj)
  ψ = shifted(h, xkj, max.(-Δk, l_bound_kj), min.(Δk, u_bound_kj), selected)

  Fobj_hist = zeros(maxIter + 1)
  Hobj_hist = zeros(maxIter + 1)
  IterSucc = fill(false, maxIter + 1)
  Complex_hist = zeros(Int, maxIter + 1)
  if verbose > 0
    #! format: off
    @info @sprintf "%6s %8s %8s %8s %8s %7s %7s %8s %7s %7s %7s %7s %1s %7s %7s" "outer" "inner" "sub" "f(x)" "h(x)" "√ξ1/√ν" "√ξ" "ρ" "Δ" "‖x‖" "‖s‖" "‖Bₖ‖" "TR" "μk" "‖xs-μk‖"
    #! format: on
  end

  local ξk1
  local ξk2
  local ξ1
  local νInv
  prox_evals = 0
  k = 0
  ϵk = options.opt_RIPM.ϵ0
  ϵ_increment = copy(ϵk) # tmp value
  Kzul = one(R) / 2
  Kzuu = R(1.0e20)

  fkj = obj(f, xkj)
  φkj = - μk * (sum(log.(xkj_m_lvar)) + sum(log.(uvar_m_xkj)))
  ∇fkj = grad(f, xkj)
  IterSucc[1] = true
  ∇ϕ1kj = copy(∇fkj)
  ∇ϕ2kj = copy(∇fkj)
  @. ∇ϕ1kj[ilow] -= μk / xkj_m_lvar
  @. ∇ϕ2kj[ilow] -= z_l_kj
  @. ∇ϕ1kj[iupp] += μk / uvar_m_xkj
  @. ∇ϕ2kj[iupp] += z_u_kj
  ∇fkj⁻ = copy(∇fkj)
  z_l_kj⁻ = copy(z_l_kj)
  z_u_kj⁻ = copy(z_u_kj)
  Θkj = fill!(similar(xkj), zero(R))
  Θkj[ilow] += z_l_kj ./ xkj_m_lvar
  Θkj[iupp] += z_u_kj ./ uvar_m_xkj
  for i in 1:n
    if Θkj[i] ≥ threshXinvZ
      Θkj[i] = threshXinvZ
    end
  end

  quasiNewtTest = isa(f, QuasiNewtonModel)
  Bkj = hess_op(f, xkj)
  λmax = opnorm(Bkj)
  νInv = (1 + θ) * (λmax + maximum(Θkj))
  ν = 1 / (νInv + 1 / (Δk * α))

  optimal = false
  tired = k ≥ maxIter || elapsed_time > maxTime
  Δkj = Δk
  n_eval = 0
  Bkjdiag = diag(Bkj)
  TRscale = one(R) ./ max.(Bkjdiag .+ Θkj, one(R))
  TR_l = similar(xkj)
  TR_u = similar(xkj)

  while !(optimal || tired)
    k = k + 1
    elapsed_time = time() - start_time

    # Δkj = Δk
    resetQN && (λmax = one(R)) # opnorm(Bkj)
    # νInv = (1 + θ) * (λmax + maximum(z_l_kj ./ xkj_m_lvar))

    # subproblem: min f(xₖ) + h(xₖ) + φₖ(xₖ)
    bnd_offset_l = (nlow > 0) ? δk * minimum(xkj_m_lvar) : zero(R)
    bnd_offset_u = (nupp > 0) ? δk * minimum(uvar_m_xkj) : zero(R)
    @assert (nlow == 0 || bnd_offset_l > 0) && (nupp == 0 || bnd_offset_u > 0)
    l_bound_kj[ilow] .+= bnd_offset_l
    u_bound_kj[iupp] .-= bnd_offset_u
    diag!(Bkj, Bkjdiag)
    TRscale .= one(R) ./ max.(Bkjdiag .+ Θkj, μk)
    TR_l .= scaleTR ? max.((-Δkj) .* TRscale, l_bound_kj) : max.(-Δkj, l_bound_kj)
    TR_u .= scaleTR ? min.(Δkj .* TRscale, u_bound_kj) : min.(Δkj, u_bound_kj)
    set_bounds!(ψ, TR_l, TR_u)
    # ∇ϕ1kj = ∇fkj - μk Xkj⁻¹ e
    @. ∇ϕ1kj = ∇fkj
    @. ∇ϕ1kj[ilow] -= μk / xkj_m_lvar
    @. ∇ϕ1kj[iupp] += μk / uvar_m_xkj
    @. ∇ϕ2kj = ∇fkj
    @. ∇ϕ2kj[ilow] -= z_l_kj
    @. ∇ϕ2kj[iupp] += z_u_kj
    j = 0
    optimal_inner = false
    tired_inner = j ≥ maxIter_inner
    ϵk_used = ϵk
    # ϵk_used = max(ϵ, min(μk * 100, R(1.0e-2)))

    while !(optimal_inner || tired_inner)

      n_eval += 1
      Fobj_hist[n_eval] = fkj
      Hobj_hist[n_eval] = hkj
      if n_eval > maxIter
        tired = true
        tired_inner = true
        continue
      end
      j = j + 1
      # subproblem inner:
      # ϕ1(d) = (∇f(xkj) - μk Xkj⁻¹e)ᵀd 
      ϕ1(d) = dot(∇ϕ1kj, d)
      mk1(d) = ϕ1(d) + ψ(d)
      ϕ2(d) = dot(∇ϕ2kj, d)
      mk2(d) = ϕ2(d) + ψ(d)

      # Take first proximal gradient step s1 and see if current xk is nearly stationary.
      # s1 minimizes φ1(s) + ‖s‖² / 2 / ν + ψ(s) ⟺ s1 ∈ prox{νψ}(-ν∇φ1(0)).
      prox!(s, ψ, -ν * ∇ϕ1kj, ν)
      prox_evals += 1
      # ξk1 = hkj - dot(∇fkj, s) + dot(z_l_kj, s) - ψ(s) + max(1, abs(hkj)) * 10 * eps()
      ξk1 = hkj - mk1(s) + max(1, abs(hkj)) * 10 * eps()
      ξk1 > 0 || error("TR: first prox-gradient step should produce a decrease but ξk1 = $(ξk1)")
      if useξz
        prox!(s2, ψ, -ν * ∇ϕ2kj, ν)
        ξk2 = hkj - mk2(s2) + max(1, abs(hkj)) * 10 * eps()
      else
        ξk2 = ξk1
      end
      # println(" j = $j  ,    ξk1 = $(sqrt(ξk1)) ,     ξk2 = $(sqrt(ξk2))")

      compl_l = norm(xkj_m_lvar .* z_l_kj .- μk)
      compl_u = norm(uvar_m_xkj .* z_u_kj .- μk)

      if ξk2 ≥ 0 && j == 1
        # ϵ_increment = ϵri * sqrt(ξk2 / ν)
        ϵ_increment = (k == 1) ? (stop_ν ? ϵri1 * sqrt(ξk2 / ν) : ϵri1 * sqrt(ξk2)) :
          (stop_ν ? ϵri * sqrt(ξk2 / ν) : ϵri * sqrt(ξk2))
        ϵk_used += ϵ_increment  # make stopping test absolute and relative
        if k == 1
          ϵ_increment = stop_ν ? (ϵr * sqrt(ξk2 / ν)) : (ϵr * sqrt(ξk2))
          !stop_proj_gd && (ϵ += ϵ_increment)
        end
        # ϵ_subsolver = ϵ_subsolver_init + ϵ_increment
      end

      if !stop_proj_gd
        if stop_ν
          if μk ≤ ϵμ && sqrt(ξk2 / ν) < ϵ && compl_l < ϵ && compl_u < ϵ 
            optimal = true
            optimal_inner = true
            continue
          end
        else
          if μk ≤ ϵμ && sqrt(ξk2) < ϵ && compl_l < ϵ && compl_u < ϵ 
            # if sqrt(norm(s2) / ν) < ϵ && compl < ϵ && μk ≤ 10 * μmin
            # the current xk is approximately first-order stationary
            optimal = true
            optimal_inner = true
            # crossover ?
            continue
          end
        end
      else
        ############### stop crit TR
        @. l_bound_kj = lvar - xkj
        @. u_bound_kj = uvar - xkj
        ν3 = one(R) / (λmax * (1 + θ) + 1 / (α * Δkj))
        Δ2 = Δkj
        TR_l .= scaleTR ? max.((-Δ2) .* TRscale, l_bound_kj) : max.(-Δ2, l_bound_kj)
        TR_u .= scaleTR ? min.(Δ2 .* TRscale, u_bound_kj) : min.(Δ2, u_bound_kj)
        set_bounds!(ψ, TR_l, TR_u)
        ϕ3(d) = dot(∇fkj, d)
        mk3(d) = ϕ3(d) + ψ(d)
        prox!(s3, ψ, -ν3 * ∇fkj, ν3)
        ξ3 = hkj - mk3(s3) + max(1, abs(hkj)) * 10 * eps()
        ξ3 > 0 || error("RIPM: first prox-gradient step should produce a decrease but ξ3 = $(ξ3)")
        # println("ξ3 = $ξ3,   √ν⁻¹ξ3 = $(sqrt(ξ3 / ν3))")
        if ξ3 ≥ 0 && k == 1 && j == 1
          ϵ_increment = ϵr * sqrt(ξ3 / ν3)
          ϵ += ϵ_increment  # make stopping test absolute and relative
        end
        if μk ≤ ϵμ && sqrt(ξ3 / ν3) < ϵ && compl_l < ϵ && compl_u < ϵ 
          # if sqrt(norm(s2) / ν) < ϵ && compl < ϵ && μk ≤ 10 * μmin
          # the current xk is approximately first-order stationary
          optimal = true
          optimal_inner = true
          # crossover ?
          continue
        end
        ###############
      end

      # if ξk1 * νInv < ϵk
      if j > 1
        # if sqrt(ξk2 / ν) < ϵk_used && compl < ϵk_used
        optimal_inner = (stop_ν ? (sqrt(ξk2 / ν) < ϵk_used) : (sqrt(ξk2) < ϵk_used)) && compl_l < ϵk && compl_u < ϵk
        # stop_proj_gd && (optimal_inner = optimal_inner && sqrt(ξ3 / ν3) < ϵk) 
        # if sqrt(norm(s2) / ν) < ϵk_used && compl < ϵk_used
          # the current xk is approximately first-order stationary
        optimal_inner && continue
      end
   
      subsolver_options.ϵa = j == 1 ? 1.0e-5 : max(ϵ_subsolver, min(1e-2, (stop_ν ? sqrt(ξk1 / ν) : sqrt(ξk1))))
      ∆_effective = min(β * χ(s), Δkj)
      subsolver_options.Δk = ∆_effective / 10
      @. l_bound_kj = lvar - xkj
      @. u_bound_kj = uvar - xkj
      bnd_offset_l = @views (nlow > 0) ? δk * minimum((xkj[ilow] .- lvar[ilow])) : zero(R)
      bnd_offset_u = @views (nupp > 0) ? δk * minimum((uvar[ilow] .- xkj[ilow])) : zero(R)
      @assert bnd_offset_l ≥ 0 && bnd_offset_u ≥ 0
      l_bound_kj[ilow] .+= bnd_offset_l
      u_bound_kj[iupp] .-= bnd_offset_u
      TRscale .= one(R) ./ max.(Bkjdiag .+ Θkj, μk)
      TR_l .= scaleTR ? max.((-∆_effective) .* TRscale, l_bound_kj) : max.(-∆_effective, l_bound_kj)
      TR_u .= scaleTR ? min.(∆_effective .* TRscale, u_bound_kj) : min.(∆_effective, u_bound_kj)
      set_bounds!(ψ, TR_l, TR_u)

      # model for subsequent prox-gradient steps and ξ
      # ϕ(d) = f(xkj) + φₖ(xkj) + (∇f(xkj) - μk Xkj⁻¹e)ᵀd + dᵀ (Bkj + z_l_kj Xkj⁻¹) d / 2
      # mul_B_XinvZ!(B_XinvZ, Bkj, X_m_lvar⁻¹Ze, ilow, d)
      ϕ(d) = @views dot(∇ϕ1kj, d) + dot(d, mul_B_Θ!(B_Θ, Bkj, Θkj, d)) / 2
      # ∇ϕ(d) = ∇f(xkj) - μk Xkj⁻¹ + (Bkj + z_l_kj Xkj⁻¹) d
      ∇ϕ!(g, d) = begin
        mul!(g, Bkj, d)
        g .+= ∇ϕ1kj .+ d .* Θkj
        g
      end

      mkj(d) = ϕ(d) + ψ(d)

      subsolver_options.ν = ν
      s, iter, _ = with_logger(subsolver_logger) do
        subsolver(ϕ, ∇ϕ!, ψ, subsolver_options, s; Bk = Bkj)
      end
      # restore initial values subsolver
      subsolver_options.ν = ν_subsolver
      subsolver_options.ϵa = ϵa_subsolver
      subsolver_options.Δk = Δk_subsolver

      prox_evals += iter

      sNorm = χ(s)
      xkjn .= xkj .+ s
      # for i=1:length(xkjn)
      #   if xkjn[i] < sqrt(eps()) * 100
      #     xkjn[i] += sqrt(eps()) * 100
      #   end
      # end
      fkjn = obj(f, xkjn)
      φkjn = @views -μk * (sum(log.(xkjn[ilow] .- lvar[ilow])) + sum(log.(uvar[iupp] .- xkjn[iupp])))
      hkjn = h(xkjn[selected])
      hkjn == -Inf && error("nonsmooth term is not proper")
  
      Δobj = fkj + φkj + hkj - (fkjn + φkjn + hkjn) + max(1, abs(fkj + φkj + hkj)) * 10 * eps()
      ξk = hkj - mkj(s) + max(1, abs(hkj)) * 10 * eps()
  
      if (ξk ≤ 0 || isnan(ξk))
        error("TR: failed to compute a step: ξk = $ξk")
      end
  
      ρkj = Δobj / ξk

      TR_stat = (η2 ≤ ρkj < Inf) ? "↗" : (ρkj < η1 ? "↘" : "=")

      if η2 ≤ ρkj < Inf
        Δkj = max(Δkj, γ * sNorm)
      end
  
      if η1 ≤ ρkj < Inf
        @. z_l_kj = @views (μk - z_l_kj * s[ilow]) / xkj_m_lvar
        @. z_u_kj = @views (μk + z_u_kj * s[iupp]) / uvar_m_xkj
        xkj .= xkjn
        @. xkj_m_lvar = @views xkj[ilow] - lvar[ilow]
        @. uvar_m_xkj = @views uvar[iupp] - xkj[iupp]
        @. l_bound_kj = lvar - xkj
        @. u_bound_kj = uvar - xkj
        proj!(z_l_kj, z_l_kj⁻, μk, xkj_m_lvar, Kzul, Kzuu)
        proj!(z_u_kj, z_u_kj⁻, μk, uvar_m_xkj, Kzul, Kzuu)
        z_l_kj⁻ .= z_l_kj
        z_u_kj⁻ .= z_u_kj
  
        #update functions
        fkj = fkjn
        φkj = φkjn
        hkj = hkjn
        shift!(ψ, xkj)
        ∇fkj = grad(f, xkj)
        ∇ϕ1kj .= ∇fkj
        @. ∇ϕ1kj[ilow] -= μk / xkj_m_lvar
        @. ∇ϕ1kj[iupp] += μk / uvar_m_xkj
        ∇ϕ2kj .= ∇fkj
        @. ∇ϕ2kj[ilow] -= z_l_kj
        @. ∇ϕ2kj[iupp] += z_u_kj
        Θkj .= zero(R)
        Θkj[ilow] += z_l_kj ./ xkj_m_lvar
        Θkj[iupp] += z_u_kj ./ uvar_m_xkj
        for i in 1:n
          if Θkj[i] ≥ threshXinvZ
            Θkj[i] = threshXinvZ
          end
        end 
        # grad!(f, xk, ∇fk)
        if quasiNewtTest
          push!(f, s, ∇fkj - ∇fkj⁻)
        end
        Bkj = hess_op(f, xkj)
        λmax = opnorm(Bkj)
        νInv = (1 + θ) * (λmax + maximum(Θkj))
        ∇fkj⁻ .= ∇fkj
        IterSucc[n_eval + 1] = true

        # update bounds
        bnd_offset_l = (nlow > 0) ? δk * minimum(xkj_m_lvar) : zero(R)
        bnd_offset_u = (nupp > 0) ? δk * minimum(uvar_m_xkj) : zero(R)
        @assert bnd_offset_l ≥ 0 && bnd_offset_u ≥ 0
        l_bound_kj[ilow] .+= bnd_offset_l
        u_bound_kj[iupp] .-= bnd_offset_u
        diag!(Bkj, Bkjdiag)
        TRscale .= one(R) ./ max.(Bkjdiag .+ Θkj, μk)
        TR_l .= scaleTR ? max.((-Δkj) .* TRscale, l_bound_kj) : max.(-Δkj, l_bound_kj)
        TR_u .= scaleTR ? min.(Δkj .* TRscale, u_bound_kj) : min.(Δkj, u_bound_kj)
        set_bounds!(ψ, TR_l, TR_u)
      end
  
      if ρkj < η1 || ρkj == Inf
        Δkj = Δkj / 2
        @. l_bound_kj = lvar - xkj
        @. u_bound_kj = uvar - xkj
        bnd_offset_l = (nlow > 0) ? δk * minimum(xkj_m_lvar) : zero(R)
        bnd_offset_u = (nupp > 0) ? δk * minimum(uvar_m_xkj) : zero(R)
        @assert bnd_offset_l ≥ 0 && bnd_offset_u ≥ 0
        l_bound_kj[ilow] .+= bnd_offset_l
        u_bound_kj[iupp] .-= bnd_offset_u
        TR_l .= scaleTR ? max.((-Δkj) .* TRscale, l_bound_kj) : max.(-Δkj, l_bound_kj)
        TR_u .= scaleTR ? min.(Δkj .* TRscale, u_bound_kj) : min.(Δkj, u_bound_kj)
        set_bounds!(ψ, TR_l, TR_u)
      end

      compl_l = norm(xkj_m_lvar .* z_l_kj .- μk)
      compl_u = norm(uvar_m_xkj .* z_u_kj .- μk)
      compl = sqrt(compl_l^2 + compl_u^2) 

      if (verbose > 0) && (k % ptf == 0)
        #! format: off
        @info @sprintf "%6d %8d %8d %8.1e %8.1e %7.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s %7.1e %7.1e" k j iter fkj hkj sqrt(ξk2 / ν) sqrt(ξk) ρkj ∆_effective χ(xkj) sNorm νInv TR_stat μk compl
        #! format: on
      end
      ν = 1 / (νInv + 1 / (Δkj * α))
      # ν = 1 / (1 + 1 / (Δkj * α))
      tired_inner = j ≥ maxIter_inner || elapsed_time > maxTime
    end
    Complex_hist[k] = j

    # if ξk1 ≥ 0 && k == 1 # && j == 1
    #   # ϵ_increment = ϵr * sqrt(ξk2 / ν)
    #   ϵ_increment = ϵr * sqrt(ξk2)
    #   ϵ += ϵ_increment  # make stopping test absolute and relative
    # end
    compl_l = norm(xkj_m_lvar .* z_l_kj .- μk)
    compl_u = norm(uvar_m_xkj .* z_u_kj .- μk)
    compl = sqrt(compl_l^2 + compl_u^2) 
    # println("ϵ = $ϵ ,  ϵk = $ϵk_used,  √ξk1/√ν = $(sqrt(ξk2 / ν)) , compl = $compl , μk = $μk")
    # if sqrt(ξk2 / ν) < ϵ && compl < ϵ && μk ≤ 10 * μmin
    optimal == true && continue

    (μk ≥ μmin) && (μk /= 10)
    # ϵk > ϵ && (ϵk /= 10)
    # (μk ≥ μmin) && min(μk /= 10, μk^R(1.5))
    ϵk = μk^R(1.01)
    (δk ≥ δmin) && (δk /= 2)

    @. l_bound_kj = lvar - xkj
    @. u_bound_kj = uvar - xkj
    # @. z_l_kj = @views -μk / l_bound_kj[ilow]
    # @. z_l_kj[iupp] = @views μk / u_bound_kj[iupp]
    # @. z_l_kj[irng] = @views μk / (-l_bound_kj[irng] + u_bound_kj[irng])
    # fkj = obj(f, xkj)
    φkj = -μk * (sum(log.(xkj_m_lvar)) + sum(log.(uvar_m_xkj)))
    # ∇fkj = grad(f, xkj)
    ∇fkj⁻ .= ∇fkj
    z_l_kj⁻ .= z_l_kj
    z_u_kj⁻ .= z_u_kj
    # hkj = h(xkj[selected])
    shift!(ψ, xkj)
    if quasiNewtTest && resetQN
      reset_data!(f)
      diag!(Bkj, Bkjdiag)
    end

    νInv = (1 + θ) * (λmax + maximum(Θkj))
    # Δkj = Δk
    Δkj = μk * 1000
    ν = 1 / (νInv + 1 / (Δkj * α))
    tired = tired || k ≥ maxIter_outer || elapsed_time > maxTime
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8s %8.1e %8.1e" k "" fkj hkj
    elseif optimal
      #! format: off
      @info @sprintf "%6d %8d %8d %8.1e %8.1e %7.1e %7.1e %8s %7.1e %7.1e %7.1e %7.1e" k 0 1 fkj hkj sqrt(ξk2 / ν) sqrt(ξk1) "" Δkj χ(xkj) χ(s) νInv
      #! format: on
      @info "RIPM: terminating with √ξ1/√ν = $(sqrt(ξk2 / ν))"
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

  @. xkj_m_lvar = @views xkj[ilow] - lvar[ilow]
  @. uvar_m_xkj = @views uvar[iupp] - xkj[iupp]
  # crossover!(xkj, z_l_kj, z_u_kj, ilow, iupp, μk, lvar)
  crossover && crossover!(xkj, z_l_kj, z_u_kj, ilow, iupp, lvar, uvar, μk)

  z_l_kj_out = fill!(similar(xkj), zero(R))
  z_l_kj_out[ilow] .= z_l_kj
  z_u_kj_out = fill!(similar(xkj), zero(R))
  z_u_kj_out[iupp] .= z_u_kj
  stats = GenericExecutionStats(f)
  set_status!(stats, status)
  set_solution!(stats, xkj)
  has_bounds(f) && set_bounds_multipliers!(stats, z_l_kj_out, z_u_kj_out)
  set_objective!(stats, fkj + hkj)
  set_residuals!(stats, zero(eltype(xkj)), ξk2 ≥ 0 ? (stop_ν ? sqrt(ξk2 / ν) : sqrt(ξk2)) : ξk2)
  set_iter!(stats, k)
  set_time!(stats, elapsed_time)
  set_solver_specific!(stats, :Fhist, Fobj_hist[1:n_eval])
  set_solver_specific!(stats, :Hhist, Hobj_hist[1:n_eval])
  set_solver_specific!(stats, :IterSucc, IterSucc[1:n_eval])
  set_solver_specific!(stats, :NonSmooth, h)
  set_solver_specific!(stats, :SubsolverCounter, Complex_hist[1:k + 1])
  set_solver_specific!(stats, :ProxEvals, prox_evals)
  subsolver_options.stop_ν = subsolver_stop_ν
  return stats
end

function RIPMDH(
  f::AbstractNLPModel{R},
  h::H,
  χ::X,
  options::ROSolverOptions;
  x0::AbstractVector = f.meta.x0,
  selected::AbstractVector{<:Integer} = 1:(f.meta.nvar),
) where {R, H, X}
  start_time = time()
  elapsed_time = 0.0
  # initialize passed options
  ϵ = options.ϵa
  ϵμ = copy(ϵ)
  ϵr = options.ϵr
  ϵri = options.opt_RIPM.ϵri
  ϵri1 = options.opt_RIPM.ϵri1
  useξz = options.opt_RIPM.useξz
  stop_proj_gd = options.opt_RIPM.stop_proj_gd
  Δk = options.Δk
  Δkj = Δk
  verbose = options.verbose
  maxIter = options.maxIter
  maxIter_inner = options.opt_RIPM.maxIter_inner
  maxIter_outer = options.opt_RIPM.maxIter_outer
  maxTime = options.maxTime
  threshXinvZ = options.opt_RIPM.threshXinvZ
  resetQN = options.opt_RIPM.resetQN
  crossover = options.opt_RIPM.crossover
  η1 = options.η1
  η2 = options.η2
  γ = options.γ
  α = options.α
  θ = options.θ
  β = options.β
  spectral = options.spectral
  psb = options.psb
  hess_init_val = one(R) / options.ν
  stop_ν = options.stop_ν

  lvar = f.meta.lvar
  uvar = f.meta.uvar

  ilow, iupp = sort!([f.meta.ilow; f.meta.irng]), sort!([f.meta.iupp; f.meta.irng])
  nlow = length(ilow)
  nupp = length(iupp)

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
  xkj = copy(x0)
  n = length(xkj)
  δk = options.opt_RIPM.δ0
  δmin = options.opt_RIPM.δmin
  μk = options.opt_RIPM.μ0
  μmin = options.opt_RIPM.μmin
  # tmp fix to not strictly feasible point
  for i in 1:n
    if !(lvar[i] < xkj[i])
      xkj[i] = lvar[i] + μk
    end
    if !(xkj[i] < uvar[i])
      xkj[i] = uvar[i] - μk
    end
    if !(lvar[i] < xkj[i] < uvar[i])
      xkj[i] = (lvar[i] + uvar[i]) / 2
    end
  end
  hkj = h(xkj[selected])
  if hkj == Inf
    verbose > 0 && @info "TR: finding initial guess where nonsmooth term is finite"
    prox!(xkj, h, xkj, one(eltype(x0)))
    hkj = h(xkj[selected])
    hkj < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "TR: found point where h has value" hk
  end
  hkj == -Inf && error("nonsmooth term is not proper")

  xkjn = copy(xkj)
  s = zero(xkj)
  s2 = similar(s)
  s3 = similar(s)
  xkj_m_lvar = similar(xkj, nlow)
  uvar_m_xkj = similar(xkj, nupp)
  @. xkj_m_lvar = @views xkj[ilow] - lvar[ilow]
  @. uvar_m_xkj = @views uvar[iupp] - xkj[iupp]
  l_bound_kj = lvar - xkj
  u_bound_kj = uvar - xkj
  z_l_kj = μk ./ xkj_m_lvar
  z_u_kj = μk ./ uvar_m_xkj
  B_Θ = similar(xkj)
  ψ = shifted(h, xkj, max.(-Δk, l_bound_kj), min.(Δk, u_bound_kj), selected)

  Fobj_hist = zeros(maxIter + 1)
  Hobj_hist = zeros(maxIter + 1)
  IterSucc = fill(false, maxIter + 1)
  Complex_hist = zeros(Int, maxIter + 1)
  if verbose > 0
    #! format: off
    @info @sprintf "%6s %8s %8s %8s %8s %7s %7s %8s %7s %7s %7s %7s %1s %7s %7s" "outer" "inner" "sub" "f(x)" "h(x)" "√ξ1/√ν" "√ξ" "ρ" "Δ" "‖x‖" "‖s‖" "‖Bₖ‖" "TR" "μk" "‖xs-μk‖"
    #! format: on
  end

  local ξk1
  local ξk2
  local ξ1
  local νInv
  prox_evals = 0
  k = 0
  ϵk = options.opt_RIPM.ϵ0
  ϵ_increment = copy(ϵk) # tmp value
  Kzul = one(R) / 2
  Kzuu = R(1.0e20)

  fkj = obj(f, xkj)
  φkj = - μk * (sum(log.(xkj_m_lvar)) + sum(log.(uvar_m_xkj)))
  ∇fkj = grad(f, xkj)
  IterSucc[1] = true
  ∇ϕ1kj = copy(∇fkj)
  ∇ϕ2kj = copy(∇fkj)
  @. ∇ϕ1kj[ilow] -= μk / xkj_m_lvar
  @. ∇ϕ2kj[ilow] -= z_l_kj
  @. ∇ϕ1kj[iupp] += μk / uvar_m_xkj
  @. ∇ϕ2kj[iupp] += z_u_kj
  ∇fkj⁻ = copy(∇fkj)
  z_l_kj⁻ = copy(z_l_kj)
  z_u_kj⁻ = copy(z_u_kj)
  Θkj = fill!(similar(xkj), zero(R))
  Θkj[ilow] += z_l_kj ./ xkj_m_lvar
  Θkj[iupp] += z_u_kj ./ uvar_m_xkj
  for i in 1:n
    if Θkj[i] ≥ threshXinvZ
      Θkj[i] = threshXinvZ
    end
  end

  Dkj = spectral ? SpectralGradient(hess_init_val, length(xkj)) :
    DiagonalQN(fill!(similar(xkj), hess_init_val), psb)
  λmax = norm(Dkj.d, Inf)
  νInv = (1 + θ) * (λmax + maximum(Θkj))
  ν = 1 / (νInv + 1 / (Δk * α))
  D_p_Θkj = Dkj.d .+ Θkj

  optimal = false
  tired = k ≥ maxIter || elapsed_time > maxTime
  Δkj = Δk
  n_eval = 0

  while !(optimal || tired)
    k = k + 1
    elapsed_time = time() - start_time

    # Δkj = Δk
    resetQN && (λmax = one(R)) # opnorm(Bkj)
    # νInv = (1 + θ) * (λmax + maximum(z_l_kj ./ xkj_m_lvar))

    # subproblem: min f(xₖ) + h(xₖ) + φₖ(xₖ)
    bnd_offset_l = (nlow > 0) ? δk * minimum(xkj_m_lvar) : zero(R)
    bnd_offset_u = (nupp > 0) ? δk * minimum(uvar_m_xkj) : zero(R)
    @assert (nlow == 0 || bnd_offset_l > 0) && (nupp == 0 || bnd_offset_u > 0)
    l_bound_kj[ilow] .+= bnd_offset_l
    u_bound_kj[iupp] .-= bnd_offset_u
    set_bounds!(ψ, max.(-Δkj, l_bound_kj), min.(Δkj, u_bound_kj))
    # ∇ϕ1kj = ∇fkj - μk Xkj⁻¹ e
    @. ∇ϕ1kj = ∇fkj
    @. ∇ϕ1kj[ilow] -= μk / xkj_m_lvar
    @. ∇ϕ1kj[iupp] += μk / uvar_m_xkj
    @. ∇ϕ2kj = ∇fkj
    @. ∇ϕ2kj[ilow] -= z_l_kj
    @. ∇ϕ2kj[iupp] += z_u_kj
    j = 0
    optimal_inner = false
    tired_inner = j ≥ maxIter_inner
    ϵk_used = ϵk
    # ϵk_used = max(ϵ, min(μk * 100, R(1.0e-2)))

    while !(optimal_inner || tired_inner)

      n_eval += 1
      Fobj_hist[n_eval] = fkj
      Hobj_hist[n_eval] = hkj
      if n_eval > maxIter
        tired = true
        tired_inner = true
        continue
      end
      j = j + 1
      # subproblem inner:
      # ϕ1(d) = (∇f(xkj) - μk Xkj⁻¹e)ᵀd 
      ϕ1(d) = dot(∇ϕ1kj, d)
      mk1(d) = ϕ1(d) + ψ(d)
      ϕ2(d) = dot(∇ϕ2kj, d)
      mk2(d) = ϕ2(d) + ψ(d)

      # Take first proximal gradient step s1 and see if current xk is nearly stationary.
      # s1 minimizes φ1(s) + ‖s‖² / 2 / ν + ψ(s) ⟺ s1 ∈ prox{νψ}(-ν∇φ1(0)).
      prox!(s, ψ, -ν * ∇ϕ1kj, ν)
      ξk1 = hkj - mk1(s) + max(1, abs(hkj)) * 10 * eps()
      ξk1 > 0 || error("TR: first prox-gradient step should produce a decrease but ξk1 = $(ξk1)")
      if useξz
        prox!(s2, ψ, -ν * ∇ϕ2kj, ν)
        ξk2 = hkj - mk2(s2) + max(1, abs(hkj)) * 10 * eps()
      else
        ξk2 = ξk1
      end
      prox_evals += 1
      # println(" j = $j  ,    ξk1 = $(sqrt(ξk1)) ,     ξk2 = $(sqrt(ξk2))")

      compl_l = norm(xkj_m_lvar .* z_l_kj .- μk)
      compl_u = norm(uvar_m_xkj .* z_u_kj .- μk)

      if ξk2 ≥ 0 && j == 1
        ϵ_increment = (k == 1) ? (stop_ν ? ϵri1 * sqrt(ξk2 / ν) : ϵri1 * sqrt(ξk2)) : (stop_ν ? ϵri * sqrt(ξk2 / ν) : ϵri * sqrt(ξk2))
        ϵk_used += ϵ_increment  # make stopping test absolute and relative
        if k == 1
          ϵ_increment = (stop_ν ? ϵr * sqrt(ξk2 / ν) : ϵr * sqrt(ξk2))
          !stop_proj_gd && (ϵ += ϵ_increment)
        end
        # ϵ_subsolver = ϵ_subsolver_init + ϵ_increment
      end

      if !stop_proj_gd
        if stop_ν
          if μk ≤ ϵμ && sqrt(ξk2 / ν) < ϵ && compl_l < ϵ && compl_u < ϵ 
            optimal = true
            optimal_inner = true
            continue
          end
        else
          if μk ≤ ϵμ && sqrt(ξk2) < ϵ && compl_l < ϵ && compl_u < ϵ 
            # if sqrt(norm(s2) / ν) < ϵ && compl < ϵ && μk ≤ 10 * μmin
            # the current xk is approximately first-order stationary
            optimal = true
            optimal_inner = true
            # crossover ?
            continue
          end
        end
      else
        ############### stop crit TR
        @. l_bound_kj = lvar - xkj
        @. u_bound_kj = uvar - xkj
        ν3 = one(R) / (λmax * (1 + θ) + 1 / (α * Δkj))
        Δ2 = Δkj
        set_bounds!(ψ, max.(-Δ2, l_bound_kj), min.(Δ2, u_bound_kj))
        ϕ3(d) = dot(∇fkj, d)
        mk3(d) = ϕ3(d) + ψ(d)
        prox!(s3, ψ, -ν3 * ∇fkj, ν3)
        ξ3 = hkj - mk3(s3) + max(1, abs(hkj)) * 10 * eps()
        ξ3 > 0 || error("RIPM: first prox-gradient step should produce a decrease but ξ3 = $(ξ3)")
        # println("ξ3 = $ξ3,   √ν⁻¹ξ3 = $(sqrt(ξ3 / ν3))")
        if ξ3 ≥ 0 && k == 1 && j == 1
          ϵ_increment = ϵr * sqrt(ξ3 / ν3)
          ϵ += ϵ_increment  # make stopping test absolute and relative
        end
        if μk ≤ ϵμ && sqrt(ξ3 / ν3) < ϵ && compl_l < ϵ && compl_u < ϵ 
          # if sqrt(norm(s2) / ν) < ϵ && compl < ϵ && μk ≤ 10 * μmin
          # the current xk is approximately first-order stationary
          optimal = true
          optimal_inner = true
          # crossover ?
          continue
        end
        ###############
      end

      # if ξk1 * νInv < ϵk
      if j > 1
        # if sqrt(ξk2 / ν) < ϵk_used && compl < ϵk_used
        optimal_inner = (stop_ν ? (sqrt(ξk2 / ν) < ϵk_used) : (sqrt(ξk2) < ϵk_used)) && compl_l < ϵk && compl_u < ϵk
        # stop_proj_gd && (optimal_inner = optimal_inner && sqrt(ξ3 / ν3) < ϵk) 
        # if sqrt(norm(s2) / ν) < ϵk_used && compl < ϵk_used
          # the current xk is approximately first-order stationary
        optimal_inner && continue
      end
   
      ∆_effective = min(β * χ(s), Δkj)
      @. l_bound_kj = lvar - xkj
      @. u_bound_kj = uvar - xkj
      bnd_offset_l = @views (nlow > 0) ? δk * minimum((xkj[ilow] .- lvar[ilow])) : zero(R)
      bnd_offset_u = @views (nupp > 0) ? δk * minimum((uvar[ilow] .- xkj[ilow])) : zero(R)
      @assert bnd_offset_l ≥ 0 && bnd_offset_u ≥ 0
      l_bound_kj[ilow] .+= bnd_offset_l
      u_bound_kj[iupp] .-= bnd_offset_u
      set_bounds!(ψ, max.(-∆_effective, l_bound_kj), min.(∆_effective, u_bound_kj))

      # model for subsequent prox-gradient steps and ξ
      # ϕ(d) = f(xkj) + φₖ(xkj) + (∇f(xkj) - μk Xkj⁻¹e)ᵀd + dᵀ (Bkj + z_l_kj Xkj⁻¹) d / 2
      # mul_B_XinvZ!(B_XinvZ, Bkj, X_m_lvar⁻¹Ze, ilow, d)
      ϕ(d) = @views dot(∇ϕ1kj, d) + dot(d, mul_B_Θ!(B_Θ, Dkj, Θkj, d)) / 2

      mkj(d) = ϕ(d) + ψ(d)

      @. D_p_Θkj = Dkj.d + Θkj
      iprox!(s, ψ, ∇ϕ1kj, D_p_Θkj)
      prox_evals += 1

      sNorm = χ(s)
      xkjn .= xkj .+ s
      # for i=1:length(xkjn)
      #   if xkjn[i] < sqrt(eps()) * 100
      #     xkjn[i] += sqrt(eps()) * 100
      #   end
      # end
      fkjn = obj(f, xkjn)
      φkjn = @views -μk * (sum(log.(xkjn[ilow] .- lvar[ilow])) + sum(log.(uvar[iupp] .- xkjn[iupp])))
      hkjn = h(xkjn[selected])
      hkjn == -Inf && error("nonsmooth term is not proper")
  
      Δobj = fkj + φkj + hkj - (fkjn + φkjn + hkjn) + max(1, abs(fkj + φkj + hkj)) * 10 * eps()
      ξk = hkj - mkj(s) + max(1, abs(hkj)) * 10 * eps()
  
      if (ξk ≤ 0 || isnan(ξk))
        error("TR: failed to compute a step: ξk = $ξk")
      end
  
      ρkj = Δobj / ξk

      TR_stat = (η2 ≤ ρkj < Inf) ? "↗" : (ρkj < η1 ? "↘" : "=")

      if η2 ≤ ρkj < Inf
        Δkj = max(Δkj, γ * sNorm)
      end
  
      if η1 ≤ ρkj < Inf
        @. z_l_kj = @views (μk - z_l_kj * s[ilow]) / xkj_m_lvar
        @. z_u_kj = @views (μk + z_u_kj * s[iupp]) / uvar_m_xkj
        xkj .= xkjn
        @. xkj_m_lvar = @views xkj[ilow] - lvar[ilow]
        @. uvar_m_xkj = @views uvar[iupp] - xkj[iupp]
        @. l_bound_kj = lvar - xkj
        @. u_bound_kj = uvar - xkj
        proj!(z_l_kj, z_l_kj⁻, μk, xkj_m_lvar, Kzul, Kzuu)
        proj!(z_u_kj, z_u_kj⁻, μk, uvar_m_xkj, Kzul, Kzuu)
        z_l_kj⁻ .= z_l_kj
        z_u_kj⁻ .= z_u_kj
        bnd_offset_l = (nlow > 0) ? δk * minimum(xkj_m_lvar) : zero(R)
        bnd_offset_u = (nupp > 0) ? δk * minimum(uvar_m_xkj) : zero(R)
        @assert bnd_offset_l ≥ 0 && bnd_offset_u ≥ 0
        l_bound_kj[ilow] .+= bnd_offset_l
        u_bound_kj[iupp] .-= bnd_offset_u
        set_bounds!(ψ, max.(-Δkj, l_bound_kj), min.(Δkj, u_bound_kj))
  
        #update functions
        fkj = fkjn
        φkj = φkjn
        hkj = hkjn
        shift!(ψ, xkj)
        ∇fkj = grad(f, xkj)
        ∇ϕ1kj .= ∇fkj
        @. ∇ϕ1kj[ilow] -= μk / xkj_m_lvar
        @. ∇ϕ1kj[iupp] += μk / uvar_m_xkj
        ∇ϕ2kj .= ∇fkj
        @. ∇ϕ2kj[ilow] -= z_l_kj
        @. ∇ϕ2kj[iupp] += z_u_kj
        Θkj .= zero(R)
        Θkj[ilow] += z_l_kj ./ xkj_m_lvar
        Θkj[iupp] += z_u_kj ./ uvar_m_xkj
        for i in 1:n
          if Θkj[i] ≥ threshXinvZ
            Θkj[i] = threshXinvZ
          end
        end 
        # grad!(f, xk, ∇fk)
        push!(Dkj, s, ∇fkj - ∇fkj⁻)
        λmax = norm(Dkj.d, Inf)
        νInv = (1 + θ) * (λmax + maximum(Θkj))
        ∇fkj⁻ .= ∇fkj
        IterSucc[n_eval + 1] = true
      end
  
      if ρkj < η1 || ρkj == Inf
        Δkj = Δkj / 2
        @. l_bound_kj = lvar - xkj
        @. u_bound_kj = uvar - xkj
        bnd_offset_l = (nlow > 0) ? δk * minimum(xkj_m_lvar) : zero(R)
        bnd_offset_u = (nupp > 0) ? δk * minimum(uvar_m_xkj) : zero(R)
        @assert bnd_offset_l ≥ 0 && bnd_offset_u ≥ 0
        l_bound_kj[ilow] .+= bnd_offset_l
        u_bound_kj[iupp] .-= bnd_offset_u
        set_bounds!(ψ, max.(-Δkj, l_bound_kj), min.(Δkj, u_bound_kj))
      end

      compl_l = norm(xkj_m_lvar .* z_l_kj .- μk)
      compl_u = norm(uvar_m_xkj .* z_u_kj .- μk)
      compl = sqrt(compl_l^2 + compl_u^2) 

      if (verbose > 0) && (k % ptf == 0)
        #! format: off
        @info @sprintf "%6d %8d %8d %8.1e %8.1e %7.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s %7.1e %7.1e" k j 1 fkj hkj sqrt(ξk2 / ν) sqrt(ξk) ρkj ∆_effective χ(xkj) sNorm νInv TR_stat μk compl
        #! format: on
      end
      ν = 1 / (νInv + 1 / (Δkj * α))
      # ν = 1 / (1 + 1 / (Δkj * α))
      tired_inner = j ≥ maxIter_inner || elapsed_time > maxTime
    end
    Complex_hist[k] = j

    # if ξk1 ≥ 0 && k == 1 # && j == 1
    #   # ϵ_increment = ϵr * sqrt(ξk2 / ν)
    #   ϵ_increment = ϵr * sqrt(ξk2)
    #   ϵ += ϵ_increment  # make stopping test absolute and relative
    # end
    compl_l = norm(xkj_m_lvar .* z_l_kj .- μk)
    compl_u = norm(uvar_m_xkj .* z_u_kj .- μk)
    compl = sqrt(compl_l^2 + compl_u^2) 
    # println("ϵ = $ϵ ,  ϵk = $ϵk_used,  √ξk1/√ν = $(sqrt(ξk2 / ν)) , compl = $compl , μk = $μk")
    # if sqrt(ξk2 / ν) < ϵ && compl < ϵ && μk ≤ 10 * μmin
    optimal == true && continue

    (μk ≥ μmin) && (μk /= 10)
    # ϵk > ϵ && (ϵk /= 10)
    # (μk ≥ μmin) && min(μk /= 10, μk^R(1.5))
    ϵk = μk^R(1.01)
    (δk ≥ δmin) && (δk /= 2)

    @. l_bound_kj = lvar - xkj
    @. u_bound_kj = uvar - xkj
    # @. z_l_kj = @views -μk / l_bound_kj[ilow]
    # @. z_l_kj[iupp] = @views μk / u_bound_kj[iupp]
    # @. z_l_kj[irng] = @views μk / (-l_bound_kj[irng] + u_bound_kj[irng])
    # fkj = obj(f, xkj)
    φkj = -μk * (sum(log.(xkj_m_lvar)) + sum(log.(uvar_m_xkj)))
    # ∇fkj = grad(f, xkj)
    ∇fkj⁻ .= ∇fkj
    z_l_kj⁻ .= z_l_kj
    z_u_kj⁻ .= z_u_kj
    # hkj = h(xkj[selected])
    shift!(ψ, xkj)
    if resetQN
      reset!(Dkj)
    end

    νInv = (1 + θ) * (λmax + maximum(Θkj))
    # Δkj = Δk
    Δkj = μk * 1000
    ν = 1 / (νInv + 1 / (Δkj * α))
    tired = tired || k ≥ maxIter_outer || elapsed_time > maxTime
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8s %8.1e %8.1e" k "" fkj hkj
    elseif optimal
      #! format: off
      @info @sprintf "%6d %8d %8d %8.1e %8.1e %7.1e %7.1e %8s %7.1e %7.1e %7.1e %7.1e" k 0 1 fkj hkj sqrt(ξk2 / ν) sqrt(ξk1) "" Δkj χ(xkj) χ(s) νInv
      #! format: on
      @info "RIPM: terminating with √ξ1/√ν = $(stop_ν ? sqrt(ξk2 / ν) : sqrt(ξk2))"
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

  @. xkj_m_lvar = @views xkj[ilow] - lvar[ilow]
  @. uvar_m_xkj = @views uvar[iupp] - xkj[iupp]
  # crossover!(xkj, z_l_kj, z_u_kj, ilow, iupp, μk, lvar)
  crossover && crossover!(xkj, z_l_kj, z_u_kj, ilow, iupp, lvar, uvar, μk)

  z_l_kj_out = fill!(similar(xkj), zero(R))
  z_l_kj_out[ilow] .= z_l_kj
  z_u_kj_out = fill!(similar(xkj), zero(R))
  z_u_kj_out[iupp] .= z_u_kj
  stats = GenericExecutionStats(f)
  set_status!(stats, status)
  set_solution!(stats, xkj)
  has_bounds(f) && set_bounds_multipliers!(stats, z_l_kj_out, z_u_kj_out)
  set_objective!(stats, fkj + hkj)
  set_residuals!(stats, zero(eltype(xkj)), ξk2 ≥ 0 ? (stop_ν ? sqrt(ξk2 / ν) : sqrt(ξk2)) : ξk2)
  set_iter!(stats, k)
  set_time!(stats, elapsed_time)
  set_solver_specific!(stats, :Fhist, Fobj_hist[1:n_eval])
  set_solver_specific!(stats, :Hhist, Hobj_hist[1:n_eval])
  set_solver_specific!(stats, :IterSucc, IterSucc[1:n_eval])
  set_solver_specific!(stats, :NonSmooth, h)
  set_solver_specific!(stats, :SubsolverCounter, Complex_hist[1:k+1])
  set_solver_specific!(stats, :ProxEvals, prox_evals)
  return stats
end
