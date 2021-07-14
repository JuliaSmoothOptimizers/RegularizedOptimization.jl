
using Logging, ShiftedProximalOperators, Arpack, Roots
function TRalg(
  f::AbstractNLPModel,
  h::ProximableFunction,
  χ::ProximableFunction,
  options;
  x0::AbstractVector=f.meta.x0,
  s_alg = QRalg,
  subsolver_logger=Logging.NullLogger(), 
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
  τ = options.τ
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
  Complex_hist = zeros(Int, maxIter)
  @info @sprintf "%6s %8s %8s %8s %7s %7s %8s %7s %7s %7s %7s %1s" "iter" "PG iter" "f(x)" "h(x)" "ξ1" "ξ" "ρ" "Δ" "‖x‖" "‖s‖" "‖Bₖ‖" "TR"

  k = 0
  ρk = -1.0
  α = 1.0
  sNorm = 0.0
  TR_stat = ""

  # keep track of old values, initialize functions
  ∇fk = grad(f, xk)
  fk = obj(f, xk)
  hk = ψ.h(xk)
  s = zero(xk)
  ∇fk⁻ = copy(∇fk)
  funEvals = 1

  quasiNewtTest = isa(f, QuasiNewtonModel)
  Bk = hess_op(f, xk)
  # define the Hessian
  H = Symmetric(Matrix(Bk))
  νInv = (1 + θ) * maximum(abs.(eigs(H; nev=1, which=:LM)[1]))

  ξ = 0.0
  ξ1 = 0.0
  optimal = false
  tired = k ≥ maxIter

  while !(optimal || tired)
    # update count
    k = k + 1
    Fobj_hist[k] = fk
    Hobj_hist[k] = hk
    # Print values
    k % ptf == 0 && @info @sprintf "%6d %8d %8.1e %8.1e %7.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s" k funEvals fk hk ξ1 ξ ρk Δk χ(xk) sNorm νInv TR_stat

    # define inner function
    ∇φ(d) = begin
      return H * d + ∇fk
    end

    φ(d) = begin
        return 0.5 * (d' * (H * d)) + ∇fk' * d
    end

    # define model and update ρ
    mk(d) = φ(d) + ψ(d)

    # take initial step s1 and see if you can do more
    subsolver_options.ν = min(1 / νInv, Δk)
    s1 = proxtemp(ψ, -subsolver_options.ν * ∇fk, subsolver_options.ν) # -> PG on one step s1
    ξ1 = hk - mk(s1) + max(1, abs(hk)) * 10 * eps()
    ξ1 > 0 || error("TR: first prox-gradient step should produce a decrease but ξ1 = $(ξ1)")

    if ξ1 < ϵ
      # the current xk is approximately first-order stationary
      optimal = true
      @info "TR: terminating with ξ1 = $(ξ1)"
      continue
    end
    subsolver_options.ϵ = k == 1 ? 1.0e-5 : max(ϵ, min(.01, sqrt(ξ1)) * ξ1)
    set_radius!(ψ, min(β * χ(s1), Δk))
    s, funEvals, _, _, _ = with_logger(subsolver_logger) do 
      s_alg(φ, ∇φ, ψ, subsolver_options; x0 = s1)
    end
    # update Complexity history
    Complex_hist[k] += funEvals

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
      #grad!(f, xk, ∇fk)
      if quasiNewtTest
        push!(f, s, ∇fk - ∇fk⁻)
      end
      Bk = hess_op(f, xk)
      H = Symmetric(Matrix(Bk))
      νInv = (1 + θ) * maximum(abs.(eigs(H; nev=1,  v0 = randn(m,), which=:LM)[1]))
      # store previous iterates
      ∇fk⁻ .= ∇fk

      #hist update
      Complex_hist[k] += 1
    end

    if ρk < η1 || ρk == Inf
      TR_stat = "↘"
      α = .5
      Δk = α * Δk	# change to reflect trust region
      set_radius!(ψ, Δk)
    end
    tired = k ≥ maxIter

  end

  return xk, k, Fobj_hist[Fobj_hist .!= 0], Hobj_hist[Fobj_hist .!= 0], Complex_hist[Complex_hist .!= 0]
end

QRa(nlp::AbstractNLPModel, args...; kwargs...) = QRa(x -> obj(nlp, x), x -> grad(nlp, x), args...; kwargs...)

function QRa(
  f,
  ∇f,
  h,
  options;
  x0::AbstractVector=f.meta.x0,
  )
  # initialize passed options
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
  Complex_hist = zeros(Int, maxIter)
  verbose == 0 || @info @sprintf "%6s %8s %8s %7s %8s %7s %7s %7s %1s" "iter" "f(x)" "h(x)" "ξ" "ρ" "σ" "‖x‖" "‖s‖" ""

  k = 0
  ρk = -1.0
  TR_stat = ""
  σk = 1/ν

  fk = f(xk)
  ∇fk = ∇f(xk)
  hk = h(xk)

  ξ = 0.0
  optimal = false
  tired = maxIter > 0 && k ≥ maxIter

  while !(optimal || tired)
    k = k + 1

    Fobj_hist[k] = fk
    Hobj_hist[k] = hk
    k % ptf == 0 && @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e %7.1e %7.1e %1s" k fk hk ξ ρk σk norm(xk) norm(s) TR_stat
    # define model
    φk(d) = dot(∇fk, d)
    mk(d) = φk(d) + ψ(d)

    s = proxtemp(ψ, -ν * ∇fk, ν)
    mks = mk(s)
    ξ = hk - mks + max(1, abs(hk)) * 10 * eps()
    if (ξ < 0 || isnan(ξ))
      @show k, ξ, ϵ, σk, ψ(s), φk(s), ψ.χ(s + ψ.sj), ψ.Δ
      @show ψ.sj, s, ν
      error("QR: failed to compute a step: ξ = $ξ")
    end

    if ξ < ϵ
      optimal = true
      verbose == 0 || @info "QR: terminating with ξ = $ξ"
      continue
    end

    xkn .= xk .+ s
    fkn = f(xkn)
    hkn = h(xkn)
    Δobj = (fk + hk) - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ρk = Δobj / ξ

    if η2 ≤ ρk < Inf
      @debug "very successful step"
      TR_stat = "↘"
      σk = σk / γ
    else
      TR_stat = "="
    end

    if η1 ≤ ρk < Inf
      ρk < η2 && @debug "successful step"
      xk .= xkn
      fk = fkn
      hk = hkn
      ∇fk = ∇f(xk)
      shift!(ψ, xk)
      Complex_hist[k] += 1
    end

    if ρk < η1 || ρk == Inf
      @debug "unsuccessful step"
      TR_stat = "↗"
      σk = σk * γ
      # σk = max(σk * γ, 1e-6)
    end

    ν = 1 / σk
    tired = maxIter > 0 && k ≥ maxIter
  end

  return xk, k, Fobj_hist[Fobj_hist .!= 0], Hobj_hist[Fobj_hist .!= 0], Complex_hist[Complex_hist .!= 0]
end

mutable struct ShiftedNormL1B2temp{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::NormL1{R}
  xk::V0
  sj::V1
  sol::V2
  Δ::R
  χ::NormL2{R}
  shifted_twice::Bool

  function ShiftedNormL1B2temp(
    h::NormL1{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    Δ::R,
    χ::NormL2{R},
    shifted_twice::Bool,
  ) where {R <: Real}
    sol = similar(sj)
    new{R, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, Δ, χ, shifted_twice)
  end
end

(ψ::ShiftedNormL1B2temp)(y) = ψ.h(ψ.xk + ψ.sj + y) + IndBallL2(ψ.Δ)(ψ.sj + y)

shifted(h::NormL1{R}, xk::AbstractVector{R}, Δ::R, χ::NormL2{R}) where {R <: Real} =
  ShiftedNormL1B2temp(h, xk, zero(xk), Δ, χ, false)
shifted(
  ψ::ShiftedNormL1B2temp{R, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
  ShiftedNormL1B2temp(ψ.h, ψ.xk, sj, ψ.Δ, ψ.χ, true)

fun_name(ψ::ShiftedNormL1B2temp) = "shifted L1 norm with L2-norm trust region indicator"
fun_expr(ψ::ShiftedNormL1B2temp) = "t ↦ ‖xk + sj + t‖₁ + χ({‖sj + t‖₂ ≤ Δ})"
fun_params(ψ::ShiftedNormL1B2temp) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "Δ = $(ψ.Δ)"

function proxtemp(
  ψ::ShiftedNormL1B2temp{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  ProjB(y) = min.(max.(y, ψ.sj .+ q .- ψ.λ * σ), ψ.sj .+ q .+ ψ.λ * σ)
  froot(η) = η - ψ.χ(ProjB((- ψ.xk) .* (η / ψ.Δ)))

  ψ.sol .= ProjB(- ψ.xk)

  if ψ.Δ ≤ ψ.χ(ψ.sol)
    η = fzero(froot, ψ.Δ, Inf)
    ψ.sol .= ProjB((- ψ.xk) .* (η / ψ.Δ)) * (ψ.Δ / η)
  end
  ψ.sol .-= ψ.sj

  if isnan(sum(ψ.sol))
    @show ψ.sol, η, ψ.Δ, ψ.χ(ψ.sol), ψ.sj, ψ.xk, q, ψ.λ * σ
  end
  return ψ.sol
end
