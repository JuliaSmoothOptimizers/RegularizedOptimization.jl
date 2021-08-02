# Implements Algorithm 4.2 in "Interior-Point Trust-Region Method for Composite Optimization".

export QRalg

"""Interior method for Trust Region problem
    QuadReg(x, params, options)
Arguments
----------
x : Array{Float64,1}
    Initial guess for the x value used in the trust region
params : mutable structure TR_params with:
    --
    -ϵ, tolerance for primal convergence
    -Δk Float64, trust region radius
    -verbose Int, print every # options
    -maxIter Float64, maximum number of inner iterations (note: does not influence TotalCount)
options : mutable struct TR_methods
    -f_obj, smooth objective struct; eval, grad, Hess
    -ψ, nonsmooth objective struct; h, ψ, ψχprox - function projecting onto the trust region ball or ψ+χ
    --
    -FO_options, options for first order algorithm, see DescentMethods.jl for more
    -s_alg, algorithm for descent direction, see DescentMethods.jl for more

Returns
-------
x   : Array{Float64,1}
    Final value of Algorithm 4.2 trust region
k   : Int
    number of iterations used
Fobj_hist: Array{Float64,1}
    smooth function history
Hobj_hist: Array{Float64, 1}
    nonsmooth function history
Complex_hist: Array{Float64, 1}
    inner algorithm iteration count

"""
QRalg(nlp::AbstractNLPModel, args...; kwargs...) = QRalg(x -> obj(nlp, x), x -> grad(nlp, x), args...; kwargs...)

function QRalg(
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
  Complex_hist = zeros(Int, (2,maxIter))
  verbose == 0 || @info @sprintf "%6s %8s %8s %7s %8s %7s %7s %7s %1s" "iter" "f(x)" "h(x)" "√ξ" "ρ" "σ" "‖x‖" "‖s‖" ""

  k = 0
  ρk = -1.0
  TR_stat = ""
  σk = 1/ν

  fk = f(xk)
  ∇fk = ∇f(xk)
  hk = h(xk)
  Hist_gradeval = [fk + hk]

  ξ = 0.0
  optimal = false
  tired = maxIter > 0 && k ≥ maxIter

  while !(optimal || tired)
    k = k + 1

    Fobj_hist[k] = fk
    Hobj_hist[k] = hk
    k % ptf == 0 && @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e %7.1e %7.1e %1s" k fk hk sqrt(ξ) ρk σk norm(xk) norm(s) TR_stat

    # define model
    φk(d) = dot(∇fk, d)
    mk(d) = φk(d) + ψ(d)

    s = ShiftedProximalOperators.prox(ψ, -ν * ∇fk, ν)
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
      Complex_hist[1,k]+=1
      append!(Hist_gradeval, fk + hk)
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

  return xk, Hist_gradeval, Fobj_hist[1:k], Hobj_hist[1:k], Complex_hist[:,1:k]
end
