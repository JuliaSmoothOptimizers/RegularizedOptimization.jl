# Implements Algorithm 4.2 in "Interior-Point Trust-Region Method for Composite Optimization".

using NLPModelsModifiers, NLPModels, LinearAlgebra, Arpack, ShiftedProximalOperators
using ShiftedProximalOperators: prox


function TRalg(f, 
    h::ProximableFunction, 
    methods, 
    params;
    x0::AbstractVector=f.meta.x0
    )

  # initialize passed options
  ϵ = params.ϵ
  Δk = params.Δk
  verbose = params.verbose
  maxIter = params.maxIter
  η1 = params.η1
  η2 = params.η2 
  γ = params.γ
  τ = params.τ
  θ = params.θ
  β = params.β
  mem = params.mem
  if verbose == 0
    ptf = Inf
  elseif verbose == 1
    ptf = round(maxIter / 10)
  elseif verbose == 2
    ptf = round(maxIter / 100)
  else
    ptf = 1
  end

  # other methods
  FO_options = methods.FO_options
  s_alg = methods.s_alg
  χ = methods.χ 

  # initialize parameters
  xk = copy(x0)
  xkn = similar(xk)
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
  # νInv = eigmax(H) #make a Matrix? ||B_k|| = λ(B_k) # change to opNorm(Bk, 2), arPack? 
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
    ∇φ(d) = H * d + ∇fk
    φ(d) = 0.5 * (d' * (H * d)) + ∇fk' * d + fk

    # define model and update ρ
    mk(d) = φ(d) + ψ(d)

    # take initial step s1 and see if you can do more 
    FO_options.ν = min(1 / νInv, Δk)
    s1 = prox(ψ, -FO_options.ν * ∇fk, FO_options.ν) # -> PG on one step s1
    ξ1 = fk + hk - mk(s1)
    # @show s1,H, s1'*H*s1, mk(s1), φ(s1), ψ(s1), ∇fk' * s1, ∇fk
    # @show ξ1, hk, mk(s1), 0.5 * (s1' * (H * s1)), ∇fk'*s1, fk
    ξ1 > 0 || error("TR: first prox-gradient step should produce a decrease but ξ1 = $(ξ1)")
    if ξ1 < ϵ
        # the current xk is approximately first-order stationary
        optimal = true
        @info "TR: terminating with ξ1 = $(ξ1)"
        continue
    end

    FO_options.optTol = k == 1 ? 1.0e-5 : max(ϵ, min(.01, sqrt(ξ1)) * ξ1)
    set_radius!(ψ, min(β * χ(s1), Δk))
    FO_options.verbose = 0
    inner_params = TRNCmethods(FO_options = methods.FO_options, χ = χ)
    inner_options = TRNCparams(; maxIter = 20000, verbose = 0, ϵ = FO_options.optTol)
    s, funEvals, _, _, _, = QRalg2(φ, ∇φ, ψ, s1, inner_params, inner_options)
    # (s, funEvals) = s_alg(φ, ∇φ, ψ, s1, FO_options)

    # update Complexity history 
    Complex_hist[k] += funEvals

    sNorm =  χ(s)
    xkn .= xk .+ s
    fkn = obj(f, xkn)
    hkn = h(xkn)

    Δobj = fk + hk - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ξ = fk + hk - mk(s)

    if (ξ ≤ 0 || isnan(ξ))
      error("failed to compute a step")
    end

    ρk = Δobj / ξ

    if η2 ≤ ρk < Inf
      TR_stat = "↗"
      Δk = max(Δk, γ * ψ.χ(s))
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
      #update gradient & hessian 
      ∇fk = grad(f, xk)
        #grad!(f, xk, ∇fk)
      if quasiNewtTest
        push!(f, s, ∇fk - ∇fk⁻)
      end
      Bk = hess_op(f, xk)
        # define the Hessian 
      H = Symmetric(Matrix(Bk))
      νInv = (1 + θ) * maximum(abs.(eigs(H;nev=1, which=:LM)[1]))
            
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