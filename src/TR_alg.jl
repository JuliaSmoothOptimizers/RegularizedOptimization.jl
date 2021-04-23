# Implements Algorithm 4.2 in "Interior-Point Trust-Region Method for Composite Optimization".

using NLPModelsModifiers, LinearAlgebra, Arpack, ShiftedProximalOperators
using ShiftedProximalOperators:prox
export TR

"""Interior method for Trust Region problem
  TR(x, params, options)
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
  -χk, function that is the TR norm ball; defaults to l2 

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
function TR(f, h, methods, params)

  # initialize passed params
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
  xk = deepcopy(f.meta.x0)

  # initialize shift
  ψ = shifted(h, xk, Δk, χ)

  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(maxIter)
  @info @sprintf "%6s %8s %8s %8s %7s %7s %8s %7s %7s %7s %7s %1s" "iter" "PG iter" "f(x)" "h(x)" "ξ1" "ξ" "ρ" "Δ" "‖x‖" "‖s‖" "‖Bₖ‖" "TR"

  k = 0
  ρk = -1.0
  α = 1.0
  TR_stat = ""

  # main algorithm initialization
  # test to see if user provided a hessian
  # quasiNewtTest = (f_obj.Hess == LSR1Operator) || (f_obj.Hess==LBFGSOperator)
  quasiNewtTest = isa(f, QuasiNewtonModel)
  if quasiNewtTest
    Bk = hess_op(f, xk)
  else
    Bk = hess(f, xk)
  end
  # define the Hessian 
  H = Symmetric(Matrix(Bk))
  # νInv = eigmax(H) #make a Matrix? ||B_k|| = λ(B_k) # change to opNorm(Bk, 2), arPack? 
  νInv = (1 + θ) * maximum(abs.(eigs(H; nev=1, which=:LM)[1]))

  # keep track of old values, initialize functions
  ∇fk = grad(f, xk)
  fk = obj(f, xk)
  hk = ψ.h(xk)
  s = zeros(size(xk))
  ∇fk⁻ = ∇fk
  funEvals = 1

  sNorm = 0.0
  ξ = 0.0
  ξ1 = ξ
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
    φ(d) = H * d + ∇fk # (φ, ∇φ, ∇²φ)

    # define model and update ρ
    mk(d) = 0.5 * (d' * (H * d)) + ∇fk' * d + fk + ψ(d) # psik = h -> psik = h(x+d)

    # take initial step s1 and see if you can do more 
    FO_options.ν = min(1 / νInv, Δk)
    s1 = prox(ψ, -FO_options.ν * ∇fk, 1 / νInv) # -> PG on one step s1
    ξ1 = fk + hk - mk(s1)

    if ξ1 > ϵ || k == 1 
      FO_options.optTol = min(ϵ, ξ1) * ξ1 # stopping criteria for inner algorithm 
      FO_options.FcnDec = ξ1
      set_radius!(ψ, min(β * ψ.χ(s1), Δk))
      (s, funEvals) = s_alg(φ, ψ, s1, FO_options)
      ξ = fk + hk - mk(s)
    else
      s .= s1
      funEvals = 1 
      ξ = ξ1
    end

    # update Complexity history 
    Complex_hist[k] += funEvals # doesn't really count because of quadratic model 
    sNorm = ψ.χ(s)

    fkn = obj(f, xk + s)
    hkn = ψ.h(xk + s)

    Δobj = fk + hk - (fkn + hkn)
    optimal = ξ1 < ϵ

    if (ξ ≤ 0 || isnan(ξ))
      error("failed to compute a step")
    end

    ρk = (Δobj + 1e-16) / (ξ + 1e-16)

    if η2 ≤ ρk < Inf
      TR_stat = "↗"
      Δk = max(Δk, γ * ψ.χ(s))
    else
      TR_stat = "="
    end

    if η1 ≤ ρk < Inf
      xk .+= s

      # update functions
      fk = fkn
      hk = hkn

      # update gradient & hessian 

      if !optimal 
        grad!(f, xk, ∇fk)
        if quasiNewtTest
          push!(f, s, ∇fk - ∇fk⁻)
          Bk = hess_op(f, xk)
        else
          Bk = hess(f, xk)
        end
        # define the Hessian 
        H = Symmetric(Matrix(Bk))
        # β = eigmax(H) #make a Matrix? ||B_k|| = λ(B_k) # change to opNorm(Bk, 2), arPack? 
        νInv = (1 + θ) * maximum(abs.(eigs(H;nev=1, which=:LM)[1]))
            
        # store previous iterates
        ∇fk⁻ .= ∇fk
      end

      # hist update 
      Complex_hist[k] += 1
    end

    if ρk < η1 || ρk == Inf
      TR_stat = "↘"
      α = .5
      Δk = α * Δk	# * norm(s, Inf) #change to reflect trust region 
    end

    shift!(ψ, xk) # inefficient but placed here for now 
    set_radius!(ψ, Δk)
    
  end

  return xk, k, Fobj_hist[Fobj_hist .!= 0], Hobj_hist[Fobj_hist .!= 0], Complex_hist[Complex_hist .!= 0]
end