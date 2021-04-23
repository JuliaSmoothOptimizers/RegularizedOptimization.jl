# Implements Algorithm 4.2 in "Interior-Point Trust-Region Method for Composite Optimization".

using LinearAlgebra, Arpack
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
function QRalg(f, h, params, options)

  # initialize passed options
  ϵ = options.ϵ
  verbose = options.verbose
  maxIter = options.maxIter
  η1 = options.η1
  η2 = options.η2 
  σk = options.σk 
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

  # other parameters
  FO_options = params.FO_options
  s_alg = params.s_alg
  χ = params.χ


  # initialize parameters
  xk = f.meta.x0
  ψ = shifted(h, xk)

  k = 0
  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, maxIter)
  @info @sprintf "%6s %8s %8s %7s %8s %7s %7s %7s %1s" "iter" "f(x)" "h(x)" "ξ" "ρ" "σ" "‖x‖" "‖s‖" ""

  k = 0
  ρk = -1.0
  TR_stat = ""
  
  # main algorithm initialization
  ∇fk = grad(f, xk)
  fk = obj(f, xk)
  hk = ψ.h(xk) #hk = h_obj(xk)

  ν = 1 / σk
  s = zeros(size(xk))
  funEvals = 1

  sNorm = 0.0
  ξ = 0.0
  optimal = false
  tired = k ≥ maxIter

  while !(optimal || tired) 
    # update count
    k = k + 1 # inner

    Fobj_hist[k] = fk
    Hobj_hist[k] = hk
    # Print values
    k % ptf == 0 && @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e %7.1e %7.1e %1s" k fk hk ξ ρk σk χ(xk) sNorm TR_stat
    
    # define model
    φk(d) = ∇fk' * d + fk
    mk(d) = φk(d) + ψ(d) # psik = h -> psik = h(x+d)

    s = prox(ψ, -ν * ∇fk, ν) # -> PG on one step s
    sNorm = χ(s)

    fkn = obj(f, xk + s)
    hkn = ψ(s)
    Δobj = (fk + hk) - (fkn + hkn)
    ξ = (fk + hk) - mk(s)

    optimal = ξ < ϵ
    
    if (ξ ≤ 0 || isnan(ξ))
      error("failed to compute a step")
    end

    ρk = (Δobj + 1e-16) / (ξ + 1e-16)

    if η2 ≤ ρk < Inf
      TR_stat = "↗"
      σk = σk / γ
    else
      TR_stat = "="
    end

    if η1 ≤ ρk < Inf
      xk .+= s 

      #update functions 
      fk = fkn
      hk = hkn

      
      if !optimal
        ∇fk = grad(f, xk)
      end
      #update gradient 
      Complex_hist[k] += 1

      shift!(ψ, xk)
    end

    if ρk < η1 || ρk == Inf
      TR_stat = "↘"
      σk = max(σk * γ, 1e-6) # dominique σmin ok? 
    end

    ν = 1 / σk
    tired = k ≥ maxIter
      
      
  end

  return xk, k, Fobj_hist[Fobj_hist .!= 0], Hobj_hist[Fobj_hist .!= 0], Complex_hist[Complex_hist .!= 0]
end
