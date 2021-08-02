using TRNC
using ADNLPModels, NLPModelsModifiers,NLPModels
using DifferentialEquations, ProximalOperators, ForwardDiff
using LinearAlgebra, Random, Printf
Random.seed!(1234)
function ODEFH()
  # so we need a model solution, a gradient, and a Hessian of the system (along with some data to fit)
  function FH_ODE(dx, x, p, t)
    # p is parameter vector [I,μ, a, b, c]
    V, W = x 
    I, μ, a, b, c = p
    dx[1] = (V - V^3 / 3 -  W + I) / μ
    dx[2] = μ * (a * V - b * W + c)
  end

  u0 = [2.0; 0.0]
  tspan = (0.0, 20.0)
  savetime = .2

  pars_FH = [0.5, 0.08, 1.0, 0.8, 0.7]
  prob_FH = ODEProblem(FH_ODE, u0, tspan, pars_FH)

  # So this is all well and good, but we need a cost function and some parameters to fit. First, we take care of the parameters
  # We start by noting that the FHN model is actually the van-der-pol oscillator with some parameters set to zero
  # x' = μ(x - x^3/3 - y)
  # y' = x/μ -> here μ = 12.5
  # changing the parameters to p = [0, .08, 1.0, 0, 0]
  x0 = [0, .2, 1.0, 0, 0]
  prob_VDP = ODEProblem(FH_ODE, u0, tspan, x0)
  sol_VDP = solve(prob_VDP, reltol=1e-6, saveat=savetime)

  # also make some noise to fit later
  t = sol_VDP.t
  b = hcat(sol_VDP.u...)
  noise = .1 * randn(size(b))
  data = noise + b

  # so now that we have data, we want to formulate our optimization problem. This is going to be 
  # min_p ||f(p) - b||₂^2 + λ||p||₀
  # define your smooth objective function
  function CostFunc(p)
    temp_prob = remake(prob_FH, p=p)
    temp_sol = solve(temp_prob, Vern9(), abstol=1e-14, reltol=1e-14, saveat=savetime)
    tot_loss = 0.

    if any((temp_sol.retcode != :Success for s in temp_sol))
      tot_loss = Inf
    else
      temp_v = convert(Array, temp_sol)
      tot_loss = sum(((temp_v - data).^2) ./ 2)
    end
    return tot_loss
  end
return CostFunc
end

function phirod(cfunc; λ = 1.0)
  nonsmoothfunc = NormL0(λ)

  # put in your initial guesses
  xi = ones(5,)
  # this is for l0 norm 
  # Smoothfunc = LBFGSModel(ADNLPModel(cfunc, xi))
  Smoothfunc = LSR1Model(ADNLPModel(cfunc, xi))
  # Smoothfunc = ADNLPModel(cfunc, xi)
  return Smoothfunc, nonsmoothfunc
end

function runode(func, hunc; ϵ = 1e-3)
  Random.seed!(1234)
  # set all options
  params = TRNCoptions(; maxIter = 10000, verbose=10, ϵ=ϵ, β=1e16)
  subopts = params; 
  xtr, k, Fhist, Hhist, Comp_pg = TRalg(func, hunc, NormLinf(1.0), params; subsolver_options = subopts)
  # xtr, k, Fhist, Hhist, Comp_pg = TR(func, hunc, NormLinf(1.0), params; subsolver_options = subopts)
  # xtr, k, Fhist, Hhist, Comp_pg = LMTR(func, hunc,  methods, params)
  return xtr
end