using NLPModels, ADNLPModels, ProximalOperators
export s_params, TRNCparams, TRNCmethods, SmoothObj#, FObj, HObj

mutable struct s_params
  ν
  λ
  optTol
  maxIter
  verbose
  p
  FcnDec

  function s_params(ν, λ; optTol=1f-6, maxIter=10000, verbose=0, p=1.1, FcnDec=1e10)
    return new(ν,λ, optTol, maxIter, verbose, p, FcnDec)
  end
end

mutable struct TRNCparams
  ϵ # termination criteria
  Δk # trust region radius
  verbose # print every so often
  maxIter # maximum amount of inner iterations
  η1 # ρ lower bound 
  η2 # ρ upper bound 
  τ # linesearch buffer parameter 
  σk # quadratic model linesearch buffer parameter
  γ # trust region buffer 
  mem # Bk iteration memory
  θ # TR inner loop "closeness" to Bk
  β # TR size for PG steps j>1

  function TRNCparams(
    ;
    ϵ=1e-2,
    Δk=1.0,
    verbose=0,
    maxIter=10000,
    η1=1.0e-3, # ρ lower bound
    η2=0.9,  # ρ upper bound
    τ=0.01, # linesearch buffer parameter
    σk=1.0e-3, # LM parameter
    γ=3.0, # trust region buffer
    mem=5, # L-BFGS memory
    θ=1e-3,
    β=10.0
    ) # default values for trust region parameters in algorithm 4.2
    return new(ϵ, Δk, verbose, maxIter, η1, η2, τ, σk, γ, mem, θ, β)
  end
end

mutable struct SmoothObj <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters

  #functions 
  f 
  g
  function SmoothObj(f, g, x::AbstractVector{T};  name = "F(x)_smooth") where T
    meta = NLPModelMeta(length(x), x0 = x, name=name)
    return new(meta, Counters(), f, g)
  end
end 

function NLPModels.obj(nlp::SmoothObj, x::AbstractVector)
  increment!(nlp, :neval_obj)
  return nlp.f(x)
end
# function NLPModels.grad!(nlp::SmoothObj, x::AbstractVector, g :: AbstractVector)
#   increment!(nlp, :neval_grad)
#   g .= nlp.g(x)
#   return g
# end
function NLPModels.grad!(nlp::SmoothObj, x::AbstractVector, g :: AbstractVector)
  increment!(nlp, :neval_grad)
  nlp.g!(g, x)
  return g
end


mutable struct TRNCmethods
  FO_options # options for minimization routine you use for s; based on minconf_spg
  s_alg # algorithm passed that determines descent direction 
  χ # TR norm one computes for the trust region radius - default is l2 

  function TRNCmethods(;
    FO_options=s_params(1.0, 1.0),
    s_alg=PG,
    χ= NormL2(1.0),
  )
    return new(FO_options, s_alg, χ)
  end
end