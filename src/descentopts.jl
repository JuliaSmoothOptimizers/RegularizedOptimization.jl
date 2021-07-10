export TRNCoptions, SmoothObj

mutable struct TRNCoptions
  ϵ # termination criteria
  Δk # trust region radius
  verbose # print every so often
  maxIter # maximum amount of inner iterations
  η1 # ρ lower bound
  η2 # ρ upper bound
  τ # linesearch buffer parameter
  ν # initial guess for ν
  γ # trust region buffer
  θ # TR inner loop "closeness" to Bk
  β # TR size for PG steps j>1

  function TRNCoptions(
    ;
    ϵ=1e-2,
    Δk=1.0,
    verbose=0,
    maxIter=10000,
    η1=1.0e-3, # ρ lower bound
    η2=0.9,  # ρ upper bound
    τ=0.01, # linesearch buffer parameter
    ν=1.0e-3,
    γ=3.0, # trust region buffer
    θ=1e-3,
    β=10.0,
    ) # default values for trust region parameters in algorithm 4.2
    return new(ϵ, Δk, verbose, maxIter, η1, η2, τ, ν, γ, θ, β)
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
  nlp.g(g, x)
  return g
end