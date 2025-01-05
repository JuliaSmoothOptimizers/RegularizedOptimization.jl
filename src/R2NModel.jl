export R2NModel

mutable struct R2NModel{T <: Real, V <: AbstractVector{T}, G <: AbstractLinearOperator{T}} <: AbstractNLPModel{T, V}
  B :: G
  ∇f :: V
  v :: V
  σ :: T
  meta::NLPModelMeta{T, V}
  counters::Counters
end
  
  function R2NModel(
  B :: G,
  ∇f :: V,
  v :: V,
  σ :: T,
  x0 :: V
) where{T, V, G}
  meta = NLPModelMeta(
    length(∇f),
    x0 = x0, # Perhaps we should add lvar and uvar as well here.
  )
  return R2NModel(
    B :: G,
    ∇f :: V,
    v :: V,
    σ :: T,
    meta,
    Counters()
  )
end
  
function NLPModels.obj(nlp::R2NModel, x::AbstractVector)
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_obj)
  mul!(nlp.v, nlp.B, x)
  return dot(nlp.v, x)/2 + dot(nlp.∇f, x) + nlp.σ * dot(x, x) / 2 
end
  
function NLPModels.grad!(nlp::R2NModel, x::AbstractVector, g::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nvar g
  increment!(nlp, :neval_grad)
  mul!(g, nlp.B, x)
  g .+= nlp.∇f
  g .+= nlp.σ * x
  return  g
end

function NLPModels.push!(nlp::R2NModel, s::AbstractVector, y::AbstractVector)
  push!(nlp.B, s, y)
end