export R2NModel

@doc raw"""
    R2NModel(B, ∇f, v, σ, x0)

Given the unconstrained optimization problem:
```math
\min f(x),
```
this model represents the smooth R2N subproblem:
```math
\min_s \ ∇f^T s + \tfrac{1}{2} s^T B s + \tfrac{1}{2} σ \|s\|^2
```
where `B` is either an approximation of the Hessian of `f` or the Hessian itself and `∇f` represents the gradient of `f` at `x0`.
`σ > 0` is a regularization parameter and `v` is a vector of the same size as `x0` used for intermediary computations.
"""
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
  σ :: T,
  x0 :: V
) where{T, V, G}
  @assert length(x0) == length(∇f)
  meta = NLPModelMeta(
    length(∇f),
    x0 = x0, # Perhaps we should add lvar and uvar as well here.
  )
  v = similar(x0)
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
  g .+= nlp.σ .* x
  return  g
end