export LMModel

@doc raw"""
    LMModel(J, F, v, σ, x0)

Given the unconstrained optimization problem:
```math
\min \tfrac{1}{2} \| F(x) \|^2,
```
this model represents the smooth LM subproblem:
```math
\min_s \ \tfrac{1}{2} \| F(x) + J(x)s \|^2 + \tfrac{1}{2} σ \|s\|^2
```
where `J` is the Jacobian of `F` at `x0` in sparse format or as a linear operator.
`σ > 0` is a regularization parameter and `v` is a vector of the same size as `F(x0)` used for intermediary computations.
"""
mutable struct LMModel{T <: Real, V <: AbstractVector{T}, G <: Union{AbstractMatrix{T}, AbstractLinearOperator{T}}} <:
               AbstractNLPModel{T, V}
  J::G
  F::V
  v::V
  σ::T
  meta::NLPModelMeta{T, V}
  counters::Counters
end

function LMModel(J::G, F::V, σ::T, x0::V) where {T, V, G}
  @assert length(x0) == size(J, 2)
  @assert length(F) == size(J, 1)
  meta = NLPModelMeta(
    length(x0),
    x0 = x0, # Perhaps we should add lvar and uvar as well here.
  )
  v = similar(F)
  return LMModel(J::G, F::V, v::V, σ::T, meta, Counters())
end

function NLPModels.obj(nlp::LMModel, x::AbstractVector{T}) where{T}
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_obj)
  nlp.v .= nlp.F
  mul!(nlp.v, nlp.J, x, one(T), one(T))
  return ( dot(nlp.v, nlp.v) + nlp.σ * dot(x, x) ) / 2
end

function NLPModels.grad!(nlp::LMModel, x::AbstractVector{T}, g::AbstractVector{T}) where{T}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nvar g
  increment!(nlp, :neval_grad)
  nlp.v .= nlp.F
  @. g = nlp.σ .* x
  mul!(nlp.v, nlp.J, x, one(T), one(T))
  mul!(g, nlp.J', nlp.v, one(T), one(T))
  return g
end
