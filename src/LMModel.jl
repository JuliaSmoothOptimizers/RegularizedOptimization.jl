export LMModel

@doc raw"""
    LMModel(j_prod!, jt_prod, F, v, σ, xk)

Given the unconstrained optimization problem:
```math
\min \tfrac{1}{2} \| F(x) \|^2,
```
this model represents the smooth LM subproblem:
```math
\min_s \ \tfrac{1}{2} \| F(x) + J(x)s \|^2 + \tfrac{1}{2} σ \|s\|^2
```
where `J` is the Jacobian of `F` at `xk`, represented via matrix-free operations.
`j_prod!(xk, s, out)` computes `J(xk) * s`, and `jt_prod!(xk, r, out)` computes `J(xk)' * r`.

`σ > 0` is a regularization parameter and `v` is a vector of the same size as `F(xk)` used for intermediary computations.
"""
mutable struct LMModel{
  T <: Real,
  V <: AbstractVector{T},
  Jac <: Union{AbstractMatrix, AbstractLinearOperator},
} <: AbstractNLPModel{T, V}
  J::Jac
  F::V
  v::V
  xk::V
  σ::T
  meta::NLPModelMeta{T, V}
  counters::Counters
end

function LMModel(J::Jac, F::V, σ::T, xk::V) where {T, V, Jac}
  meta = NLPModelMeta(
    length(xk),
    x0 = xk, # Perhaps we should add lvar and uvar as well here.
  )
  v = similar(F)
  return LMModel(J, F, v, xk, σ, meta, Counters())
end

function NLPModels.obj(nlp::LMModel, x::AbstractVector{T}) where {T}
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_obj)
  mul!(nlp.v, nlp.J, x)
  nlp.v .+= nlp.F
  return (dot(nlp.v, nlp.v) + nlp.σ * dot(x, x)) / 2
end

function NLPModels.grad!(nlp::LMModel, x::AbstractVector{T}, g::AbstractVector{T}) where {T}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nvar g
  increment!(nlp, :neval_grad)
  mul!(nlp.v, nlp.J, x)
  nlp.v .+= nlp.F
  mul!(g, nlp.J', nlp.v)
  @. g += nlp.σ .* x
  return g
end
