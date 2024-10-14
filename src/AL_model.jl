# based on Percival.jl
# https://github.com/JuliaSmoothOptimizers/Percival.jl/blob/main/src/AugLagModel.jl

export AugLagModel
export update_cx!, update_y!, project_y!, update_μ!

using NLPModels, NLPModelsModifiers, LinearAlgebra, LinearOperators
using NLPModels: increment!, @lencheck

@doc raw"""
    AugLagModel(model, y, μ, x, fx, cx)

Given a model
```math
\min \ f(x) \quad s.t. \quad c(x) = 0, \quad l ≤ x ≤ u,
```
this new model represents the subproblem of the augmented Lagrangian method
```math
\min \ f(x) - yᵀc(x) + \tfrac{1}{2} μ \|c(x)\|^2 \quad s.t. \quad l ≤ x ≤ u,
```
where y is an estimates of the Lagrange multiplier vector and μ is the penalty parameter.

In addition to keeping `meta` and `counters` as any NLPModel, an AugLagModel also stores
- `model`: The internal model defining ``f``, ``c`` and the bounds,
- `y`: The multipliers estimate,
- `μ`: The penalty parameter,
- `x`: Reference to the last point at which the function `c(x)` was computed,
- `fx`: Reference to `f(x)`,
- `cx`: Reference to `c(x)`,
- `μc_y`: storage for y - μ * cx,
- `Jtv`: storage for jtprod(nlp, x, v).

Use the functions `update_cx!`, `update_y!` and `update_μ!` to update these values.
"""
mutable struct AugLagModel{M <: AbstractNLPModel, T <: AbstractFloat, V <: AbstractVector} <:
               AbstractNLPModel{T, V}
  meta::NLPModelMeta{T, V}
  counters::Counters
  model::M
  y::V
  μ::T
  x::V
  fx::T
  cx::V
  μc_y::V # y - μ * cx
  Jtv::Vector{T}
end

function AugLagModel(model::AbstractNLPModel{T, V}, y::V, μ::T, x::V, fx::T, cx::V) where {T, V}
  nvar, ncon = model.meta.nvar, model.meta.ncon
  @assert length(x) == nvar
  @assert length(y) == ncon
  @assert length(cx) == ncon
  μ >= 0 || error("Penalty parameter μ should be nonnegative")

  meta = NLPModelMeta(
    nvar,
    x0 = model.meta.x0,
    lvar = model.meta.lvar,
    uvar = model.meta.uvar,
    name = "AugLagModel-$(model.meta.name)",
  )

  return AugLagModel(
    meta,
    Counters(),
    model,
    y,
    μ,
    x,
    fx,
    cx,
    isassigned(y) && isassigned(cx) ? y - μ * cx : similar(y),
    zeros(T, nvar),
  )
end

"""
    update_cx!(nlp, x)

Given an `AugLagModel`, if `x != nlp.x`, then updates the internal value `nlp.cx` calling `cons`
on `nlp.model`, and reset `nlp.fx` to a NaN. Also updates `nlp.μc_y`.
"""
function update_cx!(nlp::AugLagModel, x::AbstractVector{T}) where {T}
  @assert length(x) == nlp.meta.nvar
  if x != nlp.x
    cons!(nlp.model, x, nlp.cx)
    nlp.cx .-= nlp.model.meta.lcon
    nlp.x .= x
    refresh_μc_y!(nlp)
    nlp.fx = T(NaN)
  end
end

"""
    update_fxcx!(nlp, x)

Given an `AugLagModel`, if `x != nlp.x`, then updates the internal value `nlp.cx` calling `objcons`
on `nlp.model`. Also updates `nlp.μc_y`. Returns fx only.
"""
function update_fxcx!(nlp::AugLagModel, x::AbstractVector)
  @assert length(x) == nlp.meta.nvar
  if x != nlp.x
    nlp.fx, _ = objcons!(nlp.model, x, nlp.cx)
    nlp.cx .-= nlp.model.meta.lcon
    nlp.x .= x
    refresh_μc_y!(nlp)
  elseif isnan(nlp.fx)
    nlp.fx = obj(nlp.model, x)
  end
end

function refresh_μc_y!(nlp::AugLagModel)
  nlp.μc_y .= nlp.μ .* nlp.cx .- nlp.y
end

"""
    update_y!(nlp)

Given an `AugLagModel`, update `nlp.y = -nlp.μc_y` and updates `nlp.μc_y` accordingly.
"""
function update_y!(nlp::AugLagModel)
  nlp.y .= .-nlp.μc_y
  refresh_μc_y!(nlp)
end

"""
    project_y!(nlp, ymin, ymax)

Given an `AugLagModel`, project `nlp.y` into [ymin, ymax]and updates `nlp.μc_y` accordingly.
"""
function project_y!(nlp::AugLagModel, ymin::AbstractVector{T}, ymax::AbstractVector{T}) where {T <: Real}
  nlp.y .= max.(ymin, min.(nlp.y, ymax))
  refresh_μc_y!(nlp)
end

function project_y!(nlp::AugLagModel, ymin::T, ymax::T) where {T <: Real}
  nlp.y .= max.(ymin, min.(nlp.y, ymax))
  refresh_μc_y!(nlp)
end

"""
    update_μ!(nlp, μ)

Given an `AugLagModel`, updates `nlp.μ = μ` and `nlp.μc_y` accordingly.
"""
function update_μ!(nlp::AugLagModel, μ::AbstractFloat)
  nlp.μ = μ
  refresh_μc_y!(nlp)
end

function NLPModels.obj(nlp::AugLagModel, x::AbstractVector)
  @assert length(x) == nlp.meta.nvar
  increment!(nlp, :neval_obj)
  update_fxcx!(nlp, x)
  return nlp.fx - dot(nlp.y, nlp.cx) + (nlp.μ / 2) * dot(nlp.cx, nlp.cx)
end

function NLPModels.grad!(nlp::AugLagModel, x::AbstractVector, g::AbstractVector)
  @assert length(x) == nlp.meta.nvar
  @assert length(g) == nlp.meta.nvar
  increment!(nlp, :neval_grad)
  update_cx!(nlp, x)
  grad!(nlp.model, x, g)
  g .+= jtprod!(nlp.model, x, nlp.μc_y, nlp.Jtv)
  return g
end

function NLPModels.objgrad!(nlp::AugLagModel, x::AbstractVector, g::AbstractVector)
  @assert length(x) == nlp.meta.nvar
  @assert length(g) == nlp.meta.nvar
  increment!(nlp, :neval_obj)
  increment!(nlp, :neval_grad)
  update_fxcx!(nlp, x)
  f = nlp.fx - dot(nlp.y, nlp.cx) + (nlp.μ / 2) * dot(nlp.cx, nlp.cx)
  grad!(nlp.model, x, g)
  g .+= jtprod!(nlp.model, x, nlp.μc_y, nlp.Jtv)
  return f, g
end