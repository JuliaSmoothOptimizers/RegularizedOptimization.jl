abstract type AbstractShiftedProximableNLPModel{T, V} <: AbstractRegularizedNLPModel{T, V} end

struct ShiftedProximableQuadraticNLPModel{T, V, M <: AbstractNLPModel{T, V}, H <: ShiftedProximableFunction, I, P <: AbstractRegularizedNLPModel{T, V}} <:
       AbstractShiftedProximableNLPModel{T, V}
  model::M
  h::H
  selected::I
  parent::P
end

function ShiftedProximableQuadraticNLPModel(
  reg_nlp::AbstractRegularizedNLPModel{T, V}, 
  x::V;
  l_bound_m_x::VN = nothing,
  u_bound_m_x::VN = nothing,
  ∇f::VNG = nothing,
) where {T, V, VN <: Union{V, Nothing}, VNG <: Union{V, Nothing}}
  nlp, h, selected = reg_nlp.model, reg_nlp.h, reg_nlp.selected

  @assert !(has_bounds(nlp) && isnothing(l_bound_m_x) && isnothing(u_bound_m_x)) 
    "RegularizedOptimization: bounds are required for the quadratic subproblem when the NLP has bounds."

  # FIXME: `shifted` call ignores the `selected` argument when there are no bounds!
  ψ = has_bounds(nlp) ? shifted(h, x, l_bound_m_x, u_bound_m_x, selected) : shifted(h, x)

  B = hess_op(reg_nlp, x)
  isnothing(∇f) && (∇f = grad(nlp, x))
  φ = QuadraticModel(∇f, B, x0 = x, regularize = true)

  ShiftedProximableQuadraticNLPModel(φ, ψ, selected, reg_nlp)
end

function shift!(
  reg_nlp::ShiftedProximableQuadraticNLPModel{T, V},
  x::V;
  compute_grad::Bool = true
) where{T, V}
  nlp, h = reg_nlp.parent.model, reg_nlp.parent.h
  φ, ψ = reg_nlp.model, reg_nlp.h

  if has_bounds(nlp)
    update_bounds!(ψ.l, ψ.u, nlp.meta.lvar, nlp.meta.uvar, x)
  end
  shift!(ψ, x)

  g = φ.data.c
  compute_grad && grad!(nlp, x, g)

  # The hessian is implicitely updated since it was defined as hess_op(nlp, x)
end