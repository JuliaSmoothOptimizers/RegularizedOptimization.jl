abstract type AbstractShiftedProximableNLPModel{T, V} <: AbstractRegularizedNLPModel{T, V} end

"""
    subproblem = ShiftedProximableQuadraticNLPModel(reg_nlp, x; kwargs...)

Given a regularized NLP model `reg_nlp` representing the problem

    minimize f(x) + h(x),
  
construct a shifted quadratic model around `x`:

    minimize  φ(s; x) + ½ σ ‖s‖² + ψ(s; x),

where φ(s ; x) = f(x) + ∇f(x)ᵀs + ½ sᵀBs is a quadratic approximation of f about x,
ψ(s; x) is either h(x + s) or an approximation of h(x + s), ‖⋅‖ is the ℓ₂ norm and σ > 0 is the regularization parameter.

The ShiftedProximableQuadraticNLPModel is made of the following components:

- `model <: AbstractNLPModel`: represents φ, the quadratic approximation of the smooth part of the objective function;
- `h <: ShiftedProximableFunction`: represents ψ, the shifted version of the nonsmooth part of the model;
- `selected`: the subset of variables to which the regularizer h should be applied (default: all).
- `parent`: the original regularized NLP model from which the subproblem was derived.

# Arguments
- `reg_nlp::AbstractRegularizedNLPModel{T, V}`: the regularized NLP model for which the subproblem is being constructed.
- `x::V`: the point around which the quadratic model is constructed.

# Keyword Arguments
- `l_bound_m_x::VN = nothing`: the vector of lower bounds minus `x` (i.e., l - x), required if the original NLP model has bounds.
- `u_bound_m_x::VN = nothing`: the vector of upper bounds minus `x` (i.e., u - x), required if the original NLP model has bounds.
- `∇f::VNG = nothing`: the gradient of the smooth part of the objective function at `x`. If not provided, it will be computed.

The matrix B is constructed as a `LinearOperator` and is the returned value of `hess_op(reg_nlp, x)` (see https://jso.dev/NLPModels.jl/stable/reference/#NLPModels.hess_op`).
φ is constructed as a `QuadraticModel`, (see https://github.com/JuliaSmoothOptimizers/QuadraticModels.jl).
"""
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

"""
    shift!(reg_nlp::ShiftedProximableQuadraticNLPModel, x; compute_grad = true)

Update the shifted quadratic model `reg_nlp` at the point `x`. 
i.e. given the shifted quadratic model around `y`:

    minimize  φ(s; y) + ½ σ ‖s‖² + ψ(s; y),

update it to be around `x`:

    minimize  φ(s; x) + ½ σ ‖s‖² + ψ(s; x).

# Arguments
- `reg_nlp::ShiftedProximableQuadraticNLPModel`: the shifted quadratic model to be updated.
- `x::V`: the point around which the shifted quadratic model should be updated.

# Keyword Arguments
- `compute_grad::Bool = true`: whether the gradient of the smooth part of the model should be updated.
"""
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

  # The hessian is implicitly updated since it was defined as hess_op(nlp, x)
end