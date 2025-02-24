export RegularizedExecutionStats

import SolverCore.GenericExecutionStats

# use Arpack to obtain largest eigenvalue in magnitude with a minimum of robustness
function LinearAlgebra.opnorm(B; kwargs...)
  try
    _, s, _ = tsvd(B) # TODO: This allocates; see utils.jl
    return s[1]
  catch LAPACKException # FIXME: This should be removed ASAP; see PR #159.
    return LinearAlgebra.opnorm(Matrix(B))
  end
end

ShiftedProximalOperators.iprox!(
  y::AbstractVector,
  ψ::ShiftedProximableFunction,
  g::AbstractVector,
  D::AbstractDiagonalQuasiNewtonOperator,
) = iprox!(y, ψ, g, D.d)

ShiftedProximalOperators.iprox!(
  y::AbstractVector,
  ψ::ShiftedProximableFunction,
  g::AbstractVector,
  D::SpectralGradient,
) = iprox!(y, ψ, g, fill!(similar(g), D.d[1]))

LinearAlgebra.diag(op::AbstractDiagonalQuasiNewtonOperator) = copy(op.d)
LinearAlgebra.diag(op::SpectralGradient{T}) where {T} = zeros(T, op.nrow) .* op.d[1]

"""
    GenericExecutionStats(reg_nlp :: AbstractRegularizedNLPModel{T, V})

Construct a GenericExecutionStats object from an AbstractRegularizedNLPModel. 
More specifically, construct a GenericExecutionStats on the NLPModel of reg_nlp and add three solver_specific entries namely :smooth_obj, :nonsmooth_obj and :xi.
This is useful for reducing the number of allocations when calling solve!(..., reg_nlp, stats) and should be used by default.
Warning: This should *not* be used when adding other solver_specific entries that do not have the current scalar type. 
"""
function RegularizedExecutionStats(reg_nlp::AbstractRegularizedNLPModel{T, V}) where {T, V}
  stats = GenericExecutionStats(reg_nlp.model, solver_specific = Dict{Symbol, T}())
  set_solver_specific!(stats, :smooth_obj, T(Inf))
  set_solver_specific!(stats, :nonsmooth_obj, T(Inf))
  return stats
end
