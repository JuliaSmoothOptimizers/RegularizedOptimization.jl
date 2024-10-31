export RegularizedExecutionStats

import SolverCore.GenericExecutionStats

# use Arpack to obtain largest eigenvalue in magnitude with a minimum of robustness
function LinearAlgebra.opnorm(B; kwargs...)
  m, n = size(B)
  opnorm_fcn = m == n ? opnorm_eig : opnorm_svd
  return opnorm_fcn(B; kwargs...)
end
function opnorm_eig(B; max_attempts::Int = 3)
  have_eig = false
  attempt = 0
  λ = zero(eltype(B))
  success = false
  n = size(B, 1)
  nev = 1
  ncv = max(20, 2 * nev + 1)
  while !(have_eig || attempt > max_attempts)
    attempt += 1
    (d, nconv, niter, nmult, resid) = eigs(B; nev = nev, ncv = ncv, which = :LM, ritzvec = false, check = 1)
    have_eig = nconv == 1
    if (have_eig)
      λ = abs(d[1])
      success = true
    else
      ncv = min(2 * ncv, n)
    end
  end
  return λ, success
end

function opnorm_svd(J; max_attempts::Int = 3)
  have_svd = false
  attempt = 0
  σ = zero(eltype(J))
  success = false
  n = min(size(J)...)
  nsv = 1
  ncv = 10
  while !(have_svd || attempt > max_attempts)
    attempt += 1
    (s, nconv, niter, nmult, resid) = svds(J, nsv = nsv, ncv = ncv, ritzvec = false, check = 1)
    have_svd = nconv == 1
    if (have_svd)
      σ = maximum(s.S)
      success = true
    else
      ncv = min(2 * ncv, n)
    end
  end
  return σ, success
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
