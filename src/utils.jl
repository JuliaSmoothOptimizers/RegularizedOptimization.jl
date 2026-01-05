export RegularizedExecutionStats

import SolverCore.GenericExecutionStats

function power_method!(B::M, v₀::S, v₁::S, max_iter::Int = 1) where {M, S}
  @assert max_iter >= 1 "max_iter must be at least 1."
  mul!(v₁, B, v₀)
  normalize!(v₁) # v1 = B*v0 / ‖B*v0‖
  for i = 2:max_iter
    v₀ .= v₁ # v0 = v1
    mul!(v₁, B, v₀)
    normalize!(v₁)
  end
  mul!(v₁, B, v₀)
  return abs(dot(v₀, v₁))
end

# Compute upper bounds for μ‖B‖₂, where μ ∈ (0, 1].

# For matrices, we compute the Frobenius norm.
function opnorm_upper_bound(B::AbstractMatrix) 
  return norm(B, 2)
end

# For LBFGS, using the formula Bₖ = B\_{k-1} - aₖaₖᵀ + bₖbₖᵀ, we compute
# ‖Bₖ‖₂ ≤ ‖B₀‖₂ + ∑ᵢ ‖bᵢ‖₂²  
function opnorm_upper_bound(B::LBFGSOperator{T}) where{T} 
  data = B.data
  approx = data.scaling ? 1/data.scaling_factor : T(1)
  approx += norm(data.b, 2)^2
  return approx
end

# For LSR1, we use the formula Bₖ = B\_{k-1} + σₖaₖaₖᵀ, we compute
# ‖Bₖ‖₂ ≤ ‖B₀‖₂ + ∑ᵢ |σᵢ|‖aᵢ‖₂² 
function opnorm_upper_bound(B::LSR1Operator{T}) where{T}
  data = B.data
  approx = data.scaling ? 1/data.scaling_factor : T(1)
  @inbounds for i = 1:data.mem
    if data.as[i] != 0
      approx += norm(data.a[i])^2/abs(data.as[i])
    end
  end
  return approx
end

# For diagonal operators, we compute the exact operator norm
function opnorm_upper_bound(B::AbstractDiagonalQuasiNewtonOperator)
  return norm(B.d, Inf)
end

# In the general case, we either use the power_method or Arpack, 
# Note: Arpack allocates and the power method might be unreliable.
function opnorm_upper_bound(B::AbstractLinearOperator; v₀ = nothing, v₁ = nothing, max_iter = -1)
  # Fallback to either the power_method or arpack
  if max_iter ≥ 1 
    @assert !(isnothing(v₀) && isnothing(v₁))
    return power_method!(B, v₀, v₁, max_iter = max_iter)
  else
    return opnorm(B)
  end
end

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
  n = size(B, 1)
  nev = 1
  ncv = max(20, 2 * nev + 1)

  while !(have_eig || attempt >= max_attempts)
    attempt += 1
    try
      # Estimate largest eigenvalue in absolute value
      d, nconv, niter, nmult, resid =
        eigs(B; nev = nev, ncv = ncv, which = :LM, ritzvec = false, check = 1)

      # Check if eigenvalue has converged
      have_eig = nconv == 1
      if have_eig
        λ = abs(d[1])  # Take absolute value of the largest eigenvalue
        break  # Exit loop if successful
      else
        # Increase NCV for the next attempt if convergence wasn't achieved
        ncv = min(2 * ncv, n)
      end
    catch e
      if occursin("XYAUPD_Exception", string(e)) && ncv < n
        @warn "Arpack error: $e. Increasing NCV to $ncv and retrying."
        ncv = min(2 * ncv, n)  # Increase NCV but don't exceed matrix size
      else
        rethrow(e)  # Re-raise if it's a different error
      end
    end
  end

  return λ, have_eig
end

function opnorm_svd(J; max_attempts::Int = 3)
  have_svd = false
  attempt = 0
  σ = zero(eltype(J))
  n = min(size(J)...)  # Minimum dimension of the matrix
  nsv = 1
  ncv = 10

  while !(have_svd || attempt >= max_attempts)
    attempt += 1
    try
      # Estimate largest singular value
      s, nconv, niter, nmult, resid = svds(J; nsv = nsv, ncv = ncv, ritzvec = false, check = 1)

      # Check if singular value has converged
      have_svd = nconv >= 1
      if have_svd
        σ = maximum(s.S)  # Take the largest singular value
        break  # Exit loop if successful
      else
        # Increase NCV for the next attempt if convergence wasn't achieved
        ncv = min(2 * ncv, n)
      end
    catch e
      if occursin("XYAUPD_Exception", string(e)) && ncv < n
        @warn "Arpack error: $e. Increasing NCV to $ncv and retrying."
        ncv = min(2 * ncv, n)  # Increase NCV but don't exceed matrix size
      else
        rethrow(e)  # Re-raise if it's a different error
      end
    end
  end

  return σ, have_svd
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
  set_solver_specific!(stats, :sigma, T(Inf))
  set_solver_specific!(stats, :sigma_cauchy, T(Inf))
  set_solver_specific!(stats, :radius, T(Inf))
  set_solver_specific!(stats, :prox_evals, T(Inf))
  return stats
end

function get_status(
  reg_nlp::M;
  elapsed_time = 0.0,
  iter = 0,
  optimal = false,
  improper = false,
  max_eval = Inf,
  max_time = Inf,
  max_iter = Inf,
) where {M <: AbstractRegularizedNLPModel}
  if optimal
    :first_order
  elseif improper
    :improper
  elseif iter >= max_iter
    :max_iter
  elseif elapsed_time >= max_time
    :max_time
  elseif neval_obj(reg_nlp.model) >= max_eval && max_eval >= 0
    :max_eval
  else
    :unknown
  end
end

function update_bounds!(
  l_bound_m_x::V,
  u_bound_m_x::V,
  l_bound::V,
  u_bound::V,
  xk::V,
) where {V <: AbstractVector}
  @. l_bound_m_x = l_bound - xk
  @. u_bound_m_x = u_bound - xk
end

function update_bounds!(
  l_bound_m_x::V,
  u_bound_m_x::V,
  is_subsolver::Bool,
  l_bound::V,
  u_bound::V,
  xk::V,
  Δ::T,
) where {T <: Real, V <: AbstractVector{T}}
  if is_subsolver
    @. l_bound_m_x = max(xk - Δ, l_bound)
    @. u_bound_m_x = min(xk + Δ, u_bound)
  else
    @. l_bound_m_x = max(-Δ, l_bound - xk)
    @. u_bound_m_x = min(Δ, u_bound - xk)
  end
end
