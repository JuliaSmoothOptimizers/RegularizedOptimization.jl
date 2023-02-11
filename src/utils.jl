# use Arpack to obtain largest eigenvalue in magnitude with a minimum of robustness
function LinearAlgebra.opnorm(B; kwargs...)
  _, s, _ = tsvd(B)
  return s[1]
end

ShiftedProximalOperators.iprox!(
  y::AbstractVector,
  ψ::ShiftedProximableFunction,
  g::AbstractVector,
  D::DiagonalQN,
) = iprox!(y, ψ, g, D.d)

ShiftedProximalOperators.iprox!(
  y::AbstractVector,
  ψ::ShiftedProximableFunction,
  g::AbstractVector,
  D::SpectralGradient,
) = iprox!(y, ψ, g, fill!(similar(g), D.d[1]))
