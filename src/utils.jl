# use Arpack to obtain largest eigenvalue in magnitude with a minimum of robustness
function LinearAlgebra.opnorm(B; kwargs...)
  hermitian =
    (typeof(B) <: AbstractQuasiNewtonOperator && ishermitian(B)) ||
    (typeof(B) <: Symmetric && eltype(B) <: Real) ||
    (typeof(B) <: Hermitian && eltype(B) <: Complex)
  opnorm_fcn = hermitian ? opnorm_eig : opnorm_svd
  nrm, success = opnorm_fcn(B; kwargs...)
  if !success
    nrm, _ = normest(B)
  end
  return nrm
end

function opnorm_eig(B; max_attempts::Int = 10)
  have_eig = false
  attempt = 0
  λ = zero(eltype(B))
  success = false
  n = size(B, 1)
  nev = 1
  ncv = max(20, 2 * nev + 1)
  while !(have_eig || attempt > max_attempts)
    attempt += 1
    (d, nconv, niter, nmult, resid) =
      eigs(B; nev = nev, ncv = ncv, which = :LM, ritzvec = false, check = 1)
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

function opnorm_svd(J; max_attempts::Int = 10)
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
