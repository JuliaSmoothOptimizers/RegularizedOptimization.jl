
# getting prox for dual given prox for primal
function prox_dual(z, σ, prox_primal)
    t = z - σ*prox_primal((1/σ)*z, 1/σ)
    return t
end


# getting projection for a function given its prox
function proj_prox(y, τ, prox, a, b)
  f(λ) = norm(prox(y,λ)) - τ
  λ_opt = fzero(f, [a, b])
  return prox(y, λ_opt)
end


# l1 prox
function prox_l1(y::Vector{Float64}, γ::Float64)
  return max(abs(y)-γ,0).*sign(y)
end

# projection onto l1
function proj_l1!(y, τ)
     a = 0
     b = maximum(abs(y))
     f(λ) = norm(max(abs(y) - λ, 0).*sign(y)) - τ
     λ_opt = fzero(f, [a, b])
     copy!(y, max(abs(y) - λ_opt, 0).*sign(y))
 end

# prox of the quadratic
function prox_l2s(y::Vector{Float64}, γ::Float64)
  return (1.0/(1.0+γ)) * y
end

# prox of the 2-norm
function prox_l2(y::Vector{Float64}, γ::Float64)
  return max(1.0 - γ/norm(y), 0) * y
end

# projection onto lower/upper bounds
function proj_bounds(y::Vector{Float64}, l::Vector{Float64}, u::Vector{Float64}, γ::Float64)
  return max.(min.(y, u), l)
end

# projection to SO₃
function proj2SO3!(Rt, γ::Float64)
    U, Σ, V = svd(Rt);
    BLAS.gemm!('N','T',1.0,U,V,0.0,Rt);
end

# projection onto capped simplex
function projection_capped(W0, lb, ub, h, γ)
  if h == length(W0)
    w = ones(length(W0))
    return w
  end
  if γ < Inf
    a = -1.5+minimum(W0)
    b = maximum(W0)
    f(λ) = sum(max(min(W0 - λ, ub), lb)) - h
    λ_opt = fzero(f, [a, b])
    w = max(min(W0 - λ_opt, ub), lb)
  else
    # set weights
    p = sortperm(W0)
    w = zeros(size(W0))
    w[p[1:h]]=1
  end
    return w
end
