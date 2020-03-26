using TRNC,Printf, Convex,SCS, Random, LinearAlgebra, IterativeSolvers, Roots


function hardproxtestBinf(n)

A,_ = qr(5*randn(n,n))
B = Array(A)'
A = Array(B)
# rng(2)
# vectors
x = 10*randn(n)
g = 5*randn(n)

# scalars
ν = 1/norm(A'*A)^2
λ = 10*rand()
τ = 3*rand()

# This constructs q = ν∇qᵢ(sⱼ) = Bksⱼ + gᵢ (note that i = k in paper notation)
#but it's first order tho so sⱼ = 0 and it's just ∇f(x_k)
q = g #doesn't really matter tho in the example

fval(y) = (y-(x+q)).^2/(2*ν)+λ*abs.(y)
projbox(w) = min.(max.(w,x.-τ), x.+τ)

Doptions=s_options(1/ν; maxIter=10, λ=λ, gk = g, Bk = A'*A, xk=x, Δ = τ)
# n=10

# (s,f) = hardproxBinf(q, x, ν,λ, τ)
(s, f) = hardproxBinf(fval, x, projbox, Doptions)


s_cvx = Variable(n)
problem = minimize(sumsquares(s_cvx-q)/(2*ν) + λ*norm(s_cvx+x,1), norm(s_cvx, Inf)<=τ);
solve!(problem, SCSSolver())
# cvx_precision high
# cvx_begin
#     variable s_cvx(1)
#     minimize( (s_cvx-z)^2/(2*t) + lambda*abs(s_cvx+x))
#     subject to
#         -tau <= s_cvx <= tau
# cvx_endf

return norm(s_cvx.value .- s)

end
