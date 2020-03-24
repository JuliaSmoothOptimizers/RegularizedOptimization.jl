using TRNC,Printf, Convex,SCS, Random, LinearAlgebra, IterativeSolvers, Roots


function hardproxtestBinf(n)


# n=10

x = 10*randn(n)
q= 5*randn(n)
ν = 20*rand()
λ = 10*rand()
τ = 3*rand()

(s,f) = hardproxBinf(q, x, ν,λ, τ)


s_cvx = Variable(n)
problem = minimize(sumsquares(s_cvx-q)/(2*ν) + λ*norm(s_cvx+x,1), norm(s_cvx, Inf)<=τ);
solve!(problem, SCSSolver())
# cvx_precision high
# cvx_begin
#     variable s_cvx(1)
#     minimize( (s_cvx-z)^2/(2*t) + lambda*abs(s_cvx+x))
#     subject to
#         -tau <= s_cvx <= tau
# cvx_end

return norm(s_cvx.value .- s)

end
