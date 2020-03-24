# function hardproxtestBinf
using TRNC,Printf, Convex,SCS, Random, LinearAlgebra, IterativeSolvers, Roots


n=1

x = 10*randn(n)
q= 5*randn(n)
ν = 20*rand()
λ = 10*rand()
τ = 3*rand()

(s,f) = hardproxBinf(z, x, t, lambda, tau)


s_cvx = Variable(n)
problem = minimize((s_cvx-z)^2/(2*t) + lambda*abs(s_cvx+x), norm(s_cvx, Inf)<=tau)
# cvx_precision high
# cvx_begin
#     variable s_cvx(1)
#     minimize( (s_cvx-z)^2/(2*t) + lambda*abs(s_cvx+x))
#     subject to
#         -tau <= s_cvx <= tau
# cvx_end

norm(s_cvx - s)
