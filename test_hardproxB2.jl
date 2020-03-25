using TRNC,Printf, Convex,SCS, Random, LinearAlgebra, IterativeSolvers, Roots


function hardproxtestB2(n)

# rng(2)
# vectors
x = 10*randn(n);
q = 5*randn(n);

# scalars
ν = 20*rand();
λ = 10*rand();
τ = 3*rand();


(s,f) = hardproxB2(q, x, ν, λ, τ);


s_cvx = Variable(n)
problem = minimize(sumsquares(s_cvx-q)/(2*ν) + λ*norm(s_cvx+x,1), norm(s_cvx, 2)<=τ);
solve!(problem, SCSSolver())

# cvx_precision high
# cvx_begin quiet
#     variable s_cvx(n)
#     minimize( sum_square(s_cvx-z)/(2*t) + λ*norm(s_cvx+x,1))
#     subject to
#         norm(s_cvx,2) <=τ
# cvx_end






if n==1
    @printf("Us: %1.4e    CVX: %1.4e    s: %1.4e   s_cvx: %1.4e    normdiff: %1.4e\n", f, norm(s_cvx.value-q)^2/(2*ν) + λ*norm(s_cvx.value+x,1), s, s_cvx.value, norm(s_cvx.value - s));
else
    @printf("Us: %1.4e    CVX: %1.4e    s: %1.4e   s_cvx: %1.4e    normdiff: %1.4e\n", f, norm(s_cvx.value-q)^2/(2*ν) + λ*norm(s_cvx.value+x,1), norm(s)^2, norm(s_cvx.value)^2, norm(s_cvx.value - s));
end


end
