using TRNC,Printf, Convex,SCS, Random, LinearAlgebra, IterativeSolvers, Roots


function hardproxtestl1B2(n)

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
q = A'*(A*g) - randn(n) #doesn't really matter tho in the example

Doptions=s_options(1/ν; maxIter=10, λ=λ,
    ∇fk = q, Bk = A'*A, xk=x, Δ = τ)

fval(s, bq, xi, νi) = (s.+bq).^2/(2*νi) + λ*abs.(s.+xi)
projbox(y, bq, νi) = min.(max.(y, -bq.-λ*νi),-bq.+λ*νi) # different since through dual
(s,s⁻,f,funEvals) = hardproxl1B2(fval, x, projbox, Doptions);


s_cvx = Variable(n)
problem = minimize(sumsquares(s_cvx+q)/(2*ν) + λ*norm(s_cvx+x,1), norm(s_cvx, 2)<=τ);
solve!(problem, SCS.Optimizer)



@printf("Us: %1.4e    CVX: %1.4e    s: %1.4e   s_cvx: %1.4e    normdiff: %1.4e\n", sum(f), norm(s_cvx.value.-q)^2/(2*ν) + λ*norm(s_cvx.value.+x,1), norm(s)^2, norm(s_cvx.value)^2, norm(s_cvx.value .- s));



end
