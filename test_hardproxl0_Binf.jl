using TRNC,Printf, Convex,SCS, Random, LinearAlgebra, IterativeSolvers, Roots


function hardproxtestl0Binf(n)


A,_ = qr(5*randn(n,n))

B = Array(A)'

A = Array(B)
# rng(2)
# vectors
k = Int(floor(.5*n))
p = randperm(n)
#initialize x
x = zeros(n,)
x[p[1:k]]=sign.(randn(k))

# scalars
ν = 1/norm(A'*A)^2
λ = 10*rand()
τ = .5

# This constructs q = ν∇qᵢ(sⱼ) = Bksⱼ + gᵢ (note that i = k in paper notation)
q = A'*(A*x - zeros(size(x))) #l0 it might tho - gradient of smooth function

Doptions=s_options(1/ν; maxIter=10, λ=λ,
    ∇fk = q, Bk = A'*A, xk=x, Δ = τ)

# fval(s, bq, xi, νi) = norm(s.+bq)^2/(2*νi) + λ*norm(s.+xi,0)
fval(s, bq, xi, νi) = (s.+bq).^2/(2*νi) + λ.*(.!iszero.(s.+xi,0))
projbox(y, bq, Δi) = min.(max.(y, bq.-Δi),bq.+Δi) # different since through dual
(s,s⁻,f,funEvals) = hardproxl0Binf(fval, zeros(size(x)), projbox, Doptions);


s_cvx = Variable(n)
problem = minimize(sumsquares(s_cvx+q)/(2*ν) + λ*norm(s_cvx+x,1), norm(s_cvx, Inf)<=τ);
solve!(problem, SCS.Optimizer)

@printf("Us: %1.4e    CVX: %1.4e    s: %1.4e   s_cvx: %1.4e    normdiff: %1.4e\n", sum(fval(s, q, x, ν)),sum(fval(s_cvx.value, q, x, ν)) , norm(s,1), norm(s_cvx.value,1), norm(s_cvx.value .- s));

@printf("Smooth:  Us=%1.4e vs  CVX=%1.4e    Nonsmooth: Us=%1.4e   CVX=%1.4e\n", norm(s.+q)^2,norm(s_cvx.value .+ q)^2 , norm(s+x,0), norm(s_cvx.value+x,0));



end
