using TRNC,Printf, Convex,SCS, Random, LinearAlgebra, IterativeSolvers, Roots


function hardproxtestB0Binf(n)

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
    δ = k
    τ = 1

# This constructs q = ν∇qᵢ(sⱼ) = Bksⱼ + gᵢ (note that i = k in paper notation)
#but it's first order tho so sⱼ = 0 and it's just ∇f(x_k)
q = A'*(A*x - zeros(size(x))) #doesn't really matter tho in the example

Doptions=s_options(1/ν; maxIter=10, λ=δ,
    ∇fk = q, Bk = A'*A, xk=x, Δ = τ)

fval(sb, bq, bx, νi) = (sb.+bq).^2/(2*νi)
projbox(w, bx, τi) = min.(max.(w,bx.-τi), bx.+τi)


(s,s⁻, f, funEvals) = hardproxB0Binf(fval, zeros(size(x)), projbox, Doptions)


s_cvx = Variable(n)
problem = minimize(sumsquares(s_cvx+q)/(2*ν), [norm(s_cvx, Inf)<=τ, norm(s_cvx, 1)<=δ]);
solve!(problem, SCS.Optimizer)


@printf("Us: %1.4e    CVX: %1.4e    s: %1.4e   s_cvx: %1.4e    normdiff: %1.4e\n", sum(fval(s, q, x, ν)),sum(fval(s_cvx.value, q, x, ν)) , norm(s,0), norm(s_cvx.value,0), norm(s_cvx.value .- s));

@printf("Smooth:  Us=%1.4e vs  CVX=%1.4e    Nonsmooth: Us=%1.4e   CVX=%1.4e\n", norm(s.+q)^2,norm(s_cvx.value .+ q)^2 , norm(s+x,0) - δ, norm(s_cvx.value+x,0) - δ);


end
