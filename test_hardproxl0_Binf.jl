using TRNC,Printf, Convex,SCS, Random, LinearAlgebra, IterativeSolvers, Roots


function hardproxtestl0Binf(n)

    A,_ = qr(5*randn(n,n))

    B = Array(A)'
    
    A = Array(B)
    # rng(2)
    # vectors
    x = 10*randn(n)
    x0 = zeros(n)
    k = 4
    p   = randperm(n)[1:k]
    x0 = zeros(n,)
    x0[p[1:k]]=sign.(randn(k))
    b0 = A*x0
    b = b0 + 0.005*randn(n,)
    # scalars
    ν = 1/norm(A'*A)^2
    λ = 10*rand()
    τ = 3*rand()
    
    # This constructs q = ν∇qᵢ(sⱼ) = Bksⱼ + gᵢ (note that i = k in paper notation)
    function f_obj(x) #gradient and hessian info are smooth parts, m also includes nonsmooth part
        r = A*x - b
        g = A'*r
        return norm(r)^2/2, g, A'*A
    end
    function h_obj(x)
        return λ*norm(x,0)
    end
    
    (qk, ∇qk, H) = f_obj(x)
    Hess(d) = H*d
    
    Doptions=s_options(1/ν; maxIter=100, λ=λ,
        ∇fk = ∇qk, Bk = A'*A, xk=x, Δ = τ)
    
    objInner(d) = [0.5*(d'*Hess(d)) + ∇qk'*d + qk, Hess(d) + ∇qk]

    (s,s⁻,f,funEvals) = hardproxl0Binf(objInner, x, h_obj, Doptions);
    
    
    s_cvx = Variable(n)
    problem = minimize(sumsquares(A*(x+s_cvx) - b) + λ*norm(s_cvx+x,1), norm(s_cvx, Inf)<=τ);
    solve!(problem, SCS.Optimizer)
    
    @show s
    @show s_cvx.value
    
    @printf("Us: %1.4e    CVX: %1.4e    s: %1.4e   s_cvx: %1.4e    normdiff: %1.4e\n", f_obj(x+s)[1]+h_obj(x+s),f_obj(x+s_cvx.value)[1] + h_obj(s_cvx.value.+x), norm(s)^2, norm(s_cvx.value)^2, norm(s_cvx.value .- s));
    


end
