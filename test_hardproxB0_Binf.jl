using TRNC,Printf, Convex,SCS, Random, LinearAlgebra, Roots


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
    b = A*x + 0.001*randn(size(x))
    # scalars
    ν = 1/norm(A'*A)^2
    λ = 10*rand()
    δ = k
    τ = 1

# This constructs q = ν∇qᵢ(sⱼ) = Bksⱼ + gᵢ (note that i = k in paper notation)
#but it's first order tho so sⱼ = 0 and it's just ∇f(x_k)

function f_obj(x) #gradient and hessian info are smooth parts, m also includes nonsmooth part
    r = A*x - b
    g = A'*r
    return norm(r)^2/2, g, A'*A
end
function h_obj(x)
    if norm(x,0) ≤ δ
        h = 0
    else
        h = Inf 
    end
    return λ*h 
end


(qk, ∇qk, H) = f_obj(x)
Hess(d) = H*d


Doptions=s_options(1/ν; maxIter=100, λ=δ,
    ∇fk = ∇qk, Bk = A'*A, xk=x, Δ = τ)

objInner(d) = [0.5*(d'*Hess(d)) + ∇qk'*d + qk, Hess(d) + ∇qk]


(s,s⁻, f, funEvals) = hardproxB0Binf(objInner, zeros(size(x)), h_obj, Doptions)


s_cvx = Variable(n)
problem = minimize(sumsquares(s_cvx+q)/(2*ν), [norm(s_cvx, Inf)<=τ, norm(s_cvx+x, 1)<=δ]);
solve!(problem, SCS.Optimizer)


@printf("Us: %1.4e    CVX: %1.4e    s: %1.4e   s_cvx: %1.4e    normdiff: %1.4e\n",
sum(fval(s, q, x, ν)),sum(fval(s_cvx.value, q, x, ν)) , norm(s,1), norm(s_cvx.value,1), norm(s_cvx.value .- s));

@printf("Smooth:  Us=%1.4e vs  CVX=%1.4e    Nonsmooth: Us=%1.4e   CVX=%1.4e\n",
norm(s.+q)^2,norm(s_cvx.value .+ q)^2 , norm(s+x,0), norm(s_cvx.value+x,0));


end
