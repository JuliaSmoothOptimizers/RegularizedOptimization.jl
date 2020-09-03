using TRNC,Printf, Convex,SCS, Random, LinearAlgebra, Roots


function hardproxtestB0Binf(compound)
    
    m,n = compound*25, compound*64
    p = randperm(n)
    k = compound*2

    #initialize x 
    x0 = zeros(n)
    p = randperm(n)[1:k]
    x0[p[1:k]]=sign.(randn(k))
    xk = 10*randn(n)


    A,_ = qr(5*randn((n,m)))
    B = Array(A)'
    A = Array(B)

    b0 = A*x0
    b = b0 + .005*randn(m)
    λ = 0.1*norm(A'*b, Inf)
    Δ = .3*rand()


    β = eigmax(A'*A)
    δ = k

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
        h = 1 
    end
    return λ*h 
end


(qk, ∇qk, H) = f_obj(xk)
Hess(d) = H*d


Doptions=s_options(β; maxIter=100, verbose=10, λ=δ,
    ∇fk = ∇qk, Bk = A'*A, xk=xk, Δ = Δ)

objInner(d) = [0.5*(d'*Hess(d)) + ∇qk'*d + qk, Hess(d) + ∇qk]

function objInner!(d, g)
    g[:] = ∇qk[:]
    g += Hess(d)
    return 0.5*(d'*Hess(d)) + ∇qk'*d + qk
end
function proxB0binf!(q, σ)
    ProjB(y) = min.(max.(y, -Δ), Δ)

    w = xk - q
    pp = sortperm(w,rev=true)
    w[pp[δ+1:end]].=0
    w = ProjB(w) - xk

    q[:] = w[:]
end

s = zeros(size(xk))
# s⁻, his, funEvals = PG!(objInner!, s,  proxB0binf!, Doptions)
(s,s⁻, f, funEvals) = hardproxB0Binf(objInner, s, h_obj, Doptions)




s_cvx = Variable(n)
opt = () -> SCS.Optimizer(verbose=false)
problem = minimize(sumsquares(A*(xk+s_cvx) - b), [norm(s_cvx, Inf)<=Δ, norm(s_cvx+xk, 1)<=δ]);
solve!(problem, opt)


@printf("Us: %1.4e    CVX: %1.4e    s: %1.4e   s_cvx: %1.4e    normdiff: %1.4e\n",
f_obj(xk + s)[1] + h_obj(xk + s), f_obj(xk + s_cvx.value)[1] + h_obj(xk + s_cvx.value), norm(s+xk,0), norm(s_cvx.value + xk, 0), norm(s_cvx.value .- s));

# @show s_cvx.value 
temp = sum(x->x>0, s - s⁻)
@show s - s⁻

end
