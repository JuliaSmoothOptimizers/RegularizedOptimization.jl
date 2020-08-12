export power_iteration

function power_iteration(A, x; tol=1e-10)
    
    k = maximum(size(x))*10000
    λ, x1 = pwrsub(A, x, tol, k)
    
    if λ < 0
        Bk(x) = A(x) - λ*x
        x2 = randn(size(x1))
        μ, x3 = pwrsub(Bk, x2, tol, k)
        λ = μ + λ
    end
    
    return λ, x3


end

function pwrsub(A, ak, tol, iters)

    for i =1:iters
        #this should be matrix multiplication
        a = A(ak)

        #normalize
        a = a/norm(a)

        # if abs(μ - μ_im1)<tol 
        if norm(a - ak)<tol
            break
        end

        ak = a

    end
    μ = ak'*A(ak)

    return μ, bk

end