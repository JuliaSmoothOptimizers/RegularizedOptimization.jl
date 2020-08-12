export power_iteration

function power_iteration(A, bk; tol=1e-10)
    
    k = maximum(size(bk))*10000
    μ, bk = pwrsub(A, bk, tol, k)
    
    if μ < 0
        Bk(x) = A(x) + μ*x
        μ, bk = pwrsub(Bk, bk, tol, k)
    end

    return μ, bk


end

function pwrsub(A, bk, tol, iters)

    for i =1:iters
        #this should be matrix multiplication
        b = A(bk)

        #normalize
        b = b/norm(b)
        # μ = (bk'*b)/(bk'*b)
        # μ = b'*A(bk)

        # if abs(μ - μ_im1)<tol 
        if norm(b - bk)<tol
            break
        end

        bk = b

    end
    μ = bk'*A(bk)

    return μ, bk

end