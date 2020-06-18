export power_iteration




function power_iteration(A, bk; tol=1e-10)
    
    k = maximum(size(bk))*100
    μ = norm(bk)

    for i =1:k
        μ_im1 = μ
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

    return bk'*A(bk), bk


end