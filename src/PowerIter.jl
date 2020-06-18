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
        μ = b'*A(bk)/(b'*bk)

        bk = b

        if abs(μ - μ_im1)<tol 
            break
        end



    end

    return μ, bk


end