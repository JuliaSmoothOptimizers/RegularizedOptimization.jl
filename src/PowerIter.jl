export power_iteration




function power_iteration(A, bk; tol=1e-10)
    
    k = maximum(size(bk))
    μ = norm(bk)

    for i =1:k
        μ_im1 = μ
        #this should be matrix multiplication
        b = A(bk)

        #normalize
        nb = norm(b)
        bk = b/nb

        μ = (bk'*b)/nb

        if abs(μ - μ_im1)<tol 
            break
        end



    end

    return μ, bk


end