export gradient, bfgs_update



function gradient(Fcn, p; ε=1e-6)
    grad = zeros(size(p))
    p_temp = zeros(ComplexF64, size(p)) + p
    for j = 1:length(p)
        p_temp[j]+=im*ε
        grad[j] = imag(Fcn(p_temp))/ε
        p_temp[j]-=im*ε

    end

    return grad
end


function bfgs_update(Bk, sk, yk)

    f1b = sk'*Bk*sk 
    f1t = (Bk*sk)*(sk'*Bk)

    f2t = yk*yk'
    f2b = yk'*sk 

    return Bk - f1t/f1b + f2t/f2b 

end