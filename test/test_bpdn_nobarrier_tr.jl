function bpdnNoBar(x0, xi, A, f_obj, h_obj, λ, parameters, options)
    xtr, ktr, Fhisttr, Hhisttr, Comp_pg = TR(xi, parameters, options)

    objI = f_obj(x0)[1] + λ * h_obj(x0)
    objTR = f_obj(xtr)[1] + λ * h_obj(xtr)
    if isa(A, Array)
        partest = norm(x0 - xtr) / opnorm(A)
    else
        partest = norm(x0 - xtr)
    end

    objtest = abs(objI - objTR)



    return partest, objtest  
end
