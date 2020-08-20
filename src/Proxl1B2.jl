export hardproxl1B2

function hardproxl1B2(mkB, s⁻, Rk, options)
# function HP_test(mkB, s⁻, Rk, options)
    # # %HARDPROXB2 computes the prox of the sum of shifted 1-norm and L2
    # # %constraint for a scalar variable
    xk = options.xk
    Δ = options.Δ
    
    #should be ObjInner [0.5*(d'*∇²qk(d)) + ∇qk'*d + qk, ∇²qk(d) + ∇qk]

    function prox(q, σ) #s - ν*g, ν*λ - > basically inputs the value you need

        ProjB(y) = min.(max.(y, q.-σ), q.+σ)
        froot(η) = η - norm(ProjB((-xk).*(η/Δ)))

    
        # %do the 2 norm projection
        y1 = ProjB(-xk) #start with eta = tau
        if (norm(y1)<= Δ)
            y = y1  # easy case
        else
            η = fzero(froot, 1e-10, Inf)
            y = ProjB((-xk).*(η/Δ))
        end
    
        if (norm(y)<=Δ)
            snew = y
        else
            snew = Δ.*y./norm(y)
        end
        return snew
    end 



    s,s⁻, his, funEvals = PG(mkB, s⁻,  prox, options)
    f = his[end] + Rk(xk + s)
    # @printf("Y-meth: %s    s-meth: %s    s: %1.4e   y:%1.4e\n", str, str2, s[1], y[1]);
    return s,s⁻, f, funEvals 
end