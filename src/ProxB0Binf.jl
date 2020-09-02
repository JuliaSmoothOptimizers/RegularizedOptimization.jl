export hardproxB0Binf

function hardproxB0Binf(mkB, s⁻, Rk, options)


    xk = options.xk
    Δ = options.Δ


    function prox(q, σ)
        ProjB(w) = min.(max.(w, -Δ), Δ)

        w = xk - gk
        p = sortperm(w,rev=true)
        w[p[λ+1:end]].=0
        s = ProjB(w) - xk
        # w = xk - gk
        # y = ProjB(w, zeros(size(xk)), Δ)
        # r = (1/(2*ν))*((y - (xk - gk)).^2 - (xk - gk))
        # p = sortperm(r, rev=true)
        # y[p[λ+1:end]].=0
        # s = y - xk
        return s 
    end

    s,s⁻, his, funEvals = PG(mkB, s⁻,  prox, options)
    f = his[end] + Rk(xk + s)

    return s, s⁻, f, funEvals 


end
