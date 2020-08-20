export hardproxl1Binf


function hardproxl1Binf(mkB, s⁻, Rk, options)
    # HARDPROX computes the prox of the sum of shifted 1-norm and interval
    # constraint for a scalar variables
    
    λ = options.λ
    ν = 1.0/options.β
    Δ = options.Δ
    
    function prox(q, σ)
        Fcn(yp) = (yp-xk-q).^2/(2*ν)+λ*abs.(yp)
        ProjBox(wp) = min.(max.(wp,xk.-Δ), xk.+Δ)
        
        y1 = zeros(size(x))
        f1 = Fcn(y1)
        idx = (y1.<xk.-Δ) .| (y1.>xk .+ Δ) #actually do outward since more efficient
        f1[idx] .= Inf
    
        y2 = ProjB(xk+q.-σ)
        f2 = Fcn(y2)
        y3 = ProjB(xk+q.+σ)
        f3 = Fcn(y3)

        smat = hcat(y1, y2, y3) #to get dimensions right
        fvec = hcat(f1, f2, f3)
    
        f = minimum(fvec, dims=2)
        idx = argmin(fvec, dims=2)
        s = smat[idx]-xk
    
        return dropdims(s, dims=2)
    end
    
    s,s⁻, his, funEvals = PG(mkB, s⁻,  prox, options)
    f = his[end] + Rk(xk + s)
    # @printf("Y-meth: %s    s-meth: %s    s: %1.4e   y:%1.4e\n", str, str2, s[1], y[1]);
    return s,s⁻, f, funEvals 

end
    
# # function hardproxBinf(q, x, ν, λ, Δ)
# function hardproxl1Binf(Fcn, x, ProjB, options)
# # HARDPROX computes the prox of the sum of shifted 1-norm and interval
# # constraint for a scalar variable
# s = zeros(size(x))
# f = zeros(size(x))

# λ = options.λ
# ν = 1.0/options.β
# Bk = options.Bk
# xk = options.xk
# gk = options.∇fk #note that q = gk = ∇f(x_k) for this example
# Δ = options.Δ




# # for i=1:length(q)
# # fval(yp, bq, bx, νi) = (yp-bx+bq).^2/(2*νi)+λ*abs.(yp)
# # projbox(wp, bx, Δi) = min.(max.(wp,bx.-Δi), bx.+Δi)

# y1 = zeros(size(x))
# f1 = Fcn(y1, gk, xk, ν)
# idx = (y1.<xk.-Δ) .| (y1.>xk .+ Δ) #actually do outward since more efficient
# f1[idx] .= Inf

# # if y1>x[i]-Δ && y1<x[i]+Δ
# #     f1 =fval(y1)
# # else
# #     f1 = Inf
# # end

# y2 = ProjB(xk-gk.-ν*λ, xk, Δ)
# f2 = Fcn(y2,gk,xk,ν)
# y3 = ProjB(xk-gk.+ν*λ, xk, Δ)
# f3 = Fcn(y3,gk,xk,ν)
# smat = hcat(y1, y2, y3) #to get dimensions right
# # fvec = [f1; f2; f3]
# fvec = hcat(f1, f2, f3)

# # f[i]= minimum(fvec)
# f = minimum(fvec, dims=2)
# # idx = argmin(fvec)
# idx = argmin(fvec, dims=2)
# # s[i] = smat[idx]-x[i]
# s = smat[idx]-xk
# # end

# return dropdims(s, dims=2),x, sum(f), 1 #funEvals = 1 here
# end
