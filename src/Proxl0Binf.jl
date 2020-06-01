export hardproxl0Binf

function hardproxl0Binf(Fcn, s⁻, ProjB, options)
# %hardproxl0Binf computes the prox of the sum of shifted l0-norm and linf
# %constraint for a scalar variable
λ = options.λ
ν = 1.0/options.β
Bk = options.Bk
xk = options.xk
gk = options.∇fk
Δ = options.Δ
#make the constant
c = sqrt(2*λ*ν)
w = xk-gk
st = zeros(size(s⁻))

# y = ProjB(w, zeros(size(xk)), Δ)
# r = (1/(2*ν))*((y - (xk - gk)).^2 - (xk - gk))
# p = sortperm(r, rev=true)
# y[p[Int(λ)+1:end]].=0
# s = y - xk

for i = 1:length(s⁻)
    absx = abs(w[i])
    if absx <=c
        st[i] = 0
        k = k+1; 
    else
        st[i] = w[i]
    end
end

s = ProjB(st, xk, Δ) - xk
f = sum(Fcn(s, gk, xk, ν))
# @printf("Y-meth: %s    s-meth: %s    s: %1.4e   y:%1.4e\n", str, str2, s[1], y[1]);
return s,s⁻, f,1 #funEvals=1 here


end
