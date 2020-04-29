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

st = zeros(size(s⁻))

for i = 1:length(s⁻)
    absx = abs(x[i])
    if absx <=c
        st[i] = 0
    else
        st[i] = s⁻[i]
    end
end


s = ProjB(st, xk, Δ) - xk
f = Fcn(s)
# @printf("Y-meth: %s    s-meth: %s    s: %1.4e   y:%1.4e\n", str, str2, s[1], y[1]);
return s,s⁻, f,1 #funEvals=1 here


end
