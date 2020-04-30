export hardproxB0Binf

# function hardproxB2(q, x, ν, λ, τ)
function hardproxB0Binf(Fcn, x, ProjB, options)
# %HARDPROXB2 computes the prox of the sum of shifted 1-norm and L2
# %constraint for a scalar variable
λ = options.λ #surrogate for largest entries
ν = 1.0/options.β
Bk = options.Bk
xk = options.xk
gk = options.∇fk
Δ = options.Δ


w = xk - gk
p = sortperm(w,rev=true)
w[p[λ+1:end]].=0
s = ProjB(w, xk, Δ) - xk
# w = xk - gk
# y = ProjB(w, zeros(size(xk)), Δ)
# r = (1/(2*ν))*((y - (xk - gk)).^2 - (xk - gk))
# p = sortperm(r, rev=true)
# y[p[λ+1:end]].=0
# s = y - xk
f = sum(Fcn(s, gk, xk, ν))

# @printf("Y-meth: %s    s-meth: %s    s: %1.4e   y:%1.4e\n", str, str2, s[1], y[1]);
return s,zeros(size(s)), f,1 #funEvals=1 here


end
