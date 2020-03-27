export hardproxBinf

# function hardproxBinf(q, x, ν, λ, Δ)
function hardproxBinf(Fcn, x, ProjB, options)
# HARDPROX computes the prox of the sum of shifted 1-norm and interval
# constraint for a scalar variable
s = zeros(size(x))
f = zeros(size(x))

λ = options.λ
ν = 1.0/options.β
Bk = options.Bk
xk = options.xk
gk = options.gk #note that q = gk = ∇f(x_k) for this example
Δ = options.Δ




# for i=1:length(q)
# fval(y) = (y-(x[i]+q[i]))^2/(2*ν)+λ*abs(y)
# projbox(w) = min(max(w,x[i]-Δ), x[i]+Δ)

y1 = zeros(size(x))
f1 = Fcn(y1, gk, ν)
idx = (y1.<xk.-Δ) .| (y1.>xk .+ Δ) #actually do outward since more efficient
f1[idx] .= Inf

# if y1>x[i]-Δ && y1<x[i]+Δ
#     f1 =fval(y1)
# else
#     f1 = Inf
# end

y2 = ProjB(xk+gk.-ν*λ, xk, Δ)
f2 = Fcn(y2,gk, ν)
y3 = ProjB(xk+gk.+ν*λ, xk, Δ)
f3 = Fcn(y3,gk, ν)
smat = hcat(y1, y2, y3) #to get dimensions right
# fvec = [f1; f2; f3]
fvec = hcat(f1, f2, f3)

# f[i]= minimum(fvec)
f = minimum(fvec, dims=2)
# idx = argmin(fvec)
idx = argmin(fvec, dims=2)
# s[i] = smat[idx]-x[i]
s = smat[idx]-x
# end

return s, sum(f), 1 #funEvals = 1 here
end
