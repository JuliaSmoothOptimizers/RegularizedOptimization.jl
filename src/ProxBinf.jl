export hardproxBinf

function hardproxBinf(q, x, ν, λ, τ)
# HARDPROX computes the prox of the sum of shifted 1-norm and interval
# constraint for a scalar variable
s = zeros(size(q))
f = zeros(size(q))

for i=1:length(q)
fval(s) = (s-(x[i]+q[i])^2/(2*ν)+λ*abs(s)
projbox(w) = min(max(w,x[i]-τ),x[i]+τ)

y1 = 0
if y1>x[i]-τ && y1<x[i]+τ
    f1 =fval(y1)
else
    f1 = Inf
end

y2 = projbox(q[i]-ν*λ)
f2 = fval(y2)
y3 = projbox(q[i]+ν*λ)
f3 = fval(y3)
smat = [y1, y2, y3]
fvec = [f1; f2; f3]

f[i]= mininum(fvec)
idx = argmin(fvec)
s[i] = smat[:, idx]-x[i]

end

return s, f
end
