export hardproxBinf

function hardproxBinf(z, x, t, lambda, tau)
# HARDPROX computes the prox of the sum of shifted 1-norm and interval
# constraint for a scalar variable

fval(s) = (s.-z).^2/(2*t) .+ lambda.*abs.(s.+x)
projbox(w) = min.(max.(w,-tau),tau)

s1 = 0
f1 = fval(s1)
s2 = projbox(z.-t.*lambda)
f2 = fval(s2)
s3 = projbox(z.+t.*lambda)
f3 = fval(s3)
smat = [s1; s2; s3]
fvec = [f1; f2; f3]

(f, idx) = min.(fvec)
s = smat(:, idx)

return s, f
end
