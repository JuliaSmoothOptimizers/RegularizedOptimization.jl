export hardproxB2

function hardproxB2(q, x, ν, λ, τ)
# %HARDPROXB2 computes the prox of the sum of shifted 1-norm and L2
# %constraint for a scalar variable

fval(s) = norm(s.-q).^2/(2*t) .+ λ.*norm(s.+x,1)
projbox(y) = min.(max.(y, q.-λ.*ν),q.+λ.*ν) # different since through dual
froot(η) = η - norm(projbox((-x).*(η/τ)))


# %do the 2 norm projection
y1 = projbox(-x) #start with eta = tau
if (norm(y1)<= τ)
    y = y1  # easy case
    str = "y in tau"
else
    η = find_zero(froot, τ)
    y = projbox((-x).*(η/τ))
    str = "y root"
end

if(norm(y)<=τ)
    s = y
    str2 = "within tau"
else
    s = τ.*y./norm(y)
    str2 = "out tau"
end
f = fval(s)

# fprintf('Y-meth: %s    s-meth: %s    s: %1.4f   y:%1.4f\n', str, str2, s, y);
return s,f


end
