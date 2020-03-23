export hardproxB2

function hardproxB2(z, x, t, lambda, tau)
# %HARDPROXB2 computes the prox of the sum of shifted 1-norm and L2
# %constraint for a scalar variable

fval(s)=norm(s.-z).^2./(2.*t) .+ lambda.*norm(s.+x,1)
projbox(y)= min(max(y, z.-lambda.*t),z.+lambda.*t) # different since through dual
froot(eta)= eta - norm(projbox((-x).*(eta/tau)))


# %do the 2 norm projection
y1 = projbox(-x) #start with eta = tau
if (norm(y1)<= tau)
    y = y1;  # easy case
    str = 'y in tau'
else
    eta = find_zero(froot, tau)
    y = projbox((-x).*(eta/tau))
    str = 'y root'

end

if(norm(y)<=tau)
    s = y
    str2 = 'within tau'
else
    s = tau.*y./norm(y)
    str2 = 'out tau'
end
f = fval(s)

# fprintf('Y-meth: %s    s-meth: %s    s: %1.4f   y:%1.4f\n', str, str2, s, y);
return s,f


end
