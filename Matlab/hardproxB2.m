function [s,f] = hardproxB2(z, x, t, lambda, tau)
%HARDPROXB2 computes the prox of the sum of shifted 1-norm and L2
%constraint for a scalar variable 

fval = @(s) (s-z).^2./(2*t) + lambda*abs(s+x); 
projbox = @(w) min(max(w, z-lambda*tau),z+lambda*tau); % different since through dual 
froot = @(eta) eta - norm(projbox((z-x)*(eta/tau)));

y1 = projbox(z-x); 
if (norm(y1)<= tau)
    y = y1;  % easy case
else
    eta = fzero(froot, tau);
    y = projbox((z-x)*(eta/tau));
end

if(norm(y) <=tau)
    s = y;
else 
    s = tau*y/norm(y);
end
f = fval(s);  
    
    
    


end

