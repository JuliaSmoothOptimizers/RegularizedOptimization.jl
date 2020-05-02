function [s,f] = hardproxB0Binf(q, x, nu, lambda, tau)
%HARDPROX computes the prox of the sum of shifted 0-norm and interval
%constraint for a scalar variable 
projbox = @(y) min(max(y, -tau),+tau); % different since through dual 
Fcn = @(s)norm(s+q)^2/(2*nu);


w = x - q;
[~,p] = sort(w,'descend');
w(p(lambda+1:end))=0;

s = projbox(w) ;
% w = xk - gk
% y = ProjB(w, zeros(size(xk)), Δ)
% r = (1/(2*ν))*((y - (xk - gk)).^2 - (xk - gk))
% p = sortperm(r, rev=true)
% y[p[λ+1:end]].=0
% s = y - xk
f = Fcn(s);


end

