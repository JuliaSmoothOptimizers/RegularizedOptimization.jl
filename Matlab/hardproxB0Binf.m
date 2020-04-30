function [s,f] = hardproxB0Binf(q, x, nu, lambda, tau)
%HARDPROX computes the prox of the sum of shifted 0-norm and interval
%constraint for a scalar variable 

w = xk - gk;
p = sort(w,rev=true);
w[p[λ+1:end]].=0
s = ProjB(w, xk, Δ) - xk
% w = xk - gk
% y = ProjB(w, zeros(size(xk)), Δ)
% r = (1/(2*ν))*((y - (xk - gk)).^2 - (xk - gk))
% p = sortperm(r, rev=true)
% y[p[λ+1:end]].=0
% s = y - xk
f = sum(Fcn(s, gk, xk, ν))


end

