function [s,f] = hardproxBinf(q, x, nu, lambda, tau)
%HARDPROX computes the prox of the sum of shifted 1-norm and interval
%constraint for a scalar variable 

s = zeros(size(q));
f = zeros(size(q)); 
for i=1:numel(q)

fval = @(y) (y-(x(i)+q(i)))^2/(2*nu) + lambda*abs(y); 
projbox = @(w) min(max(w,x(i)-tau),x(i)+tau);

y1 = 0; 
if y1>x(i)-tau && y1<x(i)+tau
    f1 = fval(y1);
else
    f1 = Inf;
end



y2 = projbox(x(i)+q(i)-nu*lambda); f2 = fval(y2);
y3 = projbox(x(i)+q(i)+nu*lambda); f3 = fval(y3);
smat = [y1, y2, y3];
fvec = [f1; f2; f3];

[f(i), idx] = min(fvec);
s(i) = smat(:, idx)-x(i);


end
f = sum(f); 
end

