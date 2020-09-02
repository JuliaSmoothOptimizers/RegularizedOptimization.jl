function [s,f] = hardproxB0Binf(q, x, nu, lambda, tau)
%HARDPROX computes the prox of the sum of shifted 0-norm and interval
%constraint for a scalar variable 
projbox = @(y) min(max(y, x-tau),x+tau); % different since through dual 
Fcn = @(s)norm(s+q)^2/(2*nu);


w = x - q;
% y = projbox(w);
% [~,p] = sort(y,'descend');
% y(p(lambda+1:end))=0;

[~,p] = sort(w,'descend');
w(p(lambda+1:end))=0;
y = projbox(w);
y(w==0) = 0; 
s = y - x; 


for i = 1:numel(x)
    ft = @(s1) (s1+q(i))^2/(2*nu); 
    stemp = linspace(-tau, tau, 100);
    fi = zeros(100,1); 
    for j = 1:100
        fi(j) = ft(stemp(j)); 
    end
    [mfi, ifi] = min(fi); 
    figure;
    plot(stemp, fi,'*')
    fprintf('min - f: %1.4f argmin - f: %1.4f  min - fs: %1.4f   s: %1.4f x-q: %1.4f\n', mfi, stemp(ifi),ft(s(i)), s(i), x(i)-q(i));  
   
end
% w = xk - gk
% y = ProjB(w, zeros(size(xk)), Δ)
% r = (1/(2*ν))*((y - (xk - gk)).^2 - (xk - gk))
% p = sortperm(r, rev=true)
% y[p[λ+1:end]].=0
% s = y - xk
f = Fcn(s);


end

