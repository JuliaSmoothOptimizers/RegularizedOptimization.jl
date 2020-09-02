% function hardproxtestBinf
n = 4; 
xk = 10*randn(n,1); 
q = 5*randn(n,1);
nu = 20*rand(1,1);
lambda = 10*rand(1,1); 
Deltak = 3*rand(1,1);



cvx_precision high
cvx_begin
    variable s_cvx(n)
    minimize( sum_square(s_cvx+q)/(2*nu) + lambda*norm(s_cvx+xk,1))
    subject to
        -Deltak <= s_cvx <= Deltak
cvx_end

[s,f] = hardproxBinf(q, xk, nu, lambda, Deltak);
f
s
s_cvx
norm(s_cvx - s)


% end


