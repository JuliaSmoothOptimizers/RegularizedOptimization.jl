% function hardproxtestBinf
n = 4; 
x = 10*randn(n,1); 
z = 5*randn(n,1);
t = 20*rand(1,1);
lambda = 10*rand(1,1); 
tau = 3*rand(1,1);



cvx_precision high
cvx_begin
    variable s_cvx(n)
    minimize( sum_square(s_cvx-z)/(2*t) + lambda*norm(s_cvx+x,1))
    subject to
        -tau <= s_cvx <= tau
cvx_end

[s,f] = hardproxBinf(z, x, t, lambda, tau);
f
s
s_cvx
norm(s_cvx - s)


% end


