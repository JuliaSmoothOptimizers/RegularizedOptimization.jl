function hardproxtestBinf
x = 10*randn(1); 
z = 5*randn(1);
t = 20*rand(1);
lambda = 10*rand(1); 
tau = 3*rand(1);

[s,f] = hardproxBinf(z, x, t, lambda, tau);

cvx_precision high
cvx_begin
    variable s_cvx(1)
    minimize( (s_cvx-z)^2/(2*t) + lambda*abs(s_cvx+x))
    subject to
        -tau <= s_cvx <= tau
cvx_end

norm(s_cvx - s)

end


