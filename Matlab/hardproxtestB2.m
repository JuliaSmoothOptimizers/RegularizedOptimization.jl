n = 1; 

% vectors 
x = 10*randn(n,1); 
z = 5*randn(n,1);

% scalars
t = 20*rand(1);
lambda = 10*rand(1); 
tau = 3*rand(1);

[s,f] = hardproxB2(z, x, t, lambda, tau);

cvx_precision high
cvx_begin
    variable s_cvx(n)
    minimize( sum_square(s_cvx-z)/(2*t) + lambda*norm(s_cvx+x,1))
    subject to
        norm(s_cvx,2) <=tau
cvx_end

norm(s_cvx - s)
