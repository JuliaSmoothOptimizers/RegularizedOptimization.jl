n = 40; 
% rng(2); 
% vectors 
x = 10*randn(n,1); 
q = 5*randn(n,1);

% scalars
nu = 20*rand(1);
lambda = 10*rand(1); 
Delta = 3*rand(1);



cvx_precision high
cvx_begin quiet
    variable s_cvx(n)
    minimize( sum_square(s_cvx+q)/(2*nu) + lambda*norm(s_cvx+x,1))
    subject to
        norm(s_cvx,2) <=Delta
cvx_end



[s,f] = hardproxB2(q, x, nu, lambda, Delta);


if n==1
fprintf('Us: %1.4f    CVX: %1.4f    s: %1.4f   s_cvx: %1.4f    normdiff: %1.4f\n',...
    f, sum_square(s_cvx-q)/(2*nu) + lambda*norm(s_cvx+x,1), s, s_cvx, norm(s_cvx - s)); 
else
    fprintf('Us: %1.4f    CVX: %1.4f    s: %1.4f   s_cvx: %1.4f    normdiff: %1.4f\n',...
    f, sum_square(s_cvx-q)/(2*nu) + lambda*norm(s_cvx+x,1), norm(s)^2, norm(s_cvx)^2, norm(s_cvx - s)); 

    
end


