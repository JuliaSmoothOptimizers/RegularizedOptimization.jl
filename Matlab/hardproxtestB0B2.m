% function hardproxtestBinf
n = 6;
% rng(2);
[A,~] = qr(5*randn(n,n));

A = A';

k = floor(.5*n);
p = randperm(n);

%initialize x
x = zeros(n,1);
x(p(1:k))=sign(randn(k,1));

% scalarsnor
nu = 1/norm(A'*A)^2;
l = 10*rand(1);
t = 10*rand(1);

% This constructs q = ν∇qᵢ(sⱼ) = Bksⱼ + gᵢ (note that i = k in paper notation)
q = randn(size(x));%A'*(A*x - zeros(size(x))) %l0 it might tho - gradient of smooth function
% scalars
lambda = k; 



cvx_precision high
cvx_begin quiet
    variable s_cvx(n)
    minimize( sum_square(s_cvx+q)/(2*nu))
    subject to
        norm(s_cvx,Inf) <= t
        norm(s_cvx,1) <=lambda
cvx_end



[s,f] = hardproxB0Binf(q, x, nu, lambda, t);

x
s
s_cvx

fprintf('Us: %1.4f    CVX: %1.4f    s: %1.4f   s_cvx: %1.4f    normdiff: %1.4f\n',...
    f, sum_square(s_cvx-q)/(2*nu) + lambda*norm(s_cvx+x,1), norm(s)^2, norm(s_cvx)^2, norm(s_cvx - s)); 

    



