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

% scalars
v = 1/norm(A'*A)^2;
l = 10*rand(1);
t = 10;

% This constructs q = ν∇qᵢ(sⱼ) = Bksⱼ + gᵢ (note that i = k in paper notation)
q = 10*randn(size(x));%A'*(A*x - zeros(size(x))) %l0 it might tho - gradient of smooth function

% Doptions=s_options(1/ν; maxIter=10, λ=λ, ∇fk = q, Bk = A'*A, xk=x, Δ = τ)

[s,f] = hardproxl0Binf(q,x,v, l, t); 



% cvx_precision high
cvx_begin
    variable s_cvx(n)
    minimize(sum_square(s_cvx+q)/(2*v) + l*norm(s_cvx+x,1))
    subject to
        -t <= s_cvx <= t
cvx_end

f
s
s_cvx
fc = norm(s_cvx+q)^2/(2*v) + l*norm(s_cvx+x,1)
normdiff = norm(s_cvx - s)


% end


