# Julia Testing function
# Generate Compressive Sensing Data
using PyPlot, Printf, Random, LinearAlgebra
include("DescentMethods.jl")
# include("ProxProj.jl")


#Here we just try to solve the Lasso Problem
#######
# min_x 1/2||Ax - b||^2 + λ||x||_1


m,n = 200,2000; # this is a under determine system
A = rand(m,n)
xt  = zeros(n);
k   = 10;     # nonzeros in xt
p   = randperm(n)[1:k];
for i = 1:k
    xt[p[i]] = (5.0+randn())*sign(rand()-0.5);
end
b   = A*xt;
λ   = norm(A'*b, Inf)/10.0;
L   = norm(A)^(2.0);
tol = 1e-20;

function funcF!(x,g)
    r = copy(b)
    BLAS.gemv!('N',1.0,A,x,-1.0,r)
    BLAS.gemv!('T',1.0,A,r,0.0,g)
    return norm(r)
end
function proxG!(x,α)
    n = length(x);
    for i = 1:n
        x[i] > α*λ ? x[i] -= α*λ :
        x[i] <-α*λ ? x[i] += α*λ : x[i] = 0.0;
    end
end





x1 = rand(n);
hispg = proxgrad!(x1, L, funcF!, proxG!, tol, print_freq=1000)
x2 = rand(n); 
hisf = FISTA!(x2, L, funcF!, proxG!, tol, print_freq=1000)

semilogy(hispg ,"b")
semilogy(hisf,"g")
xlabel("Iteration")
ylabel("Descent")
title("Descent Comparison")
# legend("ProxGrad", "FISTA")
savefig("temp.pdf")