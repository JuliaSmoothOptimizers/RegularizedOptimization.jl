# Julia Testing function
# Generate Compressive Sensing Data
using Plots, Printf, Random, LinearAlgebra
include("DescentMethods.jl")
include("ProxProj.jl")
include("IP_alg.jl")


#Here we just try to solve the Lasso Problem
#######
# min_x 1/2||Ax - b||^2 + λ||x||_1


m,n = 100,100; # this is a under determine system
A = rand(m,n)
x0  = rand(n,1);
b0 = A*x0;
b = b0 + 0.5*rand(m,1);
cutoff = 0.0;
l = zeros(n,1)+cutoff*ones(n,1);
u = zeros(n,1)+cutoff*ones(n,1); 

p   = randperm(n)[1:k];
for i = 1:k
    xt[p[i]] = (5.0+randn())*sign(rand()-0.5);
end
b   = A*xt;
λ   = norm(A'*b, Inf)/10.0;
L   = norm(A)^(2.0);
tol = 1e-10;

function funcF(x,g)
    r = copy(b)
    BLAS.gemv!('N',1.0,A,x,-1.0,r)
    BLAS.gemv!('T',1.0,A,r,0.0,g)
    return norm(r), g
end
function proxG(x,α)
    # n = length(x)
    for i = 1:n
        x[i] > α*λ ? x[i] -= α*λ :
        x[i] <-α*λ ? x[i] += α*λ : x[i] = 0.0;
    end
    return x
end





x1 = rand(n);
hispg = proxgrad(x1, L, funcF, proxG, tol, print_freq=1000)
x2 = rand(n);
hisf = FISTA(x2, L, funcF, proxG, tol, print_freq=1000)


plot(hispg[1:end-1], yaxis=:log, xlabel="Iteration", ylabel="Descent", title="Descent Comparison", label="ProxGrad")
plot!(hisf[1:end-1], yaxis=:log, label="FISTA")
savefig("temp.pdf")
