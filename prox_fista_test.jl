# Julia Testing function
# Generate Compressive Sensing Data
using Plots, Printf, Random, LinearAlgebra
include("DescentMethods.jl")
# include("ProxProj.jl")


#Here we just try to solve the Lasso Problem
#######
# min_x 1/2||Ax - b||^2 + λ||x||_1
# srand(123)
m,n = 200,512 # this is a under determine system

A = rand(m,n)
xt  = zeros(n)
k   = 10;     # nonzeros in xt
p   = randperm(n)[1:k]
for i = 1:k
    xt[p[i]] = (5.0+randn())*sign(rand()-0.5);
end
b   = A*xt;
L   = norm(A)^(2.0);

function funcF(x)
    r = A*x - b
    # BLAS.gemv!('N',1.0,A,x,-1.0,r)
    # g = BLAS.gemv('T',1.0,A,r)
    g = A'*r
    return norm(r), g
end
function proxG(x,λ, α)
    n = length(x)
    for i = 1:n
        x[i] > α*λ ? x[i] -= α*λ :
        x[i] <-α*λ ? x[i] += α*λ : x[i] = 0.0;
    end
    return x
    # return sign.(x).*max(abs.(x).-(λ*α)*ones(size(x)), zeros(size(x)))
end

#input β, λ
options = s_options(L; optTol = 1e-10, verbose=1)
funProj(x) = proxG(x, norm(A'*b, Inf)/50.0, L^(-1))


x1 = rand(n)
xp, hispg, fevalpg = PG(funcF, x1, funProj,options)
x2 = rand(n)
xf, hisf, fevalf = FISTA(funcF, x2, funProj, options)

@printf("l2-norm| PG: %5.5e | FISTA: %5.5e\n", norm(xp - xt), norm(xf-xt))
plot(hispg, xscale=:log10, yscale=:log10, xlabel="Iteration", ylabel="Descent", title="Descent Comparison", label="ProxGrad", marker=1)
plot!(hisf, label="FISTA", marker=1)
savefig("hist_prox_test.pdf")
plot(xp, xlabel="ith position", ylabel="x value", title="x minimized", label="ProxGrad",marker=1)
plot!(xf, label="FISTA",marker=1)
plot!(xt, label="True",marker=1)
savefig("x_prox_test.pdf")

