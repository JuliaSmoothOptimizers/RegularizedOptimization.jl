# Julia Testing function
# Generate Compressive Sensing Data
using Plots, Printf, Random, LinearAlgebra, Convex, SCS
include("prox_inner.jl")


#Here we just try to solve an easy example 
#######
# min_s 1/2||s-z||^2 + λ||s+x||_1
srand(123)
m,n = 200,200 # let's try something not too hard 
g = randn(n)
B = rand(n,n)
B = B'*B
sj = randn(n)

z = g'*sj + .5*sj'*B*sj

x  = rand(n)
k   = 10;     # nonzeros in xt
p   = randperm(n)[1:k]
for i = 1:k
    xt[p[i]] = (5.0+randn())*sign(rand()-0.5);
end
b   = A*xt;
L   = norm(A)^(2.0);


S = Variable(n)
problem = minimize(sumsquares(A * X - b), X>=l, X<=u)
solve!(problem, SCSSolver())


#input β, λ
options = s_options(L; optTol = 1e-10, verbose=1)
funProj(x) = proxG(x, norm(A'*b, Inf)/50.0, L^(-1))


x1 = rand(n)
xp, hispg, fevalpg = PG(funcF, x1, funProj,options)
x2 = rand(n)
xf, hisf, fevalf = FISTA(funcF, x2, funProj, options)

@printf("l2-norm| PG: %5.5e | FISTA: %5.5e\n", norm(xp - xt), norm(xf-xt))

