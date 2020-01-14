# Julia Testing function
# Generate Compressive Sensing Data
using TRNC, Plots,Printf, Convex,SCS, Random, LinearAlgebra, IterativeSolvers

#Here we just try to solve an easy example
#######
# min_s ||As - b||^2 + λ||s||_1
compound=1
m,n = compound*120,compound*512
p = randperm(n)
k = compound*20
#initialize x
x0 = zeros(n,)
x0[p[1:k]]=sign.(randn(k))

A,_ = qr(randn(n,m))
B = Matrix(A')
b0 = B*x0
b = b0 + 0.001*rand(n,)
λ = .1*maximum(abs.(B'*b))



S = Variable(n)
problem = minimize(sumsquares(B*S - b)/2+λ*norm(vec(S), 1))
solve!(problem, SCSSolver())

function proxp(z, α)
    return sign.(z).*max(abs.(z).-(α)*ones(size(z)), zeros(size(z)))
end

function funcF(z)
    return norm(B*z - b,2)^2/2 , B'*(B*z-b)
end

#input β, λ
pg_options=s_options(norm(B)^2; maxIter=10000, verbose=1, λ=λ, optTol=1e-6)
sp = zeros(n)
sp, hispg, fevalpg = PG(funcF, sp, proxp,pg_options)

fista_options=s_options(norm(B)^2; maxIter=10000, verbose=1, λ=λ, optTol=1e-6)
sf = zeros(n)
sf, hispg, fevalpg = PG(funcF, sf, proxp,pg_options)
@printf("VarSplit l2-norm CVX: %5.5e\n", norm(S.value - sp)/norm(S.value))
@printf("CVX: %5.5e     VarSplit: %5.5e\n", norm(B*S.value)^2/2 + λ*norm(vec(S.value),1), funcF(sp)[1]+λ*norm(sp,1))
@printf("True l2-norm CVX: %5.5e\n", norm(S.value - x0)/norm(x0))
@printf("True l2-norm VarSplit: %5.5e\n", norm(sp - x0)/norm(x0))
