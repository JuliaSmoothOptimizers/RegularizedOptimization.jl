# Julia Testing function
# Generate Compressive Sensing Data
using TRNC, Plots,Printf, Convex,SCS, Random, LinearAlgebra, IterativeSolvers

#Here we just try to solve an easy example
#######
# min_s ||As - b||^2 + λ||s+c||_1 but we taylor expand ||As-b||^2 to simulate
#the nature of the problem
compound=1
m,n = compound*200,compound*512
p = randperm(n)
k = compound*10
#initialize x
x0 = zeros(n)
p   = randperm(n)[1:k]
x0 = zeros(n,)
x0[p[1:k]]=sign.(randn(k))

A,_ = qr(randn(n,m))
B = Array(A)'
A = Array(B)

b0 = A*x0
b = b0 + 0.001*rand(m,)
λ = norm(A'*b, Inf)/10
g = -A'*b

c = .01*randn(n) #pretty important here - definitely hurts the sparsity

S = Variable(n)
problem = minimize(sumsquares(A*S)/2 + g'*S + b'*b/2 + λ*norm(vec(S+c), 1))
solve!(problem, SCS.Optimizer)

function proxp(z, α)
    return sign.(z).*max.(abs.(z).-(α)*ones(size(z)), zeros(size(z)))
end

function funcF(z)
    f = dot(b,b)/2 +dot(g,z-c) + norm(A*(z-c))^2/2
    grad = g + A'*(A*(z-c))
    return f, grad
end

pg_options=s_options(norm(A)^2; maxIter=10000, verbose=1, λ=λ, optTol=1e-6)
sp = zeros(n)
up,ups, hispg, fevalpg = PG(funcF, sp, proxp,pg_options)
sp = up-c

fista_options=s_options(norm(A)^2; maxIter=10000, verbose=5, λ=λ, optTol=1e-6)
sf = randn(n)
uf,ufs, hisf, fevalpg = FISTA(funcF, sf, proxp,pg_options)
sf = uf - c
@printf("PG l2-norm CVX: %5.5e\n", norm(S.value - sp)/norm(S.value))
@printf("FISTA l2-norm CVX: %5.5e\n", norm(S.value - sf)/norm(S.value))
@printf("CVX: %5.5e     PG: %5.5e   FISTA: %5.5e\n", norm(A*S.value)^2 + Matrix(g'*S.value)[1] + b'*b + λ*norm(vec(S.value),1), funcF(sp+c)[1]+λ*norm(sp+c,1), funcF(sf+c)[1]+λ*norm(sf+c,1))
@printf("True l2-norm CVX: %5.5e\n", norm(S.value - x0)/norm(x0))
@printf("True l2-norm PG: %5.5e\n", norm(sp - x0)/norm(x0))
@printf("True l2-norm FISTA: %5.5e\n", norm(sf - x0)/norm(x0))
