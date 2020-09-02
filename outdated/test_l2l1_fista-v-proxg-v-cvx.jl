# Julia Testing function
# Generate Compressive Sensing Data
using TRNC, Plots,Printf, Convex,SCS, Random, LinearAlgebra, IterativeSolvers
# Random.seed!(123)
#Here we just try to solve an easy example
#######
# min_s ||As - b||^2 + λ||s||_1
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



S = Variable(n)
problem = minimize(sumsquares(A*S - b)/2+λ*norm(vec(S), 1))
solve!(problem, SCS.Optimizer)
#
function proxp!(z, α)
        n = length(z);
        for i = 1:n
            z[i] > α ? z[i] -= α :
            z[i] <-α ? z[i] += α : z[i] = 0.0
        end
end

function funcF!(z, g)
    r = copy(b)
    BLAS.gemv!('N', 1.0, A, z, -1.0, r)
    BLAS.gemv!('T', 1.0, A, r, 0.0, g)
    return r'*r
end
function funcF(x)
    r = A*x - b
    g = A'*r
    return norm(r)^2, g
end
function proxp(z, α)
    return sign.(z).*max.(abs.(z).-(α)*ones(size(z)), zeros(size(z)))
end

#input β, λ
β = eigmax(A'*A)
pg_options=s_options(β; maxIter=10000, verbose=3, λ=λ, optTol=1e-10)
sp = zeros(n)
@printf("%s\n", "PG")
sp,sp⁻, hispg, fevalpg = PG(funcF, sp, proxp,pg_options)
# sp⁻, hispg, fevalpg = PG!(funcF!, sp, proxp!,pg_options)
@printf("%s\n", "FISTA")
fista_options=s_options(β; maxIter=10000, verbose=10, λ=λ, optTol=1e-10)
sf = zeros(n)
sf,sf⁻, hisf, fevalf = FISTA(funcF, sf, proxp, fista_options)
# sf⁻, hisf, fevalf = FISTA!(funcF!, sf, proxp!, fista_options)

@printf("PG l2-norm CVX: %5.5e\n", norm(S.value - sp)/norm(S.value))
@printf("FISTA l2-norm CVX: %5.5e\n", norm(S.value - sf)/norm(S.value))
@printf("CVX: %5.5e     PG: %5.5e   FISTA: %5.5e\n", norm(A*S.value)^2/2 + λ*norm(vec(S.value),1), funcF(sp)[1]+λ*norm(sp,1), funcF(sf)[1]+λ*norm(sf,1))
@printf("True l2-norm CVX: %5.5e\n", norm(S.value - x0)/norm(x0))
@printf("True l2-norm PG: %5.5e\n", norm(sp - x0)/norm(x0))
@printf("True l2-norm FISTA: %5.5e\n", norm(sf - x0)/norm(x0))
