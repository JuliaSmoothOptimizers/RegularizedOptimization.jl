# Julia Testing function
# Generate Compressive Sensing Data
using TRNC, Plots,Printf, Convex,SCS, Random, LinearAlgebra, IterativeSolvers
include("src/minconf_spg/oneProjector.jl")

#Here we just try to solve an easy example
#######
# min_s gᵀs + 1/2sᵀBks + λ||s+x||_1	s.t. ||s||_1⩽1
#######
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

c = .00*randn(n) #pretty important here - definitely hurts the sparsity

S = Variable(n)
problem = minimize(g'*S + sumsquares(A*S)/2 + b'*b/2+λ*norm(vec(S+c), 1), norm(vec(S),1)<=k)
solve!(problem, SCSSolver())

function proxp(z, α)
    return sign.(z).*max.(abs.(z).-(α)*ones(size(z)), zeros(size(z)))
end
projq(z, σ) = oneProjector(z, 1.0, σ)
# function projq(z,σ)
#     return z/max(1, norm(z, 2)/σ)
# end

vp_options=s_options(norm(A'*A)^2;maxIter=10, verbose=99, restart=100, λ=λ, η =1.0, η_factor=.9,
    gk = g, Bk = A'*A, xk=c, σ_TR = k)
s,w1,w2 = prox_split_2w(proxp, zeros(size(x0)), projq, vp_options)


# @printf("l2-norm CVX: %5.5e\n", norm(S.value - s)/norm(S.value))
# @printf("l2-norm CVX: %5.5e\n", norm(S.value - w)/norm(S.value))
@printf("l2-norm CVX vs VP: %5.5e\n", norm(S.value - s)/norm(S.value))
@printf("l2-norm CVX vs True: %5.5e\n", norm(S.value - x0)/norm(S.value))
@printf("l2-norm VP vs True: %5.5e\n", norm(x0 - s)/norm(x0))
