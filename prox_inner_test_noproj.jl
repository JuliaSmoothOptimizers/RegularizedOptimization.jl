# Julia Testing function
# Generate Compressive Sensing Data
using TRNC, Plots,Printf, Convex,SCS, Random, LinearAlgebra, IterativeSolvers
include("src/minconf_spg/oneProjector.jl")

#Here we just try to solve an easy example
#######
# min_s gᵀs + 1/2sᵀBks + λ||s+x||_1	s.t. ||s||_1⩽1
# function prox_inner_test()
# m,n = 200,200 # let's try something not too hard
# g = randn(n)
# B = rand(n,n)
# Bk = B'*B
# x  = rand(n)
# λ = 1.0
compound=1
m,n = compound*120,compound*512
p = randperm(n)
k = compound*20
#initialize x
x0 = zeros(n,)
x0[p[1:k]]=sign.(randn(k))

A = randn(m,n)
(Q,_) = qr(A')
A = Matrix(Q)
B = Matrix(A')
b0 = B*x0
b = b0 + 0.001*rand(m,)
λ = .1*maximum(abs.(B'*b))



S = Variable(n)
problem = minimize(sumsquares(B*S - b)/2+λ*norm(vec(S), 1))
solve!(problem, SCSSolver())

function proxp(z, α)
    return sign.(z).*max(abs.(z).-(α)*ones(size(z)), zeros(size(z)))
end
# projq(z, σ) = oneProjector(z, 1.0, σ)
function funcF(x)
    return norm(B*x - b,2)^2/2 , B'*(B*x-b)
end

#input β, λ
w2_options=s_options(norm(Bk)^2; maxIter=10000, verbose=1, restart=100, λ=λ)
# s2,w12,w22 = prox_split_2w(proxp, zeros(size(x)), projq, w2_options)


s1 = zeros(n)
sp, hispg, fevalpg = PG(funcF, s1, proxp,w2_options)
# x2 = rand(n)
# xf, hisf, fevalf = FISTA(funcF, x2, funProj, options)
@printf("l2-norm CVX: %5.5e\n", norm(S.value - sp)/norm(S.value))
@printf("CVX: %5.5e     PG: %5.5e\n", norm(B*S.value)^2/2 + λ*norm(vec(S.value),1), funcF(sp)[1]+λ*norm(sp,1))
@printf("l2-norm CVX: %5.5e\n", norm(S.value - x0)/norm(x0))
@printf("l2-norm PG: %5.5e\n", norm(sp - x0)/norm(x0))

# end
