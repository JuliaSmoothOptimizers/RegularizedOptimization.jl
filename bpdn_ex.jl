# Julia Testing function
# Generate Compressive Sensing Data
using TRNC, Plots,Printf, Convex,SCS, Random, LinearAlgebra

#Here we just try to solve the l2-norm^2 data misfit + l1 norm regularization over the l1 trust region with 0≦x≦1
#######
# min_x 1/2||Ax - b||^2 + λ||x||₁
# s.t. 0≦x≦1
compound = 1
#m rows, n columns, k nonzeros
m,n = compound*120,compound*512
k = compound*20
p = randperm(n)
#initialize x
x0 = zeros(n,)
x0[p[1:k]]=sign.(randn(k))

A = randn(m,n)
# (Q,_) = qr(A')
# A = Q'

b0 = A*x0
b = b0 + 0.5*rand(m,)
cutoff = 0.0;
l = -1*ones(n,)+cutoff*ones(n,)
u = ones(n,)+cutoff*ones(n,)





#define your smooth objective function
function LS(x)
    r = b
    BLAS.gemv!('N',1.0, A, x, -1.0, r)
    f = .5*norm(r)^2
    g = BLAS.gemv('T',A,r)
    h = BLAS.gemm('T', 'N', 1.0, A, A)
    return f, g, h
end

function proxG(z, α)
    return sign.(z).*max(abs.(z).-(α)*ones(size(z)), zeros(size(z)))
end
#do l2 norm for testing purposes
function projq(z,σ)
    return z/max(1, norm(z, 2)/σ)
end
#set all options
#uncomment for OTHER test
first_order_options = s_options(norm(A'*A)^(2.0) ;optTol=1.0e-4, verbose=2, maxIter=100, restart=100)
#note that for the above, default λ=1.0, η=1.0, η_factor=.9
parameters = IP_struct(LS; l=l, u=u, FO_options = first_order_options, s_alg=prox_split_2w, ψk=proxG, χ_projector=projq)
options = IP_options(;simple=0, ptf=10)
#put in your initial guesses
x = (l+u)/2
zl = ones(n,)
zu = ones(n,)

# X = Variable(n)
# problem = minimize(sumsquares(A * X - b) + norm(X,1), X>=l, X<=u)
# solve!(problem, SCSSolver())




x, zl, zu = barrier_alg(x,zl, zu, parameters, options)


#print out l2 norm difference and plot the two x values
@printf("l2-norm TR: %5.5e\n", norm(x - x0))
@printf("l2-norm CVX: %5.5e\n", norm(X.value - x0))
@printf("TR vs CVX relative error: %5.5e\n", norm(X.value - x)/norm(X.value))
# plot(x0, xlabel="i^th index", ylabel="x", title="TR vs True x", label="True x")
# plot!(x, label="tr", marker=2)
# plot!(X.value, label="cvx")
# savefig("xcomp.pdf")

# plot(b0, xlabel="i^th index", ylabel="b", title="TR vs True x", label="True b")
# plot!(b, label="Observed")
# plot!(A*x, label="A*x: TR", marker=2)
# plot!(A*X.value, label="A*x: CVX")
# savefig("bcomp.pdf")
