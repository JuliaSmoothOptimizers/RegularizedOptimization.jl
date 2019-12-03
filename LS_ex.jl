# Julia Testing function
using TRNC
using Plots, Convex, SCS, Printf,LinearAlgebra

#Here we just try to solve the l2-norm Problem over the l1 trust region
#######
# min_x 1/2||Ax - b||^2 st 0⩽x⩽1


m,n = 200,100 # this is a under determined system
# m, n = 10, 2
A = rand(m,n)
x0  = rand(n,)
b0 = A*x0
b = b0 + 0.5*rand(m,)
cutoff = 0.0
l = zeros(n,)+cutoff*ones(n,)
u = ones(n,)+cutoff*ones(n,)





# function f_obj(x) #gradient and hessian info are smooth parts, m also includes nonsmooth part
#     r = b
#     BLAS.gemv!('N',1.0, A, x, -1.0, r)
#     f = .5*norm(r)^2
#     g = BLAS.gemv('T',A,r)
#     h = BLAS.gemm('T', 'N', 1.0, A, A)
#     return f, g, h
# end
function f_obj(x)
    f = .5*norm(A*x-b)^2
    g = A'*(A*x - b)
    h = A'*A
    return f, g, h
end

function h_obj(x)
    return 0
end
#set all options
first_order_options = spg_options(;optTol=1.0e-2, progTol=1.0e-10, verbose=0,
    feasibleInit=true, curvilinear=true, bbType=true, memory=1)
parameters = IP_struct(f_obj, h_obj; l=l, u=u, FO_options = first_order_options)
options = IP_options(;ptf=1)
#put in your initial guesses
xi = (l+u)/2
zl = ones(n,)
zu = ones(n,)

X = Variable(n)
problem = minimize(sumsquares(A * X - b), X>=l, X<=u)
solve!(problem, SCSSolver())




x, zl, zu = barrier_alg(xi,zl, zu, parameters, options; is_cvx=1)


#print out l2 norm difference and plot the two x values
@printf("l2-norm TR: %5.5e\n", norm(x - x0))
@printf("l2-norm CVX: %5.5e\n", norm(X.value - x0))
@printf("TR vs CVX relative error: %5.5e\n", norm(X.value - x)/norm(X.value))
plot(x0, xlabel="i^th index", ylabel="x", title="TR vs True x", label="True x")
plot!(x, label="tr", marker=2)
plot!(X.value, label="cvx")
savefig("xcomp.pdf")

plot(b0, xlabel="i^th index", ylabel="b", title="TR vs True x", label="True b")
plot!(b, label="Observed")
plot!(A*x, label="A*x: TR", marker=2)
plot!(A*X.value, label="A*x: CVX")
savefig("bcomp.pdf")
