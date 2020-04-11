# Julia Testing function
using TRNC
using Plots, Convex, SCS, Printf,LinearAlgebra

#Here we just try to solve the l2-norm Problem over the l1 trust region
#######
# min_x 1/2||Ax - b||^2
# min_x f(x)

m,n = 200,100 # this is a under determined system
A = rand(m,n)
x0  = rand(n,)
b0 = A*x0
b = b0 + 0.5*rand(m,)
cutoff = 0.0

function f_obj(x)
    f = .5*norm(A*x-b)^2
    g = A'*(A*x - b)
    h = A'*A #-> BFGS later
    return f, g, h
end

#SPGSlim version
function tr_norm(z,α,σ)
    return z./max(1, norm(z, 2)/σ)
end

function h_obj(x)
    return 0
end

#set all options
# first_order_options = spg_options(;optTol=1.0e-1, progTol=1.0e-10, verbose=0,
    # feasibleInit=true, curvilinear=true, bbType=true, memory=1)
first_order_options = s_options(1/norm(A'*A);optTol=1.0e-1)
    #need to tighten this because you don't make any progress in the later iterations
    #put in projected gradient descent in DescentMethods.jl to solve in the inner loop with relative accuracy


# Interior Pt Algorithm
parameters = IP_struct(f_obj, h_obj;s_alg = PG, FO_options = first_order_options, χ_projector=tr_norm) #defaults to h=0, spgl1/min_confSPG
options = IP_options(;ptf=1) #print freq, ΔK init, epsC/epsD initialization, maxIter
#put in your initial guesses
xi = ones(n,)/2

X = Variable(n)
problem = minimize(sumsquares(A * X - b))
solve!(problem, SCSSolver())

TotalCount = 0


# x, zl, zu = barrier_alg(xi,zl, zu, parameters, options; is_cvx=0, mu_tol=1e-3)
# x, zl, zu, k = IntPt_TR(x, zl, zu,mu,IterCount, IPparams, IPoptions)
x, k = IntPt_TR(xi, TotalCount, parameters, options)


#print out l2 norm difference and plot the two x values
@printf("l2-norm TR: %5.5e\n", norm(x - x0))
@printf("l2-norm CVX: %5.5e\n", norm(X.value - x0))
@printf("TR vs CVX relative error: %5.5e\n", norm(X.value - x)/norm(X.value))
plot(x0, xlabel="i^th index", ylabel="x", title="TR vs True x", label="True x")
plot!(x, label="tr", marker=2)
plot!(X.value, label="cvx")
savefig("figs/ls/xcomp.pdf")

plot(b0, xlabel="i^th index", ylabel="b", title="TR vs True x", label="True b")
plot!(b, label="Observed")
plot!(A*x, label="A*x: TR", marker=2)
plot!(A*X.value, label="A*x: CVX")
savefig("figs/ls/bcomp.pdf")
