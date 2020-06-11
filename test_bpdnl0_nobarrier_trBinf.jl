# Julia Testing function
# Generate Compressive Sensing Data
using TRNC, Plots,Printf, Convex,SCS, Random, LinearAlgebra

function bpdnNoBarTrl0Binf()
#Here we just try to solve the l2-norm^2 data misfit + l1 norm regularization over the l1 trust region with -10≦x≦10
#######
# min_x 1/2||Ax - b||^2 + λ||x||₀
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
b = b0 + 0.005*randn(m,)
cutoff = 0.0
l = -2.0*ones(n,)+cutoff*ones(n,)
u = 2.0*ones(n,)+cutoff*ones(n,)
λ = norm(A'*b, Inf)/100


#define your smooth objective function
#merit function isn't just this though right?
function f_smooth(x) #gradient and hessian info are smooth parts, m also includes nonsmooth part
    r = A*x - b
    g = A'*r
    return norm(r)^2/2, g, A'*A
end

function h_nonsmooth(x)
    return λ*norm(x,0) #, g∈∂h
end

#all this should be unraveling in the hardproxB# code
fval(u, bq, xi, νi) = (u.+bq).^2/(2*νi) + λ.*(.!iszero.(u.+xi))
projbox(y, bq, τi) = min.(max.(y, bq.-τi),bq.+τi)

#set all options
β = eigmax(A'*A)
Doptions=s_options(β; λ=λ)

# first_order_options = s_options(norm(A'*A)^(2.0) ;optTol=1.0e-3, λ=λ_T, verbose=22, maxIter=5, restart=20, η = 1.0, η_factor=.9)
#note that for the above, default λ=1.0, η=1.0, η_factor=.9

parameters = IP_struct(f_smooth, h_nonsmooth; 
    FO_options = Doptions, s_alg=hardproxl0Binf, InnerFunc=fval, χ_projector=projbox)
options = IP_options(;simple=0, ptf=1, ϵD=1e-5)
# options = IP_options(;simple=0, ptf=10, ϵD = 1e-1, ϵC=1e-1, maxIter=100)
#put in your initial guesses
xi = zeros(n,)/2


X = Variable(n)
problem = minimize(sumsquares(A * X - b) + λ*norm(X,1))
solve!(problem, SCS.Optimizer)

# x, k, Fhist, Hhist = IntPt_TR(xi, parameters, options; l = l, u = u, μ = 1.0, BarIter=200)
x, k, Fhist, Hhist = IntPt_TR(xi, parameters, options)


#print out l2 norm difference and plot the two x values
@printf("l2-norm CVX vs TR: %5.5e\n", norm(X.value - x)/norm(X.value))
@printf("l2-norm CVX vs True: %5.5e\n", norm(X.value - x0)/norm(x0))
@printf("l2-norm TR vs True: %5.5e\n", norm(x0 - x)/norm(x0))

@printf("Full Objective - CVX: %5.5e     TR: %5.5e   True: %5.5e\n", f_smooth(X.value)[1] + h_nonsmooth(X.value), f_smooth(x)[1]+h_nonsmooth(x), f_smooth(x0)[1]+h_nonsmooth(x0))
@printf("f(x) - CVX: %5.5e     TR: %5.5e   True: %5.5e\n", f_smooth(X.value)[1],f_smooth(x)[1], f_smooth(x0)[1])
@printf("h(x) - CVX: %5.5e     TR: %5.5e   True: %5.5e\n", h_nonsmooth(X.value)/λ,h_nonsmooth(x)/λ, h_nonsmooth(x0)/λ)

plot(x0, xlabel="i^th index", ylabel="x", title="TR vs True x", label="True x")
plot!(x, label="tr", marker=2)
plot!(X.value, label="cvx")
savefig("figs/bpdn/LS_l0_Binf/xcomp.pdf")

plot(b0, xlabel="i^th index", ylabel="b", title="TR vs True x", label="True b")
plot!(b, label="Observed")
plot!(A*x, label="A*x: TR", marker=2)
plot!(A*X.value, label="A*x: CVX")
savefig("figs/bpdn/LS_l0_Binf/bcomp.pdf")

plot(Fhist, xlabel="k^th index", ylabel="Function Value", title="Objective Value History", label="f(x) (SPGSlim)")
plot!(Hhist, label="h(x)")
plot!(Fhist + Hhist, label="f+h")
savefig("figs/bpdn/LS_l0_Binf/objhist.pdf")
end
