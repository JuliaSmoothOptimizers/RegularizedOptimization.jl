# Julia Testing function
# Generate Compressive Sensing Data
using TRNC, Plots,Printf, Convex,SCS, Random, LinearAlgebra
include("./src/minconf_spg/oneProjector.jl")

function bpdnNoBarTrBinf()

#Here we just try to solve the l2-norm^2 data misfit + l1 norm regularization over the l1 trust region with -10≦x≦10
#######
# min_x 1/2||Ax - b||^2 + λ||x||₁
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
# l = -2.0*ones(n,)+cutoff*ones(n,)
# u = 2.0*ones(n,)+cutoff*ones(n,)
λ = norm(A'*b, Inf)/10
# λ_O = norm(A'*b, Inf)/2

#define your smooth objective function
#merit function isn't just this though right?
function f_smooth(x) #gradient and hessian info are smooth parts, m also includes nonsmooth part
    r = A*x - b
    g = A'*r
    return norm(r)^2/2, g, A'*A
end

function h_nonsmooth(x)
    return λ*norm(x,1) #, g∈∂h
end

#all this should be unraveling in the hardproxB# code
fval(yp, bq, bx, νi) = (yp-bx+bq).^2/(2*νi)+λ*abs.(yp)
projbox(wp, bx, Δi) = min.(max.(wp,bx.-Δi), bx.+Δi)

#set all options
β = eigmax(A'*A)
# Doptions=s_options(β; maxIter=1000, λ=λ, verbose=1)
Doptions=s_options(β; maxIter=1000, λ=λ, verbose=2)

# first_order_options = s_options(norm(A'*A)^(2.0) ;optTol=1.0e-3, λ=λ_T, verbose=22, maxIter=5, restart=20, η = 1.0, η_factor=.9)
#note that for the above, default λ=1.0, η=1.0, η_factor=.9

parameters = IP_struct(f_smooth, h_nonsmooth;
FO_options = Doptions, s_alg=hardproxl1Binf, InnerFunc=fval, χ_projector=projbox)
# options = IP_options(;simple=0, ptf=50, Δk = k, epsC=.2, epsD=.2, maxIter=100)
options = IP_options(;simple=0, ptf=1)
#put in your initial guesses
xi = ones(n,)/2

X = Variable(n)
problem = minimize(sumsquares(A * X - b) + λ*norm(X,1))
solve!(problem, SCS.Optimizer)

function funcF(x)
    r = A*x - b
    g = A'*r
    return norm(r)^2, g
end
function proxp(z, α)
    return sign.(z).*max.(abs.(z).-(α)*ones(size(z)), zeros(size(z)))
end

(xp, xp⁻, fsave, funEvals) =PG(
    funcF,
    xi,
    proxp,
    Doptions,
)

x, k, Fhist, Hhist = IntPt_TR(xi, parameters, options)


#print out l2 norm difference and plot the two x values
@printf("l2-norm CVX vs TR: %5.5e\n", norm(X.value - x))
@printf("l2-norm CVX vs True: %5.5e\n", norm(X.value - x0)/norm(x0))
@printf("l2-norm TR vs True: %5.5e\n", norm(x0 - x)/norm(x0))
@printf("l2-norm PG vs True: %5.5e\n", norm(x0 - xp)/norm(x0))

@printf("Full Objective - CVX: %5.5e     TR: %5.5e     PG: %5.5e   True: %5.5e\n",
f_smooth(X.value)[1] + h_nonsmooth(X.value), f_smooth(x)[1]+h_nonsmooth(x),f_smooth(xp)[1]+h_nonsmooth(xp), f_smooth(x0)[1]+h_nonsmooth(x0))
@printf("f(x) - CVX: %5.5e     TR: %5.5e    PG: %5.5e   True: %5.5e\n",
f_smooth(X.value)[1],f_smooth(x)[1], f_smooth(xp)[1], f_smooth(x0)[1])
@printf("h(x) - CVX: %5.5e     TR: %5.5e    PG: %5.5e    True: %5.5e\n",
h_nonsmooth(X.value)/λ,h_nonsmooth(x)/λ, h_nonsmooth(xp)/λ, h_nonsmooth(x0)/λ)

plot(x0, xlabel="i^th index", ylabel="x", title="TR vs True x", label="True x")
plot!(x, label="tr", marker=2)
plot!(X.value, label="cvx")
savefig("figs/bpdn/xcomp.pdf")

plot(b0, xlabel="i^th index", ylabel="b", title="TR vs True x", label="True b")
plot!(b, label="Observed")
plot!(A*x, label="A*x: TR", marker=2)
plot!(A*X.value, label="A*x: CVX")
savefig("figs/bpdn/bcomp.pdf")

plot(Fhist, xlabel="k^th index", ylabel="Function Value", title="Objective Value History", label="f(x) (SPGSlim)")
plot!(Hhist, label="h(x)")
plot!(Fhist+ Hhist, label="f+h")
savefig("figs/bpdn/objhist.pdf")
end
