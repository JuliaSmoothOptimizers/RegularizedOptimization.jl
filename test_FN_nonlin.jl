# Julia Testing function
using TRNC
using LinearAlgebra, DifferentialEquations, Plots, Random, ForwardDiff


#In this example, we demonstrate the capacity of the algorithm to minimize a nonlinear
#model with a regularizer


#Here we solve the Fitzhugh-Nagumo (FHN) Model with some extra terms we know to be zero
#The FHN model is a set of coupled ODE's 
#V' = (f(V) - W + I)/μ for f(V) = V - V^3 / 3
#W' = μ(aV - bW + c) for μ = 0.08,  b = 0.8, c = 0.7

#so we need a model solution, a gradient, and a Hessian of the system (along with some data to fit)
function FH_ODE(dx, x, p, t)
    #p is parameter vector [I,μ, a, b, c]
    V,W = x 
    I, μ, a, b, c = p
    dx[1] = (V - V^3/3 -  W + I)/μ
    dx[2] = μ*(a*V - b*W+c)
end

function FH(x, p, t)
    #p is parameter vector [I,μ, a, b, c]
    dx = zeros(size(x))
    V,W = x 
    I, μ, a, b, c = p
    return [(V - V^3/3 -  W + I)/μ;  μ*(a*V - b*W+c) ]
end

x0 = [2.0; 0.0]
# pars_FH = [.5, 1/12.5, 0.08, 1.0, 0.8, 0.7]
# pars_FH = [0.5, 0.08, 1.0, 0.8, 0.7,]
pars_FH = [0.5, 0.08, 1.0, 0.8, 0.7]
myProbFH = OrdDiffProb(FH, x0, pars_FH; tspan = Array(0.0:.5:100) )
(t, yFH) = rk4Solve(myProbFH; ϵ=1e-4)



plot(t, yFH[:,1], xlabel="Time", ylabel="Voltage", label="V", title="FH sol")
plot!(t, yFH[:,2], label="W")
savefig("figs/nonlin/LS_l1_B2/fhn_basic.pdf")


# #So this is all well and good, but we need a cost function and some parameters to fit. First, we take care of the parameters
# #We start by noting that the FHN model is actually the van-der-pol oscillator with some parameters set to zero
# #x' = μ(x - x^3/3 - y)
# #y' = x/μ -> here μ = 12.5
# #changing the parameters to p = [0, .08, 1.0, 0, 0]
tf = 10
pars_VDP = [0, .2, 1.0, 0, 0]
myProbV = OrdDiffProb(FH, x0, pars_VDP; tspan = Array(0.0:.5:tf) )
(t, yV) = rk4Solve(myProbV;ϵ =1e-4)

# #also make some noise to fit later
b = yV[:,1]
noise = .1*randn(size(b))
b = noise + b

plot(t, yV[:,1], xlabel="Time", ylabel="Voltage", label="V", title="VDP sol")
plot!(t, yV[:,2], label="W")
plot!(t, b, label="V-data")
savefig("figs/nonlin/LS_l1_B2/vdp_basic.pdf")


# #so now that we have data, we want to formulate our optimization problem. This is going to be 
# #min_p ||f(p) - b||₂^2 + λ||p||₀
# #define your smooth objective function
# #First, make the function you are going to manipulate
function f_smooth(x) #gradient and hessian info are smooth parts, m also includes nonsmooth part
    odesolve(p) = sum(rk4Solve(OrdDiffProb(FH, x0, p; tspan = Array(0.0:.5:tf));ϵ = 1e-4)[2][:,1] - b)^2
    
    return odesolve(x), gradient(odesolve, x) #complex step diff, no hessian because that's hard
end

function h_nonsmooth(x)
    return λ*norm(x,1) #, g∈∂h
end

#put in your initial guesses
pi = pars_FH

(~, sens) = f_smooth(pi)

#all this should be unraveling in the hardproxB# code
# fval(s, bq, xi, νi) = (s.+bq).^2/(2*νi) + λ*abs.(s.+xi)
# projbox(y, bq, νi) = min.(max.(y, -bq.-λ*νi),-bq.+λ*νi)
fval(yp, bq, bx, νi) = (yp-bx+bq).^2/(2*νi)+λ*abs.(yp)
projbox(wp, bx, Δi) = min.(max.(wp,bx.-Δi), bx.+Δi)
# fval(u, bq, xi, νi) = (u.+bq).^2/(2*νi) + λ.*(.!iszero.(u.+xi))
# projbox(y, bq, τi) = min.(max.(y, bq.-τi),bq.+τi)
#set all options
λ = 10000.0
Doptions=s_options(eigmax(sens*sens'); maxIter=1000, λ=λ)

parameters = IP_struct(f_smooth, h_nonsmooth;
    FO_options = Doptions, s_alg=hardproxl1B2, InnerFunc=fval, Rk=projbox)
# options = IP_options(;simple=0, ptf=50, Δk = k, epsC=.2, epsD=.2, maxIter=100)
options = IP_options(;simple=0, ptf=1, ϵD = 1e-5)



p, k, Fhist, Hhist = IntPt_TR(pi, parameters, options)

myProbTR = OrdDiffProb(FH, x0, p; tspan = Array(0.0:.5:tf) )
(t, yp) = rk4Solve(myProbFH; ϵ=1e-4)

#print out l2 norm difference and plot the two x values
@printf("l2-norm True vs TR: %5.5e\n", norm(pars_VDP - p)/norm(p))

@printf("Full Objective -  TR: %5.5e    True: %5.5e\n",
f_smooth(p)[1]+h_nonsmooth(p), f_smooth(pars_VDP)[1]+h_nonsmooth(pars_VDP))
@printf("f(x) - CVX: %5.5e     TR: %5.5e    PG: %5.5e   True: %5.5e\n",
,f_smooth(p)[1],  f_smooth(pars_VDP)[1])
@printf("h(x) - CVX: %5.5e     TR: %5.5e    PG: %5.5e    True: %5.5e\n",
h_nonsmooth(p)/λ, h_nonsmooth(pars_VDP)/λ)

plot(t, yV[:,1], xlabel="Time", ylabel="Voltage", label="VDP", title="True vs TR")
plot!(t, yp[:,1], label="tr", marker=2)
plot!(t, b, label="data")
savefig("figs/nonlin/LS_l1_B2/vcomp.pdf")


plot(Fhist, xlabel="k^th index", ylabel="Function Value", title="Objective Value History", label="f(x)")
plot!(Hhist, label="h(x)")
plot!(Fhist + Hhist, label="f+h")
savefig("figs/bpdn/LS_l1_B2/objhist.pdf")
