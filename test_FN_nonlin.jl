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
(t, yFH) = rk4Solve(myProbFH; ϵ=1e-2)



plot(t, yFH[:,1], xlabel="Time", ylabel="Voltage", label="V", title="FH sol")
plot!(t, yFH[:,2], label="W")
savefig("figs/nonlin/fhn_basic.pdf")


# #So this is all well and good, but we need a cost function and some parameters to fit. First, we take care of the parameters
# #We start by noting that the FHN model is actually the van-der-pol oscillator with some parameters set to zero
# #x' = μ(x - x^3/3 - y)
# #y' = x/μ -> here μ = 12.5
# #changing the parameters to p = [0, .08, 1.0, 0, 0]
pars_VDP = [0, .2, 1.0, 0, 0]
myProbV = OrdDiffProb(FH, x0, pars_VDP; tspan = Array(0.0:.5:100) )
(t, yV) = rk4Solve(myProbV;ϵ =1e-2)

# #also make some noie to fit later
b = yV[:,1]
noise = .1*randn(size(b))
b = noise + b

plot(t, yV[:,1], xlabel="Time", ylabel="Voltage", label="V", title="VDP sol")
plot!(t, yV[:,2], label="W")
plot!(t, b, label="V-data")
savefig("figs/nonlin/vdp_basic.pdf")


# #so now that we have data, we want to formulate our optimization problem. This is going to be 
# #min_p ||f(p) - b||₂^2 + λ||p||₀
# #define your smooth objective function
# #First, make the function you are going to manipulate
x0c = x0 + zeros(ComplexF64, size(x0))
function f_smooth(x) #gradient and hessian info are smooth parts, m also includes nonsmooth part
    odesolve(p) = norm(rk4Solve(OrdDiffProb(FH, x0c, p; tspan = Array(0.0:.5:100));ϵ = 1e-2)[2][:,1] - b)^2
    
    return odesolve(x), gradient(odesolve, x) #complex step diff
end

function h_nonsmooth(x)
    return λ*norm(x,0) #, g∈∂h
end
