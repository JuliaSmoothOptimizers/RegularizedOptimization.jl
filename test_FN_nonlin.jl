# Julia Testing function
# using TRNC
using LinearAlgebra, DifferentialEquations, Plots, Random, ForwardDiff, Zygote, DiffEqSensitivity


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
    I, μ, a, b, c
    dx[1] = (V - V^3/3 -  W + I)/μ
    dx[2] = μ*(a*x[1] - b*x[2]+c)
end

x0 = [2.0; 0.0]
tspan = (0.0, 100.0)
savetime = .5
# pars_FH = [.5, 1/12.5, 0.08, 1.0, 0.8, 0.7]
# pars_FH = [0.5, 0.08, 1.0, 0.8, 0.7,]
pars_FH = [0.5, 0.08, 1.0, 0.8, 0.7]
prob_FH = ODEProblem(FH_ODE, x0, tspan, pars_FH)
sol_FH = solve(prob_FH, reltol=1e-6, saveat=savetime)
plot(sol_FH, vars=(0,1),xlabel="Time", ylabel="Voltage", label="V", title="FH sol")
plot!(sol_FH, vars=(0,2),label="W")
savefig("figs/nonlin/fhn_basic.pdf")


#So this is all well and good, but we need a cost function and some parameters to fit. First, we take care of the parameters
#We start by noting that the FHN model is actually the van-der-pol oscillator with some parameters set to zero
#x' = μ(x - x^3/3 - y)
#y' = x/μ -> here μ = 12.5
#changing the parameters to p = [0, .08, 1.0, 0, 0]
pars_VDP = [0, .2, 1.0, 0, 0]
prob_VDP = ODEProblem(FH_ODE, x0, tspan, pars_VDP)
sol_VDP = solve(prob_VDP,reltol=1e-6, saveat=savetime)

#also make some noie to fit later
b = hcat(sol_VDP.u...)[1,:]
noise = .1*randn(size(b))
b = noise + b

plot(sol_VDP, vars=(0,1),xlabel="Time", ylabel="Voltage", label="V", title="VDP sol")
plot!(sol_VDP, vars=(0,2),label="W")
plot!(sol_VDP.t, b, label="V-data")
savefig("figs/nonlin/vdp_basic.pdf")


#so now that we have data, we want to formulate our optimization problem. This is going to be 
#min_p ||f(p) - b||₂^2 + λ||p||₀
#define your smooth objective function
#First, make the function you are going to manipulate
function Gradprob(p)
    temp_prob = remake(prob_FH, p = p)
    temp_sol = solve(temp_prob, reltol=1e-6, saveat=savetime)
    temp_v = convert(Array, temp_sol)[1,:]
    return sum((temp_v - b).^2)/2
end
function f_smooth(x) #gradient and hessian info are smooth parts, m also includes nonsmooth part
    odesolve(p::Vector) = norm(hcat(solve(ODEProblem(FH_ODE, x0, tspan,p), reltol=1e-6, saveat=savetime).u...)[1,:] - b)^2
    g = p -> ForwardDiff.gradient(odesolve, p)
    h = ForwardDiff.hessian(odesolve, x)
    return norm(r)^2/2, g(x), h
end

function h_nonsmooth(x)
    return λ*norm(x,0) #, g∈∂h
end

# du01,dp1 = Zygote.gradient((u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,saveat=0.1,sensealg=QuadratureAdjoint())),u0,p)
dp1 = Zygote.gradient((p)->sum(solve(prob_FH,Tsit5(),p=p,saveat=savetime,sensealg=QuadratureAdjoint())),p)