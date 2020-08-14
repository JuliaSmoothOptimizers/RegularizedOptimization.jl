# Julia Testing function
using TRNC
using LinearAlgebra, DifferentialEquations, Plots, Random, Zygote, DiffEqSensitivity, Printf


#In this example, we demonstrate the capacity of the algorithm to minimize a nonlinear
#model with a regularizer


#Here we solve the Fitzhugh-Nagumo (FHN) Model with some extra terms we know to be zero
#The FHN model is a set of coupled ODE's 
#V' = (f(V) - W + I)/μ for f(V) = V - V^3 / 3
#W' = μ(aV - bW + c) for μ = 0.08,  b = 0.8, c = 0.7

#so we need a model solution, a gradient, and a Hessian of the system (along with some data to fit)
function LK(du, u, p, t)
    #p is parameter vector [I,μ, a, b, c]
    du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
    du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
end


u0 = [1.0; 1.0]
tspan = (0.0, 10.0)
savetime = .1
p = [1.5,1.0,3.0,1.0]

prob_LK = ODEProblem(LK, u0, tspan, p)
sol = solve(prob_LK, reltol=1e-6, saveat=savetime)
plot(sol, vars=(0,1),xlabel="Time", ylabel="Species Number", label="Prey", title="LK sol")
plot!(sol, vars=(0,2),label="Pred")


#So this is all well and good, but we need a cost function and some parameters to fit. First, we take care of the parameters
#We start by noting that the FHN model is actually the van-der-pol oscillator with some parameters set to zero
#x' = μ(x - x^3/3 - y)
#y' = x/μ -> here μ = 12.5
#changing the parameters to p = [0, .08, 1.0, 0, 0]
pars_LKs = [1/3,1/9,2/3,1/9]
prob_LKs = remake(prob_LK, p = pars_LKs)
sol_LKs = solve(prob_LKs,reltol=1e-6, saveat=savetime)

#also make some noie to fit later
b = hcat(sol_LKs.u...)
noise = .1*randn(size(b))
data = noise + b

plot(sol_LKs, vars=(0,1),xlabel="Time", ylabel="Species Number", label="Prey", title="LKs sol")
plot!(sol_LKs, vars=(0,2),label="Pred")
plot!(sol_LKs.t, data[1,:], label="Prey-data")
plot!(sol_LKs.t, data[2,:], label="Pred-data")
# savefig("figs/nonlin/LS_l1_B2/vdp_basic.pdf")


#so now that we have data, we want to formulate our optimization problem. This is going to be 
#min_p ||f(p) - b||₂^2 + λ||p||₀
#define your smooth objective function
#First, make the function you are going to manipulate
function Gradprob(p)
    temp_prob = remake(prob_FH, p = p)
    temp_sol = solve(temp_prob, reltol=1e-6, saveat=savetime)
    tot_loss = 0.0
    if any((temp_sol.retcode!= :Success for s in temp_sol))
        tot_loss = Inf
    else
        temp_v = convert(Array, temp_sol)[1,:]
        tot_loss = sum((temp_v - b).^2)/2
    end

    return tot_loss
end
function f_smooth(x) #gradient and hessian info are smooth parts, m also includes nonsmooth part
    return Gradprob(x), Zygote.gradient(Gradprob, x)[1] #complex step diff
end

function h_nonsmooth(x)
    return λ*norm(x,0) #, g∈∂h
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
λ = 100.0
Doptions=s_options(eigmax(sens*sens'); maxIter=1000, λ=λ)


parameters = IP_struct(f_smooth, h_nonsmooth;
    FO_options = Doptions, s_alg=hardproxl1B2, InnerFunc=fval, Rk=projbox)
# options = IP_options(;simple=0, ptf=50, Δk = k, epsC=.2, epsD=.2, maxIter=100)
options = IP_options(;simple=0, ptf=1, ϵD = 1e-5)



p, k, Fhist, Hhist = IntPt_TR(pi, parameters, options)

myProbFH = remake(prob_FH, p = p)
sol = solve(myProbFH; reltol=1e-6, saveat = savetime)

#print out l2 norm difference and plot the two x values
@printf("l2-norm True vs TR: %5.5e\n", norm(pars_VDP - p)/norm(p))

@printf("Full Objective -  TR: %5.5e    True: %5.5e\n",
f_smooth(p)[1]+h_nonsmooth(p), f_smooth(pars_VDP)[1]+h_nonsmooth(pars_VDP))

@printf("f(x) -  TR: %5.5e  True: %5.5e\n",
f_smooth(p)[1],  f_smooth(pars_VDP)[1])

@printf("h(x) -  TR: %5.5e    True: %5.5e\n",
h_nonsmooth(p)/λ, h_nonsmooth(pars_VDP)/λ)

plot(t, yV[:,1], xlabel="Time", ylabel="Voltage", label="VDP", title="True vs TR")
plot!(t, yp[:,1], label="tr", marker=2)
plot!(t, b, label="data")
savefig("figs/nonlin/LS_l1_B2/vcomp.pdf")


plot(Fhist, xlabel="k^th index", ylabel="Function Value", title="Objective Value History", label="f(x)")
plot!(Hhist, label="h(x)")
plot!(Fhist + Hhist, label="f+h")
savefig("figs/bpdn/LS_l1_B2/objhist.pdf")
