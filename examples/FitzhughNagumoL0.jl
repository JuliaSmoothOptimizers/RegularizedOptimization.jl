using DifferentialEquations, Zygote, DiffEqSensitivity
using Random, LinearAlgebra, TRNC, Printf,Roots, Plots
using ProximalOperators, ProximalAlgorithms, LinearOperators


#so we need a model solution, a gradient, and a Hessian of the system (along with some data to fit)
function FH_ODE(dx, x, p, t)
    #p is parameter vector [I,μ, a, b, c]
    V,W = x 
    I, μ, a, b, c = p
    dx[1] = (V - V^3/3 -  W + I)/μ
    dx[2] = μ*(a*V - b*W+c)
end


u0 = [2.0; 0.0]
tspan = (0.0, 20.0)
savetime = .2

pars_FH = [0.5, 0.08, 1.0, 0.8, 0.7]
prob_FH = ODEProblem(FH_ODE, u0, tspan, pars_FH)


#So this is all well and good, but we need a cost function and some parameters to fit. First, we take care of the parameters
#We start by noting that the FHN model is actually the van-der-pol oscillator with some parameters set to zero
#x' = μ(x - x^3/3 - y)
#y' = x/μ -> here μ = 12.5
#changing the parameters to p = [0, .08, 1.0, 0, 0]
x0 = [0, .2, 1.0, 0, 0]
prob_VDP = ODEProblem(FH_ODE, u0, tspan, x0)
sol_VDP = solve(prob_VDP,reltol=1e-6, saveat=savetime)


#also make some noie to fit later
t = sol_VDP.t
b = hcat(sol_VDP.u...)
noise = .1*randn(size(b))
data = noise + b

#so now that we have data, we want to formulate our optimization problem. This is going to be 
#min_p ||f(p) - b||₂^2 + λ||p||₀
#define your smooth objective function
#First, make the function you are going to manipulate
function Gradprob(p)
    temp_prob = remake(prob_FH, p = p)
    temp_sol = solve(temp_prob, reltol=1e-6, saveat=savetime, verbose=false)
    tot_loss = 0.

    if any((temp_sol.retcode!= :Success for s in temp_sol))
        tot_loss = Inf
    else
        temp_v = convert(Array, temp_sol)

        tot_loss = sum((temp_v - data).^2)/2
    end

    return tot_loss
end
function f_obj(x) #gradient and hessian info are smooth parts, m also includes nonsmooth part
    fk = Gradprob(x)
    if fk==Inf 
        grad = zeros(size(x))
        # Hess = Inf*ones(size(x,1), size(x,1))
    else
        grad = Zygote.gradient(Gradprob, x)[1] 
        # Hess = Zygote.hessian(Gradprob, x)
    end
    return fk, grad
end


λ = 1.0
δ = 2
function h_obj(x)
    return norm(x,0) 
end


#put in your initial guesses
xi = ones(size(pars_FH))

# (_, _, Hessapprox) = f_obj(xi)
# β = eigmax(Hessapprox)
β= 1e6
#this is for l0 norm 
function prox(q, σ, xk, Δ)

    ProjB(y) = min.(max.(y, xk.-Δ),xk.+Δ) # define outside? 
    # @show σ/λ, λ
    c = sqrt(2*σ)
    w = xk+q
    st = zeros(size(w))

    for i = 1:length(w)
        absx = abs(w[i])
        if absx <=c
            st[i] = 0
        else
            st[i] = w[i]
        end
    end
    s = ProjB(st) - xk
    return s 
end

# function h_obj(x)
#     if norm(x,0) ≤ δ
#         h = 0
#     else
#         h = Inf
#     end
#     return h 
# end

# function prox(q, σ, xk, Δ)
#     ProjB(w) = min.(max.(w, xk.-Δ), xk.+Δ)
#     y = q + xk 
#     #find largest entries
#     p = sortperm(abs.(y), rev = true)
#     y[p[δ+1:end]].=0 #set smallest to zero 
#     y = ProjB(y)#put largest entries in projection
#     s = y - xk 

#     return s 
# end


ϵ = 1e-3
#set all options
Doptions=s_options(1.0; λ=λ, optTol = ϵ*(1e-6), verbose = 0)

params= IP_struct(f_obj, h_obj, λ; FO_options = Doptions, s_alg=PG, Rkprox=prox, HessApprox = LSR1Operator)

options = IP_options(; maxIter = 500, verbose=10, ϵD = ϵ)
#solve our problem 
function funcF(x)
    fk = Gradprob(x)
    # @show fk
    if fk==Inf 
        grad = Inf*ones(size(x))
    else
        grad = Zygote.gradient(Gradprob, x)[1] 
    end

    return fk, grad
end



x_pr, k, Fhist, Hhist, Comp_pg = IntPt_TR(xi, params, options)

T = Float64
R = real(T)
TOL = R(ϵ*1e-6)
g = NormL0(λ)


mutable struct LeastSquaresObjective
    nonlin
    b
end
  
ϕ = LeastSquaresObjective(funcF, b)
  
# definition that will allow to simply evaluate the objective: ϕ(x)
function (f::LeastSquaresObjective)(x)
    r = f.nonlin(x)[1]
    return r
end

# first import gradient and gradient! to extend them
import ProximalOperators.gradient, ProximalOperators.gradient!

function gradient!(∇fx, f::LeastSquaresObjective, x)
    r, g = f.nonlin(x)
    ∇fx .= g
    return r
end

function gradient(f::LeastSquaresObjective, x)
    ∇fx = similar(x)
    fx = gradient!(∇fx, f, x)
    return ∇fx, fx
end

# PANOC checks if the objective is quadratic
import ProximalOperators.is_quadratic
is_quadratic(::LeastSquaresObjective) = true



import ProximalAlgorithms: LBFGS, Maybe, PANOC, PANOC_iterable, PANOC_state
import ProximalAlgorithms.IterationTools: halt, sample, tee, loop
using Base.Iterators  # for take
function my_panoc(solver::PANOC{R},
        x0::AbstractArray{C};
        f = Zero(),
        A = I,
        g = Zero(),
        L::Maybe{R} = nothing,
        Fhist = zeros(0),
        Hhist = zeros(0),) where {R,C<:Union{R,Complex{R}}}
    stop(state::PANOC_state) = norm(state.res, Inf) / state.gamma <= solver.tol
    function disp((it, state))
        append!(Fhist, state.f_Ax)
        append!(Hhist, state.g_z)
        @printf(
                "%5d | %.3e | %.3e | %.3e | %9.2e | %9.2e\n",
                it,
                state.gamma,
                norm(state.res, Inf) / state.gamma,
                (state.tau === nothing ? 0.0 : state.tau),
                state.f_Ax,  # <-- added this
                state.g_z    # <-- and this
            )
    end
    # disp((it, state)) = @printf(
    # 	"%5d | %.3e | %.3e | %.3e | %9.2e | %9.2e\n",
    # 	it,
    # 	state.gamma,
    # 	norm(state.res, Inf) / state.gamma,
    # 	(state.tau === nothing ? 0.0 : state.tau),
    # 	state.f_Ax,  # <-- added this
    # 	state.g_z    # <-- and this
    # )

    gamma = if solver.gamma === nothing && L !== nothing
        solver.alpha / L
    else
        solver.gamma
    end

    iter = PANOC_iterable(
        f,
        A,
        g,
        x0,
        solver.alpha,
        solver.beta,
        gamma,
        solver.adaptive,
        LBFGS(x0, solver.memory),
    )
    iter = take(halt(iter, stop), solver.maxit)
    iter = enumerate(iter)
    if solver.verbose
        iter = tee(sample(iter, solver.freq), disp)
    end

    num_iters, state_final = loop(iter)
    if isinf(sum(state_final.z))
        x = state_final.x
    else
        x = state_final.z
    end

    return x, num_iters, Fhist, Hhist
end
# 4. call PANOC again with our own objective
@info "running PANOC with our own objective"
solver = ProximalAlgorithms.PANOC{R}(tol = TOL, verbose=true, freq=1, maxit=5000)
x2, it, PFp, PHp = my_panoc(solver, xi, f = ϕ, g = g)





import ProximalAlgorithms: LBFGS, Maybe, ZeroFPR, ZeroFPR_iterable, ZeroFPR_state


function my_zerofpr(solver::ZeroFPR{R},
    x0::AbstractArray{C};
    f = Zero(),
    A = I,
    g = Zero(),
    L::Maybe{R} = nothing,
    Fhist = zeros(0),
    Hhist = zeros(0),) where {R,C<:Union{R,Complex{R}}}
    stop(state::ZeroFPR_state) = norm(state.res, Inf) / state.gamma <= solver.tol
    function disp((it, state))
        append!(Fhist, state.f_Ax)
        append!(Hhist, state.g_xbar)
        @printf(
                "%5d | %.3e | %.3e | %.3e | %9.2e | %9.2e\n",
                it,
                state.gamma,
                norm(state.res, Inf) / state.gamma,
                (state.tau === nothing ? 0.0 : state.tau),
                state.f_Ax,  # <-- added this
                state.g_xbar    # <-- and this
            )
    end

    gamma = if solver.gamma === nothing && L !== nothing
        solver.alpha / L
    else
        solver.gamma
    end

    iter = ZeroFPR_iterable(
        f,
        A,
        g,
        x0,
        solver.alpha,
        solver.beta,
        gamma,
        solver.adaptive,
        LBFGS(x0, solver.memory),
    )
    iter = take(halt(iter, stop), solver.maxit)
    iter = enumerate(iter)
    if solver.verbose
        iter = tee(sample(iter, solver.freq), disp)
    end

    num_iters, state_final = loop(iter)

    return state_final.x,state_final.xbar, state_final.y, num_iters, Fhist, Hhist
end

@info "running ZeroFPR with our own objective"
xi = ones(size(pars_FH))
solver = ProximalAlgorithms.ZeroFPR{R}(tol = TOL, verbose=true, freq=1, maxit=5000)
x3, xbar, y, it, PF, PH = my_zerofpr(solver, xi, f = ϕ, g = g)



@info "TR relative error" norm(x_pr - x0) / norm(x0)
@info "PANOC relative error" norm(x2 - x0) / norm(x0)
@info "ZeroFPR relative error" norm(x3 - x0) / norm(x0)
@info "monotonicity" findall(>(0), diff(Fhist+Hhist))

@show x_pr
@show x2
@show x3 
@show x0'


include("../test/nonlinfig_gen.jl")
folder = "./"
hist = [Fhist+Hhist,PFp+PHp, PF+PH]
histx = [Array(1:length(Fhist)), Array(1:length(PFp)), Array(1:20:length(PF)*20)] 
labs = ["f+h: TR", "f+h: PANOC", "f+h: ZeroFPR"]
figen_non(histx, hist, labs, string(folder,"objcomp"), [" ", "kth Objective Evaluation", " Value "], 3, 0)




probx = remake(prob_FH, p = x_pr)
temp_solx = solve(probx, reltol=1e-6, saveat=savetime)
probx = remake(prob_FH, p = x2)
temp_solp = solve(probx, reltol=1e-6, saveat=savetime)
probx = remake(prob_FH, p = x3)
temp_solf = solve(probx, reltol=1e-6, saveat=savetime)


#print out l2 norm difference and plot the two x values
sol = hcat(sol_VDP.u...)
solx = hcat(temp_solx.u...)
solp = hcat(temp_solp.u...)
solf = hcat(temp_solf.u...)

yvars = [sol[1,:], sol[2,:], solx[1,:], solx[2,:], solp[1,:], solp[2,:],solf[1,:], solf[2,:], data[1,:], data[2,:]]
xvars = [t, t, t, t,t,t, t, t, t, t]
labs = ["True-V", "True-W", "TR-V", "TR-W", "PANOC-V", "PANOC-W","ZFP-V", "ZFP-W", "Data-V", "Data-W"]
figen_non(xvars, yvars, labs, string(folder, "xcomp"), [" ", "Time", "Voltage"],2, 1)
