using Random, LinearAlgebra, TRNC, Printf,Roots
using ProximalAlgorithms, ProximalOperators, LinearOperators

# min_x 1/2||Ax - b||^2 + λ||x||₁
compound = 1
m,n = compound*200,compound*512 #if you want to rapidly change problem size 
k = compound*10 #10 signals 
α = .01 #noise level 

#start bpdn stuff 
x0 = zeros(n)
p   = randperm(n)[1:k]
x0 = zeros(n,)
x0[p[1:k]]=sign.(randn(k)) #create sparse signal 

A,_ = qr(randn(n,m))
B = Array(A)'
A = Array(B)

b0 = A*x0
b = b0 + α*randn(m,)


λ = norm(A'*b, Inf)/10 #this can change around 

#define your smooth objective function
function f_obj(x) #gradient and hessian info are smooth parts, m also includes nonsmooth part
    r = A*x - b
    g = A'*r
    return norm(r)^2/2, g
end

function h_obj(x)
    return norm(x,0) #, g∈∂h
end

function  prox(q, σ, xk, Δ)
    ProjB(z) = min.(max.(z, -Δ), Δ) # define outside? 
    w = xk + q 
    c1 = (1/(2*σ/λ)).*w.^2
    c2 = λ .+ ((max.(abs.(q).-Δ, zeros(size(w)) )).^2)/(2*σ/λ)
    y = zeros(size(w))
    for i = 1:length(w)
        if c1[i] ≤ c2[i] && abs(xk[i])≤ Δ
            y[i] = 0
        else
            y[i] = xk[i] + ProjB(q[i])
        end
    end
    return y - xk 
end

# function prox(q, σ, xk, Δ)
#     ProjB(y) = min.(max.(y, xk.-Δ),xk.+Δ) # define outside? 
#     # @show σ/λ, λ
#     c = sqrt(2*σ)
#     w = xk+q
#     st = zeros(size(w))

#     for i = 1:length(w)
#         absx = abs(w[i])
#         if absx <=c
#             st[i] = 0
#         else
#             st[i] = w[i]
#         end
#     end
#     s = ProjB(st) - xk
#     return s 
# end
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
#     w = q + xk 
#     #find largest entries
#     p = sortperm(abs.(w), rev = true)
#     w[p[δ+1:end]].=0 #set smallest to zero 
#     w = ProjB(w)#put all entries in projection?
#     s = w - xk 

#     return s 
# end

#set options for inner algorithm - only requires ||Bk|| norm guess to start (and λ but that is updated in IP_alg)
#verbosity is levels: 0 = nothing, 1 -> maxIter % 10, 2 = maxIter % 100, 3+ -> print all 
β = opnorm(A)^2 #1/||Bk|| for exact Bk = A'*A
Doptions=s_options(1/β; verbose=0, λ = λ, optTol=1e-16)


ε = 1e-6
#define parameters - must feed in smooth, nonsmooth, and λ
#first order options default ||Bk|| = 1.0, no printing. PG is default inner, Rkprox is inner prox loop - defaults to 2-norm ball projection (not accurate if h=0)
parameters = IP_struct(f_obj, h_obj, λ; FO_options = Doptions, s_alg=FISTA, Rkprox=prox)#, HessApprox = LSR1Operator)
options = IP_options(; ϵD=ε, verbose = 10) #options, such as printing (same as above), tolerance, γ, σ, τ, w/e
#put in your initial guesses
xi = zeros(n,)

#input initial guess, parameters, options 
x_pr, k, Fhist, Hhist, Comp_pg = IntPt_TR(xi, parameters, options)
#final value, kth iteration, smooth history, nonsmooth history (with λ), # of evaluations in the inner PG loop 




#If you want to test PG 
function funcF(x)
    r = A*x - b
    g = A'*r
    return norm(r)^2, g
end


function proxp(z, α)
    y = zeros(size(z))
    for i = 1:length(z)
        if abs(z[i])>sqrt(2*α)
            y[i] = z[i]
        end
    end
    return y
end

Doptions.verbose = 2 #print every 100 
Doptions.ν = 1/β #guess exact step size 
xpg, xpg⁻, histpg, fevals = PG(funcF, h_obj, xi, proxp, Doptions) #takes in smooth, nonsmooth, initial guess, prox, options (with λ)
#output final, secont to last, total function history, number of evals 

T = Float64
R = real(T)
g = NormL0(λ)
TOL = R(ε)

mutable struct LeastSquaresObjective
    A
    b
end
  
ϕ = LeastSquaresObjective(A, b)
  
# definition that will allow to simply evaluate the objective: ϕ(x)
function (f::LeastSquaresObjective)(x)
    r = f.A * x - f.b
    return dot(r, r) / 2
end

# first import gradient and gradient! to extend them
import ProximalOperators.gradient, ProximalOperators.gradient!

function gradient!(∇fx, f::LeastSquaresObjective, x)
    r = f.A * x - f.b
    ∇fx .= f.A' * r
    return dot(r, r) / 2
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
x1, it, PF, PH = my_panoc(solver, xi, f = ϕ, g = g)





# 4. call PANOC again with our own objective
@info "running ZeroFPR with our own objective"
solver = ProximalAlgorithms.ZeroFPR{R}(tol = TOL, verbose=true, freq=1)
xi = zeros(n,)
x2, it = solver(xi, f = ϕ, g = g, L = opnorm(A)^2)


@info "TR relative error" norm(x_pr - x0) / norm(x0)
@info "PG relative error" norm(xpg - x0)/norm(x0)
@info "PANOC relative error" norm(x1 - x0) / norm(x0)
@info "ZeroFPR relative error" norm(x2 - x0) / norm(x0)
@info "monotonicity" findall(>(0), diff(Fhist+Hhist))