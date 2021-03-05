using Random, LinearAlgebra, TRNC, Printf,Roots
using ProximalOperators, ProximalAlgorithms
using LinearOperators
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
    return norm(x,1) #without  lambda
end

function prox(q, σ, xk, Δ)
    #------ old ----
    ProjB(wp) = min.(max.(wp,q.-σ), q.+σ)
    ProjΔ(yp) = min.(max.(yp, -Δ), Δ)
    s = ProjΔ(ProjB(-xk))
    #---- new ---
    # Proxh(wp) = max.(abs.(wp) .-σ, zeros(size(wp))).*sign.(wp)
    # ProjΔ(yp) = min.(max.(yp, xk.-Δ), xk.+Δ)
    # s = ProjΔ(Proxh(q+xk)) - xk 

    #----- test -----
    # ProjB(wp) = min.(max.(wp, q.-σ), q.+σ)
    # ProjΔ(yp) = min.(max.(yp, -Δ), Δ)
    # s = ProjΔ(ProjB(-xk))

    return s
end

#set options for inner algorithm - only requires ||Bk|| norm guess to start (and λ but that is updated in TR)
#verbosity is levels: 0 = nothing, 1 -> maxIter % 10, 2 = maxIter % 100, 3+ -> print all 
β = eigmax(A'*A) #1/||Bk|| for exact Bk = A'*A
Doptions=s_options(1/β; verbose=0, λ = λ, optTol=1e-16, p = 1.5)

ϵ = 1e-6

#define parameters - must feed in smooth, nonsmooth, and λ
#first order options default ||Bk|| = 1.0, no printing. PG is default inner, Rkprox is inner prox loop - defaults to 2-norm ball projection (not accurate if h=0)
parameterstr = TRNCstruct(f_obj, h_obj, λ; FO_options = Doptions, s_alg=PGE, χk=(s)->norm(s, Inf), ψχprox=prox)#, HessApprox = LSR1Operator)
optionstr = TRNCoptions(; ϵD=ϵ, verbose = 10, maxIter = 100) #options, such as printing (same as above), tolerance, γ, σ, τ, w/e
#put in your initial guesses
xi = ones(n,)/2


#input initial guess, parameters, options 
xtr, ktr, Fhisttr, Hhisttr, Comp_pgtr = TR(xi, parameterstr, optionstr)




function proxl1s(q, νλ,  xk, Δ)
    # return sign.(q).*max.(abs.(q+xk).-(νλ)*ones(size(q)), zeros(size(q))) - xk
    ProjB(wp) = min.(max.(wp,q.-νλ), q.+νλ)
    return ProjB(-xk) 
end



parametersLM = TRNCstruct(f_obj, h_obj, λ; FO_options = Doptions, s_alg=PGE, ψχprox=proxl1s)#, HessApprox = LSR1Operator)
optionsLM = TRNCoptions(; σk = 1/β, ϵD=ϵ, verbose = 10) #options, such as printing (same as above), tolerance, γ, σ, τ, w/e
#put in your initial guesses
xi = ones(n,)/2

#input initial guess, parameters, options 
xlm, klm, Fhistlm, Hhistlm, Comp_pglm = LM(xi, parametersLM, optionsLM)
Doptions.verbose = 10; 
Doptions.ν = 1/β
Doptions.optTol = ϵ
function f_obj2(x) #gradient and hessian info are smooth parts, m also includes nonsmooth part
    r = A*x - b
    g = A'*r
    return norm(r)^2/2, g, A'*A
end
function proxl1(q, νλ,  xk, Δ)
    return sign.(q).*max.(abs.(q).-(νλ)*ones(size(q)), zeros(size(q)))
end
xpg, _, _, _ = FISTA(f_obj2, h_obj, xi,  (x, νλ)->proxl1(x, νλ, xi, 1.0), Doptions)
#final value, kth iteration, smooth history, nonsmooth history (with λ), # of evaluations in the inner PG loop 

T = Float64
R = real(T)
f = Translate(SqrNormL2(R(1)), -b)
g = NormL1(λ)

TOL = R(ϵ)
solver = ProximalAlgorithms.PANOC{R}(tol = TOL, verbose=true, freq=1)
@info "running standard PANOC"
xi = ones(n,)/2
# the argument A changes f(x) to f(Ax), i.e., 1/2 ‖Ax - b‖²
x1, it = solver(xi, f = f, A = A, g = g, L = opnorm(A)^2)  # watch out; for some reason, this changes x0


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

# 4. call PANOC again with our own objective
@info "running ZeroFPR with our own objective"
solver = ProximalAlgorithms.ZeroFPR{R}(tol = TOL, verbose=true, freq=1)
xi = ones(n,)/2
x2, it = solver(xi, f = ϕ, g = g, L = opnorm(A)^2)


@info "TR relative error" norm(xtr - x0) / norm(x0)
@info "LM relative error" norm(xlm - x0) / norm(x0)
@info "PG relative error" norm(xpg - x0) / norm(x0)
@info "PANOC relative error" norm(x1 - x0) / norm(x0)
@info "ZeroFPR relative error" norm(x2 - x0) / norm(x0)
@info "monotonicity" findall(>(0), diff(Fhisttr+Hhisttr))