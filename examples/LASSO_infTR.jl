using Random, LinearAlgebra, TRNC, Printf,Roots
include("../test/minconf_spg/SPGSlim.jl")
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
# λ = norm(A'*b, Inf)/10

#define your smooth objective function
function f_obj(x) #gradient and hessian info are smooth parts, m also includes nonsmooth part
    r = A*x - b
    g = A'*r
    return norm(r)^2/2, g, I(size(x,1))
end

function h_obj(x)
    return norm(x,1) #without  lambda
end

#combination l1 and Binf prox
function prox(q, σ, xk, Δ)
    Fcn(yp) = (yp-xk-q).^2/2+σ*abs.(yp)
    ProjB(wp) = min.(max.(wp,xk.-Δ), xk.+Δ)
    
    y1 = zeros(size(xk))
    f1 = Fcn(y1)
    idx = (y1.<xk.-Δ) .| (y1.>xk .+ Δ) #actually do outward since more efficient
    f1[idx] .= Inf

    y2 = ProjB(xk+q.-σ)
    f2 = Fcn(y2)
    y3 = ProjB(xk+q.+σ)
    f3 = Fcn(y3)

    smat = hcat(y1, y2, y3) #to get dimensions right
    fvec = hcat(f1, f2, f3)

    f = minimum(fvec, dims=2)
    idx = argmin(fvec, dims=2)
    s = smat[idx]-xk

    return dropdims(s, dims=2)
end

#set options for inner algorithm - only requires ||Bk|| norm guess to start (and λ but that is updated in IP_alg)
#verbosity is levels: 0 = nothing, 1 -> maxIter % 10, 2 = maxIter % 100, 3+ -> print all 
β = eigmax(A'*A) #1/||Bk|| for exact Bk = A'*A
Doptions=s_options(1/β; maxIter=10000, verbose=0, λ = λ, optTol=1e-16)

ϵ = 1e-6

#define parameters - must feed in smooth, nonsmooth, and λ
#first order options default ||Bk|| = 1.0, no printing. PG is default inner, Rkprox is inner prox loop - defaults to 2-norm ball projection (not accurate if h=0)
parameters = IP_struct(f_obj, h_obj, λ; FO_options = Doptions, s_alg=PG, Rkprox=prox)
options = IP_options(; ϵD=ϵ, verbose = 10) #options, such as printing (same as above), tolerance, γ, σ, τ, w/e
#put in your initial guesses
xi = ones(n,)/2


#input initial guess, parameters, options 
x_pr, k, Fhist, Hhist, Comp_pg = IntPt_TR(xi, parameters, options)
#final value, kth iteration, smooth history, nonsmooth history (with λ), # of evaluations in the inner PG loop 


Foptions=s_options(1/β; maxIter=100000, verbose=1, λ = λ, optTol=ϵ)

#If you want to test PG 
function funcF(x)
    r = A*x - b
    g = A'*r
    return norm(r)^2, g
end
function proxp(z, α)
    return sign.(z).*max.(abs.(z).-(λ*α)*ones(size(z)), zeros(size(z)))
end

Foptions.verbose = 2 #print every 100 
Foptions.ν = 1/β #guess exact step size 
xpg, xpg⁻, histpg, fevals = PG(funcF, h_obj, xi, proxp, Foptions) #takes in smooth, nonsmooth, initial guess, prox, options (with λ)
#output final, secont to last, total function history, number of evals 
# popt = spg_options(;optTol=ϵ, progTol=1.0e-16, verbose=10, memory=10, maxIter = 10000)
# funproj(d, σ) = proxp(d, σ)
# (xpg, fsave, funEvals,_,histpg) = minConf_SPG(funcF, xi, funproj,h_obj, popt)
