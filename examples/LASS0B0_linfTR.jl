using Random, LinearAlgebra, TRNC, Printf,Roots
using ProximalAlgorithms, ProximalOperators, LinearOperators

# min_x 1/2||Ax - b||^2 + λ||x||₀; ΔB_∞
function B0Binf(compound = 1)
    
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
    #     w = q + xk 
    #     #find largest entries
    #     p = sortperm(abs.(w), rev = true)
    #     w[p[δ+1:end]].=0 #set smallest to zero 
    #     w = ProjB(w)#put all entries in projection?
    #     s = w - xk 

    #     return s 
    # end

    #set options for inner algorithm - only requires ||Bk|| norm guess to start (and λ but that is updated in TR)
    #verbosity is levels: 0 = nothing, 1 -> maxIter % 10, 2 = maxIter % 100, 3+ -> print all 
    β = opnorm(A)^2 #1/||Bk|| for exact Bk = A'*A
    Doptions=s_options(1/β; verbose=0, λ = λ, optTol=1e-16)


    ε = 1e-6
    #define parameters - must feed in smooth, nonsmooth, and λ
    #first order options default ||Bk|| = 1.0, no printing. PG is default inner, Rkprox is inner prox loop - defaults to 2-norm ball projection (not accurate if h=0)
    parameters = TRNCstruct(f_obj, h_obj, λ; FO_options = Doptions, s_alg=PG, ψχprox=prox, χk=(s)->norm(s, Inf), HessApprox = LSR1Operator)
    options = TRNCoptions(; ϵD=ε, verbose = 10, θ = 1e-3, Δk = 1.0) #options, such as printing (same as above), tolerance, γ, σ, τ, w/e
    #put in your initial guesses
    xi = zeros(n,)

    #input initial guess, parameters, options 
    x_pr, k, Fhist, Hhist, Comp_pg = TR(xi, parameters, options)
    #final value, kth iteration, smooth history, nonsmooth history (with λ), # of evaluations in the inner PG loop 


    function proxl0s(q, σ, xk, Δ)
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
        s = st - xk
        return s 
    end

    parametersLM = TRNCstruct(f_obj, h_obj, λ; FO_options = Doptions, s_alg=PGE, ψχprox=proxl0s, χk=(s)->norm(s, Inf), HessApprox = LSR1Operator)
    optionsLM = TRNCoptions(; σk = 1/β, ϵD=ϵ, verbose = 10) #options, such as printing (same as above), tolerance, γ, σ, τ, w/e

    #input initial guess
    xlm, klm, Fhistlm, Hhistlm, Comp_pglm = LM(xi, parametersLM, optionsLM)

    @info "TR relative error" norm(x_pr - x0) / norm(x0)
    @info "LM relative error" norm(xlm - x0) / norm(x0)
    @info "PG relative error" norm(xpg - x0)/norm(x0)
    @info "PANOC relative error" norm(x1 - x0) / norm(x0)
    @info "ZeroFPR relative error" norm(x2 - x0) / norm(x0)
    @info "monotonicity" findall(>(0), diff(Fhist+Hhist))


end