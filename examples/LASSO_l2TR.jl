using Random, LinearAlgebra, TRNC

# min_x 1/2||Ax - b||^2 + λ||x||₁; ΔB_1
function L1B2(compound=1)
    m, n = compound * 200, compound * 512 # if you want to rapidly change problem size 
    k = compound * 10 # 10 signals 
    α = .01 # noise level 

    # start bpdn stuff 
    x0 = zeros(n)
    p   = randperm(n)[1:k]
    x0 = zeros(n, )
    x0[p[1:k]] = sign.(randn(k)) # create sparse signal 

    A, _ = qr(randn(n, m))
    B = Array(A)'
    A = Array(B)

    b0 = A * x0
    b = b0 + α * randn(m, )


    λ = norm(A' * b, Inf) / 10 # this can change around 

    ϕ = LSR1Model(SmoothObj((x) -> .5*norm(A*x - b)^2, (x) -> A'*(A*x - b), xi))
    h = NormL1(λ)
    ε = 1e-6
    # define parameters - must feed in smooth, nonsmooth, and λ
    # first order options default ||Bk|| = 1.0, no printing. PG is default inner, Rkprox is inner prox loop - defaults to 2-norm ball projection (not accurate if h=0)
    parameters = TRNCmethods(; FO_options = Doptions, s_alg=PGnew, χk=(s)->norm(s, Inf)) 
    optionstr = TRNCoptions(; ϵ = ε, verbose=10, maxIter=100) # options, such as printing (same as above), tolerance, γ, σ, τ, w/e

    # put in your initial guesses
    xi = zeros(n,)

    # input initial guess, parameters, options 
    xtr, ktr, Fhisttr, Hhisttr, Comp_pgtr = TR(ϕ, h, parameterstr, optionstr)

   
    optionsQR = TRNCoptions(; σk=1 / β, ϵD=ϵ, verbose=10) # options, such as printing (same as above), tolerance, γ, σ, τ, w/e
    # put in your initial guesses
    xi = ones(n, ) / 2

    # input initial guess, parameters, options 
    xqr, kqr, Fhistqr, Hhistqr, Comp_pgqr = QuadReg(xi, parametersQR, optionsQR)


    @info "TR relative error" norm(xtr - x0) / norm(x0)
    @info "QR relative error" norm(xqr - x0) / norm(x0)
    @info "monotonicity" findall(>(0), diff(Fhisttr + Hhisttr))

end