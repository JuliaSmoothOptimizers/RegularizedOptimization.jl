
include("Lin_table.jl")

function bpdntests()
    compound = 1
    m, n = compound * 200, compound * 512
    k = compound * 10
    A = 5 * randn(m, n)
    x0  = rand(n, )
    xi = zeros(size(x0))
    b = A * x0 + .01 * randn(m, )

    function grad!(g, x)
        g .= A'*(A*x - b)
        return g
    end
    ϕt = LBFGSModel(SmoothObj((z) -> norm(A*z-b)^2, grad!, xi))
    λ = 0.0 
    h = NormL1(λ)
    χ = NormL2(1.0)

    # set all options 
    MI = 1000
    TOL = 1e-6
    params = TRNCoptions(; maxIter = MI, verbose = 10, ϵ = TOL, β = 1e16)
    solverp = ProximalAlgorithms.PANOC(tol=TOL, verbose=true, freq=1, maxit=MI)
    solverz = ProximalAlgorithms.ZeroFPR(tol=TOL, verbose=true, freq=1, maxit=MI)

    @info "running LS bfgs, h=0"
    folder = string("figs/ls_bfgs/", compound, "/")


    function h_objmod(x)
        if norm(x, 2) ≤ 100
            h = 0
        else
            h = Inf
        end
        return h 
    end
    function tr_norm(z, σ, x, Δ)
        return z ./ max(1, norm(z, 2) / Δ)
    end

    ϕ = LeastSquaresObjective((z) -> [norm(A * z - b)^2, A' * (A * z - b)], (z) -> 0, 0, [])
    g = ProxOp(h_objmod, (z, σ) -> tr_norm(z, σ, 1, 100), 0)
    _, _ = evalwrapper(x0, xi, A, ϕt, h, ϕ, g, λ, χ, params, solverp, solverz, folder)



    @info "running LS bfgs, h=l1, tr = linf"
    # start bpdn stuff 
    x0 = zeros(n)
    p   = randperm(n)[1:k]
    x0 = zeros(n, )
    x0[p[1:k]] = sign.(randn(k))

    A, _ = qr(randn(n, m))
    B = Array(A)'
    A = Array(B)

    b = A * x0 + .01 * randn(m, )
    λ = norm(A' * b, Inf) / 100 # this can change around 

    function grad!(g, x)
        g .= A'*(A*x - b)
        return g
    end
    ϕt = LSR1Model(SmoothObj((z) -> norm(A*z-b)^2, grad!, xi))
    h = NormL1(λ)
    χ = NormLinf(1.0)
    
    xi = zeros(size(x0))
    folder = string("figs/bpdn/LS_l1_Binf/", compound, "/")


    ϕ = LeastSquaresObjective((z) -> [norm(A * z - b)^2, A' * (A * z - b)], (x)->λ*norm(x, 1), 0, [])
    g = ProxOp((x)->λ*norm(x, 1), (z, α) -> sign.(z) .* max.(abs.(z) .- (λ * α) * ones(size(z)), zeros(size(z))), 0)

    l1binfv, l1binfp = evalwrapper(x0, xi, A, ϕt, h, ϕ, g, λ, χ, params, solverp,solverz, folder)


    @info "running LS bfgs, h=l1, tr = l2"
    xi = zeros(size(x0))
    folder = string("figs/bpdn/LS_l1_B2/", compound, "/")
    ϕt = LSR1Model(SmoothObj((z) -> norm(A*z-b)^2, grad!, xi))
    h = NormL1(λ)
    χ = NormL2(1.0)

    ϕ.count = 0
    ϕ.hist = []
    g.count = 0
    
    l1b2v, l1b2p = evalwrapper(x0, xi, A, ϕt, h, ϕ, g, λ, χ, params, solverp,solverz, folder)


    @info "running LS bfgs, h=l0, tr = linf"
    xi = zeros(size(x0))
    folder = string("figs/bpdn/LS_l0_Binf/", compound, "/")

    ϕt = LSR1Model(SmoothObj((z) -> norm(A*z-b)^2, grad!, xi))
    h = NormL0(λ)
    χ = NormLinf(1.0)

    function proxl0(z, α)
        y = zeros(size(z))
        for i = 1:length(z)
            if abs(z[i]) > sqrt(2 * α * λ)
                y[i] = z[i]
            end
        end
        return y
    end

    ϕ.nonsmooth = (x) -> λ*norm(x, 0)
    ϕ.count = 0
    ϕ.hist = []
    g.count = 0 
    g.func = (x)->λ*norm(x, 0)
    g.proxh = proxl0
    l0binfv, l0binfp = evalwrapper(x0, xi, A, ϕt, h, ϕ, g, λ, χ, params, solverp,solverz, folder)






    @info "running LS bfgs, h=B0, tr = linf"
    λ = k
    xi = zeros(size(x0))
    folder = string("figs/bpdn/LS_B0_Binf/", compound, "/")
    ϕt = LSR1Model(SmoothObj((z) -> norm(A*z-b)^2, grad!, xi))
    h = IndBallL0(λ)
    χ = NormLinf(1.0)

    function h_obj(x)
        if norm(x, 0) ≤ λ
            h = 0
        else
            h = Inf
        end
        return h 
    end

    function proxb0(q, σ)
        # find largest entries
        p = sortperm(abs.(q), rev=true)
        q[p[λ + 1:end]] .= 0 # set smallest to zero 
        return q 
    end


    ϕ.nonsmooth = h_obj 
    ϕ.count = 0
    ϕ.hist = []
    g.count = 0
    g.func = h_obj
    g.proxh = proxb0

    b0binfv, b0binfp = evalwrapper(x0, xi, A, ϕt, h, ϕ, g, λ, χ, params, solverp, solverz, folder)

    toplabs = ["\\(h=\\|\\cdot\\|_1\\), \\(\\Delta\\mathbb{B}_2\\)", "\\(h=\\|\\cdot\\|_0\\), \\(\\Delta\\mathbb{B}_\\infty\\)","\\(h=\\chi(\\cdot; \\lambda \\mathbb{B}_0)\\), \\(\\Delta\\mathbb{B}_\\infty\\)"]
    xlabs = ["True", "TR", "PANOC", "ZFP", "TR", "PANOC", "ZFP", "TR", "PANOC", "ZFP"]

    # pars = [l1b2p, l0binfp, b0binfp]
    vals = hcat(l1b2v, l0binfv[:,2:end], b0binfv[:,2:end])

    df = show_table(toplabs, vals, xlabs)
    _ = write_table(toplabs, df, string("figs/bpdn/", "bpdn-table"))
end