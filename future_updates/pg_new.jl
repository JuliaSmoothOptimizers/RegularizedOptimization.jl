using LinearAlgebra, Random, Plots
include("pgtest.jl")
# include("pgtest2.jl")
# include("panoc.jl")
# using ProximalAlgorithms, ProximalOperators

compound = 10
T = Float64


m,n = compound*25, compound*64
    p = randperm(n)
    k = compound*2

    #initialize x 
    x0 = zeros(T,n)
    p = randperm(n)[1:k]
    x0[p[1:k]]=sign.(randn(T, k))

    A,_ = qr(randn(T, (n,m)))
    B = Array(A)'
    A = Array(B)

    b0 = A*x0
    

    R = real(T)
    b = b0 + R(.001)*randn(T,m)

    λ = R(0.1)*norm(A'*b, Inf)


    β = eigmax(A'*A)

    function funcF(x)
        r = A*x - b
        g = A'*r
        return norm(r)^2, g
    end
    function funch(x)
        return λ*norm(x,1)
    end

    function proxp(v, α)
        return sign.(v).*max.(abs.(v).-(α*λ)*ones(size(v)), zeros(size(v)))
    end

    problem = GD_problem(funcF, proxp, zeros(T,size(x0)), β, λ)

    setting = GD_setting(verbose = true, tol = 1e-2, maxit = 1000, freq = 1)

    final_state_GD = GD_solver(problem, setting)
    @show norm(final_state_GD.x - x0)

    # solver = ForwardBackward(tol = 1e-15, freq = 1, verbose=true)
    # solver = PANOC(freq = 1, verbose = true)
    # x, it = solver(zeros(size(x0)), f = funcF, g = funch, prox = proxp, L = Float64(opnorm(A)^2))

    # @show norm(x - x0)

    plot(x0)
    # plot!(final_state_GD.x) #compared to other pg, seems right 
    plot!(x)