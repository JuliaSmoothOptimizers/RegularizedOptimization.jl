using LinearAlgebra, Random, Plots
include("pgtest.jl")

compound = 10
T = Float32


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
    function proxp(z, α)
        # y = copy(z)
        # for i in eachindex(z)
        #     aux = max(abs(z[i]) - α,0)
        #     y[i] = aux/(aux+α)*z[i]
        # end
        # return y
        return sign.(z).*max.(abs.(z).-(α)*ones(T, size(z)), zeros(T, size(z)))
    end

    problem = GD_problem(funcF, proxp, zeros(T,size(x0)), β, λ)

    setting = GD_setting(verbose = true, tol = 1e-2, maxit = 1000, freq = 1)

    final_state_GD = GD_solver(problem, setting)

    @show norm(final_state_GD.x - x0)

    plot(x0)
    plot!(final_state_GD.x) #compared to other pg, seems right 
    # plot!(x_out)