@testset "Lasso ($T) Descent Methods" for T in [Float32, Float64, ComplexF32, ComplexF64]
    using LinearAlgebra
    using TRNC 
    using Plots, Printf
    # using Convex, SCS

    compound = 1
    m,n = compound*200, compound*512
    p = randperm(n)
    k = compound*10

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
    @test typeof(λ) == R


    β = eigmax(A'*A)


    function proxp!(z, α)
        n = length(z);
        for i = 1:n
            z[i] > α ? z[i] -= α :
            z[i] <-α ? z[i] += α : z[i] = T(0.0)
        end
    end

    function funcF!(z, g)
        r = copy(b)
        BLAS.gemv!('N', T(1.0), A, z, T(-1.0), r)
        BLAS.gemv!('T', T(1.0), A, r, T(0.0), g)
        return r'*r
    end
    function funcF(x)
        r = A*x - b
        g = A'*r
        return norm(r)^2, g
    end
    function proxp(z, α)
        return sign.(z).*max.(abs.(z).-(α)*ones(size(z)), zeros(size(z)))
    end

    TOL = R(1e-6)

    @testset "PG" begin

        ## PG and PG! (in place)
        pg_options=s_options(β; maxIter=1000, verbose=0, λ=λ, optTol=TOL)
        x = zeros(T, n)

        x_out, x⁻_out, hispg_out, fevalpg_out = PG(funcF, x, proxp, pg_options)
        x⁻, hispg, fevalpg = PG!(funcF!, x, proxp!,pg_options)

        #check types
        @test eltype(x) == T
        @test eltype(x_out) == T
        @test eltype(x⁻_out) == T
        @test eltype(x⁻) == T

        #check func evals less than maxIter 
        @test fevalpg_out < 1000
        @test fevalpg < 1000

        #check overall accuracy
        @test norm(x - x0, Inf) <= TOL
        @test norm(x_out - x0, Inf) <= TOL

        #check relative accuracy 
        @test norm(x_out - x⁻_out, Inf) <= TOL
        @test norm(x - x⁻, Inf) <= TOL
        @test norm(x_out - x, Inf) <= TOL
        @test norm(x⁻_out - x⁻, Inf) <= TOL
        


    end

    @testset "FISTA" begin
        ## FISTA and FISTA! (in place)
        fista_options=s_options(β; maxIter=1000, verbose=0, λ=λ, optTol=TOL)
        x = zeros(T, n)

        x_out, x⁻_out, hisf_out, fevalf_out = FISTA(funcF, x, proxp, fista_options)
        x⁻, hispf, fevalf = FISTA!(funcF!, x, proxp!,fista_options)


        #check types
        @test eltype(x) == T
        @test eltype(x_out) == T
        @test eltype(x⁻_out) == T
        @test eltype(x⁻) == T

        #check func evals less than maxIter 
        @test fevalpg_out < 1000
        @test fevalpg < 1000

        #check overall accuracy
        @test norm(x - x0, Inf) <= TOL
        @test norm(x_out - x0, Inf) <= TOL

        #check relative accuracy 
        @test norm(x_out - x⁻_out, Inf) <= TOL
        @test norm(x - x⁻, Inf) <= TOL
        @test norm(x_out - x, Inf) <= TOL
        @test norm(x⁻_out - x⁻, Inf) <= TOL
        

    end


end