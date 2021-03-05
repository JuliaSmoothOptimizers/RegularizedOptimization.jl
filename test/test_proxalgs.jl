@testset "Lasso ($T) Descent Methods" for T in [Float32, Float64]#, ComplexF32, ComplexF64]
    using LinearAlgebra
    using TRNC 
    using Plots, Printf

    # compound = 1
    compound = 10
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
    @test typeof(λ) == R


    ν = 1/eigmax(A'*A)


    # function proxp!(z, α)
    #     n = length(z)
    #     for i = 1:n
    #         abs(z[i]) > α ? z[i]-= sign(z[i])*α : z[i] = T(0.0)
    #     end
    # end

    # function funcF!(z, g)
    #     r = copy(b)
    #     BLAS.gemv!('N', T(1.0), A, z, T(-1.0), r)
    #     BLAS.gemv!('T', T(1.0), A, r, T(0.0), g)
    #     return r'*r
    # end

    function funcF(x)
        r = A*x - b
        g = A'*r
        return norm(r)^2, g, A'*A
    end
    function proxp(z, α, x, Δ)
        return sign.(z).*max.(abs.(z).-(α)*ones(T, size(z)), zeros(T, size(z)))
    end

    function funcH(x)
        return λ*norm(x,1)
    end

    TOL = R(1e-6)
    MI = 100
    @testset "PG" begin

        ## PG and PG! (in place)
        pg_options=s_options(ν; maxIter=MI, verbose=0, λ=λ, optTol=TOL, p = .5)
        x = zeros(T, n)

        x_out, x⁻_out, hispg_out, fevalpg_out = PG(funcF,funcH, zeros(T, n), (z,α)->proxp(z, α, ones(T, size(x0)), 1.0), pg_options)
        x_d, x⁻_d, hispg_d, fevalpg_d = PGLnsch(funcF, funcH, zeros(T, n),(z,α)->proxp(z, α, ones(T, size(x0)), 1.0), pg_options)
        xΔ, x⁻Δ, hispgΔ, fevalpgΔ = PGΔ(funcF, funcH, zeros(T, n),(z,α)->proxp(z, α, ones(T, size(x0)), 1.0), pg_options)
        xE, x⁻E, hispgE, fevalpgE = PGLnsch(funcF, funcH, zeros(T, n),(z,α)->proxp(z, α, ones(T, size(x0)), 1.0), pg_options)

        #check types
        @test eltype(xΔ) == T
        @test eltype(x_out) == T
        @test eltype(x⁻_out) == T
        @test eltype(x⁻Δ) == T
        @test eltype(x_d) == T
        @test eltype(x⁻_d) == T

        #check func evals less than maxIter 
        @test fevalpg_out <= MI
        @test fevalpg_d <= MI
        @test fevalpgΔ <=MI
        @test fevalpgE <=MI

        #check overall accuracy
        @test norm(x_out - x0)/norm(x0) <= .2 
        @test norm(x_d - x0)/norm(x0) <= .2
        @test norm(xΔ - x0)/norm(x0) <= .2
        @test norm(xE - x0)/norm(x0) <= .2
        

        #test monotonicity
        @test sum(diff(hispg_out).<=TOL)==length(diff(hispg_out))
        @test sum(diff(hispg_d).<=TOL)==length(diff(hispg_d))
        


    end

    @testset "FISTA" begin
        ## FISTA and FISTA! (in place)
        fista_options=s_options(ν; maxIter=100, verbose=0, λ=λ, optTol=TOL)
        x = zeros(T, n)

        x_out, x⁻_out, hisf_out, fevalf_out = FISTA(funcF,funcH, zeros(T, n), (z,α)->proxp(z, α, ones(size(x0)), 1.0), fista_options)
        x_d, x⁻_d, hisf_d, fevalf_d = FISTAD(funcF, funcH, zeros(T, n), (z,α)->proxp(z, α, ones(size(x0)), 1.0), fista_options)
        


        #check types
        @test eltype(x_out) == T
        @test eltype(x⁻_out) == T
        @test eltype(x_d) == T
        @test eltype(x⁻_d) == T

        #check func evals less than maxIter 
        @test fevalf_out <= MI
        @test fevalf_d <= MI

        #check overall accuracy
        @test norm(x_out - x0)/norm(x0) <= .2
        @test norm(x_d - x0)/norm(x0) <= .2 


        #check relative accuracy 
        @test norm(x_out - x⁻_out, Inf) <= TOL*10
        @test norm(x_d - x⁻_d, Inf) <= TOL*10
        
        #test monotonicity
        temp = diff(hisf_d)
        @test sum(temp.<=TOL)==length(temp)

    end


end