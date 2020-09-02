@testset "Inner Descent Direction ($compound) Descent Methods" for compound=1:3
    using LinearAlgebra
    using TRNC 
    using Plots, Roots
    using Convex, SCS

    m,n = compound*25, compound*64
    p = randperm(n)
    k = compound*2

    #initialize x 
    x0 = zeros(n)
    p = randperm(n)[1:k]
    x0[p[1:k]]=sign.(randn(k))
    xk = 10*randn(n)


    A,_ = qr(5*randn((n,m)))
    B = Array(A)'
    A = Array(B)

    b0 = A*x0
    b = b0 + .005*randn(m)
    λ = 0.1*norm(A'*b, Inf)
    Δ = 3*rand()


    β = eigmax(A'*A)

    S = Variable(n)
    opt = () -> SCS.Optimizer(verbose=false)



    function f_obj(x) #gradient and hessian info are smooth parts, m also includes nonsmooth part
        r = A*x - b
        g = A'*r
        return norm(r)^2/2, g, A'*A
    end
    function h_obj(x)
        return λ*norm(x,1)
    end

    function proxl1b2(q, σ) #q = s - ν*g, ν*λ, xk, Δ - > basically inputs the value you need

        ProjB(y) = min.(max.(y, q.-σ), q.+σ)
        froot(η) = η - norm(ProjB((-xk).*(η/Δ)))
    
        # %do the 2 norm projection
        y1 = ProjB(-xk) #start with eta = tau
        if (norm(y1)<= Δ)
            y = y1  # easy case
        else
            η = fzero(froot, 1e-10, Inf)
            y = ProjB((-xk).*(η/Δ))
        end
    
        if (norm(y)<=Δ)
            snew = y
        else
            snew = Δ.*y./norm(y)
        end
        return snew
    end 

    function proxl1b2!(q, σ) #q = s - ν*g, ν*λ, xk, Δ - > basically inputs the value you need

        ProjB(y) = min.(max.(y, q.-σ), q.+σ)
        froot(η) = η - norm(ProjB((-xk).*(η/Δ)))
    
        # %do the 2 norm projection
        y1 = ProjB(-xk) #start with eta = tau
        if (norm(y1)<= Δ)
            y = y1  # easy case
        else
            η = fzero(froot, 1e-10, Inf)
            y = ProjB((-xk).*(η/Δ))
        end
    
        if (norm(y)<=Δ)
            q[:] = y[:]
        else
            q[:] = Δ.*y[:]./norm(y)
        end
    end 
    (qk, ∇qk, H) = f_obj(xk)
    hk = h_obj(xk)
    Hess(d) = H*d

    TOL = 1e-10
    Doptions=s_options(β; maxIter=5000, verbose =0, λ=λ, optTol = TOL, xk = xk, Δ = Δ, ∇fk = ∇qk, Bk = A'*A)
    objInner(d) = [0.5*(d'*Hess(d)) + ∇qk'*d + qk, Hess(d) + ∇qk]

    function objInner!(d, g)
        g[:] = ∇qk[:]
        g += Hess(d)
        return 0.5*(d'*Hess(d)) + ∇qk'*d + qk
    end



    @testset "S: l1 - B2" begin

        
        s⁻ = zeros(n)
        s = copy(s⁻)
        s_cvx = Variable(n)
        problem = minimize(sumsquares(A*(xk+s_cvx) - b) + λ*norm(s_cvx+xk,1), norm(s_cvx, 2)<=Δ);
        solve!(problem, opt)
        s_out, s⁻_out, _, feval = PG(objInner, s⁻, proxl1b2, Doptions)
        s⁻, _, fevals = PG!(objInner!, s, proxl1b2!, Doptions)

        #check func evals less than maxIter 
        @test feval <= 5000
        @test fevals <= 5000

        #check overall accuracy
        @test norm(s_cvx.value - s_out)/norm(s_cvx.value) <= .01
        @test norm(s_cvx.value - s)/norm(s_cvx.value) <= .01

        #check relative accuracy 
        @test norm(s_out - s⁻_out, 2) <= TOL
        @test norm(s - s⁻, 2) <= TOL


        #test function outputs
        @test f_obj(xk+s_out)[1]+h_obj(xk+s_out) < qk + hk 
        @test f_obj(xk+s)[1]+h_obj(xk+s) < qk + hk #check for decrease 
        @test (f_obj(xk+s_out)[1]+h_obj(xk+s_out) -(f_obj(xk+s_cvx.value)[1] + h_obj(s_cvx.value.+xk)))/(f_obj(xk+s_cvx.value)[1] + h_obj(s_cvx.value.+xk))<.01
        @test (f_obj(xk+s)[1]+h_obj(xk+s) -(f_obj(xk+s_cvx.value)[1] + h_obj(s_cvx.value.+xk)))/(f_obj(xk+s_cvx.value)[1] + h_obj(s_cvx.value.+xk))<.01

    end

    function proxl1binf(q, σ)
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
    function proxl1binf!(q, σ)
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
        s = dropdims(s, dims=2)
        q[:] = s[:]
    end

    @testset "S: l1 - Binf" begin

        
        s⁻ = zeros(n)
        s = copy(s⁻)
        s_cvx = Variable(n)
        problem = minimize(sumsquares(A*(xk+s_cvx) - b) + λ*norm(s_cvx+xk,1), norm(s_cvx, Inf)<=Δ);
        solve!(problem, opt)

        s_out, s⁻_out, _, feval = FISTA(objInner, s⁻, proxl1binf, Doptions)
        s⁻, _, fevals = FISTA!(objInner!, s, proxl1binf!, Doptions)

        #check func evals less than maxIter 
        @test feval <= 5000
        @test fevals <= 5000

        #check overall accuracy - new test because s_cvx is unreliable for large problems 
        # @test norm(s_cvx.value .- s_out) <= .01
        # @test norm(s_cvx.value .- s) <= .01

        #check relative accuracy 
        @test norm(s_out .- s⁻_out) <= TOL
        @test norm(s .- s⁻) <= TOL
        
        @test f_obj(xk+s_out)[1]+h_obj(xk+s_out) < qk + hk 
        @test f_obj(xk+s)[1]+h_obj(xk+s) < qk + hk 
        @test (f_obj(xk+s_out)[1]+h_obj(xk+s_out) -(f_obj(xk+s_cvx.value)[1] + h_obj(s_cvx.value.+xk)))/(f_obj(xk+s_cvx.value)[1] + h_obj(s_cvx.value.+xk))<.05
        @test (f_obj(xk+s)[1]+h_obj(xk+s) -(f_obj(xk+s_cvx.value)[1] + h_obj(s_cvx.value.+xk)))/(f_obj(xk+s_cvx.value)[1] + h_obj(s_cvx.value.+xk))<.05


end


function proxl0binf(q, σ)

    ProjB(y) = min.(max.(y, xk.-Δ),xk.+Δ) # define outside? 
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

function proxl0binf!(q, σ)

    ProjB(y) = min.(max.(y, xk.-Δ),xk.+Δ) # define outside? 
    c = sqrt(2*σ)
    q[:] += xk[:]

    for i = 1:length(q)
        absx = abs(q[i])
        if absx <=c
            q[i] = 0
        end
    end
    q[:] = ProjB(q)[:]
    q[:] -= xk[:] 
end 

@testset "S: l0 - Binf" begin

        
        s⁻ = zeros(n)
        s = copy(s⁻)
        s_out, s⁻_out, _, feval = FISTA(objInner, s⁻, proxl0binf, Doptions)
        s⁻, _, fevals = FISTA!(objInner!, s, proxl0binf!, Doptions)

        s_cvx = Variable(n)
        problem = minimize(sumsquares(A*(xk+s_cvx) - b) + λ*norm(s_cvx+xk,1), norm(s_cvx, Inf)<=Δ);
        solve!(problem, opt)

        #check func evals less than maxIter 
        @test feval <= 5000
        @test fevals <= 5000

        #check overall accuracy - new test because s_cvx is unreliable for large problems 
        # if compound ==1
        #     @test norm(s_cvx.value .- s_out) <= .01
        #     @test norm(s_cvx.value .- s) <= .01
        # end

        
        #check relative accuracy 
        @test norm(s_out .- s⁻_out) <= TOL
        @test norm(s .- s⁻) <= TOL
        
        @test f_obj(xk+s_out)[1]+h_obj(xk+s_out) < qk + hk 
        @test f_obj(xk+s)[1]+h_obj(xk+s) < qk + hk 
        @test (f_obj(xk+s_out)[1]+h_obj(xk+s_out) -(f_obj(xk+s_cvx.value)[1] + h_obj(s_cvx.value.+xk)))/(f_obj(xk+s_cvx.value)[1] + h_obj(s_cvx.value.+xk))<.05
        @test (f_obj(xk+s)[1]+h_obj(xk+s) -(f_obj(xk+s_cvx.value)[1] + h_obj(s_cvx.value.+xk)))/(f_obj(xk+s_cvx.value)[1] + h_obj(s_cvx.value.+xk))<.05


end


end