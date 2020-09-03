# Julia Testing function
# Generate Compressive Sensing Data
using Plots

function bpdnNoBarTrB0Binf(A, x0, b, b0, compound, k)
    #Here we just try to solve the l2-norm^2 data misfit + l1 norm regularization over the l1 trust region with -10≦x≦10
    #######
    # min_x 1/2||Ax - b||^2 + δ(λ||x||₀< k)
    m,n = size(A)
    #initialize x
    δ = k 
    λ = norm(A'*b, Inf)/100

    #define your smooth objective function
    #merit function isn't just this though right?
    function f_smooth(x) #gradient and hessian info are smooth parts, m also includes nonsmooth part
        r = A*x - b
        g = A'*r
        return norm(r)^2/2, g, A'*A
    end

    function h_nonsmooth(x)
        if norm(x,0) ≤ δ
            h = 0
        else
            h = Inf 
        end
        return λ*h 
    end

    #set all options
    β = eigmax(A'*A)
    Doptions=s_options(β; verbose=0, λ=λ)

    function prox(q, σ, xk, Δ)
        ProjB(w) = min.(max.(w, xk.-Δ), xk.+Δ)
        y = q + xk 
        #find largest entries
        p = sortperm(y, rev = true)
        y[p[δ+1:end]].=0 #set smallest to zero 
        y = ProjB(y)#put all entries in projection
        s = y - xk 

        # w = xk + q
        # p = sortperm(w,rev=true)
        # w[p[δ+1:end]].=0
        # s = ProjB(w) - xk
        # y = ProjB(w)
        # r = (λ/σ)*.5*((y - (xk + q)).^2 - (xk + q))
        # p = sortperm(r, rev=true)
        # y[p[δ+1:end]].=0
        # s = y - xk
        return s 
    end

    parameters = IP_struct(f_smooth, h_nonsmooth; FO_options = Doptions, s_alg=PG,  Rkprox=prox)

    options = IP_options(;verbose=10, ϵD = 1e-10, Δk = 10)
    #put in your initial guesses
    xi = ones(n,)

    x, k, Fhist, Hhist, Comp = IntPt_TR(xi, parameters, options)


    xcompmat = norm(x0 - x)/opnorm(A)^2
    fullmat = [f_smooth(x)[1]+h_nonsmooth(x), f_smooth(x0)[1]+h_nonsmooth(x0) ]
    fmat = [f_smooth(x)[1], f_smooth(x0)[1]]
    hmat = [h_nonsmooth(x)/λ,  h_nonsmooth(x0)/λ]
    #print out l2 norm difference and plot the two x values

    plot(x0, xlabel="i^th index", ylabel="x", title="TR vs True x", label="True x")
    plot!(x, label="tr", marker=2)
    savefig(string("figs/bpdn/LS_B0_Binf/xcomp", compound, ".pdf"))

    plot(b0, xlabel="i^th index", ylabel="b", title="TR vs True x", label="True b")
    plot!(b, label="Observed")
    plot!(A*x, label="A*x: TR", marker=2)
    savefig(string("figs/bpdn/LS_B0_Binf/bcomp", compound, ".pdf"))

    plot(Fhist, xlabel="k^th index", ylabel="Function Value", title="Objective Value History", label="f(x)", yaxis=:log)
    savefig(string("figs/bpdn/LS_B0_Binf/objhist", compound, ".pdf"))


    plot(Comp, xlabel="k^th index", ylabel="Function Calls per Iteration", title="Complexity History", label="TR")
    savefig(string("figs/bpdn/LS_B0_Binf/complexity", compound, ".pdf"))

    return xcompmat, fullmat, fmat, hmat
end
