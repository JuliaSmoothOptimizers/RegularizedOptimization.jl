# Julia Testing function
# Generate Compressive Sensing Data
using TRNC, Plots,Printf, Convex,SCS, Random, LinearAlgebra

function bpdnNoBarTrl0Binf(A, x0, b, b0, compound)
    #Here we just try to solve the l2-norm^2 data misfit + l1 norm regularization over the l1 trust region with -10≦x≦10
    #######
    # min_x 1/2||Ax - b||^2 + λ||x||₀
    m,n = size(A)
    λ = norm(A'*b, Inf)/10


    #define your smooth objective function
    #merit function isn't just this though right?
    function f_smooth(x) #gradient and hessian info are smooth parts, m also includes nonsmooth part
        r = A*x - b
        g = A'*r
        return norm(r)^2/2, g, A'*A
    end

    function h_nonsmooth(x)
        return λ*norm(x,1) #, g∈∂h
    end

    #all this should be unraveling in the hardproxB# code
    function prox(q, σ, xk, Δ)

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


    #set all options
    β = eigmax(A'*A)
    Doptions=s_options(β; λ=λ, verbose=0, maxIter=1000)

    # first_order_options = s_options(norm(A'*A)^(2.0) ;optTol=1.0e-3, λ=λ_T, verbose=22, maxIter=5, restart=20, η = 1.0, η_factor=.9)
    #note that for the above, default λ=1.0, η=1.0, η_factor=.9

    parameters = IP_struct(f_smooth, h_nonsmooth; 
        FO_options = Doptions, s_alg=PG, Rkprox=prox)
    options = IP_options(; ϵD=1e-10)

    #put in your initial guesses
    xi = zeros(n,)/2


    X = Variable(n)
    opt = () -> SCS.Optimizer(verbose=false)
    problem = minimize(sumsquares(A * X - b) + λ*norm(X,1))
    solve!(problem, opt)


    x, k, Fhist, Hhist, Comp = IntPt_TR(xi, parameters, options)





    #print out l2 norm difference and plot the two x values
    xcompmat = [norm(x0 - x)/opnorm(A)^2, norm(X.value - x)/opnorm(A)^2, norm(X.value - x0)/opnorm(A)^2]
    fullmat = [f_smooth(x)[1]+h_nonsmooth(x), f_smooth(X.value)[1] + h_nonsmooth(X.value), f_smooth(x0)[1]+h_nonsmooth(x0) ]
    fmat = [f_smooth(x)[1], f_smooth(X.value)[1],  f_smooth(x0)[1]]
    hmat = [h_nonsmooth(x)/λ, h_nonsmooth(X.value)/λ,  h_nonsmooth(x0)/λ]




    plot(x0, xlabel="i^th index", ylabel="x", title="TR vs True x", label="True x")
    plot!(x, label="tr", marker=2)
    plot!(X.value, label="cvx")
    savefig(string("figs/bpdn/LS_l0_Binf/xcomp", compound, ".pdf"))

    plot(b0, xlabel="i^th index", ylabel="b", title="TR vs True x", label="True b")
    plot!(b, label="Observed")
    plot!(A*x, label="A*x: TR", marker=2)
    plot!(A*X.value, label="A*x: CVX")
    savefig(string("figs/bpdn/LS_l0_Binf/bcomp", compound, ".pdf"))

    plot(Fhist, xlabel="k^th index", ylabel="Function Value", title="Objective Value History", label="f(x)", yaxis=:log)
    plot!(Hhist, label="h(x)")
    @show Fhist 
    @show Hhist 
    plot!(Fhist + Hhist, label="f+h")
    savefig(string("figs/bpdn/LS_l0_Binf/objhist", compound, ".pdf"))

    plot(Comp, xlabel="k^th index", ylabel="Function Calls per Iteration", title="Complexity History", label="TR")
    savefig(string("figs/bpdn/LS_l0_Binf/complexity", compound, ".pdf"))
    

    return xcompmat, fullmat, fmat, hmat
end
