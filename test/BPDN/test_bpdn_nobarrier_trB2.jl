# Julia Testing function
# Generate Compressive Sensing Data
using Plots

#Here we just try to solve the l2-norm^2 data misfit + l1 norm regularization over the l1 trust region with -10≦x≦10
#######
function bpdnNoBarTrB2(A, x0, b, b0, compound)

    #Here we just try to solve the l2-norm^2 data misfit + l1 norm regularization over the l1 trust region with -10≦x≦10
    #######
    # min_x 1/2||Ax - b||^2 + λ||x||₁
    m,n = size(A)
    λ = norm(A'*b, Inf)/100


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

function prox(q, σ, xk, Δ) #q = s - ν*g, ν*λ, xk, Δ - > basically inputs the value you need

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
    #set all options
    β = eigmax(A'*A)
    Doptions=s_options(β;maxIter=1000, verbose =0, λ=λ)


    parameters = IP_struct(f_smooth, h_nonsmooth; FO_options = Doptions, s_alg=PG, Rkprox=prox)
    options = IP_options(;ϵD = 1e-10)
    #put in your initial guesses
    xi = ones(n,)/2


    X = Variable(n)
    problem = minimize(sumsquares(A * X - b) + λ*norm(X,1))
    opt = () -> SCS.Optimizer(verbose=false)
    solve!(problem, opt)

    function funcF(x)
        r = A*x - b
        g = A'*r
        return norm(r)^2, g
    end
    function proxp(z, α)
        return sign.(z).*max.(abs.(z).-(α)*ones(size(z)), zeros(size(z)))
    end

    (xp, xp⁻, fsave, funEvals) =PG(
        funcF,
        xi,
        proxp,
        Doptions,
    )

    x, k, Fhist, Hhist, Comp = IntPt_TR(xi, parameters, options)


#print out l2 norm difference and plot the two x values
  xcompmat = [norm(x0 - x)/opnorm(A)^2,norm(x0 - xp)/opnorm(A)^2, norm(X.value - x)/opnorm(A)^2, norm(X.value - x0)/opnorm(A)^2]
  fullmat = [f_smooth(x)[1]+h_nonsmooth(x),f_smooth(xp)[1]+h_nonsmooth(xp), f_smooth(X.value)[1] + h_nonsmooth(X.value), f_smooth(x0)[1]+h_nonsmooth(x0) ]
  fmat = [f_smooth(x)[1], f_smooth(xp)[1],f_smooth(X.value)[1], f_smooth(x0)[1]]
  hmat = [h_nonsmooth(x)/λ, h_nonsmooth(xp)/λ, h_nonsmooth(X.value)/λ, h_nonsmooth(x0)/λ]


    plot(x0, xlabel="i^th index", ylabel="x", title="TR vs True x", label="True x")
    plot!(x, label="tr", marker=2)
    plot!(X.value, label="cvx")
    savefig(string("figs/bpdn/LS_l1_B2/xcomp", compound, ".pdf"))

    plot(b0, xlabel="i^th index", ylabel="b", title="TR vs True x", label="True b")
    plot!(b, label="Observed")
    plot!(A*x, label="A*x: TR", marker=2)
    plot!(A*X.value, label="A*x: CVX")
    savefig(string("figs/bpdn/LS_l1_B2/bcomp", compound, ".pdf"))

    plot(Fhist, xlabel="k^th index", ylabel="Function Value", title="Objective Value History", label="f(x)", yaxis=:log)
    plot!(Hhist, label="h(x)")
    plot!(Fhist + Hhist, label="f+h")
    savefig(string("figs/bpdn/LS_l1_B2/objhist", compound, ".pdf"))

    plot(Comp, xlabel="k^th index", ylabel="Function Calls per Iteration", title="Complexity History", label="TR")
    savefig(string("figs/bpdn/LS_l1_B2/complexity", compound, ".pdf"))

    return xcompmat, fullmat, fmat, hmat
end
