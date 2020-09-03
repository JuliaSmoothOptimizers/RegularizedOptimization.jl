# Julia Testing function
using TRNC
using Plots, Convex, SCS, Printf,LinearAlgebra

#Here we just try to solve the l2-norm Problem over the l1 trust region
#######
# min_x 1/2||Ax - b||^2
function LSnobarBFGS(A, x0, b, b0, compound)
    m,n= size(A)

    function f_obj(x)
        f = .5*norm(A*x-b)^2
        g = A'*(A*x - b)
        return f, g
    end

    #SPGSlim version
    function tr_norm_spg(z,α,σ)
        return z./max(1, norm(z, 2)/σ)
    end

    function tr_norm(z,σ, x, Δ)
        return z./max(1, norm(z, 2)/σ)
    end

    function h_obj(x)
        return 0
    end

    #set all options
    first_order_options_spgslim = spg_options(;optTol=1.0e-1, progTol=1.0e-10, verbose=0,
        feasibleInit=true, curvilinear=true, bbType=true, memory=1)
    first_order_options_proj = s_options(1/eigmax(A'*A);maxIter=1000, verbose=0, optTol=1.0e-3)
        #need to tighten this because you don't make any progress in the later iterations


    # Interior Pt Algorithm
    parameters_spgslim = IP_struct(f_obj, h_obj;
        FO_options = first_order_options_spgslim, Rkprox=tr_norm_spg) #defaults to h=0, spgl1/min_confSPG
    parameters_proj = IP_struct(f_obj, h_obj;
        s_alg = PG, FO_options = first_order_options_proj, Rkprox=tr_norm)
    # parameters = IP_struct(f_obj, h_obj;FO_options = first_order_options, χ_projector=tr_norm) #defaults to h=0, spgl1/min_confSPG
    options_spgslim = IP_options(; ϵD =1e-4); #ϵD=1e-3) #print freq, ΔK init, epsC/epsD initialization, maxIter
    options_proj= IP_options(; ϵD=1e-4)

    #put in your initial guesses
    xi = ones(n,)/2

    X = Variable(n)
    problem = minimize(sumsquares(A * X - b))
    opt = () -> SCS.Optimizer(verbose=false)
    solve!(problem, opt)


    x_spg, k, Fhist_spg, Hhist_spg, Comp_spg = IntPt_TR(xi, parameters_spgslim, options_spgslim)
    x_pg, k, Fhist_pg, Hhist_pg, Comp_pg = IntPt_TR(xi, parameters_proj, options_proj)

    plot(x0, xlabel="i^th index", ylabel="x", title="TR vs True x", label="True x")
    plot!(x_spg, label="tr-spg", marker=2)
    plot!(x_pg, label="tr-pr", marker=3)
    plot!(X.value, label="cvx")
    savefig(string("figs/ls_bfgs/xcomp",compound,".pdf"))

    plot(b0, xlabel="i^th index", ylabel="b", title="TR vs True x", label="True b")
    plot!(b, label="Observed")
    plot!(A*x_spg, label="A*x: TR-spg", marker=2)
    plot!(A*x_pg, label="A*x: TR-pr", marker=3)
    plot!(A*X.value, label="A*x: CVX")
    savefig(string("figs/ls_bfgs/bcomp",compound,".pdf"))

    plot(Fhist_spg, xlabel="k^th index", ylabel="Function Value", title="Objective Value History", label="f(x) (SPGSlim)", yaxis=:log)
    plot!(Hhist_spg, label="h(x) (SPGSlim)")
    plot!(Fhist_spg + zeros(size(Fhist_spg)), label="f+h (SPGSlim)")
    plot!(Hhist_pg, label="h(x) (Prox-grad)")
    plot!(Fhist_pg, label="f(x) (Prox-grad)")
    plot!(Fhist_pg+ zeros(size(Fhist_pg)), label="f+h (Prox-grad)")
    savefig(string("figs/ls_bfgs/objhist", compound,".pdf"))


    plot(Comp_pg, xlabel="k^th index", ylabel="Function Calls per Iteration", title="Complexity History", label="TR")
    plot!(Comp_spg, label="SPG")
    savefig(string("figs/ls_bfgs/complexity",compound,".pdf"))   


    return norm(x_spg - x0)/opnorm(A)^2, norm(x_pg - x0)/opnorm(A)^2, norm(X.value - x0)/opnorm(A)^2, norm(X.value - x_spg)/opnorm(A)^2, norm(X.value - x_pg)/opnorm(A)^2 #, sum(Comp_pg), sum(Comp_spg) 

end