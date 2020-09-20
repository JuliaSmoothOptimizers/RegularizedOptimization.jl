# Julia Testing function
using Plots
include("lstable.jl")
#Here we just try to solve the l2-norm Problem over the l1 trust region
#######
# min_x 1/2||Ax - b||^2
function LSnobar(A, x0, b, b0, compound)
    m,n= size(A)
    function f_obj(x)
        f = .5*norm(A*x-b)^2
        g = A'*(A*x - b)
        # h = A'*A #-> BFGS later
        h(d) = A'*(A*d)
        return f, g, h
    end
    function f_pg(x)
        f = .5*norm(A*x-b)^2
        g = A'*(A*x - b)
        # h = A'*A #-> BFGS later
        return f, g
    end


    #SPGSlim version
    function tr_norm(z,σ, x, Δ)
        return z./max(1, norm(z, 2)/σ)
    end

    function h_obj(x)
        return 0
    end

    #set all options
    first_order_options_proj = s_options(1/eigmax(A'*A);maxIter=1000, verbose=0)
        #need to tighten this because you don't make any progress in the later iterations


    # Interior Pt Algorithm
    parameters_proj = IP_struct(f_obj, h_obj; s_alg = PG, FO_options = first_order_options_proj, Rkprox=tr_norm)
    options_proj= IP_options(;verbose=0, ϵD=1e-4)

    #put in your initial guesses
    xi = ones(n,)/2

    x_pr, k, Fhist, Hhist, Comp_pg = IntPt_TR(xi, parameters_proj, options_proj)
    xpg, xpg⁻, histpg, fevals = PGLnsrch(f_pg, h_obj, xi, (z, σ) -> tr_norm(z, σ, 1, 2), first_order_options_proj)
   
    





    fp = f_obj(x_pr)[1]+h_obj(p_pr)
    fpt =  (f_obj(x0)[1]+h_obj(x0))
    fpo =  (f_obj(xpg)[1]+h_obj(xpg))

    objtest = abs(fp - fpt)
    partest = norm(x_pr - x0)


    ftab = [f_obj(x_pr)[1], f_obj(x0)[1], f_obj(xpg)[1]]'
    htab = [h_obj(x_pr)/λ, h_obj(x0)/λ, h_obj(xpg)/λ ]'
    objtab = [fpt, fp, fpo]'
    vals = vcat(objtab, ftab, htab, [partest, norm(xpg - x0), 0 ]')
    pars = hcat(x0, x_pr, xpg)


    xvars = [x_pr, x0, xpg]; xlabs = ["TR", "True", "PG"]
    titles = ["Basis Comparison", "ith Index", " "]
    figen(xvars, labs, "figs/ls/xcomp", ["Basis Comparison", "ith Index", " "], 1)




    bvars = [A*x_pr, A*x0, A*xpg]; 
    figen(bvars, labs, "figs/ls/bcomp", ["Signal Comparison", "ith Index", " "], 1)
    
    
    hist = [Fhist + zeros(size(Fhist)), Fhist, zeros(size(Fhist)), 
            histpg, histpg, zeros(size(histpg))] 
    labs = ["f+g: TR", "f: TR", "h: TR", "f+g: PG", "f: PG", "h: PG"]
    figen(hist, labs, "figs/ls/objcomp", ["Objective History", "kth Iteration", " Objective Value "], 3)
 

    figen(hist, labs, "figs/ls/complexity", ["Complexity History", "kth Iteration", " Objective Function Evaluations "], 1)

    dp, df = show_table(pars, vals)
    _ = write_table(dp, df, "figs/ls/ls")



    return partest, objtest  

end