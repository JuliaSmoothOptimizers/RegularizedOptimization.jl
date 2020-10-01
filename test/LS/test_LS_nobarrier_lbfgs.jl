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

    function tr_norm(z,σ, x, Δ)
        return z./max(1, norm(z, 2)/σ)
    end

    function proxp(z,σ)
        return z
    end
    
    function h_obj(x)
        return 0
    end
    λ = 1.0 

   #set all options
   first_order_options_proj = s_options(1/eigmax(A'*A);maxIter=10000, verbose=0)
   #need to tighten this because you don't make any progress in the later iterations


    # Interior Pt Algorithm
    parameters_proj = IP_struct(f_obj, h_obj; s_alg = PG, FO_options = first_order_options_proj, Rkprox=tr_norm)
    options_proj= IP_options(;verbose=0, ϵD=1e-4)

    #put in your initial guesses
    xi = ones(n,)/2

    x_pr, k, Fhist, Hhist, Comp_pg = IntPt_TR(xi, parameters_proj, options_proj)
    xpg, xpg⁻, histpg, fevals = PGLnsch(f_obj, h_obj, xi, proxp, first_order_options_proj)

    folder = string("figs/ls_bfgs/", compound, "/")

    fp = f_obj(x_pr)[1]+h_obj(p_pr)
    fpt =  (f_obj(x0)[1]+h_obj(x0))
    fpo =  (f_obj(xpg)[1]+h_obj(xpg))

    objtest = abs(fp - fpt)/norm(A,2)
    partest = norm(x_pr - x0)/norm(A,2)


    ftab = [f_obj(x_pr)[1], f_obj(xpg)[1], f_obj(x0)[1]]'
    htab = [h_obj(x_pr)/λ, h_obj(xpg)/λ, h_obj(x0)/λ ]'
    objtab = [fp,fpo, fpt]'
    vals = vcat(objtab, ftab, htab, [partest, norm(xpg - x0), 0 ]')
    pars = hcat(x0, x_pr, xpg)


    xvars = [x_pr, x0, xpg]; xlabs = ["TR", "True", "PG"]
    titles = ["Basis Comparison", "ith Index", " "]
    figen(xvars, xlabs, string(folder,"xcomp"), ["Basis Comparison", "ith Index", " "], 1, 0)




    bvars = [A*x_pr, b0, A*xpg]; 
    figen(bvars, xlabs,string(folder,"bcomp"), ["Signal Comparison", "ith Index", " "], 1, 0)
    
    
    hist = [Fhist + zeros(size(Fhist)), Fhist, zeros(size(Fhist)), 
            histpg, histpg, zeros(size(histpg))] 
    labs = ["f+g: TR", "f: TR", "h: TR", "f+g: PG", "f: PG", "h: PG"]
    figen(hist, labs, string(folder,"objcomp"), ["Objective History", "kth Iteration", " Objective Value "], 3, 1)
 
    figen([Comp_pg], "TR", string(folder,"complexity"), ["Complexity History", "kth Iteration", " Objective Function Evaluations "], 1, 1)

    
    dp, df = show_table(pars, vals)
    _ = write_table(dp, df, string(folder,"ls"))



    return partest, objtest  

end