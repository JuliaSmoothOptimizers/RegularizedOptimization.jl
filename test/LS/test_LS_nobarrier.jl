# Julia Testing function
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
        # h(d) = A'*(A*d)
        return f, g
    end
    function f_pg(x)
        f = .5*norm(A*x-b)^2
        g = A'*(A*x - b)
        # h = A'*A #-> BFGS later
        return f, g
    end

    λ = 1.0 
    #SPGSlim version
    function tr_norm(z,σ, x, Δ)
        return z./max(1, norm(z, 2)/σ)
    end
    function proxp(z,σ)
        return z
    end
    function h_obj(x)
        return 0
    end

    #set all options
    first_order_options_proj = s_options(1/eigmax(A'*A);maxIter=1000, verbose=0)
        #need to tighten this because you don't make any progress in the later iterations


    # Interior Pt Algorithm
    parameters_proj = IP_struct(f_obj, h_obj; s_alg = PG, FO_options = first_order_options_proj, Rkprox=tr_norm)
    options_proj= IP_options(;verbose=0, ϵD=1e-10)

    #put in your initial guesses
    xi = ones(n,)/2

    x_pr, k, Fhist, Hhist, Comp_pg = IntPt_TR(xi, parameters_proj, options_proj)
    # xpg, xpg⁻, histpg, fevals = PGLnsch(f_pg, h_obj, xi, proxp, first_order_options_proj)
    popt = spg_options(;optTol=1.0e-10, progTol=1.0e-10, verbose=0, memory=5, maxIter = 1000)
	funproj(d) = oneProjector(d, 1.0, 1.0)
	# funproj(d) = proxp(d, λ)
	(xpg, fsave, funEvals,_,histpg) = minConf_SPG(f_pg, xi, (d)->proxp(d, 1.0), popt)
   
    

    folder = string("figs/ls/", compound, "/")


    fp = f_obj(x_pr)[1]+h_obj(p_pr)
    fpt =  (f_obj(x0)[1]+h_obj(x0))
    fpo =  (f_obj(xpg)[1]+h_obj(xpg))

    objtest = abs(fp - fpt)/opnorm(A)
    partest = norm(x_pr - x0)/opnorm(A)


    ftab = [f_obj(x_pr)[1], f_obj(xpg)[1], f_obj(x0)[1]]'
    htab = [h_obj(x_pr)/λ, h_obj(xpg)/λ, h_obj(x0)/λ ]'
    objtab = [fp,fpo, fpt]'
    vals = vcat(objtab, ftab, htab, [partest, norm(xpg - x0)/opnorm(A), 0 ]')
    pars = hcat(x0, x_pr, xpg)


    xvars = [x_pr, x0, xpg]; xlabs = ["TR", "True", "MC"]
    titles = ["Basis Comparison", "ith Index", " "]
    figen(xvars, xlabs, string(folder,"xcomp"), ["Basis Comparison", "ith Index", " "], 1, 0)




    bvars = [A*x_pr, b0, A*xpg]; 
    figen(bvars, xlabs,string(folder,"bcomp"), ["Signal Comparison", "ith Index", " "], 1, 0)
    
    
    # hist = [Fhist + zeros(size(Fhist)), Fhist, ones(size(Fhist)).*(1e-16), 
        # histpg, histpg, ones(size(histpg)).*(1e-16)] 
    # labs = ["f+g: TR", "f: TR", "h: TR", "f+g: PG", "f: PG", "h: PG"]
    hist = [Fhist, histpg[1,:]]
    histx = [Array(1:length(Fhist)), histpg[2,:]] 
    labs = ["f+h: TR", "f+h: MC"]
    figen_non(histx, hist, labs, string(folder,"objcomp"), ["Objective History", "kth Objective Evaluation", " Objective Value "], 3, 1)
 
    figen([Comp_pg], ["TR"], string(folder,"complexity"), ["Complexity History", "kth Iteration", " Objective Function Evaluations "], 1, 0)

    
    dp, df = show_table(pars, vals)
    _ = write_table(dp, df, string(folder,"ls"))



    return partest, objtest  

end