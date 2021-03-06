# Julia Testing function
# Generate Compressive Sensing Data

function bpdnNoBar(x0,xi,A, f_obj, h_obj,ϕ,g,λ,parameters, options, solverp,solverz, folder, tabname)
	#Here we just try to solve the l2-norm^2 data misfit + l1 norm regularization over the l1 trust region with -10≦x≦10
	#######
	# min_x 1/2||Ax - b||^2 + λ||x||₁
    @info "running TR with our own objective"
    xtr, ktr, Fhisttr, Hhisttr, Comp_pg = TR(xi, parameters, options)
    ival = f_obj(xi)[1]+λ*h_obj(xi); 

    xi2 = copy(xi)
    @info "running PANOC with our own objective"
    xpanoc, kpanoc, Fhistpanoc, Hhistpanoc = my_panoc(solverp, xi, f = ϕ, g = g)

    
    @info "running ZeroFPR with our own objective"
	xz, kz, Fhistz, Hhistz = my_zerofpr(solverz, xi2, f = ϕ, g = g)


    xvars = [x0, xtr, xpanoc, xz]
    # nvars = [0, ktr, kpanoc, kz]
    xlabs = ["True", "TR", "PANOC", "ZFP"]


    hist = [Fhisttr+Hhisttr, vcat(ival, Fhistpanoc+Hhistpanoc), vcat(ival, Fhistz+Hhistz)]
    partest, objtest = fig_preproc(f_obj, h_obj, xvars, xlabs, hist, [Comp_pg], A, λ, folder, tabname)



    return partest, objtest  
end
