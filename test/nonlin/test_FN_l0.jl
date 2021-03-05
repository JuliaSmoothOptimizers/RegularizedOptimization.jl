# Julia Testing function
#In this example, we demonstrate the capacity of the algorithm to minimize a nonlinear
#model with a regularizer
function FHNONLIN(x0,xi, A, f_obj, h_obj,ϕ,g,λ,parameters, options, solver, folder, tabname)

	@info "running TR with our own objective"
	xtr, ktr, Fhisttr, Hhisttr, Comp_pg = TR(xi, parameters, options)
    @info "running PANOC with our own objective"
    xpanoc, kpanoc, Fhistpanoc, Hhistpanoc = my_panoc(solver, xi, f = ϕ, g = g)

    xvars = [x0, xtr, xpanoc]
    xlabs = ["True", "TR", "PANOC"]


	hist = [Fhisttr+Hhisttr, Fhistpanoc+Hhistpanoc]
    partest, objtest = fig_preproc(f_obj, h_obj, xvars, xlabs, hist,[Comp_pg], A, λ, folder, tabname) 


	#print out l2 norm difference and plot the two x values


	return partest, objtest
end