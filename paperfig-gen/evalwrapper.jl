# Julia Testing function
# Generate Compressive Sensing Data

function evalwrapper(x0, xi, A, ϕtr, h, ϕ, g, λ, methods, params, solverp, solverz, folder)
    xi2 = copy(xi)
    xi3 = copy(xi)

    @info "running TR with our own objective"
    xtr, ktr, Fhisttr, Hhisttr, Comp_pg = TRalg(ϕtr, h, methods, params)
    proxnum = [0, sum(Comp_pg)]

    ival = obj(ϕtr, xi) + h(xi); 
    ϕ.hist = [ival] 
    
    @info "running PANOC with our own objective"
    xpanoc, kpanoc = my_panoc(solverp, xi2, f=ϕ, g=g)
    histpanoc = ϕ.hist
    append!(proxnum, g.count)
    
    @info "running ZeroFPR with our own objective"
    ϕ.hist = [ival] 
    ϕ.count = 0
    g.count = 0
	xz, kz = my_zerofpr(solverz, xi3, f=ϕ, g=g)
    histz = ϕ.hist
    append!(proxnum, g.count)

    xvars = [x0, xtr, xpanoc, xz]
    xlabs = ["True", "TR", "PANOC", "ZFP"]


    hist = [Fhisttr + Hhisttr, vcat(ival, histpanoc), vcat(ival, histz)]
    fig_preproc(xvars,xlabs, hist, [Comp_pg], A, folder)
    vals, pars = tab_preproc(ϕtr, g.func, xvars,proxnum, hist, A, λ)
    return vals, pars
    
end
