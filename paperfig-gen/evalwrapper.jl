# Julia Testing function
# Generate Compressive Sensing Data

function evalwrapper(x0, xi, A, ϕtr, h, ϕ, g, λ, χ, params, solverp, solverz, folder)
    xi2 = copy(xi)
    xi3 = copy(xi)
    ϕ2 = ϕ
    g2 = g

    @info "running TR with our own objective"
    xtr, ktr, Fhisttr, Hhisttr, Comp_pg = TR(ϕtr, h, χ, params; s_alg = PG)
    proxnum = [0, sum(Comp_pg[2,:])]

    ival = obj(ϕtr, xi) + h(xi); 
    ϕ.hist = [ival] 
    
    @info "running PANOC with our own objective"
    xpanoc, kpanoc = my_panoc(solverp, xi2, f=ϕ, g=g)
    histpanoc = ϕ.hist
    append!(proxnum, g.count)

    @info "running ZeroFPR with our own objective"
    ϕ2.hist = [ival] 
	xz, kz, xb = my_zerofpr(solverz, xi3, f=ϕ2, g=g2)
    histz = ϕ.hist
    append!(proxnum, g2.count)

    xvars = [x0, xtr, xpanoc, xb]
    xlabs = ["True", "TR-PG", "PANOC", "ZFP"]
# @show histz, g.func(xz), norm(xz - xpanoc), g.func(xb), norm(xb - xpanoc)
    hist = [ktr, vcat(ival, histpanoc), vcat(ival, histz)]
    fig_preproc(xvars,xlabs, hist, [Comp_pg[2,:]], A, folder)
    vals, pars = tab_preproc(ϕtr, g.func, xvars,proxnum, hist, A, λ)
    return vals, pars
    
end
