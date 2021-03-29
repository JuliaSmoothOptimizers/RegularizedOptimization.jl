function figen(xvars, labels, savestring, titles, typeswitch, yax)

    lstyle = []
    marks = []
    colors = []

    for i = 1:length(labels)
        if occursin("True", labels[i])
            push!(lstyle, :solid)
            push!(marks, :none)
        elseif occursin("TR", labels[i]) || occursin("QR", labels[i])
            push!(lstyle, :dash)
            push!(marks, :none)
        elseif occursin("PANOC", labels[i]) || || occursin("PG", labels[i])
            push!(lstyle, :dot)
            push!(marks,  :circle)
        elseif occursin("ZFP", labels[i])
            push!(lstyle, :dashdotdot)
            push!(marks,  :circle)
        else
            push!(lstyle, :auto)
            push!(marks,  :circle)
        end

        if occursin("True", labels[i])
            push!(colors, :darkgray)
        else
            push!(colors, :black)
        end

    end
    # if typeswitch == 1

    #     lstyle = [:solid, :dot, :dash, :dashdot]

    # elseif typeswitch == 2
    #     lstyle = [:solid,:solid, :dot,:dot, :dash, :dash, :dashdot, :dashdot]
    # else
    #     lstyle = [:solid,:solid,:solid,:dot, :dot,:dot, :dash, :dash,:dash, :dashdot, :dashdot, :dashdot]
        
    # end
    # marks = [:circle, :cross, :xcross, :diamond, :hline, :ltriangle, :utriangle, :vline, :rect]

    if yax==1
        plot(xvars[1], color = colors[1], label=labels[1], linewidth = 1, linestyle = lstyle[1], title = titles[1], xlabel=titles[2], ylabel=titles[3], tickfontsize = 14, xguidefontsize=18, yguidefontsize=18,legendfontsize=18)
    else
        plot(xvars[1], color = colors[1], label=labels[1], linewidth = 1, linestyle = lstyle[1], title = titles[1], xlabel=titles[2], ylabel=titles[3], yscale = :log10, tickfontsize = 14, xguidefontsize=18, yguidefontsize=18,legendfontsize=18)
    end

    for i = 2:length(xvars)

        plot!(xvars[i], linewidth = 1, linestyle = lstyle[i],color = colors[i], label=labels[i])

    end
    # tkstring = string(savestring, ".pdf")
    tkstring = string(savestring, ".tikz")
    # texstring = string(savestring, ".tex")
    savefig(tkstring)
    # run(`mv $texstring $tkstring`)

end