using Plots


function figen_non(xvars, yvars, labels, savestring, titles, typeswitch, yax)

    if typeswitch == 1

        lstyle = [:solid, :dot, :dash, :dashdot]

    elseif typeswitch == 2
        lstyle = [:solid,:solid, :dot,:dot, :dash, :dash, :dashdot, :dashdot]
    else
        lstyle = [:solid,:solid,:solid,:dot, :dot,:dot, :dash, :dash,:dash, :dashdot, :dashdot, :dashdot]
        
    end
    marks = [:circle, :cross, :xcross, :diamond, :hline, :ltriangle, :utriangle, :vline, :rect]

    if yax==1
        plot(xvars,yvars[1], label=labels[1], linewidth = 2, marker=1, linestyle = lstyle[1], markershape = marks[1], title = titles[1], xlabel=titles[2], ylabel=titles[3])
    else
        plot(xvars,yvars[1], label=labels[1], linewidth = 2, marker=1, linestyle = lstyle[1], markershape = marks[1], title = titles[1], xlabel=titles[2], ylabel=titles[3], yscale = :log10)
    end

    for i = 2:length(yvars)

        plot!(xvars,yvars[i],linestyle = lstyle[i], markershape = marks[i], label=labels[i])

    end
    tkstring = string(savestring, ".pdf")
    # tkstring = string(savestring, ".tikz")
    # texstring = string(savestring, ".tex")
    savefig(tkstring)
    # run(`mv $texstring $tkstring`)

end