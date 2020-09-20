using Plots


function figen(xvars, labels, savestring, titles, typeswitch)

    if typeswitch == 1

        lstyle = [:solid, :dot, :dash, :dashdot]
        marks = [:circle, :cross, :x, :xcross]

    elseif typeswitch == 2
        lstyle = [:solid,:solid, :dot,:dot, :dash, :dash, :dashdot, :dashdot]
    else
        lstyle = [:solid,:solid,:solid,:dot, :dot,:dot, :dash, :dash,:dash, :dashdot, :dashdot, :dashdot]
        
    end
    marks = [:circle, :cross, :x, :xcross, :diamond, :hline, :ltriangle, :utriangle, :vline, :rect]

    plot(xvars[1],labels[1], linewidth = 4, marker=2, linestyle = lstyle[1], markershape = marks[1], title = titles[1], xlabel=titles[2], ylabel=titles[3])

    for i = 2:length(xvars)

        plot!(xvars[i],linestyle = lstyle[i], markershape = marks[i], label=labels[i])

    end
    tkstring = string(savestring, ".tikz")
    texstring = string(savestring, ".tex")
    savefig(tkstring)
    run(`mv $texstring $tkstring`)

end