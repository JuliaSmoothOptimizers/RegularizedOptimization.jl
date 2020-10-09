using Plots


function figen_non(xvars, yvars, labels, savestring, titles, typeswitch, yax)

    lstyle = []
    marks = []
    colors = []

    for i = 1:length(labels)
        if occursin("True", labels[i])
            push!(lstyle, :solid)
            push!(marks, :none)
        elseif occursin("TR", labels[i])
            push!(lstyle, :dash)
            push!(marks, :none)
        elseif occursin("MC", labels[i])
            push!(lstyle, :dot)
            push!(marks,  :none)
        else
            push!(lstyle, :none)
            push!(marks,  :circle)
        end

        if i % 2==0
            push!(colors, :darkgray)
        else
            push!(colors, :black)
        end

    end

    # if typeswitch == 1

    #     lstyle = [:solid, :dot, :dash, :none]

    # elseif typeswitch == 2
    #     lstyle = [:solid,:solid, :dot,:dot, :dash, :dash, :dashdot, :dashdot]
    # else
    #     lstyle = [:solid,:solid,:solid,:dot, :dot,:dot, :dash, :dash,:dash, :dashdot, :dashdot, :dashdot]
        
    # end
    # marks = [:circle, :cross, :xcross, :diamond, :hline, :ltriangle, :utriangle, :vline, :rect]
    # marks = [:circle, :cross, :xcross, :diamond, :hline, :ltriangle, :utriangle, :vline, :rect]

    if yax==1
        plot(xvars[1],yvars[1],color = colors[1], label=labels[1], linewidth = 2, linestyle = lstyle[1], title = titles[1], xlabel=titles[2], ylabel=titles[3])
    else
        plot(xvars[1],yvars[1],color = colors[1],  label=labels[1], linewidth = 2, linestyle = lstyle[1], title = titles[1], xlabel=titles[2], ylabel=titles[3], yscale = :log10)
    end

    for i = 2:length(yvars)

        if occursin("Data", labels[i])
            scatter!(xvars[i], yvars[i], markershape=marks[i], markersize = 2, markercolor = colors[i], label = labels[i])
        else
            plot!(xvars[i],yvars[i],linestyle = lstyle[i], color = colors[i], label=labels[i])
        end

    end
    tkstring = string(savestring, ".pdf")
    # tkstring = string(savestring, ".tikz")
    # texstring = string(savestring, ".tex")
    savefig(tkstring)
    # run(`mv $texstring $tkstring`)

end