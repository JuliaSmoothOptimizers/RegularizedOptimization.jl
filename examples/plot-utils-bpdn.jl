using PGFPlots

function plot_bpdn(Comp_pg, objdec, x, sol, name = "tr-qr")
    a = Axis(
        [
            Plots.Linear(1:length(x), x, mark = "none", legendentry = "computed"),
            Plots.Linear(1:length(sol), sol, mark = "none", legendentry = "exact"),
        ],
        xlabel = "index",
        ylabel = "signal",
        legendStyle = "at={(1.0,1.0)}, anchor=north east, draw=none, font=\\scriptsize",
    )
    save("bpdn-$(name).pdf", a)

    b = Axis(
        Plots.Linear(1:length(Comp_pg), Comp_pg, mark = "none"),
        xlabel = "outer iterations",
        ylabel = "inner iterations",
        ymode = "log",
    )
    save("bpdn-inner-outer-$(name).pdf", b)

    c = Axis(
        Plots.Linear(1:length(objdec), objdec, mark = "none"),
        xlabel = "\$ k^{th}\$  \$ \\nabla f \$ Call",
        ylabel = "Objective Value",
        ymode = "log",
    )
    save("bpdn-objdec-$(name).pdf", c)
end
