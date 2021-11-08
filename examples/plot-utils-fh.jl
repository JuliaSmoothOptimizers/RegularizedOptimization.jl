using PGFPlots

function plot_fh(outstruct, F, data, name = "tr-qr")
  Comp_pg = outstruct.solver_specific[:SubsolverCounter]
  objdec = outstruct.solver_specific[:Fhist] + outstruct.solver_specific[:Hhist]
  F1 = @view F[1:2:end-1]
  F2 = @view F[2:2:end]
  data1 = @view data[1:2:end-1]
  data2 = @view data[2:2:end]
  a = Axis([Plots.Linear(1:length(F1), F1, mark="none", legendentry="V"),
            Plots.Linear(1:length(F2), F2, mark="none", legendentry="W"),
            Plots.Linear(1:length(data1), data1, onlyMarks=true, markSize=1, mark="o", legendentry="V data"),
            Plots.Linear(1:length(data2), data2, onlyMarks=true, markSize=1, mark="*", legendentry="W data"),
          ],
          xlabel="time",
          ylabel="voltage",
          legendStyle = "at={(1.0,1.0)}, anchor=north east, draw=none, font=\\scriptsize"
          )
  save("fh-$(name).pdf", a)

  b = Axis(Plots.Linear(1:length(Comp_pg), Comp_pg, mark="none"),
          xlabel="outer iterations",
          ylabel="inner iterations",
          ymode="log",
          )
  save("fh-inner-outer-$(name).pdf", b)

  c = Axis(Plots.Linear(1:length(objdec), objdec, mark = "none"),
          xlabel="\$ k^{th}\$  \$ \\nabla f \$ Call",
          ylabel="Objective Value",
          ymode = "log"
        )
  save("fh-objdec-$(name).pdf",c)
end

