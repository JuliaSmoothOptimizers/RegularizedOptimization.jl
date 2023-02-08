using PGFPlots

function plot_nnmf(outstruct, Avec, m, n, k, name = "tr-qr")
  Comp_pg = outstruct.solver_specific[:SubsolverCounter]
  objdec = outstruct.solver_specific[:Fhist] + outstruct.solver_specific[:Hhist]
  x = outstruct.solution
  A = reshape(Avec, m, n)
  W = reshape(x[1:(m * k)], m, k)
  H = reshape(x[(m * k + 1):end], k, n)
  WH = W * H
  
  a = GroupPlot(2,2)
  push!(a, Axis(Plots.Image(A, (1, m), (1, n), colormap=ColorMaps.Named("Jet")), xlabel = "A matrix (reference)"))
  push!(a, Axis(Plots.Image(WH, (1, m), (1, n), colormap=ColorMaps.Named("Jet")), xlabel = "WH matrix"))
  push!(a, Axis(Plots.Image(H, (1, k), (1, n), colormap=ColorMaps.Named("Jet")), xlabel = "H matrix"))
  push!(a, Axis(Plots.Image(abs.(A - WH), (1, m), (1, n), colormap=ColorMaps.Named("Jet")), xlabel = "|A-WH| matrix"))
  save("nnmf-$(name).pdf", a)

  b = Axis(
    Plots.Linear(1:length(Comp_pg), Comp_pg, mark = "none"),
    xlabel = "outer iterations",
    ylabel = "inner iterations",
    ymode = "log",
  )
  save("nnmf-inner-outer-$(name).pdf", b)

  c = Axis(
    Plots.Linear(1:length(objdec), objdec, mark = "none"),
    xlabel = "\$ k^{th}\$  \$ \\nabla f \$ Call",
    ylabel = "Objective Value",
    ymode = "log",
  )
  save("nnmf-objdec-$(name).pdf", c)
end
