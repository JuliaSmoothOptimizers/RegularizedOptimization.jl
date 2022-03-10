using PGFPlots
using SolverCore

function plot_sol(x; legend = "generic")
  Plots.Linear(1:length(x), x, mark = "none", legendentry = legend)
end

function plot_sols()
  Axis(
    xlabel = "index",
    ylabel = "signal",
    legendStyle = "at={(1.0,1.0)}, anchor=north east, draw=none, font=\\scriptsize",
  )
end

function plot_sols(x; kwargs...)
  a = plot_sols()
  push!(a, plot_sol(x; kwargs...))
  a
end

function plot_inner_outer(comp; legend = "generic")
  Plots.Linear(1:length(comp), comp, mark = "none", legendentry = legend)
end

function plot_inner_outers()
  Axis(
    xlabel = "outer iterations",
    ylabel = "inner iterations",
    ymode = "log",
    legendStyle = "at={(1.0,1.0)}, anchor=north east, draw=none, font=\\scriptsize",
  )
end

function plot_inner_outers(comp; kwargs...)
  a = plot_inner_outers()
  push!(a, plot_inner_outer(comp; kwargs...))
  a
end

function plot_obj_decrease(objdec; legend = "generic")
  Plots.Linear(1:length(objdec), objdec, legendentry = legend)
end

function plot_obj_decreases()
  Axis(
    xlabel = "\$ k^{th}\$  \$ \\nabla f \$ Call",
    ylabel = "Objective Value",
    ymode = "log",
    legendStyle = "at={(1.0,1.0)}, anchor=north east, draw=none, font=\\scriptsize",
  )
end

function plot_obj_decreases(objdec; kwargs...)
  a = plot_obj_decreases()
  push!(a, plot_obj_decrease(objdec; kwargs...))
  a
end

function plot_bpdn(outstruct::GenericExecutionStats, sol::AbstractVector, name::AbstractString = "generic")
  a = plot_sols(sol, legend = "exact")
  x = outstruct.solution
  push!(a, plot_sol(x, legend = "computed"))
  save("bpdn-$(name).pdf", a)

  Comp_pg = outstruct.solver_specific[:SubsolverCounter]
  save("bpdn-inner-outer-$(name).pdf", plot_inner_outers(Comp_pg))

  objdec = outstruct.solver_specific[:Fhist] + outstruct.solver_specific[:Hhist]
  save("bpdn-objdec-$(name).pdf", plot_obj_decreases(objdec))
end

function plot_bpdn(outstructs, sol, names)
  a = plot_sols(sol, legend = "exact")
  b = plot_obj_decreases()
  for (outstruct, name) ∈ zip(outstructs, names)
    x = outstruct.solution
    push!(a, plot_sol(x, legend = name))

    objdec = outstruct.solver_specific[:Fhist] + outstruct.solver_specific[:Hhist]
    push!(b, plot_obj_decrease(objdec, legend = name))
  end

  save("bpdn-solutions.pdf", a)
  save("bpdn-decreases.pdf", b)
  nothing
end

