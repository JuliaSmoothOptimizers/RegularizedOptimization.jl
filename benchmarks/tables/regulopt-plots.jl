using PGFPlotsX
using Colors
using LaTeXStrings
using PrettyTables, LaTeXStrings
using Random
using LinearAlgebra
using ProximalOperators
using NLPModels,
  NLPModelsModifiers,
  RegularizedProblems,
  RegularizedOptimization,
  ShiftedProximalOperators,
  SolverBenchmark
using Printf

# utils for extracting stats / display table
modelname(nlp::LSR1Model) = "LSR1"
modelname(nlp::LBFGSModel) = "LBFGS"
modelname(nlp::SpectralGradientModel) = "SpectralGradient"
modelname(nlp::DiagonalQNModel) = "DiagonalQN"
subsolvername(subsolver::Symbol) = subsolver == :None ? "" : string("-", subsolver)
function options_str(
  options::ROSolverOptions,
  solver::Symbol,
  subsolver_options::ROSolverOptions,
  subsolver::Symbol,
)
  if solver == :TRDH
    out_str = !options.spectral ? (options.psb ? "-PSB" : "-Andrei") : "-Spec"
    out_str = (options.reduce_TR) ? out_str : string(out_str, "-noredTR")
  elseif solver == :TR && subsolver == :TRDH
    out_str = !subsolver_options.spectral ? (subsolver_options.psb ? "-PSB" : "-Andrei") : "-Spec"
    out_str = (subsolver_options.reduce_TR) ? out_str : string(out_str, "-noredTR")
  else
    out_str = ""
  end
  return out_str
end

function benchmark_plot(
  f::AbstractNLPModel,
  selected,
  h,
  solvers,
  subsolvers,
  solver_options,
  subsolver_options,
  random_seed::Int;
  measured::Symbol = :obj, # set to :grad to eval grad
  xmode::String = "log",
  ymode::String = "log",
)
  solver_names = [
    "$(solver)$(subsolvername(subsolver))$(options_str(opt, solver, subsolver_opt, subsolver))"
    for (solver, opt, subsolver, subsolver_opt) in
    zip(solvers, solver_options, subsolvers, subsolver_options)
  ]
  n_solvers = length(solver_names)
  objdecs = Vector{Float64}[] 
  coords = Coordinates{2}[]
  obj_min = Float64(Inf)

  reset!(f)
  for (solver, subsolver, opt, sub_opt) in
      zip(solvers, subsolvers, solver_options, subsolver_options)
    @info " using $solver with subsolver = $subsolver"
    args = solver == :R2 ? () : (NormLinf(1.0),)
    Random.seed!(random_seed)
    if subsolver == :None
      solver_out = eval(solver)(f, h, args..., opt, x0 = f.meta.x0, selected = selected)
    else
      solver_out = eval(solver)(
        f,
        h,
        args...,
        opt,
        x0 = f.meta.x0,
        subsolver = eval(subsolver),
        subsolver_options = sub_opt,
        selected = selected,
      )
    end
    objdec = solver_out.solver_specific[:Fhist] + solver_out.solver_specific[:Hhist]
    measured == :grad && (objdec = objdec[solver_out.solver_specific[:IterSucc]])
    obj_min = min(minimum(objdec), obj_min)
    objdec[end] = obj(f, solver_out.solution) + h(solver_out.solution[selected])
    push!(objdecs, objdec)
    reset!(f)
  end
  for i in 1:length(objdecs)
    objdec = objdecs[i]
    println(length(objdec))
    push!(
      coords,
      # Coordinates([(k, objdec[k] - obj_min) for k in 1:length(objdec)]),
      Coordinates([(k, objdec[k] - obj_min) for k in 1:length(objdec)][1:10:end]),
    )
  end

  colors = distinguishable_colors(
    n_solvers,
    [RGB(1, 0, 0), RGB(0, 0, 1), RGB(0, 1, 0), RGB(0, 0, 0)],#, RGB(1, 0, 1)],
    dropseed = false,
  )
  l_plots = [@pgf Plot({color = colors[i]}, coords[i]) for i in 1:n_solvers]
  
  @pgf Axis(
    {
      xlabel = "iterations",
      ylabel = L"$(f + h)(x_k)$",
      ymode = ymode,
      xmode = xmode,
      no_markers,
      legend_style = {
        nodes={scale=0.8},
        font = "\\tiny",
      },
      legend_pos="south west",
      # legend_pos="south east",
    },
    Tuple(l_plots)...,
    Legend(solver_names),
  )
end
