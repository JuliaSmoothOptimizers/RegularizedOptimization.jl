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
  if solver ∈ [:R2DH, :TRDH]
    out_str = !options.spectral ? (options.psb ? "-PSB" : (options.andrei ? "-Andrei" : (options.wolk ? "-Wolk" : "-DBFGS"))) : "-Spec"
    out_str = (options.reduce_TR) ? out_str : string(out_str, "-noredTR")
  elseif solver == :TR && subsolver ∈ [:R2DH, :TRDH]
    out_str = !subsolver_options.spectral ? (subsolver_options.psb ? "-PSB" : (subsolver_options.andrei ? "-Andrei" : (subsolver_options.wolk ? "-Wolk" : "-DBFGS"))) : "-Spec"
    out_str = (subsolver_options.reduce_TR) ? out_str : string(out_str, "-noredTR")
  elseif solver == :R2N && subsolver ∈ [:R2DH, :TRDH]
    out_str = !subsolver_options.spectral ? (subsolver_options.psb ? "-PSB" : (subsolver_options.andrei ? "-Andrei" : (subsolver_options.wolk ? "-Wolk" : "-DBFGS"))) : "-Spec"
    out_str = (subsolver_options.reduce_TR) ? out_str : string(out_str, "-noredTR")
  else
    out_str = ""
  end
  return out_str
end
grad_evals(nlp::AbstractNLPModel) = neval_grad(nlp)
grad_evals(nls::AbstractNLSModel) = neval_jtprod_residual(nls) + neval_jprod_residual(nls)
obj_evals(nlp::AbstractNLPModel) = neval_obj(nlp)
obj_evals(nls::AbstractNLSModel) = neval_residual(nls)
function nb_prox_evals(stats, solver::Symbol)
  if solver ∈ [:TR, :R2, :TRDH, :R2DH, :R2N]
    prox_evals = sum(stats.solver_specific[:SubsolverCounter])
  else
    error("not implemented")
  end
  return prox_evals
end

acc = vec -> length(findall(x -> x < 1, vec)) / length(vec) * 100 # for SVM

function benchmark_table(
  f::AbstractNLPModel,
  selected,
  sol,
  h,
  λ,
  solvers,
  subsolvers,
  solver_options,
  subsolver_options,
  pb_name::String,
  random_seed::Int;
  tex::Bool = false,
  nls_train::Union{Nothing, AbstractNLSModel} = nothing, # for SVM
  nls_test::Union{Nothing, AbstractNLSModel} = nothing, # for SVM
)
  solver_names = [
    "$(solver)$(subsolvername(subsolver))$(options_str(opt, solver, subsolver_opt, subsolver))"
    for (solver, opt, subsolver, subsolver_opt) in
    zip(solvers, solver_options, subsolvers, subsolver_options)
  ]

  nf_evals = []
  n∇f_evals = []
  nprox_evals = []
  solver_stats = []

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
    push!(nf_evals, obj_evals(f))
    push!(n∇f_evals, grad_evals(f))
    push!(nprox_evals, nb_prox_evals(solver_out, solver))
    push!(solver_stats, solver_out)
    reset!(f)
  end

  if tex
    if length(sol) == 0
      header = [
        "solver",
        L"$f(x)$",
        L"$h(x) / \lambda$",
        L"$\sqrt{\xi / \nu}$",
        L"$\# \ f$",
        L"$\# \ \nabla f$",
        L"$\# \ prox$",
        L"$t$ ($s$)",
      ]
    else
      header = [
        "solver",
        "\$f(x)\$",
        L"$h(x)/\lambda$",
        L"$\sqrt{\xi / \nu}$",
        pb_name[1:3] == "SVM" ? L"$(Train, Test)$" : L"$\|x-x_T\|_2$",
        L"$\# \ f$",
        L"$\# \ \nabla f$",
        L"$\# \ prox$",
        L"$t$ ($s$)",
      ]
    end
  else
    if length(sol) == 0
      header = ["solver", "f(x)", "h(x)/λ", "√(ξ/ν)", "# f", "# ∇f", "# prox", "t (s)"]
    else
      header = [
        "solver",
        "f(x)",
        "h(x)/λ",
        "√ξ/√ν",
        pb_name[1:3] == "SVM" ? "(Train, Test)" : "||x-x*||",
        "# f",
        "# ∇f",
        "# prox",
        "t(s)",
      ]
    end
  end

  nh = length(header)
  n_solvers = length(solver_names)
  data = Matrix{Any}(undef, n_solvers, nh)
  for i = 1:n_solvers
    sname = solver_names[i]
    solver_out = solver_stats[i]
    x = solver_out.solution
    fx = solver_out.solver_specific[:Fhist][end]
    hx = solver_out.solver_specific[:Hhist][end]
    ξ = solver_out.dual_feas
    nf = nf_evals[i]
    n∇f = n∇f_evals[i]
    nprox = nprox_evals[i]
    t = solver_out.elapsed_time
    if length(sol) == 0
      data[i, :] .= [sname, fx, hx / λ, ξ, nf, n∇f, nprox, t]
    else
      if pb_name[1:3] == "SVM"
        string(round(t, digits = 2))
        err = "($(
          round(acc(residual(nls_train, solver_out.solution)), digits=1)), $(
            round(acc(residual(nls_test, solver_out.solution)), digits = 1)))"
      else
        err = norm(x - sol)
      end
      data[i, :] .= [sname, fx, hx / λ, ξ, err, nf, n∇f, nprox, t]
    end
  end

  h_format = h isa NormL0 ? "%i" : "%7.1e"
  if length(sol) == 0
    print_formats = ft_printf(["%s", "%7.2e", h_format, "%7.1e", "%i", "%i", "%i", "%7.1e"], 1:nh)
  else
    if pb_name[1:3] == "SVM"
      print_formats =
        ft_printf(["%s", "%7.2e", h_format, "%7.1e", "%7s", "%i", "%i", "%i", "%7.1e"], 1:nh)
    else
      print_formats =
        ft_printf(["%s", "%7.2e", h_format, "%7.1e", "%7.1e", "%i", "%i", "%i", "%7.1e"], 1:nh)
    end
  end

  title = "$pb_name $(modelname(f)) $(typeof(h).name.name)"
  if (length(sol) != 0) && pb_name[1:3] != "SVM"
    title = string(title, " \$f(x_T) = $(@sprintf("%.2e", obj(model, sol)))\$")
  end
  if tex
    pretty_table(
      data;
      header = header,
      title = title,
      backend = Val(:latex),
      formatters = (
        print_formats,
        (v, i, j) -> (j == 1 ? v : SolverBenchmark.safe_latex_AbstractFloat(v)),
      ),
    )
  else
    pretty_table(data; header = header, title = title, formatters = (print_formats,), display_size = (-1,-1))
  end
  return solver_names, solver_stats
end

# λ = norm(grad(model, rand(model.meta.nvar)), Inf) / 100000
# h = NormL1(λ)
# benchmark_table(f, selected, [], h, λ, solvers, subsolvers, solver_options, subsolver_options,
#                 "NNMF with m = $m, n = $n, k = $k, ν = 1.0e-3,")

# header = ["TR LSR1 L0Box", "R2 LSR1 L0Box", "LM L0Box", "LMTR L0Box"]
# TR_out = TR(f, h, χ, options, x0 = f.meta.x0)
# n∇f_TR = neval_grad(f)
# prox_evals_TR = sum(TR_out.solver_specific[:SubsolverCounter])
# reset!(f)
# R2_out = R2(f, h, options, x0 = f.meta.x0)
# n∇f_R2 = neval_grad(f)
# prox_evals_R2 = R2_out.iter
# reset!(f)
# LM_out = LM(nls_model, h, options, x0 = nls_model.meta.x0)
# n∇f_LM = neval_jtprod_residual(nls_model) + neval_jprod_residual(nls_model)
# prox_evals_LM = sum(LM_out.solver_specific[:SubsolverCounter])
# reset!(nls_model)
# LMTR_out = LMTR(nls_model, h, χ, options, x0 = nls_model.meta.x0)
# n∇f_LMTR = neval_jtprod_residual(nls_model) + neval_jprod_residual(nls_model)
# prox_evals_LMTR = sum(LMTR_out.solver_specific[:SubsolverCounter])
# reset!(nls_model)
# n∇f_evals = [n∇f_TR, n∇f_R2, n∇f_LM, n∇f_LMTR]
# nprox_evals = [prox_evals_TR, prox_evals_R2, prox_evals_LM, prox_evals_LMTR]

# solver_stats = [TR_out, R2_out, LM_out, LMTR_out]
