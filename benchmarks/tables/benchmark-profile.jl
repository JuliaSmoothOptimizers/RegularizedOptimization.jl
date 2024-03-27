using LaTeXStrings
using PrettyTables, LaTeXStrings
using Random
using LinearAlgebra
using ProximalOperators
using Plots
using BenchmarkProfiles
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

function benchmark_prof(
  pb::Symbol, #:nnmf, :bpdn
  solvers,
  solver_names,
  nb_prob::Int,
  random_seed::Int;
  measured::Symbol = :obj, # set to :grad to eval grad
)

  if pb == :nnmf
    m, n, k = 100, 50, 5
    λ = 1.0e-1
  elseif pb == :bpdn
    compound = 1
  else
    error("Problem not supported")
  end
  n_solvers = length(solvers)
  data = zeros(nb_prob, n_solvers)

  current_seed = random_seed
  nb_data_min = Int(10^16) # min number of data for each solver on every pb
  for i=1:nb_prob
    current_seed = random_seed + i - 1
    Random.seed!(current_seed)
    if pb == :nnmf
      model, nls_model, A, selected = nnmf_model(m, n, k)
      h = NormL0(λ)
    elseif pb == :bpdn
      model, nls_model, sol = bpdn_model(compound, bounds = false)
      selected = 1:length(sol)
      λ = norm(grad(model, zeros(model.meta.nvar)), Inf) / 10
      reset!(model)
      h = NormL0(λ)
    end
    f = LSR1Model(model)
    @info "pb $i"
    for (j, solver, name) in zip(1:n_solvers, solvers, solver_names)
      solver_out = solver(f, h, selected)
      @info "pb: $i  solver: $name  status = $(solver_out.status)  obj = $(solver_out.objective)"
      if solver_out.status == :first_order
        data[i, j] = neval_grad(f)
      else
        data[i, j] = +Inf
      end
      reset!(f)
    end
  end

  performance_profile(PlotsBackend(), data, solver_names, legend=:bottomright, title = String(measured))
end

ν = 1.0
ϵ = 1.0e-5
ϵi = 1.0e-3
ϵri = 1.0e-6
maxIter = 2000
maxIter_inner = 100
function TR_R2(f, h, selected; ϵ = ϵ, ϵi = ϵi, ϵri = ϵri, maxIter = maxIter, maxIter_inner = maxIter_inner)
  opt = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = 0, maxIter = maxIter)
  sub_opt = ROSolverOptions(ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner)
  TR(f, h, NormLinf(1.0), opt, x0 = f.meta.x0, subsolver_options = sub_opt, selected = selected)
end

function TR_TRDH_PSB(f, h, selected; ϵ = ϵ, ϵi = ϵi, ϵri = ϵri, maxIter = maxIter, maxIter_inner = maxIter_inner)
  opt = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = 0, maxIter = maxIter)
  sub_opt = ROSolverOptions(ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner, psb = true, reduce_TR = false)
  TR(f, h, NormLinf(1.0), opt, x0 = f.meta.x0, subsolver_options = sub_opt, selected = selected, subsolver = TRDH)
end

function TR_TRDH_Andrei(f, h, selected; ϵ = ϵ, ϵi = ϵi, ϵri = ϵri, maxIter = maxIter, maxIter_inner = maxIter_inner)
  opt = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = 0, maxIter = maxIter)
  sub_opt = ROSolverOptions(ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner, spectral = false, psb = false, reduce_TR = false)
  TR(f, h, NormLinf(1.0), opt, x0 = f.meta.x0, subsolver_options = sub_opt, selected = selected, subsolver = TRDH)
end

function TR_TRDH_Spec(f, h, selected; ϵ = ϵ, ϵi = ϵi, ϵri = ϵri, maxIter = maxIter, maxIter_inner = maxIter_inner)
  opt = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = 0, maxIter = maxIter)
  sub_opt = ROSolverOptions(ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner, spectral = true, reduce_TR = false)
  TR(f, h, NormLinf(1.0), opt, x0 = f.meta.x0, subsolver_options = sub_opt, selected = selected, subsolver = TRDH)
end

function TRDH_Spec(f, h, selected; ϵ = ϵ, ϵi = ϵi, ϵri = ϵri, maxIter = maxIter, maxIter_inner = maxIter_inner)
  opt = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, maxIter = maxIter, spectral = true, reduce_TR = false)
  TRDH(f, h, NormLinf(1.0), opt, x0 = f.meta.x0, selected = selected)
end

function R2_None(f, h, selected; ϵ = ϵ, ϵi = ϵi, ϵri = ϵri, maxIter = maxIter, maxIter_inner = maxIter_inner)
  opt = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, maxIter = maxIter, spectral = true, reduce_TR = false)
  R2(f, h, opt; x0 = f.meta.x0, selected = selected)
end


benchmark_prof(
  :nnmf,
  [TRDH_Spec, TR_R2, TR_TRDH_PSB, TR_TRDH_Andrei, TR_TRDH_Spec],
  ["TRDH_Spec", "TR-R2", "TR-TRDH-PSB", "TR_TRDH_Andrei", "TR_TRDH_Spec"],
  50,
  1234;
  measured = :grad, # set to :grad to eval grad
)

benchmark_prof(
  :bpdn,
  [R2_None, TRDH_Spec, TR_R2, TR_TRDH_PSB],
  ["R2", "TRDH-Spec", "TR-R2", "TR-TRDH-PSB"],
  50,
  1234;
  measured = :grad, # set to :grad to eval grad
)