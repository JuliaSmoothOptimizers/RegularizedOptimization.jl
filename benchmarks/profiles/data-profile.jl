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

function data_prof_nnmf(
  solvers,
  solver_names,
  nb_prob::Int,
  random_seed::Int;
  measured::Symbol = :obj, # set to :grad to eval grad
)

  m, n, k = 100, 50, 5
  λ = 1.0e-1 
  h = NormL1(λ)
  n_solvers = length(solvers)
  objdecs = Vector{Float64}[] 

  current_seed = random_seed
  nb_data_min = Int(10^16) # min number of data for each solver on every pb
  for i=1:nb_prob
    Random.seed!(current_seed)
    model, nls_model, A, selected = nnmf_model(m, n, k)
    f = LSR1Model(model)
    @info "pb $i"
    for (solver, name) in zip(solvers, solver_names)
      @info "pb: $i  solver: $name"
      args = name == "R2" ? () : (NormLinf(1.0),)
      solver_out = solver(f, h, selected)
      objdec = solver_out.solver_specific[:Fhist] + solver_out.solver_specific[:Hhist]
      measured == :grad && unique!(objdec)
      nb_data_min = min(nb_data_min, length(objdec))
      push!(objdecs, objdec)
      reset!(f)
    end
  end

  data = zeros(nb_data_min, nb_prob, n_solvers)
  idx_objdec = 1
  for i=1:nb_prob
    for idx_solv in 1:n_solvers
      objdec = objdecs[idx_objdec]
      data[:, i, idx_solv] .= objdec[1:nb_data_min]
      idx_objdec += 1
    end
  end

  data_profile(PlotsBackend(), data, ones(nb_prob), solver_names, legend=:topleft, τ= 1.0e-2)
end

ν = 1.0
ϵ = 1.0e-4
ϵi = 1.0e-4
ϵri = 1.0e-4
maxIter = 200
maxIter_inner = 100
function TR_R2(f, h, selected; ϵ = ϵ, ϵi = ϵi, ϵri = ϵri, maxIter = maxIter, maxIter_inner = maxIter_inner)
  opt = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = 0, maxIter = maxIter)
  sub_opt = ROSolverOptions(ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner)
  TR(f, h, NormLinf(1.0), opt, x0 = f.meta.x0, subsolver_options = sub_opt, selected = selected)
end

function TR_TRDH(f, h, selected; ϵ = ϵ, ϵi = ϵi, ϵri = ϵri, maxIter = maxIter, maxIter_inner = maxIter_inner)
  opt = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = 0, maxIter = maxIter)
  sub_opt = ROSolverOptions(ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner)
  TR(f, h, NormLinf(1.0), opt, x0 = f.meta.x0, subsolver_options = sub_opt, selected = selected, subsolver = TRDH)
end

data_prof_nnmf(
  [TR_R2, TR_TRDH],
  ["TR-R2", "TR-TRDH"],
  5,
  1234;
  measured = :grad, # set to :grad to eval grad
)