include("regulopt-tables.jl")
using ADNLPModels, DifferentialEquations

display_sol = true

Random.seed!(1234)
data, simulate, resid, misfit, x0 = RegularizedProblems.FH_smooth_term()
model = ADNLPModel(misfit, ones(5))
f = LBFGSModel(model)

λ = 1.0e1
h = NormL1(λ)
ν = 1.0e0
verbose = 0 #10
maxIter = 500
maxIter_inner = 200 # max iter for subsolver
ϵ = 1.0e-4
ϵi = 1.0e-3
ϵri = 1.0e-6
options =
  ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = verbose, maxIter = maxIter, spectral = true)
optionsbis =
  ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = verbose, maxIter = maxIter, spectral = true, reduce_TR = false)
options2 = ROSolverOptions(spectral = false, psb = true, ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner)
options2bis = ROSolverOptions(spectral = false, psb = true, ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner, reduce_TR = false)
options3 = ROSolverOptions(spectral = false, psb = false, ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner)
options3bis = ROSolverOptions(spectral = false, psb = false, ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner, reduce_TR = false)
options4 = ROSolverOptions(spectral = true, ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner)
options4bis =
  ROSolverOptions(spectral = true, ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner, reduce_TR = false)
options5 = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = true,
)
options5bis = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = true,
  reduce_TR = false,
)
options6 = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = false,
)
options6bis = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = false,
  reduce_TR = false,
)

solvers = [:R2, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TR, :TR, :TR, :TR, :TR, :TR, :TR]
subsolvers = [:None, :None, :None, :None, :None, :None, :None, :R2, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH]
solver_options = [
  options,
  options,
  optionsbis,
  options5,
  options5bis,
  options6,
  options6bis,
  options,
  options,
  options,
  options,
  options,
  options,
  options,
  options,
]
subsolver_options = [
  options2,
  options2,
  options2,
  options2,
  options2,
  options2,
  options2,
  options2,
  options2,
  options2bis,
  options3,
  options3bis,
  options4,
  options4bis,
] # n'importe lequel si subsolver = :None
subset = 2:length(solvers) # issues with R2 alone

names, stats = benchmark_table(
  f,
  1:(f.meta.nvar),
  x0,
  h,
  λ,
  solvers[subset],
  subsolvers[subset],
  solver_options[subset],
  subsolver_options[subset],
  "FH with ν = $ν, λ = $λ",
  tex = true,
);

if display_sol
  data = zeros(length(subset) + 1, 5)
  data[1, :] .= x0
  for i=1:length(subset)
    data[i+1, :] .= stats[i].solution
  end
  pretty_table(
    data;
    header = [L"$x_1$", L"$x_2$", L"$x_3$", L"$x_4$", L"$x_5$"],
    row_names = vcat(["True"], names),
    title = "Solution FH",
    formatters = ft_printf("%1.2f"),
    backend = Val(:latex),
  )
end
