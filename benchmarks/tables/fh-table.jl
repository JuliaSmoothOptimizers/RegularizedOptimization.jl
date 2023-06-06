include("regulopt-tables.jl")
include("regulopt-plots.jl")
using ADNLPModels, DifferentialEquations

display_sol = true

random_seed = 12345
Random.seed!(random_seed)

cstr = true
ctr_val = cstr ? 0.5 : -Inf
lvar = [-Inf, ctr_val, -Inf, -Inf, -Inf]
uvar = fill(Inf, 5)
data, simulate, resid, misfit, x0 = RegularizedProblems.FH_smooth_term()
model = ADNLPModel(misfit, ones(5), lvar, uvar)
f = LBFGSModel(model)
λ = cstr ? 4.0e1 : 1.0e1

h = cstr ? NormL1(λ) : NormL0(λ)
ν = 1.0e0
verbose = 0 #10
maxIter = 500
maxIter_inner = 200 # max iter for subsolver
ϵ = 1.0e-4
ϵi = 1.0e-3
ϵri = 1.0e-6
Mmonotone = 5
options =
  ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = verbose, maxIter = maxIter, spectral = true, Mmonotone = Mmonotone)
options_nrTR = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = true,
  reduce_TR = false,
  Mmonotone = Mmonotone,
)
options2 = ROSolverOptions(spectral = false, psb = true, ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner, Mmonotone = Mmonotone)
options2_nrTR = ROSolverOptions(
  spectral = false,
  psb = true,
  ϵa = ϵi,
  ϵr = ϵri,
  maxIter = maxIter_inner,
  reduce_TR = false,
  Mmonotone = Mmonotone,
)
options3 =
  ROSolverOptions(spectral = false, psb = false, ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner, Mmonotone = Mmonotone)
options3_nrTR = ROSolverOptions(
  spectral = false,
  psb = false,
  ϵa = ϵi,
  ϵr = ϵri,
  maxIter = maxIter_inner,
  reduce_TR = false,
  Mmonotone = Mmonotone,
)
options4 = ROSolverOptions(spectral = true, ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner, Mmonotone = Mmonotone)
options4_nrTR =
  ROSolverOptions(spectral = true, ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner, reduce_TR = false, Mmonotone = Mmonotone)
options5 = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = true,
  Mmonotone = Mmonotone,
)
options5_nrTR = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = true,
  reduce_TR = false,
  Mmonotone = Mmonotone,
)
options6 = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = false,
  Mmonotone = Mmonotone,
)
options6_nrTR = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = false,
  reduce_TR = false,
  Mmonotone = Mmonotone,
)

solvers = [:R2, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TR, :TR, :TR, :TR, :TR, :TR, :TR]
subsolvers =
  [:None, :None, :None, :None, :None, :None, :None, :R2, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH]
solver_options = [
  options,
  options,
  options_nrTR,
  options5,
  options5_nrTR,
  options6,
  options6_nrTR,
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
  options2_nrTR,
  options3,
  options3_nrTR,
  options4,
  options4_nrTR,
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
  "FH with ν = $ν, λ = $λ, M = $Mmonotone",
  random_seed,
  tex = true,
);

if display_sol
  data = zeros(length(subset) + 1, 5)
  data[1, :] .= x0
  for i = 1:length(subset)
    data[i + 1, :] .= stats[i].solution
  end
  pretty_table(
    data;
    header = [L"$x_1$", L"$x_2$", L"$x_3$", L"$x_4$", L"$x_5$"],
    row_names = vcat(["True"], names),
    title = "Solution FH",
    formatters = ft_printf("%1.2f"),
    # backend = Val(:latex),
  )
end

subset = [8, 9, 10, 11, 12, 13, 14]

p = benchmark_plot(
  f,
  1:(f.meta.nvar),
  h,
  solvers[subset],
  subsolvers[subset],
  solver_options[subset],
  subsolver_options[subset],
  random_seed;
  measured = :grad,
  xmode = "linear",
  ymode = "log", 
)