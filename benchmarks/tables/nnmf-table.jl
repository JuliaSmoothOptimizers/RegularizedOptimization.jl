include("regulopt-tables.jl")
include("regulopt-plots.jl")

random_seed = 1234
Random.seed!(random_seed)
m, n, k = 100, 50, 5
model, nls_model, A, selected = nnmf_model(m, n, k)
f = LSR1Model(model)
λ = 1.0e-3 # norm(grad(model, rand(model.meta.nvar)), Inf) / 100
h = NormL0(λ)
ν = 1.0
ϵ = 1.0e-5
ϵi = 1.0e-3
ϵri = 1.0e-6
maxIter = 500
maxIter_inner = 100
Mmonotone = 10
verbose = 0 #10
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

subset = 1:length(solvers)

benchmark_table(
  f,
  selected,
  [],
  h,
  λ,
  solvers[subset],
  subsolvers[subset],
  solver_options[subset],
  subsolver_options[subset],
  "NNMF with m = $m, n = $n, k = $k, ν = $ν, λ = $λ, M = $Mmonotone",
  random_seed,
  tex = true,
);

subset = [8, 9, 10, 11, 12, 13, 14]

opt =
  ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = verbose, maxIter = maxIter, spectral = true, Mmonotone = 0)
optPSB =
  ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = verbose, maxIter = maxIter, spectral = false, psb = true, Mmonotone = 0)
opt2 =
  ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = verbose, maxIter = maxIter, spectral = true, Mmonotone = 2)
sopt = ROSolverOptions(spectral = true, ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner, Mmonotone = 0)
# subset = [1,2,3,4,5,6,7]
solvers2 = [:TRDH, :TRDH, :TR, :TR]
subsolvers2 = [:None, :None, :R2, :TRDH]
solver_options2 = [opt, opt2, opt, opt]
subsolver_options2 = [sopt, sopt, sopt, sopt]

p = benchmark_plot(
  f,
  1:(f.meta.nvar),
  h,
  solvers2,
  subsolvers2,
  solver_options2,
  subsolver_options2,
  random_seed;
  xmode = "linear",
  ymode = "log", 
)

# path = raw"C:\Users\Geoffroy Leconte\Documents\doctorat\biblio\papiers\indef-pg\figs\objplots"
# pgfsave(
#   joinpath(path, "nnmf.tikz"),
#   p;
#   include_preamble = true,
# )