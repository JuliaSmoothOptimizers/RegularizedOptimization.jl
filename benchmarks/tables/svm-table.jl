include("regulopt-tables.jl")
include("regulopt-plots.jl")
using MLDatasets

random_seed = 1234
Random.seed!(random_seed)
nlp_train, nls_train, sol_train = RegularizedProblems.svm_train_model()
nlp_test, nls_test, sol_test = RegularizedProblems.svm_test_model()
f = LBFGSModel(nlp_train)
f_test = LBFGSModel(nlp_test)
λ = 1.0e-1 #norm(grad(model, rand(model.meta.nvar)), Inf) / 10
h = NormL1(λ)

ν = 1.0e0
verbose = 0 #10
ϵ = 1.0e-4
ϵi = 1.0e-3
ϵri = 1.0e-6
maxIter = 1000
maxIter_inner = 100
Mmonotone = 10
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
  1:(f.meta.nvar),
  (sol_train, sol_test),
  h,
  λ,
  solvers[subset],
  subsolvers[subset],
  solver_options[subset],
  subsolver_options[subset],
  "SVM with ν = $ν, λ = $λ",
  random_seed,
  nls_train = nls_train,
  nls_test = nls_test,
  tex = true,
);

subset = [8, 9, 10, 11, 12, 13, 14]

opt =
  ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = verbose, maxIter = maxIter, spectral = true, Mmonotone = 0)
optPSB =
  ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = verbose, maxIter = maxIter, spectral = false, psb = true, Mmonotone = 0)
opt5 =
  ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = verbose, maxIter = maxIter, spectral = true, Mmonotone = 5)
sopt = ROSolverOptions(spectral = true, ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner, Mmonotone = 0)
soptPSB = ROSolverOptions(spectral = false, psb = true, ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner, Mmonotone = 0)
solvers2 = [:R2, :TRDH, :TRDH, :TR, :TR]
subsolvers2 = [:None, :None, :None, :R2, :TRDH]
solver_options2 = [opt, opt, opt5, opt, opt]
subsolver_options2 = [sopt, sopt, sopt, sopt, soptPSB]

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
#   joinpath(path, "svm.tikz"),
#   p;
#   include_preamble = true,
# )