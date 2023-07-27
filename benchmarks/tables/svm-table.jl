include("regulopt-tables.jl")
using MLDatasets

Random.seed!(1234)
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
maxIter = 1500
maxIter_inner = 100
options =
  ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = verbose, maxIter = maxIter, spectral = true)
options_nrTR = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = true,
  reduce_TR = false,
)
options2 = ROSolverOptions(spectral = false, psb = true, ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner)
options2_nrTR = ROSolverOptions(
  spectral = false,
  psb = true,
  ϵa = ϵi,
  ϵr = ϵri,
  maxIter = maxIter_inner,
  reduce_TR = false,
)
options3 =
  ROSolverOptions(spectral = false, psb = false, ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner)
options3_nrTR = ROSolverOptions(
  spectral = false,
  psb = false,
  ϵa = ϵi,
  ϵr = ϵri,
  maxIter = maxIter_inner,
  reduce_TR = false,
)
options4 = ROSolverOptions(spectral = true, ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner)
options4_nrTR =
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
options5_nrTR = ROSolverOptions(
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
options6_nrTR = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = false,
  reduce_TR = false,
)

options7 = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = false,
  andrei = false
)

options7_nrTR = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = false,
  andrei = false,
  reduce_TR = false,
)

options8 = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = false,
  andrei = false,
  wolk = true
)

options8_nrTR = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = false,
  andrei = false,
  wolk = true,
  reduce_TR = false,
)

solvers = [:R2, :R2DH, :R2DH, :R2DH, :R2DH, :R2DH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TR, :TR, :TR, :TR, :TR, :TR, :TR, :TR, :TR, :TR, :TR, :TR, :TR, :TR]
subsolvers =
  [:None, :None, :None, :None, :None, :None, :None, :None, :None, :None, :None, :None, :None, :None, :None, :None, :R2, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :R2DH, :R2DH, :R2DH]
solver_options = [
  options,
  options,
  options5,
  options6,
  options7,
  options8,
  options,
  options_nrTR,
  options5,
  options5_nrTR,
  options6,
  options6_nrTR,
  options7,
  options7_nrTR,
  options8,
  options8_nrTR,
  options,
  options,
  options,
  options,
  options,
  options,
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
  options7,
  options7_nrTR,
  options8,
  options8_nrTR,
  options8,
  options7,
  options,
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
  nls_train = nls_train,
  nls_test = nls_test,
  tex = false,
);
