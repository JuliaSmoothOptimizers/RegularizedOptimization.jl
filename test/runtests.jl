using JET, LinearAlgebra, Random, Test
using ProximalOperators
using ADNLPModels,
  OptimizationProblems,
  OptimizationProblems.ADNLPProblems,
  NLPModels,
  NLPModelsModifiers,
  RegularizedProblems,
  RegularizedOptimization,
  SolverCore

Random.seed!(0)
include("utils.jl")
include("callbacks.jl")

include("test-AL.jl")

include("bpdn/test-bpdn.jl")
include("bpdn/test-bpdn-bounds.jl")
include("bpdn/test-bpdn-allocs.jl")
