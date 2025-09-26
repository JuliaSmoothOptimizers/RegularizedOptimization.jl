using LinearAlgebra: length
using LinearAlgebra, Random, Test
using ProximalOperators
using ADNLPModels,
  OptimizationProblems,
  OptimizationProblems.ADNLPProblems,
  NLPModels,
  NLPModelsModifiers,
  RegularizedProblems,
  RegularizedOptimization,
  SolverCore

for (root, dirs, files) in walkdir(@__DIR__)
  for file in files
    if isnothing(match(r"^test-.*\.jl$", file))
      continue
    end
    title = titlecase(replace(splitext(file[6:end])[1], "-" => " "))
    @testset "$title" begin
      include(file)
    end
  end
end
