using NLPModels, CUTEst
using ProximalOperators
using RegularizedOptimization

problem_name = "HS8"
nlp = CUTEstModel(problem_name)
@assert !has_bounds(nlp)
@assert equality_constrained(nlp)

h = NormL1(1.0)

stats = AL(nlp, h, atol = 1e-6, verbose = 1)
print(stats)

using RegularizedProblems

regnlp = RegularizedNLPModel(nlp, h)
stats = AL(regnlp, atol = 1e-6, verbose = 1)
print(stats)

solver = ALSolver(regnlp)
stats = solve!(solver, regnlp, atol = 1e-6, verbose = 1)
print(stats)

using SolverCore

stats = GenericExecutionStats(nlp)
solver = ALSolver(regnlp)
stats = solve!(solver, regnlp, stats, atol = 1e-6, verbose = 1)
print(stats)

callback =
  (regnlp, solver, stats) -> begin
    @info "iter $(stats.iter), obj $(stats.objective), status $(stats.status)"
  end
stats = AL(nlp, h, atol = 1e-6, verbose = 1, callback = callback)
print(stats)

finalize(nlp)