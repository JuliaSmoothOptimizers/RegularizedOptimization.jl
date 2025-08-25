using NLPModels, ADNLPModels, OptimizationProblems, OptimizationProblems.ADNLPProblems
using ProximalOperators
using RegularizedProblems, RegularizedOptimization

problem_name = "HS8"
nlp = hs8()
@assert !has_bounds(nlp)
@assert equality_constrained(nlp)

h = NormL1(1.0)

stats = AL(nlp, h, atol = 1e-3, verbose = 1)
print(stats)

regnlp = RegularizedNLPModel(nlp, h)
stats = AL(regnlp, atol = 1e-3, verbose = 1)
print(stats)

macro wrappedallocs(expr)
  kwargs = [a for a in expr.args if isa(a, Expr)]
  args = [a for a in expr.args if isa(a, Symbol)]

  argnames = [gensym() for a in args]
  kwargs_dict = Dict{Symbol, Any}(a.args[1] => a.args[2] for a in kwargs if a.head == :kw)
  quote
    function g($(argnames...); kwargs_dict...)
      @allocated $(Expr(expr.head, argnames..., kwargs...))
    end
    $(Expr(:call, :g, [esc(a) for a in args]...))
  end
end

solver = ALSolver(regnlp)
stats = RegularizedExecutionStats(regnlp)
println("NB ALLOCS = $(@wrappedallocs solve!(solver, regnlp, stats, atol = 1e-3))")
#print(stats)
finalize(nlp)

error("done")
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

callback =
  (regnlp, solver, stats) -> begin
    @info "iter $(stats.iter), f $(stats.solver_specific[:smooth_obj]), h $(stats.solver_specific[:nonsmooth_obj])"
  end
stats = AL(nlp, h, atol = 1e-6, verbose = 1, callback = callback)
print(stats)

finalize(nlp)