using LinearAlgebra: length
using LinearAlgebra, Random
using ProximalOperators
using NLPModels, NLPModelsModifiers, RegularizedProblems, RegularizedOptimization, SolverCore

Random.seed!(123)   

compound = 1
nz = 10 * compound
bpdn, bpdn_nls, sol = bpdn_model(compound)
λ = norm(grad(bpdn, zeros(bpdn.meta.nvar)), Inf) / 10

h = NormL0(λ)
reg_nlp = RegularizedNLPModel(LBFGSModel(bpdn), h)
solver = R2NSolver(reg_nlp)
stats = RegularizedExecutionStats(reg_nlp)

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

solve!(solver, reg_nlp, stats, σk = 1.0, atol = 1e-6, rtol = 1e-6, verbose = 1, sub_kwargs = (max_iter = 100, ))
#println(@report_opt target_modules=(RegularizedOptimization,) solve!(solver, reg_nlp, stats, σk = 1.0, atol = 1e-6, rtol = 1e-6))
error("done")

avg_num_iter_compute_true = 0
avg_num_iter_compute_false = 0
avg_time_compute_true = 0
avg_time_compute_false = 0

callback = (nlp, solver, stats) -> begin
  if stats.iter == 0
    set_solver_specific!(stats, :total_iter, 0)
  else
    set_solver_specific!(stats, :total_iter, stats.solver_specific[:total_iter] + solver.substats.iter)
  end
end

Nrun = 50
for i = 1:Nrun
    println(i)
    local x0 = 100 * randn(Float64, reg_nlp.model.meta.nvar)

    solve!(solver, reg_nlp, stats; x = x0, σk = 1.0, atol = 1e-8, rtol = 1e-8,
           compute_opnorm = true, callback = callback)
    global avg_num_iter_compute_true += stats.solver_specific[:total_iter]
    global avg_time_compute_true += stats.elapsed_time

    solve!(solver, reg_nlp, stats; x = x0, σk = 1.0, atol = 1e-8, rtol = 1e-8,
           compute_opnorm = false, callback = callback)
    global avg_num_iter_compute_false += stats.solver_specific[:total_iter]
    global avg_time_compute_false += stats.elapsed_time
end

println("Average number of iterations with compute_opnorm = true : $(avg_num_iter_compute_true / Nrun)")
println("Average computation time with compute_opnorm = true : $(avg_time_compute_true / Nrun)")
println("")
println("Average number of iterations with compute_opnorm = false : $(avg_num_iter_compute_false / Nrun)")
println("Average computation time with compute_opnorm = false : $(avg_time_compute_false / Nrun)")

