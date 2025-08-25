using RegularizedOptimization, RegularizedProblems, NLPModels, NLPModelsModifiers, LinearAlgebra, ProximalOperators, Random
using JET
Random.seed!(1234)

compound = 1 
_, bpdn, _ = bpdn_model(compound)
options = ROSolverOptions(ϵa = 1e-6, ϵr = 1e-6, verbose = 10, reduce_TR = true)
λ = norm(grad(bpdn, zeros(bpdn.meta.nvar)), Inf) / 10
x0 = zeros(bpdn.meta.nvar)
χ = NormLinf(1.0)
h  = NormL1(λ)
nlp = bpdn

out = LM(nlp, h, options, x0 = x0, subsolver = R2DHSolver)
reg_nlp = RegularizedNLSModel(nlp, h)
solver = LMSolver(reg_nlp, subsolver = R2DHSolver)#, subsolver = TRDHSolver)
stats = RegularizedExecutionStats(reg_nlp)
solve!(solver, reg_nlp, stats, x=x0, atol = 1e-6, rtol = 1e-6, verbose = 1)


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

println(@wrappedallocs solve!(solver, reg_nlp, stats, atol = 1e-6, rtol = 1e-6))
error("done")
#println(@wrappedallocs solve!(solver, reg_nlp, stats))

#solve!(solver, reg_nlp, stats, x=x0, atol = 1e-6, rtol = 1e-6, verbose = 1)
reset!(solver.subpb.model.B)
out = TR(nlp, h, χ, options, x0 = x0)#, subsolver = TRDH)
error("done")
#nlp.model.D.d = [1.0]
nlp.op.d = [1.0]
solver = TRDHSolver(reg_nlp, χ = χ)
stats = RegularizedExecutionStats(reg_nlp)
solve!(solver, reg_nlp, stats, x=x0, atol = 1e-8, rtol = 1e-8, verbose = 1, reduce_TR = true)



println(@wrappedallocs solve!(solver, reg_nlp, stats))
#@report_opt solve!(solver, reg_nlp, stats)

