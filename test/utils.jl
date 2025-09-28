"""
    @wrappedallocs(expr)

Given an expression, this macro wraps that expression inside a new function
which will evaluate that expression and measure the amount of memory allocated
by the expression. Wrapping the expression in a new function allows for more
accurate memory allocation detection when using global variables (e.g. when
at the REPL).

This code is based on that of https://github.com/JuliaAlgebra/TypedPolynomials.jl/blob/master/test/runtests.jl

For example, `@wrappedallocs(x + y)` produces:

```julia
function g(x1, x2)
    @allocated x1 + x2
end
g(x, y)
```

You can use this macro in a unit test to verify that a function does not
allocate:

```
@test @wrappedallocs(x + y) == 0
```
"""
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

function test_solver_basic(reg_nlp::AbstractRegularizedNLPModel, solver_sym::Symbol; expected_output = :first_order, constructor_kwargs = NamedTuple(), solver_kwargs = NamedTuple())
  
  # Test calling with a regularized NLP
  solver = eval(solver_sym)
  reset!(reg_nlp.model)
  out1 = solver(reg_nlp; merge(constructor_kwargs, solver_kwargs)...)
  @test typeof(out1.solution) == typeof(reg_nlp.model.meta.x0)
  @test length(out1.solution) == reg_nlp.model.meta.nvar
  @test typeof(out1.dual_feas) == eltype(out1.solution)
  @test out1.status == expected_output

  if expected_output == :max_eval
    @test neval_obj(reg_nlp.model) == solver_kwargs[:max_eval] + 1
  end

  if expected_output == :max_iter
    @test out1.iter == solver_kwargs[:max_iter] + 1
  end

  # Test calling with solve!
  solver_constructor = eval(Symbol(string(solver_sym), "Solver"))
  workspace = solver_constructor(reg_nlp; constructor_kwargs...)
  out2 = RegularizedExecutionStats(reg_nlp)

  solve!(workspace, reg_nlp, out2; solver_kwargs...)
  @test typeof(out2.solution) == typeof(reg_nlp.model.meta.x0)
  @test length(out2.solution) == reg_nlp.model.meta.nvar
  @test typeof(out2.dual_feas) == eltype(out2.solution)
  @test out2.status == expected_output

  # Test that both methods gave the same solution
  if expected_output == :first_order # Else there are issues with uninitialized values
    @test out2.solution == out1.solution
    @test out2.iter == out1.iter 
    @test out2.dual_feas == out1.dual_feas
    @test out2.solver_specific[:smooth_obj] == out1.solver_specific[:smooth_obj]
    @test out2.solver_specific[:nonsmooth_obj] == out1.solver_specific[:nonsmooth_obj]
  end

  # In case the problem has bounds, check feasibility
  if has_bounds(reg_nlp.model) && expected_output == :first_order
    l_bound, u_bound = reg_nlp.model.meta.lvar, reg_nlp.model.meta.uvar
    @test all(out2.solution .>= l_bound)
    @test all(out2.solution .<= u_bound)
  end
  
  # Test Type stability 
  @test_opt target_modules=(RegularizedOptimization,) function_filter=(@nospecialize(f)->(f!=LinearAlgebra.opnorm)) solve!(workspace, reg_nlp, out2; solver_kwargs...)
end