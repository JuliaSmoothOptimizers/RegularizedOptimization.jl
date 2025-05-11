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

@testset "allocs" begin
  for (h, h_name) ∈ ((NormL1(λ), "l1"),)
    for (solver, solver_name) ∈
        ((:R2Solver, "R2"), (:R2DHSolver, "R2DH"), (:R2NSolver, "R2N"), (:R2NSolver, "R2N"))
      @testset "$(solver_name)" begin
        solver_name == "R2N" && continue #FIXME
        reg_nlp = RegularizedNLPModel(LBFGSModel(bpdn), h)
        solver = eval(solver)(reg_nlp)
        stats = RegularizedExecutionStats(reg_nlp)
        @test @wrappedallocs(solve!(solver, reg_nlp, stats, ν = 1.0, atol = 1e-6, rtol = 1e-6)) == 0
        @test stats.status == :first_order
      end
    end
  end
  for (h, h_name) ∈ (
    (NormL1(0.1), "l1"),
    (NormLp(0.1, 1.6, bpdn.meta.nvar), "lp"),
    # (NormTVp(0.1, 1.6, bpdn.meta.nvar), "tvp"), hits max time limit
  )
    for (solver, solver_name) ∈ ((:iR2Solver, "iR2"),)
      @testset "$(solver_name)" begin
        reg_nlp = RegularizedNLPModel(LBFGSModel(bpdn), h)
        solver = eval(solver)(reg_nlp)
        stats = RegularizedExecutionStats(reg_nlp)
        @test @wrappedallocs(
          solve!(solver, reg_nlp, stats, ν = 1.0, atol = 1e-6, rtol = 1e-6, verbose = 0)
        ) == 0
        @test stats.status == :first_order
      end
    end
  end
end
